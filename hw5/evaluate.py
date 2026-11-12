import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from typing import Any, TypedDict, cast

from bfcl_eval.constants.enums import Language
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.inputs import TokensPrompt
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

BASE_MODEL_ID = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
BFCL_MODEL_NAME_FOR_SCORING = "Qwen/Qwen3-8B-FC"
BENCHMARK_PATH = Path("bfcl.jsonl")
CATEGORY_GROUPS = {
    "simple": ["simple"],
    "multiple": ["multiple", "live_multiple"],
    "parallel": ["parallel"],
    "parallel_multiple": ["parallel_multiple"],
    "irrelevance": ["irrelevance", "live_irrelevance"],
}
MAX_NEW_TOKENS = 1024

type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)


class BenchmarkExample(TypedDict):
    id: str
    question: list[dict[str, Any]]
    function: list[dict[str, Any]]
    category: str
    ground_truth: list[dict[str, Any]]


def convert_bfcl_schema_to_json_schema(value: JsonValue) -> JsonValue:
    if not isinstance(value, dict):
        return value

    converted_schema = {
        key: convert_bfcl_schema_to_json_schema(item) for key, item in value.items()
    }
    schema_type = converted_schema.get("type")
    if isinstance(schema_type, str):
        converted_schema["type"] = {
            "dict": "object",
            "float": "number",
            "tuple": "array",
            "any": "string",
        }.get(schema_type, schema_type)

    return converted_schema


def parse_and_score_response(
    example: BenchmarkExample,
    response: CompletionOutput,
    chat_request: ChatCompletionRequest,
    tool_parser: Qwen3CoderToolParser,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    response_text = response.text
    tool_parser.tools = cast(list[Any], chat_request.tools)
    parsed_response = tool_parser.extract_tool_calls(response_text, chat_request)
    tool_calls = [
        {call.function.name: json.loads(call.function.arguments)}
        for call in parsed_response.tool_calls
    ]

    has_unbalanced_tags = any(
        response_text.count(opening_tag) != response_text.count(closing_tag)
        for opening_tag, closing_tag in [
            ("<tool_call>", "</tool_call>"),
            ("<function=", "</function>"),
            ("<parameter=", "</parameter>"),
        ]
    )
    has_unparsed_calls = response_text.count("<tool_call>") != len(tool_calls)
    has_unparsed_tool_markup = not tool_calls and any(
        tag in response_text for tag in ("<tool", "<function", "<parameter")
    )

    if (
        response.finish_reason == "length"
        or not response_text.strip()
        or has_unbalanced_tags
        or has_unparsed_calls
        or has_unparsed_tool_markup
    ):
        return tool_calls, {
            "valid": False,
            "error": "Пустой, обрезанный или некорректно размеченный ответ.",
        }

    if "irrelevance" in example["category"]:
        return tool_calls, {
            "valid": not tool_calls,
            "error": "Вызов инструмента не требовался." if tool_calls else None,
        }

    if not tool_calls:
        return tool_calls, {"valid": False, "error": "Ожидался вызов инструмента."}

    validation_result = ast_checker(
        example["function"],
        tool_calls,
        example["ground_truth"],
        Language.PYTHON,
        example["category"],
        model_name=BFCL_MODEL_NAME_FOR_SCORING,
    )
    return tool_calls, validation_result


def generate_responses(
    model_name_or_path: str,
    examples: list[BenchmarkExample],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[ChatCompletionRequest], list[RequestOutput]]:
    chat_requests: list[ChatCompletionRequest] = []
    tokenized_prompts: list[TokensPrompt] = []

    for example in examples:
        tools: list[Any] = [
            {"type": "function", "function": convert_bfcl_schema_to_json_schema(tool)}
            for tool in example["function"]
        ]
        chat_requests.append(
            ChatCompletionRequest.model_validate(
                {
                    "model": model_name_or_path,
                    "messages": example["question"],
                    "tools": tools,
                }
            )
        )
        prompt_token_ids = tokenizer.apply_chat_template(
            example["question"],
            tools=tools,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokenized_prompts.append(
            {"prompt_token_ids": cast(list[int], prompt_token_ids)}
        )

    inference_engine = LLM(
        model=model_name_or_path,
        revision=BASE_REVISION if model_name_or_path == BASE_MODEL_ID else None,
        tokenizer=BASE_MODEL_ID,
        tokenizer_revision=BASE_REVISION,
        generation_config="vllm",
        dtype="bfloat16",
        language_model_only=True,
        max_model_len=4096,
        max_num_seqs=64,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        seed=42,
    )
    generated_responses = inference_engine.generate(
        tokenized_prompts,
        SamplingParams(
            temperature=0, max_tokens=MAX_NEW_TOKENS, skip_special_tokens=False
        ),
    )
    return chat_requests, generated_responses


def score_and_save_predictions(
    examples: list[BenchmarkExample],
    chat_requests: list[ChatCompletionRequest],
    generated_responses: list[RequestOutput],
    tool_parser: Qwen3CoderToolParser,
    predictions_path: Path,
) -> dict[str, Any]:
    correctness_by_category: dict[str, list[bool]] = {
        category: []
        for categories in CATEGORY_GROUPS.values()
        for category in categories
    }

    with predictions_path.open("w") as predictions_file:
        for example, chat_request, generated_response in zip(
            examples, chat_requests, generated_responses, strict=True
        ):
            response = generated_response.outputs[0]
            tool_calls, validation_result = parse_and_score_response(
                example, response, chat_request, tool_parser
            )
            prediction = {
                "id": example["id"],
                "category": example["category"],
                "response": response.text,
                "calls": tool_calls,
                "check": validation_result,
                "finish_reason": response.finish_reason,
                "prompt_tokens": len(
                    cast(list[int], generated_response.prompt_token_ids)
                ),
                "completion_tokens": len(response.token_ids),
            }
            correctness_by_category[example["category"]].append(
                validation_result["valid"]
            )
            predictions_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    category_metrics = {
        category: {
            "correct": sum(correctness_flags),
            "total": len(correctness_flags),
            "accuracy": mean(correctness_flags),
        }
        for category, correctness_flags in correctness_by_category.items()
    }
    group_accuracies = {
        group: mean(category_metrics[category]["accuracy"] for category in categories)
        for group, categories in CATEGORY_GROUPS.items()
    }

    return {
        "overall": mean(group_accuracies.values()),
        "groups": group_accuracies,
        "categories": category_metrics,
    }


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description="Оценка Qwen3.5-2B на учебном срезе BFCL v2.",
        add_help=False,
        usage=argparse.SUPPRESS,
    )
    argument_group = argument_parser.add_argument_group("Параметры")
    argument_group.add_argument(
        "-h", "--help", action="help", help="Показать справку и выйти."
    )
    argument_group.add_argument(
        "model",
        nargs="?",
        default=BASE_MODEL_ID,
        metavar="МОДЕЛЬ",
        help=(
            "Модель Hugging Face или папка с полными весами "
            f"(по умолчанию {BASE_MODEL_ID})."
        ),
    )
    argument_group.add_argument(
        "--output",
        type=Path,
        metavar="ПАПКА",
        help="Папка результатов (по умолчанию results/<имя модели>).",
    )
    args = argument_parser.parse_args()

    start_time = time.monotonic()
    examples: list[BenchmarkExample] = [
        json.loads(line) for line in BENCHMARK_PATH.read_text().splitlines()
    ]
    output_dir = args.output or Path("results") / Path(args.model).name
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION)
    tool_parser = Qwen3CoderToolParser(tokenizer)
    chat_requests, generated_responses = generate_responses(
        args.model, examples, tokenizer
    )
    accuracy_metrics = score_and_save_predictions(
        examples,
        chat_requests,
        generated_responses,
        tool_parser,
        output_dir / "predictions.jsonl",
    )
    metrics = {
        **accuracy_metrics,
        "model": args.model,
        "base_revision": BASE_REVISION,
        "benchmark_sha256": hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest(),
        "decoding": {
            "temperature": 0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "enable_thinking": False,
            "seed": 42,
        },
        "versions": {
            package: version(package)
            for package in (
                "numpy",
                "vllm",
                "transformers",
                "torch",
                "trl",
                "bfcl-eval",
            )
        },
        "elapsed_seconds": round(time.monotonic() - start_time, 1),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"\nИтоговая точность: {metrics['overall']:.2%}")
    print(f"Результаты сохранены в {output_dir}")


if __name__ == "__main__":
    main()
