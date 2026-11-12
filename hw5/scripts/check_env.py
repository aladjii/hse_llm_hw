import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import BASE_MODEL_ID, MODEL_REVISION, ROOT, sha256


def main() -> None:
    import torch
    from evaluate import parse_and_score_response
    from reward import make_reward
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    function = {
        "name": "lookup",
        "description": "найти город",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    tools = [{"type": "function", "function": function}]
    response = "<tool_call>\n<function=lookup>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
    target = [{"name": "lookup", "arguments": {"city": "Paris"}}]
    reward = make_reward(tokenizer, "exact", 1024)
    ids = tokenizer.encode(response, add_special_tokens=False)
    scores = reward(
        completions=[response],
        completion_ids=[ids],
        tools_json=[json.dumps(tools)],
        target_json=[json.dumps(target)],
    )
    assert scores == [1.0], scores
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "city"}],
            "tools": tools,
        }
    )
    example = {
        "id": "test",
        "category": "simple",
        "question": [],
        "function": [function],
        "ground_truth": [{"lookup": {"city": ["Paris"]}}],
    }
    _, checked = parse_and_score_response(
        example,
        SimpleNamespace(text=response, finish_reason="stop"),
        request,
        Qwen3CoderToolParser(tokenizer),
    )
    assert checked["valid"], checked
    for bad in ["", "<tool_call>", "<tool_call></tool_call>"]:
        bad_ids = tokenizer.encode(bad, add_special_tokens=False)
        assert reward(
            completions=[bad],
            completion_ids=[bad_ids],
            tools_json=[json.dumps(tools)],
            target_json=["[]"],
        ) == [0.0]
    assert torch.cuda.is_available(), "для этой проверки нужна выделенная GPU с CUDA"
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "vllm",
                "transformers",
                "trl",
                "datasets",
                "peft",
                "bfcl-eval",
                "numpy",
            ]
        },
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "benchmark_sha256": sha256(ROOT / "bfcl.jsonl"),
        "evaluate_sha256": sha256(ROOT / "evaluate.py"),
        "reward_and_official_ast_smoke": "passed",
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
