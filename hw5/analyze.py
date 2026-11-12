import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prepare import BASE_MODEL_ID, MODEL_REVISION, ROOT


def quantiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {
        name: ordered[int(q * (len(ordered) - 1))]
        for name, q in [
            ("min", 0),
            ("p50", 0.5),
            ("p95", 0.95),
            ("p99", 0.99),
            ("max", 1),
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--output", type=Path, default=ROOT / "data/eda")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for name in ["train", "validation"]:
        lengths, answers, calls, counts = [], [], [], Counter()
        with (args.output / f"{name}_lengths.jsonl").open("w") as output:
            for line in (args.data / f"{name}.jsonl").read_text().splitlines():
                row = json.loads(line)
                messages, tools = (
                    json.loads(row["messages_json"]),
                    json.loads(row["tools_json"]),
                )
                ids = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=True,
                    return_dict=False,
                )
                target = json.loads(row["target_json"])
                assistant: dict[str, Any] = {
                    "role": "assistant",
                    "content": "" if target else row["reference_text"],
                }
                if target:
                    assistant["tool_calls"] = [
                        {"type": "function", "function": call} for call in target
                    ]
                complete = tokenizer.apply_chat_template(
                    messages + [assistant],
                    tools=tools,
                    enable_thinking=False,
                    tokenize=True,
                    return_dict=False,
                )
                # Шаблон может заменить префикс генерации, поэтому длина эталона оценочная.
                answer_length = max(0, len(complete) - len(ids))
                lengths.append(len(ids))
                answers.append(answer_length)
                calls.append(len(target))
                counts[row["kind"]] += 1
                output.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "prompt_tokens": len(ids),
                            "reference_tokens_estimate": answer_length,
                            "calls": len(target),
                        }
                    )
                    + "\n"
                )
        summary[name] = {
            "examples": len(lengths),
            "kinds": dict(counts),
            "prompt_tokens": quantiles(lengths),
            "reference_tokens_estimate": quantiles(answers),
            "calls": dict(Counter(calls)),
            "prompt_over_3072": sum(length > 3072 for length in lengths),
            "reference_over_1024": sum(length > 1024 for length in answers),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
