import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
BASE_MODEL_ID = "Qwen/Qwen3.5-2B"
DATA_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: normalize_schema(item) for key, item in value.items()}
    if isinstance(result.get("type"), str):
        result["type"] = {
            "dict": "object",
            "float": "number",
            "tuple": "array",
            "any": "string",
        }.get(result["type"], result["type"])
    return result


class JsonLiterals(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in {"true", "false", "null"}:
            return ast.Constant(
                value={"true": True, "false": False, "null": None}[node.id]
            )
        return node


def extract_tools(system: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    marker = "Here is a list of functions in JSON format that you can invoke:"
    if marker not in system:
        raise ValueError("неподдерживаемый формат описания инструментов")
    raw_tools, _ = json.JSONDecoder().raw_decode(system.split(marker, 1)[1].lstrip())
    tools, aliases = [], {}
    for tool in raw_tools:
        name = tool["name"]
        alias = re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_")
        if not alias or alias[0].isdigit():
            alias = "tool_" + alias
        if alias in aliases.values() or len(alias) > 64:
            raise ValueError("имя инструмента неоднозначно или длиннее 64 символов")
        aliases[name] = alias
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": alias,
                    "description": tool["description"],
                    "parameters": normalize_schema(tool["parameters"]),
                },
            }
        )
    return tools, aliases


def parse_reference(text: str, aliases: dict[str, str]) -> list[dict[str, Any]]:
    text = text.strip()
    if not text.startswith("["):
        return []
    if not text.endswith("]"):
        raise ValueError("не закрыт список эталонных вызовов")
    remaining = text[1:-1].strip()
    calls = []
    while remaining:
        name = next(
            (
                name
                for name in sorted(aliases, key=len, reverse=True)
                if remaining.startswith(name)
                and remaining[len(name) :].lstrip().startswith("(")
            ),
            None,
        )
        if name is None:
            raise ValueError("неизвестная функция в эталоне")
        arguments = remaining[len(name) :].lstrip()
        depth, quote, escaped, end = 0, "", False, None
        for i, char in enumerate(arguments):
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = ""
            elif char in "\"'":
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            raise ValueError("не закрыты аргументы эталонного вызова")
        node = ast.parse("f" + arguments[:end], mode="eval").body
        if not isinstance(node, ast.Call) or node.args:
            raise ValueError("аргументы эталонного вызова должны быть именованными")
        kwargs = {}
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg in kwargs:
                raise ValueError(
                    "эталонный аргумент повторяется или передан через распаковку"
                )
            kwargs[keyword.arg] = ast.literal_eval(JsonLiterals().visit(keyword.value))
        calls.append({"name": aliases[name], "arguments": kwargs})
        remaining = arguments[end:].strip()
        if remaining and not remaining.startswith(","):
            raise ValueError("после эталонного вызова найден посторонний текст")
        remaining = remaining[1:].lstrip() if remaining else ""
    if not calls:
        raise ValueError("пустой список эталонных вызовов")
    return calls


def expand_dialogue(row: dict[str, Any], dialogue_id: str) -> list[dict[str, Any]]:
    tools, aliases = extract_tools(row["system"])
    history, examples = [], []
    for index, message in enumerate(row["conversations"]):
        role, text = message["from"], message["value"]
        if role == "assistant":
            calls = parse_reference(text, aliases)
            if history and history[-1]["role"] == "user":
                examples.append(
                    {
                        "id": f"{dialogue_id}:{index}",
                        "dialogue_id": dialogue_id,
                        "messages_json": canonical(history),
                        "tools_json": canonical(tools),
                        "target_json": canonical(calls),
                        "reference_text": text,
                        "kind": "call" if calls else "no_call",
                        "tool_count": len(tools),
                    }
                )
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": "" if calls else text,
            }
            if calls:
                assistant["tool_calls"] = [
                    {"type": "function", "function": call} for call in calls
                ]
            history.append(assistant)
        elif role in {"user", "tool"}:
            history.append({"role": role, "content": text})
        else:
            raise ValueError("неподдерживаемая роль в диалоге")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/raw/toolace/data.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("`validation_fraction` должен находиться строго между 0 и 1")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.source.read_text())
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    rejected, seen = [], set()
    for row in rows:
        dialogue_id = hashlib.sha256(canonical(row).encode()).hexdigest()
        if dialogue_id in seen:
            rejected.append(
                {"dialogue_id": dialogue_id, "reason": "duplicate dialogue"}
            )
            continue
        seen.add(dialogue_id)
        try:
            examples = expand_dialogue(row, dialogue_id)
        except (ValueError, SyntaxError, KeyError, TypeError) as error:
            rejected.append({"dialogue_id": dialogue_id, "reason": str(error)})
            continue
        # Разделяем до разворачивания диалогов, чтобы их части оставались в одной выборке.
        fraction = (
            int(
                hashlib.sha256(f"{args.seed}:{dialogue_id}".encode()).hexdigest()[:16],
                16,
            )
            / 2**64
        )
        splits["validation" if fraction < args.validation_fraction else "train"].extend(
            examples
        )
    validation_prompts = {
        (row["messages_json"], row["tools_json"]) for row in splits["validation"]
    }
    before = len(splits["train"])
    splits["train"] = [
        row
        for row in splits["train"]
        if (row["messages_json"], row["tools_json"]) not in validation_prompts
    ]
    for name, examples in splits.items():
        (args.output / f"{name}.jsonl").write_text(
            "".join(canonical(row) + "\n" for row in examples)
        )
    (args.output / "rejected.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in rejected)
    )
    manifest = {
        "dataset": "Team-ACE/ToolACE",
        "revision": DATA_REVISION,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "raw_dialogues": len(rows),
        "source_sha256": sha256(args.source),
        "rejected_dialogues": len(rejected),
        "train_prompts_matching_validation_removed": before - len(splits["train"]),
        "rejection_reasons": dict(Counter(row["reason"] for row in rejected)),
        "splits": {
            name: {
                "examples": len(examples),
                "dialogues": len({row["dialogue_id"] for row in examples}),
                "kinds": dict(Counter(row["kind"] for row in examples)),
                "tool_counts": dict(Counter(row["tool_count"] for row in examples)),
                "sha256": sha256(args.output / f"{name}.jsonl"),
            }
            for name, examples in splits.items()
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
