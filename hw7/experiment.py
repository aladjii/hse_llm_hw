"""Запускает офлайн-эксперименты по сжатию Qwen3 на MASSIVE."""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

SEED = 42
MODELS = Path("/gpfs/fs1/DATA/HFs")
DATA = Path.home() / "llm_data.zip"
ARTIFACTS = Path("/gpfs/fs_2/SHARED/adzhiev/massive_ru_ru_ed58ac4/artifacts")
BASE = "Qwen3-8B"
TEACHER = "Qwen3-32B"
MAX_NEW = 128
SLOT_PATTERN = re.compile(r"\[([^:\[\]]+?)\s*:\s*([^\[\]]*?)\]")


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    gold: str


def dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def save_json(path: Path, obj: Any) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)
        file.write("\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def dataset(archive: Path, member: str) -> pa.Table:
    with ZipFile(archive) as source:
        return pq.read_table(pa.BufferReader(source.read(f"hw7_data/{member}")))


def massive(archive: Path) -> tuple[list[Example], list[Example], list[str], list[str]]:
    train, dev = (
        dataset(archive, f"massive/ru-RU/{split}/0000.parquet")
        for split in ("train", "validation")
    )
    features = json.loads(train.schema.metadata[b"huggingface"])["info"]["features"]
    intents = features["intent"]["names"]
    types = set()

    def convert(table: pa.Table, record_types: bool) -> list[Example]:
        examples = []
        for row in table.select(["id", "utt", "intent", "annot_utt"]).to_pylist():
            slots = [[key.strip(), value.strip()] for key, value in SLOT_PATTERN.findall(row["annot_utt"])]
            if record_types:
                types.update(key for key, _ in slots)
            examples.append(Example(str(row["id"]), row["utt"], dump({
                "intent": intents[row["intent"]], "slots": slots,
            })))
        return examples

    return convert(train, True), convert(dev, False), intents, sorted(types)


def instruction(intents: list[str], types: list[str]) -> str:
    return (
        'Определи intent и извлеки slots. Верни только JSON вида '
        '{"intent":"...","slots":[["тип","значение"],...]}. '
        'Значения slots копируй из запроса. Если slots нет, верни пустой список. '
        'Не добавляй объяснения.\nДопустимые intents: ' + ", ".join(intents)
        + "\nДопустимые типы slots: " + ", ".join(types)
    )


def encode(tokenizer: Any, system: str, texts: Sequence[str]) -> list[list[int]]:
    return tokenizer.apply_chat_template(
        [[{"role": "system", "content": system}, {"role": "user", "content": text}] for text in texts],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False,
    )


def load_model(path: Path) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
    ).eval()
    return model, tokenizer


def trim_answer(ids: list[int], eos_ids: set[int], pad_id: int) -> list[int]:
    for index, token in enumerate(ids):
        if token in eos_ids:
            return ids[:index + 1]
    while ids and ids[-1] == pad_id:
        ids.pop()
    return ids


@torch.inference_mode()
def generate(
    model: Any, tokenizer: Any, system: str, rows: Sequence[Example], batch_size: int,
) -> Iterator[dict[str, Any]]:
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    stop_tokens = set(eos_ids)
    pad_id = tokenizer.pad_token_id
    config = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=MAX_NEW, use_cache=True,
        pad_token_id=pad_id, eos_token_id=eos_ids,
    )
    start = time.monotonic()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        prompt_ids = encode(tokenizer, system, [row.text for row in batch])
        inputs = tokenizer.pad({"input_ids": prompt_ids}, padding=True, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, generation_config=config)
        suffixes = output[:, inputs.input_ids.shape[1]:].cpu().tolist()
        answers = [trim_answer(ids, stop_tokens, pad_id) for ids in suffixes]
        for row, prompt, ids, text in zip(
            batch, prompt_ids, answers, tokenizer.batch_decode(answers, skip_special_tokens=True), strict=True,
        ):
            yield {
                "id": row.id, "text": row.text, "gold": row.gold,
                "teacher_answer": text, "prompt_ids": prompt, "answer_ids": ids,
            }
        if offset % (batch_size * 10) == 0 or offset + batch_size >= len(rows):
            print(f"генерация: {min(offset + batch_size, len(rows))}/{len(rows)}, "
                  f"{time.monotonic() - start:.0f} с", flush=True)


def parse_answer(text: str, intents: set[str], types: set[str] | None) -> tuple[str, Counter] | None:
    try:
        obj = json.loads(text.strip())
    except ValueError:
        return None
    if not isinstance(obj, dict) or set(obj) != {"intent", "slots"}:
        return None
    intent, slots = obj["intent"], obj["slots"]
    if not isinstance(intent, str) or intent not in intents or not isinstance(slots, list):
        return None
    values: Counter[tuple[str, str]] = Counter()
    for slot in slots:
        if not isinstance(slot, list) or len(slot) != 2:
            return None
        key, value = slot
        if (not isinstance(key, str) or (types is not None and key not in types)
                or not isinstance(value, str) or not value.strip()):
            return None
        values[key, " ".join(value.casefold().split())] += 1
    return intent, values


def compute_metrics(rows: Sequence[dict[str, Any]], intents: list[str], types: list[str]) -> dict:
    if not rows:
        raise ValueError("параметр `rows` не должен быть пустым")
    intent_set, type_set = set(intents), set(types)
    valid = correct = exact = tp = fp = fn = 0
    for row in rows:
        # редкий тип game_type встречается только в dev и остаётся в gold.
        gold = parse_answer(row["gold"], intent_set, None)
        if gold is None:
            raise ValueError(f"невалидный эталон: id={row['id']}")
        pred = parse_answer(row["prediction"], intent_set, type_set)
        slots = pred[1] if pred is not None else Counter()
        tp += (slots & gold[1]).total()
        fp += (slots - gold[1]).total()
        fn += (gold[1] - slots).total()
        if pred is not None:
            valid += 1
            correct += pred[0] == gold[0]
            exact += pred == gold
    n = len(rows)
    denom = 2 * tp + fp + fn
    slot_f1 = 2 * tp / denom if denom else float(valid == n)
    return {
        "quality": (correct / n + slot_f1) / 2,
        "intent_accuracy": correct / n, "slot_micro_f1": slot_f1,
        "frame_exact_match": exact / n, "valid_json_rate": valid / n, "n": n,
    }


def evaluate(
    model: Any, tokenizer: Any, system: str, rows: Sequence[Example],
    batch_size: int, target: Path, intents: list[str], types: list[str],
) -> dict:
    predictions = []
    with (target / "dev_predictions.jsonl").open("x", encoding="utf-8") as file:
        for row in generate(model, tokenizer, system, rows, batch_size):
            entry = {key: row[key] for key in ("id", "text", "gold")}
            entry["prediction"] = row["teacher_answer"]
            predictions.append(entry)
            file.write(dump(entry) + "\n")
    metrics = compute_metrics(predictions, intents, types)
    save_json(target / "dev.json", metrics)
    print(dump(metrics), flush=True)
    return metrics


def setup_run(args: argparse.Namespace) -> Path:
    path = args.artifacts / args.run_name
    path.mkdir(parents=True, exist_ok=False)
    (path / "checkpoints").mkdir()
    (path / "metrics").mkdir()
    repo = Path(__file__).resolve().parent.parent

    def git(*cmd: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *cmd], text=True).strip()

    params = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    params.update(seed=SEED, max_new_tokens=MAX_NEW, gpu=torch.cuda.get_device_name(0),
                  versions={name: version(name) for name in ("torch", "transformers", "pyarrow", "peft")})
    save_json(path / "config.yaml", params)
    with (path / "git_meta.txt").open("x", encoding="utf-8") as file:
        print(f"repo={repo}\nbranch={git('branch', '--show-current')}\nsha={git('rev-parse', 'HEAD')}\n"
              f"dirty={bool(git('status', '--porcelain', '--untracked-files=no'))}\n"
              f"saved_at={datetime.now(timezone.utc).isoformat()}", file=file)
    return path


class RunLog:
    def __init__(self, terminal: TextIO, file: TextIO) -> None:
        self.terminal = terminal
        self.file = file

    def write(self, text: str) -> int:
        self.file.write(text)
        return self.terminal.write(text)

    def flush(self) -> None:
        self.file.flush()
        self.terminal.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


def run(args: argparse.Namespace, folder: Path) -> None:
    train, dev, intents, types = massive(args.archive)
    system = instruction(intents, types)
    dev = dev[:args.dev_limit]
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    # cuDNN на этом окружении не строит план для маски с левым паддингом.
    torch.backends.cuda.enable_cudnn_sdp(False)
    print(f"прогон {args.run_name}: train={len(train)}, dev={len(dev)}", flush=True)
    if args.mode in ("baseline", "teacher"):
        model, tokenizer = load_model(args.models / (TEACHER if args.mode == "teacher" else BASE))
        evaluate(model, tokenizer, system, dev, args.batch, folder / "metrics", intents, types)
        if args.mode == "teacher":
            examples = random.Random(SEED).sample(train, min(args.samples, len(train)))
            answers = folder / "checkpoints" / "teacher_answers"
            answers.mkdir()
            stats: Counter[str] = Counter()
            intent_set, type_set = set(intents), set(types)
            with (answers / "answers.jsonl").open("x", encoding="utf-8", buffering=1) as file:
                for row in generate(model, tokenizer, system, examples, args.batch):
                    row["valid"] = parse_answer(row["teacher_answer"], intent_set, type_set) is not None
                    stats["total"] += 1
                    stats["valid"] += row["valid"]
                    file.write(dump(row) + "\n")
            save_json(folder / "metrics" / "train_answers.json", stats)
        return
    if args.mode == "quant":
        from quantize import quant_run

        quant_run(args, folder, train, dev, intents, types, system)
        return
    if args.mode == "student":
        from student import student_run

        student_run(args, folder, dev, intents, types, system)
        return
    raise ValueError(f"неизвестное значение параметра `mode`: {args.mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "teacher", "quant", "student"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--archive", type=Path, default=DATA)
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--source", choices=("massive", "wikitext"), default="massive")
    parser.add_argument("--method", choices=("awq", "gptq", "smoothquant"))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--cal-length", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--teacher-run", default="teacher_qwen3_32b")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("параметр `run-name` должен быть именем каталога")
    folder = setup_run(args)
    with (folder / "run.log").open("x", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(RunLog(sys.stdout, log)), redirect_stderr(RunLog(sys.stderr, log)):
            try:
                run(args, folder)
            except Exception:
                traceback.print_exc()
                raise SystemExit(1) from None


if __name__ == "__main__":
    main()
