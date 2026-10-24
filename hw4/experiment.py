import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET = "dair-ai/emotion"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATASET_REVISION = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
SEED = 42
ROOT = Path("/gpfs/fs_2/SHARED/adzhiev/hw4_emotion_v1")
BASE_LORA = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}
CONFIGS = {
    "base": {"method": "base"},
    "full_ft": {"method": "full", "learning_rate": 2e-5},
    "lora_base": dict(method="lora", **BASE_LORA),
    "lora_r4": dict(method="lora", **(BASE_LORA | {"r": 4})),
    "lora_r64": dict(method="lora", **(BASE_LORA | {"r": 64})),
    "lora_alpha8": dict(method="lora", **(BASE_LORA | {"lora_alpha": 8})),
    "lora_alpha128": dict(method="lora", **(BASE_LORA | {"lora_alpha": 128})),
    "lora_dropout0": dict(method="lora", **(BASE_LORA | {"lora_dropout": 0.0})),
    "lora_dropout02": dict(method="lora", **(BASE_LORA | {"lora_dropout": 0.2})),
    "lora_qv": dict(
        method="lora", **(BASE_LORA | {"target_modules": ["q_proj", "v_proj"]})
    ),
    "lora_all_linear": dict(
        method="lora", **(BASE_LORA | {"target_modules": "all-linear"})
    ),
}


def save_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def provenance(out: Path, config: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=False)
    with (out / "config.yaml").open("x") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    repo = Path(__file__).resolve().parent.parent
    details = {
        "repo": str(repo),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for key, command in {
        "branch": ["git", "branch", "--show-current"],
        "commit": ["git", "rev-parse", "HEAD"],
        "dirty": ["git", "status", "--porcelain"],
    }.items():
        value = subprocess.check_output(command, cwd=repo, text=True).strip()
        details[key] = bool(value) if key == "dirty" else value
    with (out / "git_meta.txt").open("x") as stream:
        yaml.safe_dump(details, stream)


def digest(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def deduplicate(
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    seen: set[str] = set()
    result = {}
    # Keep evaluation texts first so a repeated text cannot leak into training.
    for split in ("test", "validation", "train"):
        result[split] = []
        for row in splits[split]:
            text = " ".join(row["text"].split())
            if text not in seen:
                seen.add(text)
                result[split].append({"text": text, "label": int(row["label"])})
    return result


def prompt(text: str) -> list[dict[str, str]]:
    mapping = ", ".join(f"{index}={label}" for index, label in enumerate(LABELS))
    return [
        {
            "role": "system",
            "content": (
                "Classify the emotion expressed in the text. "
                f"Labels: {mapping}. Reply with exactly one digit from 0 to 5."
            ),
        },
        {"role": "user", "content": text},
    ]


def prepare(root: Path) -> None:
    out = root / "artifacts" / "prepare_emotion_v1"
    provenance(out, {"model": MODEL, "dataset": DATASET, "seed": SEED})
    model = snapshot_download(
        MODEL,
        revision=MODEL_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.tiktoken", "README.md"],
    )
    source = snapshot_download(
        DATASET,
        revision=DATASET_REVISION,
        repo_type="dataset",
        allow_patterns=["split/*.parquet", "README.md"],
    )
    raw = load_dataset(
        "parquet",
        data_files={
            split: str(next((Path(source) / "split").glob(f"{split}-*.parquet")))
            for split in ("train", "validation", "test")
        },
    )
    rows = deduplicate({split: list(part) for split, part in raw.items()})
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    data = root / "data"
    data.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "model": MODEL,
        "model_revision": Path(model).name,
        "dataset": DATASET,
        "dataset_revision": Path(source).name,
        "model_path": model,
        "labels": LABELS,
        "seed": SEED,
        "splits": {},
    }
    for split, items in rows.items():
        encoded = tokenizer.apply_chat_template(
            [prompt(row["text"]) for row in items],
            tokenize=True,
            add_generation_prompt=True,
        )
        for row, ids in zip(items, encoded, strict=True):
            row["input_ids"] = ids
        if max(len(row["input_ids"]) for row in items) > 512:
            raise ValueError("`max_length` exceeds 512; inspect the dataset")
        save_json(data / f"{split}.json", items)
        manifest["splits"][split] = {
            "original": len(raw[split]),
            "count": len(items),
            "sha256": digest(items),
            "max_tokens": max(len(row["input_ids"]) for row in items),
            "class_counts": [sum(row["label"] == i for row in items) for i in range(6)],
        }
    generator = random.Random(SEED)
    indices = []
    for label in range(6):
        indices.append(
            generator.choice(
                [i for i, row in enumerate(rows["test"]) if row["label"] == label]
            )
        )
    indices += generator.sample(
        [i for i in range(len(rows["test"])) if i not in indices], 2
    )
    manifest["manual_indices"] = indices
    save_json(data / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def scores(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    confusion = [[0] * 6 for _ in range(6)]
    for actual, predicted in zip(labels, predictions, strict=True):
        confusion[actual][predicted] += 1
    f1 = []
    for label in range(6):
        true_positive = confusion[label][label]
        denominator = sum(confusion[label]) + sum(row[label] for row in confusion)
        f1.append(2 * true_positive / denominator if denominator else 0.0)
    return {
        "accuracy": sum(a == b for a, b in zip(labels, predictions, strict=True))
        / len(labels),
        "macro_f1": sum(f1) / 6,
        "per_class_f1": f1,
        "confusion_matrix": confusion,
        "count": len(labels),
    }


def collate(rows: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    ids = [
        [pad_id] * (width - len(row["input_ids"])) + row["input_ids"] for row in rows
    ]
    mask = [
        [0] * (width - len(row["input_ids"])) + [1] * len(row["input_ids"])
        for row in rows
    ]
    return {
        "input_ids": torch.tensor(ids),
        "attention_mask": torch.tensor(mask),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, digit_ids: torch.Tensor
) -> tuple[dict[str, Any], list[int]]:
    model.eval()
    predictions, labels = [], []
    for batch in loader:
        target = batch.pop("labels")
        inputs = {key: value.cuda() for key, value in batch.items()}
        logits = model(**inputs, logits_to_keep=1).logits[:, -1].float()
        predictions.extend(logits[:, digit_ids].argmax(-1).tolist())
        labels.extend(target.tolist())
    return scores(labels, predictions), predictions


def run(root: Path, name: str, suffix: str, epochs: int) -> None:
    set_seed(SEED)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = json.loads((root / "data" / "manifest.json").read_text())
    spec = CONFIGS[name]
    config = dict(
        spec,
        run_name=name + suffix,
        seed=SEED,
        epochs=epochs,
        micro_batch=32,
        accumulation=2,
        effective_batch=64,
        learning_rate=spec.get("learning_rate", 2e-4),
        weight_decay=0.01,
        scheduler="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        precision="bfloat16",
        attention="sdpa",
        max_length=512,
        loss="full-vocabulary next-token cross-entropy at the answer position",
        evaluation="argmax over six digit token logits",
        data=manifest,
        packages={
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "datasets", "peft", "accelerate")
        },
    )
    out = root / "artifacts" / config["run_name"]
    provenance(out, config)
    run_log = (out / "run.log").open("x")
    rows = {
        split: json.loads((root / "data" / f"{split}.json").read_text())
        for split in ("train", "validation", "test")
    }
    for split, items in rows.items():
        if digest(items) != manifest["splits"][split]["sha256"]:
            raise ValueError(f"`{split}` data hash does not match the manifest")
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model_path"], local_files_only=True
    )
    digits = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(6)]
    if any(len(ids) != 1 for ids in digits):
        raise ValueError("`digit_ids` must each contain exactly one token")
    digit_ids = torch.tensor([ids[0] for ids in digits], device="cuda")
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_path"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).cuda()
    model.config.use_cache = False
    if spec["method"] == "lora":
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                bias="none",
                **{key: spec[key] for key in BASE_LORA},
            ),
        )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = (
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if spec["method"] != "base"
        else 0
    )

    def batch_collator(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return collate(items, tokenizer.pad_token_id)

    loader = DataLoader(
        rows["train"],
        batch_size=config["micro_batch"],
        shuffle=True,
        collate_fn=batch_collator,
        generator=torch.Generator().manual_seed(SEED),
    )
    result: dict[str, Any] = {
        "run_name": config["run_name"],
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": 100 * trainable / total,
        "gpu": torch.cuda.get_device_name(),
        "train_seconds": 0.0,
        "peak_allocated_gib": 0.0,
        "peak_reserved_gib": 0.0,
        "optimizer_steps": 0,
        "train_examples": 0,
    }
    if spec["method"] != "base":
        model.train()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            fused=True,
        )
        steps_per_epoch = (len(loader) + config["accumulation"] - 1) // config[
            "accumulation"
        ]
        steps = epochs * steps_per_epoch
        from transformers import get_cosine_schedule_with_warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=round(steps * config["warmup_ratio"]),
            num_training_steps=steps,
        )
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with (out / "metrics.jsonl").open("x") as log:
            for epoch in range(epochs):
                for batch_index, batch in enumerate(loader):
                    group_start = (
                        batch_index // config["accumulation"] * config["accumulation"]
                    )
                    group_count = min(
                        config["effective_batch"],
                        len(rows["train"]) - group_start * config["micro_batch"],
                    )
                    targets = digit_ids[batch.pop("labels").cuda()]
                    inputs = {key: value.cuda() for key, value in batch.items()}
                    logits = model(**inputs, logits_to_keep=1).logits[:, -1].float()
                    loss = torch.nn.functional.cross_entropy(
                        logits, targets, reduction="sum"
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("`loss` is not finite")
                    (loss / group_count).backward()
                    result["train_examples"] += len(targets)
                    if (batch_index + 1) % config[
                        "accumulation"
                    ] == 0 or batch_index + 1 == len(loader):
                        norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config["max_grad_norm"]
                        )
                        if not torch.isfinite(norm):
                            raise FloatingPointError("`gradient_norm` is not finite")
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        result["optimizer_steps"] += 1
                        if result["optimizer_steps"] % 25 == 0:
                            record = {
                                "epoch": epoch + 1,
                                "step": result["optimizer_steps"],
                                "loss": loss.item() / len(targets),
                                "learning_rate": scheduler.get_last_lr()[0],
                            }
                            log.write(json.dumps(record) + "\n")
                            log.flush()
                            print(json.dumps(record), flush=True)
                            print(json.dumps(record), file=run_log, flush=True)
        torch.cuda.synchronize()
        result.update(
            train_seconds=time.perf_counter() - start,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
        del optimizer, scheduler
        model.save_pretrained(out / "checkpoints", safe_serialization=True)
        tokenizer.save_pretrained(out / "checkpoints")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for split in ("validation", "test"):
        evaluation_loader = DataLoader(
            rows[split], batch_size=64, collate_fn=batch_collator
        )
        metrics, predictions = evaluate(model, evaluation_loader, digit_ids)
        result[split] = metrics
        save_json(
            out / f"{split}_predictions.json",
            [
                {
                    "index": i,
                    "text": row["text"],
                    "label": row["label"],
                    "prediction": prediction,
                }
                for i, (row, prediction) in enumerate(
                    zip(rows[split], predictions, strict=True)
                )
            ],
        )
    torch.cuda.synchronize()
    result["evaluation_seconds"] = time.perf_counter() - start
    save_json(out / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)
    print(json.dumps(result, indent=2), file=run_log)
    run_log.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=["prepare", *CONFIGS])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("`epochs` must be positive")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if args.name == "prepare":
        prepare(args.root)
    else:
        run(args.root, args.name, args.suffix, args.epochs)


if __name__ == "__main__":
    main()
