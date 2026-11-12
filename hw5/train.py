import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from prepare import BASE_MODEL_ID, MODEL_REVISION, ROOT, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.max_steps is not None:
        if args.max_steps < 1:
            parser.error("`max_steps` должен быть положительным")
        config["training"]["max_steps"] = args.max_steps
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from reward import make_reward
    from transformers import (
        AutoModelForImageTextToText,
        AutoTokenizer,
        TrainerCallback,
        set_seed,
    )
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "для обучения нужна GPU с CUDA; запустите скрипт в выделенных ресурсах кластера"
        )
    settings = config["training"]
    set_seed(settings["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    length_stats: dict[str, Any] = {}

    def load_split(name: str) -> Dataset:
        rows, lengths, rejected = [], [], []
        for line in (args.data / f"{name}.jsonl").read_text().splitlines():
            row = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                json.loads(row["messages_json"]),
                tools=json.loads(row["tools_json"]),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            length = len(tokenizer.encode(prompt, add_special_tokens=False))
            lengths.append(length)
            if length > config["max_prompt_tokens"]:
                rejected.append(row["id"])
                continue
            row["prompt"] = prompt
            rows.append(row)
        lengths.sort()
        length_stats[name] = {
            "before": len(lengths),
            "after": len(rows),
            "rejected_ids": rejected,
            "min": lengths[0],
            "median": lengths[len(lengths) // 2],
            "p95": lengths[int(0.95 * (len(lengths) - 1))],
            "max": lengths[-1],
        }
        if not rows:
            raise ValueError(f"в `{name}` не осталось примеров после ограничения длины")
        dataset = Dataset.from_list(rows)
        if name == "validation":
            dataset = dataset.shuffle(seed=settings["seed"]).select(
                range(min(config["validation_examples"], len(dataset)))
            )
        return dataset

    train_data, validation_data = load_split("train"), load_split("validation")
    (args.output / "prompt_lengths.json").write_text(
        json.dumps(length_stats, indent=2) + "\n"
    )
    (args.output / "validation_ids.json").write_text(
        json.dumps(list(validation_data["id"]), indent=2) + "\n"
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config._name_or_path = str(args.model.resolve())

    class JsonLog(TrainerCallback):
        def on_log(
            self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any
        ) -> None:
            with (Path(args.output_dir) / "history.jsonl").open("a") as output:
                output.write(
                    json.dumps({"step": state.global_step, **(logs or {})}) + "\n"
                )

    training_args = GRPOConfig(output_dir=str(args.output), **settings)
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        peft_config=LoraConfig(task_type="CAUSAL_LM", **config["lora"]),
        reward_funcs=make_reward(
            tokenizer,
            config["reward"],
            settings["max_completion_length"],
            sample_path=args.output / "reward_examples.jsonl",
        ),
        train_dataset=train_data,
        eval_dataset=validation_data,
        callbacks=[JsonLog()],
    )
    versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
    }
    source_hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in sorted(ROOT.glob("*.py"))
    }
    metadata = {
        "config": config,
        "model_path": str(args.model.resolve()),
        "base_revision": MODEL_REVISION,
        "data_manifest": json.loads((args.data / "manifest.json").read_text()),
        "benchmark_sha256": sha256(ROOT / "bfcl.jsonl"),
        "evaluate_sha256": sha256(ROOT / "evaluate.py"),
        "source_sha256": source_hashes,
        "versions": versions,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "train_examples": len(train_data),
        "validation_examples": len(validation_data),
        "status": "running",
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in trainer.model.parameters()
            if parameter.requires_grad
        ),
        "trainable_names": [
            name
            for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad
        ],
    }
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    metadata["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    start = time.monotonic()
    try:
        trainer.train()
        trainer.save_model(str(args.output / "adapter"))
        trainer.save_state()
        selection = {
            "criterion": "maximum ToolACE validation reward; BFCL not used for selection",
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "last_step": trainer.state.global_step,
            "adapter": "adapter",
            "loaded_best_at_end": settings["load_best_model_at_end"],
        }
        if not selection["best_checkpoint"]:
            selection["criterion"] = (
                "last step; smoke run ended before the first validation"
            )
        (args.output / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n"
        )
        metadata["status"] = "complete"
    except BaseException as error:
        metadata["status"], metadata["error"] = "failed", str(error)
        raise
    finally:
        metadata["elapsed_seconds"] = round(time.monotonic() - start, 1)
        metadata["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
