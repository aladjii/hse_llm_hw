import argparse
import json
from pathlib import Path

import yaml
from experiment import CONFIGS, LABELS, ROOT, provenance, save_json, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    manifest = json.loads((args.root / "data" / "manifest.json").read_text())
    bundle = {"dataset": manifest, "runs": {}, "manual": []}
    table = [
        "| Run | r | alpha | dropout | targets | Accuracy, % | Macro-F1, % | Trainable | Seconds | Peak allocated, GiB |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name in CONFIGS:
        out = args.root / "artifacts" / (name + args.suffix)
        metrics = json.loads((out / "metrics.json").read_text())
        config = yaml.safe_load((out / "config.yaml").read_text())
        predictions = json.loads((out / "test_predictions.json").read_text())
        recalculated = scores(
            [row["label"] for row in predictions],
            [row["prediction"] for row in predictions],
        )
        if recalculated != metrics["test"]:
            raise ValueError(f"`{name}` metrics do not match saved predictions")
        bundle["runs"][name] = {"config": config, "metrics": metrics}
        targets = config.get("target_modules", "—")
        if isinstance(targets, list):
            targets = ",".join(target.replace("_proj", "") for target in targets)
        table.append(
            f"| {name} | {config.get('r', '—')} | {config.get('lora_alpha', '—')} "
            f"| {config.get('lora_dropout', '—')} | {targets} "
            f"| {100 * metrics['test']['accuracy']:.2f} "
            f"| {100 * metrics['test']['macro_f1']:.2f} "
            f"| {metrics['trainable_parameters']:,} | {metrics['train_seconds']:.1f} "
            f"| {metrics['peak_allocated_gib']:.2f} |"
        )
        if name in ("base", "full_ft", "lora_base"):
            for index in manifest["manual_indices"]:
                record = predictions[index]
                bundle["manual"].append(
                    {
                        "run": name,
                        "index": index,
                        "text": record["text"],
                        "reference": LABELS[record["label"]],
                        "prediction": LABELS[record["prediction"]],
                    }
                )
    out = args.root / "artifacts" / ("summary_emotion_v1" + args.suffix)
    provenance(out, {"runs": list(CONFIGS), "suffix": args.suffix, "data": manifest})
    save_json(out / "results.json", bundle)
    with (out / "tables.md").open("x") as stream:
        stream.write("\n".join(table) + "\n")
    print("\n".join(table))
    print(json.dumps(bundle["manual"], indent=2))


if __name__ == "__main__":
    main()
