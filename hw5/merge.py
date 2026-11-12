import argparse
import json
import shutil
from pathlib import Path

from prepare import MODEL_REVISION, ROOT, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16
    )
    model = model.to(args.device)
    model = PeftModel.from_pretrained(
        model, args.adapter, local_files_only=True
    ).merge_and_unload(safe_merge=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
    for filename in json.loads((ROOT / "data/model_assets.json").read_text())["files"]:
        shutil.copy2(args.model / filename, args.output / filename)
    (args.output / "export.json").write_text(
        json.dumps(
            {
                "base_revision": MODEL_REVISION,
                "model": str(args.model.resolve()),
                "adapter": str(args.adapter.resolve()),
                "adapter_config_sha256": sha256(args.adapter / "adapter_config.json"),
                "merged": True,
                "dtype": "bfloat16",
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
