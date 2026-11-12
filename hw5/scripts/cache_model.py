import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import MODEL_REVISION, ROOT, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    assets = json.loads((ROOT / "data/model_assets.json").read_text())
    expected = assets["config"]
    actual = json.loads((model / "config.json").read_text())
    for key in ("model_type", "text_config", "vision_config"):
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"поле `{key}` в конфигурации `model` отличается от требуемой версии"
            )
    for filename, expected_hash in assets["files"].items():
        if sha256(model / filename) != expected_hash:
            raise ValueError(f"файл `{filename}` отличается от исходной версии модели")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    for name in shards:
        if not (model / name).is_file():
            raise FileNotFoundError(model / name)
    repo = ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3.5-2B"
    snapshot = repo / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    sources = [
        model / name
        for name in [
            "config.json",
            "model.safetensors.index.json",
            *assets["files"],
            *shards,
        ]
    ]
    for source in sources:
        destination = snapshot / source.name
        if destination.is_symlink() and destination.resolve() == source.resolve():
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.symlink_to(source.resolve())
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs/main").write_text(MODEL_REVISION)
    print(f"токенизатор и {len(shards)} локальных частей весов доступны в {snapshot}")
    print(
        "конфигурация проверена; происхождение локальных весов нужно сверить отдельно"
    )


if __name__ == "__main__":
    main()
