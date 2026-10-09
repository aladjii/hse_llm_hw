"""Раннер экспериментов ДЗ-2.

Для каждой конфигурации запускает отдельный процесс, в котором патчатся
глобальные настройки модуля train (TRAINING_CONFIG / ATTENTION_IMPLEMENTATION /
TRAIN_SECONDS), после чего вызывается train.main().

Все раны идут последовательно на одной и той же GPU,
у каждой конфигурации свой output_dir. Логи пишутся в logs/<name>.log,
метрики в results/<name>/summary.json (+ config.json с параметрами рана).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name -> dict(attention=..., seconds=..., train={переопределения TRAINING_CONFIG})
CONFIGS: dict[str, dict] = {
    # --- одиночные механизмы ---
    "baseline": dict(
        attention="sdpa", seconds=90, train={},
        note="шаблон: bs32 x accum2, SDPA, без оптимизаций",
    ),
    "bs64x1": dict(
        attention="sdpa", seconds=90,
        train=dict(per_device_train_batch_size=64, gradient_accumulation_steps=1),
        note="крупнее микробатч (64x1 вместо 32x2), эффективный батч тот же 64",
    ),
    "bs128x1": dict(
        attention="sdpa", seconds=90,
        train=dict(per_device_train_batch_size=128, gradient_accumulation_steps=1),
        note="ещё крупнее микробатч (128x1), эффективный батч 128",
    ),
    "grad_ckpt": dict(
        attention="sdpa", seconds=90,
        train=dict(gradient_checkpointing=True),
        note="activation checkpointing",
    ),
    "fa2": dict(
        attention="flash_attention_2", seconds=90, train={},
        note="FlashAttention-2 вместо SDPA",
    ),
    "liger": dict(
        attention="sdpa", seconds=90,
        train=dict(use_liger_kernel=True),
        note="liger kernels (fused RMSNorm/SwiGLU/CrossEntropy)",
    ),
    "packing": dict(
        attention="sdpa", seconds=90,
        train=dict(packing=True),
        note="sequence packing (bfd): убирает паддинг-пустоту",
    ),
    "padding_free": dict(
        attention="flash_attention_2", seconds=90,
        train=dict(padding_free=True),
        note="padding-free батчи через position_ids (требует FA2)",
    ),
    "compile": dict(
        attention="sdpa", seconds=300,
        train=dict(torch_compile=True),
        note="torch.compile (динамический паддинг даёт разные длины -> рекопиляции)",
    ),
    "offload": dict(
        attention="sdpa", seconds=90,
        train=dict(activation_offloading=True),
        note="offloading активаций на CPU",
    ),
    "offload_fa2": dict(
        attention="flash_attention_2", seconds=90,
        train=dict(activation_offloading=True),
        note="offloading активаций + FA2 (без attn_bias, обход бага SDPA-пути)",
    ),
    "bs64x1_ckpt": dict(
        attention="sdpa", seconds=90,
        train=dict(per_device_train_batch_size=64, gradient_accumulation_steps=1, gradient_checkpointing=True),
        note="микробатч 64x1 + activation checkpointing (без чекпойнтинга 64x1 не влезает в память)",
    ),
    "bs128x1_ckpt": dict(
        attention="sdpa", seconds=90,
        train=dict(per_device_train_batch_size=128, gradient_accumulation_steps=1, gradient_checkpointing=True),
        note="микробатч 128x1 + activation checkpointing",
    ),
    # --- сочетания ---
    "pack_fa2_liger": dict(
        attention="flash_attention_2", seconds=90,
        train=dict(packing=True, use_liger_kernel=True),
        note="packing + FA2 + liger",
    ),
    "pack_fa2_compile": dict(
        attention="flash_attention_2", seconds=90,
        train=dict(packing=True, torch_compile=True),
        note="packing + FA2 + torch.compile (liger+compile дают NaN loss, см. лог)",
    ),
    "pack_fa2_liger_compile": dict(
        attention="flash_attention_2", seconds=90,
        train=dict(packing=True, use_liger_kernel=True, torch_compile=True),
        note="packing + FA2 + liger + torch.compile",
    ),
    "pf_fa2_bs64": dict(
        attention="flash_attention_2", seconds=90,
        train=dict(padding_free=True, per_device_train_batch_size=64, gradient_accumulation_steps=1),
        note="padding-free + FA2 + микробатч 64",
    ),
}

DRIVER = r'''
import json, sys
import train

spec = json.loads(sys.argv[1])
train.ATTENTION_IMPLEMENTATION = spec.pop("attention")
train.TRAIN_SECONDS = float(spec.pop("seconds"))
train.TRAINING_CONFIG.update(spec.pop("train"))
train.main()
'''


def run_one(name: str, gpu: str, seconds: float | None, suffix: str) -> int:
    spec = CONFIGS[name]
    out_dir = f"results/{name}{suffix}"
    out_path = HERE / out_dir
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.parent.mkdir(exist_ok=True)
    (HERE / "logs").mkdir(exist_ok=True)

    run_spec = dict(
        attention=spec["attention"],
        seconds=seconds if seconds is not None else spec["seconds"],
        train=dict(spec["train"], output_dir=out_dir),
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="8",
    )
    cmd = [sys.executable, "-c", DRIVER, json.dumps(run_spec)]
    log_path = HERE / "logs" / f"{name}{suffix}.log"
    print(f"=== {name}{suffix}: {spec['note']} -> {out_dir} (gpu {gpu}, {run_spec['seconds']}s)", flush=True)
    with log_path.open("w") as log:
        rc = subprocess.run(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc == 0 and (out_path / "summary.json").exists():
        (out_path / "config.json").write_text(
            json.dumps(dict(run_spec, note=spec["note"], config_overrides=spec["train"]), indent=2) + "\n"
        )
        summary = json.loads((out_path / "summary.json").read_text())
        print(
            f"    tokens/s={summary['tokens_per_second']:.0f}  "
            f"peak_alloc={summary['peak_allocated_gib']:.1f}GiB  "
            f"val_loss={summary['validation_loss']:.4f}  "
            f"opt_steps={summary['optimizer_steps']}",
            flush=True,
        )
    else:
        print(f"    FAILED rc={rc}, см. {log_path}", flush=True)
    return rc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", help="имена конфигураций из CONFIGS (или 'all')")
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--seconds", type=float, default=None, help="переопределить длительность рана")
    parser.add_argument("--suffix", default="", help="суффикс output_dir, например _300s")
    args = parser.parse_args()

    names = list(CONFIGS) if args.names == ["all"] else args.names
    for name in names:
        if name not in CONFIGS:
            sys.exit(f"Неизвестная конфигурация: {name}. Доступны: {', '.join(CONFIGS)}")
    failed: list[str] = []
    for name in names:
        if run_one(name, args.gpu, args.seconds, args.suffix) != 0:
            failed.append(name)
    if failed:
        sys.exit(f"Провалились: {', '.join(failed)}")


if __name__ == "__main__":
    main()
