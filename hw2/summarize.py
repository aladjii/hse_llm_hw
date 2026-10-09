"""
Сводная таблица экспериментов по results/*/summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from run_experiments import CONFIGS

HERE = Path(__file__).resolve().parent
FAILED = {
    "bs64x1": "OOM (80 GiB)",
    "bs128x1": "OOM (80 GiB)",
    "offload": "краш: TRL offloading + SDPA (attn_bias stride)",
    "pack_fa2_liger_compile": "краш: NaN loss (liger + torch.compile)",
}

MECH = {
    "baseline": "— (шаблон)",
    "bs64x1": "батч 64×1",
    "bs128x1": "батч 128×1",
    "bs64x1_ckpt": "батч 64×1 + checkpointing",
    "bs128x1_ckpt": "батч 128×1 + checkpointing",
    "grad_ckpt": "activation checkpointing",
    "fa2": "FlashAttention-2",
    "liger": "liger kernels",
    "compile": "torch.compile",
    "offload": "offloading активаций",
    "offload_fa2": "offloading + FA2",
    "packing": "sequence packing",
    "padding_free": "padding-free + FA2",
    "pf_fa2_bs64": "padding-free + FA2 + батч 64×1",
    "pack_fa2_liger": "packing + FA2 + liger",
    "pack_fa2_compile": "packing + FA2 + compile",
    "pack_fa2_liger_compile": "packing + FA2 + liger + compile",
}


def load(name: str) -> dict | None:
    path = HERE / "results" / name / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def backfill_config(name: str) -> None:
    if name not in CONFIGS:
        return
    cfg_path = HERE / "results" / name / "config.json"
    if load(name) and not cfg_path.exists():
        spec = CONFIGS[name]
        cfg_path.write_text(
            json.dumps(
                dict(
                    attention=spec["attention"],
                    seconds=spec["seconds"],
                    train=dict(spec["train"], output_dir=f"results/{name}"),
                    note=spec["note"],
                ),
                indent=2,
            )
            + "\n"
        )


def row(name: str, s: dict, base_tps: float) -> str:
    tps = s["tokens_per_second"]
    tps_s = f"{tps:.0f}" if tps else "—"
    speedup = f"{tps / base_tps:.2f}" if tps else "—"
    return (
        f"| {name} | {MECH.get(name, name)} | {CONFIGS[name]['attention'] if name in CONFIGS else ''} "
        f"| {tps_s} | {speedup} | {s['peak_allocated_gib']:.1f} | "
        f"{s['validation_loss']:.3f} | {s['optimizer_steps']} | {s['train_seconds']:.0f} |"
    )


def main() -> None:
    order = [
        "baseline", "bs64x1", "bs128x1", "bs64x1_ckpt", "bs128x1_ckpt", "grad_ckpt",
        "fa2", "liger", "compile", "offload", "offload_fa2",
        "packing", "padding_free", "pf_fa2_bs64",
        "pack_fa2_liger", "pack_fa2_compile", "pack_fa2_liger_compile",
    ]
    base = load("baseline")
    base_tps = base["tokens_per_second"]

    print("### Короткие раны (90 c, если не указано иное)\n")
    print("| Конфигурация | Механизмы | attention | tokens/s | S | peak GiB | val loss | шагов | c |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for name in order:
        backfill_config(name)
        s = load(name)
        if s:
            print(row(name, s, base_tps))
        elif name in FAILED:
            note = "300 c" if name == "compile" else ""
            print(f"| {name} | {MECH.get(name, name)} | {CONFIGS[name]['attention']} | {FAILED[name]} | | | | | {note} |")

    print("\n### Финальные раны (300 c, одна и та же GPU)\n")
    base300 = load("baseline_300s")
    if base300:
        print("| Конфигурация | tokens/s | S (300 c) | peak GiB | val loss | шагов |")
        print("|---|---:|---:|---:|---:|---:|")
        print(f"| baseline_300s | {base300['tokens_per_second']:.0f} | 1.00 | {base300['peak_allocated_gib']:.1f} | {base300['validation_loss']:.3f} | {base300['optimizer_steps']} |")
        for name in ("final", "pack_fa2_compile_300s", "pack_fa2_liger_300s", "pf_fa2_bs64_300s"):
            s = load(name)
            if s and s["tokens_per_second"]:
                label = "final (итоговый train.py)" if name == "final" else name
                print(f"| {label} | {s['tokens_per_second']:.0f} | {s['tokens_per_second'] / base300['tokens_per_second']:.2f} | {s['peak_allocated_gib']:.1f} | {s['validation_loss']:.3f} | {s['optimizer_steps']} |")


if __name__ == "__main__":
    main()
