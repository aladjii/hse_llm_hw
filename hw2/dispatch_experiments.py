"""Диспетчер экспериментов ДЗ-2 с распределением по GPU.

Правило: одновременно не более USE_GPUS (2) наших ранов,
каждый на своей GPU. GPU, занятые чужими задачами, не трогаем.

Каждый ран — отдельный процесс с CUDA_VISIBLE_DEVICES=<gpu>, конфигурации берутся
из run_experiments.CONFIGS. Готовые (results/<name>/summary.json) пропускаются,
так что диспетчер можно перезапускать.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from run_experiments import CONFIGS, DRIVER, HERE

IDLE_MIB = 2000  # GPU считается свободной, если занято меньше этого
USE_GPUS = 2  # сколько GPU одновременно держим под свои раны
SCAN_SECONDS = 15


def gpu_used_mib() -> dict[int, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return {int(idx): int(mib) for idx, mib in (line.split(", ") for line in out.strip().splitlines())}


def launch(name: str, gpu: int) -> dict:
    spec = CONFIGS[name]
    out_dir = f"results/{name}"
    out_path = HERE / out_dir
    if out_path.exists():
        shutil.rmtree(out_path)
    run_spec = dict(
        attention=spec["attention"],
        seconds=spec["seconds"],
        train=dict(spec["train"], output_dir=out_dir),
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="8",
    )
    log = (HERE / "logs" / f"{name}.log").open("w")
    print(f"[{time.strftime('%H:%M:%S')}] START {name} на GPU {gpu}: {spec['note']}", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-c", DRIVER, json.dumps(run_spec)],
        cwd=HERE,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return dict(name=name, proc=proc, log=log, out_path=out_path, run_spec=run_spec, note=spec["note"])


def reap(job: dict) -> None:
    job["log"].close()
    summary_path = job["out_path"] / "summary.json"
    if job["proc"].returncode == 0 and summary_path.exists():
        (job["out_path"] / "config.json").write_text(
            json.dumps(dict(job["run_spec"], note=job["note"]), indent=2) + "\n"
        )
        s = json.loads(summary_path.read_text())
        tps = f"{s['tokens_per_second']:.0f}" if s["tokens_per_second"] is not None else "None"
        print(
            f"[{time.strftime('%H:%M:%S')}] DONE  {job['name']}: tokens/s={tps} "
            f"peak={s['peak_allocated_gib']:.1f}GiB val_loss={s['validation_loss']:.4f} "
            f"steps={s['optimizer_steps']}",
            flush=True,
        )
    else:
        print(f"[{time.strftime('%H:%M:%S')}] FAIL  {job['name']} rc={job['proc'].returncode} (logs/{job['name']}.log)", flush=True)


def main() -> None:
    queue = sys.argv[1:] or [n for n in CONFIGS if n != "baseline"]
    active: dict[int, dict] = {}
    failed: list[str] = []
    while queue or active:
        for gpu, job in list(active.items()):
            if job["proc"].poll() is not None:
                reap(job)
                if job["proc"].returncode != 0:
                    failed.append(job["name"])
                del active[gpu]
        used = gpu_used_mib()
        my_gpus = set(active)
        idle = sorted(g for g, mib in used.items() if g not in my_gpus and mib < IDLE_MIB)
        cap = USE_GPUS
        while queue and len(active) < cap and idle:
            name = queue.pop(0)
            if (HERE / "results" / name / "summary.json").exists():
                print(f"[{time.strftime('%H:%M:%S')}] SKIP  {name}: summary.json уже есть", flush=True)
                continue
            gpu = idle.pop(0)
            active[gpu] = launch(name, gpu)
        if queue or active:
            time.sleep(SCAN_SECONDS)
    print(f"Диспетчер завершился. Провалились: {failed or 'ничего'}", flush=True)


if __name__ == "__main__":
    main()
