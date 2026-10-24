import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiment import CONFIGS, ROOT, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--names", nargs="+", choices=list(CONFIGS), default=list(CONFIGS)
    )
    args = parser.parse_args()
    if len(set(args.names)) != len(args.names):
        parser.error("`names` must be unique")
    if args.epochs < 1:
        parser.error("`epochs` must be positive")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    pending = list(args.names)
    active = {}
    failed = []
    log_dir = args.root / "artifacts"
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        if (log_dir / (name + args.suffix)).exists():
            raise FileExistsError(log_dir / (name + args.suffix))
    dispatch = log_dir / ("dispatch_emotion_v1" + args.suffix)
    provenance(
        dispatch,
        {
            "names": args.names,
            "epochs": args.epochs,
            "suffix": args.suffix,
            "configs": CONFIGS,
        },
    )
    log = (dispatch / "run.log").open("x")
    while pending or active:
        for gpu in devices:
            if pending and gpu not in active:
                name = pending.pop(0)
                # The worker creates its own artifact directory and refuses to overwrite it.
                out = log_dir / (name + args.suffix)
                if out.exists():
                    raise FileExistsError(out)
                environment = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=gpu,
                    HF_HUB_OFFLINE="1",
                    HF_DATASETS_OFFLINE="1",
                    HF_HUB_DISABLE_TELEMETRY="1",
                    WANDB_MODE="disabled",
                    TOKENIZERS_PARALLELISM="false",
                    OMP_NUM_THREADS="8",
                )
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("experiment.py")),
                        name,
                        "--root",
                        str(args.root),
                        "--suffix",
                        args.suffix,
                        "--epochs",
                        str(args.epochs),
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                active[gpu] = (name, process)
                print(f"started {name} on GPU {gpu}", flush=True)
        for gpu, (name, process) in list(active.items()):
            status = process.poll()
            if status is not None:
                if status:
                    failed.append(name)
                print(f"finished {name}: exit={status}", flush=True)
                del active[gpu]
        time.sleep(2)
    print(json.dumps({"failed": failed}), flush=True)
    log.close()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
