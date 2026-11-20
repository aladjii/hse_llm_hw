import argparse
import json
import os
import signal
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import monotonic, sleep
from typing import NamedTuple


def ngram(count: int) -> list[str]:
    config = {
        "method": "ngram",
        "num_speculative_tokens": count,
        "prompt_lookup_min": 2,
        "prompt_lookup_max": 4,
    }
    return ["--speculative-config", json.dumps(config)]


CONFIGS = {
    "default": [],
    "batch4096": ["--max-num-batched-tokens", "4096"],
    "batch2048": ["--max-num-batched-tokens", "2048"],
    "seq64": ["--max-num-seqs", "64"],
    "seq64_batch4096": ["--max-num-seqs", "64", "--max-num-batched-tokens", "4096"],
    "eager": ["--enforce-eager"],
    "ngram3": ngram(3),
    "ngram5": ngram(5),
}
SPEC_RPMS = (60, 240, 480, 960)
LOAD_RPMS = (120, 240, 480, 720, 960, 1200, 1560, 1920)
SLA_S = 15
STEP_RPM = 30


class Trial(NamedTuple):
    phase: str
    config: str
    rpm: int


def ip_port() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def run_trial(root: Path, gpu: str, cpus: list[int], trial: Trial) -> dict:
    out = root / "metrics" / trial.phase / f"{trial.config}-{trial.rpm}"
    out.mkdir(parents=True)
    cache = root / "cache" / f"gpu-{gpu}"
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": gpu,
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "VLLM_CACHE_ROOT": str(cache / "vllm"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PORT": ip_port(),
    }
    affinity = ["taskset", "-c", ",".join(map(str, cpus))]
    serve = ["bash", "serve.sh", *CONFIGS[trial.config]]
    if trial.phase == "load":
        bench = ["bash", "load_test.sh", str(trial.rpm), str(out / "profile")]
    else:
        bench = ["bash", "benchmark.sh", str(out / "profile"), str(trial.rpm)]
    print(f"start {trial.phase}/{trial.config}-{trial.rpm} gpu={gpu}", flush=True)
    with (out / "server.log").open("xb") as server_log:
        server = subprocess.Popen(
            affinity + serve, stdout=server_log, stderr=subprocess.STDOUT,
            env=env, start_new_session=True,
        )
        try:
            deadline = monotonic() + 720
            with (out / "server.log").open("rb") as reader:
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("сервер завершился до готовности")
                    if monotonic() > deadline:
                        raise RuntimeError("сервер не запустился за 720 секунд")
                    if b"Application startup complete" in reader.read():
                        break
                    sleep(0.5)
            with (out / "client.log").open("xb") as client_log:
                subprocess.run(
                    affinity + bench, stdout=client_log, stderr=subprocess.STDOUT,
                    env=env, check=True, timeout=1000,
                )
            result = json.loads((out / "profile" / "summary.json").read_text())
        except (RuntimeError, subprocess.SubprocessError) as exc:
            result = {"error": str(exc)}
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
    result.update(trial._asdict(), gpu=gpu)
    with (out / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def valid(row: dict) -> bool:
    return (
        row.get("errors") == 0
        and row.get("p90_s") is not None
        and row.get("p90_schedule_delay_s", 1.0) < 0.1
    )


def within_sla(row: dict) -> bool:
    return valid(row) and row["p90_s"] < SLA_S


def provenance(root: Path) -> None:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "--no-optional-locks", *args], text=True).strip()

    with (root / "git_meta.txt").open("x") as stream:
        cwd = Path.cwd()
        if any((path / ".git").exists() for path in (cwd, *cwd.parents)):
            print(f"repo: {git('rev-parse', '--show-toplevel')}", file=stream)
            print(f"branch: {git('branch', '--show-current')}", file=stream)
            print(f"sha: {git('rev-parse', 'HEAD')}", file=stream)
            print(f"dirty: {bool(git('status', '--porcelain'))}", file=stream)
        else:
            print(f"source: {cwd}\nrepo: n/a\nbranch: n/a\nsha: n/a\ndirty: n/a", file=stream)
        print(f"saved_at: {datetime.now(UTC).isoformat()}", file=stream)
    config = {
        "model": "Qwen/Qwen3-8B",
        "model_dir": os.environ.get("MODEL_DIR", "/gpfs/fs1/DATA/HFs/Qwen3-8B"),
        "data_sha256": sha256(Path("data/requests.jsonl").read_bytes()).hexdigest(),
        "configs": CONFIGS,
        "tuning_rpm": 120,
        "spec_rpms": SPEC_RPMS,
        "load_rpms": LOAD_RPMS,
        "load_duration_s": 300,
        "sla_p90_s": SLA_S,
        "load_grid_rpm": STEP_RPM,
        "warmup_count": 40,
        "warmup_concurrency": 8,
        "request_timeout_s": 120,
        "temperature": 0,
        "seed": 42,
        "max_tokens": 512,
        "enable_thinking": False,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "node": socket.gethostname(),
    }
    with (root / "config.yaml").open("x") as stream:
        json.dump(config, stream, indent=2)


def experiment(root: Path) -> None:
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(gpus) != 8 or not all(gpus) or "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("нужен узел Slurm с восемью выделенными GPU")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < len(gpus):
        raise RuntimeError("нужно не меньше одного CPU на GPU")
    root.mkdir(parents=True)
    provenance(root)
    print(f"artifacts: {root}", flush=True)

    def wave(trials: list[Trial]) -> list[dict]:
        with ThreadPoolExecutor(max_workers=8) as pool:
            return [
                worker.result()
                for worker in [
                    pool.submit(run_trial, root, gpus[i], cpus[i::8], trial)
                    for i, trial in enumerate(trials)
                ]
            ]

    tuning = wave([Trial("tune", config, 120) for config in CONFIGS])
    if not valid(tuning[0]):
        raise RuntimeError("не получен корректный замер для бейзлайна")
    spec = min(
        (row for row in tuning if row["config"].startswith("ngram") and valid(row)),
        key=lambda row: row["mean_s"],
        default=None,
    )
    if spec is not None:
        wave([Trial("spec", c, rpm) for rpm in SPEC_RPMS for c in ("default", spec["config"])])
    winner = min((row for row in tuning if valid(row)), key=lambda row: row["mean_s"])
    name = winner["config"]
    print(f"winner from tune: {name}", flush=True)
    load = wave([Trial("load", name, rpm) for rpm in LOAD_RPMS])
    for _ in range(3):
        usable = [row for row in load if within_sla(row)]
        if not usable:
            if min(row["rpm"] for row in load) == 120:
                candidates = [30, 60, 90]
            else:
                break
        else:
            low = max(row["rpm"] for row in usable)
            rejected = [row["rpm"] for row in load if row["rpm"] > low and "error" not in row]
            if not rejected:
                candidates = [r for r in range(low + 240, low + 1921, 240) if r <= 4800]
            else:
                high = min(rejected)
                count = (high - low - 1) // STEP_RPM
                candidates = sorted({
                    low + round((count + 1) * i / (min(count, 8) + 1)) * STEP_RPM
                    for i in range(1, min(count, 8) + 1)
                })
        tested = {row["rpm"] for row in load}
        todo = [Trial("load", name, rpm) for rpm in candidates if rpm not in tested]
        if not todo:
            break
        load.extend(wave(todo))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    experiment(args.root)


if __name__ == "__main__":
    main()
