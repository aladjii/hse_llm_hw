#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1

exec "$HOME/venvs/hse/bin/vllm" serve \
  "${MODEL_DIR:-/gpfs/fs1/DATA/HFs/Qwen3-8B}" \
  --served-model-name Qwen/Qwen3-8B --host 127.0.0.1 \
  --port "${PORT:-8000}" "$@"
