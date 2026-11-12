#!/usr/bin/env bash
# Подключите файл в выделенных ресурсах кластера перед загрузкой модели.
HW5_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$HW5_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$HW5_ROOT/results/.matplotlib"
