#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
: "${HW5_MODEL:?укажите каталог исходной модели в HW5_MODEL}"
name="${1:?запуск: bash scripts/run_one.sh grpo_exact}"
test -f "configs/$name.json"
mkdir -p logs results
test ! -e "runs/$name"
test ! -e "checkpoints/$name"
test ! -e "results/$name"
python3 -u train.py "configs/$name.json" --model "$HW5_MODEL" --output "runs/$name" 2>&1 | tee "logs/$name.train.log"
python3 -u merge.py --model "$HW5_MODEL" --adapter "runs/$name/adapter" --output "checkpoints/$name" 2>&1 | tee "logs/$name.merge.log"
python3 -u evaluate.py "./checkpoints/$name" --output "results/$name" 2>&1 | tee "logs/$name.evaluate.log"
