#!/usr/bin/env bash
# Исходный запуск экспериментов в одном выделении восьми GPU.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
: "${HW5_MODEL:?укажите каталог исходной модели в HW5_MODEL}"
export TOKENIZERS_PARALLELISM=false
mkdir -p logs results runs checkpoints

echo "[$(date -Is)] проверка окружения"
python3 scripts/cache_model.py --model "$HW5_MODEL" 2>&1 | tee logs/cache_model.log
python3 scripts/check_env.py > results/environment.json 2> logs/environment.log
python3 -m pip check > logs/pip-check.txt 2>&1 || true

echo "[$(date -Is)] оценка исходной модели на GPU 0"
CUDA_VISIBLE_DEVICES=0 python3 -u evaluate.py Qwen/Qwen3.5-2B --output results/base > logs/base.evaluate.log 2>&1 &
base_pid=$!

echo "[$(date -Is)] пробное обучение на GPU 7"
CUDA_VISIBLE_DEVICES=7 python3 -u train.py configs/grpo_exact.json --model "$HW5_MODEL" \
  --output runs/smoke --max-steps 2 > logs/smoke.train.log 2>&1
python3 merge.py --model "$HW5_MODEL" --adapter runs/smoke/adapter \
  --output checkpoints/smoke > logs/smoke.merge.log 2>&1
CUDA_VISIBLE_DEVICES=7 python3 -u evaluate.py ./checkpoints/smoke \
  --output results/smoke > logs/smoke.evaluate.log 2>&1

echo "[$(date -Is)] запуск основных экспериментов"
CUDA_VISIBLE_DEVICES=1 python3 -u train.py configs/grpo_exact.json --model "$HW5_MODEL" \
  --output runs/grpo_exact > logs/grpo_exact.train.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=2 python3 -u train.py configs/grpo_shaped.json --model "$HW5_MODEL" \
  --output runs/grpo_shaped > logs/grpo_shaped.train.log 2>&1 &
p2=$!
CUDA_VISIBLE_DEVICES=3 python3 -u train.py configs/dr_grpo_shaped.json --model "$HW5_MODEL" \
  --output runs/dr_grpo_shaped > logs/dr_grpo_shaped.train.log 2>&1 &
p3=$!
wait $p1 $p2 $p3
wait $base_pid

echo "[$(date -Is)] объединение весов и оценка"
CUDA_VISIBLE_DEVICES=1 python3 -u merge.py --model "$HW5_MODEL" --adapter runs/grpo_exact/adapter \
  --output checkpoints/grpo_exact > logs/grpo_exact.merge.log 2>&1 &
m1=$!
CUDA_VISIBLE_DEVICES=2 python3 -u merge.py --model "$HW5_MODEL" --adapter runs/grpo_shaped/adapter \
  --output checkpoints/grpo_shaped > logs/grpo_shaped.merge.log 2>&1 &
m2=$!
CUDA_VISIBLE_DEVICES=3 python3 -u merge.py --model "$HW5_MODEL" --adapter runs/dr_grpo_shaped/adapter \
  --output checkpoints/dr_grpo_shaped > logs/dr_grpo_shaped.merge.log 2>&1 &
m3=$!
wait $m1 $m2 $m3

CUDA_VISIBLE_DEVICES=1 python3 -u evaluate.py ./checkpoints/grpo_exact \
  --output results/grpo_exact > logs/grpo_exact.evaluate.log 2>&1 &
e1=$!
CUDA_VISIBLE_DEVICES=2 python3 -u evaluate.py ./checkpoints/grpo_shaped \
  --output results/grpo_shaped > logs/grpo_shaped.evaluate.log 2>&1 &
e2=$!
CUDA_VISIBLE_DEVICES=3 python3 -u evaluate.py ./checkpoints/dr_grpo_shaped \
  --output results/dr_grpo_shaped > logs/dr_grpo_shaped.evaluate.log 2>&1 &
e3=$!
wait $e1 $e2 $e3

python3 summarize.py --plots
echo "[$(date -Is)] завершено"
