#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# для чистого кеша запускаем новый сервер перед каждой точкой.
exec "$HOME/venvs/hse/bin/python" -B bench.py \
  --rpm "${2:-120}" --output "${1:-results/$(date +%Y%m%d-%H%M%S)}"
