#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

rpm="${1:-120}"
exec "$HOME/venvs/hse/bin/python" -B bench.py \
  --rpm "$rpm" --duration 300 \
  --output "${2:-results/load-${rpm}rpm-$(date +%Y%m%d-%H%M%S)}"
