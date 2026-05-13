#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${1:-outputs/alpha158_baseline}"
mkdir -p "$OUT_DIR"

python examples/benchmarks/LightGBM/run_alpha158_baseline_and_save.py \
  --output "$OUT_DIR/metrics.json" \
  > "$OUT_DIR/run.log" 2>&1

echo "saved log to $OUT_DIR/run.log"
echo "saved metrics to $OUT_DIR/metrics.json"
