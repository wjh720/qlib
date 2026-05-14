#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

METRICS_JSON="${1:-outputs/all_factors_library/metrics.json}"
OUT_DIR="${2:-outputs/all_factors_library/plots}"

python scripts/plot_qlib_long_asset_curves.py \
  --metrics-json "$METRICS_JSON" \
  --mlruns-dir mlruns \
  --output-dir "$OUT_DIR" \
  --prefix all_factors_library
