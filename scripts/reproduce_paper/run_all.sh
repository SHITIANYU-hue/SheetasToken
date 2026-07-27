#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/00_validate_release.sh"
for DATASET in industrytab_614 industrytab_1k; do
  "$SCRIPT_DIR/01_train_sat.sh" "$DATASET"
  "$SCRIPT_DIR/02_run_comparisons.sh" "$DATASET"
  "$SCRIPT_DIR/04_benchmark_latency.sh" "$DATASET" 42
done
"$SCRIPT_DIR/03_run_sensitivity.sh"
"$SCRIPT_DIR/05_aggregate_results.sh"
