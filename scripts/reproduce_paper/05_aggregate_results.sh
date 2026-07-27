#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

for DATASET in industrytab_614 industrytab_1k; do
  python scripts/reproduce_paper/aggregate_results.py \
    --run-dir "$RUN_ROOT/$DATASET" \
    --output "$RUN_ROOT/$DATASET/summary.json"
done

python scripts/full_corpus/plot_training_curves.py \
  --input-root "$RUN_ROOT" \
  --output "$RUN_ROOT/figures/full_corpus_training_curves.pdf"
