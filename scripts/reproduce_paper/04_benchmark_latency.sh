#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

DATASET="${1:-industrytab_1k}"
SEED="${2:-42}"
dataset_config "$DATASET"
SEED_DIR="$DATASET_RUN_DIR/seed${SEED}"

python scripts/full_corpus/benchmark_inference_latency.py \
  --base-cache "$SEED_DIR/base_bge_cache.pt" \
  --tuned-cache "$SEED_DIR/finetuned_bge_cache.pt" \
  --finetuned-state "$SEED_DIR/finetuned_bge_last4.pt" \
  --mlp-checkpoint "$SEED_DIR/frozen_bge_mlp_top50.pt" \
  --gnn-checkpoint "$SEED_DIR/gated_gnn_top50.pt" \
  --cross-checkpoint "$SEED_DIR/cross_encoder_top50.pt" \
  --data-dir "$DATA_DIR" \
  --dependency-edges "$DATA_DIR/dependency_edges.json" \
  --dependency-types "$DEPENDENCY_TYPES" \
  --output "$SEED_DIR/inference_latency.json" \
  --device cuda --warmup 20 --repeats 3
