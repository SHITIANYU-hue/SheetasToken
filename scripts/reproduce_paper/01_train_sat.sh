#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_bge_model

DATASET="${1:-industrytab_1k}"
dataset_config "$DATASET"
mkdir -p "$DATASET_RUN_DIR"

for SEED in 42 43 44; do
  SPLIT_SEED="$(split_seed_for "$SEED")"
  SEED_GATE_INIT="$(gate_init_for "$DATASET" "$SEED")"
  SEED_DIR="$DATASET_RUN_DIR/seed${SEED}"
  mkdir -p "$SEED_DIR"

  python scripts/full_corpus/bge_retrain_experiments.py \
    --mode cache --model "$BGE_MODEL" --data-dir "$DATA_DIR" \
    --output-dir "$SEED_DIR" --cache "$SEED_DIR/base_bge_cache.pt" \
    --candidate-k 50 --batch-size 64 --max-length 256 \
    --eval-ratio "$EVAL_RATIO" --split-seed "$SPLIT_SEED" --seed "$SEED"

  python scripts/full_corpus/bge_retrain_experiments.py \
    --mode finetune --model "$BGE_MODEL" --data-dir "$DATA_DIR" \
    --output-dir "$SEED_DIR" --cache "$SEED_DIR/base_bge_cache.pt" \
    --candidate-k 50 --epochs 3 --batch-size 64 --learning-rate 2e-4 \
    --max-length 256 --eval-ratio "$EVAL_RATIO" \
    --split-seed "$SPLIT_SEED" --seed "$SEED"

  python scripts/full_corpus/bge_retrain_experiments.py \
    --mode rerank --architecture mlp --model "$BGE_MODEL" \
    --data-dir "$DATA_DIR" --output-dir "$SEED_DIR" \
    --cache "$SEED_DIR/finetuned_bge_cache.pt" \
    --candidate-k 50 --epochs 20 --batch-size 64 --learning-rate 2e-4 \
    --eval-ratio "$EVAL_RATIO" --split-seed "$SPLIT_SEED" --seed "$SEED"

  python scripts/full_corpus/gated_graph_refine.py \
    --cache "$SEED_DIR/finetuned_bge_cache.pt" \
    --mlp-checkpoint "$SEED_DIR/frozen_bge_mlp_top50.pt" \
    --output-dir "$SEED_DIR" --name gated_gnn_top50 \
    --candidate-k 50 --layers 1 --gate-init "$SEED_GATE_INIT" \
    --epochs 20 --batch-size 64 --learning-rate 5e-4 \
    --dependency-edges "$DATA_DIR/dependency_edges.json" \
    --dependency-types "$DEPENDENCY_TYPES" --seed "$SEED"
done
