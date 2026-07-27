#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
dataset_config industrytab_1k

SEED_DIR="$DATASET_RUN_DIR/seed42"
CACHE="$SEED_DIR/finetuned_bge_cache.pt"
ROOT="$DATASET_RUN_DIR/sensitivity"
mkdir -p "$ROOT/new/k10" "$ROOT/new/k20" "$ROOT/new/layers2" \
  "$ROOT/new/layers3" "$ROOT/new/gate6" "$ROOT/new/gate8" "$ROOT/channels"

if [[ ! -f "$CACHE" ]]; then
  echo "Run 01_train_sat.sh industrytab_1k first." >&2
  exit 2
fi

run_mlp() {
  local k="$1"
  local out="$ROOT/new/k${k}"
  python scripts/full_corpus/bge_retrain_experiments.py \
    --mode rerank --architecture mlp --model "${BGE_MODEL:-unused}" \
    --data-dir "$DATA_DIR" --output-dir "$out" --cache "$CACHE" \
    --candidate-k "$k" --epochs 20 --batch-size 64 --learning-rate 2e-4 \
    --eval-ratio 0.1 --split-seed 42 --seed 42
}

run_gnn() {
  local out="$1" name="$2" k="$3" layers="$4" gate="$5" channels="$6" mlp="$7"
  python scripts/full_corpus/gated_graph_refine.py \
    --cache "$CACHE" --mlp-checkpoint "$mlp" --output-dir "$out" --name "$name" \
    --candidate-k "$k" --layers "$layers" --gate-init "$gate" \
    --epochs 20 --batch-size 64 --learning-rate 5e-4 \
    --dependency-edges "$DATA_DIR/dependency_edges.json" \
    --dependency-types "$channels" --seed 42
}

run_mlp 10
run_mlp 20
run_gnn "$ROOT/new/k10" gnn_k10 10 1 -4 "$DEPENDENCY_TYPES" \
  "$ROOT/new/k10/frozen_bge_mlp_top10.pt"
run_gnn "$ROOT/new/k20" gnn_k20 20 1 -4 "$DEPENDENCY_TYPES" \
  "$ROOT/new/k20/frozen_bge_mlp_top20.pt"

MLP="$SEED_DIR/frozen_bge_mlp_top50.pt"
run_gnn "$ROOT/channels" ref 50 1 -4 "formula_reference,summary_source" "$MLP"
run_gnn "$ROOT/new/layers2" layers2 50 2 -4 "$DEPENDENCY_TYPES" "$MLP"
run_gnn "$ROOT/new/layers3" layers3 50 3 -4 "$DEPENDENCY_TYPES" "$MLP"
run_gnn "$ROOT/new/gate6" gate6 50 1 -6 "$DEPENDENCY_TYPES" "$MLP"
run_gnn "$ROOT/new/gate8" gate8 50 1 -8 "$DEPENDENCY_TYPES" "$MLP"
run_gnn "$ROOT/channels" join 50 1 -4 "join_key" "$MLP"
run_gnn "$ROOT/channels" agg 50 1 -4 "aggregation" "$MLP"
run_gnn "$ROOT/channels" formula 50 1 -4 "formula_reference" "$MLP"
run_gnn "$ROOT/channels" summary 50 1 -4 "summary_source" "$MLP"
run_gnn "$ROOT/channels" nojoin 50 1 -4 \
  "aggregation,formula_reference,summary_source" "$MLP"
run_gnn "$ROOT/channels" all 50 1 -4 \
  "join_key,aggregation,formula_reference,summary_source" "$MLP"

python scripts/full_corpus/plot_bge_sensitivity.py \
  --input-root "$ROOT" --output-dir "$ROOT/figures"
