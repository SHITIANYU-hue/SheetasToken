#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_bge_model

DATASET="${1:-industrytab_1k}"
dataset_config "$DATASET"

for SEED in 42 43 44; do
  SPLIT_SEED="$(split_seed_for "$SEED")"
  SEED_DIR="$DATASET_RUN_DIR/seed${SEED}"
  CACHE="$SEED_DIR/base_bge_cache.pt"
  TUNED_CACHE="$SEED_DIR/finetuned_bge_cache.pt"
  if [[ ! -f "$CACHE" || ! -f "$TUNED_CACHE" ]]; then
    echo "Run 01_train_sat.sh $DATASET first (missing seed $SEED cache)." >&2
    exit 2
  fi

  python scripts/full_corpus/cache_to_rag_json.py \
    --cache "$CACHE" --output "$SEED_DIR/zero_shot_bge_top50.json" \
    --retrieval-k 50

  python baselines/end_to_end_rag_llm.py \
    --method rag --data-dir "$DATA_DIR" --output "$SEED_DIR/rag.json" \
    --retrieval-cache "$SEED_DIR/zero_shot_bge_top50.json" \
    --embedding-model "$BGE_MODEL" --retrieval-k 5 --final-k 5 \
    --eval-ratio "$EVAL_RATIO" --seed "$SPLIT_SEED"

  python baselines/end_to_end_rag_llm.py \
    --method rag_llm --data-dir "$DATA_DIR" \
    --output "$SEED_DIR/rag_llm.json" \
    --retrieval-cache "$SEED_DIR/zero_shot_bge_top50.json" \
    --embedding-model "$BGE_MODEL" --ollama-model "$OLLAMA_MODEL" \
    --ollama-host "$OLLAMA_HOST" --retrieval-k 50 --final-k 5 \
    --rag-max-headers 12 --num-ctx 32768 --fill-short-output \
    --eval-ratio "$EVAL_RATIO" --seed "$SPLIT_SEED" --resume

  python baselines/end_to_end_rag_llm.py \
    --method llm_full --data-dir "$DATA_DIR" \
    --output "$SEED_DIR/llm_only.json" \
    --ollama-model "$OLLAMA_MODEL" --ollama-host "$OLLAMA_HOST" \
    --final-k 5 --full-max-headers 3 --num-ctx 32768 \
    --fill-short-output --eval-ratio "$EVAL_RATIO" \
    --seed "$SPLIT_SEED" --resume

  python scripts/full_corpus/bge_retrain_experiments.py \
    --mode cross --model "$BGE_MODEL" --data-dir "$DATA_DIR" \
    --output-dir "$SEED_DIR" --cache "$TUNED_CACHE" \
    --initial-state "$SEED_DIR/finetuned_bge_last4.pt" \
    --candidate-k 50 --epochs 3 --batch-size 64 --learning-rate 2e-4 \
    --max-length 256 --eval-ratio "$EVAL_RATIO" \
    --split-seed "$SPLIT_SEED" --seed "$SEED"
done
