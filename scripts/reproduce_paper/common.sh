#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BGE_MODEL="${BGE_MODEL:-}"
RUN_ROOT="${RUN_ROOT:-outputs/paper_reproduction}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:latest}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

require_bge_model() {
  if [[ -z "$BGE_MODEL" || ! -d "$BGE_MODEL" ]]; then
    echo "Set BGE_MODEL to a downloaded BAAI/bge-base-en-v1.5 directory." >&2
    exit 2
  fi
}

dataset_config() {
  local dataset="$1"
  case "$dataset" in
    industrytab_1k)
      DATA_DIR="data/industrytab_1k"
      EVAL_RATIO="0.1"
      FIXED_SPLIT_SEED="42"
      GATE_INIT="-4"
      DEPENDENCY_TYPES="formula_reference,summary_source"
      ;;
    industrytab_614)
      DATA_DIR="data/industrytab_614"
      EVAL_RATIO="0.2"
      FIXED_SPLIT_SEED=""
      GATE_INIT="-6"
      DEPENDENCY_TYPES="aggregation,formula_reference,summary_source"
      ;;
    *)
      echo "Unknown dataset: $dataset (expected industrytab_614 or industrytab_1k)" >&2
      exit 2
      ;;
  esac
  DATASET_RUN_DIR="$RUN_ROOT/$dataset"
}

split_seed_for() {
  local training_seed="$1"
  if [[ -n "$FIXED_SPLIT_SEED" ]]; then
    echo "$FIXED_SPLIT_SEED"
  else
    echo "$training_seed"
  fi
}

gate_init_for() {
  local dataset="$1"
  local training_seed="$2"
  if [[ "$dataset" == "industrytab_1k" && "$training_seed" != "42" ]]; then
    echo "-2"
  else
    echo "$GATE_INIT"
  fi
}
