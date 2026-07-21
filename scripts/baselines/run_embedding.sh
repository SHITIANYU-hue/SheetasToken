#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL_NAME="${MODEL_NAME:-BAAI/bge-base-en-v1.5}"
OUTPUT="${OUTPUT:-outputs/baselines/embedding_bge_base.json}"

python -m baselines.embedding_retrieval \
  --model-name "${MODEL_NAME}" \
  --output "${OUTPUT}" \
  --top-k 10 \
  --metric-ks 1,3,5,10

