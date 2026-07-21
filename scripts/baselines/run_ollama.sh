#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-qwen2.5:1.5b-instruct-q4_K_M}"
OUTPUT="${OUTPUT:-outputs/baselines/llm_ollama_qwen2_5_1_5b_q4.json}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

python -m baselines.llm_selector \
  --provider ollama \
  --candidate-scope labeled \
  --model "${MODEL}" \
  --ollama-host "${OLLAMA_HOST}" \
  --ollama-num-ctx 8192 \
  --output "${OUTPUT}" \
  --top-k 10 \
  --metric-ks 1,3,5,10 \
  --resume
