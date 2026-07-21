#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-gpt-5.4-mini}"
OUTPUT="${OUTPUT:-outputs/baselines/llm_gpt_5_4_mini.json}"

python -m baselines.llm_selector \
  --model "${MODEL}" \
  --output "${OUTPUT}" \
  --top-k 10 \
  --metric-ks 1,3,5,10 \
  --resume

