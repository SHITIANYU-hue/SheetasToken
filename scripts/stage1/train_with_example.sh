#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/stage1/train_with_example.sh
#
# Optional overrides:
#   MODEL_NAME=bert-base-uncased
#   DATA_DIR=data
#   BEST_MODEL_DIR=best_model_with_example
#   FINAL_MODEL_DIR=final_model_with_example
#   TB_DIR=runs/stage1_biencoder_with_example

cd "$(dirname "$0")/../.."

MODEL_NAME="${MODEL_NAME:-bert-base-uncased}"
DATA_DIR="${DATA_DIR:-data}"
BEST_MODEL_DIR="${BEST_MODEL_DIR:-best_model_with_example}"
FINAL_MODEL_DIR="${FINAL_MODEL_DIR:-final_model_with_example}"
TB_DIR="${TB_DIR:-runs/stage1_biencoder_with_example}"

echo "Using model: ${MODEL_NAME}"
echo "Using data dir: ${DATA_DIR}"

python models/stage1/biencoder_model_with_example.py \
  --data-dir "${DATA_DIR}" \
  --model-name "${MODEL_NAME}" \
  --train-features-file "${DATA_DIR}/sheets.json" \
  --eval-features-file "${DATA_DIR}/sheets.json" \
  --use-tensorboard \
  --tensorboard-logdir "${TB_DIR}" \
  --best-model-dir "${BEST_MODEL_DIR}" \
  --final-model-dir "${FINAL_MODEL_DIR}" \
  --num-epochs 50 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-length 256 \
  --embedding-strategy cls
