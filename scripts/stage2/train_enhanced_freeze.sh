#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/stage2/train_enhanced_freeze.sh
#
# Optional overrides:
#   MODEL_NAME=bert-base-uncased
#   DATA_DIR=data
#   STAGE1_CKPT=best_model_with_example/classifier.pt
#   OUTPUT_DIR=outputs/stage2_enhanced_with_example_freeze
#   TB_DIR=runs/stage2_enhanced_with_example_freeze

cd "$(dirname "$0")/../.."

MODEL_NAME="${MODEL_NAME:-bert-base-uncased}"
DATA_DIR="${DATA_DIR:-data}"
STAGE1_CKPT="${STAGE1_CKPT:-best_model_with_example/classifier.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage2_enhanced_with_example_freeze}"
TB_DIR="${TB_DIR:-runs/stage2_enhanced_with_example_freeze}"

echo "Using model: ${MODEL_NAME}"
echo "Using Stage1 checkpoint: ${STAGE1_CKPT}"
echo "Using data dir: ${DATA_DIR}"

python models/stage2/stage2_gtn_v2.py \
  --run-name stage2_enhanced_with_example_freeze \
  --output-dir "${OUTPUT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --features-file "${DATA_DIR}/sheets.json" \
  --model-name "${MODEL_NAME}" \
  --stage1-checkpoint "${STAGE1_CKPT}" \
  --freeze-backbone \
  --use-tensorboard \
  --tensorboard-logdir "${TB_DIR}" \
  --num-epochs 50 \
  --batch-size 8 \
  --learning-rate 1.5e-5 \
  --max-length 256 \
  --max-query-length 64 \
  --max-workspace-size 10 \
  --max-header-texts 12 \
  --eval-ratio 0.2 \
  --embedding-strategy cls \
  --normalize-embeddings \
  --num-gcn-layers 1 \
  --gtn-layers 2 \
  --gtn-channels 4 \
  --negative-ratio 0.5 \
  --tau 0.07 \
  --lambda-align 0.10 \
  --lambda-node 0.10 \
  --pos-weight 2.0 \
  --seed 42
