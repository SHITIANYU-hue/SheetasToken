#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

python data/prepare_public_metadata.py --check
python data/prepare_public_metadata.py \
  --input data/industrytab_614/sheets.json --check
python data/prepare_public_metadata.py \
  --input data/industrytab_1k/sheets.json --check
python -m compileall -q baselines scripts/full_corpus data
PYTHONPATH=. pytest -q tests

python baselines/end_to_end_rag_llm.py \
  --method llm_full --data-dir data --output /tmp/sheetastoken_smoke.json \
  --dry-run --max-queries 1
