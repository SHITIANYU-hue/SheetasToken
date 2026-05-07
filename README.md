# SheetAgent Paper Repository

This repository contains the code for our two-stage spreadsheet retrieval pipeline, including the Stage 1 sheet encoder, the Stage 2 graph retriever, and the experiment scripts used in the paper.

## Overview

Our system separates spreadsheet understanding into two stages:

- **Stage 1: Sheet Token Encoder**
  - Learns reusable sheet-level representations from pairwise sheet supervision.
  - Supports two main variants:
    - `with_example`: sheet serialization includes column examples
    - `wo_example`: sheet serialization excludes column examples

- **Stage 2: Graph Retriever**
  - Performs query-conditioned cross-sheet retrieval over a candidate workspace.
  - Supports two main variants:
    - `baseline`: shallower graph retriever
    - `enhanced`: graph-enhanced retriever with stronger relational composition

The final paper model uses:

- **Stage 1 with examples**
- **Stage 2 enhanced**
- **frozen Stage 1 encoder during Stage 2 training**

---

## Repository Structure

```text
.
├── api/                                  # Optional API serving code
├── configs/                              # Configuration files
├── data/                                 # Training / evaluation data
├── docs/                                 # Notes or documentation
├── models/
│   ├── stage1/
│   │   ├── biencoder_model.py            # Legacy Stage 1 baseline (reference only)
│   │   ├── biencoder_model_with_example.py
│   │   └── biencoder_model_wo_example.py
│   └── stage2/
│       ├── stage2_gtn_baseline.py
│       └── stage2_gtn_v2.py
├── scripts/
│   ├── stage1/
│   │   ├── train_with_example.sh
│   │   └── train_wo_example.sh
│   └── stage2/
│       ├── train_baseline_freeze.sh
│       └── train_enhanced_freeze.sh
├── utils/                                # Utility functions
├── requirements.txt
└── README.md
```

---

## Main Files

### Stage 1
- `models/stage1/biencoder_model_with_example.py`  
  Stage 1 encoder using example-enhanced sheet serialization.

- `models/stage1/biencoder_model_wo_example.py`  
  Stage 1 encoder without column examples.

- `models/stage1/biencoder_model.py`  
  Legacy / early Stage 1 baseline, kept for reference only.  
  Current paper experiments use the two variants above.

### Stage 2
- `models/stage2/stage2_gtn_baseline.py`  
  Shallow graph retriever used as the architecture ablation / shadow model.

- `models/stage2/stage2_gtn_v2.py`  
  Enhanced graph retriever used as the full model.

---

## Data Format

The code expects the dataset under `data/`.

Typical files include:

- `data/sheets.json`  
  Sheet metadata and serialized sheet content.

- `data/train.json`  
  Pairwise Stage 1 supervision data.

- `data/query.json`  
  Query-conditioned Stage 2 retrieval data.

Adjust paths if your local setup differs.

---

## Environment Setup

Install dependencies first:

```bash
pip install -r requirements.txt
```

The scripts default to the Hugging Face model name `bert-base-uncased`.

If you want to use a local pretrained model snapshot, you can override `MODEL_NAME` when running a script.

Example:

```bash
MODEL_NAME=/path/to/local/model bash scripts/stage2/train_enhanced_freeze.sh
```

---

## Training Scripts

### Stage 1

Train Stage 1 with example-enhanced serialization:

```bash
bash scripts/stage1/train_with_example.sh
```

Train Stage 1 without column examples:

```bash
bash scripts/stage1/train_wo_example.sh
```

### Stage 2

Train the Stage 2 baseline retriever with frozen Stage 1:

```bash
bash scripts/stage2/train_baseline_freeze.sh
```

Train the Stage 2 enhanced retriever with frozen Stage 1:

```bash
bash scripts/stage2/train_enhanced_freeze.sh
```

---

## Optional Script Overrides

The shell scripts support environment-variable overrides.

Common overrides include:

- `MODEL_NAME`
- `DATA_DIR`
- `STAGE1_CKPT`
- `OUTPUT_DIR`
- `TB_DIR`
- `BEST_MODEL_DIR`
- `FINAL_MODEL_DIR`

Example:

```bash
MODEL_NAME=/path/to/local/model \
STAGE1_CKPT=best_model_with_example/classifier.pt \
bash scripts/stage2/train_enhanced_freeze.sh
```

This makes the scripts usable on both local machines and remote servers without hardcoding machine-specific paths.

---

## Paper Experiment Mapping

### Full Model
- Stage 1: `with_example`
- Stage 2: `enhanced`
- Stage 1 encoder frozen during Stage 2 training

### Architecture Ablation
- Stage 1: `with_example`
- Stage 2: `baseline`
- Stage 1 encoder frozen during Stage 2 training

### Feature Ablation
- Stage 1: `wo_example`
- Stage 2: `enhanced`
- Stage 1 encoder frozen during Stage 2 training

---

## Outputs

Training scripts typically write outputs to:

- `runs/...` for TensorBoard logs
- `outputs/...` for experiment outputs
- `best_model_*` / `final_model_*` for Stage 1 checkpoints

These training artifacts are local experiment outputs and should generally not be committed to Git.

---

## Recommended Git Ignore

A typical `.gitignore` should include at least:

```gitignore
best_model/
best_model_with_example/
best_model_wo_example/
final_model/
final_model_with_example/
final_model_wo_example/
outputs/
runs/
*.log
__pycache__/
```

You can expand this as needed for your environment.


## Citation

If you use this repository, please cite the associated paper.
