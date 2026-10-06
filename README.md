# Spreadsheet Retrieval Code

This repository contains code for two-stage retrieval over multiple spreadsheet
sheets. A BGE-based encoder retrieves candidates from the complete corpus, and a
gated relational graph model reranks the retrieved top-50 sheets.

The encoder serializes sheet names, dimensions, and column headers. Corpus
embeddings can be cached and reused across queries. The main retrieval path
uses metadata rather than cell contents or example values.

## Data

Two metadata-only corpus variants are included:

| Directory | Sheets | Queries |
|---|---:|---:|
| `data/industrytab_614/` | 614 | 1,453 |
| `data/industrytab_1k/` | 1,002 | 1,797 |

The larger corpus expands the smaller one; they are related collections.
Top-level files under `data/` are compatibility copies of the larger corpus.
Use an explicit `--data-dir` to select the variant. The obsolete 134-query file
is retained only for provenance.

Sheet records contain IDs, names, dimensions, and column names. Query files
contain relevance and hard-negative annotations; dependency files contain typed
relations between sheet IDs. Raw spreadsheets, cell examples, model
checkpoints, and complete runtime archives are not included. An anonymized
[supplemental evidence bundle](artifacts/rebuttal/README.md) provides measured
results, per-query predictions, input/run mappings, and grouped split definitions.
See [data/README.md](data/README.md) for schemas and validation commands.

The original [query construction source](scripts/data_generation/README.md)
documents filename grouping, positive-set unions, and negative sampling.
[Retrieval cases](artifacts/rebuttal/retrieval_cases.json) provide concrete
rankings, sheet names, graph controls, and candidate relations for inspecting
beneficial and harmful reranking.


---

## Overview

The retrieval pipeline has two stages:

- **Stage 1: Sheet Encoder** fine-tunes BGE for query--sheet retrieval and
  caches embeddings of sheet names, dimensions, and column headers.
- **Stage 2: Graph Retriever** trains a listwise base scorer and initializes
  a gated relational GNN from that scorer. It reranks real full-corpus top-50
  candidates without inserting annotated positives.

The main path uses metadata only. The legacy implementations additionally
provide example-enhanced and alternative graph variants.

---

## Repository Structure

```text
.
├── api/                                  # Optional API serving code
├── baselines/                            # Embedding and LLM comparisons
├── configs/                              # Configuration files
├── data/                                 # Training / evaluation data
├── models/
│   ├── stage1/
│   │   ├── biencoder_model.py            # Legacy Stage 1 baseline (reference only)
│   │   ├── biencoder_model_with_example.py
│   │   └── biencoder_model_wo_example.py
│   └── stage2/
│       ├── stage2_gtn_baseline.py
│       └── stage2_gtn_v2.py
├── scripts/
│   ├── data_generation/                  # Original query-template source
│   ├── full_corpus/                      # Current encoder and gated GNN
│   ├── rebuttal/                         # Supplemental experiment controls
│   ├── reproduce_paper/                  # Canonical end-to-end reproduction
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

- `scripts/full_corpus/bge_retrain_experiments.py`: BGE adaptation, embedding
  caches, listwise scoring, and cross-encoder comparisons.
- `scripts/full_corpus/gated_graph_refine.py`: gated relational refinement of
  retrieved candidates.
- `baselines/end_to_end_rag_llm.py`: strict full-corpus BGE and local-LLM
  comparison methods.

### Legacy Stage 1
- `models/stage1/biencoder_model_with_example.py`
  Stage 1 encoder using example-enhanced sheet serialization.

- `models/stage1/biencoder_model_wo_example.py`
  Stage 1 encoder without column examples.

- `models/stage1/biencoder_model.py`
  Legacy / early Stage 1 baseline, kept for reference only.
  Current experiments use `scripts/full_corpus/`.

### Legacy Stage 2
- `models/stage2/stage2_gtn_baseline.py`
  Legacy shallow graph retriever for architecture comparisons.

- `models/stage2/stage2_gtn_v2.py`
  Legacy enhanced graph retriever.

---

## Data Format

Current full-corpus experiments may use top-level `data/` for IndustryTab-1K,
or point `--data-dir` (or `DATA_DIR`) explicitly at
`data/industrytab_614` or `data/industrytab_1k`.

Typical files include:

- `data/<dataset>/sheets.json`
  Sheet IDs, names, dimensions, and column headers.

- `data/<dataset>/train.json`
  Historical sheet-pair annotations used by the legacy implementations.

- `data/<dataset>/query.json`
  Queries with relevant-sheet, negative-sheet, and dependency annotations.

The current pipeline trains from query--sheet labels rather than the historical
pair file. `dependency_edges.json` supplies typed sheet relations. Adjust paths
if your local setup differs.

---

## Environment Setup

Install dependencies first:

```bash
pip install -r requirements.txt
```

The current full-corpus pipeline uses `BAAI/bge-base-en-v1.5`. Download a
model snapshot and set `BGE_MODEL` to its local path. CUDA is used for the
training and latency protocol. The legacy shell scripts default to
`bert-base-uncased`.

If you want to use a local pretrained model snapshot, you can override `MODEL_NAME` when running a script.

Example:

```bash
MODEL_NAME=/path/to/local/model bash scripts/stage2/train_enhanced_freeze.sh
```

---

## Training Scripts

### Current Full-Corpus Pipeline

Train both corpus variants with separate encoder, scorer, and graph runs for
seeds 42, 43, and 44:

```bash
export BGE_MODEL=/path/to/BAAI/bge-base-en-v1.5
export RUN_ROOT=outputs/reproduction
bash scripts/reproduce_paper/01_train_sat.sh industrytab_614
bash scripts/reproduce_paper/01_train_sat.sh industrytab_1k
```

See [scripts/full_corpus/README.md](scripts/full_corpus/README.md) for the
individual training commands. For validation, comparison methods, sensitivity
analysis, result aggregation, and latency, follow
[scripts/reproduce_paper/README.md](scripts/reproduce_paper/README.md).

### Legacy Stage 1

Train Stage 1 with example-enhanced serialization:

```bash
bash scripts/stage1/train_with_example.sh
```

Train Stage 1 without column examples:

```bash
bash scripts/stage1/train_wo_example.sh
```

### Legacy Stage 2

Train the Stage 2 baseline retriever with frozen Stage 1:

```bash
bash scripts/stage2/train_baseline_freeze.sh
```

Train the Stage 2 enhanced retriever with frozen Stage 1:

```bash
bash scripts/stage2/train_enhanced_freeze.sh
```

### Zero-shot Baselines

Three no-training comparison systems are included:

- **Frozen embedding retrieval**: `BAAI/bge-base-en-v1.5` cosine retrieval over all sheets.
- **Full-corpus LLM selector**: an OpenAI model selects sheet IDs directly from the complete sheet catalog.
- **Local LLM selector**: a controlled-candidate Ollama run, with an explicit Q4_K_M 1.5B model as the default.

Run them with:

```bash
bash scripts/baselines/run_embedding.sh

export OPENAI_API_KEY=...
bash scripts/baselines/run_llm.sh

bash scripts/baselines/run_ollama.sh
```

These comparison scripts use the same sheet serialization and report Precision, Recall, HitRate, MRR, and nDCG at K. See [`baselines/README.md`](baselines/README.md) for dry runs, cost-safe smoke tests, model overrides, and output details.

---

## Optional Script Overrides

The shell scripts support environment-variable overrides.

Common overrides include:

- `BGE_MODEL` and `RUN_ROOT` for the current full-corpus pipeline
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

## Experiment Entry Points

| Experiment | Entry point |
|---|---|
| Full-corpus encoder, base scorer, and gated GNN | `scripts/reproduce_paper/01_train_sat.sh` |
| Embedding, local-LLM, and cross-encoder comparisons | `scripts/reproduce_paper/02_run_comparisons.sh` |
| Candidate, depth, gate, and relation sensitivity | `scripts/reproduce_paper/03_run_sensitivity.sh` |
| Online latency | `scripts/reproduce_paper/04_benchmark_latency.sh` |
| Three-seed aggregation and plots | `scripts/reproduce_paper/05_aggregate_results.sh` |
| Graph controls, representation views, and grouped splits | `scripts/rebuttal/` |

See [scripts/rebuttal/README.md](scripts/rebuttal/README.md) for supplemental
experiment commands and [api/README_api.md](api/README_api.md) for the
optional retrieval API.

---

## Outputs

Training scripts typically write outputs to:

- `runs/...` for TensorBoard logs
- `outputs/...` for experiment outputs
- `best_model_*` / `final_model_*` for Stage 1 checkpoints
