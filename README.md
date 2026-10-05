# Spreadsheet Retrieval Code

This repository contains code for two-stage retrieval over multiple spreadsheet
sheets. A BGE-based encoder retrieves candidates from the complete corpus, and a
gated relational graph model reranks the retrieved top-50 sheets.

The encoder serializes sheet names, dimensions, and column headers. Corpus
embeddings can be cached and reused across queries. The main retrieval path
uses metadata rather than cell contents or example values.

## Setup

```bash
pip install -r requirements.txt
```

The full-corpus scripts use `BAAI/bge-base-en-v1.5`. Model snapshots and
checkpoints are supplied separately. The optional local-LLM baselines require
an Ollama installation and the selected model.

## Training and evaluation

Use [scripts/reproduce_paper/README.md](scripts/reproduce_paper/README.md) for
release validation, training, comparison methods, three-seed aggregation,
sensitivity analysis, and latency measurement.

The main code entry points are:

- [scripts/full_corpus/](scripts/full_corpus/): encoder training, candidate
  scoring, gated graph refinement, and latency measurement.
- [baselines/end_to_end_rag_llm.py](baselines/end_to_end_rag_llm.py): full-corpus
  embedding retrieval and local-LLM comparison methods.
- [scripts/rebuttal/README.md](scripts/rebuttal/README.md): matched graph
  controls, representation comparisons, grouped splits, and checkpoint replay.
- [api/README_api.md](api/README_api.md): optional retrieval API.

The `models/stage1`, `models/stage2`, and `scripts/stage1`/`stage2` directories
contain legacy implementations. Use `scripts/full_corpus` for the current
retrieval pipeline.

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
checkpoints, and result archives are not included. See [data/README.md](data/README.md)
for schemas and validation commands.

## Local artifacts

Training results, checkpoints, TensorBoard logs, and local archive configuration
are ignored by Git. Historical checkpoint replay requires separately supplied
archives and a path manifest; see the supplemental experiment guide and its
configuration template.
