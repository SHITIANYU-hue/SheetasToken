# Reproducing the Sheet as Token paper

This directory is the canonical, step-by-step entry point for reproducing the
KDD paper. It covers both datasets, every method in the main comparison,
three-seed aggregation, sensitivity figures, and online latency.

## 1. What is reproduced

| Paper method/result | Reproduction entry point |
|---|---|
| LLM-only (Qwen3.5-9B) | `02_run_comparisons.sh` |
| RAG (zero-shot BGE) | `02_run_comparisons.sh` |
| RAG+LLM | `02_run_comparisons.sh` |
| SAT Stage 1 (fine-tuned BGE) | `01_train_sat.sh` |
| SAT Cross-Encoder | `02_run_comparisons.sh` |
| SAT gated GNN | `01_train_sat.sh` |
| Candidate/depth/gate/channel sensitivity | `03_run_sensitivity.sh` |
| Accuracy and latency table | `04_benchmark_latency.sh` |
| Mean and sample standard deviation | `05_aggregate_results.sh` |
| Stage 1/Stage 2 training curves | `05_aggregate_results.sh` |

The release uses exactly the paper serialization: sheet name, shape, and up
to 12 column headers. Cell and example values are not required.

## 2. Environment

Use Python 3.11 or 3.12 and a CUDA GPU for the reported training and timing
protocol:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download `BAAI/bge-base-en-v1.5` once and point `BGE_MODEL` to the local
snapshot. The experiment code deliberately uses `local_files_only=True` to
prevent an unnoticed backbone revision during reproduction.

Install Ollama and make the paper model available as `qwen3.5:latest`. The
reported server artifact was the 9B Q4_K_M variant; verify your Ollama model
metadata before comparing latency.

```bash
export BGE_MODEL=/absolute/path/to/bge-base-en-v1.5
export OLLAMA_MODEL=qwen3.5:latest
export OLLAMA_HOST=http://127.0.0.1:11434
export RUN_ROOT=outputs/paper_reproduction
```

## 3. Validate the public release

```bash
bash scripts/reproduce_paper/00_validate_release.sh
```

This validates the metadata-only schemas, checks both corpus sizes, compiles
the Python entry points, runs the unit tests, and performs a 1,002-sheet
full-context dry run.

## 4. Train SAT on both datasets

```bash
bash scripts/reproduce_paper/01_train_sat.sh industrytab_614
bash scripts/reproduce_paper/01_train_sat.sh industrytab_1k
```

The script runs seeds 42, 43, and 44 and writes every checkpoint/cache beneath
`$RUN_ROOT/<dataset>/seed<seed>/`.

The protocol is encoded rather than left to manual interpretation:

| Setting | IndustryTab-614 | IndustryTab-1K |
|---|---:|---:|
| Sheets / queries | 614 / 1,453 | 1,002 / 1,797 |
| Test split | 20%, separately for seeds 42--44 | 10%, fixed split seed 42 |
| Training seeds | 42, 43, 44 | 42, 43, 44 |
| Stage 1 | final four BGE layers, 3 epochs | same |
| Candidate pool | real full-corpus top-50 | same |
| Stage 2 base/GNN | 20 / 20 epochs | same |
| GNN residual gate | -6 | -4 (seed 42), -2 (seeds 43/44) |
| Dependency channels | aggregation, formula, summary | formula, summary |
| Positive injection | none | none |

`--split-seed` is separate from `--seed` so the fixed IndustryTab-1K split
does not accidentally change across training seeds.

The IndustryTab-1K gate values above follow the archived runs used to compute
the paper's reported 0.9222 mean: the seed-42 sensitivity reference uses
gate -4, while the validation-selected seed-43/44 runs use gate -2. Encoding
this per-seed setting is necessary to reproduce the published aggregate.

## 5. Run every main-table comparison

Start Ollama, then run:

```bash
bash scripts/reproduce_paper/02_run_comparisons.sh industrytab_614
bash scripts/reproduce_paper/02_run_comparisons.sh industrytab_1k
```

This uses the exact evaluation indices stored in each Stage 1 cache. RAG
retrieves from the complete corpus, RAG+LLM reranks its real top-50, and
LLM-only receives the compact catalog of all sheet names. `--resume` makes
the long Ollama runs restartable.

## 6. Reproduce sensitivity results and figures

After the IndustryTab-1K seed-42 run exists:

```bash
bash scripts/reproduce_paper/03_run_sensitivity.sh
```

It runs top-K (10/20/50), depth (1/2/3), gate (-4/-6/-8), and dependency
channel variants, then creates:

- `bge_sensitivity_metrics.pdf`
- `bge_sensitivity_convergence.pdf`
- PNG previews and `bge_sensitivity_summary.json`

under `$RUN_ROOT/industrytab_1k/sensitivity/figures/`.

## 7. Reproduce latency

```bash
bash scripts/reproduce_paper/04_benchmark_latency.sh industrytab_614 42
bash scripts/reproduce_paper/04_benchmark_latency.sh industrytab_1k 42
```

The paper uses an NVIDIA A40, online batch size one, 20 warmups, and three
timed repeats. Accuracy should reproduce across compatible CUDA hardware;
latency is hardware- and Ollama-build-dependent and should only be compared
directly on the reported A40 setup.

## 8. Aggregate the tables

```bash
bash scripts/reproduce_paper/05_aggregate_results.sh
```

The resulting `summary.json` files contain each seed plus the arithmetic mean
and sample standard deviation used by the paper. This step also regenerates
`$RUN_ROOT/figures/full_corpus_training_curves.pdf` and its PNG preview from
the recorded Stage 1, MLP, and GNN histories.

## One-command execution

After configuring BGE and Ollama, the full expensive workflow is:

```bash
bash scripts/reproduce_paper/run_all.sh
```

This performs GPU training and many local-LLM calls; use the numbered scripts
individually when running on a scheduler.

## Expected headline checks

Small numerical variation can occur across CUDA/PyTorch versions. The paper's
headline three-seed NDCG@5 values are approximately:

| Dataset | Zero-shot BGE RAG | SAT |
|---|---:|---:|
| IndustryTab-614 | 0.6344 | 0.9173 |
| IndustryTab-1K | 0.6287 | 0.9222 |

If results differ materially, first check the dataset directory, split seed,
BGE snapshot, Ollama model size/quantization, candidate K, and dependency
channels recorded in each output JSON.
