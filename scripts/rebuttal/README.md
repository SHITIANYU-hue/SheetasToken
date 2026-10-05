# Supplemental experiments and checkpoint replay

This directory contains matched graph controls, representation comparisons,
family/schema splits, metadata graph reconstruction, and result audits. The
main-paper reproduction commands are in [../reproduce_paper/README.md](../reproduce_paper/README.md).

## Local archive configuration

Historical replay requires the original encoder caches, checkpoints, and saved
baseline predictions. These archives are not included in this repository.
Example-value and CSV diagnostics additionally require the original input files;
the public metadata release omits examples. Install PyTorch, transformers,
NumPy, and pytest in the experiment environment.

Create a local configuration from [archive_manifest.example.json](archive_manifest.example.json):

```bash
mkdir -p docs
cp scripts/rebuttal/archive_manifest.example.json docs/server_archive_manifest.json
```

Replace `/path/to/archive_root` and the individual paths with your archive
locations. Keep the listed archive filenames and layout, or update the paths to
match your copies. The manifest records the six original corpus/seed runs and
their checkpoint arguments. `docs/` is local and ignored by Git. The private
manifest, internal audit notes, manuscript, and response draft are not published
with this code. Scripts accepting `--manifest` can also use a configuration at
another location.

## Graph construction and grouped splits

These commands use the public metadata and graph files:

```bash
for DATASET in industrytab_614 industrytab_1k; do
  python scripts/rebuttal/reconstruct_metadata_graph.py \
    --data-dir "data/$DATASET" \
    --output "outputs/rebuttal/graph_reconstruction/$DATASET/graph.json" \
    --audit-against "data/$DATASET/dependency_edges.json"
done
python scripts/rebuttal/make_grouped_splits.py
```

The name-only reconstruction exactly matches the archived formula, summary,
and aggregation pair sets. Its rules were inferred post hoc; it is not the
recovered historical generator or a validated cell-formula extractor. It reads
no query labels during construction. Historical rule-selection independence
remains unresolved.

Groups merge normalized filename families and identical complete header
signatures without query labels. Complete gold sets then assign queries to
partitions; mixed-partition queries are excluded without relabeling. This is
family/schema isolation within the existing benchmark.

## Training and replay with supplied archives

Run from the repository root. Each training queue uses the selected GPU.

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python scripts/rebuttal/train_dense_only.py --control dense
python scripts/rebuttal/train_dense_only.py --control node_permuted
python scripts/rebuttal/audit_and_analyze.py --audit-encoder
python scripts/rebuttal/audit_baselines.py
python scripts/rebuttal/representation_diagnostics.py
python scripts/rebuttal/transfer_diagnostic.py

python scripts/rebuttal/run_followup.py --queue views_a
python scripts/rebuttal/run_followup.py --queue views_b
python scripts/rebuttal/run_followup.py --queue grouped
python scripts/rebuttal/audit_followup.py

python -m pytest -q tests/test_rebuttal_metrics.py \
  tests/test_grouped_split_integrity.py tests/test_representation_training.py
```

`train_dense_only.py` reuses existing result JSONs. To retrain, use a fresh
`--output-dir` and supply the corresponding `--dense-root` / `--permuted-root`
to the audit. Original archives are not overwritten. Full results, predictions,
checks, and logs are written under the ignored `outputs/rebuttal/` directory.
The gradient-replay test requires CUDA; the grouped-split test requires the
generated split artifacts.

## Interpretation of the controls

- Dense-only GNN retains four dense channels and the frozen scorer but removes
  explicit relations; it consequently has fewer trainable graph parameters.
- Node permutation retains all selected channels and model capacity. It
  preserves graph topology and global degree distribution, but changes the
  degree assigned to each named sheet. It is not fixed-node degree-preserving
  edge rewiring.
- Paired bootstrap resamples query-ID clusters, keeping observed seed outcomes
  together. Subgroup intervals are exploratory without multiplicity correction.
- Inference-time column/example diagnostics reuse the archived encoder.
  Separately trained sheet, column, and example views use three seeds, matched
  query/update budgets, three epochs, a 256-token limit, and gradient caching.
  Their token computation differs.
- CSV diagnostics use 192 verified sheets and 25 eligible original test queries.
  Adaptive blocks cover all rows without truncation. Cell language and the small
  subset limit extrapolation to the complete benchmark.
- Transfer across the overlapping 614/1K corpora does not establish independent
  domain generalization. Grouped runs train new encoder/scorer/GNN pipelines;
  train-sheet negatives exclude held-out identities and test search uses all
  1,002 sheets.
- Baseline audits replay saved predictions and cross-encoder checkpoints.
  They do not rerun LLM generation or establish new isolated latency results.

## Original-path BGE baseline comparison

The archived baseline used SentenceTransformers FP32; the training cache used
an AutoModel mixed-precision path. Their small ranking differences should not
be treated as interchangeable baseline runs. To replay the original path:

```bash
python baselines/end_to_end_rag_llm.py --method rag \
  --data-dir data/industrytab_1k \
  --output outputs/rebuttal/baseline_precision/fp32_sentence_transformer_local.json \
  --embedding-model BAAI/bge-base-en-v1.5 --batch-size 64 --local-files-only
python scripts/rebuttal/compare_baseline_replay.py \
  --original /path/to/original/bge_full_corpus_top5.json
```

The verified original replay used SentenceTransformers 5.6.0 and PyTorch 2.5.1
on MPS. Supply the cached model and original result archive separately.
