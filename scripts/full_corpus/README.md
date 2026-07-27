# Strict full-corpus experiments

These scripts implement the deployable two-stage experiment used in the KDD
paper:

1. fine-tune `BAAI/bge-base-en-v1.5` with query--sheet contrastive learning;
2. retrieve top-50 candidates from the complete corpus;
3. train a listwise base scorer;
4. initialize a gated relational GNN from that scorer and refine the same
   candidates without injecting positives.

`bge_retrain_experiments.py` supports cache construction, BGE fine-tuning,
base/graph reranking, and a cross-encoder diagnostic. Use `--eval-ratio 0.2`
for the current 1,453-query IndustryTab-614 workload and `--eval-ratio 0.1`
for the 1,797-query IndustryTab-1K workload. The metadata-only variants are
included at `data/industrytab_614/` (small/original, 614 sheets) and
`data/industrytab_1k/` (large/expanded, 1,002 sheets), respectively.
`gated_graph_refine.py` consumes the tuned cache and base scorer checkpoint.

`gated_graph_refine.py` also contains optional score-message and calibration
utilities for controlled follow-up experiments. They are disabled by default
and are not part of the released paper results.

The selected dependency channels were chosen using seed-42 validation NDCG
and then fixed:

| Dataset | Dependency channels |
|---|---|
| IndustryTab-614 | aggregation, formula-reference, summary-source |
| IndustryTab-1K | formula-reference, summary-source |

Paper-facing metrics and model checkpoints are maintained separately from this
public code-only repository.

## Reproduce the default IndustryTab-1K pipeline

The commands below rebuild the paper pipeline from the public metadata. The
implementation intentionally loads the Hugging Face model with
`local_files_only=True`; set `BGE_MODEL` to a downloaded
`BAAI/bge-base-en-v1.5` snapshot.

```bash
export BGE_MODEL=/path/to/BAAI/bge-base-en-v1.5
export RUN_DIR=outputs/reproduce_industrytab_1k

# Zero-shot full-corpus BGE cache (fixed seed-42 90/10 split).
python scripts/full_corpus/bge_retrain_experiments.py \
  --mode cache --model "$BGE_MODEL" --data-dir data \
  --output-dir "$RUN_DIR" --cache "$RUN_DIR/base_bge_cache.pt" \
  --candidate-k 50 --eval-ratio 0.1 --seed 42

# Query--sheet fine-tuning of the final four BGE layers.
python scripts/full_corpus/bge_retrain_experiments.py \
  --mode finetune --model "$BGE_MODEL" --data-dir data \
  --output-dir "$RUN_DIR" --cache "$RUN_DIR/base_bge_cache.pt" \
  --candidate-k 50 --epochs 3 --eval-ratio 0.1 --split-seed 42 --seed 42

# Listwise base scorer used to initialize the graph model.
python scripts/full_corpus/bge_retrain_experiments.py \
  --mode rerank --architecture mlp --model "$BGE_MODEL" --data-dir data \
  --output-dir "$RUN_DIR/seed42" \
  --cache "$RUN_DIR/finetuned_bge_cache.pt" \
  --candidate-k 50 --eval-ratio 0.1 --seed 42

# Paper GNN: repeat with seeds 42, 43, and 44.
for SEED in 42 43 44; do
  GATE_INIT=-4
  if [ "$SEED" != 42 ]; then GATE_INIT=-2; fi
  python scripts/full_corpus/gated_graph_refine.py \
    --cache "$RUN_DIR/finetuned_bge_cache.pt" \
    --mlp-checkpoint "$RUN_DIR/seed42/frozen_bge_mlp_top50.pt" \
    --output-dir "$RUN_DIR/gnn_seed${SEED}" --name "gnn_seed${SEED}" \
    --candidate-k 50 --layers 1 --gate-init "$GATE_INIT" \
    --dependency-edges data/dependency_edges.json \
    --dependency-types formula_reference,summary_source \
    --seed "$SEED"
done
```

For IndustryTab-614, replace `--data-dir data` with
`--data-dir data/industrytab_614`, use `--eval-ratio 0.2`, and select
`aggregation,formula_reference,summary_source` with gate initialization -6.
The paper reports means and standard deviations over seeds 42, 43, and 44.

For the complete commands, including the fixed IndustryTab-1K split,
validation-selected per-seed gate settings, comparison methods, and result
aggregation, use [`../reproduce_paper/README.md`](../reproduce_paper/README.md).

The Cross-Encoder diagnostic uses the same tuned cache:

```bash
python scripts/full_corpus/bge_retrain_experiments.py \
  --mode cross --model "$BGE_MODEL" --data-dir data \
  --output-dir "$RUN_DIR/cross_seed42" \
  --cache "$RUN_DIR/finetuned_bge_cache.pt" \
  --initial-state "$RUN_DIR/finetuned_bge_last4.pt" \
  --candidate-k 50 --epochs 3 --eval-ratio 0.1 --split-seed 42 --seed 42
```
