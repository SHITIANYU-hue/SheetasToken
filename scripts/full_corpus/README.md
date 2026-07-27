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
for IndustryTab-1K. These expanded datasets are not included in this public
code repository. `gated_graph_refine.py` consumes the tuned cache and base
scorer checkpoint.

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
