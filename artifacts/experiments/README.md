# Supplemental experiment evidence

This directory provides experiment measurements and predictions for inspection.
It contains benchmark query text, sheet IDs, rankings, metrics, split definitions,
input hashes, and replay checks. It does not contain raw spreadsheets, cell
examples, checkpoint tensors, cache tensors, or server logs.

| File | Contents |
|---|---|
| [manifest.json](manifest.json) | Original input/code hashes, six original corpus/seed runs, checkpoint/cache mappings, arguments, and public-input hashes. Archive paths use `/path/to/archive_root` placeholders. |
| [results_summary.json](results_summary.json) | Aggregate and subgroup results, paired bootstrap intervals, replay checks, graph controls, representation/CSV diagnostics, transfer, and grouped experiments. |
| [per_query_predictions.json](per_query_predictions.json) | 1,407 original query/seed evaluations, including five methods, gold sets, hard negatives, ranked IDs, and candidate-loss/expansion diagnostics. |
| [retrieval_cases.json](retrieval_cases.json) | Four outcome-selected retrieval illustrations, preserving the original prediction records and adding sheet names and selected candidate relations touching gold sheets. Includes graph-control comparisons and source hashes. |
| [grouped_per_query.json](grouped_per_query.json) | 711 family/schema-held-out query/seed evaluations with Stage 1, scorer, and full graph rankings and metrics. |
| [representation_per_query.json](representation_per_query.json) | 537 query/seed evaluations from separately trained metadata, column, and example input views. |
| [grouped_splits/seed42.json](grouped_splits/seed42.json), [seed43.json](grouped_splits/seed43.json), [seed44.json](grouped_splits/seed44.json) | Family/schema components, sheet/query partitions, excluded cross-partition queries, overlap checks, and input hashes. |
| [export_checks.json](export_checks.json) | Export checks, record counts, and SHA-256 hashes of the eight evidence files. |

`query_index` is the zero-based index in the corresponding released `query.json`.
Sheet IDs resolve against that corpus's `sheets.json`. Gold sets and hard-negative
sets are the original annotations. Repeated training-seed evaluations are not
distinct queries. Grouped and separately trained representation predictions use
IndustryTab-1K; original predictions specify their corpus in `dataset`.

In `results_summary.json`, `analysis` holds original aggregate/subgroup metrics;
`audit` holds graph counts and replay checks; `followup` holds separately trained
representation and grouped results. `representation` contains inference-time and
CSV diagnostics, and `transfer` contains expanded-corpus transfer results.
The provided labels measure relevant-sheet retrieval rather than verified
answer sufficiency or executed formula/join correctness.

Historical input hashes in `manifest.json` identify the archived experiments.
The separate `public_input_sha256` hashes identify the currently released files;
public sheet metadata excludes private example values. Matching serialization
checks concern the main name/shape/header inputs, not byte-identical raw metadata.
Paths identify archive layout without exposing deployment locations. Original
checkpoints, caches, and private example/CSV inputs must be supplied separately
to rerun replay and value-feature diagnostics.

The [experiment commands](../../scripts/experiments/README.md) explain training and
replay. To regenerate this export from local experiment artifacts, run:

```bash
python scripts/experiments/export_public_artifacts.py
python scripts/experiments/export_retrieval_cases.py
```

The exporter replaces private roots, omits server deployment/backup metadata,
checks identity/endpoint/credential patterns, verifies every exported query and
available gold/hard-negative set against the public benchmark, and verifies
ranked sheet IDs. Numeric results and prediction IDs are preserved.
