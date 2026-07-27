# Public metadata-only snapshot

The public snapshot contains structural metadata and relevance supervision,
not spreadsheet cell contents.

## Dataset variants

| Directory | Scale | Sheets | Current queries | Legacy queries | Training pairs | Dependency edges |
|---|---|---:|---:|---:|---:|---:|
| `industrytab_614/` | small/original | 614 | 1,453 | 134 | 1,842 | 10,775 |
| `industrytab_1k/` | large/expanded | 1,002 | 1,797 | -- | 3,006 | 19,850 |

IndustryTab-1K expands the original corpus; the two variants are not intended
to be disjoint datasets. [`datasets.json`](datasets.json) records the same
distinction in a machine-readable form.

The main files directly under `data/` (`sheets.json`, `query.json`,
`train.json`, and `dependency_edges.json`) are compatibility copies of the
large IndustryTab-1K variant, making 1,002 sheets the repository default.
`query_legacy_134.json` remains a provenance-only snapshot from
IndustryTab-614. Use an explicit dataset directory when reproducing a
particular table:

```bash
# Default / large corpus
--data-dir data
# Equivalent explicit path
--data-dir data/industrytab_1k
# Small/original corpus
--data-dir data/industrytab_614
```

Each dataset directory contains:

| File | Public contents |
|---|---|
| `sheets.json` | sheet ID, sheet name, dimensions, column names, source tag |
| `query.json` | current full-corpus query, relevance, hard-negative, and dependency annotations |
| `query_legacy_134.json` | obsolete arXiv query snapshot; IndustryTab-614 only |
| `dependency_edges.json` | typed sheet-ID relations for gated-GNN experiments |
| `train.json` | sheet-pair IDs, labels, and optional matched column-name pairs |

## What is deliberately excluded

- raw XLSX/CSV files;
- cell values and example values;
- formulas, comments, charts, formatting, and embedded objects;
- value statistics or row-level content.

The reported BGE, Cross-Encoder, RAG/LLM, and gated-GNN paths serialize only
the sheet name, dimensions, and up to 12 column names. The GNN additionally
uses `dependency_edges.json`; this graph contains sheet IDs and typed edges,
not cell values.

## Validate the release

```bash
python data/prepare_public_metadata.py \
  --input data/industrytab_614/sheets.json --check

python data/prepare_public_metadata.py \
  --input data/industrytab_1k/sheets.json --check
```

To sanitize another compatible `sheets.json`:

```bash
python data/prepare_public_metadata.py \
  --input /path/to/private/sheets.json \
  --output /path/to/public/sheets.json
```
