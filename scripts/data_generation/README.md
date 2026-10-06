# Query construction source

[gen_query.py](gen_query.py) preserves the original query-template script byte
for byte. The source is available directly in this release; its original
project path was `dataset/training_data_new/gen_query.py`.

Source SHA-256:
`959d94ba951540303f423e0cb9b012144c05721d91f291ab220dcee9c2027108`

The script parses sheet names into company, year, quarter, and category groups,
constructs query templates and positive-set unions, and samples negatives from
IDs outside the positive set using seed 42. The Q1 sales and attendance/salary
examples discussed in the dataset description are visible in the source.
Several cross-category templates use bounded subsets of groups; the script
does not validate exhaustive support or execute a spreadsheet answer.

## Run on the public legacy snapshot

The script expects `sheets.json` and `query.json` in its working directory and
uses the legacy `sheet_ids` / `sheet_ids_negative` fields. It writes the combined
query collection back to that working directory. The following command copies
inputs to a temporary directory, leaving the released benchmark unchanged:

```bash
python3 - <<'PY'
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path.cwd()
work = Path(tempfile.mkdtemp(prefix="sat-query-generation-"))
shutil.copy2(root / "data/industrytab_614/sheets.json", work / "sheets.json")
shutil.copy2(root / "data/industrytab_614/query_legacy_134.json", work / "query.json")
subprocess.run([sys.executable, str(root / "scripts/data_generation/gen_query.py")],
               cwd=work, check=True)
print(f"Generated queries: {work / 'query.json'}")
PY
```

This source makes the original grouping/template logic inspectable. It is not
an end-to-end generator for the released expanded 1,453/1,797-query collections,
which use `positive_sheet_ids` / `negative_sheet_ids` and additional workload
annotations. Those exact evaluated collections are provided under
[data/](../../data/README.md). Query generation is separate from
[selected graph reconstruction](../experiments/reconstruct_metadata_graph.py).
