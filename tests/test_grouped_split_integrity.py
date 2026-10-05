"""Verify held-out family/schema identities and original gold sets stay isolated."""
import importlib.util,json
import pytest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
@pytest.mark.skipif(not (ROOT/'outputs/rebuttal/grouped_splits/seed42.json').exists(),reason='requires generated grouped split artifacts')
def test_grouped_partition_has_no_family_or_schema_overlap():
    module_path=ROOT/'scripts/rebuttal/make_grouped_splits.py';spec=importlib.util.spec_from_file_location('grouping',module_path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    sheets=json.loads((ROOT/'data/industrytab_1k/sheets.json').read_text());queries=json.loads((ROOT/'data/industrytab_1k/query.json').read_text())
    for seed in [42,43,44]:
        split=json.loads((ROOT/f'outputs/rebuttal/grouped_splits/seed{seed}.json').read_text());partition={k:set(split[k+'_sheet_ids']) for k in ['train','val','test']}
        for key1,key2 in [('train','val'),('train','test'),('val','test')]:
            a=partition[key1];b=partition[key2];assert not a&b
            assert not {m.family(sheets[i]['name']) for i in a}&{m.family(sheets[i]['name']) for i in b}
            sig=lambda i:tuple(sorted(c['name'].strip().lower() for c in sheets[i]['columns']))
            assert not {sig(i) for i in a}&{sig(i) for i in b}
        positions={'train':split['train_core'],'val':split['val_positions'],'test':split['eval_positions']}
        all_positions=[]
        for k,indices in positions.items():
            for i in indices:assert set(map(str,queries[i]['positive_sheet_ids']))<=partition[k]
            all_positions+=indices
        assert len(set(all_positions))==len(all_positions)
        assert set(all_positions)|set(split['excluded_cross_partition_queries'])==set(range(len(queries)))
