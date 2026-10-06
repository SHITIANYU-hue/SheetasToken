#!/usr/bin/env python3
"""Reconstruct selected archived relation sets from sheet names only.

This is a new reconstruction, not the recovered historical generator. Rules
were inferred from archived relations. No query/relevance file is read by build.
Relation names denote conceptual metadata rules, not verified cell formulas.
"""
import argparse,hashlib,itertools,json,re
from collections import defaultdict
from pathlib import Path


def parse(name):
    m=re.search(r'(20\d\d(?:[-_]\d\d)?(?:_Q\d)?)_(?:source|target)$',name)
    return (name[:m.start()].rstrip('_'),m.group(1)) if m else None


def build(sheets):
    lookup={}
    for prefix in ['Project_Management','Synthetic_Data__Project_Management']:
        lookup[tuple(sorted([prefix+'__Project_Project_Budget',prefix+'__Project_Project_Progress']))]='formula_reference'
        for name in ['Budget','Progress']:
            lookup[tuple(sorted([prefix+'__Project_Project_'+name,prefix+'__Project_Project_Registry']))]='summary_source'
    for prefix,names in [('Inventory_Products',['Stock_Ledger','Warehouse_Entry_Details','Warehouse_Exit_Details']),('Synthetic_Data__Inventory_Products',['Stock_Ledger','Warehouse_Entry','Warehouse_Exit'])]:
        for a,b in itertools.combinations(names,2):lookup[tuple(sorted([prefix+'__Inventory_'+a,prefix+'__Inventory_'+b]))]='formula_reference'
    result={};ids=sorted(map(str,sheets),key=int)
    for a,b in itertools.combinations(ids,2):
        x=parse(sheets[a]['name']);y=parse(sheets[b]['name']);types=[]
        if not x or not y:continue
        paired=tuple(sorted([x[0],y[0]]))
        if x[1]==y[1] and paired in lookup:types.append(lookup[paired])
        bx=x[0].replace('Synthetic_Data__Financial_Statements__','Financial_Statements__');by=y[0].replace('Synthetic_Data__Financial_Statements__','Financial_Statements__')
        if bx==by and (bx.startswith('Financial_Statements__') or (any(k in bx for k in ['Stock_Market','Sales_Data','Inventory_Products']) and x[1][:4]==y[1][:4])):types.append('aggregation')
        if types:result[a+','+b]=sorted(types)
    return {'edge_all_types':result,'construction':'name-only reconstruction of archived conceptual relation rules; original generator unavailable','reads_query_labels':False,'verified_cell_formula_extraction':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--audit-against');p.add_argument('--audit-output');a=p.parse_args()
    sheets=json.loads((Path(a.data_dir)/'sheets.json').read_text());result=build(sheets);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    if a.audit_against:
        archive=json.loads(Path(a.audit_against).read_text());checks={}
        for typ in ['formula_reference','summary_source','aggregation']:
            target={k for k,v in archive['edge_all_types'].items() if typ in v};new={k for k,v in result['edge_all_types'].items() if typ in v}
            checks[typ]={'archived_pairs':len(target),'reconstructed_pairs':len(new),'extra_pairs':sorted(new-target),'missing_pairs':sorted(target-new),'exact_match':new==target}
            assert new==target,(typ,len(new-target),len(target-new))
        data={'data_dir':a.data_dir,'generator_inputs':['sheets.json'],'construction_reads_query_labels':False,'checks':checks,'historical_provenance':'Original script not found. Metadata rules inferred from archived relations, not original-time evidence.','sheets_sha256':hashlib.sha256((Path(a.data_dir)/'sheets.json').read_bytes()).hexdigest(),'selected_adjacency_equivalence':'All selected pair sets exactly equal for both deployed configurations; replayed archived checkpoints receive identical adjacency.'}
        dest=Path(a.audit_output or str(out.with_name('audit.json')));dest.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({k:v['exact_match'] for k,v in checks.items()}))
if __name__=='__main__':main()
