#!/usr/bin/env python3
"""Group known filename families and identical schemas before label assignment."""
import hashlib,json,random,re
from collections import defaultdict
from pathlib import Path

def family(name):
    name=(name[len('Synthetic_Data__'):] if name.startswith('Synthetic_Data__') else name).replace('Stock_Market__','Stock_Market_Data__')
    for a,b in [('Inventory_Inventory_Check_','Inventory_Inventory_Check_Records_'),('Inventory_Warehouse_Entry_','Inventory_Warehouse_Entry_Details_'),('Inventory_Warehouse_Exit_','Inventory_Warehouse_Exit_Details_')]:
        if b not in name:name=name.replace(a,b)
    name=re.sub(r'_(source|target)$','',name)
    return re.sub(r'20\d\d(?:[-_]\d\d)?(?:_Q\d)?','YEAR',name)

def main():
    root=Path(__file__).resolve().parents[2];s=json.loads((root/'data/industrytab_1k/sheets.json').read_text());q=json.loads((root/'data/industrytab_1k/query.json').read_text());parent={i:i for i in s}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(a)]=find(b)
    seen_family={};seen_headers={}
    for i,v in s.items():
        for seen,key in [(seen_family,family(v['name'])),(seen_headers,tuple(sorted(c['name'].strip().lower() for c in v['columns'])))]:
            if key in seen:union(i,seen[key])
            else:seen[key]=i
    groups=defaultdict(list)
    for i in s:groups[find(i)].append(i)
    group_list=sorted([sorted(g,key=int) for g in groups.values()],key=lambda g:int(g[0]));sid_group={i:j for j,g in enumerate(group_list) for i in g}
    destination=root/'outputs/experiments/grouped_splits';destination.mkdir(parents=True,exist_ok=True)
    for seed in [42,43,44]:
        rng=random.Random(seed);order=list(range(len(group_list)));rng.shuffle(order)
        # Allocate by group size, using random order for equal-sized groups.
        order.sort(key=lambda i:-len(group_list[i]));allocation={k:[] for k in ['train','val','test']};sizes={k:0 for k in allocation};target={'train':.70*len(s),'val':.15*len(s),'test':.15*len(s)}
        for g in order:
            k=max(allocation,key=lambda k:(target[k]-sizes[k])/target[k]);allocation[k].append(g);sizes[k]+=len(group_list[g])
        sheet_sets={k:{i for g in gs for i in group_list[g]} for k,gs in allocation.items()}
        queries={k:[] for k in allocation};excluded=[]
        for index,row in enumerate(q):
            gold=set(map(str,row['positive_sheet_ids']));dest=[k for k,ids in sheet_sets.items() if gold<=ids]
            if len(dest)==1:queries[dest[0]].append(index)
            else:excluded.append(index)
        for k in queries:rng.shuffle(queries[k])
        assert all(set(map(str,q[i]['positive_sheet_ids']))<=sheet_sets[k] for k,ids in queries.items() for i in ids)
        assert not(sheet_sets['train']&sheet_sets['test'] or sheet_sets['train']&sheet_sets['val'] or sheet_sets['val']&sheet_sets['test'])
        data={'seed':seed,'grouping':'union of normalized filename family and identical complete sorted header signature; grouping uses no query labels','family_rule': 'strip source/target and year/month/quarter; remove Synthetic_Data prefix; explicit inventory/stock aliases','groups':group_list,'group_assignments':allocation,'sheet_counts':sizes,'train_sheet_ids':sorted(sheet_sets['train'],key=int),'val_sheet_ids':sorted(sheet_sets['val'],key=int),'test_sheet_ids':sorted(sheet_sets['test'],key=int),'train_core':queries['train'],'val_positions':queries['val'],'eval_positions':queries['test'],'excluded_cross_partition_queries':excluded,'query_counts':{k:len(v) for k,v in queries.items()},'schema_overlap_train_test':0,'family_overlap_train_test':0,'limitation':'metadata-derived template/family groups; incomplete original workbook provenance; shared query language/templates remain possible','data_sha256':{p:hashlib.sha256((root/'data/industrytab_1k'/p).read_bytes()).hexdigest() for p in ['sheets.json','query.json']}}
        (destination/f'seed{seed}.json').write_text(json.dumps(data,indent=2)+'\n');print(seed,len(group_list),sizes,data['query_counts'],'excluded',len(excluded))
if __name__=='__main__':main()
