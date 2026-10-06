#!/usr/bin/env python3
"""Exploratory transfer from IndustryTab-614 encoders to added IndustryTab-1K sheets.

The corpora overlap. Only queries whose complete positive set consists of added
sheet identities are used in the unseen-identity subset. This is not a disjoint
real-world workbook/domain benchmark.
"""

import argparse
import gc
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'full_corpus'))
from bge_retrain_experiments import QUERY_PREFIX,encode_texts,ranking_metrics,serialize_sheet


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='docs/server_archive_manifest.json')
    p.add_argument('--output',default='outputs/experiments/transfer/transfer.json')
    args=p.parse_args();torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.15)
    manifest=json.loads(Path(args.manifest).read_text())
    source=json.loads(Path('data/industrytab_614/sheets.json').read_text())
    target=json.loads(Path('data/industrytab_1k/sheets.json').read_text())
    target={str(x.get('sheet_id',k)):x for k,x in target.items()}
    old_names={x['name'] for x in source.values()}
    added={sid for sid,s in target.items() if s['name'] not in old_names}
    budget={sid for sid,s in target.items() if s['name'].startswith('Budget_Planning')}
    target_runs=[r for r in manifest['runs'] if r['dataset']=='industrytab_1k']
    source_runs=[r for r in manifest['runs'] if r['dataset']=='industrytab_614']
    reference=torch.load(target_runs[0]['cache'],map_location='cpu',weights_only=False)
    positions=reference['eval_positions']
    groups={'all_target_test':positions,
            'added_identity_positives_only':[i for i in positions if set(reference['queries'][i]['positives'])<=added],
            'new_category_budget_planning':[i for i in positions if set(reference['queries'][i]['positives'])<=budget]}
    zero=torch.load(Path(manifest['experiment_root'])/'results/base_bge_cache.pt',map_location='cpu',weights_only=False)
    assert zero['eval_positions']==positions and zero['queries']==reference['queries']
    result={'protocol':{'source':'IndustryTab-614 trained Stage 1 encoders; no 1K parameter adaptation',
                        'target_corpus_sheets':len(target),'added_sheet_identities':len(added),
                        'overlap_caveat':'Source/target corpora overlap in 614 names; added-sheet entities/templates may also overlap.',
                        'groups':{name:{'test_queries':len(rows),'query_indices':[reference['queries'][i]['query_index'] for i in rows]} for name,rows in groups.items()}},
            'zero_shot':{name:ranking_metrics(zero,rows,zero['candidates'][rows]) for name,rows in groups.items() if rows},'runs':[]}
    model_dir=Path(manifest['experiment_root'])/'models/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a'
    tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True)
    texts=[serialize_sheet(target[sid]) for sid in reference['sheet_ids']]
    queries=[QUERY_PREFIX+reference['queries'][i]['query'] for i in positions]
    for source_run,target_run in zip(source_runs,target_runs):
        assert source_run['seed']==target_run['seed']
        model=AutoModel.from_pretrained(str(model_dir),local_files_only=True).to(device).eval()
        state=torch.load(Path(source_run['cache']).parent/'finetuned_bge_last4.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state,strict=False)
        sheet_embeddings=encode_texts(model,tokenizer,texts,device,16,256)
        query_embeddings=encode_texts(model,tokenizer,queries,device,16,256)
        ranks=(query_embeddings@sheet_embeddings.T).topk(50,dim=1).indices
        by_position={position:row for row,position in enumerate(positions)}
        target_cache=torch.load(target_run['cache'],map_location='cpu',weights_only=False)
        metrics={name:{'transfer_614_encoder':ranking_metrics(reference,rows,ranks[[by_position[i] for i in rows]]),
                       'native_1k_encoder':ranking_metrics(target_cache,rows,target_cache['candidates'][rows])}
                 for name,rows in groups.items() if rows}
        result['runs'].append({'seed':source_run['seed'],'metrics':metrics})
        print(json.dumps(result['runs'][-1]),flush=True)
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2)+'\n')
        del model,target_cache;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':main()
