#!/usr/bin/env python3
"""Verify follow-up training outputs and compute paired exploratory comparisons."""
import argparse,gc,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_representation import evaluate, groups_for_view
from transformers import AutoModel, AutoTokenizer
import bge_retrain_experiments as core
from audit_and_analyze import (query_metrics,means,clustered_delta,write_json,verify_metrics,graph_model,Reranker,candidate_labels,predict_reranker,ranking_metrics,load_dependency_channels,predict)


def encoder_check(root,folder,view,device,manifest):
    cache=torch.load(folder/'cache.pt',map_location='cpu',weights_only=False)
    corpus=core.load_corpus(root/'data/industrytab_1k',42,.1)
    if view=='examples':corpus['sheets']=json.loads((Path(manifest['datasets']['industrytab_1k']['source_dir'])/'sheets.json').read_text())
    model_path=str(Path(manifest['experiment_root'])/'models/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a')
    model=AutoModel.from_pretrained(model_path,local_files_only=True).to(device)
    partial=torch.load(folder/'last4.pt',map_location='cpu',weights_only=False);_,unexpected=model.load_state_dict(partial,strict=False);assert not unexpected
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    _,sheet,query,candidates,scores=evaluate(model,tokenizer,corpus,groups_for_view(corpus,view),list(range(len(corpus['queries']))),device,16)
    positions=cache['eval_positions'];expected=json.loads((folder/'result.json').read_text())['test'];check=verify_metrics(ranking_metrics(cache,positions,candidates[positions]),expected,'reencoded-followup')
    cosine=float(torch.nn.functional.cosine_similarity(sheet,cache['sheet_embeddings']).min());assert cosine>.9999
    check.update({'sheet_embedding_min_cosine':cosine,'query_embedding_max_abs_error':float((query-cache['query_embeddings']).abs().max()),'partial_checkpoint_tensors':len(partial)})
    del model,cache;gc.collect();torch.cuda.empty_cache();return check


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',default='docs/server_archive_manifest.json');args=parser.parse_args();manifest=json.loads(Path(args.manifest).read_text())
    root=Path(__file__).resolve().parents[2];dest=root/'outputs/experiments/followup_audit';records=[];viewruns=[]
    device=torch.device('cuda',0);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
    for seed in [42,43,44]:
        files={v:json.loads((root/f'outputs/experiments/retrained/{v}/seed{seed}/result.json').read_text()) for v in ['sheet','columns','examples']}
        indices={v:[p['query_index'] for p in r['predictions']] for v,r in files.items()};assert indices['sheet']==indices['columns']==indices['examples']
        annotations=json.loads((root/'data/industrytab_1k/query.json').read_text())
        for j,i in enumerate(indices['sheet']):
            ann=annotations[i];methods={v:query_metrics(set(map(str,ann['positive_sheet_ids'])),r['predictions'][j]['ranked_ids'],list(map(str,ann.get('hard_negative_sheet_ids',[])))) for v,r in files.items()}
            records.append({'seed':seed,'query_index':i,'query':ann['query'],'metrics':methods})
        for v,r in files.items():
            subset=[x for x in records if x['seed']==seed];actual=means(subset,v);check=verify_metrics(actual,r['test'],f'retrained-{v}-{seed}');viewruns.append({'seed':seed,'view':v,'metrics':r['test'],'metric_check':check,'encoder_replay':encoder_check(root,root/f'outputs/experiments/retrained/{v}/seed{seed}',v,device,manifest),'protocol':r['protocol'],'history':r['history']})
    views={'runs':viewruns,'means':{v:means(records,v) for v in ['sheet','columns','examples']},'columns_minus_sheet':clustered_delta(records,'sheet','columns'),'examples_minus_sheet':clustered_delta(records,'sheet','examples')}
    write_json(dest/'representation_per_query.json',records)
    device=torch.device('cuda',0);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15);grouprecords=[];groupruns=[]
    for seed in [42,43,44]:
        folder=root/f'outputs/experiments/grouped/seed{seed}';split=json.loads((root/f'outputs/experiments/grouped_splits/seed{seed}.json').read_text());result=json.loads((folder/'result.json').read_text())
        cache=torch.load(folder/'cache.pt',map_location='cpu',weights_only=False);pos=cache['eval_positions'];train=set(split['train_sheet_ids'])
        for i in cache['train_core']:
            assert {cache['sheet_ids'][j] for j in cache['candidates'][i].tolist()}<=train
            assert set(cache['queries'][i]['positives'])<=train
        assert pos==split['eval_positions'] and cache['val_positions']==split['val_positions']
        labels=candidate_labels(cache,50);checkpoint=torch.load(folder/'mlp/frozen_bge_mlp_top50.pt',map_location='cpu',weights_only=False)
        mlp=Reranker(cache['sheet_embeddings'].size(1),'mlp');mlp.load_state_dict(checkpoint['state_dict'],strict=True);mlp=mlp.to(device).eval()
        ranks={'stage1':cache['candidates'][pos],'mlp':predict_reranker(mlp,cache,pos,50,labels,device,16)}
        graphfile=root/'outputs/experiments/graph_reconstruction/industrytab_1k/graph.json'
        dep=load_dependency_channels(str(graphfile),cache['sheet_ids'],['formula_reference','summary_source'])
        old=load_dependency_channels(str(root/'data/industrytab_1k/dependency_edges.json'),cache['sheet_ids'],['formula_reference','summary_source']);assert torch.equal(dep,old)
        model=graph_model({'mlp_checkpoint':str(folder/'mlp/frozen_bge_mlp_top50.pt')},cache,dep,str(folder/'gnn/full_gnn.pt'),device)
        ranks['gnn']=predict(model,cache,pos,50,labels,device,16,dep)
        expected={'stage1':result['test'],'mlp':json.loads((folder/'mlp/frozen_bge_mlp_top50.json').read_text())['reranked'],'gnn':json.loads((folder/'gnn/full_gnn.json').read_text())['reranked']}
        checks={m:verify_metrics(ranking_metrics(cache,pos,r),expected[m],f'grouped-{m}-{seed}') for m,r in ranks.items()}
        for j,i in enumerate(pos):
            q=cache['queries'][i];rankids={m:[cache['sheet_ids'][k] for k in rank[j,:5].tolist()] for m,rank in ranks.items()}
            grouprecords.append({'seed':seed,'query_index':q['query_index'],'query':q['query'],'family_schema_component':next(g for g,ids in enumerate(split['groups']) if q['positives'][0] in ids),'gold_ids':q['positives'],'ranked_ids':rankids,'metrics':{m:query_metrics(q['positives'],ids,q['hard_negatives']) for m,ids in rankids.items()}})
        groupruns.append({'seed':seed,'zero':result['zero'],'metrics':expected,'checks':checks,'sheet_counts':split['sheet_counts'],'query_counts':split['query_counts'],'excluded_queries':len(split['excluded_cross_partition_queries']),'train_candidates_with_heldout_sheets':0,'reconstructed_adjacency_equals_archived':True,'encoder_replay':encoder_check(root,folder,'sheet',device,manifest)})
        print(json.dumps(groupruns[-1]),flush=True);del model,mlp,cache,dep,old;gc.collect();torch.cuda.empty_cache()
    grouped={'runs':groupruns,'means':{v:means(grouprecords,v) for v in ['stage1','mlp','gnn']},'gnn_minus_mlp':clustered_delta(grouprecords,'mlp','gnn'),'mlp_minus_stage1':clustered_delta(grouprecords,'stage1','mlp')}
    group_clusters=[{**r,'query_index':r['family_schema_component']} for r in grouprecords]
    family_ci=clustered_delta(group_clusters,'mlp','gnn');family_ci['family_schema_components']=family_ci.pop('query_clusters');family_ci['resampling']='family/schema connected components; all observed query and seed outcomes retained'
    grouped['gnn_minus_mlp_family_bootstrap']=family_ci
    write_json(dest/'grouped_per_query.json',grouprecords);write_json(dest/'summary.json',{'retrained':views,'grouped':grouped})
    print('Follow-up audit complete.',flush=True)
if __name__=='__main__':main()
