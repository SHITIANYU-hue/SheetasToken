#!/usr/bin/env python3
"""Matched query-sheet training for whole metadata, column mean, or examples.

Two-pass gradient caching preserves one contrastive batch while replaying each
microbatch's dropout RNG; checkpoint selection uses only validation NDCG.
"""
import argparse, copy, gc, hashlib, json, math, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'full_corpus'))
import bge_retrain_experiments as core
from representation_diagnostics import column_chunks, with_examples


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def groups_for_view(corpus,view):
    return [(column_chunks(s) if view=='columns' else [with_examples(s) if view=='examples' else core.serialize_sheet(s)]) for s in [corpus['sheets'][i] for i in corpus['sheet_ids']]]


def encode(model,tokenizer,texts,device,micro,training=False):
    result=[];states=[]
    for start in range(0,len(texts),micro):
        encoded=tokenizer(texts[start:start+micro],padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
        if training:states.append((torch.get_rng_state(),torch.cuda.get_rng_state(device)))
        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.float16):
            e=F.normalize(model(**encoded,return_dict=True).last_hidden_state[:,0].float(),dim=-1)
        result.append(e)
    return torch.cat(result),states


def replay(model,tokenizer,texts,gradient,states,device,micro):
    for j,start in enumerate(range(0,len(texts),micro)):
        encoded=tokenizer(texts[start:start+micro],padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
        with torch.random.fork_rng(devices=[device.index]):
            torch.set_rng_state(states[j][0]);torch.cuda.set_rng_state(states[j][1],device)
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                e=F.normalize(model(**encoded,return_dict=True).last_hidden_state[:,0].float(),dim=-1)
            e.backward(gradient[start:start+micro])


def evaluate(model,tokenizer,corpus,groups,positions,device,micro,allowed=None):
    model.eval();flat=[t for group in groups for t in group]
    emb,_=encode(model,tokenizer,flat,device,micro)
    owners=torch.tensor([i for i,g in enumerate(groups) for _ in g],device=device)
    n=len(groups); sheet=torch.zeros(n,emb.size(1),device=device).index_add_(0,owners,emb)
    sheet=sheet/torch.tensor([len(g) for g in groups],device=device)[:,None]
    query,_=encode(model,tokenizer,[core.QUERY_PREFIX+corpus['queries'][p]['query'] for p in positions],device,micro)
    scores=query@sheet.T
    if allowed is not None:
        mask=torch.ones(n,dtype=torch.bool,device=device);mask[allowed]=False;scores[:,mask]=-float('inf')
    vals,candidates=scores.topk(min(50,len(allowed) if allowed is not None else n),dim=1)
    mini={'sheet_ids':corpus['sheet_ids'],'queries':corpus['queries']}
    metrics=core.ranking_metrics(mini,positions,candidates.cpu())
    return metrics,sheet.cpu(),query.cpu(),candidates.cpu(),vals.cpu()


def main():
    p=argparse.ArgumentParser();p.add_argument('--view',choices=['sheet','columns','examples'],default='sheet');p.add_argument('--seed',type=int,required=True)
    p.add_argument('--manifest',default='docs/server_archive_manifest.json');p.add_argument('--split-file');p.add_argument('--output-dir',required=True);p.add_argument('--micro-batch',type=int,default=16);p.add_argument('--memory-fraction',type=float,default=.22);p.add_argument('--epochs',type=int,default=3)
    args=p.parse_args();root=Path(__file__).resolve().parents[2];out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():print('Already complete',out);return
    manifest=json.loads(Path(args.manifest).read_text());run=next(r for r in manifest['runs'] if r['dataset']=='industrytab_1k' and r['seed']==args.seed)
    model_path=Path(manifest['experiment_root'])/'models/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a'
    corpus=core.load_corpus(root/'data/industrytab_1k',42,.1)
    # Private examples are needed only for this specific input view.
    if args.view=='examples':
        private=Path(manifest['datasets']['industrytab_1k']['source_dir'])/'sheets.json'
        original=json.loads(private.read_text());assert set(map(str,original))==set(corpus['sheet_ids'])
        for i,s in corpus['sheets'].items():assert core.serialize_sheet(s)==core.serialize_sheet(original[i])
        corpus['sheets']=original
    split=json.loads(Path(args.split_file).read_text()) if args.split_file else None
    if split:
        for key in ['train_core','val_positions','eval_positions']:corpus[key]=split[key]
        allowed=[corpus['sheet_ids'].index(i) for i in split['train_sheet_ids']]
    else:allowed=None
    groups=groups_for_view(corpus,args.view)
    torch.set_num_threads(4);device=torch.device('cuda',0);torch.cuda.set_per_process_memory_fraction(args.memory_fraction);random.seed(args.seed);torch.manual_seed(args.seed)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True);model=AutoModel.from_pretrained(model_path,local_files_only=True).to(device)
    for param in model.parameters():param.requires_grad=False
    for layer in model.encoder.layer[-4:]:
        for param in layer.parameters():param.requires_grad=True
    if model.pooler:
        for param in model.pooler.parameters():param.requires_grad=True
    # Fixed negative-mining input shared across views: pretrained whole metadata.
    if split:
        base,_,_,train_candidates,_=evaluate(model,tokenizer,corpus,groups_for_view(corpus,'sheet'),list(range(len(corpus['queries']))),device,args.micro_batch,allowed)
        base_cache={'sheet_ids':corpus['sheet_ids'],'queries':corpus['queries'],'candidates':train_candidates}
    else:
        base_cache=torch.load(Path(manifest['experiment_root'])/'results/base_bge_cache.pt',map_location='cpu',weights_only=False)
        assert [x['query_index'] for x in base_cache['queries']]==[x['query_index'] for x in corpus['queries']]
        for key in ['train_core','val_positions','eval_positions']:assert base_cache[key]==corpus[key]
    train=corpus['train_core'];val=corpus['val_positions'];test=corpus['eval_positions'];bs=64;lr=2e-4
    params=[x for x in model.parameters() if x.requires_grad];optimizer=AdamW(params,lr=lr,weight_decay=.01);steps=math.ceil(len(train)/bs)
    scheduler=get_linear_schedule_with_warmup(optimizer,max(1,steps//2),steps*args.epochs)
    scaler=torch.amp.GradScaler('cuda');history=[];best=-1;beststate=None;started=time.time()
    zero,*_=evaluate(model,tokenizer,corpus,groups,test,device,args.micro_batch)
    for epoch in range(1,args.epochs+1):
        dataset=core.QuerySheetPairs(base_cache,train,epoch,args.seed);loader=DataLoader(dataset,batch_size=bs,shuffle=True);model.train();total=0
        for qp,pi,hi in loader:
            qt=[core.QUERY_PREFIX+corpus['queries'][i]['query'] for i in qp.tolist()]
            ids=pi.tolist()+hi.tolist();st=[t for i in ids for t in groups[i]];owner=torch.tensor([j for j,i in enumerate(ids) for _ in groups[i]],device=device)
            count=torch.tensor([len(groups[i]) for i in ids],device=device)[:,None]
            q,rq=encode(model,tokenizer,qt,device,args.micro_batch,True);s,rs=encode(model,tokenizer,st,device,args.micro_batch,True)
            q=q.detach().requires_grad_();s=s.detach().requires_grad_();pooled=torch.zeros(len(ids),s.size(1),device=device).index_add_(0,owner,s)/count
            loss=F.cross_entropy(q@pooled.T/.02,torch.arange(len(qt),device=device));optimizer.zero_grad();scaler.scale(loss).backward()
            replay(model,tokenizer,qt,q.grad,rq,device,args.micro_batch);replay(model,tokenizer,st,s.grad,rs,device,args.micro_batch)
            scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(params,1);scaler.step(optimizer);scaler.update();scheduler.step();total+=loss.item()
        metrics,*_=evaluate(model,tokenizer,corpus,groups,val,device,args.micro_batch)
        record={'epoch':epoch,'train_loss':total/len(loader),'val':metrics};history.append(record);print(json.dumps(record),flush=True)
        if metrics['NDCG@5']>best:
            best=metrics['NDCG@5'];beststate={n:t.detach().cpu().clone() for n,t in model.state_dict().items() if n.startswith(tuple(f'encoder.layer.{i}.' for i in range(8,12))) or n.startswith('pooler.')}
    model.load_state_dict(beststate,strict=False)
    positions=list(range(len(corpus['queries'])));metrics,se,qe,candidates,scores=evaluate(model,tokenizer,corpus,groups,positions,device,args.micro_batch)
    if split:
        # Train negatives and graph candidates must never include held-out sheets.
        _,_,_,tc,ts=evaluate(model,tokenizer,corpus,groups,train,device,args.micro_batch,allowed)
        candidates[train]=tc;scores[train]=ts
    schema,shape=core.global_priors(corpus)
    cache={'sheet_ids':corpus['sheet_ids'],'queries':corpus['queries'],'sheet_embeddings':se,'query_embeddings':qe,'candidates':candidates,'candidate_scores':scores,'schema':schema,'shape':shape,'train_core':train,'val_positions':val,'eval_positions':test,'train_positions':train+val,'model':str(model_path),'split_seed':42 if not split else split['seed']}
    # Main rerankers expect these exact prior names.
    torch.save(cache,out/'cache.pt');torch.save(beststate,out/'last4.pt')
    testmetrics=core.ranking_metrics(cache,test,candidates[test])
    predicted=[{'query_index':corpus['queries'][i]['query_index'],'query':corpus['queries'][i]['query'],'gold_ids':corpus['queries'][i]['positives'],'ranked_ids':[corpus['sheet_ids'][j] for j in candidates[i,:5].tolist()]} for i in test]
    write(out/'result.json',{'view':args.view,'seed':args.seed,'zero':zero,'test':testmetrics,'history':history,'protocol':{'epochs':args.epochs,'batch_size':bs,'learning_rate':lr,'max_length':256,'train_queries':len(train),'val_queries':len(val),'test_queries':len(test),'column_aggregation':'mean of normalized chunk scores; no mean-vector renormalization','negative_mining':'fixed pretrained whole-metadata top50; train-sheet-only for grouped splits','training':'two-pass exact RNG replay gradient caching','split_file':args.split_file,'validation_selection':'NDCG@5; no test-selected aggregation'},'predictions':predicted,'runtime_seconds':time.time()-started})
    print('Complete',out,flush=True)

if __name__=='__main__':main()
