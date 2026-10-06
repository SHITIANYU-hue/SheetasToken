#!/usr/bin/env python3
"""Verify saved baseline predictions and replay the three 1K cross-encoders."""

import argparse
import gc
import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/'scripts/full_corpus'))
from baselines.end_to_end_rag_llm import ranking_metrics as baseline_metrics
from bge_retrain_experiments import CrossEncoder,cross_scores,load_corpus,ranking_metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='docs/server_archive_manifest.json')
    p.add_argument('--output',default='outputs/experiments/baselines/audit.json')
    args=p.parse_args();torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.15)
    manifest=json.loads(Path(args.manifest).read_text())
    base=Path(manifest['server_root'])/'codex_llm_baselines/results'
    specs=[('industrytab_1k',42,'bge_full_corpus_top5.json'),
           ('industrytab_1k',42,'bge_top50_qwen3_5_9b_strict_t0.json'),
           ('industrytab_1k',42,'qwen3_5_9b_full_catalog_names.json')]
    for seed in [42,43,44]:
        specs.extend(('industrytab_614',seed,f'industrytab_614/{method}_seed{seed}.json')
                     for method in ['rag_eval','rag_llm_qwen35_9b','llm_full_qwen35_9b'])
    result={'saved_prediction_checks':[],'cross_encoder_replay':[]}
    for dataset,seed,relative in specs:
        path=base/relative;o=json.loads(path.read_text());predictions=o['predictions']
        corpus=load_corpus(root/'data'/dataset,42 if dataset=='industrytab_1k' else seed,
                           .1 if dataset=='industrytab_1k' else .2)
        original=json.loads((root/'data'/dataset/'query.json').read_text())
        expected_indices={corpus['queries'][i]['query_index'] for i in corpus['eval_positions']}
        assert {x['query_index'] for x in predictions}==expected_indices
        for row in predictions:
            gold=set(map(str,original[row['query_index']]['positive_sheet_ids']))
            assert gold==set(map(str,row['relevant_sheet_ids']))
            assert row['query']==original[row['query_index']]['query']
            assert set(map(str,row['predicted_sheet_ids']))<=set(corpus['sheet_ids'])
        metrics=baseline_metrics(predictions,5)
        differences={k:abs(v-o['metrics'][k]) for k,v in metrics.items() if k in o['metrics']}
        assert max(differences.values(),default=0)<=1e-6,(relative,differences)
        result['saved_prediction_checks'].append({'dataset':dataset,'seed':seed,'source':str(path),
            'queries':len(predictions),'metrics':metrics,
            'failed_predictions':sum(row.get('status', 'succeeded') not in ['ok', 'success', 'succeeded'] for row in predictions),
            'max_absolute_metric_error':max(differences.values(),default=0)})
    model_dir=Path(manifest['experiment_root'])/'models/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a'
    tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True)
    for run in [r for r in manifest['runs'] if r['dataset']=='industrytab_1k']:
        seed=run['seed'];experiment=Path(manifest['experiment_root'])/'results'
        cross_path=experiment/('cross_tuned/cross_encoder_top50.pt' if seed==42
                               else f'cross_three_seeds/seed{seed}/cross_encoder_top50.pt')
        recorded=json.loads(cross_path.with_suffix('.json').read_text())
        corpus=load_corpus(root/'data/industrytab_1k',42,.1)
        cache=torch.load(run['cache'],map_location='cpu',weights_only=False)
        model=CrossEncoder(str(model_dir)).to(device).eval()
        initial=torch.load(Path(run['cache']).parent/'finetuned_bge_last4.pt',map_location='cpu',weights_only=False)
        model.backbone.load_state_dict(initial,strict=False)
        partial=torch.load(cross_path,map_location='cpu',weights_only=False)
        model.load_state_dict(partial,strict=False)
        positions=cache['eval_positions']
        scores=cross_scores(model,tokenizer,corpus,cache,positions,50,device,16,256)
        alpha=recorded['protocol']['raw_score_alpha']
        order=(scores+alpha*cache['candidate_scores'][positions]).argsort(1,descending=True)
        ranking=torch.gather(cache['candidates'][positions],1,order)
        metrics=ranking_metrics(cache,positions,ranking)
        differences={k:abs(v-recorded['reranked'][k]) for k,v in metrics.items()}
        assert max(differences.values())<=1e-6,(seed,differences)
        result['cross_encoder_replay'].append({'seed':seed,'source':str(cross_path),'metrics':metrics,
                                              'max_absolute_metric_error':max(differences.values())})
        print(json.dumps(result['cross_encoder_replay'][-1]),flush=True)
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2)+'\n')
        del model,cache;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    print('Saved baseline predictions and cross-encoder checkpoints verified.',flush=True)


if __name__=='__main__':main()
