#!/usr/bin/env python3
"""Compare original baseline predictions with a fresh original-path replay."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--fresh',default='outputs/experiments/baseline_precision/fp32_sentence_transformer_local.json');p.add_argument('--output',default='outputs/experiments/baseline_precision/resolution.json');a=p.parse_args()
    old=json.loads(Path(a.original).read_text());fresh=json.loads(Path(a.fresh).read_text());x={r['query_index']:r for r in old['predictions']};y={r['query_index']:r for r in fresh['predictions']};assert set(x)==set(y)
    for i in x:assert x[i]['query']==y[i]['query'] and set(x[i]['relevant_sheet_ids'])==set(y[i]['relevant_sheet_ids'])
    result={'original':a.original,'fresh':a.fresh,'fresh_environment':'Local Mac replay: SentenceTransformers 5.6.0; torch2.5.1; automatic MPS; FP32 model; CLS/L2; snapshot max_seq_length512; stable NumPy score sort; batch64','snapshot':'a5beb1e3e68b9ab74eb54cfd186867f64f240e1a','original_metrics':old['metrics'],'fresh_metrics':fresh['metrics'],'metric_dictionary_exact_match':old['metrics']==fresh['metrics'],'top5_exact_matching_queries':sum(x[i]['predicted_sheet_ids']==y[i]['predicted_sheet_ids'] for i in x),'top50_exact_matching_queries':sum(x[i]['candidate_sheet_ids']==y[i]['candidate_sheet_ids'] for i in x),'max_stored_candidate_score_error':max(abs(a-b) for i in x for a,b in zip(x[i]['candidate_scores'],y[i]['candidate_scores'])),'conclusion':'Original local SentenceTransformer path exactly replays all archived rankings and scores. Mixed-precision AutoModel cache is a different numeric path; near-score ordering differs, without a label/metric error.'}
    assert result['metric_dictionary_exact_match'] and result['top5_exact_matching_queries']==len(x) and result['top50_exact_matching_queries']==len(x) and result['max_stored_candidate_score_error']<=1e-6
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n');print('Baseline replay matched',len(x),'queries.')
if __name__=='__main__':main()
