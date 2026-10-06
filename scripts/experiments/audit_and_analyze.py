#!/usr/bin/env python3
"""Replay archived SAT checkpoints and audit data, candidates and graph controls.

Writes measured metrics and per-query predictions. Dependency-connected positives
are a structural diagnostic, not an annotation of semantically dependency-only evidence.
"""

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'full_corpus'))
from bge_retrain_experiments import (
    QUERY_PREFIX, Reranker, candidate_ceiling, candidate_labels, encode_texts,
    load_corpus, predict_reranker, ranking_metrics, serialize_sheet,
)
from gated_graph_refine import GatedGraphFromMLP, load_dependency_channels, predict


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def query_metrics(positive_ids, ranked_ids, hard_ids):
    positives, hard = set(positive_ids), set(hard_ids)
    hits = [sid in positives for sid in ranked_ids[:5]]
    first = next((i+1 for i, hit in enumerate(hits) if hit), None)
    ideal = sum(1/math.log2(i+2) for i in range(min(5, len(positives))))
    return {
        'NDCG@5': sum(hit/math.log2(i+2) for i, hit in enumerate(hits))/ideal,
        'MacroRecall@5': sum(hits)/len(positives), 'Precision@5': sum(hits)/5,
        'Hit@5': float(any(hits)), 'MRR@5': 1/first if first else 0,
        'AllRelevant@5': float(positives.issubset(ranked_ids[:5])),
        'HN-FPR@1': float(ranked_ids[0] in hard) if hard else None,
    }


def means(records, method):
    result = {}
    for key in records[0]['metrics'][method]:
        values = [r['metrics'][method][key] for r in records
                  if r['metrics'][method][key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def clustered_delta(records, left, right, n_boot=10000):
    """Resample original query IDs, retaining all seed evaluations within a cluster."""
    clusters = defaultdict(list)
    for r in records:
        clusters[r['query_index']].append(
            r['metrics'][right]['NDCG@5']-r['metrics'][left]['NDCG@5'])
    sums = np.array([sum(v) for v in clusters.values()])
    counts = np.array([len(v) for v in clusters.values()])
    rng = np.random.default_rng(20260930)
    intervals = []
    for start in range(0, n_boot, 500):
        indices = rng.integers(len(sums), size=(min(500, n_boot-start), len(sums)))
        intervals.extend((sums[indices].sum(1)/counts[indices].sum(1)).tolist())
    deltas = [v for group in clusters.values() for v in group]
    seed_delta = {str(s): float(np.mean([
        r['metrics'][right]['NDCG@5']-r['metrics'][left]['NDCG@5']
        for r in records if r['seed']==s])) for s in sorted({r['seed'] for r in records})}
    return {
        'mean_delta': float(np.mean(deltas)), 'query_clusters': len(clusters),
        'query_seed_evaluations': len(records),
        'bootstrap_95_ci': np.quantile(intervals, [.025, .975]).tolist(),
        'per_seed_mean_delta': seed_delta,
        'wins': sum(v>1e-10 for v in deltas),
        'ties': sum(abs(v)<=1e-10 for v in deltas),
        'losses': sum(v < -1e-10 for v in deltas),
        'resampling': 'query-ID clusters; all observed training seeds retained within each cluster',
    }


def graph_model(run, cache, dependency, checkpoint_path, device):
    base = torch.load(run['mlp_checkpoint'], map_location='cpu', weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    a = checkpoint['args']
    model = GatedGraphFromMLP(
        cache['sheet_embeddings'].size(1), a['layers'], a['gate_init'],
        a['dropout'], base['state_dict'], dependency.size(0),
        a.get('score_message', False), a.get('score_gate_init', -4),
    )
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval()


def verify_metrics(actual, expected, label):
    delta = {k: abs(actual[k]-v) for k, v in expected.items()
             if k in actual and isinstance(v, (int, float))}
    assert max(delta.values(), default=0)<=1e-6, (label, delta)
    return {'max_absolute_metric_error': max(delta.values(), default=0), 'passed': True}


def audit_encoder(run, cache, corpus, device, batch_size, experiment_root):
    from transformers import AutoModel, AutoTokenizer
    model_path = Path(run['model_path'])
    if not model_path.is_absolute():
        model_path = Path(run['cache']).parents[1]/model_path
        # Archived caches store a path relative to the original experiment root.
        if not model_path.exists():
            model_path = Path(experiment_root)/run['model_path']
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device)
    state = torch.load(Path(run['cache']).parent/'finetuned_bge_last4.pt',
                       map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    sheet = encode_texts(model, tokenizer,
                         [serialize_sheet(corpus['sheets'][sid]) for sid in cache['sheet_ids']],
                         device, batch_size, 256)
    pos = cache['eval_positions']
    query = encode_texts(model, tokenizer,
                        [QUERY_PREFIX+cache['queries'][p]['query'] for p in pos],
                        device, batch_size, 256)
    sheet_error = float((sheet-cache['sheet_embeddings']).abs().max())
    query_error = float((query-cache['query_embeddings'][pos]).abs().max())
    sheet_cos = float(torch.nn.functional.cosine_similarity(sheet, cache['sheet_embeddings']).min())
    query_cos = float(torch.nn.functional.cosine_similarity(query, cache['query_embeddings'][pos]).min())
    recomputed_ranks = (query @ sheet.T).topk(50, dim=1).indices
    metrics = ranking_metrics(cache, pos, recomputed_ranks)
    expected = json.loads(Path(run['mlp_result']).read_text())['raw_bge']
    check = verify_metrics(metrics, expected, 'reencoded_stage1')
    assert min(sheet_cos, query_cos) > .9999, (sheet_cos, query_cos)
    del model
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return {**check, 'sheet_embedding_max_abs_error': sheet_error,
            'query_embedding_max_abs_error': query_error,
            'sheet_embedding_min_cosine': sheet_cos, 'query_embedding_min_cosine': query_cos,
            'metrics': metrics, 'max_length': 256,
            'pooling': 'L2-normalized final-layer CLS', 'partial_checkpoint_tensors': len(state)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='docs/server_archive_manifest.json')
    parser.add_argument('--output-dir', default='outputs/experiments/audit')
    parser.add_argument('--dense-root', default='outputs/experiments/dense_only')
    parser.add_argument('--permuted-root', default='outputs/experiments/node_permuted')
    parser.add_argument('--audit-encoder', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--memory-fraction', type=float, default=.15)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    manifest = json.loads(Path(args.manifest).read_text())
    output = Path(args.output_dir)
    audit = {'datasets': {}, 'runs': []}
    records = []
    for run in manifest['runs']:
        dataset, seed = run['dataset'], run['seed']
        d = Path('data')/dataset
        raw = json.loads((d/'query.json').read_text())
        ratio = .1 if dataset=='industrytab_1k' else .2
        corpus = load_corpus(d, 42 if dataset=='industrytab_1k' else seed, ratio)
        cache = torch.load(run['cache'], map_location='cpu', weights_only=False)
        valid = set(cache['sheet_ids'])
        for key in ['sheet_ids', 'train_core', 'val_positions', 'eval_positions']:
            assert corpus[key]==cache[key], (dataset, seed, key)
        assert corpus['queries']==cache['queries'], (dataset, seed, 'cached_labels')
        splits = [set(cache[k]) for k in ['train_core', 'val_positions', 'eval_positions']]
        assert not any(splits[i]&splits[j] for i in range(3) for j in range(i+1, 3))
        assert len(set.union(*splits))==len(cache['queries'])
        label_errors=[]
        for index,item in enumerate(raw):
            positive=set(map(str,item.get('positive_sheet_ids',item.get('sheet_ids',[]))))
            negative=set(map(str,item.get('negative_sheet_ids',item.get('sheet_ids_negative',[]))))
            hard=set(map(str,item.get('hard_negative_sheet_ids',[])))
            if not positive or (positive|negative|hard)-valid or positive&(negative|hard):
                label_errors.append(index)
        assert not label_errors, (dataset, 'invalid labels', label_errors)
        if dataset not in audit['datasets']:
            edges = json.loads((d/'dependency_edges.json').read_text())
            by_type = Counter(t for ts in edges['edge_all_types'].values() for t in ts)
            selected = set(run['gnn_args']['dependency_types'].split(','))
            pairs = {k for k,types in edges['edge_all_types'].items() if selected.intersection(types)}
            per_split_queries=[set(cache['queries'][p]['query'] for p in ss) for ss in splits]
            audit['datasets'][dataset]={
                'num_sheets':len(valid), 'num_queries':len(raw), 'invalid_label_records':label_errors,
                'query_type_counts':dict(Counter(x.get('query_type','unknown') for x in raw)),
                'queries_with_more_than_5_positives':sum(len(x.get('positive_sheet_ids',x.get('sheet_ids',[])))>5 for x in raw),
                'duplicate_query_texts':len(raw)-len({x['query'] for x in raw}),
                'train_test_exact_text_overlap':len(per_split_queries[0]&per_split_queries[2]),
                'retained_edge_counts':edges.get('stats',{}).get('edge_type_counts'),
                'multilabel_edge_counts':dict(by_type), 'selected_types':sorted(selected),
                'selected_unique_undirected_pairs':len(pairs),
                'selected_directed_matrix_entries':2*len(pairs),
                'graph_source':'edge_all_types; symmetrized; no query-specific edges read by this runner',
            }
        all_scores = cache['query_embeddings'] @ cache['sheet_embeddings'].T
        candidates = cache['candidates']
        selected_scores = all_scores.gather(1,candidates)
        score_error=float((selected_scores-cache['candidate_scores']).abs().max())
        other_scores=all_scores.clone().scatter_(1,candidates,float('-inf'))
        max_outside_margin=float((other_scores.max(1).values-selected_scores.min(1).values).max())
        assert score_error<=1e-5 and max_outside_margin<=1e-5,(score_error,max_outside_margin)
        assert all(len(set(row))==len(row) for row in candidates.tolist())
        pos=cache['eval_positions'];labels=candidate_labels(cache,50)
        base=Reranker(cache['sheet_embeddings'].size(1),'mlp')
        base.load_state_dict(torch.load(run['mlp_checkpoint'],map_location='cpu',weights_only=False)['state_dict'],strict=True)
        base=base.to(device).eval()
        ranks={'stage1':candidates[pos],
               'mlp':predict_reranker(base,cache,pos,50,labels,device,args.batch_size)}
        dependency=load_dependency_channels(str(d/'dependency_edges.json'),cache['sheet_ids'],
                                            run['gnn_args']['dependency_types'].split(','))
        graph=graph_model(run,cache,dependency,run['gnn_checkpoint'],device)
        ranks['gnn']=predict(graph,cache,pos,50,labels,device,args.batch_size,dependency)
        checks={}
        for method in ranks:
            expected=json.loads(Path(run['mlp_result'] if method!='gnn' else run['gnn_result']).read_text())['raw_bge' if method=='stage1' else 'reranked']
            checks[method]=verify_metrics(ranking_metrics(cache,pos,ranks[method]),expected,method)
        dense_path=Path(args.dense_root)/dataset/f'seed{seed}'/'dense_gnn_top50.pt'
        if dense_path.exists():
            dense_dependency=torch.zeros(0,len(valid),len(valid))
            dense=graph_model(run,cache,dense_dependency,str(dense_path),device)
            ranks['dense_gnn']=predict(dense,cache,pos,50,labels,device,args.batch_size,dense_dependency)
            expected=json.loads(dense_path.with_suffix('.json').read_text())['reranked']
            checks['dense_gnn']=verify_metrics(ranking_metrics(cache,pos,ranks['dense_gnn']),expected,'dense_gnn')
            del dense
        permuted_path=Path(args.permuted_root)/dataset/f'seed{seed}'/'node_permuted_gnn_top50.pt'
        if permuted_path.exists():
            permuted_dependency=load_dependency_channels(
                str(permuted_path.parent/'permuted_dependency_edges.json'),cache['sheet_ids'],
                run['gnn_args']['dependency_types'].split(','))
            permuted_model=graph_model(run,cache,permuted_dependency,str(permuted_path),device)
            ranks['permuted_gnn']=predict(permuted_model,cache,pos,50,labels,device,args.batch_size,permuted_dependency)
            expected=json.loads(permuted_path.with_suffix('.json').read_text())['reranked']
            checks['permuted_gnn']=verify_metrics(ranking_metrics(cache,pos,ranks['permuted_gnn']),expected,'permuted_gnn')
            del permuted_model,permuted_dependency
        adjacency=dependency.any(0)
        id_index={sid:i for i,sid in enumerate(cache['sheet_ids'])}
        for row,position in enumerate(pos):
            item=cache['queries'][position];annotation=raw[item['query_index']]
            gold=set(item['positives']);candidate_ids=[cache['sheet_ids'][i] for i in candidates[position].tolist()]
            gold_indices=[id_index[sid] for sid in gold]
            connected={sid for sid in gold if adjacency[id_index[sid],gold_indices].any()}
            candidate_indices=candidates[position]
            neighbours=adjacency[candidate_indices].any(0).nonzero().flatten().tolist()
            expanded_indices=set(candidate_indices.tolist())|set(neighbours)
            expanded={cache['sheet_ids'][i] for i in expanded_indices}
            budget=len(expanded)
            direct_indices=all_scores[position].topk(budget).indices.tolist()
            matched_dense={cache['sheet_ids'][i] for i in direct_indices}
            ranked={method:[cache['sheet_ids'][i] for i in rank[row].tolist()] for method,rank in ranks.items()}
            records.append({
                'dataset':dataset,'seed':seed,'query_index':item['query_index'],'query':item['query'],
                'query_type':annotation.get('query_type','unknown'),
                'dependency_type':annotation.get('dependency_type','unknown'),
                'is_dependency_heavy':annotation.get('is_dependency_heavy',False),
                'gold_ids':sorted(gold,key=int),'hard_negative_ids':item['hard_negatives'],
                'ranked_ids':ranked,'metrics':{method:query_metrics(gold,ids,item['hard_negatives']) for method,ids in ranked.items()},
                'candidate':{'recall50':len(gold&set(candidate_ids))/len(gold),
                             'all_relevant50':gold.issubset(candidate_ids),
                             'oracle_recall5':min(5,len(gold&set(candidate_ids)))/len(gold),
                             'connected_positive_count':len(connected),
                             'connected_positive_lost':len(connected-set(candidate_ids)),
                             'annotation_dependency_positive_count':len(set(map(str,annotation.get('dependency_positive_sheet_ids',[])))&gold),
                             'annotation_dependency_positive_lost':len((set(map(str,annotation.get('dependency_positive_sheet_ids',[])))&gold)-set(candidate_ids)),
                             'expanded_pool_size':budget,
                             'expanded_recall':len(gold&expanded)/len(gold),
                             'matched_budget_dense_recall':len(gold&matched_dense)/len(gold)},
            })
        details={'dataset':dataset,'seed':seed,'metric_replay':checks,
                 'candidate_scores_max_abs_error':score_error,
                 'max_outside_candidate_score_margin':max_outside_margin,
                 'candidate_ceiling':candidate_ceiling(cache,pos,50)}
        if args.audit_encoder:details['encoder_replay']=audit_encoder(run,cache,corpus,device,args.batch_size,manifest['experiment_root'])
        audit['runs'].append(details)
        print(json.dumps(details),flush=True)
        write_json(output/'audit.json',audit)
        write_json(output/'per_query_predictions.json',records)
        del base,graph,cache,ranks,all_scores,other_scores,dependency
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    analysis={}
    for dataset in {r['dataset'] for r in records}:
        rows=[r for r in records if r['dataset']==dataset]
        methods=list(rows[0]['metrics'])
        assert all(set(r['metrics'])==set(methods) for r in rows), 'Some controls are still incomplete.'
        groups={'all':rows,'dependency_heavy':[r for r in rows if r['is_dependency_heavy']],
                'not_dependency_heavy':[r for r in rows if not r['is_dependency_heavy']]}
        groups.update({t:[r for r in rows if r['query_type']==t] for t in sorted({r['query_type'] for r in rows})})
        report={}
        for label,group in groups.items():
            if not group:continue
            report[label]={'query_seed_evaluations':len(group),'unique_queries':len({r['query_index'] for r in group}),
                'metrics':{method:means(group,method) for method in methods},
                'gnn_minus_mlp':clustered_delta(group,'mlp','gnn')}
            if 'dense_gnn' in methods:
                report[label]['gnn_minus_dense']=clustered_delta(group,'dense_gnn','gnn')
            if 'permuted_gnn' in methods:
                report[label]['gnn_minus_permuted']=clustered_delta(group,'permuted_gnn','gnn')
            candidates=[r['candidate'] for r in group]
            connected=[r for r in candidates if r['connected_positive_count']]
            annotated=[r for r in candidates if r['annotation_dependency_positive_count']]
            report[label]['candidate']={
                'recall50':float(np.mean([r['recall50'] for r in candidates])),
                'all_relevant50':float(np.mean([r['all_relevant50'] for r in candidates])),
                'oracle_recall5':float(np.mean([r['oracle_recall5'] for r in candidates])),
                'connected_positive_eligible_evaluations':len(connected),
                'connected_positive_recall50':float(np.mean([1-r['connected_positive_lost']/r['connected_positive_count'] for r in connected])) if connected else None,
                'connected_positive_lost_queries':sum(r['connected_positive_lost']>0 for r in connected),
                'annotation_dependency_positive_eligible_evaluations':len(annotated),
                'annotation_dependency_positive_recall50':float(np.mean([1-r['annotation_dependency_positive_lost']/r['annotation_dependency_positive_count'] for r in annotated])) if annotated else None,
                'mean_expanded_pool_size':float(np.mean([r['expanded_pool_size'] for r in candidates])),
                'expanded_recall':float(np.mean([r['expanded_recall'] for r in candidates])),
                'matched_budget_dense_recall':float(np.mean([r['matched_budget_dense_recall'] for r in candidates])),
            }
        analysis[dataset]=report
    write_json(output/'analysis.json',analysis)
    print('Wrote verified audit, subgroup analysis and per-query predictions.',flush=True)


if __name__=='__main__':main()
