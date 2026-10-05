#!/usr/bin/env python3
"""Controlled metadata chunk/value diagnostics and audited CSV row/block retrieval.

Chunk score aggregations are reported separately; none is selected on test scores.
Row/block experiments only use sheets whose CSV name, dimensions and columns
match the release, and only original test queries with all positives in that subset.
"""

import argparse
from collections import defaultdict
import csv
import gc
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'full_corpus'))
from bge_retrain_experiments import QUERY_PREFIX, encode_texts, ranking_metrics, serialize_sheet


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def column_chunks(sheet):
    prefix=serialize_sheet({**sheet,'columns':[]})
    headers=[str(c.get('name','') if isinstance(c,dict) else c).strip()
             for c in sheet.get('columns',[])[:12]]
    return [prefix+' ; column: '+h for h in headers if h] or [prefix]


def with_examples(sheet):
    parts=[serialize_sheet({**sheet,'columns':[]})]
    cols=[]
    for c in sheet.get('columns',[])[:12]:
        if isinstance(c,dict):
            name=str(c.get('name','')).strip();value=str(c.get('example','')).strip()
        else:name=str(c).strip();value=''
        if name:cols.append(name+(f' [example: {value}]' if value else ''))
    if cols:parts.append('columns: '+' | '.join(cols))
    return ' ; '.join(parts)


def group_scores(query_embeddings, chunk_embeddings, owners, n_sheets):
    scores=query_embeddings @ chunk_embeddings.T
    grouped=defaultdict(list)
    for i,owner in enumerate(owners):grouped[owner].append(i)
    result={a:torch.zeros(len(query_embeddings),n_sheets) for a in ['max','mean','top3_mean']}
    for owner,indices in grouped.items():
        selected=scores[:,indices]
        result['max'][:,owner]=selected.max(1).values
        result['mean'][:,owner]=selected.mean(1)
        result['top3_mean'][:,owner]=selected.topk(min(3,len(indices)),dim=1).values.mean(1)
    return result


def audit_csv_mapping(root, sheets):
    by_name=defaultdict(list)
    for base in [root/'proactivesheetagent/dataset',root/'sheetagent_paper']:
        for path in base.rglob('*.csv'):by_name[path.stem].append(path)
    accepted={};rejected={};data={};candidates={};translations=defaultdict(set)
    for sid,sheet in sheets.items():
        matches=by_name[sheet['name']]
        if len(matches)!=1:
            rejected[sid]='missing CSV' if not matches else 'ambiguous CSV'
            continue
        path=matches[0]
        try:
            with path.open(encoding='utf-8-sig',newline='') as file:
                reader=csv.reader(file);header=next(reader);rows=list(reader)
            expected=[str(c['name'] if isinstance(c,dict) else c) for c in sheet['columns']]
            if len(rows)!=sheet['num_rows'] or len(header)!=sheet['num_cols'] or len(header)!=len(expected):
                rejected[sid]={'reason':'metadata mismatch','csv_shape':[len(rows),len(header)],
                               'metadata_shape':[sheet['num_rows'],sheet['num_cols']],
                               'headers_match':header==expected}
                continue
            if not rows:
                rejected[sid]='empty CSV';continue
            candidates[sid]=(path,header,rows,expected)
            for raw_name,en_name in zip(header,expected):translations[raw_name].add(en_name)
        except (UnicodeDecodeError,StopIteration,csv.Error) as error:
            rejected[sid]=str(error)
    comparisons={}
    for sid,(path,header,rows,expected) in candidates.items():
        if any(len(translations[h])!=1 for h in header):
            rejected[sid]='ambiguous raw-header translation';continue
        numeric_checks=0;numeric_mismatches=0
        for index,column in enumerate(sheets[sid]['columns']):
            if not isinstance(column,dict):continue
            example=str(column.get('example','')).strip()
            raw=next((row[index] for row in rows if len(row)>index and row[index].strip()),'')
            try:
                x=float(example);y=float(raw)
            except ValueError:continue
            if not math.isfinite(x) or not math.isfinite(y):continue
            numeric_checks+=1
            numeric_mismatches+=not math.isclose(x,y,rel_tol=1e-8,abs_tol=1e-8)
        comparisons[sid]={'checked':numeric_checks,'mismatches':numeric_mismatches}
        if numeric_mismatches:
            rejected[sid]={'reason':'numeric example mismatch',**comparisons[sid]};continue
        accepted[sid]=str(path);data[sid]=(expected,rows)
    return accepted,rejected,data,{
        'header_translation':{k:sorted(v) for k,v in translations.items()},
        'numeric_example_checks':comparisons,
        'policy':'unique exact sheet filename and dimensions; positional headers with globally consistent translation; numeric examples checked against first nonempty CSV values',
        'cell_text_language':'original CSV text retained; headers use release English names; complete cell translation provenance unavailable',
    }


def csv_chunks(sheet, rows, header, block_size, tokenizer=None):
    prefix=serialize_sheet(sheet)
    if block_size==0:
        chunks=[];current=[]
        for row in rows:
            value=' | '.join(f'{name}: {str(cell)[:128]}'
                             for name,cell in zip(header[:12],row[:12]))
            candidate=prefix+' ; rows: '+' ; '.join(current+[value])
            if current and len(tokenizer(candidate,add_special_tokens=True)['input_ids'])>256:
                chunks.append(prefix+' ; rows: '+' ; '.join(current));current=[]
            current.append(value)
        if current:chunks.append(prefix+' ; rows: '+' ; '.join(current))
        return chunks
    chunks=[]
    for start in range(0,len(rows),block_size):
        values=[]
        for row in rows[start:start+block_size]:
            values.append(' | '.join(f'{name}: {str(value)[:128]}'
                                    for name,value in zip(header[:12],row[:12])))
        chunks.append(prefix+' ; rows: '+' ; '.join(values))
    return chunks


def token_stats(tokenizer,texts,max_length=256):
    lengths=[len(ids) for ids in tokenizer(texts,add_special_tokens=True,truncation=False)['input_ids']]
    return {'inputs':len(texts),'mean_tokens':float(np.mean(lengths)),
            'max_tokens':max(lengths),'truncated_fraction':float(np.mean(np.array(lengths)>max_length))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='docs/server_archive_manifest.json')
    parser.add_argument('--output-dir',default='outputs/rebuttal/representation')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--memory-fraction',type=float,default=.15)
    args=parser.parse_args()
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    manifest=json.loads(Path(args.manifest).read_text());out=Path(args.output_dir)
    model_dir=Path(manifest['experiment_root'])/'models/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a'
    tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True)
    output={'protocol':{'max_length':256,'max_headers':12,'column_aggregations':['max','mean','top3_mean'],
                        'value_feature':'one existing example per column; same encoder without value-specific retraining',
                        'row_block_scope':'metadata-verified CSV subset; original test indices; all gold sheets must be present',
                        'test_selected_hyperparameters':False},'runs':[]}
    raw_data={};mapping_info=None
    for dataset in ['industrytab_614','industrytab_1k']:
        runs=[r for r in manifest['runs'] if r['dataset']==dataset]
        private=Path(manifest['datasets'][dataset]['source_dir'])
        sheets=json.loads((private/'sheets.json').read_text())
        sheets={str(s.get('sheet_id',k)):s for k,s in sheets.items()}
        if dataset=='industrytab_1k':
            accepted,rejected,raw_data,mapping_details=audit_csv_mapping(Path(manifest['server_root']),sheets)
            mapping_info={'accepted_csvs':accepted,'rejected_csvs':rejected,'accepted_count':len(accepted),**mapping_details}
            write_json(out/'csv_mapping_audit.json',mapping_info)
            print(f'CSV mapping: {len(accepted)} accepted sheets',flush=True)
        zero_sheet=None;zero_query=None;zero_column=None;zero_example=None
        for index,run in enumerate(runs):
            cache=torch.load(run['cache'],map_location='cpu',weights_only=False)
            ids=cache['sheet_ids'];positions=cache['eval_positions']
            sheet_texts=[serialize_sheet(sheets[sid]) for sid in ids]
            example_texts=[with_examples(sheets[sid]) for sid in ids]
            chunks=[];owners=[]
            for i,sid in enumerate(ids):
                values=column_chunks(sheets[sid]);chunks.extend(values);owners.extend([i]*len(values))
            query_texts=[QUERY_PREFIX+x['query'] for x in cache['queries']]
            model=AutoModel.from_pretrained(str(model_dir),local_files_only=True).to(device).eval()
            started=time.time()
            if index==0:
                zero_sheet=encode_texts(model,tokenizer,sheet_texts,device,args.batch_size,256)
                zero_query=encode_texts(model,tokenizer,query_texts,device,args.batch_size,256)
                zero_chunks=encode_texts(model,tokenizer,chunks,device,args.batch_size,256)
                zero_column=group_scores(zero_query,zero_chunks,owners,len(ids))
                zero_example=encode_texts(model,tokenizer,example_texts,device,args.batch_size,256)
                output.setdefault('input_statistics',{})[dataset]={
                    'sheet':token_stats(tokenizer,sheet_texts),'with_examples':token_stats(tokenizer,example_texts),
                    'column_chunks':token_stats(tokenizer,chunks)}
                if dataset=='industrytab_1k':
                    subset_ids=[sid for sid in ids if sid in raw_data]
                    good=set(subset_ids)
                    eligible=[p for p in positions if set(cache['queries'][p]['positives']).issubset(good)]
                    if len(eligible)>=5:
                        mini={'sheet_ids':subset_ids,'queries':cache['queries']}
                        all_indices=[ids.index(sid) for sid in subset_ids]
                        query=zero_query[eligible]
                        results={'sheet_schema':ranking_metrics(mini,eligible,(query @ zero_sheet[all_indices].T).argsort(1,descending=True))}
                        csv_stats={}
                        for block_size,label in [(1,'row'),(10,'block10'),(0,'block256')]:
                            row_chunks=[];row_owners=[]
                            for i,sid in enumerate(subset_ids):
                                header,rows=raw_data[sid]
                                values=csv_chunks(sheets[sid],rows,header,block_size,tokenizer)
                                row_chunks.extend(values);row_owners.extend([i]*len(values))
                            embeddings=encode_texts(model,tokenizer,row_chunks,device,args.batch_size,256)
                            csv_stats[label]=token_stats(tokenizer,row_chunks)
                            for agg,scores in group_scores(query,embeddings,row_owners,len(subset_ids)).items():
                                results[label+'_'+agg]=ranking_metrics(mini,eligible,scores.argsort(1,descending=True))
                        output['csv_subset']={'sheets':len(subset_ids),'test_queries':len(eligible),
                            'query_indices':[cache['queries'][p]['query_index'] for p in eligible],
                            'gold_definition':'all original positive sheets present; original labels unchanged',
                            'backbone':'zero-shot BGE; same backbone for sheet/row/block',
                            'limitations':mapping_details['cell_text_language'],
                            'metrics':results,'input_statistics':csv_stats}
                        write_json(out/'representation.json',output)
            state=torch.load(Path(run['cache']).parent/'finetuned_bge_last4.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state,strict=False)
            tuned_chunks=encode_texts(model,tokenizer,chunks,device,args.batch_size,256)
            tuned_column=group_scores(cache['query_embeddings'],tuned_chunks,owners,len(ids))
            tuned_example=encode_texts(model,tokenizer,example_texts,device,args.batch_size,256)
            metrics={
                'zero_sheet':ranking_metrics(cache,positions,(zero_query[positions] @ zero_sheet.T).argsort(1,descending=True)),
                'zero_sheet_examples':ranking_metrics(cache,positions,(zero_query[positions] @ zero_example.T).argsort(1,descending=True)),
                'tuned_sheet':ranking_metrics(cache,positions,cache['candidates'][positions]),
                'tuned_sheet_examples':ranking_metrics(cache,positions,(cache['query_embeddings'][positions] @ tuned_example.T).argsort(1,descending=True)),
            }
            for agg,scores in zero_column.items():
                metrics['zero_columns_'+agg]=ranking_metrics(cache,positions,scores[positions].argsort(1,descending=True))
            for agg,scores in tuned_column.items():
                metrics['tuned_columns_'+agg]=ranking_metrics(cache,positions,scores[positions].argsort(1,descending=True))
            output['runs'].append({'dataset':dataset,'seed':run['seed'],'metrics':metrics,
                                  'runtime_seconds':time.time()-started})
            print(json.dumps(output['runs'][-1]),flush=True)
            write_json(out/'representation.json',output)
            del model,cache,tuned_chunks,tuned_column,tuned_example
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    print('Completed full-corpus metadata diagnostics and audited CSV-subset retrieval.',flush=True)


if __name__=='__main__':main()
