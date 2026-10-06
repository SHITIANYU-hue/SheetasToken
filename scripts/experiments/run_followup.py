#!/usr/bin/env python3
"""Run an explicit queue of retrained or grouped experiments on one GPU."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='docs/server_archive_manifest.json');p.add_argument('--queue',choices=['views_a','views_b','grouped'],required=True);a=p.parse_args();root=Path(__file__).resolve().parents[2];os.chdir(root)
    manifest=json.loads(Path(a.manifest).read_text());model=str(Path(manifest['experiment_root'])/'models/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a')
    jobs=([('sheet',s) for s in [42,43,44]]+[('examples',s) for s in [42,43,44]]) if a.queue=='views_a' else ([('columns',s) for s in [42,43,44]] if a.queue=='views_b' else [('sheet',s) for s in [42,43,44]])
    for view,seed in jobs:
        out=Path('outputs/experiments')/('grouped' if a.queue=='grouped' else 'retrained')/(f'seed{seed}' if a.queue=='grouped' else f'{view}/seed{seed}')
        out.mkdir(parents=True,exist_ok=True);cmd=[sys.executable,'scripts/experiments/train_representation.py','--manifest',a.manifest,'--view',view,'--seed',str(seed),'--output-dir',str(out)]
        if a.queue=='grouped':cmd+=['--split-file',f'outputs/experiments/grouped_splits/seed{seed}.json']
        print('START',view,seed,flush=True)
        with (out/'runner.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        if a.queue=='grouped':
            launcher='import torch,runpy,sys;torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15);sys.path.insert(0,"scripts/full_corpus");script=sys.argv.pop(1);runpy.run_path(script,run_name="__main__")'
            jobs2=[('mlp',[sys.executable,'-c',launcher,'scripts/full_corpus/bge_retrain_experiments.py','--mode','rerank','--architecture','mlp','--model',model,'--data-dir','data/industrytab_1k','--cache',str(out/'cache.pt'),'--output-dir',str(out/'mlp'),'--candidate-k','50','--seed',str(seed)]),('gnn',[sys.executable,'-c',launcher,'scripts/full_corpus/gated_graph_refine.py','--cache',str(out/'cache.pt'),'--mlp-checkpoint',str(out/'mlp/frozen_bge_mlp_top50.pt'),'--output-dir',str(out/'gnn'),'--name','full_gnn','--candidate-k','50','--dependency-edges','outputs/experiments/graph_reconstruction/industrytab_1k/graph.json','--dependency-types','formula_reference,summary_source','--layers','1','--gate-init','-4','--seed',str(seed)])]
            for name,cmd in jobs2:
                path=out/name/('frozen_bge_mlp_top50.json' if name=='mlp' else 'full_gnn.json')
                if path.exists():continue
                with (out/(name+'_training.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',view,seed,flush=True)
    print('QUEUE COMPLETE',a.queue,flush=True)
if __name__=='__main__':main()
