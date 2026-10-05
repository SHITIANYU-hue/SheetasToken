#!/usr/bin/env python3
"""Train matched dense-only graph controls from the archived MLP checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='docs/server_archive_manifest.json')
    parser.add_argument('--output-dir')
    parser.add_argument('--control', choices=['dense', 'node_permuted'], default='dense')
    parser.add_argument('--memory-fraction', type=float, default=0.15)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(Path(args.manifest).read_text())
    output_root = args.output_dir or ('outputs/rebuttal/dense_only' if args.control=='dense'
                                     else 'outputs/rebuttal/node_permuted')
    launcher = (
        'import runpy,sys,torch; from pathlib import Path; torch.set_num_threads(4); '
        f'torch.cuda.set_per_process_memory_fraction({args.memory_fraction}); '
        'script=sys.argv.pop(1); sys.path.insert(0,str(Path(script).parent)); '
        'runpy.run_path(script,run_name="__main__")'
    )
    for run in manifest['runs']:
        out = Path(output_root) / run['dataset'] / f"seed{run['seed']}"
        out.mkdir(parents=True, exist_ok=True)
        name = 'dense_gnn_top50' if args.control=='dense' else 'node_permuted_gnn_top50'
        result = out / (name+'.json')
        if result.exists():
            print(f'Reusing completed result: {result}', flush=True)
            continue
        command = [sys.executable, '-c', launcher,
                   str(root/'scripts/full_corpus/gated_graph_refine.py'),
                   '--cache', run['cache'], '--mlp-checkpoint', run['mlp_checkpoint'],
                   '--output-dir', str(out), '--name', name,
                   '--candidate-k', '50', '--layers', '1',
                   '--gate-init', str(run['gnn_args']['gate_init']),
                   '--epochs', '20', '--batch-size', '64', '--learning-rate', '5e-4',
                   '--dependency-types', '' if args.control=='dense' else run['gnn_args']['dependency_types'],
                   '--seed', str(run['seed'])]
        if args.control=='node_permuted':
            graph_path=root/'data'/run['dataset']/'dependency_edges.json'
            payload=json.loads(graph_path.read_text())
            sheet_ids=list(json.loads((graph_path.parent/'sheets.json').read_text()))
            permuted=sheet_ids.copy()
            random.Random(20260930+run['seed']).shuffle(permuted)
            mapping=dict(zip(sheet_ids,permuted))
            relabelled={}
            for key,types in payload['edge_all_types'].items():
                a,b=key.split(',')
                new_key=','.join(sorted([mapping[a],mapping[b]],key=int))
                assert new_key not in relabelled
                relabelled[new_key]=types
            assert len(relabelled)==len(payload['edge_all_types'])
            graph={'edge_all_types':relabelled,'node_permutation':mapping,
                   'source':str(graph_path),'permutation_seed':20260930+run['seed'],
                   'control':'node relabeling preserves relation counts and graph topology, not degrees of each named sheet'}
            permuted_path=out/'permuted_dependency_edges.json'
            permuted_path.write_text(json.dumps(graph,indent=2)+'\n')
            command.extend(['--dependency-edges',str(permuted_path)])
        cache_hash = hashlib.sha256(Path(run['cache']).read_bytes()).hexdigest()
        (out/'command.json').write_text(json.dumps({
            'argv': command, 'cache_sha256': cache_hash,
            'control': ('four dense channels; no explicit dependency channels'
                        if args.control=='dense' else graph['control']),
        }, indent=2)+'\n')
        print(f"Starting {run['dataset']} seed {run['seed']}", flush=True)
        with (out/'training.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics = json.loads(result.read_text())['reranked']
        print(json.dumps({'dataset': run['dataset'], 'seed': run['seed'],
                          'metrics': metrics}), flush=True)


if __name__ == '__main__':
    main()
