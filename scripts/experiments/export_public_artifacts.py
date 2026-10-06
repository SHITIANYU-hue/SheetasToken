#!/usr/bin/env python3
"""Export sanitized, inspectable experiment results without raw cell data."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'artifacts/experiments'
SOURCES = {
    'manifest.json': 'docs/server_archive_manifest.json',
    'results_summary.json': 'docs/experiments_results_summary.json',
    'per_query_predictions.json': 'outputs/experiments/audit/per_query_predictions.json',
    'grouped_per_query.json': 'outputs/experiments/followup_audit/grouped_per_query.json',
    'representation_per_query.json': 'outputs/experiments/followup_audit/representation_per_query.json',
    **{f'grouped_splits/seed{seed}.json': f'outputs/experiments/grouped_splits/seed{seed}.json'
       for seed in (42, 43, 44)},
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def sanitize(value, replacements):
    if isinstance(value, dict):
        return {key: sanitize(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
    return value


def numeric_values(value, path=()):
    if isinstance(value, dict):
        return {p: number for key, item in value.items()
                for p, number in numeric_values(item, path + (key,)).items()}
    if isinstance(value, list):
        return {p: number for index, item in enumerate(value)
                for p, number in numeric_values(item, path + (index,)).items()}
    return {path: value} if isinstance(value, (int, float)) else {}


def check_strings(value):
    patterns = [
        r'/(?:Users|root|home|mnt|Volumes)/',
        r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
        r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        r'\b(?:ghp_|github_pat_|sk-proj-|AKIA)[A-Za-z0-9_-]{12,}',
        r'https?://|ssh://',
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    ]
    for text in strings(value):
        if any(re.search(pattern, text) for pattern in patterns):
            raise ValueError('Potential private path, identity, endpoint, or credential in export')


def check_predictions(rows, queries, sheets, default_dataset=None):
    for row in rows:
        dataset = row.get('dataset', default_dataset)
        reference = queries[dataset][row['query_index']]
        assert row['query'] == reference['query'], 'Query differs from public benchmark'
        if 'gold_ids' in row:
            assert set(map(str, row['gold_ids'])) == set(map(str, reference['positive_sheet_ids']))
        if 'hard_negative_ids' in row:
            assert set(map(str, row['hard_negative_ids'])) == set(map(str, reference.get('hard_negative_sheet_ids', [])))
        for ranking in row.get('ranked_ids', {}).values():
            assert set(map(str, ranking)) <= sheets[dataset], 'Unknown sheet ID in ranking'


def main():
    private_manifest = read(ROOT / SOURCES['manifest.json'])
    archive_root = private_manifest['server_root'].rstrip('/')
    replacements = [(archive_root, '/path/to/archive_root'), (str(ROOT), '/path/to/workspace')]
    public = {name: sanitize(read(ROOT / source), replacements) for name, source in SOURCES.items()}
    manifest = public['manifest.json']
    for key in ('aligned_server_dir', 'initial_aligned_copy', 'previous_server_git_head',
                'previous_server_backup_branch'):
        manifest.pop(key, None)
    manifest['path_convention'] = 'Archive paths are placeholders; supply original checkpoints and caches separately.'
    manifest['public_input_sha256'] = {
        dataset: {name: digest(ROOT / 'data' / dataset / name)
                  for name in ('sheets.json', 'query.json', 'dependency_edges.json')}
        for dataset in ('industrytab_614', 'industrytab_1k')
    }
    queries = {dataset: read(ROOT / 'data' / dataset / 'query.json')
               for dataset in manifest['datasets']}
    sheets = {dataset: set(read(ROOT / 'data' / dataset / 'sheets.json'))
              for dataset in manifest['datasets']}
    for name, default in [('per_query_predictions.json', None),
                          ('grouped_per_query.json', 'industrytab_1k'),
                          ('representation_per_query.json', 'industrytab_1k')]:
        check_predictions(public[name], queries, sheets, default)
    for value in public.values():
        check_strings(value)
    for name, source in SOURCES.items():
        original = read(ROOT / source)
        assert numeric_values(original) == numeric_values(public[name]), 'Numeric evidence changed'
        if name.endswith('per_query.json') or name == 'per_query_predictions.json':
            assert original == public[name], 'Prediction evidence changed'
    DEST.mkdir(parents=True, exist_ok=True)
    for name, value in public.items():
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    checks = {
        'private_archive_paths_replaced': True,
        'server_deployment_and_backup_metadata_omitted': True,
        'identity_endpoint_and_credential_pattern_scan_passed': True,
        'all_exported_queries_match_public_benchmark': True,
        'all_exported_gold_and_hard_negative_sets_match_public_benchmark': True,
        'all_exported_ranked_ids_exist_in_public_benchmark': True,
        'all_numeric_results_preserved': True,
        'all_prediction_records_unchanged': True,
        'raw_spreadsheets_cell_examples_checkpoints_and_logs_included': False,
        'prediction_records': {name: len(public[name]) for name in
                              ('per_query_predictions.json', 'grouped_per_query.json',
                               'representation_per_query.json')},
        'sha256': {name: digest(DEST / name) for name in public},
    }
    (DEST / 'export_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(f'Exported {len(public)} sanitized result files; privacy and public-data checks passed.')


if __name__ == '__main__':
    main()
