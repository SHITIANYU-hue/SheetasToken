"""Metric semantics for exploratory subsets without hard-negative annotations."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/full_corpus'))
from bge_retrain_experiments import ranking_metrics


def test_missing_hard_negative_annotations_are_undefined():
    cache = {
        'sheet_ids': ['a', 'b', 'c', 'd', 'e'],
        'queries': [{'positives': ['a', 'b'], 'hard_negatives': []}],
    }
    metrics = ranking_metrics(cache, [0], torch.tensor([[0, 1, 2, 3, 4]]))
    assert metrics['NDCG@5'] == 1
    assert metrics['MacroRecall@5'] == 1
    assert metrics['HN-FPR@1'] is None
    assert metrics['hn_eligible'] == 0


def test_hard_negative_denominator_uses_annotated_queries_only():
    cache = {
        'sheet_ids': ['a', 'b', 'c', 'd', 'e'],
        'queries': [{'positives': ['a'], 'hard_negatives': ['b']},
                    {'positives': ['a'], 'hard_negatives': []}],
    }
    metrics = ranking_metrics(cache, [0, 1], torch.tensor([[1, 0, 2, 3, 4], [0, 1, 2, 3, 4]]))
    assert metrics['HN-FPR@1'] == 1
    assert metrics['hn_eligible'] == 1
    assert metrics['hn_fp'] == 1
