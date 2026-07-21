#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baselines.common import (
    evaluate_predictions,
    load_queries,
    load_sheets,
    parse_ks,
    serialize_sheet,
    write_json_atomic,
)


DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-shot dense sheet retrieval with a frozen embedding model"
    )
    parser.add_argument("--sheets-file", default="data/sheets.json")
    parser.add_argument("--queries-file", default="data/query.json")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--query-prefix", default=DEFAULT_QUERY_PREFIX)
    parser.add_argument("--output", default="outputs/baselines/embedding_bge_base.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--metric-ks", default="1,3,5,10")
    parser.add_argument("--max-columns", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--include-examples", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Fail instead of downloading the embedding model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and print the experiment size without loading a model",
    )
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    metric_ks = parse_ks(args.metric_ks)
    sheets = load_sheets(args.sheets_file)
    queries = load_queries(args.queries_file, sheets.keys(), args.max_queries)
    sheet_ids = sorted(sheets.keys(), key=_sheet_sort_key)
    sheet_texts = [
        serialize_sheet(sheets[sheet_id], args.max_columns, args.include_examples)
        for sheet_id in sheet_ids
    ]

    if args.dry_run:
        return {
            "method": "zero_shot_embedding",
            "dry_run": True,
            "num_sheets": len(sheet_ids),
            "num_queries": len(queries),
            "sheet_text_characters": sum(map(len, sheet_texts)),
            "model_name": args.model_name,
        }

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required; run `pip install -r requirements.txt`"
        ) from exc

    started = time.perf_counter()
    model = SentenceTransformer(
        args.model_name,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    sheet_embeddings = np.asarray(
        model.encode(
            sheet_texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    )
    query_texts = [args.query_prefix + item["query"] for item in queries]
    query_embeddings = np.asarray(
        model.encode(
            query_texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    )
    scores = query_embeddings @ sheet_embeddings.T

    predictions: List[Dict[str, Any]] = []
    top_k = min(args.top_k, len(sheet_ids))
    for row, query_item in enumerate(queries):
        top_indices = np.argsort(-scores[row], kind="stable")[:top_k]
        predictions.append(
            {
                **query_item,
                "predicted_sheet_ids": [sheet_ids[index] for index in top_indices],
                "scores": [
                    round(float(scores[row, index]), 8) for index in top_indices
                ],
            }
        )

    report: Dict[str, Any] = {
        "method": "zero_shot_embedding",
        "config": {
            "model_name": args.model_name,
            "query_prefix": args.query_prefix,
            "top_k": args.top_k,
            "metric_ks": list(metric_ks),
            "max_columns": args.max_columns,
            "include_examples": args.include_examples,
        },
        "dataset": {"num_sheets": len(sheet_ids), "num_queries": len(queries)},
        "metrics": evaluate_predictions(predictions, metric_ks),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "predictions": predictions,
    }
    write_json_atomic(args.output, report)
    return report


def _sheet_sort_key(sheet_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(sheet_id))
    except ValueError:
        return (1, sheet_id)


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run(args)
    if report.get("dry_run"):
        print(report)
        return
    print(f"Saved: {Path(args.output)}")
    print(report["metrics"])


if __name__ == "__main__":
    main()
