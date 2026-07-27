#!/usr/bin/env python3
"""Export a full-corpus retrieval cache for the RAG/LLM baseline runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retrieval-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    sheet_ids = list(map(str, cache["sheet_ids"]))
    predictions = []
    for position in cache["eval_positions"]:
        query = cache["queries"][position]
        candidate_indices = cache["candidates"][position, : args.retrieval_k].tolist()
        candidate_scores = cache["candidate_scores"][
            position, : args.retrieval_k
        ].tolist()
        predictions.append(
            {
                "query_index": int(query["query_index"]),
                "query": query["query"],
                "relevant_sheet_ids": list(map(str, query["positives"])),
                "hard_negative_sheet_ids": list(
                    map(str, query.get("hard_negatives", []))
                ),
                "candidate_sheet_ids": [
                    sheet_ids[index] for index in candidate_indices
                ],
                "candidate_scores": [
                    round(float(score), 8) for score in candidate_scores
                ],
            }
        )
    payload = {
        "method": "rag",
        "source_cache": str(Path(args.cache)),
        "protocol": {
            "num_sheets": len(sheet_ids),
            "num_total_queries": len(cache["queries"]),
            "num_eval_queries": len(predictions),
            "retrieval_k": args.retrieval_k,
        },
        "predictions": predictions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "num_eval_queries": len(predictions),
                "num_sheets": len(sheet_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
