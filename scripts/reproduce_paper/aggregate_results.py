#!/usr/bin/env python3
"""Aggregate the three paper seeds into a reviewable JSON summary."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = ("NDCG@5", "MacroRecall@5", "Hit@5", "MRR@5", "HN-FPR@1")
SOURCES = {
    "LLM-only": ("llm_only.json", "metrics"),
    "RAG": ("rag.json", "metrics"),
    "RAG+LLM": ("rag_llm.json", "metrics"),
    "SAT (Stage 1 only)": ("finetuned_bge.json", "eval"),
    "SAT (Cross-Encoder)": ("cross_encoder_top50.json", "reranked"),
    "SAT": ("gated_gnn_top50.json", "reranked"),
}


def nested(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value: Any = payload
    for part in key.split("."):
        value = value[part]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary: dict[str, Any] = {"run_dir": str(args.run_dir), "methods": {}}
    for method, (filename, metric_key) in SOURCES.items():
        rows = []
        missing = []
        for seed in (42, 43, 44):
            path = args.run_dir / f"seed{seed}" / filename
            if not path.exists():
                missing.append(str(path))
                continue
            metrics = nested(json.loads(path.read_text()), metric_key)
            rows.append({"seed": seed, **{key: metrics[key] for key in METRICS}})
        aggregate = {}
        for key in METRICS:
            values = [float(row[key]) for row in rows]
            if values:
                aggregate[key] = {
                    "mean": round(statistics.fmean(values), 6),
                    "std": round(statistics.stdev(values), 6)
                    if len(values) > 1
                    else 0.0,
                }
        summary["methods"][method] = {
            "seeds": rows,
            "aggregate": aggregate,
            "missing": missing,
        }

    output = args.output or args.run_dir / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
