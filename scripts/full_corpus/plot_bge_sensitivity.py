#!/usr/bin/env python3
"""Generate BGE full-corpus sensitivity figures and an aggregate JSON table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SWEEPS = {
    "Candidate pool": [
        ("Top-10", "new/k10/gnn_k10.json"),
        ("Top-20", "new/k20/gnn_k20.json"),
        ("Top-50 (ref.)", "channels/ref.json"),
    ],
    "GNN depth": [
        ("1 layer (ref.)", "channels/ref.json"),
        ("2 layers", "new/layers2/layers2.json"),
        ("3 layers", "new/layers3/layers3.json"),
    ],
    "Residual gate": [
        ("-8", "new/gate8/gate8.json"),
        ("-6", "new/gate6/gate6.json"),
        ("-4 (ref.)", "channels/ref.json"),
    ],
    "Dependency channels": [
        ("Join", "channels/join.json"),
        ("Aggregation", "channels/agg.json"),
        ("Formula", "channels/formula.json"),
        ("Summary", "channels/summary.json"),
        ("A+F+S", "channels/nojoin.json"),
        ("All", "channels/all.json"),
        ("F+S (ref.)", "channels/ref.json"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_runs(root: Path) -> dict[str, list[tuple[str, dict]]]:
    return {
        sweep: [(label, json.loads((root / path).read_text())) for label, path in rows]
        for sweep, rows in SWEEPS.items()
    }


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def metrics_figure(runs: dict[str, list[tuple[str, dict]]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.0), constrained_layout=True)
    colors = ["#0072B2", "#009E73", "#CC79A7"]
    metric_keys = ["NDCG@5", "MacroRecall@5", "MRR@5"]
    metric_labels = ["NDCG@5", "Macro R@5", "MRR@5"]
    for axis, (title, rows) in zip(axes.flat, runs.items()):
        x = np.arange(len(rows))
        width = 0.24
        for offset, (key, label, color) in enumerate(
            zip(metric_keys, metric_labels, colors)
        ):
            values = [row["reranked"][key] for _, row in rows]
            axis.bar(
                x + (offset - 1) * width,
                values,
                width,
                color=color,
                label=label,
                alpha=0.9,
            )
        axis.set_title(title)
        axis.set_ylabel("Full-corpus test score")
        axis.set_xticks(x)
        axis.set_xticklabels(
            [label for label, _ in rows],
            rotation=25 if len(rows) > 3 else 0,
            ha="right" if len(rows) > 3 else "center",
        )
        minimum = min(
            row["reranked"][key]
            for _, row in rows
            for key in metric_keys
        )
        axis.set_ylim(max(0.0, minimum - 0.05), 0.98)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        axis.legend(loc="lower right")
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def convergence_figure(
    runs: dict[str, list[tuple[str, dict]]], output: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.2))
    palette = plt.get_cmap("tab10")
    for axis, (title, rows) in zip(axes.flat, runs.items()):
        for index, (label, row) in enumerate(rows):
            history = row["history"]
            epochs = [item.get("epoch", item.get("joint_epoch", 0)) for item in history]
            ndcg = [item["validation"]["NDCG@5"] for item in history]
            is_reference = "ref." in label
            axis.plot(
                epochs,
                ndcg,
                label=label,
                color="#D55E00" if is_reference else palette(index),
                linewidth=2.4 if is_reference else 1.3,
                alpha=1.0 if is_reference else 0.85,
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation NDCG@5")
        axis.set_xlim(0, 20)
        axis.set_xticks([0, 5, 10, 15, 20])
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.legend(loc="best", ncol=2 if len(rows) > 4 else 1)
    fig.suptitle("BGE full-corpus sensitivity: validation convergence", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.5, w_pad=1.5)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(
        output.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)


def write_summary(runs: dict[str, list[tuple[str, dict]]], output: Path) -> None:
    payload = {
        "dataset": "IndustryTab-1K",
        "protocol": {
            "sheets": 1002,
            "eval_queries": 179,
            "full_corpus": True,
            "positive_injection": False,
            "stage1": "fine-tuned BAAI/bge-base-en-v1.5",
            "seed": 42,
            "selection_metric": "validation NDCG@5",
        },
        "sweeps": {},
    }
    for sweep, rows in runs.items():
        payload["sweeps"][sweep] = [
            {
                "setting": label,
                "best_validation_NDCG@5": row["protocol"]["best_validation_ndcg"],
                **{
                    key: row["reranked"][key]
                    for key in [
                        "NDCG@5",
                        "MacroRecall@5",
                        "Hit@5",
                        "MRR@5",
                        "HN-FPR@1",
                    ]
                },
            }
            for label, row in rows
        ]
    output.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    style()
    runs = load_runs(args.input_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_figure(runs, args.output_dir / "bge_sensitivity_metrics.pdf")
    convergence_figure(runs, args.output_dir / "bge_sensitivity_convergence.pdf")
    write_summary(runs, args.output_dir / "bge_sensitivity_summary.json")


if __name__ == "__main__":
    main()
