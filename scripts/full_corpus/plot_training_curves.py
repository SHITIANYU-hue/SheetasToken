#!/usr/bin/env python3
"""Plot the paper's Stage 1 and Stage 2 histories from reproduction outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = (
    ("industrytab_614", "IndustryTab-614"),
    ("industrytab_1k", "IndustryTab-1K"),
)
SEEDS = (42, 43, 44)
FILES = {
    "stage1": "finetuned_bge.json",
    "mlp": "frozen_bge_mlp_top50.json",
    "gnn": "gated_gnn_top50.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def histories(root: Path, dataset: str, stage: str) -> list[list[dict]]:
    result = []
    for seed in SEEDS:
        path = root / dataset / f"seed{seed}" / FILES[stage]
        result.append(json.loads(path.read_text())["history"])
    return result


def band(axis, x, rows, color, label, marker=None) -> None:
    values = np.asarray(rows, dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    axis.plot(x, mean, color=color, linewidth=2, marker=marker, label=label)
    axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.2), constrained_layout=True)
    for row, (dataset, title) in enumerate(DATASETS):
        stage1 = histories(args.input_root, dataset, "stage1")
        mlp = histories(args.input_root, dataset, "mlp")
        gnn = histories(args.input_root, dataset, "gnn")

        left = axes[row, 0]
        right_scale = left.twinx()
        stage1_epochs = np.asarray([item["epoch"] for item in stage1[0]])
        band(
            left,
            stage1_epochs,
            [[item["val"]["NDCG@5"] for item in history] for history in stage1],
            "#0072B2",
            "Validation NDCG@5",
            "o",
        )
        band(
            right_scale,
            stage1_epochs,
            [[item["train_loss"] for item in history] for history in stage1],
            "#D55E00",
            "Training loss",
            "s",
        )
        left.set_title(f"{title}: Stage 1 BGE adaptation")
        left.set_xlabel("Epoch")
        left.set_ylabel("Validation NDCG@5", color="#0072B2")
        right_scale.set_ylabel("Contrastive loss", color="#D55E00")
        left.set_xticks(stage1_epochs)
        left.grid(axis="y", alpha=0.25, linewidth=0.6)
        lines = left.lines + right_scale.lines
        left.legend(lines, [line.get_label() for line in lines], loc="best")

        right = axes[row, 1]
        mlp_epochs = np.asarray([item["epoch"] for item in mlp[0]])
        gnn_epochs = np.asarray([item["epoch"] for item in gnn[0]])
        band(
            right,
            mlp_epochs,
            [[item["val"]["NDCG@5"] for item in history] for history in mlp],
            "#009E73",
            "SAT (MLP)",
        )
        band(
            right,
            gnn_epochs,
            [
                [item["validation"]["NDCG@5"] for item in history]
                for history in gnn
            ],
            "#CC79A7",
            "SAT (GNN)",
        )
        right.set_title(f"{title}: Stage 2 validation")
        right.set_xlabel("Epoch")
        right.set_ylabel("Validation NDCG@5")
        right.set_xlim(0, 20)
        right.set_xticks([0, 5, 10, 15, 20])
        right.grid(alpha=0.25, linewidth=0.6)
        right.legend(loc="best")

    fig.suptitle(
        "Distribution-matched full-corpus training dynamics (mean ± s.d., three seeds)",
        fontsize=11,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
