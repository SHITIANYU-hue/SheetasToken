#!/usr/bin/env python3
"""Gated relational message passing initialized from a trained MLP reranker."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from bge_retrain_experiments import (
    CachedDataset,
    Reranker,
    candidate_ceiling,
    candidate_labels,
    ranking_metrics,
    reranker_loss,
)


DEPENDENCY_TYPES = ["join_key", "aggregation", "formula_reference", "summary_source"]


def load_dependency_channels(
    path: str | None,
    sheet_ids: list[str],
    selected_types: list[str],
) -> torch.Tensor:
    if not path:
        return torch.zeros(0, len(sheet_ids), len(sheet_ids))
    payload = json.loads(Path(path).read_text())
    index = {sid: position for position, sid in enumerate(sheet_ids)}
    channels = torch.zeros(len(selected_types), len(sheet_ids), len(sheet_ids))
    for key, edge_types in payload.get("edge_all_types", {}).items():
        source, target = key.split(",", 1)
        if source not in index or target not in index:
            continue
        i, j = index[source], index[target]
        for edge_type in edge_types:
            if edge_type in selected_types:
                channel = selected_types.index(edge_type)
                channels[channel, i, j] = 1.0
                channels[channel, j, i] = 1.0
    return channels


class DependencyDataset(CachedDataset):
    def __init__(
        self,
        cache: dict,
        positions: list[int],
        candidate_k: int,
        labels: torch.Tensor,
        dependency: torch.Tensor,
    ):
        super().__init__(cache, positions, candidate_k, labels)
        self.dependency = dependency

    def __getitem__(self, index: int):
        base = super().__getitem__(index)
        candidates = base[-1]
        dependency = self.dependency[:, candidates][:, :, candidates]
        return (*base, dependency)


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--mlp-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--candidate-k", type=int, default=50)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--gate-init", type=float, default=-6.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--joint-learning-rate", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--score-message",
        action="store_true",
        help="Propagate normalized base relevance scores over relation channels.",
    )
    p.add_argument("--score-gate-init", type=float, default=-4.0)
    p.add_argument("--dependency-edges")
    p.add_argument(
        "--dependency-types",
        default="join_key,aggregation,formula_reference,summary_source",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(adjacency.size(-1), device=adjacency.device)[None]
    adjacency = F.relu(adjacency) * (1 - eye)
    return adjacency / adjacency.sum(-1, keepdim=True).clamp(min=1e-9)


def relation_channels(
    query: torch.Tensor,
    nodes: torch.Tensor,
    schema: torch.Tensor,
    shape: torch.Tensor,
    dependency: torch.Tensor,
) -> list[torch.Tensor]:
    normalized_nodes = F.normalize(nodes, dim=-1)
    normalized_query = F.normalize(query, dim=-1)
    semantic = (
        torch.bmm(normalized_nodes, normalized_nodes.transpose(1, 2)) + 1
    ) * 0.5
    query_similarity = (normalized_nodes * normalized_query[:, None]).sum(-1)
    query_graph = (
        torch.einsum("bi,bj->bij", query_similarity, query_similarity) + 1
    ) * 0.5
    channels = [
        normalize_adjacency(semantic),
        normalize_adjacency(query_graph),
        normalize_adjacency(schema),
        normalize_adjacency(shape),
    ]
    channels.extend(
        normalize_adjacency(channel) for channel in dependency.unbind(dim=1)
    )
    return channels


class GatedRelationLayer(nn.Module):
    def __init__(self, dim: int, gate_init: float, dropout: float, num_channels: int):
        super().__init__()
        self.values = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(num_channels))
        self.channel_gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_channels),
        )
        self.update = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.residual_gate = nn.Parameter(torch.tensor(gate_init))

    def forward(
        self,
        query: torch.Tensor,
        nodes: torch.Tensor,
        schema: torch.Tensor,
        shape: torch.Tensor,
        dependency: torch.Tensor,
    ) -> torch.Tensor:
        channels = relation_channels(query, nodes, schema, shape, dependency)
        messages = torch.stack(
            [torch.bmm(adjacency, value(nodes)) for adjacency, value in zip(channels, self.values)],
            dim=2,
        )
        query_expanded = query[:, None].expand_as(nodes)
        weights = torch.softmax(self.channel_gate(torch.cat([nodes, query_expanded], dim=-1)), dim=-1)
        message = (messages * weights[:, :, :, None]).sum(dim=2)
        update = self.update(torch.cat([nodes, message, query_expanded], dim=-1))
        gate = torch.sigmoid(self.residual_gate)
        return F.normalize(nodes + gate * update, dim=-1)


class GatedGraphFromMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        layers: int,
        gate_init: float,
        dropout: float,
        mlp_state: dict,
        dependency_channels: int,
        score_message: bool = False,
        score_gate_init: float = -4.0,
    ):
        super().__init__()
        self.base = Reranker(dim, "mlp")
        self.base.load_state_dict(mlp_state, strict=True)
        self.score_message = score_message
        self.graph_layers = nn.ModuleList(
            GatedRelationLayer(dim, gate_init, dropout, 4 + dependency_channels) for _ in range(layers)
        )
        if score_message:
            score_channels = 4 + dependency_channels
            score_hidden = max(16, score_channels * 4)
            self.score_refiner = nn.Sequential(
                nn.Linear(score_channels + 1, score_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(score_hidden, 1),
            )
            nn.init.zeros_(self.score_refiner[-1].weight)
            nn.init.zeros_(self.score_refiner[-1].bias)
            self.score_gate = nn.Parameter(torch.tensor(score_gate_init))

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def unfreeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad = True

    def forward(
        self,
        query: torch.Tensor,
        nodes: torch.Tensor,
        raw_scores: torch.Tensor,
        schema: torch.Tensor,
        shape: torch.Tensor,
        dependency: torch.Tensor,
    ) -> torch.Tensor:
        base_logits = (
            self.base(query, nodes, raw_scores, schema, shape)
            if self.score_message
            else None
        )
        q = F.normalize(self.base.query_proj(query), dim=-1)
        h = F.normalize(self.base.node_proj(nodes), dim=-1)
        initial_h = h
        for layer in self.graph_layers:
            h = layer(q, h, schema, shape, dependency)
        query_expanded = q[:, None].expand_as(h)
        rank = torch.linspace(1.0, 0.0, h.size(1), device=h.device)[None, :, None].expand(
            h.size(0), -1, -1
        )
        features = torch.cat(
            [
                h,
                query_expanded,
                (h - query_expanded).abs(),
                h * query_expanded,
                raw_scores[:, :, None],
                rank,
            ],
            dim=-1,
        )
        graph_logits = (
            self.base.score_scale * raw_scores
            + self.base.residual(features).squeeze(-1)
        )
        if not self.score_message:
            return graph_logits
        assert base_logits is not None
        centered = base_logits - base_logits.mean(dim=1, keepdim=True)
        normalized_scores = centered / centered.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp(min=1e-6)
        channels = torch.stack(
            relation_channels(q, initial_h, schema, shape, dependency),
            dim=1,
        )
        neighbor_scores = torch.einsum(
            "bcij,bj->bci", channels, normalized_scores
        ).transpose(1, 2)
        score_features = torch.cat(
            [normalized_scores[:, :, None], neighbor_scores],
            dim=-1,
        )
        score_delta = self.score_refiner(score_features).squeeze(-1)
        return graph_logits + torch.sigmoid(self.score_gate) * score_delta


@torch.inference_mode()
def predict(
    model: nn.Module,
    cache: dict,
    positions: list[int],
    candidate_k: int,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
    dependency: torch.Tensor,
) -> torch.Tensor:
    loader = DataLoader(
        DependencyDataset(cache, positions, candidate_k, labels, dependency),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    output = []
    for _, query, nodes, raw, schema, shape, _, candidates, batch_dependency in loader:
        logits = model(
            query.to(device),
            nodes.to(device),
            raw.to(device),
            schema.to(device),
            shape.to(device),
            batch_dependency.to(device),
        )
        order = logits.argsort(1, descending=True).cpu()
        output.append(torch.gather(candidates, 1, order))
    return torch.cat(output)


def main() -> None:
    args = args_parser()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    selected_dependency_types = [
        value.strip() for value in args.dependency_types.split(",") if value.strip()
    ]
    dependency = load_dependency_channels(
        args.dependency_edges,
        cache["sheet_ids"],
        selected_dependency_types,
    )
    checkpoint = torch.load(args.mlp_checkpoint, map_location="cpu", weights_only=False)
    labels = candidate_labels(cache, args.candidate_k)
    model = GatedGraphFromMLP(
        cache["sheet_embeddings"].size(1),
        args.layers,
        args.gate_init,
        args.dropout,
        checkpoint["state_dict"],
        dependency.size(0),
        args.score_message,
        args.score_gate_init,
    ).to(device)
    model.freeze_base()
    loader = DataLoader(
        DependencyDataset(cache, cache["train_core"], args.candidate_k, labels, dependency),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, len(loader)),
        num_training_steps=len(loader) * args.epochs,
    )
    initial_ranking = predict(
        model,
        cache,
        cache["val_positions"],
        args.candidate_k,
        labels,
        device,
        args.batch_size,
        dependency,
    )
    initial_validation = ranking_metrics(cache, cache["val_positions"], initial_ranking)
    best_validation = float(initial_validation["NDCG@5"])
    best_state = copy.deepcopy(model.state_dict())
    history = [{
        "epoch": 0,
        "validation": initial_validation,
        "gates": [torch.sigmoid(layer.residual_gate).item() for layer in model.graph_layers],
    }]
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for _, query, nodes, raw, schema, shape, batch_labels, _, batch_dependency in loader:
            logits = model(
                query.to(device),
                nodes.to(device),
                raw.to(device),
                schema.to(device),
                shape.to(device),
                batch_dependency.to(device),
            )
            loss, _ = reranker_loss(logits, batch_labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                1.0,
            )
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        validation_ranking = predict(
            model,
            cache,
            cache["val_positions"],
            args.candidate_k,
            labels,
            device,
            args.batch_size,
            dependency,
        )
        validation = ranking_metrics(cache, cache["val_positions"], validation_ranking)
        record = {
            "epoch": epoch,
            "loss": total_loss / len(loader),
            "validation": validation,
            "gates": [
                torch.sigmoid(layer.residual_gate).item() for layer in model.graph_layers
            ],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["NDCG@5"] > best_validation:
            best_validation = float(validation["NDCG@5"])
            best_state = copy.deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state)
    if args.joint_learning_rate > 0:
        model.unfreeze_base()
        optimizer = AdamW(model.parameters(), lr=args.joint_learning_rate, weight_decay=0.01)
        for epoch in range(1, 6):
            model.train()
            for _, query, nodes, raw, schema, shape, batch_labels, _, batch_dependency in loader:
                logits = model(
                    query.to(device),
                    nodes.to(device),
                    raw.to(device),
                    schema.to(device),
                    shape.to(device),
                    batch_dependency.to(device),
                )
                loss, _ = reranker_loss(logits, batch_labels.to(device))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            validation_ranking = predict(
                model,
                cache,
                cache["val_positions"],
                args.candidate_k,
                labels,
                device,
                args.batch_size,
                dependency,
            )
            validation = ranking_metrics(cache, cache["val_positions"], validation_ranking)
            record = {
                "joint_epoch": epoch,
                "validation": validation,
                "gates": [
                    torch.sigmoid(layer.residual_gate).item() for layer in model.graph_layers
                ],
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if validation["NDCG@5"] > best_validation:
                best_validation = float(validation["NDCG@5"])
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    eval_ranking = predict(
        model,
        cache,
        cache["eval_positions"],
        args.candidate_k,
        labels,
        device,
        args.batch_size,
        dependency,
    )
    raw_ranking = cache["candidates"][cache["eval_positions"], : args.candidate_k]
    result = {
        "experiment": args.name,
        "protocol": {
            "candidate_k": args.candidate_k,
            "layers": args.layers,
            "gate_init": args.gate_init,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "joint_learning_rate": args.joint_learning_rate,
            "score_message": args.score_message,
            "score_gate_init": args.score_gate_init,
            "seed": args.seed,
            "dependency_edges": args.dependency_edges,
            "dependency_types": selected_dependency_types,
            "best_validation_ndcg": best_validation,
        },
        "candidate_ceiling": candidate_ceiling(cache, cache["eval_positions"], args.candidate_k),
        "raw_bge": ranking_metrics(cache, cache["eval_positions"], raw_ranking),
        "reranked": ranking_metrics(cache, cache["eval_positions"], eval_ranking),
        "history": history,
        "runtime_seconds": time.time() - started,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "args": vars(args)}, output / f"{args.name}.pt")
    (output / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
