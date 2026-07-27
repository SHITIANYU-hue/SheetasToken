#!/usr/bin/env python3
"""BGE retrieval, reranking, and query-sheet fine-tuning experiments."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cache", "rerank", "finetune", "cross"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache")
    p.add_argument("--external-eval-json")
    p.add_argument("--initial-state")
    p.add_argument("--architecture", choices=["graph", "mlp"], default="graph")
    p.add_argument("--candidate-k", type=int, default=20)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split-seed",
        type=int,
        help="Seed used only for the query split (default: use --seed).",
    )
    p.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of queries reserved for the final evaluation split.",
    )
    p.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of the remaining training queries reserved for validation.",
    )
    return p.parse_args()


def sheet_key(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def serialize_sheet(sheet: dict[str, Any], max_headers: int = 12) -> str:
    parts = []
    name = str(sheet.get("name", "")).strip()
    if name:
        parts.append(f"name: {name}")
    parts.append(f"shape: {sheet.get('num_rows', '?')} x {sheet.get('num_cols', '?')}")
    headers = []
    for column in list(sheet.get("columns", []))[:max_headers]:
        value = column.get("name", "") if isinstance(column, dict) else column
        value = str(value).strip()
        if value:
            headers.append(value)
    if headers:
        parts.append("columns: " + " | ".join(headers))
    return " ; ".join(parts)


def load_corpus(
    data_dir: Path,
    seed: int,
    eval_ratio: float = 0.1,
    validation_ratio: float = 0.1,
) -> dict[str, Any]:
    raw_sheets = json.loads((data_dir / "sheets.json").read_text())
    sheets = {
        str(record.get("sheet_id", key)): record
        for key, record in raw_sheets.items()
        if isinstance(record, dict)
    }
    sheet_ids = sorted(sheets, key=sheet_key)
    valid = set(sheet_ids)
    raw_queries = json.loads((data_dir / "query.json").read_text())
    queries = []
    for original_index, item in enumerate(raw_queries):
        query = str(item.get("query", "")).strip()
        positives = list(
            dict.fromkeys(
                str(sid)
                for sid in item.get("positive_sheet_ids", item.get("sheet_ids", []))
                if str(sid) in valid
            )
        )
        if query and positives:
            queries.append(
                {
                    "query_index": original_index,
                    "query": query,
                    "positives": positives,
                    "hard_negatives": [
                        str(sid)
                        for sid in item.get(
                            "hard_negative_sheet_ids",
                            item.get("sheet_ids_negative", item.get("negative_sheet_ids", [])),
                        )
                        if str(sid) in valid
                    ],
                }
            )
    shuffled = list(range(len(queries)))
    random.Random(seed).shuffle(shuffled)
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("--eval-ratio must be between 0 and 1")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 and 1")
    eval_size = max(1, int(eval_ratio * len(queries)))
    train_positions = shuffled[:-eval_size]
    eval_positions = shuffled[-eval_size:]
    val_size = max(1, int(validation_ratio * len(train_positions)))
    train_core = train_positions[:-val_size]
    val_positions = train_positions[-val_size:]
    return {
        "sheets": sheets,
        "sheet_ids": sheet_ids,
        "queries": queries,
        "split_seed": seed,
        "train_positions": train_positions,
        "train_core": train_core,
        "val_positions": val_positions,
        "eval_positions": eval_positions,
    }


def encode_texts(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    training: bool = False,
) -> torch.Tensor:
    outputs = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                hidden = model(**encoded, return_dict=True).last_hidden_state[:, 0]
            outputs.append(F.normalize(hidden.float(), dim=-1).cpu() if not training else F.normalize(hidden.float(), dim=-1))
    return torch.cat(outputs, dim=0)


def global_priors(corpus: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    sheets = corpus["sheets"]
    ids = corpus["sheet_ids"]
    header_sets = []
    shapes = []
    for sid in ids:
        sheet = sheets[sid]
        header_sets.append(
            {
                str(column.get("name", "")).strip().lower()
                for column in sheet.get("columns", [])[:12]
                if isinstance(column, dict) and str(column.get("name", "")).strip()
            }
        )
        shapes.append((float(sheet.get("num_rows", 0) or 0), float(sheet.get("num_cols", 0) or 0)))
    n = len(ids)
    schema = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            union = header_sets[i] | header_sets[j]
            value = len(header_sets[i] & header_sets[j]) / max(1, len(union))
            schema[i, j] = schema[j, i] = value
    shape_values = torch.tensor(shapes)
    row = 1.0 / (1.0 + (shape_values[:, None, 0] - shape_values[None, :, 0]).abs())
    col = 1.0 / (1.0 + (shape_values[:, None, 1] - shape_values[None, :, 1]).abs())
    shape = 0.5 * (row + col)
    shape.fill_diagonal_(0)
    return schema, shape


def build_cache(
    corpus: dict[str, Any],
    model_path: str,
    output: Path,
    batch_size: int,
    max_length: int,
    device: torch.device,
    external_eval_json: str | None = None,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    sheet_texts = [serialize_sheet(corpus["sheets"][sid]) for sid in corpus["sheet_ids"]]
    query_texts = [QUERY_PREFIX + item["query"] for item in corpus["queries"]]
    started = time.time()
    sheet_embeddings = encode_texts(model, tokenizer, sheet_texts, device, batch_size, max_length)
    query_embeddings = encode_texts(model, tokenizer, query_texts, device, batch_size, max_length)
    scores = query_embeddings @ sheet_embeddings.T
    candidate_scores, candidates = torch.topk(scores, k=50, dim=1)
    schema, shape = global_priors(corpus)
    payload = {
        "sheet_ids": corpus["sheet_ids"],
        "queries": corpus["queries"],
        "split_seed": corpus["split_seed"],
        "train_positions": corpus["train_positions"],
        "train_core": corpus["train_core"],
        "val_positions": corpus["val_positions"],
        "eval_positions": corpus["eval_positions"],
        "sheet_embeddings": sheet_embeddings,
        "query_embeddings": query_embeddings,
        "candidates": candidates,
        "candidate_scores": candidate_scores,
        "schema": schema,
        "shape": shape,
        "model_path": model_path,
        "runtime_seconds": time.time() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    if external_eval_json:
        external = json.loads(Path(external_eval_json).read_text())
        by_query = {row["query"]: row["candidate_sheet_ids"] for row in external["predictions"]}
        exact5 = exact50 = 0
        for position in corpus["eval_positions"]:
            query = corpus["queries"][position]["query"]
            expected = list(map(str, by_query[query]))
            actual = [corpus["sheet_ids"][index] for index in candidates[position].tolist()]
            exact5 += actual[:5] == expected[:5]
            exact50 += actual[:50] == expected[:50]
        print(f"external candidate agreement exact_top5={exact5}/179 exact_top50={exact50}/179", flush=True)
    del model
    return payload


def candidate_labels(cache: dict[str, Any], k: int) -> torch.Tensor:
    ids = cache["sheet_ids"]
    labels = torch.zeros(len(cache["queries"]), k)
    for position, item in enumerate(cache["queries"]):
        positives = set(item["positives"])
        for rank, sheet_index in enumerate(cache["candidates"][position, :k].tolist()):
            labels[position, rank] = float(ids[sheet_index] in positives)
    return labels


def ranking_metrics(
    cache: dict[str, Any],
    positions: list[int],
    ordered_candidate_indices: torch.Tensor,
    k: int = 5,
) -> dict[str, float | int]:
    ids = cache["sheet_ids"]
    totals = {"NDCG@5": 0.0, "MacroRecall@5": 0.0, "Precision@5": 0.0, "Hit@5": 0.0, "MRR@5": 0.0}
    hn_eligible = hn_fp = 0
    for row, position in enumerate(positions):
        item = cache["queries"][position]
        positives = set(item["positives"])
        ranked = [ids[index] for index in ordered_candidate_indices[row, :k].tolist()]
        hits = [sid in positives for sid in ranked]
        captured = sum(hits)
        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
        idcg = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(positives))))
        first = next((rank + 1 for rank, hit in enumerate(hits) if hit), None)
        totals["NDCG@5"] += dcg / idcg
        totals["MacroRecall@5"] += captured / len(positives)
        totals["Precision@5"] += captured / k
        totals["Hit@5"] += float(captured > 0)
        totals["MRR@5"] += 1 / first if first else 0
        hard = set(item["hard_negatives"])
        if hard:
            hn_eligible += 1
            hn_fp += int(ranked[0] in hard)
    count = len(positions)
    return {
        **{name: round(value / count, 6) for name, value in totals.items()},
        "HN-FPR@1": round(hn_fp / hn_eligible, 6),
        "hn_eligible": hn_eligible,
        "hn_fp": hn_fp,
    }


def candidate_ceiling(cache: dict[str, Any], positions: list[int], k: int) -> dict[str, float]:
    ids = cache["sheet_ids"]
    recall = hit = all_relevant = oracle = 0.0
    for position in positions:
        positives = set(cache["queries"][position]["positives"])
        candidates = {ids[index] for index in cache["candidates"][position, :k].tolist()}
        captured = len(positives & candidates)
        recall += captured / len(positives)
        hit += float(captured > 0)
        all_relevant += float(positives <= candidates)
        oracle += min(captured, 5) / len(positives)
    n = len(positions)
    return {
        f"CandidateRecall@{k}": recall / n,
        f"CandidateHit@{k}": hit / n,
        f"AllRelevant@{k}": all_relevant / n,
        "OracleRecall@5": oracle / n,
    }


class DenseGAT(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.src = nn.Linear(dim, 1, bias=False)
        self.dst = nn.Linear(dim, 1, bias=False)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        energy = F.leaky_relu(self.src(h) + self.dst(h).transpose(1, 2), 0.2)
        energy = energy.masked_fill(adj <= 0, -1e9)
        alpha = torch.softmax(energy, dim=-1) * (adj > 0)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return self.norm(x + self.out(torch.bmm(self.dropout(alpha), h)))


class Reranker(nn.Module):
    def __init__(self, dim: int, architecture: str):
        super().__init__()
        self.architecture = architecture
        self.query_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.node_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.score_scale = nn.Parameter(torch.tensor(10.0))
        if architecture == "graph":
            self.channel_logits_1 = nn.Parameter(torch.zeros(4))
            self.channel_logits_2 = nn.Parameter(torch.zeros(4))
            self.gat = nn.ModuleList([DenseGAT(dim), DenseGAT(dim)])
        self.residual = nn.Sequential(
            nn.Linear(dim * 4 + 2, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        query: torch.Tensor,
        nodes: torch.Tensor,
        raw_scores: torch.Tensor,
        schema: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        q = F.normalize(self.query_proj(query), dim=-1)
        h = F.normalize(self.node_proj(nodes), dim=-1)
        if self.architecture == "graph":
            sem = (torch.bmm(h, h.transpose(1, 2)) + 1) * 0.5
            qsim = (h * q[:, None]).sum(-1)
            qgraph = (torch.einsum("bi,bj->bij", qsim, qsim) + 1) * 0.5
            channels = torch.stack([sem, qgraph, schema, shape], dim=1)
            w1 = torch.softmax(self.channel_logits_1, 0)
            w2 = torch.softmax(self.channel_logits_2, 0)
            a1 = (channels * w1[None, :, None, None]).sum(1).relu()
            a2 = (channels * w2[None, :, None, None]).sum(1).relu()
            adj = torch.bmm(a1, a2)
            eye = torch.eye(adj.size(-1), device=adj.device)[None]
            adj = adj * (1 - eye)
            adj = adj / adj.sum(-1, keepdim=True).clamp(min=1e-9)
            for layer in self.gat:
                h = layer(h, adj)
        qx = q[:, None].expand_as(h)
        rank = torch.linspace(1.0, 0.0, h.size(1), device=h.device)[None, :, None].expand(h.size(0), -1, -1)
        features = torch.cat([h, qx, (h - qx).abs(), h * qx, raw_scores[:, :, None], rank], dim=-1)
        return self.score_scale * raw_scores + self.residual(features).squeeze(-1)


class CachedDataset(Dataset):
    def __init__(self, cache: dict[str, Any], positions: list[int], k: int, labels: torch.Tensor):
        self.cache = cache
        self.positions = positions
        self.k = k
        self.labels = labels

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        position = self.positions[index]
        candidates = self.cache["candidates"][position, : self.k]
        return (
            torch.tensor(position),
            self.cache["query_embeddings"][position],
            self.cache["sheet_embeddings"][candidates],
            self.cache["candidate_scores"][position, : self.k],
            self.cache["schema"][candidates][:, candidates],
            self.cache["shape"][candidates][:, candidates],
            self.labels[position],
            candidates,
        )


def reranker_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    valid_positive = labels.sum(1) > 0
    if valid_positive.any():
        positive_logits = logits.masked_fill(labels <= 0, -1e9)
        listwise = (
            torch.logsumexp(logits[valid_positive], dim=1)
            - torch.logsumexp(positive_logits[valid_positive], dim=1)
        ).mean()
        hardest_positive = logits.masked_fill(labels <= 0, 1e9).min(1).values
        hardest_negative = logits.masked_fill(labels > 0, -1e9).max(1).values
        pairwise = F.relu(0.2 - hardest_positive[valid_positive] + hardest_negative[valid_positive]).mean()
    else:
        listwise = logits.sum() * 0
        pairwise = logits.sum() * 0
    positive_count = labels.sum().clamp(min=1)
    negative_count = (1 - labels).sum().clamp(min=1)
    pos_weight = (negative_count / positive_count).clamp(max=20)
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    loss = listwise + 0.5 * bce + 0.2 * pairwise
    return loss, {"listwise": listwise.item(), "bce": bce.item(), "pairwise": pairwise.item()}


@torch.inference_mode()
def predict_reranker(
    model: nn.Module,
    cache: dict[str, Any],
    positions: list[int],
    k: int,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(CachedDataset(cache, positions, k, labels), batch_size=batch_size, shuffle=False)
    rankings = []
    model.eval()
    for _, query, nodes, raw, schema, shape, _, candidates in loader:
        logits = model(query.to(device), nodes.to(device), raw.to(device), schema.to(device), shape.to(device))
        order = logits.argsort(1, descending=True).cpu()
        rankings.append(torch.gather(candidates, 1, order))
    return torch.cat(rankings)


def train_reranker(args: argparse.Namespace, cache: dict[str, Any], output_dir: Path, device: torch.device) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    labels = candidate_labels(cache, args.candidate_k)
    train_dataset = CachedDataset(cache, cache["train_core"], args.candidate_k, labels)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    model = Reranker(cache["sheet_embeddings"].size(1), args.architecture).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, len(loader)),
        num_training_steps=max(1, len(loader) * args.epochs),
    )
    best_val = -1.0
    best_state = None
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "listwise": 0.0, "bce": 0.0, "pairwise": 0.0}
        for _, query, nodes, raw, schema, shape, batch_labels, _ in loader:
            logits = model(query.to(device), nodes.to(device), raw.to(device), schema.to(device), shape.to(device))
            loss, parts = reranker_loss(logits, batch_labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            totals["loss"] += loss.item()
            for key in parts:
                totals[key] += parts[key]
        val_rank = predict_reranker(
            model, cache, cache["val_positions"], args.candidate_k, labels, device, args.batch_size
        )
        val_metrics = ranking_metrics(cache, cache["val_positions"], val_rank)
        record = {
            "epoch": epoch,
            **{key: value / len(loader) for key, value in totals.items()},
            "val": val_metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_metrics["NDCG@5"] > best_val:
            best_val = float(val_metrics["NDCG@5"])
            best_state = copy.deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state)
    eval_rank = predict_reranker(
        model, cache, cache["eval_positions"], args.candidate_k, labels, device, args.batch_size
    )
    raw_rank = cache["candidates"][cache["eval_positions"], : args.candidate_k]
    result = {
        "experiment": f"frozen_bge_{args.architecture}_reranker_top{args.candidate_k}",
        "protocol": {
            "train_queries": len(cache["train_core"]),
            "validation_queries": len(cache["val_positions"]),
            "eval_queries": len(cache["eval_positions"]),
            "num_sheets": len(cache["sheet_ids"]),
            "candidate_k": args.candidate_k,
            "seed": args.seed,
            "best_validation_ndcg": best_val,
        },
        "candidate_ceiling": candidate_ceiling(cache, cache["eval_positions"], args.candidate_k),
        "raw_bge": ranking_metrics(cache, cache["eval_positions"], raw_rank),
        "reranked": ranking_metrics(cache, cache["eval_positions"], eval_rank),
        "history": history,
        "runtime_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frozen_bge_{args.architecture}_top{args.candidate_k}"
    torch.save({"state_dict": best_state, "args": vars(args)}, output_dir / f"{stem}.pt")
    (output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


class QuerySheetPairs(Dataset):
    def __init__(self, cache: dict[str, Any], positions: list[int], epoch: int, seed: int):
        self.cache = cache
        self.positions = positions
        self.epoch = epoch
        self.seed = seed

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> tuple[int, int, int]:
        position = self.positions[index]
        item = self.cache["queries"][position]
        id_to_index = {sid: i for i, sid in enumerate(self.cache["sheet_ids"])}
        rng = random.Random(self.seed * 100003 + self.epoch * 1009 + position)
        positive = id_to_index[rng.choice(item["positives"])]
        positive_ids = set(item["positives"])
        hard = next(
            index
            for index in self.cache["candidates"][position].tolist()
            if self.cache["sheet_ids"][index] not in positive_ids
        )
        return position, positive, hard


def raw_metrics_for_model(
    model: nn.Module,
    tokenizer,
    corpus: dict[str, Any],
    positions: list[int],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    sheet_texts = [serialize_sheet(corpus["sheets"][sid]) for sid in corpus["sheet_ids"]]
    query_texts = [QUERY_PREFIX + corpus["queries"][position]["query"] for position in positions]
    sheet_emb = encode_texts(model, tokenizer, sheet_texts, device, batch_size, max_length)
    query_emb = encode_texts(model, tokenizer, query_texts, device, batch_size, max_length)
    scores = query_emb @ sheet_emb.T
    candidate_scores, candidates = torch.topk(scores, 50, dim=1)
    mini_cache = {
        "sheet_ids": corpus["sheet_ids"],
        "queries": [corpus["queries"][position] for position in positions],
    }
    metrics = ranking_metrics(mini_cache, list(range(len(positions))), candidates)
    return metrics, sheet_emb, query_emb, candidates, candidate_scores


def fine_tune_bge(args: argparse.Namespace, corpus: dict[str, Any], base_cache: dict[str, Any], output_dir: Path, device: torch.device) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer in model.encoder.layer[-4:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    if getattr(model, "pooler", None):
        for parameter in model.pooler.parameters():
            parameter.requires_grad = True
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(base_cache["train_core"]) / args.batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, steps_per_epoch // 2),
        num_training_steps=steps_per_epoch * args.epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sheet_texts = [serialize_sheet(corpus["sheets"][sid]) for sid in corpus["sheet_ids"]]
    best_val = -1.0
    best_trainable = None
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        dataset = QuerySheetPairs(base_cache, base_cache["train_core"], epoch, args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model.train()
        total = 0.0
        for query_positions, positive_indices, hard_indices in loader:
            queries = [QUERY_PREFIX + corpus["queries"][position]["query"] for position in query_positions.tolist()]
            positives = [sheet_texts[index] for index in positive_indices.tolist()]
            hard = [sheet_texts[index] for index in hard_indices.tolist()]
            query_encoded = tokenizer(
                queries, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt"
            )
            sheet_encoded = tokenizer(
                positives + hard, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt"
            )
            query_encoded = {k: v.to(device) for k, v in query_encoded.items()}
            sheet_encoded = {k: v.to(device) for k, v in sheet_encoded.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                q = F.normalize(model(**query_encoded, return_dict=True).last_hidden_state[:, 0].float(), dim=-1)
                s = F.normalize(model(**sheet_encoded, return_dict=True).last_hidden_state[:, 0].float(), dim=-1)
                logits = q @ s.T / 0.02
                targets = torch.arange(q.size(0), device=device)
                loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
        val_metrics, _, _, _, _ = raw_metrics_for_model(
            model,
            tokenizer,
            corpus,
            base_cache["val_positions"],
            device,
            max(16, args.batch_size),
            args.max_length,
        )
        record = {"epoch": epoch, "train_loss": total / len(loader), "val": val_metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_metrics["NDCG@5"] > best_val:
            best_val = float(val_metrics["NDCG@5"])
            best_trainable = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
                if any(name.startswith(f"encoder.layer.{index}.") for index in range(8, 12))
                or name.startswith("pooler.")
            }
    assert best_trainable is not None
    state = model.state_dict()
    state.update(best_trainable)
    model.load_state_dict(state)
    model.eval()
    all_metrics, sheet_emb, query_emb, candidates, candidate_scores = raw_metrics_for_model(
        model,
        tokenizer,
        corpus,
        list(range(len(corpus["queries"]))),
        device,
        max(16, args.batch_size),
        args.max_length,
    )
    schema, shape = global_priors(corpus)
    tuned_cache = {
        **{
            key: base_cache[key]
            for key in [
                "sheet_ids",
                "queries",
                "train_positions",
                "train_core",
                "val_positions",
                "eval_positions",
            ]
        },
        "split_seed": base_cache.get("split_seed"),
        "sheet_embeddings": sheet_emb,
        "query_embeddings": query_emb,
        "candidates": candidates,
        "candidate_scores": candidate_scores,
        "schema": schema,
        "shape": shape,
        "model_path": args.model,
        "fine_tuned": True,
    }
    eval_raw = ranking_metrics(
        tuned_cache,
        tuned_cache["eval_positions"],
        tuned_cache["candidates"][tuned_cache["eval_positions"]],
    )
    result = {
        "experiment": "query_sheet_finetuned_bge_last4",
        "protocol": {
            "train_queries": len(base_cache["train_core"]),
            "validation_queries": len(base_cache["val_positions"]),
            "eval_queries": len(base_cache["eval_positions"]),
            "split_seed": base_cache.get("split_seed"),
            "epochs": args.epochs,
            "best_validation_ndcg": best_val,
        },
        "eval": eval_raw,
        "history": history,
        "runtime_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tuned_cache, output_dir / "finetuned_bge_cache.pt")
    torch.save(best_trainable, output_dir / "finetuned_bge_last4.pt")
    (output_dir / "finetuned_bge.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


class CrossEncoder(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path, local_files_only=True)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for layer in self.backbone.encoder.layer[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.backbone(**encoded, return_dict=True).last_hidden_state[:, 0]
        return self.head(hidden).squeeze(-1)


def cross_pairs(
    cache: dict[str, Any],
    positions: list[int],
    k: int,
    epoch: int,
    seed: int,
) -> list[tuple[int, int, float]]:
    pairs = []
    ids = cache["sheet_ids"]
    for position in positions:
        positives = set(cache["queries"][position]["positives"])
        candidate_list = cache["candidates"][position, :k].tolist()
        positive_candidates = [index for index in candidate_list if ids[index] in positives]
        negative_candidates = [index for index in candidate_list if ids[index] not in positives]
        rng = random.Random(seed * 100003 + epoch * 1009 + position)
        if positive_candidates:
            for index in positive_candidates[:2]:
                pairs.append((position, index, 1.0))
        for index in negative_candidates[: min(8, len(negative_candidates))]:
            pairs.append((position, index, 0.0))
        rng.shuffle(pairs[-10:])
    random.Random(seed + epoch).shuffle(pairs)
    return pairs


@torch.inference_mode()
def cross_scores(
    model: CrossEncoder,
    tokenizer,
    corpus: dict[str, Any],
    cache: dict[str, Any],
    positions: list[int],
    k: int,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    model.eval()
    outputs = []
    flat = [(position, index) for position in positions for index in cache["candidates"][position, :k].tolist()]
    for start in range(0, len(flat), batch_size):
        chunk = flat[start : start + batch_size]
        queries = [corpus["queries"][position]["query"] for position, _ in chunk]
        sheets = [serialize_sheet(corpus["sheets"][corpus["sheet_ids"][index]]) for _, index in chunk]
        encoded = tokenizer(
            queries,
            sheets,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs.append(model(encoded).float().cpu())
    return torch.cat(outputs).view(len(positions), k)


def train_cross(args: argparse.Namespace, corpus: dict[str, Any], cache: dict[str, Any], output_dir: Path, device: torch.device) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = CrossEncoder(args.model).to(device)
    if args.initial_state:
        tuned = torch.load(args.initial_state, map_location="cpu", weights_only=False)
        backbone_state = model.backbone.state_dict()
        matched = {
            name: tensor
            for name, tensor in tuned.items()
            if name in backbone_state and backbone_state[name].shape == tensor.shape
        }
        backbone_state.update(matched)
        model.backbone.load_state_dict(backbone_state)
        print(f"loaded initial tuned backbone tensors={len(matched)}", flush=True)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_val = -1
    best_state = None
    best_alpha = 0.0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        pairs = cross_pairs(cache, cache["train_core"], args.candidate_k, epoch, args.seed)
        model.train()
        total = 0.0
        for start in range(0, len(pairs), args.batch_size):
            chunk = pairs[start : start + args.batch_size]
            queries = [corpus["queries"][position]["query"] for position, _, _ in chunk]
            sheets = [serialize_sheet(corpus["sheets"][corpus["sheet_ids"][index]]) for _, index, _ in chunk]
            labels = torch.tensor([label for _, _, label in chunk], device=device)
            encoded = tokenizer(
                queries, sheets, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt"
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(encoded)
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=torch.tensor(4.0, device=device))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        val_scores = cross_scores(
            model, tokenizer, corpus, cache, cache["val_positions"], args.candidate_k, device, args.batch_size, args.max_length
        )
        raw = cache["candidate_scores"][cache["val_positions"], : args.candidate_k]
        candidates = cache["candidates"][cache["val_positions"], : args.candidate_k]
        epoch_best = (-1.0, 0.0, None)
        for alpha in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            order = (val_scores + alpha * raw).argsort(1, descending=True)
            ranking = torch.gather(candidates, 1, order)
            metrics = ranking_metrics(cache, cache["val_positions"], ranking)
            if metrics["NDCG@5"] > epoch_best[0]:
                epoch_best = (float(metrics["NDCG@5"]), alpha, metrics)
        record = {"epoch": epoch, "train_loss": total / math.ceil(len(pairs) / args.batch_size), "val": epoch_best[2], "alpha": epoch_best[1]}
        history.append(record)
        print(json.dumps(record), flush=True)
        if epoch_best[0] > best_val:
            best_val = epoch_best[0]
            best_alpha = epoch_best[1]
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
                if name.startswith("backbone.encoder.layer.10.")
                or name.startswith("backbone.encoder.layer.11.")
                or name.startswith("head.")
            }
    assert best_state is not None
    state = model.state_dict()
    state.update(best_state)
    model.load_state_dict(state)
    eval_scores = cross_scores(
        model, tokenizer, corpus, cache, cache["eval_positions"], args.candidate_k, device, args.batch_size, args.max_length
    )
    raw = cache["candidate_scores"][cache["eval_positions"], : args.candidate_k]
    candidates = cache["candidates"][cache["eval_positions"], : args.candidate_k]
    order = (eval_scores + best_alpha * raw).argsort(1, descending=True)
    ranking = torch.gather(candidates, 1, order)
    result = {
        "experiment": f"bge_cross_encoder_top{args.candidate_k}",
        "protocol": {"best_validation_ndcg": best_val, "raw_score_alpha": best_alpha},
        "candidate_ceiling": candidate_ceiling(cache, cache["eval_positions"], args.candidate_k),
        "raw_bge": ranking_metrics(cache, cache["eval_positions"], candidates),
        "reranked": ranking_metrics(cache, cache["eval_positions"], ranking),
        "history": history,
        "runtime_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_dir / f"cross_encoder_top{args.candidate_k}.pt")
    (output_dir / f"cross_encoder_top{args.candidate_k}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = load_corpus(
        Path(args.data_dir),
        args.seed if args.split_seed is None else args.split_seed,
        args.eval_ratio,
        args.validation_ratio,
    )
    if args.mode == "cache":
        output = Path(args.cache or output_dir / "base_bge_cache.pt")
        cache = build_cache(
            corpus,
            args.model,
            output,
            args.batch_size,
            args.max_length,
            device,
            args.external_eval_json,
        )
        raw = cache["candidates"][cache["eval_positions"]]
        result = {
            "raw_bge": ranking_metrics(cache, cache["eval_positions"], raw),
            "top10": candidate_ceiling(cache, cache["eval_positions"], 10),
            "top20": candidate_ceiling(cache, cache["eval_positions"], 20),
            "top50": candidate_ceiling(cache, cache["eval_positions"], 50),
        }
        print(json.dumps(result, indent=2))
    else:
        if not args.cache:
            raise ValueError("--cache is required for this mode")
        cache = torch.load(args.cache, map_location="cpu", weights_only=False)
        if args.mode == "rerank":
            train_reranker(args, cache, output_dir, device)
        elif args.mode == "finetune":
            fine_tune_bge(args, corpus, cache, output_dir, device)
        else:
            train_cross(args, corpus, cache, output_dir, device)


if __name__ == "__main__":
    main()
