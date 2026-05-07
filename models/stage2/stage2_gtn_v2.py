#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 GTN-style training with optional negatives and DDP support.

Design goals:
- Reuse the current Stage1 encoder/backbone weights from classifier.pt.
- Keep the current query -> workspace retrieval setup.
- Replace the current adjacency fusion with a GTN-style adjacency generator:
    candidate graphs -> 1x1 channel selection -> sequential adjacency composition -> GCN.
- Support negative sheets inside each workspace.
- Keep training robust even if query.json only provides positive sheet_ids.

Expected data/query.json item formats (all supported):
1) {"query": str, "sheet_ids": [positive ids]}
2) {"query": str, "sheet_ids": [...], "negative_sheet_ids": [...]}
3) {"query": str, "workspace_sheet_ids": [...], "labels": [0/1,...]}

The dataset will fall back to negative sampling from all sheets if explicit negatives are absent.
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from math import exp
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> torch.device:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_distributed() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def all_gather_tensor(t: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return t
    gathered = [torch.zeros_like(t) for _ in range(get_world_size())]
    dist.all_gather(gathered, t)
    return torch.cat(gathered, dim=0)


def reduce_mean(value: float, device: torch.device) -> float:
    if not is_dist():
        return float(value)
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= get_world_size()
    return t.item()


@dataclass
class EvalMetrics:
    loss: float
    retrieval_acc: float
    mean_pos_cos: float
    mean_neg_cos: float
    mean_set_cos: float
    node_auc_proxy: float
    loss_infonce: float = 0.0
    loss_align: float = 0.0
    loss_node: float = 0.0


class SheetTextSerializer:
    def __init__(self, sheet_feature_map: Dict[str, Dict], max_header_texts: int = 12, include_shape_feature: bool = True) -> None:
        self.sheet_feature_map = sheet_feature_map
        self.max_header_texts = max_header_texts
        self.include_shape_feature = include_shape_feature

    def to_text(self, sheet_id: str) -> str:
        feat = self.sheet_feature_map.get(str(sheet_id))
        if not feat:
            return f"sheet_id: {sheet_id}"
        segments: List[str] = []
        name = feat.get("name")
        if name:
            segments.append(f"name: {name}")
        if self.include_shape_feature:
            nr = feat.get("num_rows", "?")
            nc = feat.get("num_cols", "?")
            segments.append(f"shape: {nr} x {nc}")
        columns = feat.get("columns", [])
        column_names: List[str] = []
        for col in columns[: self.max_header_texts]:
            if isinstance(col, dict):
                col_name = str(col.get("name", "")).strip()
            else:
                col_name = str(col).strip()
            if col_name:
                column_names.append(col_name)
        if column_names:
            segments.append("columns: " + " | ".join(column_names))
        return " ; ".join(segments) if segments else f"sheet_id: {sheet_id}"

    def header_set(self, sheet_id: str) -> set:
        feat = self.sheet_feature_map.get(str(sheet_id), {})
        columns = feat.get("columns", [])
        out = set()
        for col in columns[: self.max_header_texts]:
            if isinstance(col, dict):
                txt = str(col.get("name", "")).strip().lower()
            else:
                txt = str(col).strip().lower()
            if txt:
                out.add(txt)
        return out

    def shape(self, sheet_id: str) -> Tuple[float, float]:
        feat = self.sheet_feature_map.get(str(sheet_id), {})
        return float(feat.get("num_rows", 0.0) or 0.0), float(feat.get("num_cols", 0.0) or 0.0)


class NegativeAwareQueryDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer,
        max_length: int = 256,
        max_query_length: int = 64,
        max_workspace_size: int = 10,
        features_file: Optional[str] = None,
        max_header_texts: int = 12,
        include_shape_feature: bool = True,
        eval_ratio: float = 0.1,
        sample_seed: int = 42,
        neg_ratio: float = 0.5,
        min_negatives: int = 1,
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_query_length = max_query_length
        self.max_workspace_size = max_workspace_size
        self.features_file = features_file
        self.eval_ratio = eval_ratio
        self.sample_seed = sample_seed
        self.neg_ratio = neg_ratio
        self.min_negatives = min_negatives
        self.serializer = SheetTextSerializer(self._load_sheet_feature_map(), max_header_texts=max_header_texts, include_shape_feature=include_shape_feature)
        self.all_sheet_ids = sorted(self.serializer.sheet_feature_map.keys())
        self.data = self._load_data()

    def _dataset_file(self) -> str:
        return os.path.join(self.data_dir, "query.json")

    def _infer_features_file(self) -> Optional[str]:
        if self.features_file:
            return self.features_file
        for path in [os.path.join(self.data_dir, "sheets.json"), os.path.join(self.data_dir, "sheet_features.json")]:
            if os.path.exists(path):
                return path
        return None

    def _load_sheet_feature_map(self) -> Dict[str, Dict]:
        path = self._infer_features_file()
        if not path:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out: Dict[str, Dict] = {}
        if isinstance(raw, dict):
            for key, item in raw.items():
                if isinstance(item, dict):
                    sid = item.get("sheet_id", key)
                    out[str(sid)] = item
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and item.get("sheet_id") is not None:
                    out[str(item["sheet_id"])] = item
        return out

    def _encode_text(self, text: str, max_length: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return out

    def _split_indices(self, n: int) -> List[int]:
        rng = random.Random(self.sample_seed)
        indices = list(range(n))
        rng.shuffle(indices)
        eval_size = max(1, int(n * self.eval_ratio))
        if eval_size >= n:
            eval_size = 1
        train_size = n - eval_size
        if self.split == "nway_train":
            return indices[:train_size]
        if self.split == "nway_eval":
            return indices[train_size:]
        raise ValueError(f"Unsupported split: {self.split}")

    def _normalize_item(self, item: Dict) -> Optional[Dict]:
        query = str(item.get("query", "")).strip()
        if not query:
            return None

        if "workspace_sheet_ids" in item and "labels" in item:
            ws = [str(x) for x in item.get("workspace_sheet_ids", []) if str(x) in self.serializer.sheet_feature_map]
            labels = [int(x) for x in item.get("labels", [])][: len(ws)]
            if ws and labels and len(ws) == len(labels):
                positive_sheet_ids = [sid for sid, y in zip(ws, labels) if y == 1]
                negative_sheet_ids = [sid for sid, y in zip(ws, labels) if y == 0]
                if positive_sheet_ids:
                    return {
                        "query": query,
                        "positive_sheet_ids": positive_sheet_ids,
                        "negative_sheet_ids": negative_sheet_ids,
                    }

        positive_sheet_ids = [str(x) for x in item.get("sheet_ids", []) if str(x) in self.serializer.sheet_feature_map]
        negative_raw = item.get("sheet_ids_negative", item.get("negative_sheet_ids", []))
        negative_sheet_ids = [str(x) for x in negative_raw if str(x) in self.serializer.sheet_feature_map]
        if positive_sheet_ids:
            return {
                "query": query,
                "positive_sheet_ids": positive_sheet_ids,
                "negative_sheet_ids": negative_sheet_ids,
            }
        return None

    def _load_data(self) -> List[Dict]:
        path = self._dataset_file()
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cleaned = []
        for item in raw:
            obj = self._normalize_item(item)
            if obj:
                cleaned.append(obj)
        if not cleaned:
            raise ValueError("No valid samples found in query.json")
        chosen = self._split_indices(len(cleaned))
        return [cleaned[i] for i in chosen]

    def __len__(self) -> int:
        return len(self.data)

    def _sample_workspace(self, item: Dict, idx: int) -> Tuple[List[str], List[int]]:
        positives = list(dict.fromkeys(item["positive_sheet_ids"]))
        negatives = list(dict.fromkeys(item.get("negative_sheet_ids", [])))
        rng = random.Random(self.sample_seed * 100003 + idx)

        max_neg = min(self.max_workspace_size - 1, max(self.min_negatives, int(round(self.max_workspace_size * self.neg_ratio))))
        max_neg = min(max_neg, self.max_workspace_size - 1)
        max_pos = max(1, self.max_workspace_size - max_neg)
        sampled_pos = positives if len(positives) <= max_pos else rng.sample(positives, max_pos)

        remaining_slots = self.max_workspace_size - len(sampled_pos)
        neg_candidates = [sid for sid in negatives if sid not in sampled_pos]
        if len(neg_candidates) < remaining_slots:
            positive_set = set(positives)
            neg_pool = [sid for sid in self.all_sheet_ids if sid not in positive_set and sid not in neg_candidates]
            if len(neg_pool) > (remaining_slots - len(neg_candidates)):
                neg_candidates.extend(rng.sample(neg_pool, remaining_slots - len(neg_candidates)))
            else:
                neg_candidates.extend(neg_pool)
        sampled_neg = neg_candidates[:remaining_slots]

        workspace = sampled_pos + sampled_neg
        labels = [1] * len(sampled_pos) + [0] * len(sampled_neg)
        pairs = list(zip(workspace, labels))
        rng.shuffle(pairs)
        workspace = [x[0] for x in pairs]
        labels = [x[1] for x in pairs]
        return workspace, labels

    def _build_schema_prior(self, sheet_ids: List[str], node_mask: torch.Tensor) -> torch.Tensor:
        n = len(sheet_ids)
        adj = torch.zeros(self.max_workspace_size, self.max_workspace_size, dtype=torch.float)
        headers = [self.serializer.header_set(sid) for sid in sheet_ids]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                hi, hj = headers[i], headers[j]
                score = 0.0 if (not hi and not hj) else len(hi & hj) / max(1, len(hi | hj))
                adj[i, j] = score
        valid_pair = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
        return adj * valid_pair

    def _build_shape_prior(self, sheet_ids: List[str], node_mask: torch.Tensor) -> torch.Tensor:
        n = len(sheet_ids)
        adj = torch.zeros(self.max_workspace_size, self.max_workspace_size, dtype=torch.float)
        shapes = [self.serializer.shape(sid) for sid in sheet_ids]
        for i in range(n):
            r1, c1 = shapes[i]
            for j in range(n):
                if i == j:
                    continue
                r2, c2 = shapes[j]
                row_sim = 1.0 / (1.0 + abs(r1 - r2))
                col_sim = 1.0 / (1.0 + abs(c1 - c2))
                adj[i, j] = 0.5 * (row_sim + col_sim)
        valid_pair = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
        return adj * valid_pair

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        query_text = item["query"]
        workspace_ids, labels = self._sample_workspace(item, idx)

        q = self._encode_text(query_text, self.max_query_length)
        ws_input_ids, ws_attention_mask, ws_token_type_ids = [], [], []
        node_mask_list: List[float] = []
        label_list: List[float] = []
        padded_sheet_ids: List[str] = []

        for i in range(self.max_workspace_size):
            if i < len(workspace_ids):
                sid = workspace_ids[i]
                enc = self._encode_text(self.serializer.to_text(sid), self.max_length)
                ws_input_ids.append(enc["input_ids"])
                ws_attention_mask.append(enc["attention_mask"])
                ws_token_type_ids.append(enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])))
                node_mask_list.append(1.0)
                label_list.append(float(labels[i]))
                padded_sheet_ids.append(sid)
            else:
                ws_input_ids.append(torch.zeros(self.max_length, dtype=torch.long))
                ws_attention_mask.append(torch.zeros(self.max_length, dtype=torch.long))
                ws_token_type_ids.append(torch.zeros(self.max_length, dtype=torch.long))
                node_mask_list.append(0.0)
                label_list.append(0.0)
                padded_sheet_ids.append("")

        node_mask = torch.tensor(node_mask_list, dtype=torch.float)
        labels_t = torch.tensor(label_list, dtype=torch.float)
        schema_prior = self._build_schema_prior(workspace_ids, node_mask)
        shape_prior = self._build_shape_prior(workspace_ids, node_mask)

        return {
            "query_input_ids": q["input_ids"],
            "query_attention_mask": q["attention_mask"],
            "query_token_type_ids": q.get("token_type_ids", torch.zeros_like(q["input_ids"])),
            "workspace_input_ids": torch.stack(ws_input_ids, dim=0),
            "workspace_attention_mask": torch.stack(ws_attention_mask, dim=0),
            "workspace_token_type_ids": torch.stack(ws_token_type_ids, dim=0),
            "node_mask": node_mask,
            "labels": labels_t,
            "schema_prior": schema_prior,
            "shape_prior": shape_prior,
            "sheet_ids": padded_sheet_ids,
        }


class TextBiEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        local_files_only: bool = True,
        embedding_strategy: str = "cls",
        use_layer_mix: bool = False,
        use_extra_position_embedding: bool = False,
        position_embedding_scale: float = 1.0,
        max_length: int = 256,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_strategy = embedding_strategy
        self.use_layer_mix = use_layer_mix
        self.use_extra_position_embedding = use_extra_position_embedding
        self.position_embedding_scale = position_embedding_scale
        self.normalize_embeddings = normalize_embeddings
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden_size = self.backbone.config.hidden_size
        self.hidden_size = hidden_size
        self.embedding_dim = self._get_embedding_dim(hidden_size)
        if self.use_extra_position_embedding:
            self.extra_position_embedding = nn.Embedding(max_length, hidden_size)
        if self.use_layer_mix:
            num_layers = self.backbone.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def _get_embedding_dim(self, hidden_size: int) -> int:
        if self.embedding_strategy in {"cls", "mean", "max"}:
            return hidden_size
        if self.embedding_strategy in {"cls_mean_concat", "mean_max_concat"}:
            return hidden_size * 2
        if self.embedding_strategy == "cls_mean_max_concat":
            return hidden_size * 3
        raise ValueError(f"Unsupported embedding_strategy: {self.embedding_strategy}")

    def _masked_mean_pool(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (sequence_output * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _masked_max_pool(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).bool()
        masked = sequence_output.masked_fill(~mask, -1e9)
        return masked.max(dim=1).values

    def _build_embedding(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls_emb = sequence_output[:, 0, :]
        mean_emb = self._masked_mean_pool(sequence_output, attention_mask)
        max_emb = self._masked_max_pool(sequence_output, attention_mask)
        if self.embedding_strategy == "cls":
            emb = cls_emb
        elif self.embedding_strategy == "mean":
            emb = mean_emb
        elif self.embedding_strategy == "max":
            emb = max_emb
        elif self.embedding_strategy == "cls_mean_concat":
            emb = torch.cat([cls_emb, mean_emb], dim=-1)
        elif self.embedding_strategy == "mean_max_concat":
            emb = torch.cat([mean_emb, max_emb], dim=-1)
        elif self.embedding_strategy == "cls_mean_max_concat":
            emb = torch.cat([cls_emb, mean_emb, max_emb], dim=-1)
        else:
            raise ValueError(f"Unsupported embedding_strategy: {self.embedding_strategy}")
        return F.normalize(emb, dim=-1) if self.normalize_embeddings else emb

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=self.use_layer_mix,
            return_dict=True,
        )
        if self.use_layer_mix:
            hidden_states = torch.stack(outputs.hidden_states, dim=0)
            layer_probs = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            sequence_output = (hidden_states * layer_probs).sum(dim=0)
        else:
            sequence_output = outputs.last_hidden_state
        if self.use_extra_position_embedding:
            seq_len = sequence_output.size(1)
            position_ids = torch.arange(seq_len, device=sequence_output.device).unsqueeze(0)
            pos_emb = self.extra_position_embedding(position_ids)
            sequence_output = sequence_output + self.position_embedding_scale * pos_emb
        return self._build_embedding(sequence_output, attention_mask)


class DenseGCNLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        out = torch.bmm(adj, h)
        out = self.out_proj(out)
        out = self.norm(out + x)
        return out * node_mask.unsqueeze(-1)


class GTNAdjacencyGenerator(nn.Module):
    def __init__(self, num_input_channels: int, num_output_channels: int = 4, num_layers: int = 2, add_identity_channel: bool = True, eps: float = 1e-9) -> None:
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.num_layers = num_layers
        self.add_identity_channel = add_identity_channel
        self.eps = eps
        # q1,q2 for first layer; then one selector per later layer.
        self.first_q1_logits = nn.Parameter(torch.zeros(num_output_channels, num_input_channels))
        self.first_q2_logits = nn.Parameter(torch.zeros(num_output_channels, num_input_channels))
        if num_layers > 1:
            self.later_logits = nn.ParameterList([
                nn.Parameter(torch.zeros(num_output_channels, num_input_channels)) for _ in range(num_layers - 1)
            ])
        else:
            self.later_logits = nn.ParameterList([])

    def _row_normalize(self, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        # adj: [B,C,N,N]
        bsz, chans, n, _ = adj.shape
        valid_pair = node_mask.unsqueeze(1).unsqueeze(2) * node_mask.unsqueeze(1).unsqueeze(3)
        adj = F.relu(adj) * valid_pair
        eye = torch.eye(n, device=adj.device).view(1, 1, n, n)
        adj = adj + eye * node_mask.unsqueeze(1).unsqueeze(-1)
        row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        return adj / row_sum

    def _mix(self, channels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        # channels: [B,K,N,N], logits: [C,K] -> [B,C,N,N]
        weights = torch.softmax(logits, dim=-1)
        mixed = torch.einsum("ck,bknm->bcnm", weights, channels)
        return mixed

    def forward(self, channels: torch.Tensor, node_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        q1 = self._mix(channels, self.first_q1_logits)
        q2 = self._mix(channels, self.first_q2_logits)
        cur = torch.matmul(q1, q2)
        cur = self._row_normalize(cur, node_mask)
        all_weights = {
            "first_q1_weights": torch.softmax(self.first_q1_logits, dim=-1).detach(),
            "first_q2_weights": torch.softmax(self.first_q2_logits, dim=-1).detach(),
        }
        for i, logits in enumerate(self.later_logits, start=2):
            q = self._mix(channels, logits)
            cur = torch.matmul(cur, q)
            cur = self._row_normalize(cur, node_mask)
            all_weights[f"layer_{i}_weights"] = torch.softmax(logits, dim=-1).detach()
        return cur, all_weights


class Stage2GTNModelV2(nn.Module):
    def __init__(
        self,
        model_name: str,
        local_files_only: bool = True,
        embedding_strategy: str = "cls",
        use_layer_mix: bool = False,
        use_extra_position_embedding: bool = False,
        position_embedding_scale: float = 1.0,
        max_length: int = 256,
        normalize_embeddings: bool = True,
        graph_dropout: float = 0.1,
        num_gcn_layers: int = 1,
        freeze_backbone: bool = True,
        gtn_channels: int = 4,
        gtn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = TextBiEncoder(
            model_name=model_name,
            local_files_only=local_files_only,
            embedding_strategy=embedding_strategy,
            use_layer_mix=use_layer_mix,
            use_extra_position_embedding=use_extra_position_embedding,
            position_embedding_scale=position_embedding_scale,
            max_length=max_length,
            normalize_embeddings=normalize_embeddings,
        )
        dim = self.encoder.embedding_dim
        self.query_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.node_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.adj_generator = GTNAdjacencyGenerator(num_input_channels=5, num_output_channels=gtn_channels, num_layers=gtn_layers)
        self.gcn_layers = nn.ModuleList([DenseGCNLayer(dim, dropout=graph_dropout) for _ in range(num_gcn_layers)])
        self.channel_merge = nn.Sequential(nn.Linear(dim * gtn_channels, dim), nn.GELU(), nn.LayerNorm(dim))
        self.set_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        self.node_classifier = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        if freeze_backbone:
            for p in self.encoder.backbone.parameters():
                p.requires_grad = False

    def load_stage1_checkpoint(self, ckpt_path: str) -> None:
        if not ckpt_path or not os.path.exists(ckpt_path):
            return
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state.get("state_dict", state)
        encoder_state = self.encoder.state_dict()
        mapped = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                new_k = f"backbone.{k[len('backbone.'):] }"
                new_k = new_k.replace(" ", "")
            else:
                continue
            if new_k in encoder_state and encoder_state[new_k].shape == v.shape:
                mapped[new_k] = v
        missing, unexpected = self.encoder.load_state_dict(mapped, strict=False)
        if is_main_process():
            print(f"Loaded Stage1 backbone weights from {ckpt_path}; matched={len(mapped)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    def _build_channels(self, query_emb: torch.Tensor, node_embs: torch.Tensor, node_mask: torch.Tensor, schema_prior: torch.Tensor, shape_prior: torch.Tensor) -> torch.Tensor:
        norm_nodes = F.normalize(node_embs, dim=-1)
        q = F.normalize(query_emb, dim=-1)
        sem = torch.bmm(norm_nodes, norm_nodes.transpose(1, 2))
        sem = (sem + 1.0) * 0.5
        q_sim = (norm_nodes * q.unsqueeze(1)).sum(dim=-1)
        q_graph = torch.einsum("bi,bj->bij", q_sim, q_sim)
        q_graph = (q_graph + 1.0) * 0.5
        n = node_embs.size(1)
        identity = torch.eye(n, device=node_embs.device).unsqueeze(0).expand(node_embs.size(0), -1, -1)
        channels = torch.stack([sem, q_graph, schema_prior, shape_prior, identity], dim=1)
        valid_pair = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        channels = channels * valid_pair.unsqueeze(1)
        return channels

    def _pool_set(self, query_emb: torch.Tensor, node_embs: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        q_expand = query_emb.unsqueeze(1).expand_as(node_embs)
        gate_in = torch.cat([node_embs, q_expand], dim=-1)
        gate = self.set_gate(gate_in).squeeze(-1)
        gate = gate.masked_fill(node_mask <= 0, -1e9)
        attn = torch.softmax(gate, dim=-1)
        attn = attn * node_mask
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        set_emb = torch.bmm(attn.unsqueeze(1), node_embs).squeeze(1)
        return F.normalize(set_emb, dim=-1)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        workspace_input_ids: torch.Tensor,
        workspace_attention_mask: torch.Tensor,
        node_mask: torch.Tensor,
        schema_prior: torch.Tensor,
        shape_prior: torch.Tensor,
        query_token_type_ids: Optional[torch.Tensor] = None,
        workspace_token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, num_nodes, seq_len = workspace_input_ids.shape
        query_emb = self.encoder.encode(query_input_ids, query_attention_mask, query_token_type_ids)
        flat_ids = workspace_input_ids.view(bsz * num_nodes, seq_len)
        flat_mask = workspace_attention_mask.view(bsz * num_nodes, seq_len)
        flat_tti = workspace_token_type_ids.view(bsz * num_nodes, seq_len) if workspace_token_type_ids is not None else None
        node_embs = self.encoder.encode(flat_ids, flat_mask, flat_tti).view(bsz, num_nodes, -1)

        query_emb = self.query_proj(query_emb)
        node_embs = self.node_proj(node_embs)
        if self.encoder.normalize_embeddings:
            query_emb = F.normalize(query_emb, dim=-1)
            node_embs = F.normalize(node_embs, dim=-1)
        node_embs = node_embs * node_mask.unsqueeze(-1)

        channels = self._build_channels(query_emb, node_embs, node_mask, schema_prior, shape_prior)
        composed_adj, aux = self.adj_generator(channels, node_mask)

        channel_outputs = []
        for c in range(composed_adj.size(1)):
            h = node_embs
            adj_c = composed_adj[:, c, :, :]
            for gcn in self.gcn_layers:
                h = gcn(h, adj_c, node_mask)
            channel_outputs.append(h)
        h_cat = torch.cat(channel_outputs, dim=-1)
        h = self.channel_merge(h_cat) * node_mask.unsqueeze(-1)

        set_emb = self._pool_set(query_emb, h, node_mask)
        query_emb = F.normalize(query_emb, dim=-1)
        q_expand = query_emb.unsqueeze(1).expand_as(h)
        node_logits = self.node_classifier(torch.cat([h, q_expand], dim=-1)).squeeze(-1)
        node_logits = node_logits.masked_fill(node_mask <= 0, -1e9)
        node_probs = torch.sigmoid(torch.where(node_mask > 0, node_logits, torch.zeros_like(node_logits)))

        node_cos = (F.normalize(h, dim=-1) * query_emb.unsqueeze(1)).sum(dim=-1) * node_mask
        set_cos = (query_emb * set_emb).sum(dim=-1)
        return {
            "query_emb": query_emb,
            "node_embs": h,
            "set_emb": set_emb,
            "node_cos": node_cos,
            "set_cos": set_cos,
            "node_logits": node_logits,
            "node_probs": node_probs,
            "composed_adj": composed_adj,
            "aux": aux,
        }


class Stage2LossV2(nn.Module):
    def __init__(self, tau: float = 0.07, lambda_align: float = 0.10, lambda_node: float = 0.20, pos_weight: float = 2.0) -> None:
        super().__init__()
        self.tau = tau
        self.lambda_align = lambda_align
        self.lambda_node = lambda_node
        self.pos_weight = pos_weight

    def forward(self, outputs: Dict[str, torch.Tensor], node_mask: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        query_emb = outputs["query_emb"]
        set_emb = outputs["set_emb"]
        node_cos = outputs["node_cos"]
        node_logits = outputs["node_logits"]

        all_query = all_gather_tensor(query_emb)
        all_set = all_gather_tensor(set_emb)
        logits = torch.matmul(query_emb, all_set.transpose(0, 1)) / self.tau
        start = get_rank() * query_emb.size(0)
        targets = torch.arange(query_emb.size(0), device=query_emb.device) + start
        loss_infonce = F.cross_entropy(logits, targets)

        pos_mask = labels * node_mask
        neg_mask = (1.0 - labels) * node_mask
        pos_denom = pos_mask.sum(dim=-1).clamp(min=1.0)
        neg_denom = neg_mask.sum(dim=-1).clamp(min=1.0)
        mean_pos_cos = (node_cos * pos_mask).sum(dim=-1) / pos_denom
        mean_neg_cos = (node_cos * neg_mask).sum(dim=-1) / neg_denom
        loss_align = (1.0 - mean_pos_cos + F.relu(mean_neg_cos)).mean()

        valid = node_mask > 0
        node_bce = F.binary_cross_entropy_with_logits(
            node_logits[valid],
            labels[valid],
            pos_weight=torch.tensor(self.pos_weight, device=node_logits.device),
        ) if valid.any() else torch.tensor(0.0, device=node_logits.device)

        loss = loss_infonce + self.lambda_align * loss_align + self.lambda_node * node_bce

        pred = logits.argmax(dim=-1)
        retrieval_acc = (pred == targets).float().mean().item()
        probs = torch.sigmoid(torch.where(valid, node_logits, torch.zeros_like(node_logits)))
        node_auc_proxy = (((probs > 0.5).float() == labels) * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

        logs = {
            "loss": float(loss.item()),
            "loss_infonce": float(loss_infonce.item()),
            "loss_align": float(loss_align.item()),
            "loss_node": float(node_bce.item()),
            "retrieval_acc": float(retrieval_acc),
            "mean_pos_cos": float(mean_pos_cos.mean().item()),
            "mean_neg_cos": float(mean_neg_cos.mean().item()),
            "mean_set_cos": float(outputs["set_cos"].mean().item()),
            "node_auc_proxy": float(node_auc_proxy.item()),
        }
        return loss, logs


class Stage2TrainerV2:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = setup_distributed()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        seed_everything(args.seed + get_rank())

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=not args.allow_download)
        self.train_dataset = NegativeAwareQueryDataset(
            data_dir=args.data_dir,
            split="nway_train",
            tokenizer=self.tokenizer,
            max_length=args.max_length,
            max_query_length=args.max_query_length,
            max_workspace_size=args.max_workspace_size,
            features_file=args.features_file,
            max_header_texts=args.max_header_texts,
            include_shape_feature=not args.disable_shape_feature,
            eval_ratio=args.eval_ratio,
            sample_seed=args.seed,
            neg_ratio=args.negative_ratio,
            min_negatives=args.min_negatives,
        )
        self.eval_dataset = NegativeAwareQueryDataset(
            data_dir=args.data_dir,
            split="nway_eval",
            tokenizer=self.tokenizer,
            max_length=args.max_length,
            max_query_length=args.max_query_length,
            max_workspace_size=args.max_workspace_size,
            features_file=args.features_file,
            max_header_texts=args.max_header_texts,
            include_shape_feature=not args.disable_shape_feature,
            eval_ratio=args.eval_ratio,
            sample_seed=args.seed,
            neg_ratio=args.negative_ratio,
            min_negatives=args.min_negatives,
        )
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True) if is_dist() else None
        self.eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False) if is_dist() else None
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=(self.train_sampler is None), sampler=self.train_sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=args.batch_size, shuffle=False, sampler=self.eval_sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

        self.model = Stage2GTNModelV2(
            model_name=args.model_name,
            local_files_only=not args.allow_download,
            embedding_strategy=args.embedding_strategy,
            use_layer_mix=args.use_layer_mix,
            use_extra_position_embedding=args.use_extra_position_embedding,
            position_embedding_scale=args.position_embedding_scale,
            max_length=max(args.max_length, args.max_query_length),
            normalize_embeddings=args.normalize_embeddings,
            graph_dropout=args.graph_dropout,
            num_gcn_layers=args.num_gcn_layers,
            freeze_backbone=args.freeze_backbone,
            gtn_channels=args.gtn_channels,
            gtn_layers=args.gtn_layers,
        )
        self.model.load_stage1_checkpoint(args.stage1_checkpoint)
        self.model.to(self.device)
        self.raw_model = self.model
        if is_dist():
            self.model = DDP(self.model, device_ids=[self.local_rank] if self.device.type == "cuda" else None, find_unused_parameters=False)

        self.loss_fn = Stage2LossV2(tau=args.tau, lambda_align=args.lambda_align, lambda_node=args.lambda_node, pos_weight=args.pos_weight)
        self.optimizer = AdamW((p for p in self.raw_model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
        total_steps = max(1, len(self.train_loader) * args.num_epochs)
        warmup_steps = int(total_steps * args.warmup_ratio) if args.warmup_steps <= 0 else args.warmup_steps
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        self.ckpt_dir = os.path.join(args.output_dir, args.run_name)
        self.tb_writer = None
        self.global_step = 0
        if is_main_process():
            os.makedirs(self.ckpt_dir, exist_ok=True)
            if args.use_tensorboard:
                tb_dir = os.path.join(args.tensorboard_logdir, args.run_name)
                os.makedirs(tb_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                self.tb_writer.add_text("run/model_name", args.model_name)
                self.tb_writer.add_text("run/embedding_strategy", args.embedding_strategy)
                self.tb_writer.add_text("run/data_dir", args.data_dir)
                self.tb_writer.add_text("run/tensorboard_dir", tb_dir)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        return out

    @staticmethod
    def _safe_entropy_from_prob(prob: float) -> float:
        p = min(max(float(prob), 1e-8), 1.0 - 1e-8)
        return -(p * torch.log(torch.tensor(p)) + (1.0 - p) * torch.log(torch.tensor(1.0 - p))).item()

    def _summarize_entropy(self, retrieval_acc: float, mean_pos_cos: float, mean_neg_cos: float) -> float:
        margin = max(-1.0, min(1.0, (mean_pos_cos - mean_neg_cos) / 2.0))
        pseudo_conf = 0.5 * (retrieval_acc + (margin + 1.0) / 2.0)
        return self._safe_entropy_from_prob(pseudo_conf)

    def save_checkpoint(self, name: str, epoch: int, best_metric: float) -> None:
        if not is_main_process():
            return
        path = os.path.join(self.ckpt_dir, name)
        torch.save({
            "state_dict": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "args": vars(self.args),
        }, path)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        total = {"loss": 0.0, "loss_infonce": 0.0, "loss_align": 0.0, "loss_node": 0.0, "retrieval_acc": 0.0, "mean_pos_cos": 0.0, "mean_neg_cos": 0.0, "mean_set_cos": 0.0, "node_auc_proxy": 0.0}
        steps = 0
        wall_start = time.time()
        total_samples = 0
        for batch in self.train_loader:
            step_start = time.time()
            batch = self._move_batch(batch)
            outputs = self.model(
                query_input_ids=batch["query_input_ids"],
                query_attention_mask=batch["query_attention_mask"],
                workspace_input_ids=batch["workspace_input_ids"],
                workspace_attention_mask=batch["workspace_attention_mask"],
                node_mask=batch["node_mask"],
                schema_prior=batch["schema_prior"],
                shape_prior=batch["shape_prior"],
                query_token_type_ids=batch["query_token_type_ids"],
                workspace_token_type_ids=batch["workspace_token_type_ids"],
            )
            loss, logs = self.loss_fn(outputs, batch["node_mask"], batch["labels"])
            self.optimizer.zero_grad()
            loss.backward()
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            batch_size = int(batch["query_input_ids"].size(0))
            total_samples += batch_size
            for k in total:
                total[k] += logs[k]
            steps += 1
            if self.tb_writer is not None and is_main_process():
                step_time = max(1e-9, time.time() - step_start)
                samples_per_sec = batch_size / step_time
                self.tb_writer.add_scalar("train/loss_step", logs["loss"], self.global_step)
                self.tb_writer.add_scalar("train/loss_infonce_step", logs["loss_infonce"], self.global_step)
                self.tb_writer.add_scalar("train/loss_align_step", logs["loss_align"], self.global_step)
                self.tb_writer.add_scalar("train/loss_node_step", logs["loss_node"], self.global_step)
                self.tb_writer.add_scalar("train/retrieval_acc_step", logs["retrieval_acc"], self.global_step)
                self.tb_writer.add_scalar("train/step_time_sec", step_time, self.global_step)
                self.tb_writer.add_scalar("train/samples_per_sec", samples_per_sec, self.global_step)
                self.tb_writer.add_scalar("optimizer/lr_step", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.global_step += 1
        steps = max(1, steps)
        for k in total:
            total[k] = reduce_mean(total[k] / steps, self.device)
        epoch_time = time.time() - wall_start
        total_samples = reduce_mean(float(total_samples), self.device)
        total["epoch_time_sec"] = epoch_time
        total["avg_steps_per_sec"] = steps / max(epoch_time, 1e-9)
        total["avg_samples_per_sec"] = total_samples / max(epoch_time, 1e-9)
        total["entropy"] = self._summarize_entropy(total["retrieval_acc"], total["mean_pos_cos"], total["mean_neg_cos"])
        total["lr"] = self.optimizer.param_groups[0]["lr"]
        return total

    def evaluate(self) -> EvalMetrics:
        self.model.eval()
        total = {"loss": 0.0, "retrieval_acc": 0.0, "mean_pos_cos": 0.0, "mean_neg_cos": 0.0, "mean_set_cos": 0.0, "node_auc_proxy": 0.0, "loss_infonce": 0.0, "loss_align": 0.0, "loss_node": 0.0}
        steps = 0
        with torch.no_grad():
            for batch in self.eval_loader:
                batch = self._move_batch(batch)
                outputs = self.model(
                    query_input_ids=batch["query_input_ids"],
                    query_attention_mask=batch["query_attention_mask"],
                    workspace_input_ids=batch["workspace_input_ids"],
                    workspace_attention_mask=batch["workspace_attention_mask"],
                    node_mask=batch["node_mask"],
                    schema_prior=batch["schema_prior"],
                    shape_prior=batch["shape_prior"],
                    query_token_type_ids=batch["query_token_type_ids"],
                    workspace_token_type_ids=batch["workspace_token_type_ids"],
                )
                _, logs = self.loss_fn(outputs, batch["node_mask"], batch["labels"])
                total["loss"] += logs["loss"]
                total["retrieval_acc"] += logs["retrieval_acc"]
                total["mean_pos_cos"] += logs["mean_pos_cos"]
                total["mean_neg_cos"] += logs["mean_neg_cos"]
                total["mean_set_cos"] += logs["mean_set_cos"]
                total["node_auc_proxy"] += logs["node_auc_proxy"]
                total["loss_infonce"] += logs["loss_infonce"]
                total["loss_align"] += logs["loss_align"]
                total["loss_node"] += logs["loss_node"]
                steps += 1
        steps = max(1, steps)
        for k in total:
            total[k] = reduce_mean(total[k] / steps, self.device)
        return EvalMetrics(**total)

    def train(self) -> None:
        best_metric = -1.0
        if is_main_process():
            print(f"train_size={len(self.train_dataset)} eval_size={len(self.eval_dataset)}", flush=True)
        for epoch in range(1, self.args.num_epochs + 1):
            t0 = time.time()
            train_logs = self.train_epoch(epoch)
            eval_metrics = self.evaluate()
            epoch_time = time.time() - t0
            eval_entropy = self._summarize_entropy(eval_metrics.retrieval_acc, eval_metrics.mean_pos_cos, eval_metrics.mean_neg_cos)
            if is_main_process():
                print(
                    f"[Epoch {epoch}/{self.args.num_epochs}] train_loss={train_logs['loss']:.4f} train_retrieval_acc={train_logs['retrieval_acc']:.4f} "
                    f"eval_loss={eval_metrics.loss:.4f} eval_retrieval_acc={eval_metrics.retrieval_acc:.4f} eval_mean_pos_cos={eval_metrics.mean_pos_cos:.4f} "
                    f"eval_mean_neg_cos={eval_metrics.mean_neg_cos:.4f} eval_node_auc_proxy={eval_metrics.node_auc_proxy:.4f} time={epoch_time:.1f}s",
                    flush=True,
                )
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("loss/train", train_logs["loss"], epoch)
                    self.tb_writer.add_scalar("loss/eval", eval_metrics.loss, epoch)
                    self.tb_writer.add_scalar("loss_infonce/train", train_logs["loss_infonce"], epoch)
                    self.tb_writer.add_scalar("loss_infonce/eval", eval_metrics.loss_infonce, epoch)
                    self.tb_writer.add_scalar("loss_align/train", train_logs["loss_align"], epoch)
                    self.tb_writer.add_scalar("loss_align/eval", eval_metrics.loss_align, epoch)
                    self.tb_writer.add_scalar("loss_node/train", train_logs["loss_node"], epoch)
                    self.tb_writer.add_scalar("loss_node/eval", eval_metrics.loss_node, epoch)
                    self.tb_writer.add_scalar("accuracy/train", train_logs["retrieval_acc"], epoch)
                    self.tb_writer.add_scalar("accuracy/eval", eval_metrics.retrieval_acc, epoch)
                    self.tb_writer.add_scalar("retrieval_acc/train", train_logs["retrieval_acc"], epoch)
                    self.tb_writer.add_scalar("retrieval_acc/eval", eval_metrics.retrieval_acc, epoch)
                    self.tb_writer.add_scalar("entropy/train", train_logs["entropy"], epoch)
                    self.tb_writer.add_scalar("entropy/eval", eval_entropy, epoch)
                    self.tb_writer.add_scalar("cosine/mean_pos/train", train_logs["mean_pos_cos"], epoch)
                    self.tb_writer.add_scalar("cosine/mean_pos/eval", eval_metrics.mean_pos_cos, epoch)
                    self.tb_writer.add_scalar("cosine/mean_neg/train", train_logs["mean_neg_cos"], epoch)
                    self.tb_writer.add_scalar("cosine/mean_neg/eval", eval_metrics.mean_neg_cos, epoch)
                    self.tb_writer.add_scalar("cosine/mean_set/train", train_logs["mean_set_cos"], epoch)
                    self.tb_writer.add_scalar("cosine/mean_set/eval", eval_metrics.mean_set_cos, epoch)
                    self.tb_writer.add_scalar("node_auc_proxy/train", train_logs["node_auc_proxy"], epoch)
                    self.tb_writer.add_scalar("node_auc_proxy/eval", eval_metrics.node_auc_proxy, epoch)
                    self.tb_writer.add_scalar("timing/epoch_time_sec", train_logs["epoch_time_sec"], epoch)
                    self.tb_writer.add_scalar("timing/avg_steps_per_sec", train_logs["avg_steps_per_sec"], epoch)
                    self.tb_writer.add_scalar("timing/avg_samples_per_sec", train_logs["avg_samples_per_sec"], epoch)
                    self.tb_writer.add_scalar("optimizer/lr", train_logs["lr"], epoch)
            metric = eval_metrics.retrieval_acc
            if metric > best_metric:
                best_metric = metric
                self.save_checkpoint("best.pt", epoch, best_metric)
                if self.tb_writer is not None and is_main_process():
                    self.tb_writer.add_scalar("best/eval_accuracy", best_metric, epoch)
                    self.tb_writer.add_text("checkpoint/best_model", os.path.abspath(os.path.join(self.ckpt_dir, "best.pt")), epoch)
            self.save_checkpoint("last.pt", epoch, best_metric)
        if self.tb_writer is not None:
            self.tb_writer.add_text("checkpoint/final_model", os.path.abspath(os.path.join(self.ckpt_dir, "last.pt")), self.args.num_epochs)
            try:
                self.tb_writer.add_hparams({
                    "learning_rate": self.args.learning_rate,
                    "batch_size": self.args.batch_size,
                    "weight_decay": self.args.weight_decay,
                    "gtn_layers": self.args.gtn_layers,
                    "gtn_channels": self.args.gtn_channels,
                    "num_gcn_layers": self.args.num_gcn_layers,
                    "negative_ratio": self.args.negative_ratio,
                    "lambda_align": self.args.lambda_align,
                    "lambda_node": self.args.lambda_node,
                    "tau": self.args.tau,
                }, {"hparam/best_eval_accuracy": best_metric})
            except Exception:
                pass
            self.tb_writer.flush()
            self.tb_writer.close()



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage2 GTN-style training with negatives")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default="outputs/stage2_gtn_v2")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--features-file", default=None)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--use-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-logdir", default="runs/stage2_gtn_v2")
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=-1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-query-length", type=int, default=64)
    parser.add_argument("--max-workspace-size", type=int, default=10)
    parser.add_argument("--max-header-texts", type=int, default=12)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--embedding-strategy", default="cls")
    parser.add_argument("--normalize-embeddings", action="store_true")
    parser.add_argument("--use-layer-mix", action="store_true")
    parser.add_argument("--use-extra-position-embedding", action="store_true")
    parser.add_argument("--position-embedding-scale", type=float, default=1.0)
    parser.add_argument("--graph-dropout", type=float, default=0.1)
    parser.add_argument("--num-gcn-layers", type=int, default=1)
    parser.add_argument("--gtn-channels", type=int, default=4)
    parser.add_argument("--gtn-layers", type=int, default=2)
    parser.add_argument("--negative-ratio", type=float, default=0.5)
    parser.add_argument("--min-negatives", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--lambda-align", type=float, default=0.10)
    parser.add_argument("--lambda-node", type=float, default=0.20)
    parser.add_argument("--pos-weight", type=float, default=2.0)
    parser.add_argument("--disable-shape-feature", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    trainer = Stage2TrainerV2(args)
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
