#!/usr/bin/env python3
"""Benchmark online full-corpus retrieval and reranking latency.

Offline model loading and sheet embedding construction are excluded. Query
tokenization, query encoding, full-corpus similarity search, candidate feature
assembly, host-to-device transfer, reranking, and top-k selection are included.
All measurements use online batch size one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bge_retrain_experiments import (
    QUERY_PREFIX,
    CrossEncoder,
    Reranker,
    serialize_sheet,
)
from gated_graph_refine import GatedGraphFromMLP, load_dependency_channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--tuned-cache", required=True)
    parser.add_argument("--finetuned-state", required=True)
    parser.add_argument("--mlp-checkpoint", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--cross-checkpoint")
    parser.add_argument("--data-dir")
    parser.add_argument("--dependency-edges", required=True)
    parser.add_argument(
        "--dependency-types", default="formula_reference,summary_source"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)

    def percentile(value: float) -> float:
        index = min(len(ordered) - 1, round((len(ordered) - 1) * value))
        return ordered[index] * 1000

    return {
        "queries": len(samples),
        "mean_ms": statistics.fmean(samples) * 1000,
        "median_ms": statistics.median(samples) * 1000,
        "p95_ms": percentile(0.95),
        "std_ms": statistics.pstdev(samples) * 1000,
    }


def load_encoder(
    model_path: str,
    device: torch.device,
    finetuned_state: str | None = None,
) -> tuple[object, torch.nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    if finetuned_state:
        update = torch.load(finetuned_state, map_location="cpu", weights_only=False)
        state = model.state_dict()
        state.update(update)
        model.load_state_dict(state)
    return tokenizer, model.to(device).eval()


@torch.inference_mode()
def benchmark_retrieval(
    cache: dict,
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    candidate_k: int,
    max_length: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    positions = list(cache["eval_positions"])
    sheet_embeddings = cache["sheet_embeddings"].to(device)

    def invoke(position: int) -> None:
        text = QUERY_PREFIX + cache["queries"][position]["query"]
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            hidden = model(**encoded, return_dict=True).last_hidden_state[:, 0]
        query = F.normalize(hidden.float(), dim=-1)
        torch.topk(query @ sheet_embeddings.T, k=candidate_k, dim=1)

    for position in positions[:warmup]:
        invoke(position)
    synchronize(device)

    samples = []
    for _ in range(repeats):
        for position in positions:
            synchronize(device)
            started = time.perf_counter()
            invoke(position)
            synchronize(device)
            samples.append(time.perf_counter() - started)
    return summarize(samples)


def candidate_inputs(
    cache: dict,
    position: int,
    candidate_k: int,
    dependency: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    candidates = cache["candidates"][position, :candidate_k]
    return (
        cache["query_embeddings"][position][None].to(device),
        cache["sheet_embeddings"][candidates][None].to(device),
        cache["candidate_scores"][position, :candidate_k][None].to(device),
        cache["schema"][candidates][:, candidates][None].to(device),
        cache["shape"][candidates][:, candidates][None].to(device),
        dependency[:, candidates][:, :, candidates][None].to(device),
    )


@torch.inference_mode()
def benchmark_reranker(
    cache: dict,
    model: torch.nn.Module,
    device: torch.device,
    candidate_k: int,
    dependency: torch.Tensor,
    use_dependency: bool,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    positions = list(cache["eval_positions"])

    def invoke(position: int) -> None:
        query, nodes, raw, schema, shape, dep = candidate_inputs(
            cache, position, candidate_k, dependency, device
        )
        if use_dependency:
            logits = model(query, nodes, raw, schema, shape, dep)
        else:
            logits = model(query, nodes, raw, schema, shape)
        torch.topk(logits, k=5, dim=1)

    for position in positions[:warmup]:
        invoke(position)
    synchronize(device)

    samples = []
    for _ in range(repeats):
        for position in positions:
            synchronize(device)
            started = time.perf_counter()
            invoke(position)
            synchronize(device)
            samples.append(time.perf_counter() - started)
    return summarize(samples)


@torch.inference_mode()
def benchmark_cross_encoder(
    cache: dict,
    sheets: dict[str, dict],
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    candidate_k: int,
    max_length: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    positions = list(cache["eval_positions"])

    def invoke(position: int) -> None:
        candidates = cache["candidates"][position, :candidate_k]
        query = cache["queries"][position]["query"]
        encoded = tokenizer(
            [query] * candidate_k,
            [
                serialize_sheet(sheets[cache["sheet_ids"][index]])
                for index in candidates.tolist()
            ],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(encoded)
        torch.topk(logits, k=5)

    for position in positions[:warmup]:
        invoke(position)
    synchronize(device)

    samples = []
    for _ in range(repeats):
        for position in positions:
            synchronize(device)
            started = time.perf_counter()
            invoke(position)
            synchronize(device)
            samples.append(time.perf_counter() - started)
    return summarize(samples)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    base = torch.load(args.base_cache, map_location="cpu", weights_only=False)
    tuned = torch.load(args.tuned_cache, map_location="cpu", weights_only=False)
    if list(base["eval_positions"]) != list(tuned["eval_positions"]):
        raise ValueError("Base and tuned caches use different evaluation queries")

    output: dict[str, object] = {
        "protocol": {
            "online_batch_size": 1,
            "warmup_queries": args.warmup,
            "repeats": args.repeats,
            "eval_queries": len(tuned["eval_positions"]),
            "corpus_sheets": len(tuned["sheet_ids"]),
            "max_length": args.max_length,
            "excluded": ["model loading", "offline sheet embedding construction"],
            "included": [
                "query tokenization and encoding",
                "full-corpus similarity and top-k selection",
                "candidate feature assembly and host-to-device transfer",
                "reranker forward pass and top-5 selection",
            ],
        },
        "hardware": {
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "torch": torch.__version__,
        },
    }

    tokenizer, encoder = load_encoder(base["model_path"], device)
    output["rag_bge_stage1_top5"] = benchmark_retrieval(
        base,
        tokenizer,
        encoder,
        device,
        5,
        args.max_length,
        args.warmup,
        args.repeats,
    )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tokenizer, encoder = load_encoder(
        tuned["model_path"], device, args.finetuned_state
    )
    output["sat_stage1_top50"] = benchmark_retrieval(
        tuned,
        tokenizer,
        encoder,
        device,
        50,
        args.max_length,
        args.warmup,
        args.repeats,
    )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if bool(args.cross_checkpoint) != bool(args.data_dir):
        raise ValueError("--cross-checkpoint and --data-dir must be provided together")
    if args.cross_checkpoint:
        raw_sheets = json.loads(
            (Path(args.data_dir) / "sheets.json").read_text(encoding="utf-8")
        )
        sheets = {
            str(record.get("sheet_id", key)): record
            for key, record in raw_sheets.items()
            if isinstance(record, dict)
        }
        cross = CrossEncoder(tuned["model_path"]).to(device).eval()
        tuned_backbone = torch.load(
            args.finetuned_state, map_location="cpu", weights_only=False
        )
        backbone_state = cross.backbone.state_dict()
        backbone_state.update(
            {
                name: tensor
                for name, tensor in tuned_backbone.items()
                if name in backbone_state
            }
        )
        cross.backbone.load_state_dict(backbone_state)
        update = torch.load(
            args.cross_checkpoint, map_location="cpu", weights_only=False
        )
        state = cross.state_dict()
        state.update(update)
        cross.load_state_dict(state)
        output["sat_stage2_cross_encoder_top50"] = benchmark_cross_encoder(
            tuned,
            sheets,
            tokenizer,
            cross,
            device,
            50,
            args.max_length,
            args.warmup,
            args.repeats,
        )
        del cross
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mlp_checkpoint = torch.load(
        args.mlp_checkpoint, map_location="cpu", weights_only=False
    )
    dim = tuned["sheet_embeddings"].size(1)
    mlp = Reranker(dim, "mlp").to(device).eval()
    mlp.load_state_dict(mlp_checkpoint["state_dict"])
    empty_dependency = torch.zeros(0, len(tuned["sheet_ids"]), len(tuned["sheet_ids"]))
    output["sat_stage2_mlp_top50"] = benchmark_reranker(
        tuned,
        mlp,
        device,
        50,
        empty_dependency,
        False,
        args.warmup,
        args.repeats,
    )
    del mlp
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gnn_checkpoint = torch.load(
        args.gnn_checkpoint, map_location="cpu", weights_only=False
    )
    gnn_args = gnn_checkpoint["args"]
    dependency_types = [
        value.strip() for value in args.dependency_types.split(",") if value.strip()
    ]
    dependency = load_dependency_channels(
        args.dependency_edges, tuned["sheet_ids"], dependency_types
    )
    gnn = GatedGraphFromMLP(
        dim,
        int(gnn_args["layers"]),
        float(gnn_args["gate_init"]),
        float(gnn_args["dropout"]),
        mlp_checkpoint["state_dict"],
        dependency.size(0),
        bool(gnn_args.get("score_message", False)),
        float(gnn_args.get("score_gate_init", -4.0)),
    ).to(device).eval()
    gnn.load_state_dict(gnn_checkpoint["state_dict"])
    output["sat_stage2_gnn_top50"] = benchmark_reranker(
        tuned,
        gnn,
        device,
        50,
        dependency,
        True,
        args.warmup,
        args.repeats,
    )

    stage1 = output["sat_stage1_top50"]["mean_ms"]
    output["derived_mean_ms"] = {
        "sat_stage1_only": stage1,
        "sat_mlp_end_to_end": stage1
        + output["sat_stage2_mlp_top50"]["mean_ms"],
        "sat_gnn_end_to_end": stage1
        + output["sat_stage2_gnn_top50"]["mean_ms"],
    }
    if "sat_stage2_cross_encoder_top50" in output:
        output["derived_mean_ms"]["sat_cross_encoder_end_to_end"] = (
            stage1 + output["sat_stage2_cross_encoder_top50"]["mean_ms"]
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
