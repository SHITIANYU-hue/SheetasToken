from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_KS = (1, 3, 5, 10)


def load_sheets(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load either the repository's dict format or a list of sheet records."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sheets: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = ((str(i), item) for i, item in enumerate(raw))
    else:
        raise ValueError(f"Unsupported sheets JSON type: {type(raw).__name__}")

    for fallback_id, item in items:
        if not isinstance(item, dict):
            continue
        sheet_id = str(item.get("sheet_id", fallback_id))
        if sheet_id in sheets:
            raise ValueError(f"Duplicate sheet_id: {sheet_id}")
        sheets[sheet_id] = item

    if not sheets:
        raise ValueError(f"No valid sheets found in {path}")
    return sheets


def load_queries(
    path: str | Path,
    valid_sheet_ids: Iterable[str],
    max_queries: int = 0,
) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("query.json must contain a list")

    valid_ids = set(map(str, valid_sheet_ids))
    queries: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        relevant = _deduplicate(str(x) for x in item.get("sheet_ids", []))
        relevant = [sheet_id for sheet_id in relevant if sheet_id in valid_ids]
        negative_raw = item.get(
            "sheet_ids_negative", item.get("negative_sheet_ids", [])
        )
        labeled_candidates = _deduplicate([*relevant, *(str(x) for x in negative_raw)])
        labeled_candidates = [
            sheet_id for sheet_id in labeled_candidates if sheet_id in valid_ids
        ]
        if not query or not relevant:
            continue
        queries.append(
            {
                "query_index": index,
                "query": query,
                "relevant_sheet_ids": relevant,
                "labeled_candidate_sheet_ids": labeled_candidates,
            }
        )
        if max_queries > 0 and len(queries) >= max_queries:
            break

    if not queries:
        raise ValueError(f"No valid queries found in {path}")
    return queries


def serialize_sheet(
    sheet: Mapping[str, Any],
    max_columns: int = 12,
    include_examples: bool = False,
) -> str:
    """Match the Stage 2 sheet serialization used by the trained retriever."""
    parts: List[str] = []
    name = str(sheet.get("name", "")).strip()
    if name:
        parts.append(f"name: {name}")
    parts.append(f"shape: {sheet.get('num_rows', '?')} x {sheet.get('num_cols', '?')}")

    column_texts: List[str] = []
    for column in list(sheet.get("columns", []))[:max_columns]:
        if isinstance(column, dict):
            column_name = str(column.get("name", "")).strip()
            example = str(column.get("example", "")).strip()
            text = (
                f"{column_name}: {example}"
                if include_examples and example
                else column_name
            )
        else:
            text = str(column).strip()
        if text:
            column_texts.append(text)
    if column_texts:
        parts.append("columns: " + " | ".join(column_texts))
    return " ; ".join(parts)


def parse_ks(value: str | Sequence[int]) -> Tuple[int, ...]:
    if isinstance(value, str):
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        values = [int(k) for k in value]
    ks = tuple(sorted(set(values)))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("Metric cutoffs must be positive integers")
    return ks


def evaluate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, float | int]:
    """Compute macro retrieval metrics over multi-label query relevance."""
    cutoffs = parse_ks(ks)
    if not predictions:
        raise ValueError("No predictions to evaluate")

    totals: Dict[str, float] = {}
    for k in cutoffs:
        totals[f"Precision@{k}"] = 0.0
        totals[f"Recall@{k}"] = 0.0
        totals[f"HitRate@{k}"] = 0.0
        totals[f"nDCG@{k}"] = 0.0
        totals[f"MRR@{k}"] = 0.0

    for item in predictions:
        relevant = set(map(str, item.get("relevant_sheet_ids", [])))
        ranked = _deduplicate(str(x) for x in item.get("predicted_sheet_ids", []))
        if not relevant:
            raise ValueError(
                "Each evaluated query must have at least one relevant sheet"
            )

        for k in cutoffs:
            top = ranked[:k]
            hits = sum(sheet_id in relevant for sheet_id in top)
            totals[f"Precision@{k}"] += hits / k
            totals[f"Recall@{k}"] += hits / len(relevant)
            totals[f"HitRate@{k}"] += float(hits > 0)

            dcg = sum(
                (1.0 / math.log2(rank + 2))
                for rank, sheet_id in enumerate(top)
                if sheet_id in relevant
            )
            ideal_hits = min(len(relevant), k)
            idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
            totals[f"nDCG@{k}"] += dcg / idcg if idcg else 0.0

            reciprocal_rank = 0.0
            for rank, sheet_id in enumerate(top, start=1):
                if sheet_id in relevant:
                    reciprocal_rank = 1.0 / rank
                    break
            totals[f"MRR@{k}"] += reciprocal_rank

    count = len(predictions)
    metrics: Dict[str, float | int] = {"num_queries": count}
    metrics.update({name: round(value / count, 6) for name, value in totals.items()})
    return metrics


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temp_path, target)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def sheet_sort_key(sheet_id: str) -> tuple[int, int | str]:
    """Sort numeric sheet identifiers numerically before lexical identifiers."""
    value = str(sheet_id)
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _deduplicate(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))
