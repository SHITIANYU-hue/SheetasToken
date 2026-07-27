#!/usr/bin/env python3
"""Create or validate a metadata-only SheetasToken public data snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SHEET_FIELDS = {"sheet_id", "name", "num_rows", "num_cols", "columns", "source"}
COLUMN_FIELDS = {"name"}
CURRENT_QUERY_FIELDS = {
    "query",
    "query_type",
    "dependency_type",
    "positive_sheet_ids",
    "negative_sheet_ids",
    "is_dependency_heavy",
    "hard_negative_sheet_ids",
    "hard_negative_types",
    "dependency_sheet_ids",
    "dependency_positive_sheet_ids",
    "dependency_edge_sheet_ids",
    "dependency_edges",
    "split",
}
LEGACY_QUERY_FIELDS = {"query", "sheet_ids", "sheet_ids_negative"}


def sanitize_sheet(key: str, record: dict[str, Any]) -> dict[str, Any]:
    columns = []
    for column in record.get("columns", []):
        name = column.get("name", "") if isinstance(column, dict) else column
        name = str(name).strip()
        if name:
            columns.append({"name": name})
    return {
        "sheet_id": record.get("sheet_id", key),
        "name": str(record.get("name", "")).strip(),
        "num_rows": int(record.get("num_rows", 0) or 0),
        "num_cols": int(record.get("num_cols", 0) or 0),
        "columns": columns,
        "source": str(record.get("source", "")).strip(),
    }


def sanitize(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("sheets.json must be an object keyed by sheet ID")
    return {
        str(key): sanitize_sheet(str(key), record)
        for key, record in payload.items()
        if isinstance(record, dict)
    }


def validate(payload: Any) -> list[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["sheets.json must be an object keyed by sheet ID"]
    for key, record in payload.items():
        if not isinstance(record, dict):
            errors.append(f"{key}: sheet record is not an object")
            continue
        extra = set(record) - SHEET_FIELDS
        if extra:
            errors.append(f"{key}: disallowed sheet fields: {sorted(extra)}")
        for index, column in enumerate(record.get("columns", [])):
            if not isinstance(column, dict):
                errors.append(f"{key}.columns[{index}]: not an object")
                continue
            extra = set(column) - COLUMN_FIELDS
            if extra:
                errors.append(
                    f"{key}.columns[{index}]: disallowed fields: {sorted(extra)}"
                )
            if not str(column.get("name", "")).strip():
                errors.append(f"{key}.columns[{index}]: empty column name")
    return errors


def validate_queries(
    payload: Any,
    valid_sheet_ids: set[str],
    allowed_fields: set[str],
    source: str,
) -> list[str]:
    errors = []
    if not isinstance(payload, list):
        return [f"{source}: expected a list"]
    id_fields = {
        "sheet_ids",
        "sheet_ids_negative",
        "positive_sheet_ids",
        "negative_sheet_ids",
        "hard_negative_sheet_ids",
        "dependency_sheet_ids",
        "dependency_positive_sheet_ids",
        "dependency_edge_sheet_ids",
    }
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            errors.append(f"{source}[{index}]: not an object")
            continue
        extra = set(record) - allowed_fields
        if extra:
            errors.append(f"{source}[{index}]: disallowed fields: {sorted(extra)}")
        if not str(record.get("query", "")).strip():
            errors.append(f"{source}[{index}]: empty query")
        for field in id_fields & set(record):
            unknown = {
                str(value)
                for value in record[field]
                if str(value) not in valid_sheet_ids
            }
            if unknown:
                errors.append(
                    f"{source}[{index}].{field}: unknown sheet IDs {sorted(unknown)}"
                )
    return errors


def validate_dependencies(payload: Any, valid_sheet_ids: set[str]) -> list[str]:
    if not isinstance(payload, dict):
        return ["dependency_edges.json: expected an object"]
    errors = []
    for index, edge in enumerate(payload.get("edges", [])):
        if not isinstance(edge, list) or len(edge) != 2:
            errors.append(f"dependency_edges.json.edges[{index}]: invalid edge")
            continue
        unknown = {str(value) for value in edge if str(value) not in valid_sheet_ids}
        if unknown:
            errors.append(
                f"dependency_edges.json.edges[{index}]: unknown IDs {sorted(unknown)}"
            )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sheets.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the input without writing a sanitized copy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if args.check:
        errors = validate(payload)
        valid_sheet_ids = {
            str(record.get("sheet_id", key))
            for key, record in payload.items()
            if isinstance(record, dict)
        }
        current_queries = args.input.parent / "query.json"
        legacy_queries = args.input.parent / "query_legacy_134.json"
        dependencies = args.input.parent / "dependency_edges.json"
        if current_queries.exists():
            errors.extend(
                validate_queries(
                    json.loads(current_queries.read_text(encoding="utf-8")),
                    valid_sheet_ids,
                    CURRENT_QUERY_FIELDS,
                    current_queries.name,
                )
            )
        if legacy_queries.exists():
            errors.extend(
                validate_queries(
                    json.loads(legacy_queries.read_text(encoding="utf-8")),
                    valid_sheet_ids,
                    LEGACY_QUERY_FIELDS,
                    legacy_queries.name,
                )
            )
        if dependencies.exists():
            errors.extend(
                validate_dependencies(
                    json.loads(dependencies.read_text(encoding="utf-8")),
                    valid_sheet_ids,
                )
            )
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"OK: {len(payload)} sheets and accompanying metadata-only files")
        return
    if args.output is None:
        raise SystemExit("--output is required unless --check is used")
    sanitized = sanitize(payload)
    errors = validate(sanitized)
    if errors:
        raise SystemExit("\n".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(sanitized)} metadata-only sheet records to {args.output}")


if __name__ == "__main__":
    main()
