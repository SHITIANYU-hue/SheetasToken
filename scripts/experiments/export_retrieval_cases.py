#!/usr/bin/env python3
"""Extract inspectable retrieval cases from the already-public predictions."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = ((1554, 42, "benefit_with_graph_controls"),
         (1508, 42, "reranking_degradation"),
         (548, 43, "hard_negative_promotion_and_candidate_loss"),
         (1476, 42, "benefit_also_reached_by_permuted_control"))


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {
        "predictions": ROOT / "artifacts/experiments/per_query_predictions.json",
        "queries": ROOT / "data/industrytab_1k/query.json",
        "sheets": ROOT / "data/industrytab_1k/sheets.json",
        "graph": ROOT / "data/industrytab_1k/dependency_edges.json",
    }
    predictions, queries, sheets, graph = (read(paths[k]) for k in paths)
    cases = []
    for index, seed, role in CASES:
        matches = [p for p in predictions if p["dataset"] == "industrytab_1k"
                   and p["query_index"] == index and p["seed"] == seed]
        assert len(matches) == 1, (index, seed)
        record = matches[0]
        query = queries[index]
        assert record["query"] == query["query"]
        assert set(record["gold_ids"]) == set(map(str, query["positive_sheet_ids"]))
        assert set(record["hard_negative_ids"]) == set(map(str, query["hard_negative_sheet_ids"]))
        candidates = set(record["ranked_ids"]["stage1"])
        for ranking in record["ranked_ids"].values():
            assert len(ranking) == len(set(ranking)) == 50
            assert set(ranking) == candidates
        gold = set(record["gold_ids"])
        relations = []
        for pair, types in graph["edge_all_types"].items():
            endpoints = pair.split(",")
            selected = sorted(set(types) & {"formula_reference", "summary_source"})
            if selected and set(endpoints) <= candidates and set(endpoints) & gold:
                relations.append({"sheet_ids": endpoints, "relation_types": selected})
        ids = candidates | gold | set(record["hard_negative_ids"])
        assert ids <= sheets.keys()
        cases.append({
            "illustration": role,
            "record": record,
            "sheet_names": {i: sheets[i]["name"] for i in sorted(ids, key=int)},
            "selected_candidate_relations_touching_gold": relations,
        })
    payload = {
        "protocol": "Outcome-selected illustrations, not prevalence estimates or causal edge attribution.",
        "query_index_convention": "Zero-based indices into industrytab_1k/query.json.",
        "source_sha256": {k: digest(p) for k, p in paths.items()},
        "checks": {"records_unchanged_from_public_predictions": True,
                   "queries_and_gold_and_hard_negatives_match_release": True,
                   "all_methods_rerank_identical_top50_candidate_sets": True,
                   "relation_pairs_are_selected_archived_candidate_edges": True},
        "cases": cases,
    }
    output = ROOT / "artifacts/experiments/retrieval_cases.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Exported and checked {len(cases)} retrieval cases.")


if __name__ == "__main__":
    main()
