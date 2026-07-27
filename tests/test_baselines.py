import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baselines.common import (
    evaluate_predictions,
    parse_ks,
    serialize_sheet,
    write_json_atomic,
)
from baselines.llm_selector import (
    SheetSelection,
    build_arg_parser,
    build_catalog,
    build_system_prompt,
    run,
    sanitize_selection,
)


class PublicReleaseDataTests(unittest.TestCase):
    def test_default_data_is_industrytab_1k(self):
        root = Path(__file__).resolve().parents[1]
        data = root / "data"
        sheets = json.loads((data / "sheets.json").read_text())
        queries = json.loads((data / "query.json").read_text())
        pairs = json.loads((data / "train.json").read_text())
        dependencies = json.loads((data / "dependency_edges.json").read_text())

        self.assertEqual(len(sheets), 1002)
        self.assertEqual(len(queries), 1797)
        self.assertEqual(len(pairs), 3006)
        self.assertEqual(len(dependencies["edges"]), 19850)
        self.assertEqual(
            (data / "sheets.json").read_bytes(),
            (data / "industrytab_1k" / "sheets.json").read_bytes(),
        )
        self.assertEqual(
            (data / "query.json").read_bytes(),
            (data / "industrytab_1k" / "query.json").read_bytes(),
        )


class CommonBaselineTests(unittest.TestCase):
    def test_sheet_serialization_matches_stage2_shape(self):
        sheet = {
            "name": "Sales",
            "num_rows": 10,
            "num_cols": 2,
            "columns": [
                {"name": "Region", "example": "West"},
                {"name": "Revenue", "example": "100"},
            ],
        }
        self.assertEqual(
            serialize_sheet(sheet),
            "name: Sales ; shape: 10 x 2 ; columns: Region | Revenue",
        )
        self.assertIn("Region: West", serialize_sheet(sheet, include_examples=True))

    def test_metrics_for_known_ranking(self):
        metrics = evaluate_predictions(
            [
                {
                    "relevant_sheet_ids": ["a", "c"],
                    "predicted_sheet_ids": ["a", "b", "c"],
                }
            ],
            (1, 3),
        )
        self.assertEqual(metrics["Precision@1"], 1.0)
        self.assertEqual(metrics["Recall@1"], 0.5)
        self.assertEqual(metrics["Precision@3"], round(2 / 3, 6))
        self.assertEqual(metrics["Recall@3"], 1.0)
        self.assertEqual(metrics["MRR@3"], 1.0)

    def test_parse_ks_and_atomic_json(self):
        self.assertEqual(parse_ks("10,1,3,3"), (1, 3, 10))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            write_json_atomic(path, {"ok": True})
            self.assertIn('"ok": true', path.read_text())


class LLMSelectorTests(unittest.TestCase):
    def test_catalog_prompt_and_selection_sanitization(self):
        sheets = {
            "1": {
                "sheet_id": 1,
                "name": "A",
                "num_rows": 1,
                "num_cols": 1,
                "columns": [],
            },
            "2": {
                "sheet_id": 2,
                "name": "B",
                "num_rows": 1,
                "num_cols": 1,
                "columns": [],
            },
        }
        catalog = build_catalog(sheets, max_columns=12, include_examples=False)
        prompt = build_system_prompt(catalog, top_k=2)
        self.assertIn("[1] name: A", prompt)
        self.assertEqual(
            sanitize_selection(["bytesheet_2", "missing", "2", "1"], {"1", "2"}, 2),
            ["2", "1"],
        )

    def test_llm_run_with_structured_response(self):
        class FakeResponses:
            def parse(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    id="response_test",
                    output_parsed=SheetSelection(sheet_ids=["2"]),
                    usage=SimpleNamespace(
                        input_tokens=100,
                        output_tokens=5,
                        total_tokens=105,
                        input_tokens_details=SimpleNamespace(cached_tokens=80),
                    ),
                )

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.responses = FakeResponses()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sheets_path = root / "sheets.json"
            queries_path = root / "query.json"
            output_path = root / "report.json"
            sheets_path.write_text(
                json.dumps(
                    {
                        "1": {
                            "sheet_id": 1,
                            "name": "A",
                            "num_rows": 1,
                            "num_cols": 1,
                            "columns": [],
                        },
                        "2": {
                            "sheet_id": 2,
                            "name": "B",
                            "num_rows": 1,
                            "num_cols": 1,
                            "columns": [],
                        },
                    }
                )
            )
            queries_path.write_text(json.dumps([{"query": "find B", "sheet_ids": [2]}]))
            args = build_arg_parser().parse_args(
                [
                    "--sheets-file",
                    str(sheets_path),
                    "--queries-file",
                    str(queries_path),
                    "--output",
                    str(output_path),
                    "--top-k",
                    "1",
                    "--metric-ks",
                    "1",
                    "--max-retries",
                    "1",
                ]
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}), patch(
                "openai.OpenAI", FakeOpenAI
            ):
                report = run(args)

            self.assertEqual(report["metrics"]["Recall@1"], 1.0)
            self.assertEqual(report["usage"]["cached_input_tokens"], 80)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
