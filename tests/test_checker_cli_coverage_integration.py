from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts import evaluate_physics_eval_sets
from scripts import evaluate_question_level_sets
from scripts import run_verifier


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _synthetic_dataset() -> list[dict[str, Any]]:
    rows = []
    for sample_id in ("checker_fail", "semantic_fail", "missing", "valid_nonlegacy"):
        prediction = f"Synthetic answer for {sample_id}."
        rows.append(
            {
                "id": sample_id,
                "prediction": prediction,
                "eval_split": "correct" if sample_id in {"checker_fail", "valid_nonlegacy"} else "error",
                "expected_has_physics_error": sample_id not in {"checker_fail", "valid_nonlegacy"},
                "physics_error_gt": [
                    {
                        "error_id": f"{sample_id}_error",
                        "error_text": "The synthetic physical claim is incorrect.",
                        "answer_quote": prediction,
                        "start_char": 0,
                        "end_char": len(prediction),
                        "span_valid": True,
                    }
                ],
            }
        )
    return rows


def _synthetic_results() -> list[dict[str, Any]]:
    return [
        {
            "id": "checker_fail",
            "checker_gate_mode": "dual_evidence",
            "checker_status": "partial_failure",
            "checker_failure_count": 1,
            "selection_strategy": "semantic_tree_selection",
            "diagnostics": [],
        },
        {
            "id": "semantic_fail",
            "checker_gate_mode": "dual_evidence",
            "checker_status": "complete",
            "checker_failure_count": 0,
            "selection_strategy": "semantic_error",
            "semantic_selection_error": "synthetic retrieval failure",
            "diagnostics": [],
        },
        {
            "id": "valid_nonlegacy",
            "checker_gate_mode": "dual_evidence",
            "checker_status": "complete",
            "checker_failure_count": 0,
            "selection_strategy": "semantic_tree_selection",
            "diagnostics": [],
        },
    ]


def _synthetic_audit() -> list[dict[str, Any]]:
    return [
        {
            "id": "valid_nonlegacy",
            "experience_code_checks": [
                {
                    "result": "fail",
                    "rule": "experience_code::synthetic",
                    "message": "Synthetic audit-only failure.",
                    "evidence": "Synthetic evidence.",
                }
            ],
        }
    ]


class CheckerCliCoverageIntegrationTests(unittest.TestCase):
    def test_evaluator_identity_is_typed_unique_and_single_arm(self) -> None:
        for module in (evaluate_question_level_sets, evaluate_physics_eval_sets):
            with self.subTest(module=module.__name__):
                indexed = module._index_by_id(
                    [{"id": 0}, {"id": "0"}],
                    label="synthetic",
                )
                self.assertEqual(len(indexed), 2)
                with self.assertRaisesRegex(ValueError, "duplicate typed sample ID"):
                    module._index_by_id(
                        [{"id": 0}, {"id": 0}],
                        label="synthetic",
                    )
                with self.assertRaisesRegex(ValueError, "multiple checker_gate_mode"):
                    module._result_identity(
                        [
                            {"id": "a", "checker_gate_mode": "legacy"},
                            {"id": "b", "checker_gate_mode": "dual_evidence"},
                        ]
                    )

    def test_run_verifier_forwards_checker_options_and_preserves_failure_status(self) -> None:
        created: list[Any] = []

        class _AvailableMatcher:
            available = True

        class FakeVerifier:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                self._unified_v2_mode = True
                self.semantic_matcher = _AvailableMatcher()
                created.append(self)

            def run_batch(self, samples: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
                self.run_batch_kwargs = kwargs
                return [
                    {
                        "id": samples[0]["id"],
                        "checker_gate_mode": "dual_evidence_consistency",
                        "checker_status": "partial_failure",
                        "checker_failures": [
                            {"rule_id": "synthetic_rule_1"},
                            {"rule_id": "synthetic_rule_2"},
                        ],
                        "selection_strategy": "semantic_tree_selection",
                        "semantic_selection_error": "",
                        "diagnostics": [],
                        "score": 0.0,
                    }
                ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.json"
            catalog_path = root / "catalog.json"
            output_path = root / "results.json"
            audit_path = root / "audit.json"
            full_path = root / "full.json"
            _write_json(input_path, [{"id": "synthetic", "question": "Q", "prediction": "A"}])
            _write_json(catalog_path, {"metadata": {"catalog_type": "unified_rules_v2"}})

            argv = [
                "run_verifier.py",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--symbolic-output",
                str(audit_path),
                "--full-output",
                str(full_path),
                "--unified-catalog",
                str(catalog_path),
                "--unified-retrieval-mode",
                "semantic",
                "--checker-gate-mode",
                "dual_evidence_consistency",
                "--checker-json-attempts",
                "3",
                "--no-symbolic-check",
                "--no-llm-cache",
                "--progress-interval",
                "0",
                "--checkpoint-every",
                "0",
                "--continue-on-semantic-error",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(run_verifier, "PhysicsRuleVerifier", FakeVerifier),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as raised:
                    run_verifier.main()

            main_rows = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(raised.exception.code, 4)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].kwargs["checker_gate_mode"], "dual_evidence_consistency")
        self.assertEqual(created[0].kwargs["checker_json_attempts"], 3)
        self.assertEqual(main_rows[0]["checker_gate_mode"], "dual_evidence_consistency")
        self.assertEqual(main_rows[0]["checker_status"], "partial_failure")
        self.assertEqual(main_rows[0]["checker_failure_count"], 2)

    def test_question_evaluator_reports_failures_as_coverage_not_tn_or_fn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "dataset.json"
            results_path = root / "results.json"
            audit_path = root / "audit.json"
            output_path = root / "metrics.json"
            _write_json(dataset_path, _synthetic_dataset())
            _write_json(results_path, _synthetic_results())
            _write_json(audit_path, _synthetic_audit())

            argv = [
                "evaluate_question_level_sets.py",
                "--dataset",
                str(dataset_path),
                "--results",
                str(results_path),
                "--audit",
                str(audit_path),
                "--output",
                str(output_path),
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                evaluate_question_level_sets.main()

            report = json.loads(output_path.read_text(encoding="utf-8"))

        summary = report["summary"]
        self.assertEqual(summary["dataset_size"], 4)
        self.assertEqual(summary["scored_size"], 1)
        self.assertEqual(summary["failed_size"], 3)
        self.assertEqual(summary["coverage"], 0.25)
        self.assertEqual(
            summary["failure_by_stage"],
            {
                "checker_failure": 1,
                "semantic_retrieval_failure": 1,
                "missing_result": 1,
            },
        )
        self.assertEqual(
            {key: summary[key] for key in ("tp", "fp", "tn", "fn")},
            {"tp": 0, "fp": 0, "tn": 1, "fn": 0},
        )
        valid = next(row for row in report["details"] if row["id"] == "valid_nonlegacy")
        self.assertTrue(valid["scored"])
        self.assertFalse(valid["pred_has_error"])
        self.assertEqual(valid["pred_finding_count"], 0)

    def test_error_evaluator_excludes_failures_and_nonlegacy_audit_findings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "dataset.json"
            results_path = root / "results.json"
            audit_path = root / "audit.json"
            output_path = root / "metrics.json"
            _write_json(dataset_path, _synthetic_dataset())
            _write_json(results_path, _synthetic_results())
            _write_json(audit_path, _synthetic_audit())

            argv = [
                "evaluate_physics_eval_sets.py",
                "--dataset",
                str(dataset_path),
                "--results",
                str(results_path),
                "--audit",
                str(audit_path),
                "--output",
                str(output_path),
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                evaluate_physics_eval_sets.main()

            report = json.loads(output_path.read_text(encoding="utf-8"))

        summary = report["summary"]
        self.assertEqual(summary["dataset_size"], 4)
        self.assertEqual(summary["scored_size"], 1)
        self.assertEqual(summary["failed_size"], 3)
        self.assertEqual(summary["coverage"], 0.25)
        self.assertEqual(
            summary["failure_by_stage"],
            {
                "checker_failure": 1,
                "semantic_retrieval_failure": 1,
                "missing_result": 1,
            },
        )
        self.assertEqual(summary["total_gt_errors"], 1)
        self.assertEqual(summary["matched_gt_errors"], 0)
        self.assertEqual(summary["sample_trigger_ratio"], 0.0)
        valid = next(row for row in report["details"] if row["id"] == "valid_nonlegacy")
        self.assertTrue(valid["scored"])
        self.assertFalse(valid["pred_has_error"])
        self.assertEqual(valid["pred_finding_count"], 0)


if __name__ == "__main__":
    unittest.main()
