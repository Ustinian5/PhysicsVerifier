from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from scripts.build_checker_common_valid import _audit_llm_trace, _object_sha256
from scripts.evaluate_checker_mechanism_gate import (
    ARMS,
    REPETITIONS,
    MechanismEvaluationError,
    _validate_formal_evaluator_source_identity,
    evaluate_mechanism_gate,
)
from core.semantic_rule_checker import SEMANTIC_RULE_CHECKER_PROMPT_VERSION


RULE_ID = "gen_0123456789abcdef"


class MechanismGateEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset = self._dataset()
        self.dataset_path = self.root / "dataset.json"
        self._write_json(self.dataset_path, self.dataset)
        self.retrieval_path = self.root / "target-binding.json"
        self.catalog_path = self.root / "catalog.json"
        self.frozen_manifest_path = self.root / "frozen-manifest.json"
        self._write_json(self.retrieval_path, {"fixture": "target-binding"})
        self._write_json(self.catalog_path, {"fixture": "catalog"})
        self._write_json(self.frozen_manifest_path, {"fixture": "manifest"})

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _span(source: str, text: str, quote: str) -> Dict[str, Any]:
        start = text.index(quote)
        return {
            "source": source,
            "quote": quote,
            "start_char": start,
            "end_char": start + len(quote),
        }

    def _case(
        self,
        mechanism: str,
        *,
        subtype: str = "none",
    ) -> Dict[str, Any]:
        question = "An isolated two-cart system undergoes a short collision."
        context = "The net external impulse is negligible during the collision."
        applicability_quote = "isolated two-cart system"
        prediction_by_mechanism = {
            "true_violation": "The total momentum decreases because the collision is inelastic.",
            "applicable_correct": "The total momentum remains constant although kinetic energy decreases.",
            "symbol_overlap_inapplicable": "Here p denotes pressure in a static gas, not cart momentum.",
            "equivalent_alternative": "The center-of-mass velocity is constant, an equivalent momentum argument.",
            "insufficient_information": "The information is insufficient to determine whether momentum changed.",
            "self_corrected": (
                "I first claimed that total momentum decreases. That claim is wrong; "
                "the isolated system's total momentum remains constant."
            ),
        }
        key = subtype if mechanism == "insufficient_or_self_corrected" else mechanism
        prediction = prediction_by_mechanism[key]
        applicability = mechanism != "symbol_overlap_inapplicable"
        current_violation = mechanism == "true_violation"
        consistency: Any = None
        if mechanism == "true_violation":
            consistency = "confirmed_violation"
        elif mechanism == "equivalent_alternative":
            consistency = "equivalent_or_alternative"
        elif subtype == "self_corrected":
            consistency = "self_corrected"
        elif subtype == "insufficient_information":
            consistency = "uncertain"
        gt_evidence = {
            "applicability_spans": (
                [self._span("question", question, applicability_quote)] if applicability else []
            ),
            "violation_spans": (
                [
                    self._span(
                        "prediction",
                        prediction,
                        "total momentum decreases",
                    )
                ]
                if current_violation
                else []
            ),
            "superseded_claim_spans": (
                [
                    self._span(
                        "prediction",
                        prediction,
                        "total momentum decreases",
                    )
                ]
                if subtype == "self_corrected"
                else []
            ),
            "correction_spans": (
                [
                    self._span(
                        "prediction",
                        prediction,
                        "total momentum remains constant",
                    )
                ]
                if subtype == "self_corrected"
                else []
            ),
        }
        return {
            "schema_version": "p3_checker_mechanism_case_v1",
            "id": f"p3::{RULE_ID}::{mechanism}",
            "question": question,
            "context": context,
            "prediction": prediction,
            "target_rule_id": RULE_ID,
            "target_rule": {
                "rule_id": RULE_ID,
                "domain_id": "mechanics",
                "domain": "Mechanics",
                "topic_id": "mechanics.momentum",
                "topic": "Momentum",
                "cluster_id": "momentum_checks",
                "origin": "gen",
                "symbolic_primitive": "equation_equivalence",
                "has_symbolic_primitive": True,
                "trigger_scope_proxy": "broad_proxy",
            },
            "mechanism": mechanism,
            "mechanism_subtype": subtype,
            "expected": {
                "applicability": applicability,
                "current_violation": current_violation,
                "publish": current_violation,
                "consistency_status": consistency,
            },
            "gt_evidence": gt_evidence,
            "case_content_sha256": _object_sha256([question, context, prediction]),
            "gt_provenance": {
                "generator_model": "gemini-3-flash-preview",
                "prompt_version": "synthetic-test-v1",
                "plan_sha256": "1" * 64,
                "raw_response_sha256": "2" * 64,
                "validation_status": "schema_validated",
            },
        }

    def _dataset(self) -> List[Dict[str, Any]]:
        return [
            self._case("true_violation"),
            self._case("applicable_correct"),
            self._case("symbol_overlap_inapplicable"),
            self._case("equivalent_alternative"),
            self._case(
                "insufficient_or_self_corrected",
                subtype="self_corrected",
            ),
        ]

    @staticmethod
    def _diagnostic(arm: str, prediction: str, *, contradictory: bool = False) -> Dict[str, Any]:
        quote = "total momentum decreases"
        start = prediction.index(quote)
        diagnostic: Dict[str, Any] = {
            "severity": "error",
            "rule": RULE_ID,
            "message": "The claim violates momentum conservation.",
            "symbol": "p",
            "evidence": {
                "source": "prediction",
                "quote": quote,
                "location": {
                    "start_char": start,
                    "end_char": start + len(quote),
                    "locatable_valid": True,
                },
            },
        }
        if arm != "legacy":
            diagnostic.update(
                {
                    "checker_gate_mode": arm,
                    "checker_evidence_gate": {
                        "passed": not contradictory,
                        "reasons": ["synthetic_contradiction"] if contradictory else [],
                    },
                    "applicability": {"applies": not contradictory},
                    "violation": {"present": True},
                }
            )
            if arm == "dual_evidence_consistency":
                diagnostic["consistency"] = {
                    "status": (
                        "equivalent_or_alternative"
                        if contradictory
                        else "confirmed_violation"
                    )
                }
        return diagnostic

    def _base_result_rows(self, arm: str, configuration_sha256: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in self.dataset:
            publish = item["mechanism"] == "true_violation"
            diagnostics = (
                [self._diagnostic(arm, item["prediction"])] if publish else []
            )
            rows.append(
                {
                    "id": item["id"],
                    "candidate_source": "frozen_target_binding",
                    "unified_retrieval_mode": "target_binding",
                    "selection_strategy": "target_rule_binding",
                    "retrieval_score_kind": "fixed_control_0_1",
                    "target_rule_id": RULE_ID,
                    "retrieved_rules": [
                        {
                            "rule_id": RULE_ID,
                            "candidate_source": "frozen_target_binding",
                            "score": 1.0,
                            "score_kind": "fixed_control_0_1",
                            "partial": False,
                            "executable": True,
                            "domain": "Mechanics",
                            "topic_id": "mechanics.momentum",
                            "topic": "Momentum",
                            "publish_gate": {
                                "publishable": True,
                                "reasons": [],
                                "score": 1.0,
                                "score_kind": "fixed_control_0_1",
                                "min_publish_score": 0.0,
                                "selection_strategy": "target_rule_binding",
                            },
                        }
                    ],
                    "used_rules": [RULE_ID],
                    "checker_gate_mode": arm,
                    "checker_min_confidence": 0.8,
                    "checker_status": (
                        "valid_with_diagnostics" if diagnostics else "valid_empty"
                    ),
                    "checker_failure_count": 0,
                    "checker_failures": [],
                    "checker_decisions": [
                        {
                            "rule_id": RULE_ID,
                            "status": (
                                "valid_with_diagnostics" if diagnostics else "valid_empty"
                            ),
                            "attempt_count": 1,
                            "attempts": [
                                {
                                    "attempt": 1,
                                    "status": (
                                        "valid_json" if arm == "legacy" else "valid_object"
                                    ),
                                }
                            ],
                            "checker_gate_mode": arm,
                            "cache_hit": False,
                            "published_diagnostic_count": len(diagnostics),
                        }
                    ],
                    "candidate_diagnostics": copy.deepcopy(diagnostics),
                    "diagnostics": diagnostics,
                    "replay_completed": True,
                    "replay_config_sha256": configuration_sha256,
                    "replay": {
                        "system_arm": arm,
                        "checker_attempted": True,
                        "checker_succeeded": True,
                        "bottom_up_enabled": False,
                    },
                }
            )
        return rows

    def _write_run(
        self,
        arm: str,
        repetition: int,
        *,
        row_mutation: Any = None,
        configuration_mutation: Any = None,
    ) -> Tuple[Path, Path]:
        prefix = f"{arm}.r{repetition}"
        trace_path = self.root / f"{prefix}.trace.jsonl"
        trace_records = [
            {
                "parse_status": (
                    "json.loads_ok" if arm == "legacy" else "valid_object"
                ),
                "raw_response": json.dumps({"run": prefix, "case": row["id"]}),
                "model": "qwen3-30b-a3b-instruct-2507",
                "checker_mode": arm,
                "trace_meta": {
                    "sample_id": row["id"],
                    "rule_id": RULE_ID,
                    "attempt": 1,
                },
            }
            for row in self.dataset
        ]
        trace_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in trace_records
            ),
            encoding="utf-8",
        )
        result_path = self.root / f"{prefix}.results.json"
        report_path = self.root / f"{prefix}.report.json"
        retrieval_sha256 = hashlib.sha256(self.retrieval_path.read_bytes()).hexdigest()
        catalog_sha256 = hashlib.sha256(self.catalog_path.read_bytes()).hexdigest()
        frozen_manifest_sha256 = hashlib.sha256(
            self.frozen_manifest_path.read_bytes()
        ).hexdigest()
        configuration: Dict[str, Any] = {
            "run_kind": "development",
            "system_arm": arm,
            "candidate_source": "frozen_target_binding",
            "checker_cache_enabled": False,
            "checker_model": "qwen3-30b-a3b-instruct-2507",
            "checker_prompt_version": SEMANTIC_RULE_CHECKER_PROMPT_VERSION,
            "checker_min_confidence": 0.8,
            "llm_trace_include_prompts": False,
            "llm_trace_path": str(trace_path),
            "retrieval_source": "frozen_target_binding",
            "retrieval_execution": False,
            "bottom_up_enabled": False,
            "target_mapping_sha256": "8" * 64,
            "dataset_sha256": hashlib.sha256(self.dataset_path.read_bytes()).hexdigest(),
            "retrieval_trace_sha256": retrieval_sha256,
            "unified_catalog_sha256": catalog_sha256,
            "frozen_manifest_sha256": frozen_manifest_sha256,
            "selection_projection_sha256": "4" * 64,
            "source_identity": {"source_tree_sha256": "5" * 64},
            "runtime_identity": {"is_conda": True, "package_set_sha256": "6" * 64},
            "api_transport_identity": {"endpoint_sha256": "7" * 64},
            "output_path": str(result_path),
            "report_path": str(report_path),
        }
        if configuration_mutation:
            configuration_mutation(configuration)
        configuration_sha256 = _object_sha256(configuration)
        rows = self._base_result_rows(arm, configuration_sha256)
        if row_mutation:
            row_mutation(rows)
        self._write_json(result_path, rows)
        invalid_count = sum(
            row.get("replay_completed") is not True
            or int(row.get("checker_failure_count") or 0) > 0
            or row.get("checker_status")
            not in {"complete", "complete_no_rules", "ok", "success", "valid_empty", "valid_with_diagnostics"}
            for row in rows
        )
        retryable_count = sum(row.get("replay_completed") is not True for row in rows)
        report_status = "complete"
        if invalid_count:
            report_status = (
                "incomplete_failures" if retryable_count else "complete_with_failures"
            )
        report = {
            "schema_version": 1,
            "report_type": "checker_only_frozen_target_binding_replay",
            "status": report_status,
            "candidate_source": "frozen_target_binding",
            "target_mapping_sha256": "8" * 64,
            "configuration": configuration,
            "configuration_sha256": configuration_sha256,
            "output": {
                "path": str(result_path),
                "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "record_count": len(rows),
            },
            "inputs": {
                "dataset": {
                    "path": str(self.dataset_path),
                    "sha256": configuration["dataset_sha256"],
                },
                "retrieval_trace": {
                    "path": str(self.retrieval_path),
                    "sha256": retrieval_sha256,
                },
                "unified_catalog": {
                    "path": str(self.catalog_path),
                    "sha256": catalog_sha256,
                },
            },
            "frozen_manifest": {
                "path": str(self.frozen_manifest_path),
                "sha256": frozen_manifest_sha256,
                "selection_projection_sha256": "4" * 64,
            },
            "llm_trace": _audit_llm_trace(trace_path),
            "statistics": {
                "total_samples": len(self.dataset),
                "processed_samples": len(rows),
                "pending_samples": 0,
                "failed_samples": invalid_count,
            },
        }
        self._write_json(report_path, report)
        return result_path, report_path

    def _matrix(self, *, mutations: Mapping[Tuple[str, int], Any] | None = None):
        specs = []
        for arm in ARMS:
            for repetition in REPETITIONS:
                mutation = (mutations or {}).get((arm, repetition))
                result_path, report_path = self._write_run(
                    arm,
                    repetition,
                    row_mutation=mutation,
                )
                specs.append((arm, repetition, result_path, report_path))
        return specs

    def _refresh_report_output(self, result_path: Path, report_path: Path) -> None:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["output"]["sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        self._write_json(report_path, report)

    def test_perfect_full_matrix_reports_metrics_and_candidate_acceptance(self) -> None:
        report = evaluate_mechanism_gate(
            dataset_path=self.dataset_path,
            formal=False,
            run_specs=self._matrix(),
        )
        self.assertEqual(report["dataset"]["record_count"], 5)
        self.assertEqual(report["global_common_valid"]["record_count"], 5)
        self.assertEqual(
            report["cells"]["dual_evidence::r1"]["metrics"]["true_violation"]["recall"],
            1.0,
        )
        self.assertEqual(
            report["cells"]["dual_evidence::r1"]["metrics"]["max_negative_fpr"],
            0.0,
        )
        self.assertEqual(report["triplicate_by_arm"]["legacy"]["agreement"], 1.0)
        self.assertTrue(report["candidate_acceptance"]["overall_pass"])
        self.assertEqual(
            report["candidate_acceptance"]["candidate_arm"],
            "dual_evidence_consistency",
        )
        self.assertEqual(report["dataset"]["composition"]["mechanism"]["applicable_correct"], 1)
        self.assertEqual(
            report["dataset"]["composition"]["mechanism"]["symbol_overlap_inapplicable"],
            1,
        )

    def test_failure_is_coverage_not_false_negative_or_true_negative(self) -> None:
        def fail_first(rows: List[Dict[str, Any]]) -> None:
            rows[0].update(
                {
                    "replay_completed": False,
                    "checker_status": "failed",
                    "checker_failure_count": 1,
                    "checker_failures": [{"kind": "transport_failure"}],
                    "checker_decisions": [],
                    "diagnostics": [],
                }
            )

        report = evaluate_mechanism_gate(
            dataset_path=self.dataset_path,
            formal=False,
            run_specs=self._matrix(mutations={("legacy", 1): fail_first}),
        )
        cell = report["cells"]["legacy::r1"]
        self.assertEqual(cell["coverage"], 0.8)
        self.assertEqual(cell["failure_by_stage"], {"transport_failure": 1})
        self.assertEqual(cell["metrics"]["true_violation"]["valid_count"], 0)
        self.assertIsNone(cell["metrics"]["true_violation"]["recall"])
        self.assertEqual(report["triplicate_by_arm"]["legacy"]["record_count"], 4)
        excluded = report["common_valid_by_repetition"]["1"]["excluded_records"]
        self.assertEqual(excluded[0]["typed_id"]["value"], self.dataset[0]["id"])
        self.assertEqual(excluded[0]["failures"]["legacy"]["failure_stage"], "transport_failure")

    def test_no_rule_and_target_decision_errors_are_invalid(self) -> None:
        def no_rule(rows: List[Dict[str, Any]]) -> None:
            rows[1]["checker_status"] = "complete_no_rules"
            rows[1]["checker_decisions"] = []

        def duplicate_decision(rows: List[Dict[str, Any]]) -> None:
            rows[2]["checker_decisions"].append(copy.deepcopy(rows[2]["checker_decisions"][0]))

        specs = self._matrix(
            mutations={
                ("legacy", 1): no_rule,
                ("dual_evidence", 1): duplicate_decision,
            }
        )
        report = evaluate_mechanism_gate(
            dataset_path=self.dataset_path,
            formal=False,
            run_specs=specs,
        )
        self.assertEqual(
            report["cells"]["legacy::r1"]["failure_by_stage"],
            {"target_rule_missing": 1},
        )
        self.assertEqual(
            report["cells"]["dual_evidence::r1"]["failure_by_stage"],
            {"target_decision_count_invalid": 1},
        )

    def test_self_corrected_probe_and_protocol_contradiction_are_counted(self) -> None:
        def publish_self_corrected(rows: List[Dict[str, Any]]) -> None:
            item = self.dataset[4]
            rows[4]["diagnostics"] = [
                self._diagnostic(
                    "dual_evidence_consistency",
                    item["prediction"],
                    contradictory=True,
                )
            ]
            rows[4]["candidate_diagnostics"] = copy.deepcopy(rows[4]["diagnostics"])
            rows[4]["checker_status"] = "valid_with_diagnostics"
            rows[4]["checker_decisions"][0]["status"] = "valid_with_diagnostics"
            rows[4]["checker_decisions"][0]["published_diagnostic_count"] = 1

        report = evaluate_mechanism_gate(
            dataset_path=self.dataset_path,
            formal=False,
            run_specs=self._matrix(
                mutations={("dual_evidence_consistency", 1): publish_self_corrected}
            ),
        )
        metrics = report["cells"]["dual_evidence_consistency::r1"]["metrics"]
        self.assertEqual(metrics["self_corrected"]["probe_diagnostic_count"], 1)
        self.assertEqual(metrics["protocol_contradiction_count"], 1)
        self.assertEqual(
            metrics["negative_mechanisms"]["insufficient_or_self_corrected"]["fpr"],
            1.0,
        )
        self.assertFalse(report["candidate_acceptance"]["overall_pass"])
        self.assertFalse(
            report["candidate_acceptance"]["repetitions"]["1"]["passed"]
        )
        self.assertIsNone(
            report["cells"]["legacy::r1"]["metrics"]["protocol_contradiction_count"]
        )

    def test_triplicate_agreement_detects_boolean_decision_change(self) -> None:
        def false_positive(rows: List[Dict[str, Any]]) -> None:
            item = self.dataset[1]
            rows[1]["diagnostics"] = [self._diagnostic("dual_evidence", item["prediction"].replace("remains constant", "decreases"))]
            # Use a locatable synthetic diagnostic on the modified quote by pointing at
            # the existing result only; evaluator is testing the final Boolean vector.
            rows[1]["diagnostics"][0]["evidence"] = {
                "quote": "total momentum remains constant",
                "location": {"start_char": 4, "end_char": 35},
            }
            rows[1]["candidate_diagnostics"] = copy.deepcopy(rows[1]["diagnostics"])
            rows[1]["checker_status"] = "valid_with_diagnostics"
            rows[1]["checker_decisions"][0]["status"] = "valid_with_diagnostics"
            rows[1]["checker_decisions"][0]["published_diagnostic_count"] = 1

        report = evaluate_mechanism_gate(
            dataset_path=self.dataset_path,
            formal=False,
            run_specs=self._matrix(mutations={("dual_evidence", 2): false_positive}),
        )
        agreement = report["triplicate_by_arm"]["dual_evidence"]
        self.assertEqual(agreement["record_count"], 5)
        self.assertEqual(agreement["consistent_count"], 4)
        self.assertEqual(agreement["agreement"], 0.8)
        self.assertFalse(agreement["agreement_ge_0_95"])
        self.assertTrue(report["candidate_acceptance"]["overall_pass"])

    def test_self_corrected_any_published_target_diagnostic_fails_probe_gate(self) -> None:
        def publish_on_correction(rows: List[Dict[str, Any]]) -> None:
            item = self.dataset[4]
            diagnostic = self._diagnostic(
                "dual_evidence_consistency", item["prediction"]
            )
            quote = "total momentum remains constant"
            start = item["prediction"].index(quote)
            diagnostic["evidence"] = {
                "source": "prediction",
                "quote": quote,
                "location": {
                    "start_char": start,
                    "end_char": start + len(quote),
                    "locatable_valid": True,
                },
            }
            rows[4]["diagnostics"] = [diagnostic]
            rows[4]["candidate_diagnostics"] = [copy.deepcopy(diagnostic)]
            rows[4]["checker_status"] = "valid_with_diagnostics"
            rows[4]["checker_decisions"][0]["status"] = "valid_with_diagnostics"
            rows[4]["checker_decisions"][0]["published_diagnostic_count"] = 1

        report = evaluate_mechanism_gate(
            dataset_path=self.dataset_path,
            formal=False,
            run_specs=self._matrix(
                mutations={
                    ("dual_evidence_consistency", 1): publish_on_correction
                }
            ),
        )
        record = report["cells"]["dual_evidence_consistency::r1"]["metrics"]
        self.assertEqual(record["self_corrected"]["probe_diagnostic_count"], 1)
        self.assertEqual(
            record["self_corrected"]["superseded_overlap_diagnostic_count"], 0
        )
        evaluated = report["global_common_valid"]["cell_metrics"][
            "dual_evidence_consistency::r1"
        ]
        self.assertEqual(evaluated["self_corrected"]["probe_diagnostic_count"], 1)
        self.assertFalse(report["candidate_acceptance"]["overall_pass"])

    def test_trace_content_must_match_sample_rule_mode_and_model(self) -> None:
        result_path, report_path = self._write_run("legacy", 1)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        trace_path = Path(report["configuration"]["llm_trace_path"])
        records = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records[0]["trace_meta"]["rule_id"] = "gen_ffffffffffffffff"
        trace_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
            encoding="utf-8",
        )
        report["llm_trace"] = _audit_llm_trace(trace_path)
        self._write_json(report_path, report)
        with self.assertRaisesRegex(MechanismEvaluationError, "wrong target rule"):
            evaluate_mechanism_gate(
                dataset_path=self.dataset_path,
                formal=False,
                run_specs=[("legacy", 1, result_path, report_path)],
                require_full_matrix=False,
            )

    def test_trace_attempt_sequence_and_status_must_match_decision(self) -> None:
        result_path, report_path = self._write_run("dual_evidence", 1)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        trace_path = Path(report["configuration"]["llm_trace_path"])
        records = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records[0]["parse_status"] = "schema_failure"
        trace_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
            encoding="utf-8",
        )
        report["llm_trace"] = _audit_llm_trace(trace_path)
        self._write_json(report_path, report)
        with self.assertRaisesRegex(MechanismEvaluationError, "trace status contradicts"):
            evaluate_mechanism_gate(
                dataset_path=self.dataset_path,
                formal=False,
                run_specs=[("dual_evidence", 1, result_path, report_path)],
                require_full_matrix=False,
            )

        result_path, report_path = self._write_run("dual_evidence", 2)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        trace_path = Path(report["configuration"]["llm_trace_path"])
        records = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records.append(copy.deepcopy(records[0]))
        trace_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
            encoding="utf-8",
        )
        report["llm_trace"] = _audit_llm_trace(trace_path)
        self._write_json(report_path, report)
        with self.assertRaisesRegex(MechanismEvaluationError, "duplicate attempt"):
            evaluate_mechanism_gate(
                dataset_path=self.dataset_path,
                formal=False,
                run_specs=[("dual_evidence", 2, result_path, report_path)],
                require_full_matrix=False,
            )

    def test_formal_evaluator_source_identity_must_be_clean_and_frozen(self) -> None:
        source = {
            "git_available": True,
            "git_head": "a" * 40,
            "git_branch": "codex/test",
            "git_dirty": False,
            "git_status_sha256": "b" * 64,
            "git_tracked_diff_sha256": "c" * 64,
            "source_tree_sha256": "d" * 64,
            "source_file_count": 12,
        }
        self.assertEqual(
            _validate_formal_evaluator_source_identity(source, copy.deepcopy(source)),
            source,
        )
        dirty = {**source, "git_dirty": True}
        with self.assertRaisesRegex(MechanismEvaluationError, "clean Git"):
            _validate_formal_evaluator_source_identity(dirty, source)
        drifted_generation = {**source, "source_tree_sha256": "e" * 64}
        with self.assertRaisesRegex(MechanismEvaluationError, "differs"):
            _validate_formal_evaluator_source_identity(source, drifted_generation)

    def test_formal_matrix_cache_and_identity_checks_fail_closed(self) -> None:
        with self.assertRaisesRegex(MechanismEvaluationError, "generation manifest"):
            evaluate_mechanism_gate(
                dataset_path=self.dataset_path,
                formal=True,
                run_specs=self._matrix(),
            )

        one = self._write_run("legacy", 1)
        with self.assertRaisesRegex(MechanismEvaluationError, "3x3 matrix"):
            evaluate_mechanism_gate(
                dataset_path=self.dataset_path,
                formal=False,
                run_specs=[("legacy", 1, *one)],
            )

        inconsistent_result, inconsistent_report = self._write_run("dual_evidence", 1)
        inconsistent_rows = json.loads(
            inconsistent_result.read_text(encoding="utf-8")
        )
        inconsistent_rows[0].update(
            {
                "replay_completed": False,
                "checker_status": "failed",
                "checker_failure_count": 1,
                "checker_failures": [{"kind": "transport_failure"}],
            }
        )
        self._write_json(inconsistent_result, inconsistent_rows)
        self._refresh_report_output(inconsistent_result, inconsistent_report)
        with self.assertRaisesRegex(MechanismEvaluationError, "failed_samples"):
            evaluate_mechanism_gate(
                dataset_path=self.dataset_path,
                formal=False,
                run_specs=[
                    (
                        "dual_evidence",
                        1,
                        inconsistent_result,
                        inconsistent_report,
                    )
                ],
                require_full_matrix=False,
            )

        bad_root = self.root / "bad-cache"
        bad_root.mkdir()
        original_root = self.root
        try:
            self.root = bad_root
            self.dataset_path = bad_root / "dataset.json"
            self._write_json(self.dataset_path, self.dataset)
            result_path, report_path = self._write_run(
                "legacy",
                1,
                configuration_mutation=lambda config: config.update(
                    {"checker_cache_enabled": True}
                ),
            )
            with self.assertRaisesRegex(MechanismEvaluationError, "disable.*Checker cache"):
                evaluate_mechanism_gate(
                    dataset_path=self.dataset_path,
                    formal=False,
                    run_specs=[("legacy", 1, result_path, report_path)],
                    require_full_matrix=False,
                )
        finally:
            self.root = original_root
            self.dataset_path = original_root / "dataset.json"

    def test_dataset_span_and_expected_schema_fail_closed(self) -> None:
        invalid = copy.deepcopy(self.dataset)
        invalid[0]["gt_evidence"]["violation_spans"][0]["end_char"] += 1
        invalid_path = self.root / "invalid.json"
        self._write_json(invalid_path, invalid)
        with self.assertRaisesRegex(MechanismEvaluationError, "exact source span"):
            evaluate_mechanism_gate(
                dataset_path=invalid_path,
                formal=False,
                run_specs=[],
                require_full_matrix=False,
            )

        contradictory = copy.deepcopy(self.dataset)
        contradictory[1]["expected"]["applicability"] = False
        contradictory_path = self.root / "contradictory.json"
        self._write_json(contradictory_path, contradictory)
        with self.assertRaisesRegex(MechanismEvaluationError, "contradicts mechanism/subtype"):
            evaluate_mechanism_gate(
                dataset_path=contradictory_path,
                formal=False,
                run_specs=[],
                require_full_matrix=False,
            )


if __name__ == "__main__":
    unittest.main()
