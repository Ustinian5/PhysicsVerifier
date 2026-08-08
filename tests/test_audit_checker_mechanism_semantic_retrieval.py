from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from core.unified_semantic_matcher import UnifiedSemanticMatcher
from scripts.audit_checker_mechanism_semantic_retrieval import (
    CLAIM_BOUNDARY,
    SemanticRetrievalAuditError,
    _object_sha256,
    audit_semantic_retrieval,
)


TARGET_RULE_ID = "gen_0000000000000001"
WRONG_RULE_ID = "gen_0000000000000002"


class SemanticRetrievalAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.catalog = self._catalog()
        self.dataset = self._dataset()
        self.traces = self._traces()
        self.catalog_path = self.root / "catalog.json"
        self.dataset_path = self.root / "dataset.json"
        self.trace_path = self.root / "trace.json"
        self._write(self.catalog_path, self.catalog)
        self._write(self.dataset_path, self.dataset)
        self._write(self.trace_path, self.traces)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def _write(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _rule(rule_id: str, title: str, trigger: str) -> Dict[str, Any]:
        return {
            "rule_id": rule_id,
            "title": title,
            "summary": title,
            "trigger": trigger,
            "check_logic": "Check conservation of total momentum.",
            "error_type": "physics",
            "symbolic_hint": {
                "primitive": "equation_equivalence",
                "canonical": "p_i=p_f",
                "required_symbols": ["p"],
            },
        }

    @classmethod
    def _catalog(cls) -> Dict[str, Any]:
        return {
            "metadata": {
                "catalog_type": "unified_rules_v2",
                "total_domains": 1,
                "total_topics": 1,
                "topics_with_rules": 1,
                "total_scenario_clusters": 1,
                "total_executable_rules": 2,
            },
            "domains": [
                {
                    "id": "mechanics",
                    "name": "Mechanics",
                    "topics": [
                        {
                            "id": "mechanics.momentum",
                            "name": "Momentum",
                            "rules": [
                                cls._rule(TARGET_RULE_ID, "Momentum conservation", "A"),
                                cls._rule(
                                    WRONG_RULE_ID,
                                    "Pressure symbol disambiguation",
                                    "A much longer trigger for pressure notation",
                                ),
                            ],
                            "scenario_clusters": [
                                {
                                    "id": "momentum_checks",
                                    "name": "Momentum checks",
                                    "summary": "Momentum checks.",
                                    "rule_ids": [TARGET_RULE_ID, WRONG_RULE_ID],
                                    "rule_groups": [],
                                }
                            ],
                        }
                    ],
                }
            ],
        }

    @staticmethod
    def _span(source: str, text: str, quote: str) -> Dict[str, Any]:
        start = text.index(quote)
        return {
            "source": source,
            "quote": quote,
            "start_char": start,
            "end_char": start + len(quote),
        }

    def _case(self, mechanism: str, *, subtype: str = "none") -> Dict[str, Any]:
        question = "An isolated two-cart system undergoes a short collision."
        context = "The net external impulse is negligible during the collision."
        predictions = {
            "true_violation": "The total momentum decreases because the collision is inelastic.",
            "applicable_correct": (
                "The total momentum remains constant although kinetic energy decreases."
            ),
            "symbol_overlap_inapplicable": (
                "Here p denotes pressure in a static gas, not cart momentum."
            ),
            "equivalent_alternative": (
                "The center-of-mass velocity is constant, an equivalent momentum argument."
            ),
            "self_corrected": (
                "I first claimed that total momentum decreases. That claim is wrong; "
                "the isolated system's total momentum remains constant."
            ),
        }
        prediction_key = subtype if mechanism == "insufficient_or_self_corrected" else mechanism
        prediction = predictions[prediction_key]
        applicability = mechanism != "symbol_overlap_inapplicable"
        violation = mechanism == "true_violation"
        consistency: Any = None
        if mechanism == "true_violation":
            consistency = "confirmed_violation"
        elif mechanism == "equivalent_alternative":
            consistency = "equivalent_or_alternative"
        elif subtype == "self_corrected":
            consistency = "self_corrected"
        evidence = {
            "applicability_spans": (
                [self._span("question", question, "isolated two-cart system")]
                if applicability
                else []
            ),
            "violation_spans": (
                [self._span("prediction", prediction, "total momentum decreases")]
                if violation
                else []
            ),
            "superseded_claim_spans": (
                [self._span("prediction", prediction, "total momentum decreases")]
                if subtype == "self_corrected"
                else []
            ),
            "correction_spans": (
                [self._span("prediction", prediction, "total momentum remains constant")]
                if subtype == "self_corrected"
                else []
            ),
        }
        return {
            "schema_version": "p3_checker_mechanism_case_v1",
            "id": f"p3::{TARGET_RULE_ID}::{mechanism}",
            "question": question,
            "context": context,
            "prediction": prediction,
            "target_rule_id": TARGET_RULE_ID,
            "target_rule": {
                "rule_id": TARGET_RULE_ID,
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
                "current_violation": violation,
                "publish": violation,
                "consistency_status": consistency,
            },
            "gt_evidence": evidence,
            "case_content_sha256": _object_sha256(
                [question.strip(), context.strip(), prediction.strip()]
            ),
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
            self._case("insufficient_or_self_corrected", subtype="self_corrected"),
        ]

    @staticmethod
    def _gate(*, publishable: bool = True) -> Dict[str, Any]:
        return {
            "publishable": publishable,
            "reasons": [] if publishable else ["below_semantic_publish_score"],
            "score": 0.94,
            "semantic_score": 0.94,
            "score_kind": "semantic_0_1",
            "min_publish_score": 0.5,
            "precision_profile": "strict",
            "symbolic_policy": "optional",
            "selection_strategy": "semantic_tree_selection",
            "strong_anchor_hits": [],
            "precondition_hits": [],
            "violation_signature_hits": [],
            "negative_condition_hits": [],
            "evidence_requirement_hits": [],
            "llm_hint_only": False,
        }

    @classmethod
    def _candidate(
        cls,
        rule_id: str,
        *,
        publishable: bool = True,
        partial: bool | None = None,
        executable: bool | None = None,
    ) -> Dict[str, Any]:
        title = (
            "Momentum conservation"
            if rule_id == TARGET_RULE_ID
            else "Pressure symbol disambiguation"
        )
        candidate: Dict[str, Any] = {
            "rule_id": rule_id,
            "domain": "Mechanics",
            "topic_id": "mechanics.momentum",
            "topic": "Momentum",
            "cluster_id": "momentum_checks",
            "cluster": "Momentum checks",
            "title": title,
            "scope": "domain",
            "score": 0.94,
            "score_kind": "semantic_0_1",
            "semantic_score": 0.94,
            "grounding_score": 0.2,
            "publish_gate": cls._gate(publishable=publishable),
            "manual_override_reason": "",
            "evidence": {"semantic_reason": "synthetic"},
        }
        if partial is not None:
            candidate["partial"] = partial
        if executable is not None:
            candidate["executable"] = executable
        return candidate

    @classmethod
    def _trace(
        cls,
        sample_id: str,
        *,
        strategy: str = "semantic_tree_selection",
        candidates: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        if candidates is None:
            candidates = [cls._candidate(TARGET_RULE_ID)]
        failure = strategy in {"semantic_error", "semantic_unavailable"}
        empty = strategy == "semantic_tree_empty"
        if empty:
            candidates = []
        return {
            "id": sample_id,
            "topic": "Momentum",
            "verifier": "unified_v2_semantic_retrieval_only",
            "unified_mode": True,
            "unified_retrieval_mode": "semantic",
            "selection_strategy": strategy,
            "retrieval_score_kind": "semantic_0_1",
            "semantic_min_publish_score": 0.5,
            "semantic_selection_error": "RuntimeError: synthetic" if failure else "",
            "semantic_failed_stage": "rule" if failure else "",
            "semantic_input_policy": UnifiedSemanticMatcher.INPUT_POLICY,
            "background_analysis": {},
            "navigation_trace": {},
            "terminal_stage": "rule",
            "empty_reason": (
                "semantic_retrieval_error"
                if failure
                else ("no_rule_selected" if empty else "")
            ),
            "retrieved_domains": [
                {
                    "domain_id": "mechanics",
                    "domain": "Mechanics",
                    "score": 0.98,
                    "score_kind": "semantic_0_1",
                }
            ],
            "retrieved_topics": [
                {
                    "domain": "Mechanics",
                    "topic_id": "mechanics.momentum",
                    "topic": "Momentum",
                    "score": 0.97,
                    "score_kind": "semantic_0_1",
                }
            ],
            "retrieved_clusters": [
                {
                    "domain": "Mechanics",
                    "topic_id": "mechanics.momentum",
                    "topic": "Momentum",
                    "cluster_id": "momentum_checks",
                    "cluster": "Momentum checks",
                    "score": 0.96,
                    "score_kind": "semantic_0_1",
                }
            ],
            "retrieved_rules": candidates,
        }

    def _traces(self) -> List[Dict[str, Any]]:
        return [
            self._trace(self.dataset[0]["id"]),
            self._trace(
                self.dataset[1]["id"],
                candidates=[self._candidate(TARGET_RULE_ID, publishable=False)],
            ),
            self._trace(
                self.dataset[2]["id"],
                candidates=[self._candidate(WRONG_RULE_ID)],
            ),
            self._trace(self.dataset[3]["id"], strategy="semantic_tree_empty"),
            self._trace(
                self.dataset[4]["id"],
                strategy="semantic_error",
                candidates=[
                    self._candidate(
                        TARGET_RULE_ID,
                        partial=True,
                        executable=False,
                    )
                ],
            ),
        ]

    def test_classifies_stratifies_and_freezes_replay_ready_hit_subset(self) -> None:
        report_path = self.root / "audit.json"
        subset_path = self.root / "target_hits.json"
        subset_manifest = self.root / "target_hits.manifest.json"
        report = audit_semantic_retrieval(
            dataset_path=self.dataset_path,
            semantic_trace_path=self.trace_path,
            catalog_path=self.catalog_path,
            report_path=report_path,
            subset_dataset_path=subset_path,
            subset_manifest_path=subset_manifest,
        )

        self.assertEqual(
            report["overall"]["category_counts"],
            {
                "target_hit_executable": 1,
                "target_present_suppressed": 1,
                "wrong_only": 1,
                "empty": 1,
                "retrieval_failure": 1,
            },
        )
        self.assertEqual(report["overall"]["target_recall_denominator"], 3)
        self.assertEqual(report["overall"]["target_miss_count"], 2)
        self.assertAlmostEqual(report["overall"]["target_hit_executable_rate"], 1 / 3, 6)
        self.assertEqual(report["overall"]["empty_count"], 1)
        self.assertEqual(report["overall"]["retrieval_failure_count"], 1)
        self.assertIn("true_violation", report["stratified"]["mechanism"])
        self.assertIn("Mechanics", report["stratified"]["domain"])
        self.assertIn("gen", report["stratified"]["origin"])
        self.assertIn("broad_proxy", report["stratified"]["trigger_scope_proxy"])
        self.assertEqual(len(report["rule_level_summary"]), 1)
        self.assertEqual(report["claim_boundary"], CLAIM_BOUNDARY)

        subset = json.loads(subset_path.read_text(encoding="utf-8"))
        subset_trace_path = self.root / "target_hits.semantic_trace.json"
        subset_trace = json.loads(subset_trace_path.read_text(encoding="utf-8"))
        manifest = json.loads(subset_manifest.read_text(encoding="utf-8"))
        self.assertEqual([row["id"] for row in subset], [self.dataset[0]["id"]])
        self.assertEqual([row["id"] for row in subset_trace], [self.dataset[0]["id"]])
        self.assertEqual(manifest["manifest_type"], "checker_replay_frozen_retrieval")
        self.assertEqual(manifest["candidate_source"], "semantic_retrieval")
        self.assertEqual(
            manifest["retrieval_config_metadata"]["scope"],
            "secondary_semantic_retrieval_target_hit_only",
        )
        self.assertEqual(
            manifest["retrieval_config_metadata"]["ordered_typed_ids"],
            [f'str:"{self.dataset[0]["id"]}"'],
        )
        self.assertIn(
            "source_input_fingerprints", manifest["retrieval_config_metadata"]
        )
        self.assertEqual(
            report["target_hit_subset"]["ordered_typed_ids"],
            [f'str:"{self.dataset[0]["id"]}"'],
        )
        self.assertTrue(report_path.is_file())

        with self.assertRaisesRegex(SemanticRetrievalAuditError, "overwrite"):
            audit_semantic_retrieval(
                dataset_path=self.dataset_path,
                semantic_trace_path=self.trace_path,
                catalog_path=self.catalog_path,
                report_path=report_path,
            )

    def test_rejects_typed_id_order_and_every_target_binding_marker(self) -> None:
        reversed_traces = list(reversed(copy.deepcopy(self.traces)))
        self._write(self.trace_path, reversed_traces)
        with self.assertRaisesRegex(SemanticRetrievalAuditError, "typed IDs"):
            audit_semantic_retrieval(
                dataset_path=self.dataset_path,
                semantic_trace_path=self.trace_path,
                catalog_path=self.catalog_path,
            )

        for field, value in (
            ("candidate_source", "frozen_target_binding"),
            ("unified_retrieval_mode", "target_binding"),
            ("selection_strategy", "target_rule_binding"),
            ("target_rule_id", TARGET_RULE_ID),
        ):
            with self.subTest(field=field):
                traces = copy.deepcopy(self.traces)
                traces[0][field] = value
                self._write(self.trace_path, traces)
                with self.assertRaises(SemanticRetrievalAuditError):
                    audit_semantic_retrieval(
                        dataset_path=self.dataset_path,
                        semantic_trace_path=self.trace_path,
                        catalog_path=self.catalog_path,
                    )

    def test_rejects_dataset_and_trace_catalog_ownership_mismatches(self) -> None:
        bad_dataset = copy.deepcopy(self.dataset)
        bad_dataset[0]["target_rule"]["topic"] = "Foreign topic"
        self._write(self.dataset_path, bad_dataset)
        with self.assertRaisesRegex(SemanticRetrievalAuditError, "catalog-derived"):
            audit_semantic_retrieval(
                dataset_path=self.dataset_path,
                semantic_trace_path=self.trace_path,
                catalog_path=self.catalog_path,
            )

        self._write(self.dataset_path, self.dataset)
        bad_traces = copy.deepcopy(self.traces)
        bad_traces[0]["retrieved_rules"][0]["cluster"] = "Foreign cluster"
        self._write(self.trace_path, bad_traces)
        with self.assertRaisesRegex(SemanticRetrievalAuditError, "catalog"):
            audit_semantic_retrieval(
                dataset_path=self.dataset_path,
                semantic_trace_path=self.trace_path,
                catalog_path=self.catalog_path,
            )

    def test_rejects_loose_numeric_schema_and_partial_nonfailure(self) -> None:
        bad_traces = copy.deepcopy(self.traces)
        bad_traces[0]["retrieved_rules"][0]["score"] = "0.94"
        self._write(self.trace_path, bad_traces)
        with self.assertRaisesRegex(SemanticRetrievalAuditError, "JSON number"):
            audit_semantic_retrieval(
                dataset_path=self.dataset_path,
                semantic_trace_path=self.trace_path,
                catalog_path=self.catalog_path,
            )

        bad_traces = copy.deepcopy(self.traces)
        bad_traces[0]["retrieved_rules"][0]["partial"] = True
        bad_traces[0]["retrieved_rules"][0]["executable"] = False
        self._write(self.trace_path, bad_traces)
        with self.assertRaisesRegex(SemanticRetrievalAuditError, "non-executable"):
            audit_semantic_retrieval(
                dataset_path=self.dataset_path,
                semantic_trace_path=self.trace_path,
                catalog_path=self.catalog_path,
            )


if __name__ == "__main__":
    unittest.main()
