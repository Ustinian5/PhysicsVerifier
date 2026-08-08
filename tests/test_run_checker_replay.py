from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List

from scripts.run_checker_replay import (
    CANDIDATE_SOURCE_TARGET_BINDING,
    CHECKER_GATE_MODES,
    FROZEN_MANIFEST_SCHEMA_VERSION,
    ReplayValidationError,
    TARGET_BINDING_RETRIEVAL_MODE,
    TARGET_BINDING_SCORE_KIND,
    TARGET_BINDING_SELECTION_STRATEGY,
    prepare_frozen_manifest,
    run_checker_replay,
)


def _catalog() -> Dict[str, Any]:
    return {
        "metadata": {
            "version": "2.0",
            "catalog_type": "unified_rules_v2",
        },
        "domains": [
            {
                "id": "mechanics",
                "name": "Mechanics",
                "topics": [
                    {
                        "id": "mechanics.motion",
                        "name": "Motion",
                        "scenario_clusters": [
                            {
                                "id": "constant_acceleration",
                                "name": "Constant acceleration",
                                "rule_ids": ["rule_motion"],
                                "rule_groups": [
                                    {
                                        "id": "motion_checks",
                                        "rule_ids": ["rule_motion"],
                                    }
                                ],
                            }
                        ],
                        "rules": [
                            {
                                "rule_id": "rule_motion",
                                "title": "Check the velocity relation",
                                "trigger": "A constant-acceleration derivation is used.",
                                "check_logic": "Check that v = u + at is applied consistently.",
                                "source": "experience",
                                "precision_profile": "precision",
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _dataset(ids: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "id": sample_id,
            "question": f"Synthetic question {sample_id}",
            "context": "Synthetic context",
            "prediction": f"Synthetic answer {sample_id}: v=u-at",
            "answer": "REFERENCE-MUST-NOT-ENTER-CHECKER",
            "private_gt_label": "MUST-NOT-ENTER-CHECKER",
        }
        for sample_id in ids
    ]


def _target_dataset(ids: List[str]) -> List[Dict[str, Any]]:
    rows = _dataset(ids)
    for row in rows:
        row.update(
            {
                "target_rule_id": "rule_motion",
                "target_rule": {"rule_id": "rule_motion"},
                "mechanism_class": "synthetic_violation",
                "private_gt": {"applicable": True, "violation": True},
            }
        )
    return rows


def _trace(sample_id: str, *, with_rule: bool = True) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "id": sample_id,
        "topic": "Motion" if with_rule else None,
        "verifier": "unified_v2_semantic_retrieval_only",
        "unified_retrieval_mode": "semantic",
        "selection_strategy": (
            "semantic_tree_selection" if with_rule else "semantic_tree_empty"
        ),
        "retrieval_score_kind": "semantic_0_1",
        "semantic_selection_error": "",
        "semantic_failed_stage": "",
        "terminal_stage": "rule",
        "empty_reason": "" if with_rule else "no_rule_selected",
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
                "topic_id": "mechanics.motion",
                "topic": "Motion",
                "score": 0.97,
                "score_kind": "semantic_0_1",
            }
        ]
        if with_rule
        else [],
        "retrieved_clusters": [
            {
                "domain": "Mechanics",
                "topic_id": "mechanics.motion",
                "topic": "Motion",
                "cluster_id": "constant_acceleration",
                "cluster": "Constant acceleration",
                "score": 0.96,
                "score_kind": "semantic_0_1",
            }
        ]
        if with_rule
        else [],
        "retrieved_rules": [],
    }
    if with_rule:
        base["retrieved_rules"] = [
            {
                "rule_id": "rule_motion",
                "domain": "Mechanics",
                "topic_id": "mechanics.motion",
                "topic": "Motion",
                "cluster_id": "constant_acceleration",
                "cluster": "Constant acceleration",
                "title": "Trace title is not used as rule content",
                "scope": "domain",
                "score": 0.94,
                "score_kind": "semantic_0_1",
                "semantic_score": 0.94,
                "grounding_score": 0.2,
                "publish_gate": {
                    "publishable": True,
                    "reasons": [],
                    "score": 0.94,
                    "semantic_score": 0.94,
                    "score_kind": "semantic_0_1",
                    "min_publish_score": 0.0,
                    "selection_strategy": "semantic_tree_selection",
                },
                "evidence": {"semantic_reason": "synthetic"},
            }
        ]
    return base


def _target_trace(sample_id: str) -> Dict[str, Any]:
    trace = _trace(sample_id)
    trace.update(
        {
            "candidate_source": CANDIDATE_SOURCE_TARGET_BINDING,
            "target_rule_id": "rule_motion",
            "unified_retrieval_mode": TARGET_BINDING_RETRIEVAL_MODE,
            "selection_strategy": TARGET_BINDING_SELECTION_STRATEGY,
            "retrieval_score_kind": TARGET_BINDING_SCORE_KIND,
        }
    )
    for collection in (
        "retrieved_domains",
        "retrieved_topics",
        "retrieved_clusters",
    ):
        for item in trace[collection]:
            item["score"] = 1.0
            item["score_kind"] = TARGET_BINDING_SCORE_KIND
    selected = trace["retrieved_rules"][0]
    selected.update(
        {
            "score": 1.0,
            "score_kind": TARGET_BINDING_SCORE_KIND,
            "partial": False,
            "executable": True,
        }
    )
    selected.pop("semantic_score", None)
    selected.pop("grounding_score", None)
    selected["publish_gate"] = {
        "publishable": True,
        "reasons": [],
        "score": 1.0,
        "score_kind": TARGET_BINDING_SCORE_KIND,
        "min_publish_score": 0.0,
        "selection_strategy": TARGET_BINDING_SELECTION_STRATEGY,
    }
    return trace


def _diagnostic() -> Dict[str, Any]:
    quote = "v=u-at"
    return {
        "rule": "rule_motion",
        "severity": "error",
        "message": "The acceleration sign is inconsistent.",
        "evidence": {
            "quote": quote,
            "location": {
                "locatable_valid": True,
                "start_char": 21,
                "end_char": 27,
                "paragraph_index": 0,
            },
        },
    }


class _FakeChecker:
    def __init__(
        self,
        mode: str,
        handler: Callable[[Dict[str, Any], "_FakeChecker"], Dict[str, Any]],
    ) -> None:
        self.checker_mode = mode
        self.handler = handler
        self.rules_to_check: List[str] = []
        self.rule_translations: Dict[str, Any] = {}
        self.samples: List[Dict[str, Any]] = []
        self.translation_snapshots: List[Dict[str, Any]] = []
        self.llm_temperature = 0.0
        self.llm_max_output_tokens = 0
        self.checker_min_confidence = 0.8
        self.llm_trace_path = ""
        self.llm_trace_include_prompts = False

    def analyze(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        self.samples.append(dict(sample))
        self.translation_snapshots.append(json.loads(json.dumps(self.rule_translations)))
        result = self.handler(sample, self)
        if self.llm_trace_path:
            trace_record = {
                "model": "synthetic-qwen30b",
                "trace_meta": {"sample_id": sample.get("id")},
                "raw_response": json.dumps(result.get("diagnostics") or []),
                "parse_status": (
                    "parse_failed" if result.get("checker_failures") else "valid_object"
                ),
            }
            with Path(self.llm_trace_path).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace_record) + "\n")
        return result


class _FakeVerifier:
    def __init__(
        self,
        kwargs: Dict[str, Any],
        handler: Callable[[Dict[str, Any], _FakeChecker], Dict[str, Any]],
    ) -> None:
        self.kwargs = kwargs
        self.checker_gate_mode = kwargs["checker_gate_mode"]
        self.enable_symbolic_check = kwargs["enable_symbolic_check"]
        self.semantic_matcher = None
        self.semantic_checker = _FakeChecker(self.checker_gate_mode, handler)
        self.filter_calls = 0
        self.release_calls = 0

    def verify(self, _sample: Dict[str, Any]) -> Dict[str, Any]:
        raise AssertionError("verify() must never be called by Checker replay")

    def retrieve_unified_semantic_tree(self, _sample: Dict[str, Any]) -> Dict[str, Any]:
        raise AssertionError("retrieval must never be called by Checker replay")

    def _retrieve_unified_v2_semantic_tree(self, _sample: Dict[str, Any]) -> Dict[str, Any]:
        raise AssertionError("private retrieval must never be called by Checker replay")

    def _filter_low_confidence_unified_diagnostics(
        self,
        diagnostics: List[Dict[str, Any]],
        _records: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        self.filter_calls += 1
        return list(diagnostics), []

    def _apply_diagnostic_release_gate(
        self,
        diagnostics: List[Dict[str, Any]],
        _records: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        self.release_calls += 1
        published = []
        for diagnostic in diagnostics:
            enriched = dict(diagnostic)
            enriched["release_gate"] = {"publishable": True, "reasons": []}
            published.append(enriched)
        return published, []


class CheckerReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset_path = self.root / "synthetic_dataset.json"
        self.trace_path = self.root / "frozen_retrieval.json"
        self.catalog_path = self.root / "catalog.json"
        self.manifest_path = self.root / "frozen_manifest.json"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def _write(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def _freeze(self, ids: List[str], *, empty_last: bool = False) -> Dict[str, Any]:
        self._write(self.dataset_path, _dataset(ids))
        traces = [
            _trace(sample_id, with_rule=not (empty_last and index == len(ids) - 1))
            for index, sample_id in enumerate(ids)
        ]
        self._write(self.trace_path, traces)
        self._write(self.catalog_path, _catalog())
        return prepare_frozen_manifest(
            dataset_path=self.dataset_path,
            frozen_retrieval_path=self.trace_path,
            catalog_path=self.catalog_path,
            frozen_manifest_path=self.manifest_path,
            prompt_metadata={},
            retrieval_config_metadata={},
        )

    def _factory(
        self,
        handler: Callable[[Dict[str, Any], _FakeChecker], Dict[str, Any]],
        created: List[_FakeVerifier],
    ) -> Callable[..., _FakeVerifier]:
        def factory(**kwargs: Any) -> _FakeVerifier:
            verifier = _FakeVerifier(kwargs, handler)
            created.append(verifier)
            return verifier

        return factory

    def test_manifest_and_replay_are_frozen_checker_only(self) -> None:
        manifest = self._freeze(["s1", "s2"], empty_last=True)
        self.assertEqual(
            manifest["schema_version"], FROZEN_MANIFEST_SCHEMA_VERSION
        )
        self.assertEqual(manifest["candidate_source"], "semantic_retrieval")
        self.assertEqual(manifest["ordered_ids"], ["s1", "s2"])
        self.assertTrue(manifest["selection_projection"]["records_sha256"])
        self.assertEqual(manifest["prompt_metadata"], {})
        self.assertEqual(manifest["retrieval_config_metadata"], {})

        created: List[_FakeVerifier] = []

        def successful(sample: Dict[str, Any], checker: _FakeChecker) -> Dict[str, Any]:
            self.assertNotIn("answer", sample)
            self.assertNotIn("private_gt_label", sample)
            self.assertEqual(checker.rules_to_check, ["rule_motion"])
            return {
                "checker_mode": checker.checker_mode,
                "checker_status": "complete",
                "checker_decisions": [{"rule_id": "rule_motion", "decision": "violation"}],
                "checker_failures": [],
                "checker_suppressed": [],
                "diagnostics": [_diagnostic()],
            }

        output = self.root / "replay.json"
        report_path = self.root / "replay.report.json"
        report = run_checker_replay(
            dataset_path=self.dataset_path,
            frozen_retrieval_path=self.trace_path,
            catalog_path=self.catalog_path,
            frozen_manifest_path=self.manifest_path,
            output_path=output,
            report_path=report_path,
            model="synthetic-qwen30b",
            mode="legacy",
            verifier_factory=self._factory(successful, created),
        )

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["statistics"]["checker_attempted_samples"], 1)
        self.assertEqual(report["statistics"]["checker_success_samples"], 1)
        self.assertEqual(report["statistics"]["no_executable_rule_samples"], 1)
        self.assertEqual(report["configuration"]["checker_min_confidence"], 0.8)
        self.assertEqual(
            report["configuration"]["frozen_manifest_sha256"],
            report["frozen_manifest"]["sha256"],
        )
        self.assertEqual(report["llm_trace"]["record_count"], 1)
        self.assertEqual(report["llm_trace"]["raw_response_record_count"], 1)
        self.assertFalse(report["llm_trace"]["prompts_included"])
        self.assertEqual(report["inputs"]["dataset"]["sha256"], manifest["inputs"]["dataset"]["sha256"])
        self.assertEqual(len(created), 1)
        verifier = created[0]
        self.assertEqual(verifier.kwargs["unified_retrieval_mode"], "lexical")
        self.assertFalse(verifier.kwargs["enable_symbolic_check"])
        self.assertFalse(verifier.kwargs["enable_llm_cache"])
        self.assertEqual(verifier.filter_calls, 1)
        self.assertEqual(verifier.release_calls, 1)
        srd = verifier.semantic_checker.translation_snapshots[0]["rule_motion"]["srd"]
        self.assertIn("A constant-acceleration derivation is used", srd)

        rows = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual([row["id"] for row in rows], ["s1", "s2"])
        self.assertEqual(len(rows[0]["diagnostics"]), 1)
        self.assertEqual(rows[0]["symbolic_post_diagnostics"], [])
        self.assertEqual(rows[0]["checker_min_confidence"], 0.8)
        self.assertEqual(rows[0]["checker_failure_count"], 0)
        self.assertEqual(rows[0]["experience_post_diagnostics"], [])
        self.assertEqual(rows[0]["experience_code_post_diagnostics"], [])
        self.assertFalse(rows[0]["symbolic_check"]["enabled"])

    def test_all_three_modes_are_independent_system_arms(self) -> None:
        self._freeze(["s1"])

        for mode in CHECKER_GATE_MODES:
            with self.subTest(mode=mode):
                created: List[_FakeVerifier] = []

                def successful(_sample: Dict[str, Any], checker: _FakeChecker) -> Dict[str, Any]:
                    return {
                        "checker_mode": checker.checker_mode,
                        "checker_status": "complete",
                        "checker_decisions": [
                            {"rule_id": "rule_motion", "status": "valid_empty"}
                        ],
                        "checker_failures": [],
                        "checker_suppressed": [],
                        "diagnostics": [],
                    }

                report = run_checker_replay(
                    dataset_path=self.dataset_path,
                    frozen_retrieval_path=self.trace_path,
                    catalog_path=self.catalog_path,
                    frozen_manifest_path=self.manifest_path,
                    output_path=self.root / f"{mode}.json",
                    report_path=self.root / f"{mode}.report.json",
                    model="synthetic-qwen30b",
                    mode=mode,
                    verifier_factory=self._factory(successful, created),
                )
                self.assertEqual(report["configuration"]["system_arm"], mode)
                self.assertIn("not a shared-candidate", report["system_arm_interpretation"])
                self.assertEqual(created[0].kwargs["checker_gate_mode"], mode)

    def test_legacy_v1_semantic_manifest_remains_replayable(self) -> None:
        manifest = self._freeze(["s1"])
        legacy_manifest = copy.deepcopy(manifest)
        legacy_manifest["schema_version"] = 1
        legacy_manifest.pop("candidate_source", None)
        legacy_manifest["inputs"]["retrieval_trace"].pop("candidate_source", None)
        for key in (
            "candidate_source",
            "target_mapping_count",
            "target_mapping_sha256",
            "target_mapping",
        ):
            legacy_manifest["selection_projection"].pop(key, None)
        self._write(self.manifest_path, legacy_manifest)
        created: List[_FakeVerifier] = []

        def successful(_sample: Dict[str, Any], checker: _FakeChecker) -> Dict[str, Any]:
            return {
                "checker_mode": checker.checker_mode,
                "checker_status": "valid_empty",
                "checker_decisions": [
                    {"rule_id": "rule_motion", "status": "valid_empty"}
                ],
                "checker_failures": [],
                "checker_suppressed": [],
                "diagnostics": [],
            }

        report = run_checker_replay(
            dataset_path=self.dataset_path,
            frozen_retrieval_path=self.trace_path,
            catalog_path=self.catalog_path,
            frozen_manifest_path=self.manifest_path,
            output_path=self.root / "legacy-v1.results.json",
            report_path=self.root / "legacy-v1.report.json",
            model="synthetic-qwen30b",
            verifier_factory=self._factory(successful, created),
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            report["configuration"]["candidate_source"], "semantic_retrieval"
        )

    def test_target_binding_is_honest_frozen_candidate_arm_without_label_leakage(self) -> None:
        self._write(self.dataset_path, _target_dataset(["s1"]))
        self._write(self.trace_path, [_target_trace("s1")])
        self._write(self.catalog_path, _catalog())
        manifest = prepare_frozen_manifest(
            dataset_path=self.dataset_path,
            frozen_retrieval_path=self.trace_path,
            catalog_path=self.catalog_path,
            frozen_manifest_path=self.manifest_path,
        )
        self.assertEqual(
            manifest["manifest_type"], "checker_replay_frozen_target_binding"
        )
        self.assertEqual(
            manifest["candidate_source"], CANDIDATE_SOURCE_TARGET_BINDING
        )
        self.assertEqual(
            manifest["selection_projection"]["target_mapping"],
            [
                {
                    "id": "s1",
                    "target_rule_id": "rule_motion",
                    "selected_rule_id": "rule_motion",
                }
            ],
        )

        created: List[_FakeVerifier] = []

        def successful(sample: Dict[str, Any], checker: _FakeChecker) -> Dict[str, Any]:
            self.assertEqual(
                set(sample), {"id", "question", "context", "prediction"}
            )
            self.assertNotIn("target_rule_id", sample)
            self.assertNotIn("target_rule", sample)
            self.assertNotIn("mechanism_class", sample)
            self.assertNotIn("private_gt", sample)
            self.assertEqual(checker.rules_to_check, ["rule_motion"])
            return {
                "checker_mode": checker.checker_mode,
                "checker_status": "valid_empty",
                "checker_decisions": [
                    {"rule_id": "rule_motion", "status": "valid_empty"}
                ],
                "checker_failures": [],
                "checker_suppressed": [],
                "diagnostics": [],
            }

        output = self.root / "target.results.json"
        report = run_checker_replay(
            dataset_path=self.dataset_path,
            frozen_retrieval_path=self.trace_path,
            catalog_path=self.catalog_path,
            frozen_manifest_path=self.manifest_path,
            output_path=output,
            report_path=self.root / "target.report.json",
            model="synthetic-qwen30b",
            mode="dual_evidence",
            verifier_factory=self._factory(successful, created),
        )
        self.assertEqual(
            report["report_type"], "checker_only_frozen_target_binding_replay"
        )
        self.assertEqual(
            report["candidate_source"], CANDIDATE_SOURCE_TARGET_BINDING
        )
        self.assertEqual(
            report["configuration"]["retrieval_source"], "frozen_target_binding"
        )
        self.assertEqual(
            report["target_mapping_sha256"],
            manifest["selection_projection"]["target_mapping_sha256"],
        )
        row = json.loads(output.read_text(encoding="utf-8"))[0]
        self.assertEqual(row["selection_strategy"], TARGET_BINDING_SELECTION_STRATEGY)
        self.assertEqual(row["unified_retrieval_mode"], TARGET_BINDING_RETRIEVAL_MODE)
        self.assertEqual(row["target_rule_id"], "rule_motion")

    def test_target_binding_rejects_mismatch_multiple_rules_and_fake_scores(self) -> None:
        base_dataset = _target_dataset(["s1"])
        base_trace = _target_trace("s1")
        mutations = {
            "target_mismatch": (
                lambda dataset, trace: dataset[0].update(
                    {"target_rule_id": "different_rule"}
                ),
                "does not match",
            ),
            "multiple_candidates": (
                lambda dataset, trace: trace[0]["retrieved_rules"].append(
                    copy.deepcopy(trace[0]["retrieved_rules"][0])
                ),
                "exactly one rule",
            ),
            "fake_semantic_mode": (
                lambda dataset, trace: trace[0].update(
                    {"unified_retrieval_mode": "semantic"}
                ),
                "unified_retrieval_mode",
            ),
            "fake_semantic_score_kind": (
                lambda dataset, trace: trace[0]["retrieved_rules"][0].update(
                    {"score_kind": "semantic_0_1"}
                ),
                "fixed_control_0_1",
            ),
            "semantic_score_field": (
                lambda dataset, trace: trace[0]["retrieved_rules"][0].update(
                    {"semantic_score": 1.0}
                ),
                "not allowed for target binding",
            ),
            "numeric_string": (
                lambda dataset, trace: trace[0]["retrieved_rules"][0].update(
                    {"score": "1.0"}
                ),
                "JSON number",
            ),
            "boolean_score": (
                lambda dataset, trace: trace[0]["retrieved_topics"][0].update(
                    {"score": True}
                ),
                "JSON number",
            ),
            "publishable_false": (
                lambda dataset, trace: trace[0]["retrieved_rules"][0][
                    "publish_gate"
                ].update({"publishable": False}),
                "exactly true",
            ),
        }
        for name, (mutate, expected) in mutations.items():
            with self.subTest(name=name):
                dataset = copy.deepcopy(base_dataset)
                trace = [copy.deepcopy(base_trace)]
                mutate(dataset, trace)
                self._write(self.dataset_path, dataset)
                self._write(self.trace_path, trace)
                self._write(self.catalog_path, _catalog())
                with self.assertRaisesRegex(ReplayValidationError, expected):
                    prepare_frozen_manifest(
                        dataset_path=self.dataset_path,
                        frozen_retrieval_path=self.trace_path,
                        catalog_path=self.catalog_path,
                        frozen_manifest_path=self.root / f"{name}.manifest.json",
                    )

    def test_hash_mismatch_is_rejected_before_checker_creation(self) -> None:
        self._freeze(["s1"])
        trace = json.loads(self.trace_path.read_text(encoding="utf-8"))
        trace[0]["background_analysis"] = {"mutated_after_freeze": True}
        self._write(self.trace_path, trace)
        created: List[_FakeVerifier] = []

        with self.assertRaisesRegex(ReplayValidationError, "SHA256 mismatch"):
            run_checker_replay(
                dataset_path=self.dataset_path,
                frozen_retrieval_path=self.trace_path,
                catalog_path=self.catalog_path,
                frozen_manifest_path=self.manifest_path,
                output_path=self.root / "out.json",
                report_path=self.root / "report.json",
                model="synthetic-qwen30b",
                verifier_factory=self._factory(lambda _s, _c: {}, created),
            )
        self.assertEqual(created, [])

    def test_prepare_rejects_id_order_and_catalog_ownership_drift(self) -> None:
        self._write(self.dataset_path, _dataset(["s1", "s2"]))
        self._write(self.catalog_path, _catalog())
        self._write(self.trace_path, [_trace("s2"), _trace("s1")])
        with self.assertRaisesRegex(ReplayValidationError, "same IDs in the same order"):
            prepare_frozen_manifest(
                dataset_path=self.dataset_path,
                frozen_retrieval_path=self.trace_path,
                catalog_path=self.catalog_path,
                frozen_manifest_path=self.manifest_path,
            )

        bad_trace = _trace("s1")
        bad_trace["retrieved_rules"][0]["cluster_id"] = "foreign_cluster"
        self._write(self.dataset_path, _dataset(["s1"]))
        self._write(self.trace_path, [bad_trace])
        with self.assertRaisesRegex(ReplayValidationError, "does not own rule"):
            prepare_frozen_manifest(
                dataset_path=self.dataset_path,
                frozen_retrieval_path=self.trace_path,
                catalog_path=self.catalog_path,
                frozen_manifest_path=self.manifest_path,
            )

    def test_resume_skips_success_and_retries_checker_failure(self) -> None:
        self._freeze(["s1", "s2"])
        output = self.root / "resume.json"
        report_path = self.root / "resume.report.json"
        first_created: List[_FakeVerifier] = []

        def fail_second(sample: Dict[str, Any], checker: _FakeChecker) -> Dict[str, Any]:
            if sample["id"] == "s2":
                return {
                    "checker_mode": checker.checker_mode,
                    "checker_status": "partial_failure",
                    "checker_decisions": [
                        {"rule_id": "rule_motion", "status": "schema_failure"}
                    ],
                    "checker_failures": [{"kind": "invalid_json", "stage": "checker"}],
                    "checker_suppressed": [],
                    "diagnostics": [_diagnostic()],
                }
            return {
                "checker_mode": checker.checker_mode,
                "checker_status": "complete",
                "checker_decisions": [
                    {"rule_id": "rule_motion", "status": "valid_empty"}
                ],
                "checker_failures": [],
                "checker_suppressed": [],
                "diagnostics": [],
            }

        first = run_checker_replay(
            dataset_path=self.dataset_path,
            frozen_retrieval_path=self.trace_path,
            catalog_path=self.catalog_path,
            frozen_manifest_path=self.manifest_path,
            output_path=output,
            report_path=report_path,
            model="synthetic-qwen30b",
            mode="dual_evidence",
            verifier_factory=self._factory(fail_second, first_created),
        )
        self.assertEqual(first["status"], "incomplete_failures")
        first_rows = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(first_rows[0]["replay_completed"])
        self.assertFalse(first_rows[1]["replay_completed"])
        self.assertEqual(first_rows[1]["diagnostics"], [])
        self.assertEqual(first["statistics"]["failure_counts"]["invalid_json"], 1)

        llm_trace_path = output.with_name(f"{output.name}.llm_trace.jsonl")
        with llm_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "model": "synthetic-qwen30b",
                        "trace_meta": {"sample_id": "s2", "orphan_after_crash": True},
                        "raw_response": "{}",
                        "parse_status": "valid_object",
                    }
                )
                + "\n"
            )

        second_created: List[_FakeVerifier] = []

        def succeed(sample: Dict[str, Any], checker: _FakeChecker) -> Dict[str, Any]:
            return {
                "checker_mode": checker.checker_mode,
                "checker_status": "complete",
                "checker_decisions": [
                    {"rule_id": "rule_motion", "status": "valid_empty"}
                ],
                "checker_failures": [],
                "checker_suppressed": [],
                "diagnostics": [] if sample["id"] == "s2" else [_diagnostic()],
            }

        second = run_checker_replay(
            dataset_path=self.dataset_path,
            frozen_retrieval_path=self.trace_path,
            catalog_path=self.catalog_path,
            frozen_manifest_path=self.manifest_path,
            output_path=output,
            report_path=report_path,
            model="synthetic-qwen30b",
            mode="dual_evidence",
            resume=True,
            verifier_factory=self._factory(succeed, second_created),
        )
        self.assertEqual(second["status"], "complete")
        self.assertTrue(second["resume"]["enabled"])
        self.assertEqual(second["resume"]["reused_completed_samples"], 1)
        self.assertTrue(second["resume"]["orphan_trace_recovery"]["detected"])
        self.assertEqual(
            second["resume"]["orphan_trace_recovery"]["extra_record_count"],
            1,
        )
        self.assertEqual(
            [sample["id"] for sample in second_created[0].semantic_checker.samples],
            ["s2"],
        )
        final_rows = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(all(row["replay_completed"] for row in final_rows))

    def test_semantic_unavailable_is_terminal_nonzero_failure(self) -> None:
        self._write(self.dataset_path, _dataset(["s1"]))
        unavailable = _trace("s1", with_rule=False)
        unavailable.update(
            {
                "selection_strategy": "semantic_unavailable",
                "semantic_selection_error": "Semantic matcher is not available.",
                "semantic_failed_stage": "initialization",
                "terminal_stage": "initialization",
                "empty_reason": "semantic_matcher_unavailable",
            }
        )
        self._write(self.trace_path, [unavailable])
        self._write(self.catalog_path, _catalog())
        prepare_frozen_manifest(
            dataset_path=self.dataset_path,
            frozen_retrieval_path=self.trace_path,
            catalog_path=self.catalog_path,
            frozen_manifest_path=self.manifest_path,
        )
        created: List[_FakeVerifier] = []
        report = run_checker_replay(
            dataset_path=self.dataset_path,
            frozen_retrieval_path=self.trace_path,
            catalog_path=self.catalog_path,
            frozen_manifest_path=self.manifest_path,
            output_path=self.root / "unavailable.json",
            report_path=self.root / "unavailable.report.json",
            model="synthetic-qwen30b",
            verifier_factory=self._factory(lambda _s, _c: {}, created),
        )
        self.assertEqual(report["status"], "complete_with_failures")
        self.assertEqual(report["statistics"]["failed_samples"], 1)
        self.assertEqual(report["statistics"]["terminal_failure_samples"], 1)
        self.assertEqual(report["statistics"]["retryable_failure_samples"], 0)
        self.assertEqual(created, [])
        row = json.loads((self.root / "unavailable.json").read_text(encoding="utf-8"))[0]
        self.assertEqual(row["checker_failure_count"], 1)
        self.assertTrue(row["replay_completed"])


if __name__ == "__main__":
    unittest.main()
