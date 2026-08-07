from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from core.physics_rule_verifier import PhysicsRuleVerifier
from scripts import run_verifier


def _catalog() -> dict:
    return {
        "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
        "domains": [
            {
                "name": "Mechanics",
                "topics": [
                    {
                        "name": "Orbital Mechanics",
                        "summary": "Satellite motion and orbital energy.",
                        "rules": [
                            {
                                "rule_id": "orbit_energy",
                                "title": "Orbital energy consistency",
                                "trigger": "satellite orbit",
                                "check_logic": "Check the orbital-energy relation.",
                            }
                        ],
                        "scenario_clusters": [
                            {
                                "id": "orbit_decay",
                                "name": "Orbit decay",
                                "summary": "Slow orbital decay.",
                                "rule_ids": ["orbit_energy"],
                            }
                        ],
                    }
                ],
            }
        ],
    }


class _PartialFailureMatcher:
    available = True

    def __init__(self) -> None:
        self.json_retries = 0
        self.last_trace = {}
        self.last_partial_result = {}

    def select_tree_semantically(self, sample: dict, catalog: dict) -> dict:
        topic = catalog["domains"][0]["topics"][0]
        self.last_partial_result = {
            "input_policy": "background_navigation_prediction_rule_only",
            "background_analysis": {"task_focus": "satellite orbit decay"},
            "domain_judgments": [
                {"domain": "Mechanics", "score": 0.98, "reason": "orbital motion"}
            ],
            "selected_domains": ["Mechanics"],
            "topic_judgments": [],
            "selected_topics": [
                {
                    "domain": "Mechanics",
                    "topic": "Orbital Mechanics",
                    "score": 0.96,
                    "reason": "satellite orbit",
                    "topic_obj": topic,
                }
            ],
            "cluster_judgments": [],
            "selected_clusters": [],
            "rule_judgments": [],
            "selected_rules": [],
        }
        self.last_trace = {
            "input_policy": "background_navigation_prediction_rule_only",
            "background_analysis": {"task_focus": "satellite orbit decay"},
            "stages": {
                "domain": {"accepted": [{"domain": "Mechanics", "score": 0.98}]},
                "topic": {
                    "accepted": [
                        {
                            "domain": "Mechanics",
                            "topic": "Orbital Mechanics",
                            "score": 0.96,
                        }
                    ]
                },
                "cluster": {"accepted": [], "empty_reason": "invalid_json_response"},
            },
            "terminal_stage": "cluster",
            "empty_reason": "invalid_json_response",
            "status": "error",
        }
        exc = RuntimeError("cluster response was not valid JSON")
        exc.stage = "cluster"
        exc.trace = self.last_trace
        exc.partial_result = self.last_partial_result
        raise exc


class _EmptyTraceMatcher:
    available = True

    def select_tree_semantically(self, sample: dict, catalog: dict) -> dict:
        return {
            "input_policy": "background_navigation_prediction_rule_only",
            "background_analysis": {"task_focus": "classify the physical setting"},
            "navigation_trace": {
                "terminal_stage": "topic",
                "empty_reason": "no_topic_selected",
            },
            "terminal_stage": "topic",
            "empty_reason": "no_topic_selected",
            "domain_judgments": [
                {"domain": "Mechanics", "score": 0.91, "reason": "mechanical system"}
            ],
            "selected_domains": ["Mechanics"],
            "topic_judgments": [],
            "selected_topics": [],
            "cluster_judgments": [],
            "selected_clusters": [],
            "rule_judgments": [],
            "selected_rules": [],
        }


class _PartialRuleFailureMatcher:
    available = True

    def select_tree_semantically(self, sample: dict, catalog: dict) -> dict:
        topic = catalog["domains"][0]["topics"][0]
        partial_rule = {
            "rule_id": "orbit_energy",
            "title": "Orbital energy consistency",
            "domain": "Mechanics",
            "topic": "Orbital Mechanics",
            "cluster_id": "orbit_decay",
            "cluster": "Orbit decay",
            "score": 0.91,
            "reason": "first batch hit",
            "rule_obj": topic["rules"][0],
        }
        self.last_partial_result = {
            "input_policy": "background_navigation_prediction_rule_only",
            "background_analysis": {"task_focus": "satellite orbit decay"},
            "domain_judgments": [{"domain": "Mechanics", "score": 0.98, "reason": "orbit"}],
            "selected_domains": ["Mechanics"],
            "topic_judgments": [],
            "selected_topics": [
                {
                    "domain": "Mechanics",
                    "topic": "Orbital Mechanics",
                    "score": 0.96,
                    "reason": "orbit",
                    "topic_obj": topic,
                }
            ],
            "cluster_judgments": [],
            "selected_clusters": [],
            "rule_judgments": [partial_rule],
            "selected_rules": [partial_rule],
        }
        self.last_trace = {
            "input_policy": "background_navigation_prediction_rule_only",
            "background_analysis": {"task_focus": "satellite orbit decay"},
            "stages": {"rule": {"accepted": [], "empty_reason": "selection_error"}},
            "terminal_stage": "rule",
            "empty_reason": "selection_error",
            "status": "failed",
        }
        exc = RuntimeError("second rule batch failed")
        exc.stage = "rule"
        exc.trace = self.last_trace
        exc.partial_result = self.last_partial_result
        raise exc


class _SuccessfulRuleMatcher:
    available = True

    def __init__(self) -> None:
        self.last_sample: dict = {}

    def select_tree_semantically(self, sample: dict, catalog: dict) -> dict:
        self.last_sample = dict(sample)
        topic = catalog["domains"][0]["topics"][0]
        rule = topic["rules"][0]
        return {
            "input_policy": "background_navigation_prediction_rule_only",
            "background_analysis": {"task_focus": "audit orbital energy"},
            "navigation_trace": {"status": "complete", "terminal_stage": "rule"},
            "terminal_stage": "rule",
            "empty_reason": "",
            "domain_judgments": [{"domain": "Mechanics", "score": 0.98, "reason": "orbit"}],
            "selected_domains": ["Mechanics"],
            "topic_judgments": [],
            "selected_topics": [
                {
                    "domain": "Mechanics",
                    "topic": "Orbital Mechanics",
                    "score": 0.96,
                    "reason": "orbit",
                    "topic_obj": topic,
                }
            ],
            "cluster_judgments": [],
            "selected_clusters": [],
            "rule_judgments": [],
            "selected_rules": [
                {
                    "rule_id": "orbit_energy",
                    "title": "Orbital energy consistency",
                    "domain": "Mechanics",
                    "topic": "Orbital Mechanics",
                    "cluster_id": "orbit_decay",
                    "cluster": "Orbit decay",
                    "score": 0.95,
                    "reason": "auditable claim",
                    "rule_obj": rule,
                }
            ],
        }


class SemanticTraceAdapterTests(unittest.TestCase):
    def _verifier(self, root: Path, matcher: object, **kwargs: object) -> PhysicsRuleVerifier:
        catalog_path = root / "catalog.json"
        catalog_path.write_text(json.dumps(_catalog()), encoding="utf-8")
        return PhysicsRuleVerifier(
            llm_model=None,
            unified_rules_path=str(catalog_path),
            semantic_matcher=matcher,
            log_dir=str(root / "logs"),
            results_dir=str(root / "results"),
            enable_symbolic_check=False,
            **kwargs,
        )

    def test_failure_preserves_completed_levels_and_navigation_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            matcher = _PartialFailureMatcher()
            verifier = self._verifier(
                Path(temp_dir),
                matcher,
                semantic_json_attempts=4,
            )
            result = verifier.retrieve_unified_semantic_tree(
                {"id": "partial", "question": "A satellite orbit decays.", "prediction": "", "answer": ""}
            )

        self.assertEqual(matcher.json_retries, 3)
        self.assertEqual(result["selection_strategy"], "semantic_error")
        self.assertEqual(result["semantic_failed_stage"], "cluster")
        self.assertIn("not valid JSON", result["semantic_selection_error"])
        self.assertEqual(result["retrieved_domains"][0]["domain"], "Mechanics")
        self.assertEqual(result["retrieved_topics"][0]["topic"], "Orbital Mechanics")
        self.assertEqual(result["retrieved_clusters"], [])
        self.assertEqual(result["retrieved_rules"], [])
        self.assertEqual(result["background_analysis"]["task_focus"], "satellite orbit decay")
        self.assertEqual(result["navigation_trace"]["status"], "error")
        self.assertEqual(result["terminal_stage"], "cluster")
        self.assertEqual(result["empty_reason"], "invalid_json_response")
        json.dumps(result, ensure_ascii=False)

    def test_explicit_missing_unified_catalog_never_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = root / "legacy.json"
            legacy.write_text(json.dumps({"domains": []}), encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                PhysicsRuleVerifier(
                    rules_catalog_path=str(legacy),
                    unified_rules_path=str(root / "missing.json"),
                    llm_model=None,
                    enable_symbolic_check=False,
                )

    def test_legacy_mode_without_unified_catalog_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog = root / "legacy.json"
            catalog.write_text(json.dumps({"domains": []}), encoding="utf-8")
            verifier = PhysicsRuleVerifier(
                rules_catalog_path=str(catalog),
                unified_rules_path=None,
                llm_model=None,
                enable_symbolic_check=False,
                log_dir=str(root / "logs"),
                results_dir=str(root / "results"),
            )

        self.assertFalse(verifier._unified_mode)

    def test_llm_cache_can_be_disabled_for_controlled_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            verifier = self._verifier(
                Path(temp_dir),
                _EmptyTraceMatcher(),
                enable_llm_cache=False,
            )

        self.assertFalse(verifier.enable_llm_cache)
        self.assertFalse(verifier.semantic_checker.enable_cache)

    def test_full_batch_exposes_per_sample_checkpoint_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            verifier = self._verifier(Path(temp_dir), _EmptyTraceMatcher())
            verifier.verify = lambda sample: {"id": sample["id"], "diagnostics": []}
            snapshots: list[list[str]] = []

            results = verifier.run_batch(
                [{"id": "first"}, {"id": "second"}],
                progress_interval=0,
                on_result=lambda rows: snapshots.append([str(row["id"]) for row in rows]),
            )

        self.assertEqual(["first", "second"], [row["id"] for row in results])
        self.assertEqual([["first"], ["first", "second"]], snapshots)

    def test_empty_tree_exposes_reason_and_is_not_a_rule_hit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            verifier = self._verifier(Path(temp_dir), _EmptyTraceMatcher())
            result = verifier.retrieve_unified_semantic_tree(
                {"id": "empty", "question": "A physical problem.", "prediction": "", "answer": ""}
            )

        self.assertEqual(result["selection_strategy"], "semantic_tree_empty")
        self.assertEqual(result["retrieved_rules"], [])
        self.assertEqual(result["terminal_stage"], "topic")
        self.assertEqual(result["empty_reason"], "no_topic_selected")
        self.assertEqual(result["background_analysis"]["task_focus"], "classify the physical setting")

    def test_partial_rule_is_visible_but_not_executed_after_semantic_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            verifier = self._verifier(Path(temp_dir), _PartialRuleFailureMatcher())

            def unexpected_checker(_sample: dict) -> dict:
                raise AssertionError("partial semantic rules must not execute")

            verifier.semantic_checker.analyze = unexpected_checker
            result = verifier.verify(
                {
                    "id": "partial-rule",
                    "question": "A satellite orbit decays.",
                    "prediction": "A claim.",
                    "answer": "A reference answer.",
                }
            )

        self.assertEqual(result["selection_strategy"], "semantic_error")
        self.assertEqual(verifier.semantic_checker.rules_to_check, [])
        self.assertEqual(result["diagnostics"], [])
        self.assertEqual(result["retrieved_rules"][0]["rule_id"], "orbit_energy")
        self.assertTrue(result["retrieved_rules"][0]["partial"])
        self.assertFalse(result["retrieved_rules"][0]["executable"])

    def test_reference_answer_is_removed_before_checker_execution(self) -> None:
        captured_samples: list[dict] = []
        captured_generated_samples: list[dict] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            matcher = _SuccessfulRuleMatcher()
            verifier = self._verifier(Path(temp_dir), matcher)

            class CaptureExperienceEngine:
                available = True

                def run_rule(self, _rule_id: str, check_sample: dict) -> None:
                    captured_generated_samples.append(dict(check_sample))
                    return None

            verifier.enable_symbolic_check = True
            verifier.experience_code_engine = CaptureExperienceEngine()

            def capture_checker(check_sample: dict) -> dict:
                captured_samples.append(dict(check_sample))
                return {"diagnostics": []}

            verifier.semantic_checker.analyze = capture_checker
            result = verifier.verify(
                {
                    "id": "answer-boundary",
                    "question": "A satellite orbit decays.",
                    "prediction": "The final value is correct but this reasoning may be wrong.",
                    "answer": "secret reference answer",
                    "context": "",
                }
            )

        self.assertEqual(result["selection_strategy"], "semantic_tree_selection")
        self.assertNotIn("answer", matcher.last_sample)
        self.assertEqual(len(captured_samples), 1)
        self.assertNotIn("answer", captured_samples[0])
        self.assertEqual(len(captured_generated_samples), 1)
        self.assertNotIn("answer", captured_generated_samples[0])

    def test_verifier_passes_output_adapter_to_constructed_matcher(self) -> None:
        captured: dict[str, object] = {}

        class CapturingMatcher:
            available = True

            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(_catalog()), encoding="utf-8")
            with patch(
                "core.physics_rule_verifier.UnifiedSemanticMatcher",
                CapturingMatcher,
            ):
                verifier = PhysicsRuleVerifier(
                    llm_model=None,
                    unified_rules_path=str(catalog_path),
                    unified_retrieval_mode="semantic",
                    semantic_output_adapter="forced_tool_call",
                    enable_symbolic_check=False,
                    log_dir=str(root / "logs"),
                    results_dir=str(root / "results"),
                )

        self.assertIsInstance(verifier.semantic_matcher, CapturingMatcher)
        self.assertEqual(captured["structured_output_adapter"], "forced_tool_call")


class SemanticCliControlTests(unittest.TestCase):
    def _run_cli(
        self,
        root: Path,
        *,
        continue_after_error: bool,
        mode: str = "error_then_empty",
    ) -> tuple[list[dict], int, str, dict]:
        input_path = root / "samples.json"
        output_path = root / ("continued.json" if continue_after_error else "fail_fast.json")
        (root / "catalog.json").write_text(json.dumps(_catalog()), encoding="utf-8")
        if mode == "success":
            samples = [{"id": "hit", "question": "q", "prediction": "claim", "answer": "reference"}]
        elif mode in {"empty_only", "unavailable"}:
            samples = [{"id": "empty", "question": "q", "prediction": "", "answer": ""}]
        else:
            samples = [
                {"id": "bad", "question": "q1", "prediction": "", "answer": ""},
                {"id": "empty", "question": "q2", "prediction": "", "answer": ""},
            ]
        input_path.write_text(
            json.dumps(samples),
            encoding="utf-8",
        )

        created: list[object] = []

        class FakeVerifier:
            def __init__(self, **kwargs: object) -> None:
                self._unified_v2_mode = True
                self.semantic_matcher = type(
                    "AvailableMatcher",
                    (),
                    {"available": mode != "unavailable"},
                )()
                self.calls: list[str] = []
                self.kwargs = kwargs
                created.append(self)

            def retrieve_unified_semantic_tree(self, sample: dict) -> dict:
                sample_id = str(sample.get("id"))
                self.calls.append(sample_id)
                common = {
                    "id": sample_id,
                    "retrieved_domains": [],
                    "retrieved_topics": [],
                    "retrieved_clusters": [],
                    "retrieved_rules": [],
                }
                if sample_id == "bad":
                    return {
                        **common,
                        "selection_strategy": "semantic_error",
                        "semantic_failed_stage": "cluster",
                        "semantic_selection_error": "invalid JSON",
                        "terminal_stage": "cluster",
                        "empty_reason": "invalid_json_response",
                    }
                if sample_id == "hit":
                    return {
                        **common,
                        "selection_strategy": "semantic_tree_selection",
                        "semantic_failed_stage": "",
                        "semantic_selection_error": "",
                        "terminal_stage": "rule",
                        "empty_reason": "",
                        "navigation_trace": {
                            "status": "complete",
                            "stages": {
                                "rule": {
                                    "candidates": [{"rule_id": "orbit_energy", "batch_index": 1}],
                                    "accepted": [{"rule_id": "orbit_energy"}],
                                }
                            },
                        },
                        "retrieved_rules": [
                            {
                                "rule_id": "orbit_energy",
                                "title": "Orbital energy consistency",
                                "score": 0.94,
                            }
                        ],
                    }
                return {
                    **common,
                    "selection_strategy": "semantic_tree_empty",
                    "semantic_failed_stage": "",
                    "semantic_selection_error": "",
                    "terminal_stage": "topic",
                    "empty_reason": "no_topic_selected",
                }

        argv = [
            "run_verifier.py",
            "--retrieval-only",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--unified-catalog",
            str(root / "catalog.json"),
            "--unified-retrieval-mode",
            "semantic",
            "--semantic-json-attempts",
            "4",
            "--semantic-output-adapter",
            "forced_tool_call",
            "--no-llm-cache",
            "--progress-interval",
            "0",
        ]
        if continue_after_error:
            argv.append("--continue-on-semantic-error")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(sys, "argv", argv),
            patch.object(run_verifier, "PhysicsRuleVerifier", FakeVerifier),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = 0
            try:
                run_verifier.main()
            except SystemExit as exc:
                exit_code = int(exc.code)

        payload = (
            json.loads(output_path.read_text(encoding="utf-8"))
            if output_path.exists()
            else []
        )
        instance = created[0]
        combined_output = stdout.getvalue() + stderr.getvalue()
        return payload, exit_code, combined_output, {
            "calls": instance.calls,
            "kwargs": instance.kwargs,
        }

    def test_default_is_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload, exit_code, output, state = self._run_cli(
                Path(temp_dir),
                continue_after_error=False,
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(state["calls"], ["bad"])
        self.assertEqual(len(payload), 1)
        self.assertIn("errors=1", output)

    def test_continue_mode_finishes_batch_but_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload, exit_code, output, state = self._run_cli(
                Path(temp_dir),
                continue_after_error=True,
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(state["calls"], ["bad", "empty"])
        self.assertEqual(len(payload), 2)
        self.assertEqual(state["kwargs"]["semantic_json_attempts"], 4)
        self.assertEqual(state["kwargs"]["semantic_output_adapter"], "forced_tool_call")
        self.assertFalse(state["kwargs"]["enable_llm_cache"])
        self.assertIn("empty_without_rules=1", output)
        self.assertIn("semantic_tree_empty is not a successful rule hit", output)

    def test_success_saves_rules_and_trace_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload, exit_code, output, state = self._run_cli(
                Path(temp_dir),
                continue_after_error=False,
                mode="success",
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(state["calls"], ["hit"])
        self.assertEqual(payload[0]["retrieved_rules"][0]["rule_id"], "orbit_energy")
        self.assertEqual(payload[0]["navigation_trace"]["status"], "complete")
        self.assertIn("rule_hit_samples=1", output)

    def test_all_empty_batch_exits_three(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload, exit_code, output, _state = self._run_cli(
                Path(temp_dir),
                continue_after_error=True,
                mode="empty_only",
            )

        self.assertEqual(len(payload), 1)
        self.assertEqual(exit_code, 3)
        self.assertIn("without any rule hit", output)

    def test_unavailable_matcher_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload, exit_code, output, state = self._run_cli(
                Path(temp_dir),
                continue_after_error=False,
                mode="unavailable",
            )

        self.assertEqual(payload, [])
        self.assertEqual(exit_code, 2)
        self.assertEqual(state["calls"], [])
        self.assertIn("required but unavailable", output)


if __name__ == "__main__":
    unittest.main()
