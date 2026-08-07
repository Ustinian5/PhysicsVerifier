from __future__ import annotations

import unittest
from unittest.mock import patch

from core.rule_catalog_retrieval import (
    build_signal_document_frequency,
    build_topic_candidates,
    score_rule_candidate,
    score_topic_candidate,
)
from scripts.audit_rule_runtime_metadata import _load_fp_rule_counts, main as audit_main
from scripts.backfill_rule_runtime_metadata import (
    _build_reference_index,
    _build_reference_topic_index,
    _copy_reference_metadata,
    _copy_reference_topic_metadata,
    _ensure_rule_metadata,
    _ensure_topic_metadata,
    _find_reference_rule,
)
from scripts.compare_verifier_effect_metrics import _compare_one, _markdown
from scripts.manage_rule_library import main as manage_main


class RuleRuntimeMetadataTests(unittest.TestCase):
    def test_backfill_adds_runtime_metadata_without_api(self) -> None:
        rule = {
            "rule_id": "norm_test_rule",
            "title": "Orbital radius energy consistency",
            "trigger": "satellite orbital decay",
            "check_logic": "check orbital radius and energy change signs",
            "symbolic_hint": {"primitive": "formula_pattern", "canonical": "E=-GMm/(2r)", "required_symbols": ["r"]},
        }

        changed = _ensure_rule_metadata(rule, domain="Mechanics", topic="Gravitation and Kepler's Laws", overwrite=False)

        self.assertTrue(changed["match_features"])
        self.assertEqual(rule["id"], "norm_test_rule")
        self.assertEqual(rule["path"]["domain"], "Mechanics")
        self.assertIn("preconditions", rule)
        self.assertIn("violation_signatures", rule)
        self.assertIn("negative_conditions", rule)
        self.assertIn("evidence_requirements", rule)
        self.assertIn("satellite", " ".join(rule["match_features"]["trigger_keywords"]).casefold())

    def test_backfill_reuses_reference_catalog_metadata_without_api(self) -> None:
        target = {
            "rule_id": "norm_new",
            "title": "Orbital radius energy consistency",
            "trigger": "satellite orbital decay",
            "check_logic": "check orbital radius and energy change signs",
            "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []},
        }
        reference_catalog = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Gravitation",
                            "rules": [
                                {
                                    "rule_id": "exp_old",
                                    "title": target["title"],
                                    "trigger": target["trigger"],
                                    "check_logic": target["check_logic"],
                                    "match_features": {"trigger_keywords": ["orbital decay"], "object_keywords": [], "required_symbols": ["r"]},
                                    "support": {"count": 3, "sample_ids": ["a", "b", "c"]},
                                    "llm_hints": {"match_phrases": ["the orbit decays"], "discriminative_terms": ["orbital decay"]},
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        index = _build_reference_index([reference_catalog])
        reference = _find_reference_rule(target, index)
        self.assertIsNotNone(reference)

        changed = _copy_reference_metadata(target, reference or {}, overwrite=False)
        _ensure_rule_metadata(target, domain="Mechanics", topic="Gravitation", overwrite=False)

        self.assertTrue(changed["match_features"])
        self.assertEqual(target["support"]["count"], 3)
        self.assertEqual(target["llm_hints"]["discriminative_terms"], ["orbital decay"])
        self.assertEqual(target["match_features"]["trigger_keywords"], ["orbital decay"])
        self.assertEqual(target["source_rule_ids"], ["exp_old"])

    def test_backfill_restores_topic_routing_metadata(self) -> None:
        reference_catalog = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Gravitation",
                            "retrieval_hints": {
                                "scene_keywords": ["satellite orbital decay"],
                                "topic_keywords": ["orbit"],
                                "required_symbols": ["r"],
                                "llm_problem_phrases": ["a satellite loses orbital energy"],
                            },
                            "knowledge_reference": {"rule_ids": ["knowledge_1"], "keywords": ["Kepler"]},
                            "tagged_reference": {"aliases": ["orbital mechanics"]},
                            "rules": [],
                        }
                    ],
                }
            ]
        }
        index = _build_reference_topic_index([reference_catalog])
        target = {"name": "Gravitation", "rules": []}

        changed = _copy_reference_topic_metadata(
            target,
            index[("mechanics", "gravitation")],
            overwrite=False,
        )
        _ensure_topic_metadata(target, domain="Mechanics", overwrite=False)

        self.assertTrue(changed["retrieval_hints"])
        self.assertEqual(target["retrieval_hints"]["llm_problem_phrases"], ["a satellite loses orbital energy"])
        self.assertEqual(target["knowledge_reference"]["keywords"], ["Kepler"])
        self.assertEqual(target["tagged_reference"]["aliases"], ["orbital mechanics"])

    def test_backfill_generates_topic_routing_metadata_without_reference(self) -> None:
        topic = {
            "name": "Gravitation",
            "rules": [
                {
                    "rule_id": "r1",
                    "title": "Orbital energy consistency",
                    "trigger": "satellite orbital decay",
                    "scope": "domain",
                    "match_features": {"required_symbols": ["r_orbit"]},
                }
            ],
        }

        changed = _ensure_topic_metadata(topic, domain="Mechanics", overwrite=False)

        self.assertTrue(changed["retrieval_hints"])
        self.assertIn("r_orbit", topic["retrieval_hints"]["required_symbols"])
        self.assertTrue(topic["retrieval_hints"]["topic_keywords"])
        self.assertTrue(topic["retrieval_hints"]["scene_keywords"])

    def test_fp_metrics_rule_counts_accept_false_positive_replay(self) -> None:
        payload = {
            "false_positive_replay": [
                {"rule_id": "r1"},
                {"rule": "r1"},
                {"rule_match": {"rule_id": "r2"}},
            ]
        }
        with patch("scripts.audit_rule_runtime_metadata._load_json", return_value=payload):
            counts = _load_fp_rule_counts("unused.json")

        self.assertEqual(counts["r1"], 2)
        self.assertEqual(counts["r2"], 1)

    def test_audit_outputs_fp_rule_ids_ordered_by_count(self) -> None:
        catalog = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "rules": [
                                {"rule_id": "r1", "title": "Rule 1", "symbolic_hint": {}},
                                {"rule_id": "r2", "title": "Rule 2", "symbolic_hint": {}},
                            ],
                        }
                    ],
                }
            ]
        }
        fp_payload = {"false_positive_replay": [{"rule_id": "r2"}, {"rule_id": "r1"}, {"rule_id": "r2"}]}
        writes: dict[str, str] = {}

        def fake_load(path: str) -> dict:
            return fp_payload if path == "fp.json" else catalog

        class FakePath:
            def __init__(self, value: str) -> None:
                self.value = value
                self.parent = self

            def mkdir(self, *_: object, **__: object) -> None:
                return None

            def write_text(self, text: str, **_: object) -> None:
                writes[self.value] = text

        argv = [
            "audit_rule_runtime_metadata.py",
            "--catalog",
            "catalog.json",
            "--fp-metrics",
            "fp.json",
            "--rule-ids-output",
            "rule_ids.json",
            "--top",
            "2",
        ]
        with patch("scripts.audit_rule_runtime_metadata._load_json", side_effect=fake_load), patch(
            "scripts.audit_rule_runtime_metadata.Path", FakePath
        ), patch("sys.argv", argv):
            audit_main()

        self.assertEqual(writes["rule_ids.json"].strip(), '[\n  "r2",\n  "r1"\n]')

    def test_audit_separates_required_and_optional_coverage(self) -> None:
        catalog = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "retrieval_hints": {"topic_keywords": ["motion"]},
                            "knowledge_reference": {"keywords": ["motion"]},
                            "tagged_reference": {"aliases": []},
                            "rules": [
                                {
                                    "rule_id": "r1",
                                    "match_features": {"trigger_keywords": ["motion"]},
                                    "support": {"count": 0, "sample_ids": []},
                                    "preconditions": ["motion"],
                                    "violation_signatures": ["wrong direction"],
                                    "negative_conditions": ["later corrected"],
                                    "evidence_requirements": ["direction"],
                                    "symbolic_hint": {"primitive": "none"},
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        writes: dict[str, str] = {}

        class FakePath:
            def __init__(self, value: str) -> None:
                self.value = value
                self.parent = self

            def mkdir(self, *_: object, **__: object) -> None:
                return None

            def write_text(self, text: str, **_: object) -> None:
                writes[self.value] = text

        argv = ["audit_rule_runtime_metadata.py", "--catalog", "catalog.json", "--output", "audit.json"]
        with patch("scripts.audit_rule_runtime_metadata._load_json", return_value=catalog), patch(
            "scripts.audit_rule_runtime_metadata.Path", FakePath
        ), patch("sys.argv", argv):
            audit_main()

        report = __import__("json").loads(writes["audit.json"])
        self.assertTrue(report["runtime_readiness"]["ready"])
        self.assertEqual(report["coverage"]["llm_hints"], 0)
        self.assertEqual(report["missing_runtime_metadata"], [])
        self.assertEqual(report["missing_optional_metadata"][0]["missing"], ["llm_hints", "source_rule_ids"])

    def test_manage_enhance_forwards_rule_id_whitelist(self) -> None:
        calls: list[dict] = []

        def fake_load(path: str) -> object:
            if path == "catalog.json":
                return {"domains": []}
            if path == "rule_ids.json":
                return ["r2", "r1"]
            raise AssertionError(path)

        def fake_write(path: str, payload: object) -> None:
            calls.append({"write_path": path, "payload": payload})

        def fake_enhance(catalog: dict, **kwargs: object) -> dict:
            calls.append({"rule_ids_filter": kwargs.get("rule_ids_filter")})
            return catalog

        class FakePath:
            def __init__(self, value: str) -> None:
                self.value = value

            def read_text(self, **_: object) -> str:
                if self.value == "rule_ids.json":
                    return '["r2", "r1"]'
                raise AssertionError(self.value)

        argv = [
            "manage_rule_library.py",
            "enhance",
            "--catalog",
            "catalog.json",
            "--output",
            "out.json",
            "--rule-ids-file",
            "rule_ids.json",
            "--no-topic-hints",
            "--no-semantic-clusters",
        ]
        with patch("scripts.manage_rule_library.load_json", side_effect=fake_load), patch(
            "scripts.manage_rule_library.write_json", side_effect=fake_write
        ), patch("scripts.manage_rule_library.enhance_catalog", side_effect=fake_enhance), patch(
            "scripts.manage_rule_library.Path", FakePath
        ), patch("sys.argv", argv):
            manage_main()

        self.assertEqual(calls[0]["rule_ids_filter"], {"r1", "r2"})
        self.assertEqual(calls[1]["write_path"], "out.json")

    def test_metric_comparison_reports_metric_and_count_deltas(self) -> None:
        before = {"summary": {"precision": 0.4, "recall": 0.8, "f1": 0.5333, "fp": 6, "tp": 4}}
        after = {"summary": {"precision": 0.5, "recall": 0.75, "f1": 0.6, "fp": 4, "tp": 4}}

        with patch("scripts.compare_verifier_effect_metrics._load_json", side_effect=[before, after]):
            report = _compare_one("before.json", "after.json", level="question")

        self.assertAlmostEqual(report["metrics"]["precision"]["delta"], 0.1)
        self.assertAlmostEqual(report["metrics"]["recall"]["delta"], -0.05)
        self.assertEqual(report["counts"]["fp"]["delta"], -2)
        self.assertIn("| metrics | precision | 0.4000 | 0.5000 | 0.1000 |", _markdown({"comparisons": [report]}))

    def test_topic_includes_and_excludes_affect_retrieval_evidence(self) -> None:
        catalog = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Gravitation and Kepler's Laws",
                            "includes": ["satellite orbital decay"],
                            "excludes": ["electric circuit"],
                            "rules": [{"rule_id": "orbit_check"}],
                        }
                    ],
                }
            ]
        }
        candidates = build_topic_candidates(catalog)
        signal_df = build_signal_document_frequency(candidates)

        hit = score_topic_candidate(candidates[0], "A satellite orbital decay problem", signal_df=signal_df)
        excluded = score_topic_candidate(candidates[0], "An electric circuit problem", signal_df=signal_df)

        self.assertIn("satellite orbital decay", hit["evidence"]["include_hits"])
        self.assertIn("electric circuit", excluded["evidence"]["exclude_hits"])
        self.assertGreater(hit["score"], excluded["score"])

    def test_topic_candidates_exclude_topics_without_executable_rules(self) -> None:
        catalog = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {"name": "Empty", "rules": []},
                        {
                            "name": "Kinematics",
                            "rules": [{"rule_id": "motion_check"}],
                        },
                    ],
                }
            ]
        }

        candidates = build_topic_candidates(catalog)

        self.assertEqual([item["topic"] for item in candidates], ["Kinematics"])

    def test_rule_precision_fields_affect_retrieval_evidence(self) -> None:
        rule = {
            "rule_id": "r_precision",
            "title": "Current direction check",
            "match_features": {
                "trigger_keywords": ["current direction"],
                "object_keywords": ["Kirchhoff loop"],
                "required_symbols": ["I"],
            },
            "preconditions": ["Kirchhoff loop"],
            "violation_signatures": ["wrong current direction"],
            "negative_conditions": ["intermediate value later corrected"],
            "evidence_requirements": ["I"],
        }

        payload = score_rule_candidate(
            rule,
            "The Kirchhoff loop uses I but later has the wrong current direction.",
        )
        negative = score_rule_candidate(
            rule,
            "The Kirchhoff loop has I as an intermediate value later corrected.",
        )

        self.assertIn("Kirchhoff loop", payload["evidence"]["precondition_hits"])
        self.assertIn("wrong current direction", payload["evidence"]["violation_signature_hits"])
        self.assertIn("I", payload["evidence"]["evidence_requirement_hits"])
        self.assertIn("intermediate value later corrected", negative["evidence"]["negative_keyword_hits"])
        self.assertLess(negative["score"], payload["score"])


if __name__ == "__main__":
    unittest.main()
