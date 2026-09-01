from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.evaluate_unified_rules_quality import evaluate_catalog_quality


TMP_ROOT = Path(tempfile.mkdtemp(prefix="physicsverifier_quality_tests_"))
atexit.register(shutil.rmtree, TMP_ROOT, ignore_errors=True)


def _case_dir() -> Path:
    path = TMP_ROOT / f"quality_eval_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class EvaluateUnifiedRulesQualityTests(unittest.TestCase):
    def test_evaluate_catalog_quality_reports_schema_summary_and_cluster_gaps(self) -> None:
        root = _case_dir()
        catalog_path = root / "catalog.json"
        proposals_path = root / "cluster_proposals.json"
        runtime_path = root / "runtime_eval.json"
        output_path = root / "quality.json"
        catalog_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "catalog_type": "unified_rules_v2",
                        "schema_profile": "semantic_navigation_tree_minimal",
                        "total_domains": 1,
                        "total_topics": 2,
                        "topics_with_rules": 2,
                        "total_executable_rules": 3,
                        "total_scenario_clusters": 1,
                    },
                    "domains": [
                        {
                            "id": "mechanics",
                            "name": "Mechanics",
                            "summary": "Classical mechanics navigation domain.",
                            "topics": [
                                {
                                    "id": "kinematics",
                                    "name": "Kinematics",
                                    "summary": "Motion relations.",
                                    "scenario_clusters": [
                                        {
                                            "id": "timing",
                                            "name": "Timing",
                                            "summary": "Timing and displacement checks.",
                                            "rule_ids": ["r1"],
                                        }
                                    ],
                                    "rules": [
                                        {
                                            "rule_id": "r1",
                                            "title": "Timing relation",
                                            "summary": "Check timing relation.",
                                            "trigger": "time and displacement",
                                            "check_logic": "Verify kinematic relation.",
                                            "error_type": "calculation",
                                            "symbolic_hint": {"canonical": "x=vt"},
                                        },
                                        {
                                            "rule_id": "r2",
                                            "title": "Timing relation",
                                            "summary": "Check timing relation.",
                                            "trigger": "time and displacement",
                                            "check_logic": "Verify kinematic relation.",
                                            "error_type": "calculation",
                                            "symbolic_hint": {"canonical": "x=vt"},
                                        },
                                    ],
                                },
                                {
                                    "id": "dynamics",
                                    "name": "Dynamics",
                                    "summary": "Force reasoning.",
                                    "scenario_clusters": [],
                                    "rules": [
                                        {
                                            "rule_id": "r3",
                                            "title": "Force balance",
                                            "summary": "Check force balance.",
                                            "trigger": "forces",
                                            "check_logic": "Verify net force.",
                                            "error_type": "concept",
                                            "symbolic_hint": {},
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        proposals_path.write_text(
            json.dumps({"proposals": [], "failures": [{"topic_key": "mechanics::dynamics"}]}),
            encoding="utf-8",
        )
        runtime_path.write_text(
            json.dumps(
                {
                    "summary": {"sample_count": 2, "semantic_error_count": 0, "rule_selection_rate": 0.5, "average_selected_rules": 3.0},
                    "rows": [
                        {"sample_id": "s1", "topic_count": 1, "cluster_count": 1, "rule_count": 0, "semantic_selection_error": ""},
                        {"sample_id": "s2", "topic_count": 3, "cluster_count": 4, "rule_count": 6, "semantic_selection_error": ""},
                    ],
                }
            ),
            encoding="utf-8",
        )

        report = evaluate_catalog_quality(
            catalog_path=catalog_path,
            cluster_proposals_path=proposals_path,
            runtime_eval_path=runtime_path,
            output_path=output_path,
        )

        self.assertEqual(report["schema"]["schema_profile"], "semantic_navigation_tree_minimal")
        self.assertEqual(report["summary_quality"]["rules"]["missing"], 0)
        self.assertEqual(report["cluster_quality"]["unclustered_topic_count"], 1)
        self.assertEqual(report["duplication"]["duplicate_summary_group_count"], 1)
        self.assertEqual(report["cluster_proposals"]["failure_count"], 1)
        self.assertFalse(report["runtime_eval"]["stale"])
        self.assertEqual(report["runtime_eval"]["empty_rule_sample_ids"], ["s1"])
        self.assertEqual(report["runtime_eval"]["high_rule_selection_sample_ids"], ["s2"])
        self.assertEqual(report["runtime_eval"]["rule_cap"], 5)
        self.assertEqual(report["runtime_eval"]["rule_cap_violation_sample_ids"], ["s2"])
        self.assertEqual(report["runtime_eval"]["broad_topic_selection_sample_ids"], ["s2"])
        self.assertEqual(report["runtime_eval"]["broad_cluster_selection_sample_ids"], ["s2"])
        self.assertFalse(report["overall"]["readiness_gates"]["runtime_rule_cap_respected"])
        self.assertGreater(report["overall"]["blocking_gate_count"], 0)
        self.assertTrue(output_path.exists())

    def test_evaluate_catalog_quality_omits_unclustered_recommendation_when_fully_clustered(self) -> None:
        root = _case_dir()
        catalog_path = root / "catalog.json"
        catalog_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "catalog_type": "unified_rules_v2",
                        "schema_profile": "semantic_navigation_tree_minimal",
                        "total_domains": 1,
                        "total_topics": 1,
                        "topics_with_rules": 1,
                        "total_executable_rules": 1,
                        "total_scenario_clusters": 1,
                    },
                    "domains": [
                        {
                            "name": "Mechanics",
                            "topics": [
                                {
                                    "name": "Kinematics",
                                    "summary": "Motion relations.",
                                    "scenario_clusters": [
                                        {"id": "timing", "name": "Timing", "summary": "Timing checks.", "rule_ids": ["r1"]}
                                    ],
                                    "rules": [
                                        {
                                            "rule_id": "r1",
                                            "title": "Timing",
                                            "summary": "Timing checks.",
                                            "trigger": "time",
                                            "check_logic": "Check timing.",
                                            "error_type": "logic",
                                            "symbolic_hint": {},
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        report = evaluate_catalog_quality(catalog_path=catalog_path, cluster_proposals_path=None, runtime_eval_path=None)

        self.assertEqual(report["cluster_quality"]["unclustered_rule_count"], 0)
        self.assertNotIn("继续补齐尚未进入 scenario clusters 的 topic。", report["overall"]["recommended_next_steps"])

    def test_evaluate_catalog_quality_accepts_raw_retrieval_traces(self) -> None:
        root = _case_dir()
        catalog_path = root / "catalog.json"
        proposals_path = root / "proposals.json"
        runtime_path = root / "runtime.json"
        output_path = root / "quality.json"
        catalog_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "catalog_type": "unified_rules_v2",
                        "schema_profile": "semantic_navigation_tree_minimal",
                        "total_domains": 1,
                        "total_topics": 1,
                        "topics_with_rules": 1,
                        "total_executable_rules": 1,
                        "total_scenario_clusters": 1,
                    },
                    "domains": [],
                }
            ),
            encoding="utf-8",
        )
        proposals_path.write_text(
            json.dumps({"proposals": [], "failures": []}),
            encoding="utf-8",
        )
        runtime_path.write_text(
            json.dumps(
                [
                    {
                        "id": "s1",
                        "retrieved_topics": [{"topic": "T"}],
                        "retrieved_clusters": [{"cluster": "C"}],
                        "retrieved_rules": [],
                        "semantic_selection_error": "",
                    },
                    {
                        "id": "s2",
                        "retrieved_topics": [{"topic": "T"}],
                        "retrieved_clusters": [{"cluster": "C"}],
                        "retrieved_rules": [{"rule_id": "r1"}],
                        "semantic_selection_error": "",
                    },
                ]
            ),
            encoding="utf-8",
        )

        report = evaluate_catalog_quality(
            catalog_path=catalog_path,
            cluster_proposals_path=proposals_path,
            runtime_eval_path=runtime_path,
            output_path=output_path,
        )

        self.assertEqual(report["runtime_eval"]["sample_count"], 2)
        self.assertEqual(report["runtime_eval"]["rule_selection_rate"], 0.5)
        self.assertEqual(report["runtime_eval"]["average_selected_rules"], 0.5)
        self.assertEqual(report["runtime_eval"]["empty_rule_sample_ids"], ["s1"])


if __name__ == "__main__":
    unittest.main()
