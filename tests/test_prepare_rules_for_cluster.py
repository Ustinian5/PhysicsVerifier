from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.prepare_rules_for_cluster import prepare_rules_for_cluster

TMP_ROOT = Path(tempfile.mkdtemp(prefix="physicsverifier_prepare_cluster_tests_"))
atexit.register(shutil.rmtree, TMP_ROOT, ignore_errors=True)


def _case_dir() -> Path:
    path = TMP_ROOT / f"physicsverifier_prepare_cluster_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class PrepareRulesForClusterTests(unittest.TestCase):
    def test_incremental_mode_preserves_baseline_rule_identity(self) -> None:
        root = _case_dir()
        knowledge = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [{"name": "Kinematics", "rules": []}],
                }
            ]
        }
        baseline = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "rules": [
                                {
                                    "rule_id": "old_id",
                                    "title": "Existing rule",
                                    "summary": "Existing summary",
                                    "trigger": "Existing trigger",
                                    "check_logic": "Existing check",
                                    "error_type": "logic",
                                }
                            ],
                            "scenario_clusters": [],
                        }
                    ],
                }
            ]
        }
        distilled = {
            "rules": [
                {
                    "rule_id": "old_id",
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "title": "Reworded existing rule",
                    "trigger": "Reworded trigger",
                    "check_logic": "Reworded check",
                    "error_type": "logic",
                },
                {
                    "rule_id": "new_id",
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "title": "New rule",
                    "trigger": "New trigger",
                    "check_logic": "New check",
                    "error_type": "logic",
                },
            ]
        }
        paths = {
            "knowledge": root / "knowledge.json",
            "tagged": root / "tagged.json",
            "baseline": root / "baseline.json",
            "distilled": root / "distilled.json",
            "normalized": root / "normalized.json",
            "catalog": root / "catalog.json",
            "report": root / "report.json",
        }
        paths["knowledge"].write_text(json.dumps(knowledge), encoding="utf-8")
        paths["tagged"].write_text("[]", encoding="utf-8")
        paths["baseline"].write_text(json.dumps(baseline), encoding="utf-8")
        paths["distilled"].write_text(json.dumps(distilled), encoding="utf-8")

        report = prepare_rules_for_cluster(
            distilled_input=paths["distilled"],
            knowledge_path=paths["knowledge"],
            tagged_path=paths["tagged"],
            baseline_catalog_path=paths["baseline"],
            distilled_output=paths["normalized"],
            catalog_output=paths["catalog"],
            report_output=paths["report"],
            scenario_cluster_blueprints_paths=[],
            preserve_baseline_rule_ids=True,
        )

        payload = json.loads(paths["normalized"].read_text(encoding="utf-8"))
        rules = {rule["rule_id"]: rule for rule in payload["rules"]}
        self.assertEqual(set(rules), {"old_id", "new_id"})
        self.assertEqual(rules["old_id"]["title"], "Existing rule")
        self.assertEqual(report["normalization"]["preserved_by_rule_id"], 1)
        self.assertTrue(report["normalization"]["preserve_baseline_rule_ids"])

    def test_prepares_single_cluster_ready_rule_set(self) -> None:
        root = _case_dir()
        distilled_path = root / "distilled.json"
        knowledge_path = root / "knowledge.json"
        tagged_path = root / "tagged.json"
        baseline_path = root / "baseline.json"
        normalized_output = root / "distilled_for_cluster.json"
        catalog_output = root / "rules_for_cluster.json"
        report_output = root / "report.json"
        embedding_output = root / "rule_embedding_input.json"

        knowledge_path.write_text(
            json.dumps(
                {
                    "domains": [
                        {
                            "name": "Mechanics",
                            "topics": [
                                {
                                    "name": "Fluid Dynamics (Bernoulli's Equation, Continuity)",
                                    "rules": [],
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        tagged_path.write_text("[]", encoding="utf-8")
        baseline_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "catalog_type": "unified_rules_v2",
                        "schema_profile": "semantic_navigation_tree_minimal",
                        "total_executable_rules": 0,
                        "topics_with_rules": 0,
                        "total_scenario_clusters": 0,
                    },
                    "domains": [
                        {
                            "name": "Mechanics",
                            "topics": [
                                {
                                    "name": "Fluid Dynamics (Bernoulli's Equation, Continuity)",
                                    "rules": [
                                        {
                                            "rule_id": "exp_old",
                                            "title": "旧版伯努利规则",
                                            "summary": "旧版流体能量规则",
                                            "trigger": "旧版样本触发",
                                            "check_logic": "检查伯努利能量项。",
                                            "error_type": "logic",
                                            "symbolic_hint": {"primitive": "formula_pattern", "canonical": "p+rho g h+rho v^2/2", "required_symbols": ["p", "v"]},
                                        }
                                    ],
                                    "scenario_clusters": [],
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        distilled_path.write_text(
            json.dumps(
                {
                    "rules": [
                        {
                            "rule_id": "r1",
                            "domain": "Mechanics",
                            "topic": "Mechanics / Fluid Dynamics (Bernoulli's Equation, Continuity)",
                            "title": "连续性方程截面积校验",
                            "trigger": "管道流体截面积变化",
                            "check_logic": "检查 A v 是否守恒。",
                            "error_type": "calculation",
                            "symbolic_hint": {
                                "primitive": "formula_pattern",
                                "canonical": "A1 v1 = A2 v2",
                                "required_symbols": ["A", "v"],
                            },
                            "auxiliary": {
                                "node_summary": "流体连续性检查",
                                "scene_cues": ["pipe"],
                                "boundary_cues": ["incompressible"],
                                "explore_cues": ["viscosity"],
                                "evidence_sample_ids": ["s1"],
                            },
                            "count": 1,
                            "sample_ids": ["s1"],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        report = prepare_rules_for_cluster(
            distilled_input=distilled_path,
            knowledge_path=knowledge_path,
            tagged_path=tagged_path,
            baseline_catalog_path=baseline_path,
            distilled_output=normalized_output,
            catalog_output=catalog_output,
            report_output=report_output,
            embedding_input_output=embedding_output,
            scenario_cluster_blueprints_paths=[],
        )

        self.assertEqual(report["normalization"]["distilled_input_rules"], 1)
        self.assertEqual(report["normalization"]["baseline_seed_rules"], 1)
        self.assertEqual(report["normalization"]["input_rules"], 2)
        self.assertEqual(report["normalization"]["output_rules"], 2)
        self.assertEqual(report["quality"]["unmatched_topic_count"], 0)
        self.assertEqual(report["catalog"]["schema_profile"], "semantic_navigation_tree_minimal")
        self.assertEqual(report["catalog"]["total_executable_rules"], 2)
        self.assertEqual(report["embedding_input"]["rule_count"], 2)
        self.assertTrue(normalized_output.exists())
        self.assertTrue(catalog_output.exists())
        self.assertTrue(report_output.exists())
        self.assertTrue(embedding_output.exists())

        normalized = json.loads(normalized_output.read_text(encoding="utf-8"))
        rules_by_title = {rule["title"]: rule for rule in normalized["rules"]}
        rule = rules_by_title["连续性方程截面积校验"]
        self.assertEqual(rule["rule_id"], "r1")
        self.assertEqual(rule["summary"], "流体连续性检查")
        self.assertNotIn("Mechanics /", rule["topic"])
        self.assertEqual(rules_by_title["旧版伯努利规则"]["rule_id"], "exp_old")
        self.assertEqual(rules_by_title["旧版伯努利规则"]["summary"], "旧版流体能量规则")

        embedding_payload = json.loads(embedding_output.read_text(encoding="utf-8"))
        record = next(item for item in embedding_payload["rules"] if item["title"] == "连续性方程截面积校验")
        self.assertEqual(record["rule_id"], rule["rule_id"])
        self.assertEqual(record["summary"], "流体连续性检查")
        self.assertIn("流体连续性检查", record["embedding_text"])
        self.assertIn("管道流体截面积变化", record["embedding_text"])
        self.assertEqual(record["near_duplicate_key"], "Mechanics::Fluid Dynamics (Bernoulli's Equation, Continuity)::连续性方程截面积校验")

        catalog = json.loads(catalog_output.read_text(encoding="utf-8"))
        catalog_rules = catalog["domains"][0]["topics"][0]["rules"]
        catalog_rule = next(item for item in catalog_rules if item["title"] == "连续性方程截面积校验")
        self.assertEqual(catalog_rule["summary"], "流体连续性检查")


if __name__ == "__main__":
    unittest.main()
