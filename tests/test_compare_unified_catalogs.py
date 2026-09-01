from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.compare_unified_catalogs import _console_json, compare_catalogs

TMP_ROOT = Path(tempfile.mkdtemp(prefix="physicsverifier_compare_tests_"))
atexit.register(shutil.rmtree, TMP_ROOT, ignore_errors=True)


def _case_dir() -> Path:
    path = TMP_ROOT / f"physicsverifier_compare_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class CompareUnifiedCatalogsTests(unittest.TestCase):
    def test_console_json_is_ascii_safe_for_windows_shells(self) -> None:
        text = _console_json({"topic": "Schrödinger Equation"})

        self.assertTrue(all(ord(char) < 128 for char in text))
        self.assertIn("\\u00f6", text)

    def test_compare_catalogs_reports_growth_and_cluster_coverage(self) -> None:
        baseline = {
            "metadata": {"total_executable_rules": 1, "topics_with_rules": 1, "total_scenario_clusters": 1},
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "rules": [{"rule_id": "exp_a"}],
                            "scenario_clusters": [
                                {"id": "timing", "rule_ids": ["exp_a"]},
                            ],
                        }
                    ],
                }
            ],
        }
        candidate = {
            "metadata": {"total_executable_rules": 3, "topics_with_rules": 2, "total_scenario_clusters": 3},
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "rules": [{"rule_id": "exp_a"}, {"rule_id": "exp_b"}],
                            "scenario_clusters": [
                                {"id": "timing", "rule_ids": ["exp_a"]},
                                {"id": "general_reasoning", "rule_ids": ["exp_b"]},
                            ],
                        },
                        {
                            "name": "Dynamics",
                            "rules": [{"rule_id": "exp_c"}],
                            "scenario_clusters": [],
                        },
                    ],
                }
            ],
        }

        root = _case_dir()
        baseline_path = root / "baseline.json"
        candidate_path = root / "candidate.json"
        output_path = root / "comparison.json"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

        comparison = compare_catalogs(baseline_path, candidate_path, output_path)

        self.assertEqual(comparison["summary"]["rule_delta"], 2)
        self.assertEqual(comparison["summary"]["topics_with_rules_delta"], 1)
        self.assertEqual(comparison["summary"]["scenario_cluster_delta"], 2)
        kin = comparison["topics"]["Mechanics::Kinematics"]
        self.assertEqual(kin["rule_delta"], 1)
        self.assertEqual(kin["candidate_cluster_rule_coverage"], 1.0)
        self.assertEqual(kin["candidate_general_reasoning_rule_ratio"], 0.5)
        dyn = comparison["topics"]["Mechanics::Dynamics"]
        self.assertEqual(dyn["baseline_rule_count"], 0)
        self.assertEqual(dyn["candidate_rule_count"], 1)
        self.assertEqual(dyn["candidate_cluster_rule_coverage"], 0.0)
        self.assertEqual(comparison["cluster_coverage"]["unclustered_topic_count"], 1)
        self.assertEqual(comparison["cluster_coverage"]["unclustered_rule_count"], 1)
        self.assertEqual(
            comparison["cluster_coverage"]["top_unclustered_topics"][0],
            {"topic": "Mechanics::Dynamics", "rule_count": 1},
        )
        self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
