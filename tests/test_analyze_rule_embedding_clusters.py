from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.analyze_rule_embedding_clusters import analyze_embedding_clusters


TMP_ROOT = Path(tempfile.mkdtemp(prefix="physicsverifier_embedding_tests_"))
atexit.register(shutil.rmtree, TMP_ROOT, ignore_errors=True)


def _case_dir() -> Path:
    path = TMP_ROOT / f"embedding_cluster_analysis_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class AnalyzeRuleEmbeddingClustersTests(unittest.TestCase):
    def test_analyze_embedding_clusters_reports_coverage(self) -> None:
        root = _case_dir()
        input_path = root / "clusters.json"
        output_path = root / "report.json"
        input_path.write_text(
            json.dumps(
                {
                    "metadata": {"generator": "topic_local_rule_embedding_clustering_v1"},
                    "topics": [
                        {
                            "topic_key": "mechanics::kinematics",
                            "rule_count": 10,
                            "clusters": [
                                {"cluster_id": "c1", "rule_ids": ["r1", "r2", "r3", "r4"]},
                                {"cluster_id": "c2", "rule_ids": ["r5", "r6"]},
                            ],
                            "residual_rule_ids": ["r7", "r8", "r9", "r10"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        report = analyze_embedding_clusters(input_path=input_path, output_path=output_path)

        self.assertTrue(report["ready_for_labeling"])
        self.assertEqual(report["topic_count"], 1)
        self.assertEqual(report["total_clustered_rule_count"], 6)
        self.assertEqual(report["clustered_rule_ratio"], 0.6)
        self.assertTrue(output_path.exists())

    def test_strict_fails_when_clustered_ratio_too_low(self) -> None:
        root = _case_dir()
        input_path = root / "clusters.json"
        input_path.write_text(
            json.dumps(
                {
                    "topics": [
                        {
                            "topic_key": "mechanics::dynamics",
                            "rule_count": 10,
                            "clusters": [{"cluster_id": "c1", "rule_ids": ["r1"]}],
                            "residual_rule_ids": ["r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaises(SystemExit):
            analyze_embedding_clusters(
                input_path=input_path,
                min_clustered_rule_ratio=0.3,
                strict=True,
            )

    def test_missing_or_duplicate_rule_assignment_is_structurally_invalid(self) -> None:
        root = _case_dir()
        input_path = root / "clusters.json"
        input_path.write_text(
            json.dumps(
                {
                    "metadata": {"rule_count": 3, "topic_count": 1},
                    "topics": [
                        {
                            "topic_key": "mechanics::kinematics",
                            "rule_count": 3,
                            "clusters": [{"cluster_id": "c1", "rule_ids": ["r1", "r1"]}],
                            "residual_rule_ids": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        report = analyze_embedding_clusters(input_path=input_path)

        self.assertFalse(report["structural_valid"])
        self.assertFalse(report["ready_for_labeling"])
        self.assertEqual(report["invalid_topic_keys"], ["mechanics::kinematics"])

    def test_cluster_ids_must_match_current_embedding_input(self) -> None:
        root = _case_dir()
        input_path = root / "clusters.json"
        rule_input_path = root / "rules.json"
        input_path.write_text(
            json.dumps(
                {
                    "topics": [
                        {
                            "topic_key": "mechanics::kinematics",
                            "rule_count": 1,
                            "clusters": [],
                            "residual_rule_ids": ["old_id"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        rule_input_path.write_text(
            json.dumps(
                {
                    "rules": [
                        {
                            "topic_key": "mechanics::kinematics",
                            "rule_id": "new_id",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        report = analyze_embedding_clusters(
            input_path=input_path,
            rule_input_path=rule_input_path,
            min_clustered_rule_ratio=0.0,
        )

        self.assertFalse(report["source_alignment_valid"])
        self.assertFalse(report["ready_for_labeling"])


if __name__ == "__main__":
    unittest.main()
