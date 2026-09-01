from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.analyze_semantic_experience_run import analyze_run

TMP_ROOT = Path(tempfile.mkdtemp(prefix="physicsverifier_analyze_tests_"))
atexit.register(shutil.rmtree, TMP_ROOT, ignore_errors=True)


def _case_dir() -> Path:
    path = TMP_ROOT / f"physicsverifier_analyze_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class AnalyzeSemanticExperienceRunTests(unittest.TestCase):
    def test_analyze_run_counts_semantic_distilled_and_auxiliary_stats(self) -> None:
        root = _case_dir()
        semantic_path = root / "semantic.json"
        distilled_path = root / "distilled.json"
        output_path = root / "report.json"

        semantic_path.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "sample_id": "s1",
                            "topic_guess": {"domain": "Mechanics", "topic": "Kinematics"},
                            "semantic_audit": {"summary": "ok", "key_errors": []},
                            "experience_rules": [{"title": "r1"}, {"title": "r2"}],
                        },
                        {
                            "sample_id": "s2",
                            "topic_guess": {"domain": "Unknown", "topic": "Unknown"},
                            "semantic_audit": {
                                "summary": "LLM调用失败，已记录重试占位。",
                                "key_errors": [{"message": "LLM调用失败"}],
                            },
                            "experience_rules": [],
                        },
                        {
                            "sample_id": "s2",
                            "topic_guess": {"domain": "Mechanics", "topic": "Kinematics"},
                            "semantic_audit": {"summary": "duplicate", "key_errors": []},
                            "experience_rules": [],
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        distilled_path.write_text(
            json.dumps(
                {
                    "rules": [
                        {
                            "rule_id": "exp_a",
                            "domain": "Mechanics",
                            "topic": "Kinematics",
                            "auxiliary": {
                                "node_summary": "Motion timing.",
                                "scene_cues": ["timer"],
                                "boundary_cues": [],
                                "explore_cues": ["projection"],
                                "evidence_sample_ids": ["s1"],
                            },
                        },
                        {
                            "rule_id": "exp_b",
                            "domain": "Unknown",
                            "topic": "Unknown",
                            "auxiliary": {
                                "node_summary": "",
                                "scene_cues": [],
                                "boundary_cues": [],
                                "explore_cues": [],
                                "evidence_sample_ids": [],
                            },
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        report = analyze_run(
            semantic_path=semantic_path,
            distilled_path=distilled_path,
            expected_samples=3,
            output_path=output_path,
            strict=False,
        )

        self.assertEqual(report["semantic"]["sample_count"], 3)
        self.assertEqual(report["semantic"]["duplicate_sample_id_count"], 1)
        self.assertEqual(report["semantic"]["failure_placeholder_count"], 1)
        self.assertEqual(report["semantic"]["empty_rule_sample_count"], 2)
        self.assertEqual(report["semantic"]["unknown_topic_sample_count"], 1)
        self.assertEqual(report["semantic"]["topics"]["Mechanics::Kinematics"]["sample_count"], 2)
        self.assertEqual(report["semantic"]["topics"]["Mechanics::Kinematics"]["experience_rule_count"], 2)
        self.assertEqual(report["distilled"]["total_rules"], 2)
        self.assertEqual(report["distilled"]["topic_bucket_count"], 2)
        self.assertEqual(report["distilled"]["unknown_rule_count"], 1)
        self.assertEqual(report["distilled"]["auxiliary"]["rules_with_node_summary"], 1)
        self.assertEqual(report["distilled"]["auxiliary"]["rules_with_scene_cues"], 1)
        self.assertEqual(report["distilled"]["auxiliary"]["rules_with_explore_cues"], 1)
        self.assertTrue(output_path.exists())

    def test_strict_mode_raises_on_count_mismatch(self) -> None:
        root = _case_dir()
        semantic_path = root / "semantic.json"
        distilled_path = root / "distilled.json"
        semantic_path.write_text(json.dumps({"samples": []}), encoding="utf-8")
        distilled_path.write_text(json.dumps({"rules": []}), encoding="utf-8")

        with self.assertRaises(SystemExit):
            analyze_run(
                semantic_path=semantic_path,
                distilled_path=distilled_path,
                expected_samples=3,
                output_path=None,
                strict=True,
            )

    def test_strict_mode_raises_on_failure_placeholders(self) -> None:
        root = _case_dir()
        semantic_path = root / "semantic.json"
        distilled_path = root / "distilled.json"
        semantic_path.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "sample_id": "s1",
                            "topic_guess": {"domain": "Unknown", "topic": "Unknown"},
                            "semantic_audit": {
                                "summary": "LLM调用失败，已记录重试占位。",
                                "key_errors": [{"message": "LLM调用失败", "evidence": "empty response"}],
                            },
                            "experience_rules": [],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        distilled_path.write_text(json.dumps({"rules": []}), encoding="utf-8")

        with self.assertRaises(SystemExit):
            analyze_run(
                semantic_path=semantic_path,
                distilled_path=distilled_path,
                expected_samples=1,
                output_path=None,
                strict=True,
            )


if __name__ == "__main__":
    unittest.main()
