from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.check_server_run_inputs import check_inputs

TMP_ROOT = Path(tempfile.mkdtemp(prefix="physicsverifier_server_input_tests_"))
atexit.register(shutil.rmtree, TMP_ROOT, ignore_errors=True)


def _case_dir() -> Path:
    path = TMP_ROOT / f"physicsverifier_server_inputs_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class ServerRunInputsTests(unittest.TestCase):
    def test_check_inputs_reports_ready_when_required_files_and_schema_exist(self) -> None:
        root = _case_dir()
        sample_path = root / "sample.json"
        script_path = root / "generate_experience_rules.py"
        rules_catalog_path = root / "rules_catalog_top_down.json"
        unified_catalog_path = root / "rules_unified.json"
        output_path = root / "preflight.json"

        sample_path.write_text(
            json.dumps(
                [
                    {"id": "1", "question": "q1", "prediction": "p1", "answer": "a1"},
                    {"id": "2", "question": "q2", "prediction": "p2", "answer": "a2"},
                ]
            ),
            encoding="utf-8",
        )
        script_path.write_text('"auxiliary" "node_summary" "scene_cues" "boundary_cues" "explore_cues"', encoding="utf-8")
        rules_catalog_path.write_text(json.dumps({"domains": []}), encoding="utf-8")
        unified_catalog_path.write_text(
            json.dumps({"metadata": {"schema_profile": "semantic_navigation_tree_minimal"}}),
            encoding="utf-8",
        )

        report = check_inputs(
            sample_path=sample_path,
            expected_samples=2,
            run_script_path=script_path,
            rules_catalog_path=rules_catalog_path,
            unified_catalog_path=unified_catalog_path,
            output_path=output_path,
        )

        self.assertTrue(report["ready"])
        self.assertEqual(report["sample"]["count"], 2)
        self.assertEqual(report["sample"]["empty_required_field_count"], 0)
        self.assertTrue(report["run_script"]["has_auxiliary_schema"])
        self.assertEqual(report["unified_catalog"]["schema_profile"], "semantic_navigation_tree_minimal")
        self.assertTrue(output_path.exists())

    def test_check_inputs_reports_not_ready_for_missing_auxiliary_schema(self) -> None:
        root = _case_dir()
        sample_path = root / "sample.json"
        script_path = root / "generate_experience_rules.py"
        rules_catalog_path = root / "rules_catalog_top_down.json"
        unified_catalog_path = root / "rules_unified.json"

        sample_path.write_text(json.dumps([{"id": "1", "question": "q", "prediction": "p", "answer": "a"}]), encoding="utf-8")
        script_path.write_text("old prompt", encoding="utf-8")
        rules_catalog_path.write_text(json.dumps({"domains": []}), encoding="utf-8")
        unified_catalog_path.write_text(json.dumps({"metadata": {"schema_profile": "semantic_navigation_tree_minimal"}}), encoding="utf-8")

        report = check_inputs(
            sample_path=sample_path,
            expected_samples=1,
            run_script_path=script_path,
            rules_catalog_path=rules_catalog_path,
            unified_catalog_path=unified_catalog_path,
            output_path=None,
        )

        self.assertFalse(report["ready"])
        self.assertIn("run_script_missing_auxiliary_schema", report["failures"])


if __name__ == "__main__":
    unittest.main()
