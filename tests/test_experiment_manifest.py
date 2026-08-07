from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.experiment_manifest import (
    ExperimentManifest,
    capture_git_state,
    fingerprint_file,
    fingerprint_source_tree,
    probe_python_runtime,
    validate_unified_catalog,
)


def _catalog() -> dict:
    return {
        "metadata": {
            "catalog_type": "unified_rules_v2",
            "schema_profile": "semantic_navigation_tree_minimal",
            "total_domains": 1,
            "total_executable_rules": 1,
        },
        "domains": [{"name": "Mechanics", "topics": []}],
    }


class ExperimentManifestTest(unittest.TestCase):
    def test_file_fingerprint_is_portable_and_records_json_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "data" / "rows.json"
            path.parent.mkdir()
            raw = b'[{"id": 1}, {"id": 2}]'
            path.write_bytes(raw)

            fingerprint = fingerprint_file(
                "data/rows.json",
                project_root=root,
                required=True,
                json_count=True,
            )

        self.assertEqual("data/rows.json", fingerprint["path"])
        self.assertEqual(len(raw), fingerprint["size_bytes"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), fingerprint["sha256"])
        self.assertEqual(2, fingerprint["json_count"])

    def test_required_missing_file_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                fingerprint_file(
                    "missing.json",
                    project_root=Path(temp_dir),
                    required=True,
                )

    def test_unified_catalog_schema_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid = root / "valid.json"
            invalid = root / "invalid.json"
            valid.write_text(json.dumps(_catalog()), encoding="utf-8")
            invalid.write_text(json.dumps({"metadata": {}, "domains": []}), encoding="utf-8")

            fingerprint = validate_unified_catalog(valid, project_root=root)
            with self.assertRaises(ValueError):
                validate_unified_catalog(invalid, project_root=root)

        self.assertEqual("unified_rules_v2", fingerprint["catalog_type"])
        self.assertEqual(1, fingerprint["total_executable_rules"])

    def test_source_tree_hash_changes_when_dirty_source_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "core" / "module.py"
            source.parent.mkdir()
            source.write_text("VALUE = 1\n", encoding="utf-8")
            before = fingerprint_source_tree(root)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            after = fingerprint_source_tree(root)

        self.assertNotEqual(before["sha256"], after["sha256"])
        self.assertNotEqual(
            before["files"]["core/module.py"]["sha256"],
            after["files"]["core/module.py"]["sha256"],
        )

    def test_git_state_records_content_sensitive_diff_hash(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        state = capture_git_state(project_root)

        self.assertTrue(state["available"])
        self.assertIn("head", state)
        self.assertIn("branch", state)
        self.assertEqual(64, len(state["status_sha256"]))
        self.assertEqual(64, len(state["tracked_diff_sha256"]))

    def test_current_test_runtime_is_conda(self) -> None:
        runtime = probe_python_runtime(sys.executable, require_conda=True)

        self.assertTrue(runtime["is_conda"])
        self.assertEqual(str(Path(sys.executable).resolve()), runtime["executable"])
        self.assertEqual(64, len(runtime["package_set_sha256"]))

    def test_failure_refreshes_artifacts_and_redacts_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "result.json"
            manifest_path = root / "manifest.json"
            recorder = ExperimentManifest(
                manifest_path,
                {"run_id": "unit-test"},
                project_root=root,
                artifact_groups={"artifacts": {"result": artifact}},
            )
            recorder.start_stage("verifier")
            artifact.write_text("[]", encoding="utf-8")
            secret = "unit-test-secret-value"
            with patch.dict(os.environ, {"OPENAI_API_KEY": secret}):
                recorder.fail(RuntimeError(f"request failed for {secret}"))

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual("failed", payload["status"])
        self.assertTrue(payload["artifacts"]["result"]["exists"])
        self.assertNotIn(secret, json.dumps(payload))
        self.assertIn("[REDACTED]", payload["failure"]["message"])
        self.assertFalse(manifest_path.with_name("manifest.json.tmp").exists())

    def test_completion_records_dataset_and_output_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.json"
            output = root / "output.json"
            manifest_path = root / "manifest.json"
            dataset.write_text('[{"id": 1}]', encoding="utf-8")
            output.write_text("[]", encoding="utf-8")
            recorder = ExperimentManifest(
                manifest_path,
                {"run_id": "completed-test"},
                project_root=root,
                artifact_groups={
                    "datasets": {"question": dataset},
                    "artifacts": {"results": output},
                },
            )
            recorder.complete()
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(1, payload["schema_version"])
        self.assertEqual("completed", payload["status"])
        self.assertEqual(1, payload["datasets"]["question"]["json_count"])
        self.assertEqual(64, len(payload["artifacts"]["results"]["sha256"]))


if __name__ == "__main__":
    unittest.main()
