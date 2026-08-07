from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from scripts import run_physics_eval_pipeline as pipeline


def _catalog() -> dict:
    return {
        "metadata": {
            "catalog_type": "unified_rules_v2",
            "schema_profile": "semantic_navigation_tree_minimal",
            "total_domains": 1,
            "topics_with_rules": 1,
            "total_scenario_clusters": 1,
            "total_executable_rules": 1,
        },
        "domains": [{"name": "Mechanics", "topics": []}],
    }


class PhysicsEvalPipelineManifestTest(unittest.TestCase):
    def test_semantic_failure_exit_can_be_preserved_without_aborting_pipeline(self) -> None:
        with patch.object(
            pipeline.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=2),
        ):
            return_code = pipeline._run("verifier command", allowed_returncodes=(0, 2))

        self.assertEqual(2, return_code)

    def test_unified_catalog_is_required(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            pipeline.build_parser().parse_args(["--skip-build", "--skip-run"])

        self.assertIn("--unified-catalog", stderr.getvalue())

    def test_default_child_interpreter_is_current_conda_python(self) -> None:
        args = pipeline.build_parser().parse_args(["--unified-catalog", "catalog.json"])

        self.assertEqual(sys.executable, args.python)
        self.assertNotIn(".venv", args.python)

    def test_invalid_catalog_fails_before_output_directory_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog = root / "catalog.json"
            output_dir = root / "out"
            catalog.write_text(json.dumps({"metadata": {}, "domains": []}), encoding="utf-8")
            stderr = io.StringIO()

            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                pipeline.cli(
                    [
                        "--unified-catalog",
                        str(catalog),
                        "--output-dir",
                        str(output_dir),
                        "--skip-build",
                        "--skip-run",
                        "--skip-error-eval",
                        "--skip-question-eval",
                        "--no-symbolic-check",
                    ]
                )

            self.assertFalse(output_dir.exists())
            self.assertIn("catalog_type='unified_rules_v2'", stderr.getvalue())

    def test_skip_pipeline_completes_portable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog = root / "catalog.json"
            out = root / "out"
            manifest = root / "tracked" / "run.json"
            catalog.write_text(json.dumps(_catalog()), encoding="utf-8")
            out.mkdir()
            (out / "error_eval_dataset_1.json").write_text('[{"id": "wrong"}]', encoding="utf-8")
            (out / "question_eval_dataset_1_1.json").write_text(
                '[{"id": "wrong"}, {"id": "right"}]',
                encoding="utf-8",
            )

            pipeline.cli(
                [
                    "--unified-catalog",
                    str(catalog),
                    "--output-dir",
                    str(out),
                    "--manifest-output",
                    str(manifest),
                    "--run-id",
                    "unit-development",
                    "--recall-size",
                    "1",
                    "--precision-size",
                    "1",
                    "--skip-build",
                    "--skip-run",
                    "--skip-error-eval",
                    "--skip-question-eval",
                    "--no-symbolic-check",
                ]
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual("completed", payload["status"])
        self.assertEqual("unit-development", payload["run_id"])
        self.assertTrue(payload["runtime"]["is_conda"])
        self.assertFalse(payload["llm_cache"])
        self.assertTrue(payload["continue_on_semantic_error"])
        self.assertEqual("unified_rules_v2", payload["unified_catalog"]["catalog_type"])
        self.assertEqual(1, payload["datasets"]["error_dataset"]["json_count"])
        self.assertEqual(2, payload["datasets"]["question_dataset"]["json_count"])
        self.assertIn("source_tree", payload["source_state"])
        serialized = json.dumps(payload)
        self.assertNotIn("OPENAI_API_KEY\": \"", serialized)

    def test_validation_dirty_worktree_is_rejected_and_manifest_is_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog = root / "catalog.json"
            out = root / "out"
            manifest = root / "run.json"
            catalog.write_text(json.dumps(_catalog()), encoding="utf-8")
            out.mkdir()
            (out / "error_eval_dataset_1.json").write_text("[]", encoding="utf-8")
            (out / "question_eval_dataset_1_1.json").write_text("[]", encoding="utf-8")
            dirty_state = {
                "git": {
                    "available": True,
                    "head": "a" * 40,
                    "branch": "test",
                    "dirty": True,
                    "status_sha256": "b" * 64,
                    "tracked_diff_sha256": "c" * 64,
                },
                "source_tree": {"sha256": "d" * 64, "file_count": 0, "files": {}},
            }

            with patch.object(pipeline, "capture_source_state", return_value=dirty_state):
                with self.assertRaises(RuntimeError):
                    pipeline.cli(
                        [
                            "--unified-catalog",
                            str(catalog),
                            "--output-dir",
                            str(out),
                            "--manifest-output",
                            str(manifest),
                            "--run-kind",
                            "validation",
                            "--recall-size",
                            "1",
                            "--precision-size",
                            "1",
                            "--skip-build",
                            "--skip-run",
                            "--skip-error-eval",
                            "--skip-question-eval",
                            "--no-symbolic-check",
                        ]
                    )
            payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual("failed", payload["status"])
        self.assertEqual("RuntimeError", payload["failure"]["type"])
        self.assertIn("clean Git worktree", payload["failure"]["message"])


if __name__ == "__main__":
    unittest.main()
