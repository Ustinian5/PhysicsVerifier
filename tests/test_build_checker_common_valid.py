from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
from unittest.mock import patch

from scripts import evaluate_question_level_sets
from scripts.build_checker_common_valid import (
    CHECKER_GATE_MODES,
    CommonValidError,
    audit_replay_artifacts,
    build_common_valid,
)


def _canonical_sha256(value: Any) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_row(sample_id: Any) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "candidate_source": "semantic_retrieval",
        "selection_strategy": "semantic_tree_selection",
        "semantic_selection_error": "",
        "checker_status": "valid_empty",
        "checker_failure_count": 0,
        "checker_failures": [],
        "replay_completed": True,
        "diagnostics": [],
    }


class CheckerCommonValidTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        # Integer 0 must remain valid, while integer 1 and string "1" must stay
        # distinct throughout the paired-subset projection.
        self.dataset = [
            {"id": 0, "question": "Q0", "prediction": "P0"},
            {"id": 1, "question": "Q1", "prediction": "P1"},
            {"id": "1", "question": "Qs", "prediction": "Ps"},
            {"id": "tail", "question": "Qt", "prediction": "Pt"},
        ]
        self.dataset_path = self.root / "synthetic_dataset.json"
        self._write(self.dataset_path, self.dataset)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def _write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _frozen_input_paths(self) -> Tuple[Path, Path, Path]:
        retrieval_path = self.root / "synthetic-retrieval.json"
        catalog_path = self.root / "synthetic-catalog.json"
        frozen_manifest_path = self.root / "synthetic-frozen-manifest.json"
        if not retrieval_path.exists():
            self._write(retrieval_path, {"artifact": "synthetic-retrieval"})
        if not catalog_path.exists():
            self._write(catalog_path, {"artifact": "synthetic-catalog"})
        if not frozen_manifest_path.exists():
            self._write(frozen_manifest_path, {"artifact": "synthetic-manifest"})
        return retrieval_path, catalog_path, frozen_manifest_path

    def _base_configuration(self, mode: str) -> Dict[str, Any]:
        retrieval_path, catalog_path, frozen_manifest_path = self._frozen_input_paths()
        return {
            "run_kind": "development",
            "system_arm": mode,
            "checker_model": "synthetic-qwen30b",
            "checker_prompt_version": "synthetic-checker-v1",
            "checker_json_attempts": 3,
            "checker_min_confidence": 0.8,
            "checker_cache_enabled": False,
            "candidate_source": "semantic_retrieval",
            "retrieval_source": "frozen_semantic_trace",
            "retrieval_execution": False,
            "bottom_up_enabled": False,
            "target_mapping_sha256": _canonical_sha256([]),
            "llm_trace_include_prompts": False,
            "llm_temperature": 0.1,
            "llm_max_output_tokens": 2048,
            "llm_trace_path": str(self.root / f"{mode}.llm_trace.jsonl"),
            "dataset_sha256": _file_sha256(self.dataset_path),
            "retrieval_trace_sha256": _file_sha256(retrieval_path),
            "unified_catalog_sha256": _file_sha256(catalog_path),
            "selection_projection_sha256": "3" * 64,
            "frozen_manifest_sha256": _file_sha256(frozen_manifest_path),
            "source_identity": {
                "git_available": True,
                "git_head": "5" * 40,
                "git_dirty": False,
                "source_tree_sha256": "5" * 64,
                "source_file_count": 100,
            },
            "runtime_identity": {
                "executable": "/synthetic/conda/bin/python",
                "python_version": "3.11.0",
                "is_conda": True,
                "conda_env": "physicsverifier",
                "package_set_sha256": "6" * 64,
                "package_count": 10,
            },
            "api_transport_identity": {
                "endpoint_source": "OPENAI_BASE_URL",
                "endpoint_sha256": "7" * 64,
                "disable_thinking": True,
            },
        }

    def _write_arm(
        self,
        mode: str,
        rows: List[Dict[str, Any]],
        *,
        configuration_mutation: Any = None,
    ) -> Tuple[Path, Path]:
        configuration = self._base_configuration(mode)
        if configuration_mutation is not None:
            configuration_mutation(configuration)
        configuration_sha256 = _canonical_sha256(configuration)
        prepared_rows = copy.deepcopy(rows)
        for row in prepared_rows:
            row["checker_gate_mode"] = mode
            row["replay_config_sha256"] = configuration_sha256

        result_path = self.root / f"{mode}.results.json"
        report_path = self.root / f"{mode}.report.json"
        trace_path = Path(configuration["llm_trace_path"])
        self._write(result_path, prepared_rows)
        trace_fingerprint = self._write_trace(
            trace_path,
            [
                {
                    "parse_status": "valid_object",
                    "raw_response": "{}",
                    "synthetic_sample_id": row["id"],
                }
                for row in prepared_rows
            ],
        )
        failed_samples = sum(
            1
            for row in prepared_rows
            if row.get("replay_completed") is not True
            or row.get("checker_failure_count", 0) > 0
            or row.get("selection_strategy")
            in {"semantic_error", "semantic_unavailable"}
            or bool(row.get("semantic_selection_error"))
        )
        retryable = sum(
            1 for row in prepared_rows if row.get("replay_completed") is not True
        )
        status = "complete"
        if failed_samples:
            status = "incomplete_failures" if retryable else "complete_with_failures"
        candidate_source = str(configuration.get("candidate_source"))
        retrieval_path, catalog_path, frozen_manifest_path = self._frozen_input_paths()
        report = {
            "schema_version": 1,
            "report_type": (
                "checker_only_frozen_target_binding_replay"
                if candidate_source == "frozen_target_binding"
                else "checker_only_frozen_retrieval_replay"
            ),
            "candidate_source": candidate_source,
            "target_mapping_sha256": configuration["target_mapping_sha256"],
            "status": status,
            "configuration": configuration,
            "configuration_sha256": configuration_sha256,
            "inputs": {
                "dataset": {
                    "path": str(self.dataset_path),
                    "sha256": configuration["dataset_sha256"],
                },
                "retrieval_trace": {
                    "path": str(retrieval_path),
                    "sha256": configuration["retrieval_trace_sha256"],
                },
                "unified_catalog": {
                    "path": str(catalog_path),
                    "sha256": configuration["unified_catalog_sha256"],
                },
            },
            "frozen_manifest": {
                "path": str(frozen_manifest_path),
                "sha256": configuration["frozen_manifest_sha256"],
                "selection_projection_sha256": configuration[
                    "selection_projection_sha256"
                ],
            },
            "llm_trace": trace_fingerprint,
            "output": {
                "path": str(result_path),
                "sha256": _file_sha256(result_path),
                "record_count": len(prepared_rows),
            },
            "statistics": {
                "total_samples": len(self.dataset),
                "processed_samples": len(prepared_rows),
                "pending_samples": 0,
                "completed_samples": len(prepared_rows) - retryable,
                "failed_samples": failed_samples,
            },
        }
        self._write(report_path, report)
        return result_path, report_path

    @staticmethod
    def _write_trace(path: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = b"".join(
            (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            for record in records
        )
        path.write_bytes(encoded)
        parse_status_counts: Dict[str, int] = {}
        for record in records:
            status = str(record.get("parse_status") or "")
            parse_status_counts[status] = parse_status_counts.get(status, 0) + 1
        prompt_count = sum(
            1
            for record in records
            if "system_prompt" in record or "user_prompt" in record
        )
        return {
            "path": str(path),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
            "record_count": len(records),
            "raw_response_record_count": sum(
                1 for record in records if "raw_response" in record
            ),
            "prompt_record_count": prompt_count,
            "prompts_included": prompt_count > 0,
            "parse_status_counts": dict(sorted(parse_status_counts.items())),
        }

    def _three_arms(
        self,
        *,
        mutations: Mapping[str, Any] | None = None,
    ) -> Dict[str, Tuple[Path, Path]]:
        rows_by_mode: Dict[str, List[Dict[str, Any]]] = {
            mode: [_valid_row(row["id"]) for row in self.dataset]
            for mode in CHECKER_GATE_MODES
        }
        # Each arm loses a different non-zero/string sample; only integer ID 0
        # is common-valid.
        rows_by_mode["legacy"][1].update(
            {
                "checker_status": "failed",
                "checker_failure_count": 1,
                "checker_failures": [{"kind": "transport_failure"}],
                "replay_completed": False,
            }
        )
        rows_by_mode["dual_evidence"][2].update(
            {
                "selection_strategy": "semantic_error",
                "semantic_selection_error": "synthetic retrieval failure",
                "checker_status": "not_run_frozen_retrieval_error",
                "checker_failure_count": 1,
                "checker_failures": [{"kind": "frozen_retrieval_error"}],
            }
        )
        rows_by_mode["dual_evidence_consistency"][3].update(
            {
                "checker_status": "partial_failure",
                "checker_failure_count": 1,
                "checker_failures": [{"kind": "schema_failure"}],
                "replay_completed": False,
            }
        )
        return {
            mode: self._write_arm(
                mode,
                rows_by_mode[mode],
                configuration_mutation=(mutations or {}).get(mode),
            )
            for mode in CHECKER_GATE_MODES
        }

    def test_builds_typed_common_subset_and_immutable_manifest(self) -> None:
        arms = self._three_arms()
        output_dataset = self.root / "common.json"
        manifest_path = self.root / "common.manifest.json"

        manifest = build_common_valid(
            dataset_path=self.dataset_path,
            arms=arms,
            output_dataset_path=output_dataset,
            manifest_path=manifest_path,
        )

        common_rows = json.loads(output_dataset.read_text(encoding="utf-8"))
        self.assertEqual([row["id"] for row in common_rows], [0])
        self.assertTrue(manifest["immutable"])
        self.assertEqual(manifest["mode_order"], list(CHECKER_GATE_MODES))
        self.assertEqual(manifest["common_valid"]["record_count"], 1)
        self.assertEqual(manifest["common_valid"]["coverage"], 0.25)
        self.assertEqual(
            manifest["common_valid"]["ordered_typed_ids"],
            [{"type": "int", "value": 0}],
        )
        self.assertEqual(
            manifest["common_valid"]["dataset"]["sha256"],
            _file_sha256(output_dataset),
        )
        for mode in CHECKER_GATE_MODES:
            arm_manifest = manifest["inputs"]["arms"][mode]
            self.assertEqual(
                arm_manifest["results"]["sha256"],
                _file_sha256(arms[mode][0]),
            )
            self.assertEqual(
                arm_manifest["report"]["sha256"],
                _file_sha256(arms[mode][1]),
            )
            self.assertEqual(arm_manifest["coverage"], 0.75)

        audited = audit_replay_artifacts(
            dataset_path=self.dataset_path,
            mode="legacy",
            result_path=arms["legacy"][0],
            report_path=arms["legacy"][1],
        )
        self.assertEqual(len(audited["rows"]), len(self.dataset))
        self.assertEqual(audited["configuration"]["system_arm"], "legacy")
        self.assertIn("checker_failure", audited["failure_by_stage"])
        self.assertEqual(
            audited["actual_llm_trace"]["sha256"],
            audited["reported_llm_trace"]["sha256"],
        )

        with self.assertRaisesRegex(CommonValidError, "refusing to overwrite"):
            build_common_valid(
                dataset_path=self.dataset_path,
                arms=arms,
                output_dataset_path=output_dataset,
                manifest_path=manifest_path,
            )

    def test_accepts_successful_target_binding_strategy(self) -> None:
        for sample in self.dataset:
            sample.update(
                {
                    "target_rule_id": "synthetic_rule",
                    "target_rule": {"rule_id": "synthetic_rule"},
                }
            )
        self._write(self.dataset_path, self.dataset)
        rows = []
        for sample in self.dataset:
            row = _valid_row(sample["id"])
            row.update(
                {
                    "candidate_source": "frozen_target_binding",
                    "selection_strategy": "target_rule_binding",
                    "unified_retrieval_mode": "target_binding",
                    "retrieval_score_kind": "fixed_control_0_1",
                    "target_rule_id": "synthetic_rule",
                    "retrieved_rules": [
                        {
                            "rule_id": "synthetic_rule",
                            "domain": "Synthetic",
                            "topic_id": "synthetic.topic",
                            "topic": "Synthetic Topic",
                            "score": 1.0,
                            "score_kind": "fixed_control_0_1",
                            "partial": False,
                            "executable": True,
                            "publish_gate": {
                                "publishable": True,
                                "reasons": [],
                                "score": 1.0,
                                "score_kind": "fixed_control_0_1",
                                "min_publish_score": 0.0,
                                "selection_strategy": "target_rule_binding",
                            },
                        }
                    ],
                    "used_rules": ["synthetic_rule"],
                }
            )
            rows.append(row)

        def target_configuration(configuration: Dict[str, Any]) -> None:
            configuration.update(
                {
                    "candidate_source": "frozen_target_binding",
                    "retrieval_source": "frozen_target_binding",
                    "target_mapping_sha256": "8" * 64,
                }
            )

        arms = {
            mode: self._write_arm(
                mode,
                rows,
                configuration_mutation=target_configuration,
            )
            for mode in CHECKER_GATE_MODES
        }
        output_dataset = self.root / "target-common.json"
        manifest_path = self.root / "target-common.manifest.json"
        manifest = build_common_valid(
            dataset_path=self.dataset_path,
            arms=arms,
            output_dataset_path=output_dataset,
            manifest_path=manifest_path,
        )

        self.assertEqual(manifest["common_valid"]["record_count"], len(self.dataset))
        self.assertEqual(
            manifest["shared_configuration"]["candidate_source"],
            "frozen_target_binding",
        )

        legacy_results, legacy_report = arms["legacy"]
        tampered_rows = json.loads(legacy_results.read_text(encoding="utf-8"))
        tampered_rows[0]["target_rule_id"] = "different_rule"
        self._write(legacy_results, tampered_rows)
        tampered_report = json.loads(legacy_report.read_text(encoding="utf-8"))
        tampered_report["output"]["sha256"] = _file_sha256(legacy_results)
        self._write(legacy_report, tampered_report)
        with self.assertRaisesRegex(CommonValidError, "does not match dataset"):
            build_common_valid(
                dataset_path=self.dataset_path,
                arms=arms,
                output_dataset_path=self.root / "target-common-tampered.json",
                manifest_path=self.root / "target-common-tampered.manifest.json",
            )

    def test_accepts_int_and_string_same_value_but_rejects_duplicate_typed_id(self) -> None:
        arms = self._three_arms()
        # The successful build above already proves int 1 and string "1" do not
        # collide.  A repeated integer ID must fail before reports are trusted.
        duplicated = copy.deepcopy(self.dataset)
        duplicated[2]["id"] = 1
        duplicate_path = self.root / "duplicate_dataset.json"
        self._write(duplicate_path, duplicated)
        with self.assertRaisesRegex(CommonValidError, "duplicate typed ID"):
            build_common_valid(
                dataset_path=duplicate_path,
                arms=arms,
                output_dataset_path=self.root / "unused-common.json",
                manifest_path=self.root / "unused-manifest.json",
            )

        result_path, report_path = arms["legacy"]
        result_rows = json.loads(result_path.read_text(encoding="utf-8"))
        result_rows[2]["id"] = 1
        self._write(result_path, result_rows)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["output"]["sha256"] = _file_sha256(result_path)
        self._write(report_path, report)
        with self.assertRaisesRegex(CommonValidError, "duplicate typed ID"):
            build_common_valid(
                dataset_path=self.dataset_path,
                arms=arms,
                output_dataset_path=self.root / "unused-common-2.json",
                manifest_path=self.root / "unused-manifest-2.json",
            )

    def test_common_dataset_is_directly_usable_by_typed_id_evaluator(self) -> None:
        arms = self._three_arms()
        output_dataset = self.root / "common.json"
        manifest_path = self.root / "common.manifest.json"
        build_common_valid(
            dataset_path=self.dataset_path,
            arms=arms,
            output_dataset_path=output_dataset,
            manifest_path=manifest_path,
        )
        audit_path = self.root / "empty-audit.json"
        metrics_path = self.root / "legacy-common.metrics.json"
        self._write(audit_path, [])
        argv = [
            "evaluate_question_level_sets.py",
            "--dataset",
            str(output_dataset),
            "--results",
            str(arms["legacy"][0]),
            "--audit",
            str(audit_path),
            "--output",
            str(metrics_path),
        ]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            evaluate_question_level_sets.main()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(metrics["summary"]["dataset_size"], 1)
        self.assertEqual(metrics["summary"]["scored_size"], 1)
        self.assertEqual(metrics["summary"]["tn"], 1)
        self.assertEqual(metrics["details"][0]["id"], "0")

    def test_requires_exactly_three_distinct_modes(self) -> None:
        arms = self._three_arms()
        arms.pop("legacy")
        with self.assertRaisesRegex(CommonValidError, "exactly one result/report pair"):
            build_common_valid(
                dataset_path=self.dataset_path,
                arms=arms,
                output_dataset_path=self.root / "common.json",
                manifest_path=self.root / "manifest.json",
            )

    def test_rejects_result_sha_mismatch_from_report(self) -> None:
        arms = self._three_arms()
        result_path = arms["legacy"][0]
        rows = json.loads(result_path.read_text(encoding="utf-8"))
        rows[0]["diagnostics"] = [{"message": "tampered"}]
        self._write(result_path, rows)

        with self.assertRaisesRegex(CommonValidError, "output SHA256"):
            build_common_valid(
                dataset_path=self.dataset_path,
                arms=arms,
                output_dataset_path=self.root / "common.json",
                manifest_path=self.root / "manifest.json",
            )

    def test_rejects_source_runtime_api_or_input_configuration_drift(self) -> None:
        mutations = {
            "source": (
                lambda config: config["source_identity"].update(
                    {"source_tree_sha256": "a" * 64}
                ),
                "differs outside the arm allowlist",
            ),
            "runtime": (
                lambda config: config["runtime_identity"].update(
                    {"package_set_sha256": "b" * 64}
                ),
                "differs outside the arm allowlist",
            ),
            "api": (
                lambda config: config["api_transport_identity"].update(
                    {"endpoint_sha256": "c" * 64}
                ),
                "differs outside the arm allowlist",
            ),
            "input": (
                lambda config: config.update(
                    {"retrieval_trace_sha256": "d" * 64}
                ),
                "actual file",
            ),
        }
        for label, (mutation, expected) in mutations.items():
            with self.subTest(label=label):
                scenario = self.root / label
                scenario.mkdir()
                original_root = self.root
                try:
                    self.root = scenario
                    self.dataset_path = scenario / "synthetic_dataset.json"
                    self._write(self.dataset_path, self.dataset)
                    arms = self._three_arms(
                        mutations={"dual_evidence": mutation}
                    )
                    with self.assertRaisesRegex(
                        CommonValidError,
                        expected,
                    ):
                        build_common_valid(
                            dataset_path=self.dataset_path,
                            arms=arms,
                            output_dataset_path=scenario / "common.json",
                            manifest_path=scenario / "manifest.json",
                        )
                finally:
                    self.root = original_root
                    self.dataset_path = original_root / "synthetic_dataset.json"

    def test_rejects_checker_cache_even_when_all_arms_enable_it(self) -> None:
        arms = self._three_arms(
            mutations={
                mode: lambda config: config.update(
                    {"checker_cache_enabled": True}
                )
                for mode in CHECKER_GATE_MODES
            }
        )
        with self.assertRaisesRegex(CommonValidError, "disable the Checker cache"):
            build_common_valid(
                dataset_path=self.dataset_path,
                arms=arms,
                output_dataset_path=self.root / "common.json",
                manifest_path=self.root / "manifest.json",
            )

    def test_binds_actual_llm_trace_and_rejects_missing_or_tampered_file(self) -> None:
        for scenario in ("missing", "tampered"):
            with self.subTest(scenario=scenario):
                scenario_root = self.root / scenario
                scenario_root.mkdir()
                original_root = self.root
                original_dataset_path = self.dataset_path
                try:
                    self.root = scenario_root
                    self.dataset_path = scenario_root / "synthetic_dataset.json"
                    self._write(self.dataset_path, self.dataset)
                    arms = self._three_arms()
                    trace_path = scenario_root / "legacy.llm_trace.jsonl"
                    if scenario == "missing":
                        trace_path.unlink()
                        expected = "LLM trace does not exist"
                    else:
                        with trace_path.open("ab") as handle:
                            handle.write(
                                b'{"parse_status":"valid_object","raw_response":"tampered"}\n'
                            )
                        expected = "actual trace"
                    with self.assertRaisesRegex(CommonValidError, expected):
                        build_common_valid(
                            dataset_path=self.dataset_path,
                            arms=arms,
                            output_dataset_path=scenario_root / "common.json",
                            manifest_path=scenario_root / "manifest.json",
                        )
                finally:
                    self.root = original_root
                    self.dataset_path = original_dataset_path

    def test_rejects_prompt_fields_even_with_matching_trace_fingerprint(self) -> None:
        arms = self._three_arms()
        trace_path = self.root / "legacy.llm_trace.jsonl"
        fingerprint = self._write_trace(
            trace_path,
            [
                {
                    "parse_status": "valid_object",
                    "raw_response": "{}",
                    "system_prompt": "synthetic secret prompt",
                }
            ],
        )
        report_path = arms["legacy"][1]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["llm_trace"] = fingerprint
        self._write(report_path, report)
        with self.assertRaisesRegex(CommonValidError, "must not contain prompt fields"):
            build_common_valid(
                dataset_path=self.dataset_path,
                arms=arms,
                output_dataset_path=self.root / "common.json",
                manifest_path=self.root / "manifest.json",
            )

    def test_rejects_shared_llm_trace_path_between_arms(self) -> None:
        shared_path = str(self.root / "legacy.llm_trace.jsonl")
        arms = self._three_arms(
            mutations={
                "dual_evidence": lambda config: config.update(
                    {"llm_trace_path": shared_path}
                )
            }
        )
        with self.assertRaisesRegex(CommonValidError, "distinct LLM trace path"):
            build_common_valid(
                dataset_path=self.dataset_path,
                arms=arms,
                output_dataset_path=self.root / "common.json",
                manifest_path=self.root / "manifest.json",
            )

    def test_rejects_non_string_sha_and_boolean_schema_version(self) -> None:
        scenarios = {
            "integer_sha": (
                {"dual_evidence": lambda config: config["source_identity"].update(
                    {"source_tree_sha256": int("5" * 64)}
                )},
                None,
                "source_tree_sha256 is invalid",
            ),
            "boolean_schema": ({}, True, "unsupported schema_version"),
        }
        for scenario, (mutations, schema_version, expected) in scenarios.items():
            with self.subTest(scenario=scenario):
                scenario_root = self.root / scenario
                scenario_root.mkdir()
                original_root = self.root
                original_dataset_path = self.dataset_path
                try:
                    self.root = scenario_root
                    self.dataset_path = scenario_root / "synthetic_dataset.json"
                    self._write(self.dataset_path, self.dataset)
                    arms = self._three_arms(mutations=mutations)
                    if schema_version is not None:
                        report_path = arms["legacy"][1]
                        report = json.loads(report_path.read_text(encoding="utf-8"))
                        report["schema_version"] = schema_version
                        self._write(report_path, report)
                    with self.assertRaisesRegex(CommonValidError, expected):
                        build_common_valid(
                            dataset_path=self.dataset_path,
                            arms=arms,
                            output_dataset_path=scenario_root / "common.json",
                            manifest_path=scenario_root / "manifest.json",
                        )
                finally:
                    self.root = original_root
                    self.dataset_path = original_dataset_path


if __name__ == "__main__":
    unittest.main()
