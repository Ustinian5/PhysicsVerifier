from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Mapping
from unittest.mock import patch

from scripts.build_checker_mechanism_dataset import (
    EXPECTED_CATALOG_SHA256,
    MECHANISMS,
    MechanismDatasetError,
    _expected_for,
    _api_transport_identity,
    _candidate_trace_rows,
    _normalize_provider_response,
    _object_sha256,
    _prompt,
    _resume_configuration_compatible,
    _sha256_file,
    _validate_model,
    _validate_generated_payload,
    audit_generation_artifacts,
    build_mechanism_dataset,
    build_parser,
    prepare_mechanism_plan,
)
from scripts.run_checker_replay import _catalog_index, _validate_frozen_trace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "catalogs" / "rules_unified_3000.json"


def _span(source: str, text: str, quote: str) -> Dict[str, Any]:
    start = text.index(quote)
    return {
        "source": source,
        "quote": quote,
        "start_char": start,
        "end_char": start + len(quote),
    }


def _payload(rule: Mapping[str, Any]) -> Dict[str, Any]:
    cases = []
    for index, mechanism in enumerate(MECHANISMS):
        marker = f"{rule['rule_id']}-{index}"
        applicability_quote = f"condition {marker} applies"
        question = f"Synthetic question: {applicability_quote}."
        context = f"Synthetic context for {marker}."
        subtype = rule["fifth_mechanism_subtype"] if index == 4 else "none"
        applicability = []
        if mechanism != "symbol_overlap_inapplicable":
            applicability = [_span("question", question, applicability_quote)]
        violation = []
        superseded = []
        correction = []
        if mechanism == "true_violation":
            violation_quote = f"wrong relation {marker}"
            prediction = f"I assert the {violation_quote} as my final result."
            violation = [_span("prediction", prediction, violation_quote)]
        elif mechanism == "applicable_correct":
            prediction = f"I apply the catalog relation correctly for {marker}."
        elif mechanism == "symbol_overlap_inapplicable":
            prediction = f"The symbol appears in an unrelated definition for {marker}."
        elif mechanism == "equivalent_alternative":
            prediction = f"I use an algebraically equivalent alternative for {marker}."
        elif subtype == "self_corrected":
            superseded_quote = f"wrong claim {marker}"
            correction_quote = f"corrected claim {marker}"
            prediction = (
                f"Initially I state the {superseded_quote}. I explicitly withdraw it. "
                f"My final answer uses the {correction_quote}."
            )
            superseded = [_span("prediction", prediction, superseded_quote)]
            correction = [_span("prediction", prediction, correction_quote)]
        else:
            prediction = f"The supplied information cannot determine the relation for {marker}."
        cases.append(
            {
                "mechanism": mechanism,
                "mechanism_subtype": subtype,
                "question": question,
                "context": context,
                "prediction": prediction,
                "expected": _expected_for(mechanism, subtype),
                "gt_evidence": {
                    "applicability_spans": applicability,
                    "violation_spans": violation,
                    "superseded_claim_spans": superseded,
                    "correction_spans": correction,
                },
            }
        )
    return {"cases": cases}


class BuildCheckerMechanismDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.plan_path = self.root / "plan.json"
        self.plan = prepare_mechanism_plan(
            catalog_path=CATALOG_PATH,
            plan_path=self.plan_path,
            expected_catalog_sha256=EXPECTED_CATALOG_SHA256,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _paths(self) -> Dict[str, Path]:
        return {
            "dataset_path": self.root / "dataset.json",
            "candidate_trace_path": self.root / "candidate_trace.json",
            "raw_trace_path": self.root / "raw_trace.jsonl",
            "checkpoint_path": self.root / "checkpoint.json",
            "manifest_path": self.root / "generation_manifest.json",
        }

    def test_plan_has_exact_registered_quota_and_is_immutable(self) -> None:
        self.assertEqual(self.plan["catalog"]["sha256"], EXPECTED_CATALOG_SHA256)
        self.assertEqual(len(self.plan["rules"]), 60)
        self.assertEqual(self.plan["summary"]["origin_counts"], {"exp": 30, "gen": 30})
        self.assertEqual(
            self.plan["summary"]["trigger_scope_proxy_counts"],
            {"broad_proxy": 30, "narrow_proxy": 30},
        )
        self.assertEqual(
            self.plan["summary"]["has_symbolic_primitive_counts"],
            {"false": 14, "true": 46},
        )
        self.assertEqual(
            self.plan["summary"]["fifth_mechanism_subtype_counts"],
            {"insufficient_information": 30, "self_corrected": 30},
        )
        self.assertEqual(
            {row["fifth_mechanism_subtype"] for row in self.plan["rules"][:30]},
            {"self_corrected"},
        )
        self.assertEqual(
            {row["fifth_mechanism_subtype"] for row in self.plan["rules"][30:]},
            {"insufficient_information"},
        )
        self.assertFalse(self.plan["trigger_scope_proxy"]["semantic_breadth_claim"])
        with self.assertRaisesRegex(MechanismDatasetError, "immutable"):
            prepare_mechanism_plan(
                catalog_path=CATALOG_PATH,
                plan_path=self.plan_path,
                expected_catalog_sha256=EXPECTED_CATALOG_SHA256,
            )

    def test_generated_schema_rejects_bad_quote_boolean_and_duplicates(self) -> None:
        rule = self.plan["rules"][0]
        valid = _payload(rule)
        normalized = _validate_generated_payload(valid, rule=rule)
        self.assertEqual(len(normalized), 5)

        wrong_model_offset = copy.deepcopy(valid)
        wrong_model_offset["cases"][0]["gt_evidence"]["violation_spans"][0]["start_char"] += 1
        repaired = _validate_generated_payload(wrong_model_offset, rule=rule)
        self.assertEqual(
            repaired[0]["gt_evidence"]["violation_spans"],
            normalized[0]["gt_evidence"]["violation_spans"],
        )

        bad_quote = copy.deepcopy(valid)
        bad_quote["cases"][0]["gt_evidence"]["violation_spans"][0]["quote"] = "not present"
        with self.assertRaisesRegex(MechanismDatasetError, "occur exactly once"):
            _validate_generated_payload(bad_quote, rule=rule)

        bad_boolean = copy.deepcopy(valid)
        bad_boolean["cases"][0]["expected"]["publish"] = 1
        with self.assertRaisesRegex(MechanismDatasetError, "must be a boolean"):
            _validate_generated_payload(bad_boolean, rule=rule)

        duplicate = copy.deepcopy(valid)
        for field in ("question", "context", "prediction"):
            duplicate["cases"][1][field] = duplicate["cases"][0][field]
        with self.assertRaisesRegex(MechanismDatasetError, "duplicates"):
            _validate_generated_payload(duplicate, rule=rule)

    def test_max_rules_smoke_outputs_five_cases_and_honest_target_trace(self) -> None:
        calls = []

        def provider(rule: Mapping[str, Any], system: str, user: str, attempt: int) -> str:
            calls.append((rule["rule_id"], attempt, system, user))
            return json.dumps(_payload(rule), ensure_ascii=False)

        paths = self._paths()
        report = build_mechanism_dataset(
            catalog_path=CATALOG_PATH,
            plan_path=self.plan_path,
            model="gemini-3-flash-preview",
            max_rules=1,
            max_attempts=1,
            response_provider=provider,
            **paths,
        )
        self.assertTrue(report["run_complete"])
        self.assertFalse(report["complete"])
        self.assertEqual(report["dataset_case_count"], 5)
        self.assertEqual(len(calls), 1)
        dataset = json.loads(paths["dataset_path"].read_text(encoding="utf-8"))
        trace = json.loads(paths["candidate_trace_path"].read_text(encoding="utf-8"))
        self.assertEqual([row["id"] for row in dataset], [row["id"] for row in trace])
        target = self.plan["rules"][0]
        self.assertEqual({row["target_rule"]["rule_id"] for row in dataset}, {target["rule_id"]})
        self.assertEqual({row["target_rule_id"] for row in dataset}, {target["rule_id"]})
        self.assertEqual([row["mechanism"] for row in dataset], list(MECHANISMS))
        self.assertEqual(
            [row["id"] for row in dataset],
            [f"p3::{target['rule_id']}::{mechanism}" for mechanism in MECHANISMS],
        )
        for row in trace:
            self.assertEqual(row["candidate_source"], "frozen_target_binding")
            self.assertEqual(row["unified_retrieval_mode"], "target_binding")
            self.assertEqual(row["selection_strategy"], "target_rule_binding")
            self.assertEqual(row["retrieval_score_kind"], "fixed_control_0_1")
            self.assertEqual(row["topic"], target["topic"])
            self.assertEqual(len(row["retrieved_rules"]), 1)
            selected = row["retrieved_rules"][0]
            self.assertEqual(selected["rule_id"], target["rule_id"])
            self.assertEqual(selected["topic_id"], target["topic_id"])
            self.assertEqual(selected["cluster_id"], target["cluster_id"])
            self.assertEqual(selected["score"], 1.0)
            self.assertNotIn("semantic_score", selected)
            self.assertNotIn("grounding_score", selected)
            self.assertEqual(
                set(selected["publish_gate"]),
                {
                    "publishable",
                    "reasons",
                    "score",
                    "score_kind",
                    "min_publish_score",
                    "selection_strategy",
                },
            )
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        _validate_frozen_trace(
            trace,
            _catalog_index(catalog),
            dataset=dataset,
        )
        raw_records = [
            json.loads(line)
            for line in paths["raw_trace_path"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(raw_records), 1)
        self.assertNotIn("prompt", raw_records[0])
        self.assertNotIn("messages", raw_records[0])
        self.assertIn("prompt_sha256", raw_records[0])
        self.assertEqual(raw_records[0]["parse_status"], "valid")
        manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_status"], "development_incomplete")
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["ordered_ids"], [row["id"] for row in dataset])
        self.assertEqual(manifest["artifacts"]["dataset"]["record_count"], 5)
        artifact_audit = report["artifact_audit"]
        self.assertTrue(artifact_audit["valid"])
        self.assertEqual(
            artifact_audit["schema_version"], "p3_generation_artifact_audit_v1"
        )
        self.assertTrue(artifact_audit["run_complete"])
        self.assertFalse(artifact_audit["complete"])
        self.assertEqual(artifact_audit["provider_kind"], "injected")
        self.assertEqual(
            artifact_audit["artifact_paths"]["dataset"],
            str(paths["dataset_path"].resolve()),
        )
        independently_audited = audit_generation_artifacts(
            dataset_path=paths["dataset_path"],
            manifest_path=paths["manifest_path"],
        )
        self.assertEqual(
            independently_audited["configuration"], manifest["configuration"]
        )
        self.assertEqual(independently_audited["manifest"], manifest)
        self.assertEqual(independently_audited["plan"], self.plan)
        self.assertEqual(independently_audited["manifest_sha256"], manifest["manifest_sha256"])
        with self.assertRaisesRegex(MechanismDatasetError, "manifest is immutable"):
            build_mechanism_dataset(
                catalog_path=CATALOG_PATH,
                plan_path=self.plan_path,
                model="gemini-3-flash-preview",
                max_rules=1,
                max_attempts=1,
                response_provider=provider,
                **paths,
            )

        mismatched = copy.deepcopy(dataset)
        mismatched[0]["target_rule_id"] = self.plan["rules"][1]["rule_id"]
        with self.assertRaisesRegex(MechanismDatasetError, "does not match"):
            _candidate_trace_rows(mismatched, self.plan["rules"][:1])

    def test_failure_keeps_rule_and_resume_does_not_regenerate_success(self) -> None:
        first_two = self.plan["rules"][:2]
        first_calls = []

        def first_provider(rule: Mapping[str, Any], _system: str, _user: str, _attempt: int) -> str:
            first_calls.append(rule["rule_id"])
            if rule["rule_id"] == first_two[0]["rule_id"]:
                return json.dumps(_payload(rule), ensure_ascii=False)
            return "{}"

        paths = self._paths()
        first_report = build_mechanism_dataset(
            catalog_path=CATALOG_PATH,
            plan_path=self.plan_path,
            model="gemini-3-flash-preview",
            max_rules=2,
            max_attempts=1,
            response_provider=first_provider,
            **paths,
        )
        self.assertFalse(first_report["complete"])
        self.assertEqual(first_report["failed_rule_ids"], [first_two[1]["rule_id"]])
        self.assertEqual(first_report["dataset_case_count"], 5)

        system_prompt, user_prompt = _prompt(first_two[1])
        orphan = {
            "schema_version": "p3_gt_raw_response_trace_v2",
            "rule_id": first_two[1]["rule_id"],
            "selection_index": first_two[1]["selection_index"],
            "attempt": 99,
            "model": "gemini-3-flash-preview",
            "actual_model": "",
            "response_id": "",
            "provider_kind": "injected",
            "prompt_version": first_report["prompt_version"],
            "prompt_sha256": hashlib.sha256(
                (system_prompt + "\0" + user_prompt).encode("utf-8")
            ).hexdigest(),
            "raw_response": "",
            "raw_response_sha256": hashlib.sha256(b"").hexdigest(),
            "parse_status": "invalid",
            "error": "synthetic orphan",
        }
        with paths["raw_trace_path"].open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(orphan) + "\n")

        resumed_calls = []

        # Every invocation emits a new immutable audit receipt.
        paths["manifest_path"] = self.root / "resumed_generation_manifest.json"

        def resumed_provider(rule: Mapping[str, Any], _system: str, _user: str, _attempt: int) -> str:
            resumed_calls.append(rule["rule_id"])
            return json.dumps(_payload(rule), ensure_ascii=False)

        resumed = build_mechanism_dataset(
            catalog_path=CATALOG_PATH,
            plan_path=self.plan_path,
            model="gemini-3-flash-preview",
            max_rules=2,
            max_attempts=1,
            response_provider=resumed_provider,
            resume=True,
            **paths,
        )
        self.assertTrue(resumed["run_complete"])
        self.assertFalse(resumed["complete"])
        self.assertEqual(resumed["dataset_case_count"], 10)
        self.assertEqual(resumed_calls, [first_two[1]["rule_id"]])
        self.assertEqual(resumed["resume"]["orphan_raw_trace_records"], 1)

    def test_public_audit_rejects_artifact_tampering(self) -> None:
        def provider(rule: Mapping[str, Any], _system: str, _user: str, _attempt: int) -> str:
            return json.dumps(_payload(rule), ensure_ascii=False)

        paths = self._paths()
        build_mechanism_dataset(
            catalog_path=CATALOG_PATH,
            plan_path=self.plan_path,
            max_rules=1,
            max_attempts=1,
            response_provider=provider,
            **paths,
        )
        dataset = json.loads(paths["dataset_path"].read_text(encoding="utf-8"))
        dataset[0]["prediction"] += " tampered"
        paths["dataset_path"].write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            MechanismDatasetError, "does not reproduce|SHA256 does not match"
        ):
            audit_generation_artifacts(
                dataset_path=paths["dataset_path"],
                manifest_path=paths["manifest_path"],
            )

    def test_public_audit_rejects_semantic_checkpoint_tamper_with_refreshed_hashes(self) -> None:
        def provider(rule: Mapping[str, Any], _system: str, _user: str, _attempt: int) -> str:
            return json.dumps(_payload(rule), ensure_ascii=False)

        paths = self._paths()
        build_mechanism_dataset(
            catalog_path=CATALOG_PATH,
            plan_path=self.plan_path,
            max_rules=1,
            max_attempts=1,
            response_provider=provider,
            **paths,
        )
        checkpoint = json.loads(paths["checkpoint_path"].read_text(encoding="utf-8"))
        rule_id = self.plan["rules"][0]["rule_id"]
        checkpoint["states"][rule_id]["attempts_total"] = 2
        paths["checkpoint_path"].write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
        manifest["artifacts"]["checkpoint"]["size_bytes"] = paths[
            "checkpoint_path"
        ].stat().st_size
        manifest["artifacts"]["checkpoint"]["sha256"] = _sha256_file(
            paths["checkpoint_path"]
        )
        manifest["manifest_sha256"] = _object_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        paths["manifest_path"].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(MechanismDatasetError, "attempt trace count"):
            audit_generation_artifacts(
                dataset_path=paths["dataset_path"],
                manifest_path=paths["manifest_path"],
            )

    def test_exact_model_and_formal_provider_gates(self) -> None:
        paths = self._paths()

        def provider(rule: Mapping[str, Any], _system: str, _user: str, _attempt: int) -> str:
            return json.dumps(_payload(rule), ensure_ascii=False)

        with self.assertRaisesRegex(MechanismDatasetError, "must be exactly"):
            build_mechanism_dataset(
                catalog_path=CATALOG_PATH,
                plan_path=self.plan_path,
                model="gemini-3-flash-preview-extra",
                max_rules=1,
                response_provider=provider,
                **paths,
            )
        with self.assertRaisesRegex(MechanismDatasetError, "must be exactly"):
            _validate_model(" gemini-3-flash-preview ")
        with self.assertRaisesRegex(MechanismDatasetError, "forbids injected"):
            build_mechanism_dataset(
                catalog_path=CATALOG_PATH,
                plan_path=self.plan_path,
                run_kind="validation",
                response_provider=provider,
                **paths,
            )
        with self.assertRaisesRegex(MechanismDatasetError, "requires run_kind"):
            build_mechanism_dataset(
                catalog_path=CATALOG_PATH,
                plan_path=self.plan_path,
                run_kind="development",
                response_provider=provider,
                **paths,
            )

    def test_development_resume_ignores_only_non_source_worktree_drift(self) -> None:
        stored = {
            "source_identity": {
                "git_available": True,
                "git_head": "a" * 40,
                "git_branch": "test",
                "git_dirty": False,
                "git_status_sha256": "b" * 64,
                "git_tracked_diff_sha256": "c" * 64,
                "source_tree_sha256": "d" * 64,
                "source_file_count": 1,
            },
            "model": "gemini-3-flash-preview",
        }
        current = copy.deepcopy(stored)
        current["source_identity"]["git_dirty"] = True
        current["source_identity"]["git_status_sha256"] = "e" * 64
        current["source_identity"]["git_tracked_diff_sha256"] = "f" * 64
        self.assertTrue(
            _resume_configuration_compatible(
                stored, current, run_kind="development"
            )
        )
        self.assertFalse(
            _resume_configuration_compatible(stored, current, run_kind="validation")
        )
        current["source_identity"]["source_tree_sha256"] = "0" * 64
        self.assertFalse(
            _resume_configuration_compatible(
                stored, current, run_kind="development"
            )
        )

    def test_provider_metadata_and_endpoint_identity_are_not_spoofed(self) -> None:
        self.assertEqual(
            _normalize_provider_response(
                {
                    "raw_response": "{}",
                    "actual_model": "gemini-3-flash-preview",
                    "response_id": "response-1",
                },
                provider_kind="openai_compatible",
            ),
            ("{}", "gemini-3-flash-preview", "response-1"),
        )
        with self.assertRaisesRegex(MechanismDatasetError, "metadata"):
            _normalize_provider_response("{}", provider_kind="openai_compatible")
        with self.assertRaisesRegex(MechanismDatasetError, "raw response string"):
            _normalize_provider_response(
                {
                    "raw_response": "{}",
                    "actual_model": "gemini-3-flash-preview",
                    "response_id": "fake",
                },
                provider_kind="injected",
            )
        with patch.dict(
            os.environ,
            {
                "OPENAI_BASE_URL": "http://127.0.0.1:9999/v1",
                "PHYSICSVERIFIER_LLM_TIMEOUT_SEC": "15",
                "PHYSICSVERIFIER_LLM_MAX_RETRIES": "0",
            },
            clear=False,
        ):
            identity = _api_transport_identity()
        self.assertEqual(identity["endpoint_scheme"], "http")
        self.assertTrue(identity["endpoint_is_loopback"])
        self.assertEqual(identity["timeout_sec"], 15.0)
        self.assertEqual(identity["sdk_max_retries"], 0)

    def test_cli_exposes_prepare_and_max_rules_without_overwrite_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["--plan", "plan.json", "--prepare-plan-only"]
        )
        self.assertTrue(args.prepare_plan_only)
        smoke = parser.parse_args(
            [
                "--plan",
                "plan.json",
                "--dataset",
                "dataset.json",
                "--candidate-trace",
                "trace.json",
                "--raw-trace",
                "raw.jsonl",
                "--checkpoint",
                "checkpoint.json",
                "--manifest",
                "manifest.json",
                "--max-rules",
                "2",
            ]
        )
        self.assertEqual(smoke.max_rules, 2)
        self.assertFalse(any(action.dest == "overwrite_plan" for action in parser._actions))
        system, user = _prompt(self.plan["rules"][0])
        self.assertIn("strict JSON", system)
        self.assertIn("broad_proxy", json.dumps(self.plan["trigger_scope_proxy"]))
        self.assertNotIn("evaluation result", user.lower())


if __name__ == "__main__":
    unittest.main()
