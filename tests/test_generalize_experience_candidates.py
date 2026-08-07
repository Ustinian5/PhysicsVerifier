import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.generalize_experience_candidates import (
    _build_prompt,
    _call_model,
    _extract_json_object,
    _incremental_configuration_binding,
    _retry_user_prompt,
    _stable_rule_id,
    _thinking_kwargs,
    generalize_candidates,
)
from rule_framework.incremental_validation import (
    incremental_manifest_configuration_sha256,
)


class GeneralizeExperienceCandidatesTest(unittest.TestCase):
    def test_incremental_binding_rejects_manifest_changed_after_fingerprinting(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "incremental_manifest.json"
            manifest = {
                "schema_version": 2,
                "inputs": {},
                "candidate_delta": {},
                "candidate_affected_topics": [],
                "declared_change_topics": [],
                "affected_topics": [],
                "change_policy": {},
                "commands": [],
                "run_configuration": {"model": "model-a"},
            }
            manifest["configuration_sha256"] = (
                incremental_manifest_configuration_sha256(manifest)
            )
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(
                _incremental_configuration_binding(str(path))[
                    "incremental_configuration_sha256"
                ],
                manifest["configuration_sha256"],
            )

            manifest["run_configuration"]["model"] = "model-b"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                _incremental_configuration_binding(str(path))

    def test_rule_id_depends_on_source_set_not_generated_wording(self):
        first = _stable_rule_id("Mechanics", "Kinematics", ["candidate_b", "candidate_a"])
        second = _stable_rule_id("Mechanics", "Kinematics", ["candidate_a", "candidate_b"])
        self.assertEqual(first, second)

    def test_retry_prompt_rejects_scalar_output(self):
        prompt = _retry_user_prompt("original")
        self.assertIn("包含 rules 数组的 JSON 对象", prompt)
        self.assertIn("不得返回单个数字", prompt)

    def test_thinking_is_disabled_by_default(self):
        self.assertEqual(
            _thinking_kwargs(),
            {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        )
        self.assertTrue(
            _thinking_kwargs(enable_thinking=True)["extra_body"]["chat_template_kwargs"][
                "enable_thinking"
            ]
        )

    def test_deepseek_uses_its_thinking_parameter(self):
        self.assertEqual(
            _thinking_kwargs(model="deepseek-v4-flash"),
            {"extra_body": {"chat_template_kwargs": {"thinking": False}}},
        )
        self.assertTrue(
            _thinking_kwargs(
                model="deepseek-v4-flash",
                enable_thinking=True,
            )["extra_body"]["chat_template_kwargs"]["thinking"]
        )

    def test_api_timeout_enters_bounded_outer_retry(self):
        class FakeCompletions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                raise TimeoutError("request timed out")

        completions = FakeCompletions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        with patch("scripts.generalize_experience_candidates.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "failed across models"):
                _call_model(
                    client=client,
                    model="test-model",
                    system_prompt="system",
                    user_prompt="user",
                    max_tokens=256,
                    attempts=2,
                )

        self.assertEqual(len(completions.calls), 2)
        self.assertFalse(
            completions.calls[0]["extra_body"]["chat_template_kwargs"][
                "enable_thinking"
            ]
        )

    def test_route_unavailable_switches_to_fallback_without_repeating_primary(self):
        class FakeCompletions:
            def __init__(self):
                self.models = []

            def create(self, **kwargs):
                self.models.append(kwargs["model"])
                if kwargs["model"] == "unavailable-primary":
                    raise RuntimeError(
                        "503 bad_response_status_code: No available sub-groups"
                    )
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content='{"rules":[]}')
                        )
                    ]
                )

        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        result = _call_model(
            client=client,
            model="unavailable-primary",
            fallback_models=["working-fallback"],
            system_prompt="system",
            user_prompt="user",
            max_tokens=256,
            attempts=3,
        )

        self.assertEqual(
            completions.models,
            ["unavailable-primary", "working-fallback"],
        )
        self.assertEqual(result["_model_used"], "working-fallback")

    def test_prompt_requires_shared_information(self):
        _, prompt = _build_prompt(
            "Mechanics",
            "Kinematics",
            [{"rule_id": "candidate_a", "title": "A", "trigger": "B", "check_logic": "C"}],
        )
        self.assertIn("至少两条来源候选共同支持", prompt)

    def test_extract_json_object_accepts_fenced_json(self):
        self.assertEqual(
            _extract_json_object('```json\n{"rules": []}\n```'),
            {"rules": []},
        )

    def test_extract_json_object_wraps_rule_array(self):
        self.assertEqual(
            _extract_json_object('[{"title": "rule"}]'),
            {"rules": [{"title": "rule"}]},
        )

    def test_generalizes_cluster_and_keeps_sample_sources(self):
        candidate_payload = {
            "rules": [
                {
                    "rule_id": "candidate_a",
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "title": "具体数值校验 A",
                    "trigger": "样本 A",
                    "check_logic": "检查数值 A",
                    "error_type": "calculation",
                    "sample_ids": ["sample_1"],
                },
                {
                    "rule_id": "candidate_b",
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "title": "具体数值校验 B",
                    "trigger": "样本 B",
                    "check_logic": "检查数值 B",
                    "error_type": "calculation",
                    "sample_ids": ["sample_2"],
                },
                {
                    "rule_id": "candidate_c",
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "title": "不同机制",
                    "trigger": "样本 C",
                    "check_logic": "检查机制 C",
                    "error_type": "logic",
                    "sample_ids": ["sample_3"],
                },
            ]
        }
        cluster_payload = {
            "topics": [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "clusters": [
                        {
                            "cluster_id": "embedding_cluster_01",
                            "rule_ids": ["candidate_a", "candidate_b", "candidate_c"],
                        }
                    ],
                }
            ]
        }

        def fake_generate(domain, topic, candidates):
            self.assertEqual((domain, topic), ("Mechanics", "Kinematics"))
            self.assertEqual(len(candidates), 3)
            return {
                "rules": [
                    {
                        "source_candidate_ids": ["candidate_a", "candidate_b"],
                        "title": "通用数量级校验",
                        "trigger": "计算结果涉及跨数量级运算时",
                        "check_logic": "逐项检查指数与单位。",
                        "error_type": "calculation",
                        "symbolic_hint": {
                            "primitive": "none",
                            "canonical": "",
                            "required_symbols": [],
                        },
                    }
                ]
            }

        result = generalize_candidates(
            candidate_payload=candidate_payload,
            cluster_payload=cluster_payload,
            generate=fake_generate,
            output_metadata={
                "incremental_configuration_sha256": "configuration-a",
                "incremental_manifest": "workspace/incremental_manifest.json",
            },
        )

        self.assertEqual(result["metadata"]["generated_rule_count"], 1)
        self.assertEqual(result["metadata"]["pending_candidate_count"], 1)
        self.assertEqual(
            result["metadata"]["incremental_configuration_sha256"],
            "configuration-a",
        )
        self.assertEqual(
            result["metadata"]["incremental_manifest"],
            "workspace/incremental_manifest.json",
        )
        self.assertEqual(result["rules"][0]["sample_ids"], ["sample_1", "sample_2"])
        self.assertNotIn("source_candidate_ids", result["rules"][0])
        self.assertEqual(
            result["cluster_results"][0]["pending_candidate_ids"],
            ["candidate_c"],
        )

    def test_single_candidate_rule_stays_pending(self):
        candidate_payload = {
            "rules": [
                {
                    "rule_id": "candidate_a",
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "title": "单题规则",
                    "trigger": "单题条件",
                    "check_logic": "单题检查",
                    "error_type": "logic",
                    "sample_ids": ["sample_1"],
                }
            ]
        }
        cluster_payload = {
            "topics": [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "clusters": [
                        {
                            "cluster_id": "embedding_cluster_01",
                            "rule_ids": ["candidate_a"],
                        }
                    ],
                }
            ]
        }

        result = generalize_candidates(
            candidate_payload=candidate_payload,
            cluster_payload=cluster_payload,
            generate=lambda *_: {
                "rules": [
                    {
                        "source_candidate_ids": ["candidate_a"],
                        "title": "单题规则",
                        "trigger": "单题条件",
                        "check_logic": "单题检查",
                        "error_type": "logic",
                    }
                ]
            },
        )

        self.assertEqual(result["rules"], [])
        self.assertEqual(result["metadata"]["pending_candidate_count"], 1)

    def test_two_candidates_from_same_sample_stay_pending(self):
        candidate_payload = {
            "rules": [
                {
                    "rule_id": "candidate_a",
                    "domain": "Optics",
                    "topic": "Polarization",
                    "title": "规则 A",
                    "trigger": "条件 A",
                    "check_logic": "检查 A",
                    "sample_ids": ["sample_1"],
                },
                {
                    "rule_id": "candidate_b",
                    "domain": "Optics",
                    "topic": "Polarization",
                    "title": "规则 B",
                    "trigger": "条件 B",
                    "check_logic": "检查 B",
                    "sample_ids": ["sample_1"],
                },
            ]
        }
        cluster_payload = {
            "topics": [
                {
                    "domain": "Optics",
                    "topic": "Polarization",
                    "clusters": [
                        {
                            "cluster_id": "embedding_cluster_01",
                            "rule_ids": ["candidate_a", "candidate_b"],
                        }
                    ],
                }
            ]
        }
        result = generalize_candidates(
            candidate_payload=candidate_payload,
            cluster_payload=cluster_payload,
            generate=lambda *_: {
                "rules": [
                    {
                        "source_candidate_ids": ["candidate_a", "candidate_b"],
                        "title": "概括规则",
                        "trigger": "共同条件",
                        "check_logic": "共同检查",
                        "error_type": "logic",
                    }
                ]
            },
        )

        self.assertEqual(result["rules"], [])
        self.assertEqual(result["metadata"]["pending_candidate_count"], 2)

    def test_full_scope_keeps_residual_candidates_pending(self):
        candidates = {
            "rules": [
                {
                    "rule_id": candidate_id,
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "title": candidate_id,
                    "trigger": "trigger",
                    "check_logic": "logic",
                    "sample_ids": [sample_id],
                }
                for candidate_id, sample_id in [
                    ("candidate_a", "sample_1"),
                    ("candidate_b", "sample_2"),
                    ("candidate_c", "sample_3"),
                ]
            ]
        }
        clusters = {
            "topics": [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "clusters": [
                        {
                            "cluster_id": "cluster_1",
                            "rule_ids": ["candidate_a", "candidate_b"],
                        }
                    ],
                    "residual_rule_ids": ["candidate_c"],
                }
            ]
        }
        result = generalize_candidates(
            candidate_payload=candidates,
            cluster_payload=clusters,
            generate=lambda *_: {
                "rules": [
                    {
                        "source_candidate_ids": ["candidate_a", "candidate_b"],
                        "title": "general",
                        "trigger": "trigger",
                        "check_logic": "logic",
                        "error_type": "logic",
                    }
                ]
            },
            max_clusters=0,
        )

        self.assertTrue(result["metadata"]["complete"])
        self.assertEqual(result["metadata"]["scope_mode"], "full")
        self.assertEqual(result["metadata"]["input_candidate_count"], 3)
        self.assertEqual(result["residual_candidate_ids"], ["candidate_c"])
        self.assertEqual(result["pending_candidate_ids"], ["candidate_c"])

    def test_large_cluster_is_batched_without_candidate_loss(self):
        candidate_rows = [
            {
                "rule_id": f"candidate_{index}",
                "domain": "Mechanics",
                "topic": "Kinematics",
                "title": f"rule {index}",
                "trigger": "trigger",
                "check_logic": "logic",
                "sample_ids": [f"sample_{index}"],
            }
            for index in range(5)
        ]
        calls = []

        def generate(_, __, rows):
            calls.append([row["rule_id"] for row in rows])
            return {
                "rules": [
                    {
                        "source_candidate_ids": [row["rule_id"] for row in rows],
                        "title": "general",
                        "trigger": "trigger",
                        "check_logic": "logic",
                        "error_type": "logic",
                    }
                ]
            }

        result = generalize_candidates(
            candidate_payload={"rules": candidate_rows},
            cluster_payload={
                "topics": [
                    {
                        "domain": "Mechanics",
                        "topic": "Kinematics",
                        "clusters": [
                            {
                                "cluster_id": "large_cluster",
                                "rule_ids": [row["rule_id"] for row in candidate_rows],
                            }
                        ],
                        "residual_rule_ids": [],
                    }
                ]
            },
            generate=generate,
            max_candidates_per_batch=2,
        )

        self.assertEqual(calls, [["candidate_0", "candidate_1"], ["candidate_2", "candidate_3"]])
        self.assertEqual(result["metadata"]["selected_batch_count"], 3)
        self.assertEqual(result["pending_candidate_ids"], ["candidate_4"])
        self.assertEqual(
            result["metadata"]["mapped_candidate_count"] + result["metadata"]["pending_candidate_count"],
            5,
        )

    def test_full_scope_preserves_candidates_missing_from_cluster_artifact(self):
        candidates = {
            "rules": [
                {
                    "rule_id": candidate_id,
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "title": candidate_id,
                    "trigger": "trigger",
                    "check_logic": "logic",
                    "sample_ids": [sample_id],
                }
                for candidate_id, sample_id in [
                    ("candidate_a", "sample_1"),
                    ("candidate_b", "sample_2"),
                    ("candidate_unclustered", "sample_3"),
                ]
            ]
        }
        clusters = {
            "topics": [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "clusters": [
                        {
                            "cluster_id": "cluster_1",
                            "rule_ids": ["candidate_a", "candidate_b"],
                        }
                    ],
                    "residual_rule_ids": [],
                }
            ]
        }

        result = generalize_candidates(
            candidate_payload=candidates,
            cluster_payload=clusters,
            generate=lambda *_: {
                "rules": [
                    {
                        "source_candidate_ids": ["candidate_a", "candidate_b"],
                        "title": "general",
                        "trigger": "trigger",
                        "check_logic": "logic",
                        "error_type": "logic",
                    }
                ]
            },
        )

        self.assertTrue(result["metadata"]["complete"])
        self.assertEqual(result["metadata"]["input_candidate_count"], 3)
        self.assertEqual(result["unclustered_candidate_ids"], ["candidate_unclustered"])
        self.assertEqual(result["residual_candidate_ids"], ["candidate_unclustered"])
        self.assertEqual(result["pending_candidate_ids"], ["candidate_unclustered"])

    def test_continue_on_error_preserves_failed_batch_as_pending(self):
        rows = [
            {
                "rule_id": f"candidate_{index}",
                "domain": "Mechanics",
                "topic": "Kinematics",
                "title": f"rule {index}",
                "trigger": "trigger",
                "check_logic": "logic",
                "sample_ids": [f"sample_{index}"],
            }
            for index in range(4)
        ]

        def generate(_, __, candidates):
            if candidates[0]["rule_id"] == "candidate_0":
                raise RuntimeError("fake API failure")
            return {
                "rules": [
                    {
                        "source_candidate_ids": [item["rule_id"] for item in candidates],
                        "title": "general",
                        "trigger": "trigger",
                        "check_logic": "logic",
                        "error_type": "logic",
                    }
                ]
            }

        result = generalize_candidates(
            candidate_payload={"rules": rows},
            cluster_payload={
                "topics": [
                    {
                        "domain": "Mechanics",
                        "topic": "Kinematics",
                        "clusters": [
                            {"cluster_id": "cluster_1", "rule_ids": ["candidate_0", "candidate_1"]},
                            {"cluster_id": "cluster_2", "rule_ids": ["candidate_2", "candidate_3"]},
                        ],
                        "residual_rule_ids": [],
                    }
                ]
            },
            generate=generate,
            continue_on_error=True,
        )

        self.assertFalse(result["metadata"]["complete"])
        self.assertEqual(result["metadata"]["failed_batch_count"], 1)
        self.assertEqual(result["pending_candidate_ids"], ["candidate_0", "candidate_1"])
        self.assertEqual(result["metadata"]["mapped_candidate_count"], 2)

    def test_resume_reuses_successful_batch(self):
        candidate_payload = {
            "rules": [
                {
                    "rule_id": candidate_id,
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "title": candidate_id,
                    "trigger": "trigger",
                    "check_logic": "logic",
                    "sample_ids": [sample_id],
                }
                for candidate_id, sample_id in [
                    ("candidate_a", "sample_1"),
                    ("candidate_b", "sample_2"),
                ]
            ]
        }
        cluster_payload = {
            "topics": [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "clusters": [
                        {"cluster_id": "cluster_1", "rule_ids": ["candidate_a", "candidate_b"]}
                    ],
                    "residual_rule_ids": [],
                }
            ]
        }
        first = generalize_candidates(
            candidate_payload=candidate_payload,
            cluster_payload=cluster_payload,
            generate=lambda *_: {
                "rules": [
                    {
                        "source_candidate_ids": ["candidate_a", "candidate_b"],
                        "title": "general",
                        "trigger": "trigger",
                        "check_logic": "logic",
                        "error_type": "logic",
                    }
                ]
            },
        )
        second = generalize_candidates(
            candidate_payload=candidate_payload,
            cluster_payload=cluster_payload,
            generate=lambda *_: self.fail("successful batch should be reused"),
            existing_payload=first,
        )

        self.assertTrue(second["metadata"]["complete"])
        self.assertTrue(second["cluster_results"][0]["reused"])
        self.assertEqual(first["rules"], second["rules"])

    def test_resume_rejects_same_ids_when_content_or_model_context_changes(self):
        candidate_payload = {
            "rules": [
                {
                    "rule_id": candidate_id,
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "title": candidate_id,
                    "trigger": "trigger",
                    "check_logic": "logic",
                    "sample_ids": [sample_id],
                }
                for candidate_id, sample_id in [
                    ("candidate_a", "sample_1"),
                    ("candidate_b", "sample_2"),
                ]
            ]
        }
        cluster_payload = {
            "topics": [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "clusters": [
                        {
                            "cluster_id": "cluster_1",
                            "rule_ids": ["candidate_a", "candidate_b"],
                        }
                    ],
                    "residual_rule_ids": [],
                }
            ]
        }

        def generated(*_):
            return {
                "rules": [
                    {
                        "source_candidate_ids": ["candidate_a", "candidate_b"],
                        "title": "general",
                        "trigger": "trigger",
                        "check_logic": "logic",
                        "error_type": "logic",
                    }
                ]
            }

        first = generalize_candidates(
            candidate_payload=candidate_payload,
            cluster_payload=cluster_payload,
            generate=generated,
            resume_context={
                "model": "model-a",
                "incremental_configuration_sha256": "configuration-a",
            },
        )
        calls = []

        def regenerated(*args):
            calls.append(args)
            return generated(*args)

        changed_content = {
            "rules": [dict(rule) for rule in candidate_payload["rules"]]
        }
        changed_content["rules"][0]["trigger"] = "changed trigger"
        content_result = generalize_candidates(
            candidate_payload=changed_content,
            cluster_payload=cluster_payload,
            generate=regenerated,
            existing_payload=first,
            resume_context={
                "model": "model-a",
                "incremental_configuration_sha256": "configuration-a",
            },
        )
        model_result = generalize_candidates(
            candidate_payload=candidate_payload,
            cluster_payload=cluster_payload,
            generate=regenerated,
            existing_payload=first,
            resume_context={
                "model": "model-b",
                "incremental_configuration_sha256": "configuration-a",
            },
        )
        configuration_result = generalize_candidates(
            candidate_payload=candidate_payload,
            cluster_payload=cluster_payload,
            generate=regenerated,
            existing_payload=first,
            resume_context={
                "model": "model-a",
                "incremental_configuration_sha256": "configuration-b",
            },
        )

        self.assertEqual(len(calls), 3)
        self.assertFalse(content_result["cluster_results"][0].get("reused", False))
        self.assertFalse(model_result["cluster_results"][0].get("reused", False))
        self.assertFalse(
            configuration_result["cluster_results"][0].get("reused", False)
        )


if __name__ == "__main__":
    unittest.main()
