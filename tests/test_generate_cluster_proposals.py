from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.generate_cluster_proposals import (
    CLUSTER_LABEL_SYSTEM_PROMPT,
    _assert_english_only_proposal,
    _build_embedding_topic_prompt_payload,
    _build_client,
    _build_distilled_auxiliary_index,
    _build_topic_prompt_payload,
    _collect_topic_candidates,
    _chat_json,
    _extract_json_object,
    _incremental_configuration_binding,
    _cluster_label_resume_fingerprint,
    add_catalog_fallback_proposals,
    generate_cluster_proposals_from_embedding_clusters,
)
from rule_framework.incremental_validation import (
    incremental_manifest_configuration_sha256,
)


TMP_ROOT = Path("results/test_tmp")
TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _case_dir() -> Path:
    path = TMP_ROOT / f"cluster_proposal_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class GenerateClusterProposalTests(unittest.TestCase):
    def test_incremental_binding_rejects_manifest_changed_after_fingerprinting(self) -> None:
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

    def test_cluster_label_fingerprint_covers_unsampled_and_residual_membership(self) -> None:
        base = {
            "domain": "Mechanics",
            "topic": "Kinematics",
            "topic_key": "Mechanics::Kinematics",
            "rule_count": 4,
            "clusters": [
                {"cluster_id": "c1", "rule_ids": ["r1", "r2", "r3"]}
            ],
            "residual_rule_ids": ["residual_a"],
        }
        changed = json.loads(json.dumps(base))
        changed["clusters"][0]["rule_ids"][-1] = "r4"
        changed["residual_rule_ids"] = ["residual_b"]
        kwargs = {
            "rule_index": {
                rule_id: {"rule_id": rule_id, "summary": rule_id}
                for rule_id in ("r1", "r2", "r3", "r4")
            },
            "system_prompt": CLUSTER_LABEL_SYSTEM_PROMPT,
            "model": "test-model",
            "temperature": 0.0,
            "max_topics": 1,
            "min_rule_count": 1,
            "max_rules_per_cluster": 2,
            "max_output_tokens": None,
        }

        self.assertNotEqual(
            _cluster_label_resume_fingerprint(base, **kwargs),
            _cluster_label_resume_fingerprint(changed, **kwargs),
        )

    def test_extract_json_object_accepts_plain_json(self) -> None:
        data = _extract_json_object('{"topic_summary":"x","should_add_clusters":true}')
        self.assertEqual(data["topic_summary"], "x")
        self.assertTrue(data["should_add_clusters"])

    def test_extract_json_object_accepts_fenced_json(self) -> None:
        data = _extract_json_object(
            """Here is the proposal:

```json
{"topic_summary":"mechanics summary","should_add_clusters":false}
```"""
        )
        self.assertEqual(data["topic_summary"], "mechanics summary")
        self.assertFalse(data["should_add_clusters"])

    def test_extract_json_object_accepts_unclosed_fenced_json(self) -> None:
        data = _extract_json_object(
            """```json
{"topic_summary":"mechanics summary","should_add_clusters":true}"""
        )
        self.assertEqual(data["topic_summary"], "mechanics summary")
        self.assertTrue(data["should_add_clusters"])

    def test_extract_json_object_accepts_loose_wrapped_json(self) -> None:
        data = _extract_json_object(
            'I think this is the right output: {"topic_summary":"wrapped","should_add_clusters":true}'
        )
        self.assertEqual(data["topic_summary"], "wrapped")
        self.assertTrue(data["should_add_clusters"])

    def test_english_only_proposal_rejects_cjk_content(self) -> None:
        with self.assertRaises(RuntimeError):
            _assert_english_only_proposal(
                {
                    "topic_summary": "mechanics summary",
                    "rationale": "ok",
                    "clusters": [
                        {
                            "cluster_id": "bad_cluster",
                            "name": "中文名称",
                            "summary": "summary",
                            "description": "description",
                            "includes": [],
                            "excludes": [],
                            "entry_cues": [],
                            "related_clusters": [],
                        }
                    ],
                }
            )

    def test_minimal_catalog_payload_omits_removed_runtime_fields_and_adds_auxiliary(self) -> None:
        auxiliary_index = _build_distilled_auxiliary_index(
            {
                "rules": [
                    {
                        "rule_id": "exp_a",
                        "auxiliary": {
                            "node_summary": "Magnetic force direction check.",
                            "scene_cues": ["charged particle in uniform B"],
                            "boundary_cues": ["electric force dominates"],
                            "explore_cues": ["circular motion coupling"],
                            "evidence_sample_ids": ["s1", "s2"],
                        },
                    }
                ]
            }
        )
        payload = _build_topic_prompt_payload(
            {
                "domain": "Electromagnetism",
                "topic": "Magnetic Fields and Lorentz Force",
                "rule_count": 1,
                "topic_obj": {
                    "summary": "Magnetic force and charged-particle motion.",
                    "includes": ["old include should not be present"],
                    "excludes": ["old exclude should not be present"],
                    "related_topics": ["old relation should not be present"],
                    "scenario_clusters": [],
                    "rules": [
                        {
                            "rule_id": "exp_a",
                            "title": "Lorentz direction",
                            "summary": "Check magnetic force direction.",
                            "trigger": "qvB setup",
                            "check_logic": "Use right-hand rule with charge sign.",
                            "error_type": "concept",
                            "symbolic_hint": {"primitive": "formula_pattern", "canonical": "F=qvB", "required_symbols": ["q"]},
                            "scope": "old scope",
                            "support": {"count": 2, "sample_ids": ["s1"]},
                        }
                    ],
                },
            },
            auxiliary_by_rule=auxiliary_index,
        )

        self.assertNotIn("topic_includes", payload)
        self.assertNotIn("topic_excludes", payload)
        self.assertNotIn("related_topics", payload)
        rule_payload = payload["rules"][0]
        self.assertNotIn("scope", rule_payload)
        self.assertNotIn("support_count", rule_payload)
        self.assertNotIn("sample_ids", rule_payload)
        self.assertEqual(rule_payload["auxiliary"]["node_summary"], "Magnetic force direction check.")
        self.assertEqual(rule_payload["auxiliary"]["scene_cues"], ["charged particle in uniform B"])
        self.assertIn("scene_cues", payload["output_schema"]["clusters"][0])
        self.assertNotIn("includes", payload["output_schema"]["clusters"][0])

    def test_collect_topic_candidates_prioritizes_high_rule_missing_cluster_topics(self) -> None:
        catalog = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {"name": "Has Cluster", "rules": [{"rule_id": "a"}] * 100, "scenario_clusters": [{"id": "c"}]},
                        {"name": "Small Missing", "rules": [{"rule_id": "b"}] * 5, "scenario_clusters": []},
                        {"name": "Large Missing", "rules": [{"rule_id": "c"}] * 80, "scenario_clusters": []},
                    ],
                }
            ]
        }

        candidates = _collect_topic_candidates(
            catalog,
            only_missing_clusters=True,
            domain_filters=set(),
            topic_filters=set(),
            max_topics=1,
            min_rule_count=10,
        )

        self.assertEqual([(item["domain"], item["topic"], item["rule_count"]) for item in candidates], [("Mechanics", "Large Missing", 80)])

    def test_chat_json_passes_explicit_max_tokens(self) -> None:
        class _Message:
            content = '{"topic_summary":"x","should_add_clusters":false,"clusters":[]}'

        class _Choice:
            message = _Message()

        class _Completions:
            def __init__(self) -> None:
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"choices": [_Choice()]})()

        completions = _Completions()
        client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()

        result = _chat_json(
            client,
            model="test-model",
            temperature=0.0,
            system_prompt="system",
            user_prompt="user",
            max_output_tokens=8192,
        )

        self.assertFalse(result["should_add_clusters"])
        self.assertEqual(completions.kwargs["max_tokens"], 8192)

    def test_chat_json_allows_cjk_generated_text_for_bilingual_rules(self) -> None:
        class _Message:
            content = '{"topic_summary":"中文 summary","should_add_clusters":false,"clusters":[]}'

        class _Choice:
            message = _Message()

        class _Completions:
            def create(self, **kwargs):
                return type("Response", (), {"choices": [_Choice()]})()

        client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()

        result = _chat_json(
            client,
            model="test-model",
            temperature=0.0,
            system_prompt="system",
            user_prompt="user",
        )

        self.assertEqual(result["topic_summary"], "中文 summary")

    def test_build_client_passes_request_timeout(self) -> None:
        import scripts.generate_cluster_proposals as module

        original_openai = module.OpenAI
        original_httpx = module.httpx

        class _Httpx:
            class Client:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

        captured = {}

        class _OpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        try:
            module.OpenAI = _OpenAI
            module.httpx = _Httpx
            _build_client(api_key="key", base_url="https://example.test", trust_env=True, request_timeout=123.0)
        finally:
            module.OpenAI = original_openai
            module.httpx = original_httpx

        self.assertEqual(captured["timeout"], 123.0)
        self.assertEqual(captured["http_client"].kwargs["timeout"], 123.0)
        self.assertTrue(captured["http_client"].kwargs["trust_env"])

    def test_embedding_cluster_payload_samples_rules_without_reassigning_membership(self) -> None:
        payload = _build_embedding_topic_prompt_payload(
            {
                "domain": "Mechanics",
                "topic": "Kinematics",
                "topic_key": "Mechanics::Kinematics",
                "rule_count": 3,
                "clusters": [
                    {
                        "cluster_id": "embedding_cluster_01",
                        "size": 2,
                        "rule_ids": ["r1", "r2"],
                        "representative_rules": [{"rule_id": "r1", "summary": "motion summary"}],
                    }
                ],
                "residual_rule_ids": ["r3"],
            },
            rule_index={
                "r1": {"title": "t1", "summary": "s1", "trigger": "tr1", "check_logic": "c1"},
                "r2": {"title": "t2", "summary": "s2", "trigger": "tr2", "check_logic": "c2"},
            },
            max_rules_per_cluster=1,
        )

        self.assertEqual(payload["embedding_clusters"][0]["source_cluster_id"], "embedding_cluster_01")
        self.assertEqual([item["rule_id"] for item in payload["embedding_clusters"][0]["sampled_rules"]], ["r1"])
        self.assertNotIn("candidate_rule_ids", payload["output_schema"]["clusters"][0])
        self.assertIn("source_cluster_id", payload["output_schema"]["clusters"][0])

    def test_generate_from_embedding_clusters_keeps_rule_membership_deterministic(self) -> None:
        class _Message:
            content = (
                '{"topic_summary":"Kinematics timing scenes","rationale":"embedding groups are fixed",'
                '"clusters":[{"source_cluster_id":"embedding_cluster_01","cluster_id":"timing_checks",'
                '"name":"Timing Checks","summary":"Checks timing relations.","description":"Time relation scenarios.",'
                '"scene_cues":["time interval"],"boundary_cues":["force dynamics"],"explore_cues":["piecewise motion"]}]}'
            )

        class _Choice:
            message = _Message()

        class _Completions:
            def create(self, **kwargs):
                return type("Response", (), {"choices": [_Choice()]})()

        client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()
        result = generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters={
                "topics": [
                    {
                        "domain": "Mechanics",
                        "topic": "Kinematics",
                        "topic_key": "Mechanics::Kinematics",
                        "rule_count": 3,
                        "clusters": [{"cluster_id": "embedding_cluster_01", "rule_ids": ["r1", "r2"], "size": 2}],
                        "residual_rule_ids": ["r3"],
                    }
                ]
            },
            rule_input={"rules": [{"rule_id": "r1", "summary": "s1"}, {"rule_id": "r2", "summary": "s2"}]},
            client=client,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
            max_output_tokens=2048,
        )

        proposal = result["proposals"][0]
        self.assertEqual(proposal["clusters"][0]["candidate_rule_ids"], ["r1", "r2"])
        self.assertEqual(proposal["residual_rule_ids"], ["r3"])
        self.assertEqual(proposal["label_source"], "model")
        self.assertEqual(result["metadata"]["generator"], "embedding_cluster_labeling_v1")

    def test_generate_from_embedding_clusters_records_cjk_warning_without_failure(self) -> None:
        class _Message:
            content = (
                '{"topic_summary":"中文场景","rationale":"fixed",'
                '"clusters":[{"source_cluster_id":"embedding_cluster_01","cluster_id":"gas_escape",'
                '"name":"Gas Escape","summary":"流出分子的平均能量为 2kT","description":"gas escape",'
                '"scene_cues":[],"boundary_cues":[],"explore_cues":[]}]}'
            )

        class _Choice:
            message = _Message()

        class _Completions:
            def create(self, **kwargs):
                return type("Response", (), {"choices": [_Choice()]})()

        client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()
        result = generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters={
                "topics": [
                    {
                        "domain": "Thermodynamics",
                        "topic": "Kinetic Theory",
                        "topic_key": "Thermodynamics::Kinetic Theory",
                        "rule_count": 4,
                        "clusters": [{"cluster_id": "embedding_cluster_01", "rule_ids": ["r1", "r2"], "size": 2}],
                        "residual_rule_ids": [],
                    }
                ]
            },
            rule_input={"rules": [{"rule_id": "r1", "summary": "s1"}, {"rule_id": "r2", "summary": "s2"}]},
            client=client,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
        )

        self.assertTrue(result["proposals"][0]["contains_cjk_generated_text"])
        self.assertEqual(result["metadata"]["cjk_warning_count"], 1)

    def test_generate_from_embedding_clusters_saves_incremental_output_and_resumes(self) -> None:
        root = _case_dir()
        output_path = root / "cluster_proposals.json"

        class _Message:
            content = (
                '{"topic_summary":"summary","rationale":"fixed embedding clusters",'
                '"clusters":[{"source_cluster_id":"embedding_cluster_01","cluster_id":"timing_checks",'
                '"name":"Timing Checks","summary":"Checks timing.","description":"Timing scenarios.",'
                '"scene_cues":[],"boundary_cues":[],"explore_cues":[]}]}'
            )

        class _Choice:
            message = _Message()

        class _Completions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                return type("Response", (), {"choices": [_Choice()]})()

        completions = _Completions()
        client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
        embedding_clusters = {
            "topics": [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "topic_key": "Mechanics::Kinematics",
                    "rule_count": 3,
                    "clusters": [{"cluster_id": "embedding_cluster_01", "rule_ids": ["r1", "r2"], "size": 2}],
                    "residual_rule_ids": ["r3"],
                }
            ]
        }

        first = generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters=embedding_clusters,
            rule_input={"rules": [{"rule_id": "r1", "summary": "s1"}, {"rule_id": "r2", "summary": "s2"}]},
            client=client,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
            output_path=output_path,
            resume=False,
            incremental_configuration_sha256="configuration-a",
            incremental_manifest="workspace/incremental_manifest.json",
        )
        generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters=embedding_clusters,
            rule_input={"rules": [{"rule_id": "r1", "summary": "s1"}, {"rule_id": "r2", "summary": "s2"}]},
            client=client,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
            output_path=output_path,
            resume=True,
            incremental_configuration_sha256="configuration-a",
            incremental_manifest="workspace/incremental_manifest.json",
        )

        self.assertEqual(completions.calls, 1)
        self.assertTrue(output_path.exists())
        self.assertEqual(
            first["metadata"]["incremental_configuration_sha256"],
            "configuration-a",
        )
        self.assertEqual(
            first["metadata"]["incremental_manifest"],
            "workspace/incremental_manifest.json",
        )

        generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters=embedding_clusters,
            rule_input={"rules": [{"rule_id": "r1", "summary": "s1"}, {"rule_id": "r2", "summary": "s2"}]},
            client=client,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
            output_path=output_path,
            resume=True,
            incremental_configuration_sha256="configuration-b",
            incremental_manifest="workspace/other_incremental_manifest.json",
        )
        self.assertEqual(completions.calls, 1)

        generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters=embedding_clusters,
            rule_input={
                "rules": [
                    {"rule_id": "r1", "summary": "changed summary"},
                    {"rule_id": "r2", "summary": "s2"},
                ]
            },
            client=client,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
            output_path=output_path,
            resume=True,
            incremental_configuration_sha256="configuration-a",
            incremental_manifest="workspace/incremental_manifest.json",
        )
        generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters=embedding_clusters,
            rule_input={"rules": [{"rule_id": "r1", "summary": "s1"}, {"rule_id": "r2", "summary": "s2"}]},
            client=client,
            model="different-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
            output_path=output_path,
            resume=True,
            incremental_configuration_sha256="configuration-a",
            incremental_manifest="workspace/incremental_manifest.json",
        )
        generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters=embedding_clusters,
            rule_input={
                "rules": [
                    {"rule_id": "r1", "summary": "s1"},
                    {"rule_id": "r2", "summary": "s2"},
                ]
            },
            client=client,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
            output_path=output_path,
            resume=True,
            incremental_configuration_sha256="configuration-b",
            incremental_manifest="workspace/incremental_manifest.json",
        )

        # Bundle/config identity is rebound in output metadata; only behavior inputs
        # (content/model/prompt parameters) invalidate the model-result cache.
        self.assertEqual(completions.calls, 4)

    def test_generate_from_embedding_clusters_uses_fallback_labels_on_error_when_continuing(self) -> None:
        class _Completions:
            def create(self, **kwargs):
                raise RuntimeError("bad model json")

        client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()
        result = generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters={
                "topics": [
                    {
                        "domain": "Mechanics",
                        "topic": "Center of Mass",
                        "topic_key": "Mechanics::Center of Mass",
                        "rule_count": 4,
                        "clusters": [{"cluster_id": "embedding_cluster_01", "rule_ids": ["r1", "r2"], "size": 2}],
                        "residual_rule_ids": ["r3"],
                    }
                ]
            },
            rule_input={"rules": [{"rule_id": "r1", "summary": "s1"}, {"rule_id": "r2", "summary": "s2"}]},
            client=client,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
            continue_on_error=True,
        )

        self.assertEqual(result["metadata"]["failure_count"], 0)
        self.assertEqual(result["metadata"]["fallback_label_count"], 1)
        proposal = result["proposals"][0]
        self.assertEqual(proposal["topic_key"], "mechanics::center of mass")
        self.assertEqual(proposal["clusters"][0]["name"], "Center Of Mass Cluster 01")
        self.assertEqual(proposal["clusters"][0]["summary"], "Embedding-derived cluster for Center Of Mass rules.")
        self.assertEqual(proposal["clusters"][0]["candidate_rule_ids"], ["r1", "r2"])
        self.assertEqual(proposal["residual_rule_ids"], ["r3"])

    def test_resume_retries_embedding_fallback_label(self) -> None:
        root = _case_dir()
        output_path = root / "cluster_proposals.json"
        topic_item = {
            "domain": "Mechanics",
            "topic": "Kinematics",
            "topic_key": "Mechanics::Kinematics",
            "rule_count": 2,
            "clusters": [
                {
                    "cluster_id": "embedding_cluster_01",
                    "rule_ids": ["r1", "r2"],
                    "size": 2,
                }
            ],
            "residual_rule_ids": [],
        }
        fingerprint = _cluster_label_resume_fingerprint(
            topic_item,
            rule_index={"r1": {"rule_id": "r1"}, "r2": {"rule_id": "r2"}},
            system_prompt=CLUSTER_LABEL_SYSTEM_PROMPT,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
            max_output_tokens=None,
        )
        output_path.write_text(
            json.dumps(
                {
                    "metadata": {"fallback_label_count": 1},
                    "proposals": [
                        {
                            "topic_key": "mechanics::kinematics",
                            "source_fingerprint": fingerprint,
                            "label_source": "embedding_fallback",
                        }
                    ],
                    "failures": [],
                }
            ),
            encoding="utf-8",
        )

        class _Completions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                return type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "message": type(
                                        "Message",
                                        (),
                                        {
                                            "content": json.dumps(
                                                {
                                                    "topic_summary": "Motion checks.",
                                                    "rationale": "One motion cluster.",
                                                    "clusters": [
                                                        {
                                                            "source_cluster_id": "embedding_cluster_01",
                                                            "cluster_id": "motion",
                                                            "name": "Motion",
                                                            "summary": "Motion checks.",
                                                            "description": "Motion checks.",
                                                            "scene_cues": [],
                                                            "boundary_cues": [],
                                                            "explore_cues": [],
                                                        }
                                                    ],
                                                }
                                            )
                                        },
                                    )()
                                },
                            )()
                        ]
                    },
                )()

        completions = _Completions()
        client = type(
            "Client",
            (),
            {"chat": type("Chat", (), {"completions": completions})()},
        )()
        result = generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters={"topics": [topic_item]},
            rule_input={"rules": [{"rule_id": "r1"}, {"rule_id": "r2"}]},
            client=client,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
            output_path=output_path,
            resume=True,
            continue_on_error=True,
        )

        self.assertEqual(completions.calls, 1)
        self.assertEqual(result["metadata"]["fallback_label_count"], 0)
        self.assertEqual(result["proposals"][0]["label_source"], "model")

    def test_generate_from_embedding_clusters_drops_stale_failures_when_resuming(self) -> None:
        root = _case_dir()
        output_path = root / "cluster_proposals.json"
        topic_item = {
            "domain": "Mechanics",
            "topic": "Kinematics",
            "topic_key": "Mechanics::Kinematics",
            "rule_count": 2,
            "clusters": [
                {
                    "cluster_id": "embedding_cluster_01",
                    "rule_ids": ["r1", "r2"],
                    "size": 2,
                }
            ],
            "residual_rule_ids": [],
        }
        output_path.write_text(
            json.dumps(
                {
                    "metadata": {},
                    "proposals": [
                        {
                            "domain": "Mechanics",
                            "topic": "Kinematics",
                            "topic_key": "mechanics::kinematics",
                            "source_fingerprint": _cluster_label_resume_fingerprint(
                                topic_item,
                                rule_index={"r1": {"rule_id": "r1"}, "r2": {"rule_id": "r2"}},
                                system_prompt=CLUSTER_LABEL_SYSTEM_PROMPT,
                                model="test-model",
                                temperature=0.0,
                                max_topics=1,
                                min_rule_count=1,
                                max_rules_per_cluster=2,
                                max_output_tokens=None,
                            ),
                            "rule_count": 2,
                            "clusters": [],
                            "residual_rule_ids": [],
                        }
                    ],
                    "failures": [
                        {
                            "topic_key": "mechanics::kinematics",
                            "domain": "Mechanics",
                            "topic": "Kinematics",
                            "error": "old failure",
                        },
                        {
                            "topic_key": "mechanics::obsolete topic",
                            "domain": "Mechanics",
                            "topic": "Obsolete Topic",
                            "error": "old target failure",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        class _Completions:
            def create(self, **kwargs):
                raise AssertionError("completed topic should be skipped")

        client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()
        result = generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters={
                "topics": [topic_item]
            },
            rule_input={"rules": [{"rule_id": "r1"}, {"rule_id": "r2"}]},
            client=client,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
            output_path=output_path,
            resume=True,
            continue_on_error=True,
        )

        self.assertEqual(result["metadata"]["failure_count"], 0)
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["proposals"][0]["label_source"], "model")

    def test_add_catalog_fallback_proposals_adds_missing_rule_topics(self) -> None:
        payload = {
            "metadata": {"topic_count": 1},
            "proposals": [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "topic_key": "mechanics::kinematics",
                    "rule_count": 1,
                    "clusters": [],
                    "residual_rule_ids": ["r1"],
                }
            ],
            "failures": [],
        }
        catalog = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {"name": "Kinematics", "rules": [{"rule_id": "r1"}]},
                        {"name": "Relative Motion", "rules": [{"rule_id": "r2"}, {"rule_id": "r3"}]},
                    ],
                }
            ]
        }

        result = add_catalog_fallback_proposals(payload, catalog)

        self.assertEqual(result["metadata"]["topic_count"], 2)
        self.assertEqual(result["metadata"]["target_topic_count"], 2)
        self.assertEqual(result["metadata"]["failure_count"], 0)
        self.assertEqual(result["metadata"]["catalog_fallback_topic_count"], 1)
        fallback = result["proposals"][1]
        self.assertEqual(fallback["topic_key"], "mechanics::relative motion")
        self.assertEqual(fallback["label_source"], "catalog_fallback")
        self.assertEqual(fallback["residual_rule_ids"], ["r2", "r3"])


if __name__ == "__main__":
    unittest.main()
