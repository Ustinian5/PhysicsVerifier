import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.finalize_incremental_update import (
    _compose_incremental_blueprints,
    _stabilize_catalog_order,
    _validate_deterministic_formal_bundle,
    _validate_generalization_fingerprints,
    finalize_incremental_update,
)
from scripts.generate_cluster_proposals import (
    CLUSTER_LABEL_SYSTEM_PROMPT,
    _build_rule_index,
    _cluster_label_resume_fingerprint,
)
from scripts.generalize_experience_candidates import _batch_resume_fingerprint
from scripts.prepare_incremental_candidates import prepare_incremental_candidates
from scripts.prepare_rules_for_cluster import prepare_rules_for_cluster
from rule_framework.incremental_validation import (
    incremental_artifact_binding,
    incremental_manifest_configuration_sha256,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catalog(rule_ids: list[str], *, cluster_id: str = "coarse") -> dict:
    return {
        "domains": [
            {
                "name": "Mechanics",
                "topics": [
                    {
                        "name": "Kinematics",
                        "rules": [{"rule_id": rule_id} for rule_id in rule_ids],
                        "scenario_clusters": [
                            {
                                "id": cluster_id,
                                "name": "Coarse",
                                "summary": "Motion checks.",
                                "rule_ids": rule_ids,
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _input_record(path: Path) -> dict:
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _policy(**overrides: object) -> dict:
    policy = {
        "mode": "additive",
        "max_added_formal_rules": 1,
        "max_removed_formal_rules": 0,
        "max_modified_existing_rules": 0,
        "max_existing_rule_topic_moves": 0,
        "max_existing_rule_cluster_moves": 0,
        "max_existing_rule_group_moves": 0,
        "max_added_clusters": 1,
        "max_removed_clusters": 0,
        "max_changed_cluster_definitions": 0,
        "max_changed_topics": 1,
        "max_unexpected_changed_topics": 0,
        "max_added_generalized_rules": 1,
        "max_removed_generalized_rules": 0,
        "max_modified_generalized_rules": 0,
        "min_added_rule_incremental_source_ratio": 1.0,
        "allowed_recluster_topics": [],
    }
    policy.update(overrides)
    return policy


def _prepare_case(root: Path, *, base_rule_ids: list[str]) -> tuple[Path, Path, Path, Path]:
    workspace = root / "workspace"
    workspace.mkdir()
    base_path = root / "base.json"
    base_generalized = root / "base_generalized.json"
    base_formal = root / "base_formal.json"
    base_proposals = root / "base_proposals.json"
    current_candidates = root / "current_candidates.json"
    new_candidates = root / "new_candidates.json"
    knowledge = root / "knowledge.json"
    tagged = root / "tagged.json"
    _write(base_path, _catalog(base_rule_ids))
    _write(base_generalized, {"rules": [], "cluster_results": []})
    _write(base_formal, {"rules": []})
    _write(base_proposals, {"proposals": []})
    current_candidate_payload = {"rules": []}
    new_candidate_payload = {
        "rules": [
            {
                "rule_id": f"candidate_new_{suffix}",
                "domain": "Mechanics",
                "topic": "Kinematics",
                "title": f"Check motion setup {suffix}",
                "trigger": "A kinematics solution is evaluated.",
                "check_logic": f"Check motion assumption {suffix}.",
                "error_type": "modeling",
                "sample_ids": [f"sample_new_{suffix}"],
                "count": 1,
            }
            for suffix in ("a", "b")
        ]
    }
    formal_payload = {"rules": []}
    _write(current_candidates, current_candidate_payload)
    _write(new_candidates, new_candidate_payload)
    _write(
        knowledge,
        {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [{"name": "Kinematics", "rules": []}],
                }
            ]
        },
    )
    _write(tagged, [])
    merged_candidates, merge_report = prepare_incremental_candidates(
        current_payload=current_candidate_payload,
        new_payload=new_candidate_payload,
        formal_payload=formal_payload,
    )
    merged_path = workspace / "semantic_experience_distilled.json"
    candidate_rules_path = (
        workspace / "semantic_experience_distilled_for_cluster.json"
    )
    candidate_input_path = workspace / "rule_embedding_input.json"
    _write(merged_path, merged_candidates)
    _write(candidate_rules_path, merged_candidates)
    _write(
        candidate_input_path,
        {
            "metadata": {},
            "rules": [
                {
                    **rule,
                    "topic_key": "Mechanics::Kinematics",
                    "embedding_text": rule["check_logic"],
                }
                for rule in merged_candidates["rules"]
            ],
        },
    )
    inputs = {
        "base_catalog": _input_record(base_path),
        "base_generalized": _input_record(base_generalized),
        "base_formal": _input_record(base_formal),
        "base_cluster_proposals": _input_record(base_proposals),
        "current_candidates": _input_record(current_candidates),
        "new_candidates": _input_record(new_candidates),
        "knowledge": _input_record(knowledge),
        "tagged": _input_record(tagged),
        "merged_candidates": _input_record(merged_path),
        "candidate_rules_for_cluster": _input_record(candidate_rules_path),
        "candidate_embedding_input": _input_record(candidate_input_path),
    }
    affected_topics = merge_report["affected_topics"]
    candidate_delta = {
        "added_candidate_ids": merge_report["added_candidate_ids"],
        "support_updated_candidate_ids": merge_report[
            "support_updated_candidate_ids"
        ],
        "changed_candidate_ids": [
            *merge_report["added_candidate_ids"],
            *merge_report["support_updated_candidate_ids"],
        ],
    }
    manifest = {
        "schema_version": 2,
        "status": "prepared",
        "inputs": inputs,
        "affected_topics": affected_topics,
        "declared_change_topics": affected_topics,
        "candidate_affected_topics": affected_topics,
        "candidate_delta": candidate_delta,
        "change_policy": _policy(),
        "merge_summary": merge_report["summary"],
        "commands": [
            {"step": f"step_{index}", "command": f"command {index}"}
            for index in range(6)
        ],
        "run_configuration": {
            "candidate_embedding": {
                "embedding_model": "test-embedding",
                "similarity_threshold": 0.74,
                "min_cluster_size": 4,
                "batch_size": 64,
            },
            "candidate_generalization": {
                "model_chain": ["test-generalizer"],
                "temperature": 0.0,
                "max_clusters": 0,
                "max_candidates_per_batch": 12,
                "min_source_candidates": 2,
                "min_source_samples": 2,
                "max_tokens": 4000,
                "request_timeout_seconds": 120.0,
                "attempts": 2,
                "thinking_enabled": False,
            },
            "formal_embedding": {
                "embedding_model": "test-embedding",
                "similarity_threshold": 0.72,
                "min_cluster_size": 4,
                "batch_size": 64,
            },
            "cluster_labeling": {
                "model": "test-cluster-model",
                "temperature": 0.0,
                "max_topics": 0,
                "min_rule_count": 1,
                "max_rules_per_cluster": 8,
                "max_output_tokens": 8192,
            }
        },
    }
    manifest["configuration_sha256"] = incremental_manifest_configuration_sha256(
        manifest
    )
    manifest_path = workspace / "incremental_manifest.json"
    _write(manifest_path, manifest)
    configuration_sha256 = manifest["configuration_sha256"]
    candidate_clusters_path = workspace / "rule_embedding_clusters.json"
    candidate_cluster_binding = incremental_artifact_binding(
        manifest_path,
        stage="candidate_embedding",
        input_paths={"rule_input": candidate_input_path},
    )
    _write(
        candidate_clusters_path,
        {
            "metadata": {
                "generator": "topic_local_rule_embedding_clustering_v1",
                "embedding_model": "test-embedding",
                "similarity_threshold": 0.74,
                "min_cluster_size": 4,
                "batch_size": 64,
                **candidate_cluster_binding,
            },
            "topics": [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "topic_key": "Mechanics::Kinematics",
                    "rule_count": 2,
                    "clusters": [
                        {
                            "cluster_id": "embedding_cluster_01",
                            "rule_ids": [
                                "candidate_new_a",
                                "candidate_new_b",
                            ],
                        }
                    ],
                    "residual_rule_ids": [],
                }
            ],
        },
    )
    _write(workspace / "incremental_merge_report.json", merge_report)
    generalized_binding = incremental_artifact_binding(
        manifest_path,
        stage="candidate_generalization",
        input_paths={
            "candidate_rules": candidate_rules_path,
            "candidate_clusters": candidate_clusters_path,
        },
        input_sha256={"base_generalized": inputs["base_generalized"]["sha256"]},
    )
    generalized_rule = {
        "rule_id": "new",
        "domain": "Mechanics",
        "topic": "Kinematics",
        "title": "Check motion setup",
        "trigger": "A kinematics solution is evaluated.",
        "check_logic": "Check that the motion model matches the setup.",
        "error_type": "modeling",
        "sample_ids": ["sample_new_a", "sample_new_b"],
        "count": 2,
    }
    generalization_batch = {
        "domain": "Mechanics",
        "topic": "Kinematics",
        "cluster_id": "embedding_cluster_01",
        "source_cluster_id": "embedding_cluster_01",
        "batch_index": 1,
        "batch_count": 1,
        "candidates": merged_candidates["rules"],
    }
    resume_context = {
        "model_chain": ["test-generalizer"],
        "temperature": 0.0,
        "max_tokens": 4000,
        "attempts": 2,
        "request_timeout_seconds": 120.0,
        "thinking_enabled": False,
    }
    generalized = {
        "metadata": {
            "generator": "experience_candidate_generalizer_v1",
            "scope_mode": "full",
            "complete": True,
            "failed_batch_count": 0,
            "missing_candidate_count": 0,
            "min_source_candidates": 2,
            "min_source_samples": 2,
            "incremental_behavior_configuration": manifest[
                "run_configuration"
            ]["candidate_generalization"],
            **generalized_binding,
        },
        "rules": [generalized_rule],
        "cluster_results": [
            {
                "domain": "Mechanics",
                "topic": "Kinematics",
                "cluster_id": "embedding_cluster_01",
                "source_cluster_id": "embedding_cluster_01",
                "batch_index": 1,
                "batch_count": 1,
                "generated_rules": [generalized_rule],
                "input_candidate_ids": ["candidate_new_a", "candidate_new_b"],
                "mappings": [
                    {
                        "rule_id": "new",
                        "source_candidate_ids": [
                            "candidate_new_a",
                            "candidate_new_b",
                        ],
                    }
                ],
                "pending_candidate_ids": [],
                "resume_fingerprint_v2": _batch_resume_fingerprint(
                    generalization_batch,
                    min_source_candidates=2,
                    min_source_samples=2,
                    max_candidates_per_batch=12,
                    resume_context=resume_context,
                ),
            }
        ],
        "pending_candidate_ids": [],
        "residual_candidate_ids": [],
        "unclustered_candidate_ids": [],
        "missing_candidate_ids": [],
    }
    generalized_path = workspace / "semantic_experience_generalized.json"
    _write(generalized_path, generalized)
    formal_path = workspace / "semantic_experience_generalized_for_cluster.json"
    precluster_path = workspace / "catalog_precluster.json"
    formal_input_path = workspace / "formal_rule_embedding_input.json"
    prepare_rules_for_cluster(
        distilled_input=generalized_path,
        knowledge_path=knowledge,
        tagged_path=tagged,
        baseline_catalog_path=base_path,
        distilled_output=formal_path,
        catalog_output=precluster_path,
        report_output=workspace / "formal_precluster_report.json",
        embedding_input_output=formal_input_path,
        scenario_cluster_blueprints_paths=[],
        preserve_baseline_rule_ids=True,
        incremental_manifest_path=manifest_path,
    )
    formal_input = json.loads(formal_input_path.read_text(encoding="utf-8"))
    formal_rule_ids = [rule["rule_id"] for rule in formal_input["rules"]]
    formal_clusters_path = workspace / "formal_rule_embedding_clusters.json"
    formal_cluster_binding = incremental_artifact_binding(
        manifest_path,
        stage="formal_embedding",
        input_paths={"rule_input": formal_input_path},
    )
    formal_topic = {
        "domain": "Mechanics",
        "topic": "Kinematics",
        "topic_key": "Mechanics::Kinematics",
        "rule_count": len(formal_rule_ids),
        "cluster_count": 1,
        "clusters": [
            {
                "cluster_id": "embedding_cluster_01",
                "rule_ids": formal_rule_ids,
                "size": len(formal_rule_ids),
                "representative_rules": [],
            }
        ],
        "residual_rule_ids": [],
    }
    _write(
        formal_clusters_path,
        {
            "metadata": {
                "generator": "topic_local_rule_embedding_clustering_v1",
                "embedding_model": "test-embedding",
                "similarity_threshold": 0.72,
                "min_cluster_size": 4,
                "batch_size": 64,
                **formal_cluster_binding,
            },
            "topics": [formal_topic],
        },
    )
    proposal_binding = incremental_artifact_binding(
        manifest_path,
        stage="cluster_labeling",
        input_paths={
            "precluster_catalog": precluster_path,
            "formal_clusters": formal_clusters_path,
            "formal_rule_input": formal_input_path,
        },
        input_sha256={
            "base_cluster_proposals": inputs["base_cluster_proposals"]["sha256"]
        },
    )
    source_fingerprint = _cluster_label_resume_fingerprint(
        formal_topic,
        rule_index=_build_rule_index(formal_input),
        system_prompt=CLUSTER_LABEL_SYSTEM_PROMPT,
        model="test-cluster-model",
        temperature=0.0,
        max_topics=0,
        min_rule_count=1,
        max_rules_per_cluster=8,
        max_output_tokens=8192,
        incremental_configuration_sha256=configuration_sha256,
        incremental_lineage=proposal_binding["incremental_lineage"],
    )
    _write(
        workspace / "cluster_proposals.json",
        {
            "metadata": {
                "generator": "embedding_cluster_labeling_v1",
                "topic_count": 1,
                "target_topic_count": 1,
                "failure_count": 0,
                "fallback_label_count": 0,
                "model": "test-cluster-model",
                "temperature": 0.0,
                "max_topics": 0,
                "min_rule_count": 1,
                "max_rules_per_cluster": 8,
                "max_output_tokens": 8192,
                **proposal_binding,
            },
            "proposals": [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "topic_key": "mechanics::kinematics",
                    "source_fingerprint": source_fingerprint,
                    "label_source": "model",
                }
            ],
            "failures": [],
        },
    )
    return workspace, base_path, knowledge, tagged


class FinalizeIncrementalUpdateTests(unittest.TestCase):
    def test_stabilizes_existing_domain_and_topic_order(self) -> None:
        base = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [{"name": "A"}, {"name": "B"}],
                },
                {"name": "Optics", "topics": [{"name": "C"}]},
            ]
        }
        candidate = {
            "domains": [
                {"name": "Optics", "topics": [{"name": "C"}]},
                {
                    "name": "Mechanics",
                    "topics": [
                        {"name": "B"},
                        {"name": "A"},
                        {"name": "New Topic"},
                    ],
                },
                {"name": "New Domain", "topics": []},
            ]
        }

        _stabilize_catalog_order(base, candidate)

        self.assertEqual(
            [domain["name"] for domain in candidate["domains"]],
            ["Mechanics", "Optics", "New Domain"],
        )
        self.assertEqual(
            [topic["name"] for topic in candidate["domains"][0]["topics"]],
            ["A", "B", "New Topic"],
        )

    def test_additive_order_keeps_existing_rules_and_clusters_before_new_items(self) -> None:
        base = _catalog(["r1", "r2"])
        base_topic = base["domains"][0]["topics"][0]
        base_topic["scenario_clusters"] = [
            {"id": "a", "rule_ids": ["r1"]},
            {"id": "b", "rule_ids": ["r2"]},
        ]
        candidate = _catalog(["r3", "r2", "r1"])
        candidate_topic = candidate["domains"][0]["topics"][0]
        candidate_topic["scenario_clusters"] = [
            {"id": "b", "rule_ids": ["r2"]},
            {"id": "a", "rule_ids": ["r3", "r1"]},
            {"id": "c", "rule_ids": []},
        ]

        _stabilize_catalog_order(base, candidate)

        self.assertEqual(
            [rule["rule_id"] for rule in candidate_topic["rules"]],
            ["r1", "r2", "r3"],
        )
        self.assertEqual(
            [cluster["id"] for cluster in candidate_topic["scenario_clusters"]],
            ["a", "b", "c"],
        )
        self.assertEqual(candidate_topic["scenario_clusters"][0]["rule_ids"], ["r1", "r3"])

    def test_scoped_order_keeps_rule_identity_order_but_preserves_new_topology(self) -> None:
        base = _catalog(["r1", "r2"])
        base["domains"][0]["topics"][0]["scenario_clusters"] = [
            {"id": "a", "rule_ids": ["r1"]},
            {"id": "b", "rule_ids": ["r2"]},
        ]
        candidate = _catalog(["r3", "r2", "r1"])
        topic = candidate["domains"][0]["topics"][0]
        topic["scenario_clusters"] = [
            {"id": "b", "rule_ids": ["r1", "r2"]},
            {"id": "a", "rule_ids": ["r3"]},
        ]

        _stabilize_catalog_order(
            base,
            candidate,
            recluster_topic_keys=["mechanics::kinematics"],
        )

        self.assertEqual(
            [item["rule_id"] for item in topic["rules"]],
            ["r1", "r2", "r3"],
        )
        self.assertEqual(
            [item["id"] for item in topic["scenario_clusters"]],
            ["b", "a"],
        )

    def test_additive_order_stabilizes_groups_and_group_rules(self) -> None:
        base = _catalog(["r1", "r2"])
        base_cluster = base["domains"][0]["topics"][0]["scenario_clusters"][0]
        base_cluster["rule_groups"] = [
            {"id": "g1", "rule_ids": ["r1", "r2"]},
            {"id": "g2", "rule_ids": []},
        ]
        candidate = _catalog(["r3", "r2", "r1"])
        cluster = candidate["domains"][0]["topics"][0]["scenario_clusters"][0]
        cluster["rule_groups"] = [
            {"id": "g2", "rule_ids": []},
            {"id": "g-new", "rule_ids": []},
            {"id": "g1", "rule_ids": ["r3", "r2", "r1"]},
        ]

        _stabilize_catalog_order(base, candidate)

        self.assertEqual(cluster["rule_ids"], ["r1", "r2", "r3"])
        self.assertEqual(
            [group["id"] for group in cluster["rule_groups"]],
            ["g1", "g2", "g-new"],
        )
        self.assertEqual(
            cluster["rule_groups"][0]["rule_ids"],
            ["r1", "r2", "r3"],
        )

    def test_additive_blueprints_filter_out_existing_rules_and_avoid_id_collision(self) -> None:
        combined, changed, validation = _compose_incremental_blueprints(
            base_catalog=_catalog(["old"]),
            precluster_catalog=_catalog(["old", "new"]),
            generated_blueprints={
                "mechanics::kinematics": [
                    {
                        "cluster_id": "coarse",
                        "rule_groups": [
                            {"group_id": "generated", "rule_ids": ["old", "new"]}
                        ],
                    }
                ]
            },
            policy=_policy(),
        )

        self.assertTrue(validation["passed"])
        self.assertEqual(
            [item["cluster_id"] for item in combined["mechanics::kinematics"]],
            ["coarse", "coarse__incremental_01"],
        )
        self.assertEqual(
            changed["mechanics::kinematics"][0]["rule_groups"][0]["rule_ids"],
            ["new"],
        )

    def test_scoped_recluster_replaces_entire_allowed_topic_with_exact_cover(self) -> None:
        generated = {
            "mechanics::kinematics": [
                {
                    "cluster_id": "replacement",
                    "rule_groups": [
                        {
                            "group_id": "replacement_rules",
                            "rule_ids": ["new", "old"],
                        }
                    ],
                }
            ]
        }
        combined, changed, validation = _compose_incremental_blueprints(
            base_catalog=_catalog(["old"]),
            precluster_catalog=_catalog(["old", "new"]),
            generated_blueprints=generated,
            policy=_policy(
                mode="scoped_recluster",
                allowed_recluster_topics=[
                    {"domain": "Mechanics", "topic": "Kinematics"}
                ],
            ),
        )

        self.assertTrue(validation["passed"])
        self.assertEqual(combined["mechanics::kinematics"], generated["mechanics::kinematics"])
        self.assertEqual(changed["mechanics::kinematics"], generated["mechanics::kinematics"])
        self.assertEqual(
            validation["scoped_topic_audits"][0]["assigned_rule_count"],
            2,
        )

    def test_blueprint_cover_rejects_missing_duplicate_and_foreign_rules(self) -> None:
        cases = {
            "missing": [["old"]],
            "duplicate": [["old", "new"], ["old"]],
            "foreign": [["old", "new", "other"]],
        }
        for name, groups in cases.items():
            with self.subTest(name=name):
                _combined, _changed, validation = _compose_incremental_blueprints(
                    base_catalog=_catalog(["old"]),
                    precluster_catalog=_catalog(["old", "new"]),
                    generated_blueprints={
                        "mechanics::kinematics": [
                            {
                                "cluster_id": "replacement",
                                "rule_groups": [
                                    {
                                        "group_id": f"g{index}",
                                        "rule_ids": rule_ids,
                                    }
                                    for index, rule_ids in enumerate(groups)
                                ],
                            }
                        ]
                    },
                    policy=_policy(
                        mode="scoped_recluster",
                        allowed_recluster_topics=[
                            {"domain": "Mechanics", "topic": "Kinematics"}
                        ],
                    ),
                )
                audit = validation["scoped_topic_audits"][0]
                self.assertFalse(validation["passed"])
                self.assertTrue(
                    audit["missing_rule_ids"]
                    or audit["duplicate_rule_ids"]
                    or audit["foreign_rule_ids"]
                )

    def _run(
        self,
        *,
        workspace: Path,
        base_path: Path,
        knowledge: Path,
        tagged: Path,
        final_catalog: dict,
    ) -> dict:
        with (
            patch(
                "scripts.finalize_incremental_update.add_catalog_fallback_proposals",
                side_effect=lambda proposals, _catalog: proposals,
            ),
            patch(
                "scripts.finalize_incremental_update.build_generated_blueprints_from_refined_proposals",
                return_value={
                    "mechanics::kinematics": [
                        {
                            "cluster_id": "generated",
                            "name": "Generated",
                            "description": "Generated checks.",
                            "rule_groups": [
                                {
                                    "group_id": "generated_rules",
                                    "name": "Generated rules",
                                    "summary": "Generated checks.",
                                    "activation_condition": "Use for generated checks.",
                                    "rule_ids": ["new"],
                                }
                            ],
                        }
                    ]
                },
            ),
            patch(
                "scripts.finalize_incremental_update.build_unified_catalog",
                return_value=final_catalog,
            ),
            patch(
                "scripts.finalize_incremental_update.validate_catalog_structure",
                return_value={"valid": True},
            ),
            patch(
                "scripts.finalize_incremental_update.audit_rule_coarsening",
                return_value={"complete": True},
            ),
        ):
            return finalize_incremental_update(
                workspace=workspace,
                base_catalog_path=base_path,
                knowledge_path=knowledge,
                tagged_path=tagged,
            )

    def test_removed_base_rule_blocks_retrieval_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, base_path, knowledge, tagged = _prepare_case(
                root, base_rule_ids=["old"]
            )

            report = self._run(
                workspace=workspace,
                base_path=base_path,
                knowledge=knowledge,
                tagged=tagged,
                final_catalog=_catalog(["new"]),
            )

            self.assertFalse(report["ready_for_retrieval_evaluation"])
            self.assertFalse(report["promotion_ready"])
            self.assertEqual(report["change_scope"]["added_rule_ids"], ["new"])
            self.assertEqual(report["change_scope"]["removed_rule_ids"], ["old"])
            self.assertFalse(report["change_scope"]["identity_stable"])
            self.assertIn("removed_formal_rules", report["change_policy"]["violations"])

    def test_strict_additive_catalog_is_ready_for_retrieval_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, base_path, knowledge, tagged = _prepare_case(
                root, base_rule_ids=["old"]
            )

            report = self._run(
                workspace=workspace,
                base_path=base_path,
                knowledge=knowledge,
                tagged=tagged,
                final_catalog=_catalog(["old", "new"]),
            )

            self.assertTrue(report["ready_for_retrieval_evaluation"])
            self.assertTrue(report["manifest_validation"]["passed"])
            self.assertTrue(report["candidate_delta_validation"]["passed"])
            self.assertTrue(report["output_binding_validation"]["passed"])
            self.assertTrue(report["topology_validation"]["passed"])
            self.assertTrue(report["change_policy"]["passed"])
            self.assertEqual(report["lineage"]["coverage_ratio"], 1.0)
            self.assertEqual(
                report["catalog_diff"]["rule_cluster_mapping"]["changed_existing_count"],
                0,
            )

    def test_base_catalog_hash_mismatch_fails_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, base_path, knowledge, tagged = _prepare_case(
                root, base_rule_ids=["old"]
            )
            _write(base_path, _catalog(["changed_after_prepare"]))

            with patch(
                "scripts.finalize_incremental_update.build_unified_catalog"
            ) as builder:
                with self.assertRaisesRegex(ValueError, "base_catalog: sha256_mismatch"):
                    finalize_incremental_update(
                        workspace=workspace,
                        base_catalog_path=base_path,
                        knowledge_path=knowledge,
                        tagged_path=tagged,
                    )

            builder.assert_not_called()
            report = json.loads(
                (workspace / "incremental_validation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(report["manifest_validation"]["passed"])

    def test_recomputed_candidate_delta_mismatch_fails_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, base_path, knowledge, tagged = _prepare_case(
                root, base_rule_ids=["old"]
            )
            merge_report_path = workspace / "incremental_merge_report.json"
            merge_report = json.loads(merge_report_path.read_text(encoding="utf-8"))
            merge_report["affected_topics"] = []
            _write(merge_report_path, merge_report)

            with patch(
                "scripts.finalize_incremental_update.build_unified_catalog"
            ) as builder:
                with self.assertRaisesRegex(
                    ValueError, "Incremental candidate delta validation failed"
                ):
                    finalize_incremental_update(
                        workspace=workspace,
                        base_catalog_path=base_path,
                        knowledge_path=knowledge,
                        tagged_path=tagged,
                    )

            builder.assert_not_called()

    def test_output_configuration_mismatch_fails_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, base_path, knowledge, tagged = _prepare_case(
                root, base_rule_ids=["old"]
            )
            generalized_path = workspace / "semantic_experience_generalized.json"
            generalized = json.loads(generalized_path.read_text(encoding="utf-8"))
            generalized["metadata"]["incremental_configuration_sha256"] = "stale"
            _write(generalized_path, generalized)

            with patch(
                "scripts.finalize_incremental_update.build_unified_catalog"
            ) as builder:
                with self.assertRaisesRegex(
                    ValueError, "Incremental run output binding failed"
                ):
                    finalize_incremental_update(
                        workspace=workspace,
                        base_catalog_path=base_path,
                        knowledge_path=knowledge,
                        tagged_path=tagged,
                    )

            builder.assert_not_called()
            report = json.loads(
                (workspace / "incremental_validation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["manifest_validation"]["passed"])
            self.assertFalse(report["output_binding_validation"]["passed"])

    def test_incomplete_generalization_fails_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, base_path, knowledge, tagged = _prepare_case(
                root, base_rule_ids=["old"]
            )
            generalized_path = workspace / "semantic_experience_generalized.json"
            generalized = json.loads(generalized_path.read_text(encoding="utf-8"))
            generalized["metadata"]["complete"] = False
            generalized["metadata"]["failed_batch_count"] = 1
            _write(generalized_path, generalized)

            with patch(
                "scripts.finalize_incremental_update.build_unified_catalog"
            ) as builder:
                with self.assertRaisesRegex(
                    ValueError, "Incremental run output binding failed"
                ):
                    finalize_incremental_update(
                        workspace=workspace,
                        base_catalog_path=base_path,
                        knowledge_path=knowledge,
                        tagged_path=tagged,
                    )

            builder.assert_not_called()
            report = json.loads(
                (workspace / "incremental_validation.json").read_text(
                    encoding="utf-8"
                )
            )
            fields = {
                item.get("field")
                for item in report["output_binding_validation"]["mismatches"]
            }
            self.assertIn("metadata.complete", fields)
            self.assertIn("metadata.failed_batch_count", fields)

    def test_fallback_cluster_label_fails_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, base_path, knowledge, tagged = _prepare_case(
                root, base_rule_ids=["old"]
            )
            proposals_path = workspace / "cluster_proposals.json"
            proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
            proposals["metadata"]["fallback_label_count"] = 1
            proposals["proposals"][0]["label_source"] = "embedding_fallback"
            _write(proposals_path, proposals)

            with patch(
                "scripts.finalize_incremental_update.build_unified_catalog"
            ) as builder:
                with self.assertRaisesRegex(
                    ValueError, "Incremental run output binding failed"
                ):
                    finalize_incremental_update(
                        workspace=workspace,
                        base_catalog_path=base_path,
                        knowledge_path=knowledge,
                        tagged_path=tagged,
                    )

            builder.assert_not_called()
            report = json.loads(
                (workspace / "incremental_validation.json").read_text(
                    encoding="utf-8"
                )
            )
            fields = {
                item.get("field")
                for item in report["output_binding_validation"]["mismatches"]
            }
            self.assertIn("metadata.fallback_label_count", fields)
            self.assertIn("proposals.label_source", fields)

    def test_deterministic_replay_rejects_formal_content_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, base_path, knowledge, tagged = _prepare_case(
                root, base_rule_ids=["old"]
            )
            formal_path = (
                workspace / "semantic_experience_generalized_for_cluster.json"
            )
            formal = json.loads(formal_path.read_text(encoding="utf-8"))
            new_rule = next(
                rule for rule in formal["rules"] if rule["rule_id"] == "new"
            )
            new_rule["check_logic"] = "A sample-specific replacement check."

            validation = _validate_deterministic_formal_bundle(
                workspace=workspace,
                base_catalog_path=base_path,
                knowledge_path=knowledge,
                tagged_path=tagged,
                formal=formal,
                precluster_catalog=json.loads(
                    (workspace / "catalog_precluster.json").read_text(
                        encoding="utf-8"
                    )
                ),
                formal_rule_input=json.loads(
                    (workspace / "formal_rule_embedding_input.json").read_text(
                        encoding="utf-8"
                    )
                ),
            )

            self.assertFalse(validation["passed"])
            self.assertIn(
                "formal_rules",
                {item["artifact"] for item in validation["mismatches"]},
            )

    def test_generalization_fingerprint_replay_rejects_stale_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, _base_path, _knowledge, _tagged = _prepare_case(
                root, base_rule_ids=["old"]
            )
            manifest = json.loads(
                (workspace / "incremental_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            generalized = json.loads(
                (workspace / "semantic_experience_generalized.json").read_text(
                    encoding="utf-8"
                )
            )
            generalized["cluster_results"][0]["resume_fingerprint_v2"] = "stale"

            validation = _validate_generalization_fingerprints(
                manifest=manifest,
                workspace=workspace,
                generalized=generalized,
            )

            self.assertFalse(validation["passed"])
            self.assertTrue(validation["api_calls_required"])

    def test_base_generalized_hash_mismatch_fails_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, base_path, knowledge, tagged = _prepare_case(
                root, base_rule_ids=["old"]
            )
            manifest = json.loads(
                (workspace / "incremental_manifest.json").read_text(encoding="utf-8")
            )
            base_generalized = Path(manifest["inputs"]["base_generalized"]["path"])
            _write(base_generalized, {"rules": [{"rule_id": "drifted"}]})

            with patch(
                "scripts.finalize_incremental_update.build_unified_catalog"
            ) as builder:
                with self.assertRaisesRegex(
                    ValueError, "base_generalized: sha256_mismatch"
                ):
                    finalize_incremental_update(
                        workspace=workspace,
                        base_catalog_path=base_path,
                        knowledge_path=knowledge,
                        tagged_path=tagged,
                    )

            builder.assert_not_called()

    def test_post_prepare_budget_edit_fails_configuration_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, base_path, knowledge, tagged = _prepare_case(
                root, base_rule_ids=["old"]
            )
            manifest_path = workspace / "incremental_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["change_policy"]["max_added_formal_rules"] = 99
            _write(manifest_path, manifest)

            with patch(
                "scripts.finalize_incremental_update.build_unified_catalog"
            ) as builder:
                with self.assertRaisesRegex(
                    ValueError, "manifest: configuration_sha256_mismatch"
                ):
                    finalize_incremental_update(
                        workspace=workspace,
                        base_catalog_path=base_path,
                        knowledge_path=knowledge,
                        tagged_path=tagged,
                    )

            builder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
