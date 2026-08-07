import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_incremental_update import prepare_incremental_update
from rule_framework.incremental_validation import (
    incremental_manifest_configuration_sha256,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate(rule_id: str, title: str, sample_id: str) -> dict:
    return {
        "rule_id": rule_id,
        "domain": "Mechanics",
        "topic": "Kinematics",
        "title": title,
        "trigger": f"Trigger {title}",
        "check_logic": f"Check {title}",
        "error_type": "logic",
        "sample_ids": [sample_id],
        "count": 1,
    }


def _fake_prepare_rules_for_cluster(**kwargs: object) -> dict:
    distilled_output = Path(kwargs["distilled_output"])
    catalog_output = Path(kwargs["catalog_output"])
    report_output = Path(kwargs["report_output"])
    embedding_output = Path(kwargs["embedding_input_output"])
    _write(distilled_output, {"rules": []})
    _write(catalog_output, {"domains": []})
    _write(report_output, {})
    _write(embedding_output, {"metadata": {}, "rules": []})
    return {}


class PrepareIncrementalUpdateTests(unittest.TestCase):
    def test_prepares_isolated_workspace_and_cache_reusing_runbook(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {
                "current_candidates": root / "current_candidates.json",
                "new": root / "new.json",
                "generalized": root / "generalized.json",
                "formal": root / "formal.json",
                "proposals": root / "proposals.json",
                "catalog": root / "catalog.json",
                "knowledge": root / "knowledge.json",
                "tagged": root / "tagged.json",
            }
            _write(paths["current_candidates"], {"rules": [_candidate("r1", "One", "s1")]})
            new_candidate = _candidate("r2", "Two", "s2")
            new_candidate.update(
                {
                    "domain": "Mechanics",
                    "topic": "Dimensional Analysis and Scaling",
                }
            )
            _write(paths["new"], {"rules": [new_candidate]})
            _write(paths["generalized"], {"rules": [], "cluster_results": []})
            _write(paths["formal"], {"rules": []})
            _write(paths["proposals"], {"proposals": []})
            _write(
                paths["catalog"],
                {
                    "domains": [
                        {
                            "name": "Mechanics",
                            "topics": [
                                {
                                    "name": "Kinematics",
                                    "rules": [{"rule_id": "base_k"}],
                                },
                                {
                                    "name": "Dynamics",
                                    "rules": [{"rule_id": "base_d"}],
                                },
                            ],
                        },
                        {
                            "name": "Experimental Physics",
                            "topics": [
                                {
                                    "name": "Dimensional Analysis and Scaling",
                                    "rules": [],
                                }
                            ],
                        },
                    ],
                },
            )
            _write(paths["knowledge"], {"domains": []})
            _write(paths["tagged"], {"rules": []})
            workspace = root / "workspace"
            candidate_cache = root / "candidate_cache.json"
            formal_cache = root / "formal_cache.json"

            with patch(
                "scripts.prepare_incremental_update.prepare_rules_for_cluster",
                side_effect=_fake_prepare_rules_for_cluster,
            ) as prepare:
                manifest = prepare_incremental_update(
                    new_candidates_path=paths["new"],
                    workspace=workspace,
                    current_candidates_path=paths["current_candidates"],
                    current_generalized_path=paths["generalized"],
                    current_formal_path=paths["formal"],
                    current_cluster_proposals_path=paths["proposals"],
                    current_catalog_path=paths["catalog"],
                    knowledge_path=paths["knowledge"],
                    tagged_path=paths["tagged"],
                    candidate_embedding_cache_path=candidate_cache,
                    formal_embedding_cache_path=formal_cache,
                    change_mode="scoped_recluster",
                    allowed_recluster_topics=[
                        {"domain": "Mechanics", "topic": "Dynamics"}
                    ],
                )

            prepare.assert_called_once()
            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(
                manifest["affected_topics"],
                [
                    {
                        "domain": "Experimental Physics",
                        "topic": "Dimensional Analysis and Scaling",
                    },
                    {"domain": "Mechanics", "topic": "Dynamics"},
                ],
            )
            self.assertEqual(
                manifest["candidate_affected_topics"],
                [
                    {
                        "domain": "Experimental Physics",
                        "topic": "Dimensional Analysis and Scaling",
                    }
                ],
            )
            self.assertEqual(
                manifest["declared_change_topics"],
                manifest["affected_topics"],
            )
            self.assertEqual(len(manifest["commands"]), 6)
            self.assertTrue(
                manifest["commands"][0]["command"].startswith(
                    "conda run -n physicsverifier python "
                )
            )
            self.assertIn(
                str(candidate_cache).replace("\\", "/"),
                manifest["commands"][0]["command"],
            )
            self.assertEqual(
                manifest["formal_seed_catalog"],
                str(paths["catalog"]),
            )
            self.assertIn(
                str(paths["catalog"]).replace("\\", "/"),
                manifest["commands"][2]["command"],
            )
            self.assertIn(
                "--preserve-baseline-rule-ids",
                manifest["commands"][2]["command"],
            )
            self.assertIn(
                f"--incremental-manifest {workspace / 'incremental_manifest.json'}",
                manifest["commands"][1]["command"],
            )
            self.assertIn(
                f"--incremental-manifest {workspace / 'incremental_manifest.json'}",
                manifest["commands"][4]["command"],
            )
            finalize_command = manifest["commands"][5]["command"]
            self.assertIn(
                f"--knowledge {paths['knowledge']}",
                finalize_command,
            )
            self.assertIn(
                f"--tagged {paths['tagged']}",
                finalize_command,
            )
            self.assertEqual(
                manifest["candidate_delta"]["changed_candidate_ids"],
                ["r2"],
            )
            self.assertEqual(
                manifest["change_policy"]["max_added_formal_rules"],
                1,
            )
            self.assertEqual(
                manifest["change_policy"]["max_existing_rule_cluster_moves"],
                0,
            )
            self.assertEqual(
                manifest["change_policy"]["max_existing_rule_group_moves"],
                0,
            )
            self.assertEqual(
                manifest["change_policy"]["max_changed_topics"],
                2,
            )
            self.assertEqual(
                set(manifest["inputs"]),
                {
                    "base_catalog",
                    "base_generalized",
                    "base_formal",
                    "base_cluster_proposals",
                    "current_candidates",
                    "new_candidates",
                    "knowledge",
                    "tagged",
                    "merged_candidates",
                    "candidate_rules_for_cluster",
                    "candidate_embedding_input",
                },
            )
            self.assertTrue(
                all(record["sha256"] for record in manifest["inputs"].values())
            )
            self.assertEqual(
                manifest["run_configuration"]["candidate_generalization"][
                    "model_chain"
                ],
                [
                    "deepseek-v4-flash-nothinking",
                    "gemini-2.5-flash-nothinking",
                ],
            )
            self.assertEqual(
                manifest["configuration_sha256"],
                incremental_manifest_configuration_sha256(manifest),
            )
            self.assertTrue((workspace / "incremental_manifest.json").exists())
            self.assertTrue((workspace / "semantic_experience_generalized.json").exists())
            self.assertTrue((workspace / "cluster_proposals.json").exists())

    def test_explicit_scoped_recluster_runs_without_new_candidate_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            empty_rules = {"rules": []}
            current_candidates = root / "current_candidates.json"
            new_candidates = root / "new_candidates.json"
            generalized = root / "generalized.json"
            formal = root / "formal.json"
            proposals = root / "proposals.json"
            catalog = root / "catalog.json"
            knowledge = root / "knowledge.json"
            tagged = root / "tagged.json"
            for path in (current_candidates, new_candidates, formal):
                _write(path, empty_rules)
            _write(generalized, {"rules": [], "cluster_results": []})
            _write(proposals, {"proposals": []})
            _write(
                catalog,
                {
                    "domains": [
                        {
                            "name": "Mechanics",
                            "topics": [
                                {
                                    "name": "Dynamics",
                                    "rules": [{"rule_id": "old"}],
                                }
                            ],
                        }
                    ]
                },
            )
            _write(knowledge, {"domains": []})
            _write(tagged, {"rules": []})

            with patch(
                "scripts.prepare_incremental_update.prepare_rules_for_cluster",
                side_effect=_fake_prepare_rules_for_cluster,
            ):
                manifest = prepare_incremental_update(
                    new_candidates_path=new_candidates,
                    workspace=root / "workspace",
                    current_candidates_path=current_candidates,
                    current_generalized_path=generalized,
                    current_formal_path=formal,
                    current_cluster_proposals_path=proposals,
                    current_catalog_path=catalog,
                    knowledge_path=knowledge,
                    tagged_path=tagged,
                    candidate_embedding_cache_path=root / "candidate_cache.json",
                    formal_embedding_cache_path=root / "formal_cache.json",
                    change_mode="scoped_recluster",
                    max_existing_rule_cluster_moves=1,
                    max_existing_rule_group_moves=1,
                    max_removed_clusters=1,
                    allowed_recluster_topics=[
                        {"domain": "mechanics", "topic": "dynamics"}
                    ],
                )

            self.assertEqual(manifest["candidate_delta"]["changed_candidate_ids"], [])
            self.assertEqual(manifest["candidate_affected_topics"], [])
            self.assertEqual(
                manifest["declared_change_topics"],
                [{"domain": "Mechanics", "topic": "Dynamics"}],
            )
            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(len(manifest["commands"]), 6)
            self.assertEqual(manifest["change_policy"]["max_removed_clusters"], 1)


if __name__ == "__main__":
    unittest.main()
