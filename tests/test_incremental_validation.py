import unittest

from rule_framework.incremental_validation import (
    audit_added_rule_lineage,
    build_catalog_snapshot,
    build_generalized_lineage,
    compare_catalog_snapshots,
    compare_generalized_outputs,
    evaluate_change_policy,
    incremental_manifest_configuration_sha256,
    validate_change_policy,
)


def _catalog(clusters: list[dict], rule_ids: list[str] | None = None) -> dict:
    ids = rule_ids or [
        rule_id for cluster in clusters for rule_id in cluster.get("rule_ids", [])
    ]
    return {
        "domains": [
            {
                "name": "Mechanics",
                "topics": [
                    {
                        "name": "Kinematics",
                        "rules": [
                            {"rule_id": rule_id, "title": f"Rule {rule_id}"}
                            for rule_id in dict.fromkeys(ids)
                        ],
                        "scenario_clusters": clusters,
                    }
                ],
            }
        ]
    }


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


def _empty_lineage() -> dict:
    return {
        "coverage_ratio": 1.0,
        "changed_candidate_accounting": {"unaccounted": []},
        "conflicting_rule_ids": [],
        "unknown_mapping_rule_ids": [],
        "unmapped_generalized_rule_ids": [],
        "unknown_candidate_ids": [],
    }


class IncrementalValidationTests(unittest.TestCase):
    def test_additive_policy_cannot_budget_existing_topology_destruction(self) -> None:
        for field in (
            "max_existing_rule_cluster_moves",
            "max_existing_rule_group_moves",
            "max_removed_clusters",
            "max_changed_cluster_definitions",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "must remain zero"):
                    validate_change_policy(_policy(**{field: 1}))

    def test_scoped_policy_cannot_budget_cross_topic_rule_moves(self) -> None:
        with self.assertRaisesRegex(ValueError, "must remain zero"):
            validate_change_policy(
                _policy(
                    mode="scoped_recluster",
                    max_existing_rule_topic_moves=1,
                    allowed_recluster_topics=[
                        {"domain": "Mechanics", "topic": "Kinematics"}
                    ],
                )
            )

    def test_manifest_configuration_hash_binds_commands_without_self_reference(self) -> None:
        manifest = {
            "schema_version": 2,
            "inputs": {"base_catalog": {"sha256": "base"}},
            "candidate_delta": {"changed_candidate_ids": ["candidate_new"]},
            "candidate_affected_topics": [
                {"domain": "Mechanics", "topic": "Kinematics"}
            ],
            "change_policy": _policy(),
            "commands": [
                {
                    "step": "generalize",
                    "command": (
                        "python scripts/generalize_experience_candidates.py "
                        "--model model-a --output results/a.json"
                    ),
                }
            ],
            "run_configuration": {"generalization_model": "model-a"},
        }
        expected = incremental_manifest_configuration_sha256(manifest)
        bound = {
            **manifest,
            "commands": [
                {
                    **manifest["commands"][0],
                    "command": (
                        manifest["commands"][0]["command"]
                        + f" --incremental-configuration-sha256 {expected}"
                    ),
                }
            ],
        }
        changed_model = {
            **manifest,
            "commands": [
                {
                    **manifest["commands"][0],
                    "command": manifest["commands"][0]["command"].replace(
                        "model-a", "model-b"
                    ),
                }
            ],
        }
        changed_path = {
            **manifest,
            "commands": [
                {
                    **manifest["commands"][0],
                    "command": manifest["commands"][0]["command"].replace(
                        "results/a.json", "results/b.json"
                    ),
                }
            ],
        }

        self.assertEqual(
            incremental_manifest_configuration_sha256(bound), expected
        )
        self.assertNotEqual(
            incremental_manifest_configuration_sha256(changed_model), expected
        )
        self.assertNotEqual(
            incremental_manifest_configuration_sha256(changed_path), expected
        )

    def test_existing_rule_topic_move_is_a_separate_hard_gate(self) -> None:
        base = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {"name": "Kinematics", "rules": [{"rule_id": "r1"}]},
                        {"name": "Dynamics", "rules": []},
                    ],
                }
            ]
        }
        candidate = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {"name": "Kinematics", "rules": []},
                        {"name": "Dynamics", "rules": [{"rule_id": "r1"}]},
                    ],
                }
            ]
        }
        diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(candidate),
            affected_topics=[("Mechanics", "Kinematics"), ("Mechanics", "Dynamics")],
        )
        decision = evaluate_change_policy(
            catalog_diff=diff,
            generalized_diff={
                "added_rule_ids": [],
                "removed_rule_ids": [],
                "modified_rule_ids": [],
            },
            lineage=_empty_lineage(),
            policy=_policy(max_changed_topics=2),
        )

        self.assertEqual(len(diff["rule_topic_changes"]), 1)
        self.assertFalse(decision["passed"])
        self.assertIn("existing_rule_topic_moves", decision["violations"])
        self.assertIn("rule_topic_scope", decision["violations"])

    def test_domain_definition_drift_is_hard_blocked_in_both_modes(self) -> None:
        base = _catalog([{"id": "cluster", "rule_ids": ["r1"]}])
        base["domains"][0].update(
            {"id": "mechanics", "summary": "Stable navigation summary."}
        )
        candidate = _catalog([{"id": "cluster", "rule_ids": ["r1"]}])
        candidate["domains"][0].update(
            {"id": "mechanics", "summary": "Sample-specific navigation hint."}
        )
        diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(candidate),
            affected_topics=[("Mechanics", "Kinematics")],
        )

        self.assertEqual(
            diff["domains"]["definition_changed"],
            [{"domain": "Mechanics"}],
        )
        self.assertEqual(diff["topics"]["changed"], [])
        for mode in ("additive", "scoped_recluster"):
            with self.subTest(mode=mode):
                policy = _policy(mode=mode)
                if mode == "scoped_recluster":
                    policy["allowed_recluster_topics"] = [
                        {"domain": "Mechanics", "topic": "Kinematics"}
                    ]
                decision = evaluate_change_policy(
                    catalog_diff=diff,
                    generalized_diff={
                        "added_rule_ids": [],
                        "removed_rule_ids": [],
                        "modified_rule_ids": [],
                    },
                    lineage=_empty_lineage(),
                    policy=policy,
                )

                self.assertFalse(decision["passed"])
                self.assertIn(
                    "domain_definition_stable", decision["violations"]
                )

    def test_affected_topic_definition_drift_is_hard_blocked_in_both_modes(self) -> None:
        base = _catalog([{"id": "cluster", "rule_ids": ["r1"]}])
        base["domains"][0]["topics"][0]["retrieval_hints"] = {
            "scene_keywords": ["stable mechanism"]
        }
        candidate = _catalog([{"id": "cluster", "rule_ids": ["r1"]}])
        candidate["domains"][0]["topics"][0]["retrieval_hints"] = {
            "scene_keywords": ["sample-specific cue"]
        }
        diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(candidate),
            affected_topics=[("Mechanics", "Kinematics")],
        )

        self.assertEqual(
            diff["topics"]["definition_changed"],
            [{"domain": "Mechanics", "topic": "Kinematics"}],
        )
        self.assertEqual(diff["topics"]["unexpected_changed"], [])
        for mode in ("additive", "scoped_recluster"):
            with self.subTest(mode=mode):
                policy = _policy(mode=mode)
                if mode == "scoped_recluster":
                    policy["allowed_recluster_topics"] = [
                        {"domain": "Mechanics", "topic": "Kinematics"}
                    ]
                decision = evaluate_change_policy(
                    catalog_diff=diff,
                    generalized_diff={
                        "added_rule_ids": [],
                        "removed_rule_ids": [],
                        "modified_rule_ids": [],
                    },
                    lineage=_empty_lineage(),
                    policy=policy,
                )

                self.assertFalse(decision["passed"])
                self.assertIn(
                    "topic_definition_stable", decision["violations"]
                )

    def test_domain_and_topic_order_are_hard_blocked_in_both_modes(self) -> None:
        base = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {"name": "Kinematics", "rules": []},
                        {"name": "Dynamics", "rules": []},
                    ],
                },
                {
                    "name": "Electromagnetism",
                    "topics": [{"name": "Electrostatics", "rules": []}],
                },
            ]
        }
        candidate = {
            "domains": [
                {
                    "name": "Electromagnetism",
                    "topics": [{"name": "Electrostatics", "rules": []}],
                },
                {
                    "name": "Mechanics",
                    "topics": [
                        {"name": "Dynamics", "rules": []},
                        {"name": "Kinematics", "rules": []},
                    ],
                },
            ]
        }
        diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(candidate),
        )

        self.assertTrue(diff["domains"]["existing_order_changed"])
        self.assertEqual(
            diff["domains"]["existing_topic_order_changed"],
            [{"domain": "Mechanics"}],
        )
        for mode in ("additive", "scoped_recluster"):
            with self.subTest(mode=mode):
                policy = _policy(mode=mode, max_changed_topics=0)
                if mode == "scoped_recluster":
                    policy["allowed_recluster_topics"] = [
                        {"domain": "Mechanics", "topic": "Kinematics"}
                    ]
                decision = evaluate_change_policy(
                    catalog_diff=diff,
                    generalized_diff={
                        "added_rule_ids": [],
                        "removed_rule_ids": [],
                        "modified_rule_ids": [],
                    },
                    lineage=_empty_lineage(),
                    policy=policy,
                )

                self.assertFalse(decision["passed"])
                self.assertIn("domain_order_stable", decision["violations"])
                self.assertIn("topic_order_stable", decision["violations"])

    def test_navigation_order_checks_allow_order_preserving_insertions(self) -> None:
        base = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {"name": "Kinematics", "rules": []},
                        {"name": "Dynamics", "rules": []},
                    ],
                },
                {"name": "Optics", "topics": []},
            ]
        }
        candidate = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {"name": "Kinematics", "rules": []},
                        {"name": "Statics", "rules": []},
                        {"name": "Dynamics", "rules": []},
                    ],
                },
                {"name": "Thermodynamics", "topics": []},
                {"name": "Optics", "topics": []},
            ]
        }

        diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(candidate),
        )

        self.assertFalse(diff["domains"]["existing_order_changed"])
        self.assertEqual(diff["domains"]["existing_topic_order_changed"], [])

    def test_additive_hard_blocks_existing_rule_and_cluster_reordering(self) -> None:
        base = _catalog(
            [
                {"id": "a", "rule_ids": ["r1"]},
                {"id": "b", "rule_ids": ["r2"]},
            ],
            ["r1", "r2"],
        )
        candidate = _catalog(
            [
                {"id": "b", "rule_ids": ["r2"]},
                {"id": "a", "rule_ids": ["r1"]},
            ],
            ["r2", "r1"],
        )
        diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(candidate),
            affected_topics=[("Mechanics", "Kinematics")],
        )
        decision = evaluate_change_policy(
            catalog_diff=diff,
            generalized_diff={
                "added_rule_ids": [],
                "removed_rule_ids": [],
                "modified_rule_ids": [],
            },
            lineage=_empty_lineage(),
            policy=_policy(),
        )

        self.assertEqual(
            diff["topics"]["existing_rule_order_changed"],
            [{"domain": "Mechanics", "topic": "Kinematics"}],
        )
        self.assertEqual(
            diff["topics"]["existing_cluster_order_changed"],
            [{"domain": "Mechanics", "topic": "Kinematics"}],
        )
        self.assertFalse(decision["passed"])
        self.assertIn("rule_order_stable", decision["violations"])
        self.assertIn("cluster_order_scope", decision["violations"])

    def test_scoped_recluster_exempts_only_cluster_order(self) -> None:
        base = _catalog(
            [
                {"id": "a", "rule_ids": ["r1"]},
                {"id": "b", "rule_ids": ["r2"]},
            ],
            ["r1", "r2"],
        )
        cluster_reordered = _catalog(
            [
                {"id": "b", "rule_ids": ["r2"]},
                {"id": "a", "rule_ids": ["r1"]},
            ],
            ["r1", "r2"],
        )
        rule_and_cluster_reordered = _catalog(
            [
                {"id": "b", "rule_ids": ["r2"]},
                {"id": "a", "rule_ids": ["r1"]},
            ],
            ["r2", "r1"],
        )
        policy = _policy(
            mode="scoped_recluster",
            allowed_recluster_topics=[
                {"domain": "Mechanics", "topic": "Kinematics"}
            ],
        )

        decisions = []
        for candidate in (cluster_reordered, rule_and_cluster_reordered):
            diff = compare_catalog_snapshots(
                build_catalog_snapshot(base),
                build_catalog_snapshot(candidate),
                affected_topics=[("Mechanics", "Kinematics")],
            )
            decisions.append(
                evaluate_change_policy(
                    catalog_diff=diff,
                    generalized_diff={
                        "added_rule_ids": [],
                        "removed_rule_ids": [],
                        "modified_rule_ids": [],
                    },
                    lineage=_empty_lineage(),
                    policy=policy,
                )
            )

        self.assertTrue(decisions[0]["passed"])
        self.assertTrue(decisions[0]["gates"]["cluster_order_scope"])
        self.assertFalse(decisions[1]["passed"])
        self.assertIn("rule_order_stable", decisions[1]["violations"])
        self.assertNotIn("cluster_order_scope", decisions[1]["violations"])

    def test_additive_hard_blocks_group_and_group_rule_reordering(self) -> None:
        base = _catalog(
            [
                {
                    "id": "cluster",
                    "rule_ids": ["r1", "r2", "r3"],
                    "rule_groups": [
                        {"id": "g1", "rule_ids": ["r1", "r2"]},
                        {"id": "g2", "rule_ids": ["r3"]},
                    ],
                }
            ]
        )
        candidate = _catalog(
            [
                {
                    "id": "cluster",
                    "rule_ids": ["r1", "r2", "r3"],
                    "rule_groups": [
                        {"id": "g2", "rule_ids": ["r3"]},
                        {"id": "g1", "rule_ids": ["r2", "r1"]},
                    ],
                }
            ]
        )
        diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(candidate),
            affected_topics=[("Mechanics", "Kinematics")],
        )
        decision = evaluate_change_policy(
            catalog_diff=diff,
            generalized_diff={
                "added_rule_ids": [],
                "removed_rule_ids": [],
                "modified_rule_ids": [],
            },
            lineage=_empty_lineage(),
            policy=_policy(),
        )

        self.assertEqual(
            diff["clusters"]["existing_rule_group_order_changed"],
            [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "cluster_id": "cluster",
                }
            ],
        )
        self.assertEqual(
            diff["rule_groups"]["existing_rule_order_changed"],
            [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "cluster_id": "cluster",
                    "group_id": "g1",
                }
            ],
        )
        self.assertFalse(decision["passed"])
        self.assertIn("rule_group_order_scope", decision["violations"])
        self.assertIn("rule_group_rule_order_scope", decision["violations"])

    def test_scoped_recluster_allows_group_orders_only_in_allowed_topic(self) -> None:
        base = _catalog(
            [
                {
                    "id": "cluster",
                    "rule_ids": ["r1", "r2", "r3"],
                    "rule_groups": [
                        {"id": "g1", "rule_ids": ["r1", "r2"]},
                        {"id": "g2", "rule_ids": ["r3"]},
                    ],
                }
            ]
        )
        candidate = _catalog(
            [
                {
                    "id": "cluster",
                    "rule_ids": ["r1", "r2", "r3"],
                    "rule_groups": [
                        {"id": "g2", "rule_ids": ["r3"]},
                        {"id": "g1", "rule_ids": ["r2", "r1"]},
                    ],
                }
            ]
        )
        diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(candidate),
            affected_topics=[("Mechanics", "Kinematics")],
        )
        generalized_diff = {
            "added_rule_ids": [],
            "removed_rule_ids": [],
            "modified_rule_ids": [],
        }

        allowed = evaluate_change_policy(
            catalog_diff=diff,
            generalized_diff=generalized_diff,
            lineage=_empty_lineage(),
            policy=_policy(
                mode="scoped_recluster",
                max_changed_cluster_definitions=1,
                allowed_recluster_topics=[
                    {"domain": "Mechanics", "topic": "Kinematics"}
                ],
            ),
        )
        outside_scope = evaluate_change_policy(
            catalog_diff=diff,
            generalized_diff=generalized_diff,
            lineage=_empty_lineage(),
            policy=_policy(
                mode="scoped_recluster",
                max_changed_cluster_definitions=1,
                allowed_recluster_topics=[
                    {"domain": "Mechanics", "topic": "Dynamics"}
                ],
            ),
        )

        self.assertTrue(allowed["passed"])
        self.assertTrue(allowed["gates"]["rule_group_order_scope"])
        self.assertTrue(allowed["gates"]["rule_group_rule_order_scope"])
        self.assertFalse(outside_scope["passed"])
        self.assertIn("rule_group_order_scope", outside_scope["violations"])
        self.assertIn(
            "rule_group_rule_order_scope", outside_scope["violations"]
        )

    def test_group_order_checks_ignore_order_preserving_insertions(self) -> None:
        base = _catalog(
            [
                {
                    "id": "cluster",
                    "rule_ids": ["r1", "r2", "r3"],
                    "rule_groups": [
                        {"id": "g1", "rule_ids": ["r1", "r2"]},
                        {"id": "g2", "rule_ids": ["r3"]},
                    ],
                }
            ]
        )
        candidate = _catalog(
            [
                {
                    "id": "cluster",
                    "rule_ids": ["r1", "r-new", "r2", "r3"],
                    "rule_groups": [
                        {"id": "g1", "rule_ids": ["r1", "r-new", "r2"]},
                        {"id": "g-new", "rule_ids": []},
                        {"id": "g2", "rule_ids": ["r3"]},
                    ],
                }
            ]
        )

        diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(candidate),
            affected_topics=[("Mechanics", "Kinematics")],
        )

        self.assertEqual(
            diff["clusters"]["existing_rule_group_order_changed"], []
        )
        self.assertEqual(
            diff["rule_groups"]["existing_rule_order_changed"], []
        )

    def test_cluster_rule_order_is_scoped_and_ignores_insertions(self) -> None:
        base = _catalog(
            [{"id": "cluster", "rule_ids": ["r1", "r2"]}],
            ["r1", "r2"],
        )
        reordered = _catalog(
            [{"id": "cluster", "rule_ids": ["r2", "r1"]}],
            ["r1", "r2"],
        )
        inserted = _catalog(
            [{"id": "cluster", "rule_ids": ["r1", "r-new", "r2"]}],
            ["r1", "r2", "r-new"],
        )
        reordered_diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(reordered),
            affected_topics=[("Mechanics", "Kinematics")],
        )
        inserted_diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(inserted),
            affected_topics=[("Mechanics", "Kinematics")],
        )
        generalized_diff = {
            "added_rule_ids": [],
            "removed_rule_ids": [],
            "modified_rule_ids": [],
        }

        additive = evaluate_change_policy(
            catalog_diff=reordered_diff,
            generalized_diff=generalized_diff,
            lineage=_empty_lineage(),
            policy=_policy(),
        )
        scoped = evaluate_change_policy(
            catalog_diff=reordered_diff,
            generalized_diff=generalized_diff,
            lineage=_empty_lineage(),
            policy=_policy(
                mode="scoped_recluster",
                allowed_recluster_topics=[
                    {"domain": "Mechanics", "topic": "Kinematics"}
                ],
            ),
        )

        self.assertEqual(
            reordered_diff["clusters"]["existing_rule_order_changed"],
            [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "cluster_id": "cluster",
                }
            ],
        )
        self.assertFalse(additive["passed"])
        self.assertIn("cluster_rule_order_scope", additive["violations"])
        self.assertTrue(scoped["gates"]["cluster_rule_order_scope"])
        self.assertEqual(
            inserted_diff["clusters"]["existing_rule_order_changed"], []
        )

    def test_rule_cluster_move_is_detected_when_topic_rule_ids_do_not_change(self) -> None:
        base = _catalog(
            [
                {"id": "a", "name": "A", "rule_ids": ["r1"]},
                {"id": "b", "name": "B", "rule_ids": ["r2"]},
            ]
        )
        candidate = _catalog(
            [
                {"id": "a", "name": "A", "rule_ids": ["r2"]},
                {"id": "b", "name": "B", "rule_ids": ["r1"]},
            ]
        )
        diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(candidate),
            affected_topics=[("Mechanics", "Kinematics")],
        )

        self.assertEqual(diff["topics"]["rule_set_changed"], [])
        self.assertEqual(diff["rule_cluster_mapping"]["changed_existing_count"], 2)
        decision = evaluate_change_policy(
            catalog_diff=diff,
            generalized_diff={
                "added_rule_ids": [],
                "removed_rule_ids": [],
                "modified_rule_ids": [],
            },
            lineage=_empty_lineage(),
            policy=_policy(),
        )
        self.assertFalse(decision["passed"])
        self.assertIn("existing_rule_cluster_moves", decision["violations"])
        self.assertIn("recluster_scope", decision["violations"])

    def test_rule_group_move_cannot_bypass_additive_policy(self) -> None:
        base = _catalog(
            [
                {
                    "id": "cluster",
                    "rule_ids": ["r1"],
                    "rule_groups": [
                        {"id": "g1", "rule_ids": ["r1"]},
                        {"id": "g2", "rule_ids": []},
                    ],
                }
            ]
        )
        candidate = _catalog(
            [
                {
                    "id": "cluster",
                    "rule_ids": ["r1"],
                    "rule_groups": [
                        {"id": "g1", "rule_ids": []},
                        {"id": "g2", "rule_ids": ["r1"]},
                    ],
                }
            ]
        )
        diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(candidate),
            affected_topics=[("Mechanics", "Kinematics")],
        )

        self.assertEqual(
            diff["rule_cluster_mapping"]["changed_existing_count"], 0
        )
        self.assertEqual(diff["clusters"]["definition_changed"], [])
        self.assertEqual(diff["rule_group_mapping"]["changed_existing_count"], 1)
        decision = evaluate_change_policy(
            catalog_diff=diff,
            generalized_diff={
                "added_rule_ids": [],
                "removed_rule_ids": [],
                "modified_rule_ids": [],
            },
            lineage=_empty_lineage(),
            policy=_policy(),
        )

        self.assertFalse(decision["passed"])
        self.assertIn("existing_rule_group_moves", decision["violations"])
        self.assertIn("rule_group_scope", decision["violations"])

    def test_scoped_recluster_allows_budgeted_group_move_only_in_allowed_topic(self) -> None:
        base = _catalog(
            [
                {
                    "id": "cluster",
                    "rule_ids": ["r1"],
                    "rule_groups": [
                        {"id": "g1", "rule_ids": ["r1"]},
                        {"id": "g2", "rule_ids": []},
                    ],
                }
            ]
        )
        candidate = _catalog(
            [
                {
                    "id": "cluster",
                    "rule_ids": ["r1"],
                    "rule_groups": [
                        {"id": "g1", "rule_ids": []},
                        {"id": "g2", "rule_ids": ["r1"]},
                    ],
                }
            ]
        )
        diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(candidate),
            affected_topics=[("Mechanics", "Kinematics")],
        )
        generalized_diff = {
            "added_rule_ids": [],
            "removed_rule_ids": [],
            "modified_rule_ids": [],
        }

        allowed = evaluate_change_policy(
            catalog_diff=diff,
            generalized_diff=generalized_diff,
            lineage=_empty_lineage(),
            policy=_policy(
                mode="scoped_recluster",
                max_existing_rule_group_moves=1,
                allowed_recluster_topics=[
                    {"domain": "Mechanics", "topic": "Kinematics"}
                ],
            ),
        )
        outside_scope = evaluate_change_policy(
            catalog_diff=diff,
            generalized_diff=generalized_diff,
            lineage=_empty_lineage(),
            policy=_policy(
                mode="scoped_recluster",
                max_existing_rule_group_moves=1,
                allowed_recluster_topics=[
                    {"domain": "Mechanics", "topic": "Dynamics"}
                ],
            ),
        )

        self.assertTrue(allowed["passed"])
        self.assertFalse(outside_scope["passed"])
        self.assertIn("rule_group_scope", outside_scope["violations"])

    def test_undeclared_cluster_definition_and_order_change_are_reported(self) -> None:
        base = _catalog(
            [
                {"id": "a", "name": "A", "rule_ids": ["r1"]},
                {"id": "b", "name": "B", "rule_ids": ["r2"]},
            ]
        )
        candidate = _catalog(
            [
                {"id": "b", "name": "B changed", "rule_ids": ["r2"]},
                {"id": "a", "name": "A", "rule_ids": ["r1"]},
            ]
        )
        diff = compare_catalog_snapshots(
            build_catalog_snapshot(base),
            build_catalog_snapshot(candidate),
            affected_topics=[("Mechanics", "Dynamics")],
        )

        self.assertEqual(
            diff["topics"]["unexpected_changed"],
            [{"domain": "Mechanics", "topic": "Kinematics"}],
        )
        self.assertEqual(
            diff["topics"]["cluster_order_changed"],
            [{"domain": "Mechanics", "topic": "Kinematics"}],
        )
        self.assertEqual(
            [item["cluster_id"] for item in diff["clusters"]["definition_changed"]],
            ["b"],
        )

    def test_added_rule_without_changed_candidate_source_fails_lineage_gate(self) -> None:
        generalized = {
            "rules": [{"rule_id": "new_rule"}],
            "cluster_results": [
                {
                    "mappings": [
                        {
                            "rule_id": "new_rule",
                            "source_candidate_ids": ["old_candidate"],
                        }
                    ]
                }
            ],
            "pending_candidate_ids": ["new_candidate"],
        }
        candidate_bank = {
            "rules": [
                {"rule_id": "old_candidate", "sample_ids": ["old_sample"]},
                {"rule_id": "new_candidate", "sample_ids": ["new_sample"]},
            ]
        }
        lineage = audit_added_rule_lineage(
            added_rule_ids=["new_rule"],
            lineage=build_generalized_lineage(generalized, candidate_bank),
            changed_candidate_ids=["new_candidate"],
            generalized=generalized,
        )

        self.assertEqual(lineage["coverage_ratio"], 0.0)
        self.assertEqual(lineage["without_changed_source_rule_ids"], ["new_rule"])
        self.assertEqual(
            lineage["changed_candidate_accounting"]["pending"], ["new_candidate"]
        )

    def test_generalized_removal_and_addition_are_bidirectional_budget_inputs(self) -> None:
        generalized_diff = compare_generalized_outputs(
            {"rules": [{"rule_id": "old_a"}, {"rule_id": "old_b"}]},
            {"rules": [{"rule_id": "old_a"}, {"rule_id": "new_a"}]},
        )
        catalog_diff = compare_catalog_snapshots(
            build_catalog_snapshot(_catalog([], [])),
            build_catalog_snapshot(_catalog([], [])),
        )
        decision = evaluate_change_policy(
            catalog_diff=catalog_diff,
            generalized_diff=generalized_diff,
            lineage=_empty_lineage(),
            policy=_policy(),
        )

        self.assertEqual(generalized_diff["added_rule_ids"], ["new_a"])
        self.assertEqual(generalized_diff["removed_rule_ids"], ["old_b"])
        self.assertFalse(decision["passed"])
        self.assertIn("removed_generalized_rules", decision["violations"])

    def test_mapping_cannot_attribute_a_formal_rule_absent_from_generalized_rules(self) -> None:
        base_catalog = _catalog([], [])
        final_catalog = _catalog(
            [{"id": "new_cluster", "rule_ids": ["formal_injected"]}],
            ["formal_injected"],
        )
        catalog_diff = compare_catalog_snapshots(
            build_catalog_snapshot(base_catalog),
            build_catalog_snapshot(final_catalog),
            affected_topics=[("Mechanics", "Kinematics")],
        )
        generalized = {
            "rules": [],
            "cluster_results": [
                {
                    "mappings": [
                        {
                            "rule_id": "formal_injected",
                            "source_candidate_ids": ["candidate_new"],
                        }
                    ]
                }
            ],
            "pending_candidate_ids": [],
        }
        lineage = audit_added_rule_lineage(
            added_rule_ids=catalog_diff["added_rule_ids"],
            lineage=build_generalized_lineage(
                generalized,
                {"rules": [{"rule_id": "candidate_new", "sample_ids": ["s1"]}]},
            ),
            changed_candidate_ids=["candidate_new"],
            generalized=generalized,
        )
        decision = evaluate_change_policy(
            catalog_diff=catalog_diff,
            generalized_diff={
                "added_rule_ids": [],
                "removed_rule_ids": [],
                "modified_rule_ids": [],
            },
            lineage=lineage,
            policy=_policy(),
        )

        self.assertEqual(lineage["coverage_ratio"], 0.0)
        self.assertEqual(
            lineage["unknown_mapping_rule_ids"], ["formal_injected"]
        )
        self.assertFalse(decision["passed"])
        self.assertIn("lineage_consistent", decision["violations"])
        self.assertIn(
            "catalog_additions_come_from_new_generalized",
            decision["violations"],
        )


if __name__ == "__main__":
    unittest.main()
