from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


TopicKey = Tuple[str, str]
ClusterKey = Tuple[str, str, str]
RuleGroupKey = Tuple[str, str, str, str]


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _ordered_unique(values: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        item = _text(value)
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def incremental_artifact_binding(
    manifest_path: Path,
    *,
    stage: str,
    input_paths: Mapping[str, Path] | None = None,
    input_sha256: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Bind one incremental artifact to its configuration and direct parents."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Incremental manifest must contain a JSON object: {manifest_path}"
        )
    expected = _text(payload.get("configuration_sha256") or "")
    actual = incremental_manifest_configuration_sha256(payload)
    if not expected:
        raise ValueError(
            f"Incremental manifest is missing configuration_sha256: {manifest_path}"
        )
    if expected != actual:
        raise ValueError(
            "Incremental manifest configuration_sha256 does not match its current "
            f"configuration: {manifest_path}"
        )
    parents = {
        name: sha256_file(Path(path))
        for name, path in sorted((input_paths or {}).items())
    }
    for name, digest in sorted((input_sha256 or {}).items()):
        normalized_digest = _text(digest)
        if not normalized_digest:
            raise ValueError(f"Incremental artifact parent {name!r} has no SHA-256")
        parents[name] = normalized_digest
    return {
        "incremental_configuration_sha256": expected,
        "incremental_manifest": str(manifest_path).replace("\\", "/"),
        "incremental_lineage": {
            "schema_version": 1,
            "stage": _text(stage),
            "input_sha256": parents,
        },
    }


def _existing_relative_order_changed(
    base_order: Iterable[str], candidate_order: Iterable[str]
) -> bool:
    base_items = list(base_order)
    candidate_items = list(candidate_order)
    common = set(base_items).intersection(candidate_items)
    return [item for item in base_items if item in common] != [
        item for item in candidate_items if item in common
    ]


def _normalized_manifest_commands(manifest: Mapping[str, Any]) -> List[Any]:
    normalized: List[Any] = []
    for raw_item in manifest.get("commands", []) or []:
        if not isinstance(raw_item, dict):
            normalized.append(raw_item)
            continue
        item = dict(raw_item)
        command = item.get("command")
        if isinstance(command, str):
            try:
                parts = shlex.split(command)
            except ValueError:
                parts = [command]
            filtered: List[str] = []
            skip_value = False
            for part in parts:
                if skip_value:
                    skip_value = False
                    continue
                if part == "--incremental-configuration-sha256":
                    skip_value = True
                    continue
                if part.startswith("--incremental-configuration-sha256="):
                    continue
                filtered.append(part)
            item["command"] = filtered
        normalized.append(item)
    return normalized


def incremental_manifest_configuration_sha256(manifest: Mapping[str, Any]) -> str:
    return _sha256_json(
        {
            "schema_version": manifest.get("schema_version"),
            "inputs": manifest.get("inputs"),
            "candidate_delta": manifest.get("candidate_delta"),
            "candidate_affected_topics": manifest.get("candidate_affected_topics"),
            "declared_change_topics": manifest.get("declared_change_topics"),
            "affected_topics": manifest.get("affected_topics"),
            "change_policy": manifest.get("change_policy"),
            "commands": _normalized_manifest_commands(manifest),
            "run_configuration": manifest.get("run_configuration"),
        }
    )


def _topic_item(key: TopicKey) -> Dict[str, str]:
    return {"domain": key[0], "topic": key[1]}


def _domain_item(key: str) -> Dict[str, str]:
    return {"domain": key}


def _cluster_item(key: ClusterKey) -> Dict[str, str]:
    return {"domain": key[0], "topic": key[1], "cluster_id": key[2]}


def _rule_group_item(key: RuleGroupKey) -> Dict[str, str]:
    return {
        "domain": key[0],
        "topic": key[1],
        "cluster_id": key[2],
        "group_id": key[3],
    }


def _domain_definition(domain: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in domain.items() if key != "topics"}


def _topic_definition(topic: Dict[str, Any]) -> Dict[str, Any]:
    dynamic_members = {
        "rules",
        "rule_ids",
        "rule_tree",
        "clusters",
        "scenario_clusters",
    }
    return {
        key: value for key, value in topic.items() if key not in dynamic_members
    }


def _cluster_definition(cluster: Dict[str, Any]) -> Dict[str, Any]:
    definition = {
        key: value
        for key, value in cluster.items()
        if key not in {"rule_ids", "rule_groups"}
    }
    groups: List[Dict[str, Any]] = []
    for group in cluster.get("rule_groups", []) or []:
        if not isinstance(group, dict):
            continue
        groups.append({key: value for key, value in group.items() if key != "rule_ids"})
    if groups:
        definition["rule_groups"] = groups
    return definition


def build_catalog_snapshot(catalog: Dict[str, Any]) -> Dict[str, Any]:
    """Build a lossless runtime-structure snapshot for incremental comparisons."""
    domain_order: List[str] = []
    domains: Dict[str, Dict[str, Any]] = {}
    topics: Dict[TopicKey, Dict[str, Any]] = {}
    rules: Dict[str, Dict[str, Any]] = {}
    rule_topics: Dict[str, TopicKey] = {}
    rule_clusters: Dict[str, List[ClusterKey]] = {}
    rule_groups: Dict[str, List[RuleGroupKey]] = {}
    rule_group_contents: Dict[RuleGroupKey, Dict[str, Any]] = {}
    clusters: Dict[ClusterKey, Dict[str, Any]] = {}

    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = _text(domain.get("name") or domain.get("id") or "Unknown")
        domain_order.append(domain_name)
        domains[domain_name] = {
            "definition_sha256": _sha256_json(_domain_definition(domain)),
            "topic_ids": [],
        }
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            topic_name = _text(topic.get("name") or topic.get("id") or "Unknown")
            topic_key = (domain_name, topic_name)
            domains[domain_name]["topic_ids"].append(topic_name)
            topic_rules = [
                rule for rule in (topic.get("rules") or []) if isinstance(rule, dict)
            ]
            rule_order = [
                _text(rule.get("rule_id") or rule.get("id") or "")
                for rule in topic_rules
                if _text(rule.get("rule_id") or rule.get("id") or "")
            ]
            topic_clusters = [
                cluster
                for cluster in (topic.get("scenario_clusters") or topic.get("clusters") or [])
                if isinstance(cluster, dict)
            ]
            cluster_order: List[str] = []
            for index, cluster in enumerate(topic_clusters):
                cluster_id = _text(
                    cluster.get("id") or cluster.get("cluster_id") or f"@index:{index}"
                )
                cluster_key = (domain_name, topic_name, cluster_id)
                cluster_order.append(cluster_id)
                cluster_rule_ids = _ordered_unique(cluster.get("rule_ids") or [])
                clusters[cluster_key] = {
                    "definition_sha256": _sha256_json(_cluster_definition(cluster)),
                    "rule_order_sha256": _sha256_json(cluster_rule_ids),
                    "full_sha256": _sha256_json(cluster),
                    "rule_ids": cluster_rule_ids,
                    "rule_group_ids": [],
                }
                for rule_id in cluster_rule_ids:
                    rule_clusters.setdefault(rule_id, []).append(cluster_key)
                for group_index, group in enumerate(
                    cluster.get("rule_groups", []) or []
                ):
                    if not isinstance(group, dict):
                        continue
                    group_id = _text(
                        group.get("id")
                        or group.get("group_id")
                        or f"@index:{group_index}"
                    )
                    group_key = (*cluster_key, group_id)
                    group_rule_ids = _ordered_unique(group.get("rule_ids") or [])
                    clusters[cluster_key]["rule_group_ids"].append(group_id)
                    rule_group_contents[group_key] = {
                        "rule_ids": group_rule_ids,
                        "rule_order_sha256": _sha256_json(group_rule_ids),
                    }
                    for rule_id in group_rule_ids:
                        rule_groups.setdefault(rule_id, []).append(group_key)

            topics[topic_key] = {
                "definition_sha256": _sha256_json(_topic_definition(topic)),
                "rule_ids": rule_order,
                "rule_id_set": set(rule_order),
                "rule_order_sha256": _sha256_json(rule_order),
                "cluster_ids": cluster_order,
                "cluster_order_sha256": _sha256_json(cluster_order),
                "runtime_sha256": _sha256_json(topic),
            }
            for rule in topic_rules:
                rule_id = _text(rule.get("rule_id") or rule.get("id") or "")
                if not rule_id:
                    continue
                rules[rule_id] = rule
                rule_topics[rule_id] = topic_key

    return {
        "domain_order": domain_order,
        "domains": domains,
        "topics": topics,
        "rules": rules,
        "rule_topics": rule_topics,
        "rule_clusters": {
            rule_id: tuple(paths) for rule_id, paths in rule_clusters.items()
        },
        "rule_groups": {
            rule_id: tuple(paths) for rule_id, paths in rule_groups.items()
        },
        "rule_group_contents": rule_group_contents,
        "clusters": clusters,
    }


def compare_catalog_snapshots(
    base: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    affected_topics: Iterable[TopicKey] = (),
) -> Dict[str, Any]:
    base_rules = set(base["rules"])
    candidate_rules = set(candidate["rules"])
    existing_rule_ids = sorted(base_rules.intersection(candidate_rules))
    affected = set(affected_topics)

    existing_domain_order_changed = _existing_relative_order_changed(
        base.get("domain_order", []),
        candidate.get("domain_order", []),
    )
    existing_topic_order_changed_domain_keys = sorted(
        key
        for key in set(base["domains"]).intersection(candidate["domains"])
        if _existing_relative_order_changed(
            base["domains"][key].get("topic_ids", []),
            candidate["domains"][key].get("topic_ids", []),
        )
    )

    changed_domain_definition_keys = sorted(
        key
        for key in set(base["domains"]).intersection(candidate["domains"])
        if base["domains"][key]["definition_sha256"]
        != candidate["domains"][key]["definition_sha256"]
    )

    modified_existing_rule_ids = sorted(
        rule_id
        for rule_id in existing_rule_ids
        if _sha256_json(base["rules"][rule_id])
        != _sha256_json(candidate["rules"][rule_id])
    )
    rule_topic_changes = [
        {
            "rule_id": rule_id,
            "before": _topic_item(base["rule_topics"][rule_id]),
            "after": _topic_item(candidate["rule_topics"][rule_id]),
        }
        for rule_id in existing_rule_ids
        if base["rule_topics"].get(rule_id) != candidate["rule_topics"].get(rule_id)
    ]
    rule_cluster_changes = [
        {
            "rule_id": rule_id,
            "topic": _topic_item(base["rule_topics"][rule_id]),
            "before_topic": _topic_item(base["rule_topics"][rule_id]),
            "after_topic": _topic_item(candidate["rule_topics"][rule_id]),
            "before_cluster_ids": [path[2] for path in base["rule_clusters"].get(rule_id, ())],
            "after_cluster_ids": [
                path[2] for path in candidate["rule_clusters"].get(rule_id, ())
            ],
        }
        for rule_id in existing_rule_ids
        if base["rule_clusters"].get(rule_id, ())
        != candidate["rule_clusters"].get(rule_id, ())
    ]
    rule_group_changes = [
        {
            "rule_id": rule_id,
            "topic": _topic_item(base["rule_topics"][rule_id]),
            "before_topic": _topic_item(base["rule_topics"][rule_id]),
            "after_topic": _topic_item(candidate["rule_topics"][rule_id]),
            "before_group_paths": [
                {"cluster_id": path[2], "group_id": path[3]}
                for path in base["rule_groups"].get(rule_id, ())
            ],
            "after_group_paths": [
                {"cluster_id": path[2], "group_id": path[3]}
                for path in candidate["rule_groups"].get(rule_id, ())
            ],
        }
        for rule_id in existing_rule_ids
        if base["rule_groups"].get(rule_id, ())
        != candidate["rule_groups"].get(rule_id, ())
    ]

    topic_keys = set(base["topics"]).union(candidate["topics"])
    changed_topic_keys = {
        key
        for key in topic_keys
        if (base["topics"].get(key) or {}).get("runtime_sha256")
        != (candidate["topics"].get(key) or {}).get("runtime_sha256")
    }
    changed_topic_definition_keys = sorted(
        key
        for key in set(base["topics"]).intersection(candidate["topics"])
        if base["topics"][key]["definition_sha256"]
        != candidate["topics"][key]["definition_sha256"]
    )
    rule_set_changed_topics = {
        key
        for key in topic_keys
        if (base["topics"].get(key) or {}).get("rule_id_set", set())
        != (candidate["topics"].get(key) or {}).get("rule_id_set", set())
    }
    rule_order_changed_topics = {
        key
        for key in topic_keys
        if (base["topics"].get(key) or {}).get("rule_order_sha256")
        != (candidate["topics"].get(key) or {}).get("rule_order_sha256")
    }
    existing_rule_order_changed_topics = sorted(
        key
        for key in set(base["topics"]).intersection(candidate["topics"])
        if _existing_relative_order_changed(
            base["topics"][key]["rule_ids"],
            candidate["topics"][key]["rule_ids"],
        )
    )
    cluster_order_changed_topics = {
        key
        for key in topic_keys
        if (base["topics"].get(key) or {}).get("cluster_order_sha256")
        != (candidate["topics"].get(key) or {}).get("cluster_order_sha256")
    }
    existing_cluster_order_changed_topics = sorted(
        key
        for key in set(base["topics"]).intersection(candidate["topics"])
        if _existing_relative_order_changed(
            base["topics"][key]["cluster_ids"],
            candidate["topics"][key]["cluster_ids"],
        )
    )

    cluster_keys = set(base["clusters"]).union(candidate["clusters"])
    existing_rule_group_order_changed_cluster_keys = sorted(
        key
        for key in set(base["clusters"]).intersection(candidate["clusters"])
        if _existing_relative_order_changed(
            base["clusters"][key].get("rule_group_ids", []),
            candidate["clusters"][key].get("rule_group_ids", []),
        )
    )
    existing_cluster_rule_order_changed_keys = sorted(
        key
        for key in set(base["clusters"]).intersection(candidate["clusters"])
        if _existing_relative_order_changed(
            base["clusters"][key].get("rule_ids", []),
            candidate["clusters"][key].get("rule_ids", []),
        )
    )
    existing_rule_group_rule_order_changed_keys = sorted(
        key
        for key in set(base["rule_group_contents"]).intersection(
            candidate["rule_group_contents"]
        )
        if _existing_relative_order_changed(
            base["rule_group_contents"][key]["rule_ids"],
            candidate["rule_group_contents"][key]["rule_ids"],
        )
    )
    added_cluster_keys = sorted(set(candidate["clusters"]) - set(base["clusters"]))
    removed_cluster_keys = sorted(set(base["clusters"]) - set(candidate["clusters"]))
    changed_cluster_definition_keys = sorted(
        key
        for key in set(base["clusters"]).intersection(candidate["clusters"])
        if base["clusters"][key]["definition_sha256"]
        != candidate["clusters"][key]["definition_sha256"]
    )
    changed_cluster_keys = sorted(
        key
        for key in cluster_keys
        if (base["clusters"].get(key) or {}).get("full_sha256")
        != (candidate["clusters"].get(key) or {}).get("full_sha256")
    )
    unexpected_topic_keys = sorted(changed_topic_keys - affected)
    unexpected_cluster_keys = [
        key for key in changed_cluster_keys if (key[0], key[1]) not in affected
    ]

    return {
        "added_rule_ids": sorted(candidate_rules - base_rules),
        "removed_rule_ids": sorted(base_rules - candidate_rules),
        "modified_existing_rule_ids": modified_existing_rule_ids,
        "rule_topic_changes": rule_topic_changes,
        "rule_cluster_mapping": {
            "changed_existing_count": len(rule_cluster_changes),
            "changes": rule_cluster_changes,
        },
        "rule_group_mapping": {
            "changed_existing_count": len(rule_group_changes),
            "changes": rule_group_changes,
        },
        "rule_groups": {
            "existing_rule_order_changed": [
                _rule_group_item(key)
                for key in existing_rule_group_rule_order_changed_keys
            ],
        },
        "domains": {
            "definition_changed": [
                _domain_item(key) for key in changed_domain_definition_keys
            ],
            "existing_order_changed": existing_domain_order_changed,
            "existing_topic_order_changed": [
                _domain_item(key)
                for key in existing_topic_order_changed_domain_keys
            ],
        },
        "topics": {
            "changed": [_topic_item(key) for key in sorted(changed_topic_keys)],
            "definition_changed": [
                _topic_item(key) for key in changed_topic_definition_keys
            ],
            "rule_set_changed": [
                _topic_item(key) for key in sorted(rule_set_changed_topics)
            ],
            "rule_order_changed": [
                _topic_item(key) for key in sorted(rule_order_changed_topics)
            ],
            "existing_rule_order_changed": [
                _topic_item(key) for key in existing_rule_order_changed_topics
            ],
            "cluster_order_changed": [
                _topic_item(key) for key in sorted(cluster_order_changed_topics)
            ],
            "existing_cluster_order_changed": [
                _topic_item(key) for key in existing_cluster_order_changed_topics
            ],
            "unexpected_changed": [
                _topic_item(key) for key in unexpected_topic_keys
            ],
        },
        "clusters": {
            "added": [_cluster_item(key) for key in added_cluster_keys],
            "removed": [_cluster_item(key) for key in removed_cluster_keys],
            "definition_changed": [
                _cluster_item(key) for key in changed_cluster_definition_keys
            ],
            "changed": [_cluster_item(key) for key in changed_cluster_keys],
            "unexpected_changed": [
                _cluster_item(key) for key in unexpected_cluster_keys
            ],
            "existing_rule_group_order_changed": [
                _cluster_item(key)
                for key in existing_rule_group_order_changed_cluster_keys
            ],
            "existing_rule_order_changed": [
                _cluster_item(key)
                for key in existing_cluster_rule_order_changed_keys
            ],
        },
    }


def build_generalized_lineage(
    generalized: Dict[str, Any],
    candidate_bank: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_samples = {
        _text(rule.get("rule_id") or ""): _ordered_unique(rule.get("sample_ids") or [])
        for rule in candidate_bank.get("rules", []) or []
        if isinstance(rule, dict) and _text(rule.get("rule_id") or "")
    }
    generalized_rule_ids = {
        _text(rule.get("rule_id") or "")
        for rule in generalized.get("rules", []) or []
        if isinstance(rule, dict) and _text(rule.get("rule_id") or "")
    }
    lineage: Dict[str, List[str]] = {}
    conflicting_rule_ids: set[str] = set()
    unknown_mapping_rule_ids: set[str] = set()
    for result in generalized.get("cluster_results", []) or []:
        if not isinstance(result, dict):
            continue
        for mapping in result.get("mappings", []) or []:
            if not isinstance(mapping, dict):
                continue
            rule_id = _text(mapping.get("rule_id") or "")
            source_ids = sorted(
                _ordered_unique(mapping.get("source_candidate_ids") or [])
            )
            if not rule_id:
                continue
            if rule_id not in generalized_rule_ids:
                unknown_mapping_rule_ids.add(rule_id)
                continue
            if rule_id in lineage and lineage[rule_id] != source_ids:
                conflicting_rule_ids.add(rule_id)
                lineage[rule_id] = _ordered_unique([*lineage[rule_id], *source_ids])
            else:
                lineage[rule_id] = source_ids

    unknown_candidate_ids = sorted(
        {
            candidate_id
            for source_ids in lineage.values()
            for candidate_id in source_ids
            if candidate_id not in candidate_samples
        }
    )
    return {
        "source_candidate_ids_by_rule": lineage,
        "source_sample_ids_by_rule": {
            rule_id: _ordered_unique(
                sample_id
                for candidate_id in source_ids
                for sample_id in candidate_samples.get(candidate_id, [])
            )
            for rule_id, source_ids in lineage.items()
        },
        "conflicting_rule_ids": sorted(conflicting_rule_ids),
        "unknown_mapping_rule_ids": sorted(unknown_mapping_rule_ids),
        "unmapped_generalized_rule_ids": sorted(
            generalized_rule_ids - set(lineage)
        ),
        "unknown_candidate_ids": unknown_candidate_ids,
    }


def audit_added_rule_lineage(
    *,
    added_rule_ids: Iterable[str],
    lineage: Dict[str, Any],
    changed_candidate_ids: Iterable[str],
    generalized: Dict[str, Any],
) -> Dict[str, Any]:
    changed = set(_ordered_unique(changed_candidate_ids))
    source_map = lineage["source_candidate_ids_by_rule"]
    sample_map = lineage["source_sample_ids_by_rule"]
    rows: List[Dict[str, Any]] = []
    with_changed_source = 0
    for rule_id in sorted(_ordered_unique(added_rule_ids)):
        source_ids = list(source_map.get(rule_id, []))
        changed_source_ids = [item for item in source_ids if item in changed]
        if changed_source_ids:
            with_changed_source += 1
        rows.append(
            {
                "rule_id": rule_id,
                "source_candidate_ids": source_ids,
                "changed_source_candidate_ids": changed_source_ids,
                "source_sample_ids": list(sample_map.get(rule_id, [])),
            }
        )

    mapped_candidate_ids = {
        candidate_id for source_ids in source_map.values() for candidate_id in source_ids
    }
    pending_candidate_ids = set(
        _ordered_unique(
            [
                *(generalized.get("pending_candidate_ids") or []),
                *(generalized.get("residual_candidate_ids") or []),
                *(generalized.get("unclustered_candidate_ids") or []),
            ]
        )
    )
    missing_candidate_ids = set(
        _ordered_unique(generalized.get("missing_candidate_ids") or [])
    )
    unaccounted_changed_candidate_ids = sorted(
        changed - mapped_candidate_ids - pending_candidate_ids - missing_candidate_ids
    )
    count = len(rows)
    return {
        "added_rule_count": count,
        "with_changed_source_count": with_changed_source,
        "coverage_ratio": 1.0 if count == 0 else with_changed_source / count,
        "unattributed_added_rule_ids": [
            row["rule_id"] for row in rows if not row["source_candidate_ids"]
        ],
        "without_changed_source_rule_ids": [
            row["rule_id"] for row in rows if not row["changed_source_candidate_ids"]
        ],
        "conflicting_rule_ids": lineage["conflicting_rule_ids"],
        "unknown_mapping_rule_ids": lineage["unknown_mapping_rule_ids"],
        "unmapped_generalized_rule_ids": lineage[
            "unmapped_generalized_rule_ids"
        ],
        "unknown_candidate_ids": lineage["unknown_candidate_ids"],
        "changed_candidate_accounting": {
            "mapped": sorted(changed.intersection(mapped_candidate_ids)),
            "pending": sorted(changed.intersection(pending_candidate_ids)),
            "missing": sorted(changed.intersection(missing_candidate_ids)),
            "unaccounted": unaccounted_changed_candidate_ids,
        },
        "rules": rows,
    }


def compare_generalized_outputs(
    base: Dict[str, Any], candidate: Dict[str, Any]
) -> Dict[str, Any]:
    def index(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {
            _text(rule.get("rule_id") or ""): rule
            for rule in payload.get("rules", []) or []
            if isinstance(rule, dict) and _text(rule.get("rule_id") or "")
        }

    base_rules = index(base)
    candidate_rules = index(candidate)
    retained = set(base_rules).intersection(candidate_rules)
    modified = sorted(
        rule_id
        for rule_id in retained
        if _sha256_json(base_rules[rule_id]) != _sha256_json(candidate_rules[rule_id])
    )
    return {
        "base_rule_count": len(base_rules),
        "candidate_rule_count": len(candidate_rules),
        "retained_rule_count": len(retained),
        "added_rule_ids": sorted(set(candidate_rules) - set(base_rules)),
        "removed_rule_ids": sorted(set(base_rules) - set(candidate_rules)),
        "modified_rule_ids": modified,
    }


def validate_change_policy(policy: Mapping[str, Any]) -> Dict[str, Any]:
    integer_fields = (
        "max_added_formal_rules",
        "max_removed_formal_rules",
        "max_modified_existing_rules",
        "max_existing_rule_topic_moves",
        "max_existing_rule_cluster_moves",
        "max_existing_rule_group_moves",
        "max_added_clusters",
        "max_removed_clusters",
        "max_changed_cluster_definitions",
        "max_changed_topics",
        "max_unexpected_changed_topics",
        "max_added_generalized_rules",
        "max_removed_generalized_rules",
        "max_modified_generalized_rules",
    )
    required_fields = {
        "mode",
        *integer_fields,
        "min_added_rule_incremental_source_ratio",
        "allowed_recluster_topics",
    }
    missing_fields = sorted(required_fields - set(policy))
    if missing_fields:
        raise ValueError(
            "change_policy is missing required fields: " + ", ".join(missing_fields)
        )
    normalized = dict(policy)
    for field in integer_fields:
        value = int(normalized.get(field, 0))
        if value < 0:
            raise ValueError(f"change_policy.{field} must be non-negative")
        normalized[field] = value
    ratio = float(normalized.get("min_added_rule_incremental_source_ratio", 1.0))
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(
            "change_policy.min_added_rule_incremental_source_ratio must be in [0, 1]"
        )
    normalized["min_added_rule_incremental_source_ratio"] = ratio
    mode = _text(normalized.get("mode") or "additive")
    if mode not in {"additive", "scoped_recluster"}:
        raise ValueError("change_policy.mode must be additive or scoped_recluster")
    normalized["mode"] = mode
    normalized["allowed_recluster_topics"] = [
        _topic_item((
            _text(item.get("domain") or ""),
            _text(item.get("topic") or ""),
        ))
        for item in normalized.get("allowed_recluster_topics", []) or []
        if isinstance(item, dict)
        and _text(item.get("domain") or "")
        and _text(item.get("topic") or "")
    ]
    if mode == "additive" and normalized["allowed_recluster_topics"]:
        raise ValueError(
            "change_policy.allowed_recluster_topics must be empty in additive mode"
        )
    if mode == "scoped_recluster" and not normalized["allowed_recluster_topics"]:
        raise ValueError(
            "change_policy.allowed_recluster_topics must be non-empty in scoped_recluster mode"
        )
    for field in (
        "max_removed_formal_rules",
        "max_modified_existing_rules",
        "max_existing_rule_topic_moves",
    ):
        if normalized[field] != 0:
            raise ValueError(f"change_policy.{field} must remain zero")
    if mode == "additive":
        for field in (
            "max_existing_rule_cluster_moves",
            "max_existing_rule_group_moves",
            "max_removed_clusters",
            "max_changed_cluster_definitions",
        ):
            if normalized[field] != 0:
                raise ValueError(
                    f"change_policy.{field} must remain zero in additive mode"
                )
    return normalized


def evaluate_change_policy(
    *,
    catalog_diff: Dict[str, Any],
    generalized_diff: Dict[str, Any],
    lineage: Dict[str, Any],
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    budget = validate_change_policy(policy)
    allowed_recluster_topics = {
        (_text(item["domain"]), _text(item["topic"]))
        for item in budget["allowed_recluster_topics"]
    }
    cluster_changes = catalog_diff["rule_cluster_mapping"]["changes"]
    group_changes = catalog_diff["rule_group_mapping"]["changes"]
    topic_changes = catalog_diff["rule_topic_changes"]
    unallowed_topic_moves = [
        item
        for item in topic_changes
        if (item["before"]["domain"], item["before"]["topic"])
        not in allowed_recluster_topics
        or (item["after"]["domain"], item["after"]["topic"])
        not in allowed_recluster_topics
    ]
    unallowed_cluster_moves = [
        item
        for item in cluster_changes
        if (
            item["before_topic"]["domain"],
            item["before_topic"]["topic"],
        )
        not in allowed_recluster_topics
        or (item["after_topic"]["domain"], item["after_topic"]["topic"])
        not in allowed_recluster_topics
    ]
    unallowed_group_moves = [
        item
        for item in group_changes
        if (
            item["before_topic"]["domain"],
            item["before_topic"]["topic"],
        )
        not in allowed_recluster_topics
        or (item["after_topic"]["domain"], item["after_topic"]["topic"])
        not in allowed_recluster_topics
    ]
    unallowed_cluster_definition_changes = [
        item
        for item in catalog_diff["clusters"]["definition_changed"]
        if (item["domain"], item["topic"]) not in allowed_recluster_topics
    ]
    unallowed_removed_clusters = [
        item
        for item in catalog_diff["clusters"]["removed"]
        if (item["domain"], item["topic"]) not in allowed_recluster_topics
    ]
    existing_cluster_order_changes = catalog_diff["topics"][
        "existing_cluster_order_changed"
    ]
    unallowed_existing_cluster_order_changes = [
        item
        for item in existing_cluster_order_changes
        if (item["domain"], item["topic"]) not in allowed_recluster_topics
    ]
    existing_rule_group_order_changes = catalog_diff["clusters"][
        "existing_rule_group_order_changed"
    ]
    unallowed_existing_rule_group_order_changes = [
        item
        for item in existing_rule_group_order_changes
        if (item["domain"], item["topic"]) not in allowed_recluster_topics
    ]
    existing_cluster_rule_order_changes = catalog_diff["clusters"][
        "existing_rule_order_changed"
    ]
    unallowed_existing_cluster_rule_order_changes = [
        item
        for item in existing_cluster_rule_order_changes
        if (item["domain"], item["topic"]) not in allowed_recluster_topics
    ]
    existing_rule_group_rule_order_changes = catalog_diff["rule_groups"][
        "existing_rule_order_changed"
    ]
    unallowed_existing_rule_group_rule_order_changes = [
        item
        for item in existing_rule_group_rule_order_changes
        if (item["domain"], item["topic"]) not in allowed_recluster_topics
    ]
    actual = {
        "added_formal_rules": len(catalog_diff["added_rule_ids"]),
        "removed_formal_rules": len(catalog_diff["removed_rule_ids"]),
        "modified_existing_rules": len(catalog_diff["modified_existing_rule_ids"]),
        "existing_rule_topic_moves": len(topic_changes),
        "unallowed_existing_rule_topic_moves": len(unallowed_topic_moves),
        "existing_rule_cluster_moves": len(cluster_changes),
        "unallowed_existing_rule_cluster_moves": len(unallowed_cluster_moves),
        "existing_rule_group_moves": len(group_changes),
        "unallowed_existing_rule_group_moves": len(unallowed_group_moves),
        "unallowed_cluster_definition_changes": len(
            unallowed_cluster_definition_changes
        ),
        "changed_domain_definitions": len(
            catalog_diff["domains"]["definition_changed"]
        ),
        "existing_domain_order_changes": int(
            bool(catalog_diff["domains"]["existing_order_changed"])
        ),
        "existing_topic_order_changes": len(
            catalog_diff["domains"]["existing_topic_order_changed"]
        ),
        "changed_topic_definitions": len(
            catalog_diff["topics"]["definition_changed"]
        ),
        "existing_rule_order_changes": len(
            catalog_diff["topics"]["existing_rule_order_changed"]
        ),
        "existing_cluster_order_changes": len(existing_cluster_order_changes),
        "unallowed_existing_cluster_order_changes": len(
            unallowed_existing_cluster_order_changes
        ),
        "existing_rule_group_order_changes": len(
            existing_rule_group_order_changes
        ),
        "unallowed_existing_rule_group_order_changes": len(
            unallowed_existing_rule_group_order_changes
        ),
        "existing_cluster_rule_order_changes": len(
            existing_cluster_rule_order_changes
        ),
        "unallowed_existing_cluster_rule_order_changes": len(
            unallowed_existing_cluster_rule_order_changes
        ),
        "existing_rule_group_rule_order_changes": len(
            existing_rule_group_rule_order_changes
        ),
        "unallowed_existing_rule_group_rule_order_changes": len(
            unallowed_existing_rule_group_rule_order_changes
        ),
        "unallowed_removed_clusters": len(unallowed_removed_clusters),
        "added_clusters": len(catalog_diff["clusters"]["added"]),
        "removed_clusters": len(catalog_diff["clusters"]["removed"]),
        "changed_cluster_definitions": len(
            catalog_diff["clusters"]["definition_changed"]
        ),
        "changed_topics": len(catalog_diff["topics"]["changed"]),
        "unexpected_changed_topics": len(
            catalog_diff["topics"]["unexpected_changed"]
        ),
        "added_generalized_rules": len(generalized_diff["added_rule_ids"]),
        "removed_generalized_rules": len(generalized_diff["removed_rule_ids"]),
        "modified_generalized_rules": len(generalized_diff["modified_rule_ids"]),
        "added_rule_incremental_source_ratio": float(lineage["coverage_ratio"]),
        "unaccounted_changed_candidates": len(
            lineage["changed_candidate_accounting"]["unaccounted"]
        ),
        "lineage_conflicts": len(lineage["conflicting_rule_ids"]),
        "unknown_lineage_candidates": len(lineage["unknown_candidate_ids"]),
        "unknown_mapping_rules": len(lineage["unknown_mapping_rule_ids"]),
        "unmapped_generalized_rules": len(
            lineage["unmapped_generalized_rule_ids"]
        ),
        "catalog_added_without_new_generalized": len(
            set(catalog_diff["added_rule_ids"])
            - set(generalized_diff["added_rule_ids"])
        ),
    }
    comparisons = {
        "added_formal_rules": "max_added_formal_rules",
        "removed_formal_rules": "max_removed_formal_rules",
        "modified_existing_rules": "max_modified_existing_rules",
        "existing_rule_topic_moves": "max_existing_rule_topic_moves",
        "existing_rule_cluster_moves": "max_existing_rule_cluster_moves",
        "existing_rule_group_moves": "max_existing_rule_group_moves",
        "added_clusters": "max_added_clusters",
        "removed_clusters": "max_removed_clusters",
        "changed_cluster_definitions": "max_changed_cluster_definitions",
        "changed_topics": "max_changed_topics",
        "unexpected_changed_topics": "max_unexpected_changed_topics",
        "added_generalized_rules": "max_added_generalized_rules",
        "removed_generalized_rules": "max_removed_generalized_rules",
        "modified_generalized_rules": "max_modified_generalized_rules",
    }
    gates = {
        name: actual[name] <= budget[budget_name]
        for name, budget_name in comparisons.items()
    }
    gates.update(
        {
            "recluster_scope": actual["unallowed_existing_rule_cluster_moves"] == 0,
            "rule_group_scope": actual["unallowed_existing_rule_group_moves"] == 0,
            "rule_topic_scope": actual["unallowed_existing_rule_topic_moves"] == 0,
            "cluster_definition_scope": actual[
                "unallowed_cluster_definition_changes"
            ]
            == 0,
            "domain_definition_stable": actual[
                "changed_domain_definitions"
            ]
            == 0,
            "domain_order_stable": actual["existing_domain_order_changes"] == 0,
            "topic_order_stable": actual["existing_topic_order_changes"] == 0,
            "topic_definition_stable": actual[
                "changed_topic_definitions"
            ]
            == 0,
            "rule_order_stable": actual["existing_rule_order_changes"] == 0,
            "cluster_order_scope": actual[
                "unallowed_existing_cluster_order_changes"
            ]
            == 0,
            "rule_group_order_scope": actual[
                "unallowed_existing_rule_group_order_changes"
            ]
            == 0,
            "cluster_rule_order_scope": actual[
                "unallowed_existing_cluster_rule_order_changes"
            ]
            == 0,
            "rule_group_rule_order_scope": actual[
                "unallowed_existing_rule_group_rule_order_changes"
            ]
            == 0,
            "cluster_removal_scope": actual["unallowed_removed_clusters"] == 0,
            "incremental_source_coverage": actual[
                "added_rule_incremental_source_ratio"
            ]
            >= budget["min_added_rule_incremental_source_ratio"],
            "changed_candidate_accounting": actual[
                "unaccounted_changed_candidates"
            ]
            == 0,
            "lineage_consistent": actual["lineage_conflicts"] == 0
            and actual["unknown_lineage_candidates"] == 0
            and actual["unknown_mapping_rules"] == 0
            and actual["unmapped_generalized_rules"] == 0,
            "catalog_additions_come_from_new_generalized": actual[
                "catalog_added_without_new_generalized"
            ]
            == 0,
        }
    )
    violations = [name for name, passed in gates.items() if not passed]
    return {
        "passed": not violations,
        "gates": gates,
        "violations": violations,
        "budget": budget,
        "actual": actual,
    }
