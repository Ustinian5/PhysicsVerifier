from __future__ import annotations

"""Replay Checker + release-gate arms from an immutable candidate trace.

This entry point intentionally has no retrieval or bottom-up verification path.  A
frozen manifest binds the dataset, unified catalog, and either a genuine semantic
retrieval trace or an explicitly declared controlled target-rule binding. The only
executable rules are rebuilt from catalog rule IDs recorded in that trace; target
bindings are never relabeled as semantic retrieval.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.physics_rule_verifier import PhysicsRuleVerifier
from core.rule_catalog_retrieval import topic_rule_leaves
from core.semantic_rule_checker import (
    SEMANTIC_RULE_CHECKER_PROMPT_VERSION,
    _load_env_file_fallback,
)
from scripts.experiment_manifest import (
    capture_source_state,
    probe_python_runtime,
    sha256_file,
)


FROZEN_MANIFEST_SCHEMA_VERSION = 2
LEGACY_FROZEN_MANIFEST_SCHEMA_VERSION = 1
FROZEN_MANIFEST_TYPE = "checker_replay_frozen_retrieval"
TARGET_BINDING_MANIFEST_TYPE = "checker_replay_frozen_target_binding"
REPLAY_REPORT_SCHEMA_VERSION = 1
FROZEN_CHECKER_MIN_CONFIDENCE = 0.8
CANDIDATE_SOURCE_SEMANTIC = "semantic_retrieval"
CANDIDATE_SOURCE_TARGET_BINDING = "frozen_target_binding"
TARGET_BINDING_RETRIEVAL_MODE = "target_binding"
TARGET_BINDING_SELECTION_STRATEGY = "target_rule_binding"
TARGET_BINDING_SCORE_KIND = "fixed_control_0_1"
CHECKER_GATE_MODES = (
    "legacy",
    "dual_evidence",
    "dual_evidence_consistency",
)
_SUCCESS_CHECKER_STATUSES = {
    "complete",
    "ok",
    "success",
    "valid_empty",
    "valid_with_diagnostics",
}
_SENSITIVE_ENV_NAMES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)
_RUN_KINDS = ("development", "validation", "final")


class ReplayValidationError(ValueError):
    """Raised when a frozen artifact or replay checkpoint fails integrity checks."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file_prefix(path: Path, size_bytes: int) -> str:
    if size_bytes < 0:
        raise ReplayValidationError("trace prefix size must be non-negative")
    digest = hashlib.sha256()
    remaining = int(size_bytes)
    with path.open("rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ReplayValidationError("Checker LLM trace is shorter than its sidecar prefix")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise ReplayValidationError(f"{label} does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayValidationError(f"{label} is not valid JSON: {path}: {exc}") from exc


def _load_json_array(path: Path, *, label: str) -> List[Dict[str, Any]]:
    payload = _load_json(path, label=label)
    if not isinstance(payload, list):
        raise ReplayValidationError(f"{label} must be a JSON array: {path}")
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ReplayValidationError(f"{label}[{index}] must be a JSON object")
        rows.append(item)
    return rows


def _id_key(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ReplayValidationError("sample IDs must be non-empty strings or integers")
    if isinstance(value, str) and not value.strip():
        raise ReplayValidationError("sample IDs must not be empty")
    return f"{type(value).__name__}:{_canonical_json(value)}"


def _ordered_ids(rows: Sequence[Mapping[str, Any]], *, label: str) -> Tuple[List[Any], List[str]]:
    values: List[Any] = []
    keys: List[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if "id" not in row:
            raise ReplayValidationError(f"{label}[{index}] is missing id")
        try:
            key = _id_key(row.get("id"))
        except ReplayValidationError as exc:
            raise ReplayValidationError(f"invalid {label}[{index}].id: {exc}") from exc
        if key in seen:
            raise ReplayValidationError(f"duplicate ID in {label}: {row.get('id')!r}")
        seen.add(key)
        values.append(row.get("id"))
        keys.append(key)
    return values, keys


def _as_nonempty_string(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ReplayValidationError(f"{label} must be a non-empty string")
    return text


def _strict_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplayValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _strict_fixed_number(value: Any, *, expected: float, label: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise ReplayValidationError(
            f"{label} must be a JSON number fixed at {expected:g}"
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric != expected:
        raise ReplayValidationError(f"{label} must be fixed at {expected:g}")
    return numeric


def _trace_candidate_source(trace: Mapping[str, Any], *, label: str) -> str:
    explicit = str(trace.get("candidate_source") or "").strip()
    retrieval_mode = str(trace.get("unified_retrieval_mode") or "semantic").strip()
    strategy = str(trace.get("selection_strategy") or "").strip()
    target_marker = (
        explicit == CANDIDATE_SOURCE_TARGET_BINDING
        or retrieval_mode == TARGET_BINDING_RETRIEVAL_MODE
        or strategy == TARGET_BINDING_SELECTION_STRATEGY
    )
    if target_marker:
        if explicit != CANDIDATE_SOURCE_TARGET_BINDING:
            raise ReplayValidationError(
                f"{label}.candidate_source must be "
                f"{CANDIDATE_SOURCE_TARGET_BINDING!r}"
            )
        if retrieval_mode != TARGET_BINDING_RETRIEVAL_MODE:
            raise ReplayValidationError(
                f"{label}.unified_retrieval_mode must be "
                f"{TARGET_BINDING_RETRIEVAL_MODE!r}"
            )
        if strategy != TARGET_BINDING_SELECTION_STRATEGY:
            raise ReplayValidationError(
                f"{label}.selection_strategy must be "
                f"{TARGET_BINDING_SELECTION_STRATEGY!r}"
            )
        return CANDIDATE_SOURCE_TARGET_BINDING
    if explicit not in {"", CANDIDATE_SOURCE_SEMANTIC}:
        raise ReplayValidationError(
            f"{label}.candidate_source={explicit!r} is unsupported"
        )
    if retrieval_mode != "semantic":
        raise ReplayValidationError(
            f"{label}.unified_retrieval_mode must be 'semantic'"
        )
    return CANDIDATE_SOURCE_SEMANTIC


def _candidate_source_for_traces(traces: Sequence[Mapping[str, Any]]) -> str:
    sources = {
        _trace_candidate_source(trace, label=f"frozen retrieval[{index}]")
        for index, trace in enumerate(traces)
    }
    if not sources:
        # Empty datasets have no trace marker; retain the historical semantic
        # interpretation rather than inventing a target binding.
        return CANDIDATE_SOURCE_SEMANTIC
    if len(sources) != 1:
        raise ReplayValidationError(
            "frozen retrieval must not mix semantic and target-binding candidates"
        )
    return next(iter(sources))


def _catalog_index(catalog: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(catalog, dict):
        raise ReplayValidationError("unified catalog must be a JSON object")
    metadata = catalog.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("catalog_type") != "unified_rules_v2":
        raise ReplayValidationError(
            "catalog must declare metadata.catalog_type='unified_rules_v2'"
        )
    domains = catalog.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ReplayValidationError("unified catalog must contain a non-empty domains array")

    rule_index: Dict[str, Dict[str, Any]] = {}
    for domain_pos, domain in enumerate(domains):
        if not isinstance(domain, dict):
            raise ReplayValidationError(f"catalog domains[{domain_pos}] must be an object")
        domain_name = _as_nonempty_string(
            domain.get("name") or domain.get("id"),
            label=f"catalog domains[{domain_pos}].name",
        )
        domain_id = str(domain.get("id") or "").strip()
        topics = domain.get("topics")
        if not isinstance(topics, list):
            raise ReplayValidationError(f"catalog domain {domain_name!r} topics must be an array")
        for topic_pos, topic in enumerate(topics):
            if not isinstance(topic, dict):
                raise ReplayValidationError(
                    f"catalog domain {domain_name!r} topics[{topic_pos}] must be an object"
                )
            topic_name = _as_nonempty_string(
                topic.get("name") or topic.get("id"),
                label=f"catalog topic {domain_name}[{topic_pos}].name",
            )
            topic_id = _as_nonempty_string(
                topic.get("id"),
                label=f"catalog topic {domain_name}/{topic_name}.id",
            )
            topic_rules: Dict[str, Dict[str, Any]] = {}
            for rule_pos, rule in enumerate(topic_rule_leaves(topic)):
                if not isinstance(rule, dict):
                    continue
                rule_id = _as_nonempty_string(
                    rule.get("rule_id") or rule.get("id"),
                    label=f"catalog rule {domain_name}/{topic_name}[{rule_pos}].id",
                )
                if rule_id in topic_rules:
                    raise ReplayValidationError(
                        f"duplicate rule ID within topic {domain_name}/{topic_name}: {rule_id}"
                    )
                if rule_id in rule_index:
                    previous = rule_index[rule_id]
                    raise ReplayValidationError(
                        "rule IDs must be globally unique; "
                        f"{rule_id!r} appears in both "
                        f"{previous['domain_name']}/{previous['topic_name']} and "
                        f"{domain_name}/{topic_name}"
                    )
                topic_rules[rule_id] = rule

            clusters = topic.get("scenario_clusters") or []
            if not isinstance(clusters, list):
                raise ReplayValidationError(
                    f"catalog topic {domain_name}/{topic_name} scenario_clusters must be an array"
                )
            cluster_membership: Dict[str, set[str]] = {rule_id: set() for rule_id in topic_rules}
            cluster_names: Dict[str, str] = {}
            seen_cluster_ids: set[str] = set()
            for cluster_pos, cluster in enumerate(clusters):
                if not isinstance(cluster, dict):
                    raise ReplayValidationError(
                        f"catalog cluster {domain_name}/{topic_name}[{cluster_pos}] must be an object"
                    )
                cluster_id = _as_nonempty_string(
                    cluster.get("id") or cluster.get("cluster_id"),
                    label=f"catalog cluster {domain_name}/{topic_name}[{cluster_pos}].id",
                )
                if cluster_id in seen_cluster_ids:
                    raise ReplayValidationError(
                        f"duplicate cluster ID in {domain_name}/{topic_name}: {cluster_id}"
                    )
                seen_cluster_ids.add(cluster_id)
                cluster_names[cluster_id] = _as_nonempty_string(
                    cluster.get("name") or cluster_id,
                    label=(
                        f"catalog cluster {domain_name}/{topic_name}/{cluster_id}.name"
                    ),
                )
                member_ids = list(cluster.get("rule_ids") or [])
                groups = cluster.get("rule_groups") or []
                if not isinstance(groups, list):
                    raise ReplayValidationError(
                        f"catalog cluster {domain_name}/{topic_name}/{cluster_id} rule_groups must be an array"
                    )
                for group in groups:
                    if not isinstance(group, dict):
                        raise ReplayValidationError(
                            f"catalog cluster {domain_name}/{topic_name}/{cluster_id} has a non-object rule group"
                        )
                    member_ids.extend(group.get("rule_ids") or [])
                for raw_rule_id in member_ids:
                    rule_id = str(raw_rule_id or "").strip()
                    if rule_id not in topic_rules:
                        raise ReplayValidationError(
                            "cluster membership references an unknown/out-of-topic rule: "
                            f"{domain_name}/{topic_name}/{cluster_id}/{rule_id}"
                        )
                    cluster_membership[rule_id].add(cluster_id)

            for rule_id, rule in topic_rules.items():
                rule_index[rule_id] = {
                    "rule": rule,
                    "domain_id": domain_id,
                    "domain_name": domain_name,
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "topic": topic,
                    "cluster_ids": cluster_membership.get(rule_id, set()),
                    "cluster_names": {
                        cluster_id: cluster_names[cluster_id]
                        for cluster_id in cluster_membership.get(rule_id, set())
                    },
                }

    if not rule_index:
        raise ReplayValidationError("unified catalog contains no executable rule leaves")
    return rule_index


def _finite_score(value: Any, *, label: str) -> float:
    try:
        score = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ReplayValidationError(f"{label} must be numeric") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ReplayValidationError(f"{label} must be finite and between 0 and 1")
    return score


def _validate_frozen_trace(
    traces: Sequence[Mapping[str, Any]],
    rule_index: Mapping[str, Mapping[str, Any]],
    *,
    dataset: Optional[Sequence[Mapping[str, Any]]] = None,
) -> None:
    if dataset is not None and len(dataset) != len(traces):
        raise ReplayValidationError(
            "dataset and frozen retrieval must contain the same number of records"
        )
    for sample_pos, trace in enumerate(traces):
        prefix_root = f"frozen retrieval[{sample_pos}]"
        candidate_source = _trace_candidate_source(trace, label=prefix_root)
        target_binding = candidate_source == CANDIDATE_SOURCE_TARGET_BINDING
        strategy = str(trace.get("selection_strategy") or "").strip()
        semantic_strategies = {
            "semantic_tree_selection",
            "semantic_tree_empty",
            "semantic_error",
            "semantic_unavailable",
        }
        if not target_binding and strategy not in semantic_strategies:
            raise ReplayValidationError(
                f"frozen retrieval[{sample_pos}] has unsupported selection_strategy={strategy!r}"
            )
        expected_score_kind = (
            TARGET_BINDING_SCORE_KIND if target_binding else "semantic_0_1"
        )
        score_kind = str(
            trace.get("retrieval_score_kind")
            or ("" if target_binding else "semantic_0_1")
        ).strip()
        if score_kind != expected_score_kind:
            raise ReplayValidationError(
                f"frozen retrieval[{sample_pos}] must use {expected_score_kind} scores"
            )
        retrieved_rules = trace.get("retrieved_rules") or []
        if not isinstance(retrieved_rules, list):
            raise ReplayValidationError(
                f"frozen retrieval[{sample_pos}].retrieved_rules must be an array"
            )
        if strategy == "semantic_tree_selection" and not retrieved_rules:
            raise ReplayValidationError(
                f"frozen retrieval[{sample_pos}] selects a tree but records no rules"
            )
        if strategy == "semantic_tree_empty" and retrieved_rules:
            raise ReplayValidationError(
                f"frozen retrieval[{sample_pos}] is marked empty but records rules"
            )
        if target_binding and len(retrieved_rules) != 1:
            raise ReplayValidationError(
                f"frozen retrieval[{sample_pos}] target binding must contain exactly one rule"
            )

        target_rule_id = ""
        if target_binding:
            if dataset is None:
                raise ReplayValidationError(
                    "target-binding validation requires the paired dataset"
                )
            target_rule_id = _strict_nonempty_string(
                trace.get("target_rule_id"),
                label=f"{prefix_root}.target_rule_id",
            )
            dataset_target = _strict_nonempty_string(
                dataset[sample_pos].get("target_rule_id"),
                label=f"dataset[{sample_pos}].target_rule_id",
            )
            dataset_target_record = dataset[sample_pos].get("target_rule")
            if not isinstance(dataset_target_record, dict):
                raise ReplayValidationError(
                    f"dataset[{sample_pos}].target_rule must be an object"
                )
            nested_target = _strict_nonempty_string(
                dataset_target_record.get("rule_id"),
                label=f"dataset[{sample_pos}].target_rule.rule_id",
            )
            if nested_target != dataset_target:
                raise ReplayValidationError(
                    f"dataset[{sample_pos}].target_rule.rule_id does not match "
                    "target_rule_id"
                )
            if target_rule_id != dataset_target:
                raise ReplayValidationError(
                    f"dataset[{sample_pos}].target_rule_id does not match "
                    f"{prefix_root}.target_rule_id"
                )
            if target_rule_id not in rule_index:
                raise ReplayValidationError(
                    f"{prefix_root}.target_rule_id references unknown catalog rule "
                    f"{target_rule_id!r}"
                )

        retrieved_topics = trace.get("retrieved_topics") or []
        retrieved_clusters = trace.get("retrieved_clusters") or []
        if not isinstance(retrieved_topics, list) or not all(
            isinstance(item, dict) for item in retrieved_topics
        ):
            raise ReplayValidationError(
                f"frozen retrieval[{sample_pos}].retrieved_topics must be an array of objects"
            )
        if not isinstance(retrieved_clusters, list) or not all(
            isinstance(item, dict) for item in retrieved_clusters
        ):
            raise ReplayValidationError(
                f"frozen retrieval[{sample_pos}].retrieved_clusters must be an array of objects"
            )
        if target_binding:
            target_owner = rule_index[target_rule_id]
            retrieved_domains = trace.get("retrieved_domains")
            if not isinstance(retrieved_domains, list) or len(retrieved_domains) != 1:
                raise ReplayValidationError(
                    f"{prefix_root}.retrieved_domains must contain exactly one target owner"
                )
            domain_item = retrieved_domains[0]
            if not isinstance(domain_item, dict):
                raise ReplayValidationError(
                    f"{prefix_root}.retrieved_domains[0] must be an object"
                )
            expected_domain_id = _strict_nonempty_string(
                target_owner.get("domain_id"),
                label=f"catalog owner for {target_rule_id}.domain_id",
            )
            for key, expected in (
                ("domain_id", expected_domain_id),
                ("domain", target_owner.get("domain_name")),
            ):
                actual = _strict_nonempty_string(
                    domain_item.get(key),
                    label=f"{prefix_root}.retrieved_domains[0].{key}",
                )
                if actual != str(expected):
                    raise ReplayValidationError(
                        f"{prefix_root}.retrieved_domains[0].{key} does not match "
                        f"catalog ownership {expected!r}"
                    )
            _strict_fixed_number(
                domain_item.get("score"),
                expected=1.0,
                label=f"{prefix_root}.retrieved_domains[0].score",
            )
            if domain_item.get("score_kind") != TARGET_BINDING_SCORE_KIND:
                raise ReplayValidationError(
                    f"{prefix_root}.retrieved_domains[0].score_kind must be "
                    f"{TARGET_BINDING_SCORE_KIND}"
                )
            expected_cluster_count = 1 if target_owner.get("cluster_ids") else 0
            if len(retrieved_topics) != 1:
                raise ReplayValidationError(
                    f"{prefix_root}.retrieved_topics must contain exactly one target owner"
                )
            if len(retrieved_clusters) != expected_cluster_count:
                raise ReplayValidationError(
                    f"{prefix_root}.retrieved_clusters must contain exactly "
                    f"{expected_cluster_count} target owner"
                )
            top_topic = _strict_nonempty_string(
                trace.get("topic"), label=f"{prefix_root}.topic"
            )
            if top_topic != str(target_owner.get("topic_name") or ""):
                raise ReplayValidationError(
                    f"{prefix_root}.topic does not match target catalog ownership"
                )
        topic_keys: set[Tuple[str, str, str]] = set()
        for topic_pos, topic_item in enumerate(retrieved_topics):
            topic_key = (
                str(topic_item.get("domain") or "").strip(),
                str(topic_item.get("topic_id") or "").strip(),
                str(topic_item.get("topic") or "").strip(),
            )
            if not all(topic_key):
                raise ReplayValidationError(
                    f"frozen retrieval[{sample_pos}].retrieved_topics[{topic_pos}] has incomplete ownership"
                )
            if topic_key in topic_keys:
                raise ReplayValidationError(
                    f"frozen retrieval[{sample_pos}] contains duplicate retrieved topic {topic_key!r}"
                )
            topic_keys.add(topic_key)
            _finite_score(
                topic_item.get("score"),
                label=f"frozen retrieval[{sample_pos}].retrieved_topics[{topic_pos}].score",
            )
            if target_binding:
                if topic_item.get("score_kind") != TARGET_BINDING_SCORE_KIND:
                    raise ReplayValidationError(
                        f"{prefix_root}.retrieved_topics[{topic_pos}].score_kind must be "
                        f"{TARGET_BINDING_SCORE_KIND}"
                    )
                _strict_fixed_number(
                    topic_item.get("score"),
                    expected=1.0,
                    label=f"{prefix_root}.retrieved_topics[{topic_pos}].score",
                )
        cluster_keys: set[Tuple[str, str, str, str]] = set()
        for cluster_pos, cluster_item in enumerate(retrieved_clusters):
            cluster_key = (
                str(cluster_item.get("domain") or "").strip(),
                str(cluster_item.get("topic_id") or "").strip(),
                str(cluster_item.get("topic") or "").strip(),
                str(cluster_item.get("cluster_id") or "").strip(),
            )
            if not all(cluster_key):
                raise ReplayValidationError(
                    f"frozen retrieval[{sample_pos}].retrieved_clusters[{cluster_pos}] has incomplete ownership"
                )
            if cluster_key in cluster_keys:
                raise ReplayValidationError(
                    f"frozen retrieval[{sample_pos}] contains duplicate retrieved cluster {cluster_key!r}"
                )
            cluster_keys.add(cluster_key)
            _finite_score(
                cluster_item.get("score"),
                label=f"frozen retrieval[{sample_pos}].retrieved_clusters[{cluster_pos}].score",
            )
            if target_binding:
                if cluster_item.get("score_kind") != TARGET_BINDING_SCORE_KIND:
                    raise ReplayValidationError(
                        f"{prefix_root}.retrieved_clusters[{cluster_pos}].score_kind must be "
                        f"{TARGET_BINDING_SCORE_KIND}"
                    )
                _strict_fixed_number(
                    cluster_item.get("score"),
                    expected=1.0,
                    label=f"{prefix_root}.retrieved_clusters[{cluster_pos}].score",
                )
                cluster_id = str(cluster_item.get("cluster_id") or "").strip()
                expected_cluster_name = str(
                    (target_owner.get("cluster_names") or {}).get(cluster_id) or ""
                )
                cluster_name = _strict_nonempty_string(
                    cluster_item.get("cluster"),
                    label=f"{prefix_root}.retrieved_clusters[{cluster_pos}].cluster",
                )
                if cluster_name != expected_cluster_name:
                    raise ReplayValidationError(
                        f"{prefix_root}.retrieved_clusters[{cluster_pos}].cluster "
                        f"does not match catalog ownership {expected_cluster_name!r}"
                    )

        seen_rules: set[str] = set()
        for rule_pos, selected in enumerate(retrieved_rules):
            prefix = f"frozen retrieval[{sample_pos}].retrieved_rules[{rule_pos}]"
            if not isinstance(selected, dict):
                raise ReplayValidationError(f"{prefix} must be an object")
            rule_id = _as_nonempty_string(selected.get("rule_id"), label=f"{prefix}.rule_id")
            if rule_id in seen_rules:
                raise ReplayValidationError(
                    f"frozen retrieval[{sample_pos}] contains duplicate rule ID {rule_id!r}"
                )
            seen_rules.add(rule_id)
            if target_binding and rule_id != target_rule_id:
                raise ReplayValidationError(
                    f"{prefix}.rule_id must equal target_rule_id {target_rule_id!r}"
                )
            owner = rule_index.get(rule_id)
            if not isinstance(owner, Mapping):
                raise ReplayValidationError(f"{prefix} references unknown catalog rule {rule_id!r}")

            for key, expected in (
                ("domain", owner.get("domain_name")),
                ("topic_id", owner.get("topic_id")),
                ("topic", owner.get("topic_name")),
            ):
                actual = str(selected.get(key) or "").strip()
                if actual != str(expected or ""):
                    raise ReplayValidationError(
                        f"{prefix}.{key}={actual!r} does not match catalog ownership {expected!r}"
                    )

            cluster_id = str(selected.get("cluster_id") or "").strip()
            owned_clusters = set(owner.get("cluster_ids") or set())
            if cluster_id and cluster_id not in owned_clusters:
                raise ReplayValidationError(
                    f"{prefix}.cluster_id={cluster_id!r} does not own rule {rule_id!r}"
                )
            if owned_clusters and not cluster_id:
                raise ReplayValidationError(
                    f"{prefix} omits cluster ownership for clustered rule {rule_id!r}"
                )
            if target_binding and cluster_id:
                cluster_name = _strict_nonempty_string(
                    selected.get("cluster"), label=f"{prefix}.cluster"
                )
                expected_cluster_name = str(
                    (owner.get("cluster_names") or {}).get(cluster_id) or ""
                )
                if cluster_name != expected_cluster_name:
                    raise ReplayValidationError(
                        f"{prefix}.cluster does not match catalog ownership "
                        f"{expected_cluster_name!r}"
                    )
            owner_topic_key = (
                str(owner.get("domain_name") or ""),
                str(owner.get("topic_id") or ""),
                str(owner.get("topic_name") or ""),
            )
            if owner_topic_key not in topic_keys:
                raise ReplayValidationError(
                    f"{prefix} owner topic is absent from frozen retrieved_topics"
                )
            if cluster_id and (*owner_topic_key, cluster_id) not in cluster_keys:
                raise ReplayValidationError(
                    f"{prefix} owner cluster is absent from frozen retrieved_clusters"
                )

            item_score_kind = str(
                selected.get("score_kind")
                or ("" if target_binding else "semantic_0_1")
            ).strip()
            if item_score_kind != expected_score_kind:
                raise ReplayValidationError(
                    f"{prefix}.score_kind must be {expected_score_kind}"
                )
            if target_binding:
                _strict_fixed_number(
                    selected.get("score"), expected=1.0, label=f"{prefix}.score"
                )
            else:
                _finite_score(selected.get("score"), label=f"{prefix}.score")
            for optional_score in ("semantic_score", "grounding_score"):
                if optional_score in selected:
                    if target_binding:
                        raise ReplayValidationError(
                            f"{prefix}.{optional_score} is not allowed for target binding"
                        )
                    else:
                        _finite_score(
                            selected.get(optional_score),
                            label=f"{prefix}.{optional_score}",
                        )
            publish_gate = selected.get("publish_gate")
            if not isinstance(publish_gate, dict) or not isinstance(
                publish_gate.get("publishable"), bool
            ):
                raise ReplayValidationError(
                    f"{prefix}.publish_gate.publishable must be an explicit boolean"
                )
            gate_reasons = publish_gate.get("reasons") or []
            if not isinstance(gate_reasons, list) or not all(
                isinstance(reason, str) for reason in gate_reasons
            ):
                raise ReplayValidationError(
                    f"{prefix}.publish_gate.reasons must be an array of strings"
                )
            for optional_score in ("score", "semantic_score", "min_publish_score"):
                if optional_score in publish_gate:
                    if target_binding:
                        if optional_score == "semantic_score":
                            raise ReplayValidationError(
                                f"{prefix}.publish_gate.semantic_score is not allowed "
                                "for target binding"
                            )
                        _strict_fixed_number(
                            publish_gate.get(optional_score),
                            expected=0.0 if optional_score == "min_publish_score" else 1.0,
                            label=f"{prefix}.publish_gate.{optional_score}",
                        )
                    else:
                        _finite_score(
                            publish_gate.get(optional_score),
                            label=f"{prefix}.publish_gate.{optional_score}",
                        )
            gate_score_kind = str(
                publish_gate.get("score_kind")
                or ("" if target_binding else "semantic_0_1")
            ).strip()
            if gate_score_kind != expected_score_kind:
                raise ReplayValidationError(
                    f"{prefix}.publish_gate.score_kind must be {expected_score_kind}"
                )
            if target_binding:
                if publish_gate.get("publishable") is not True:
                    raise ReplayValidationError(
                        f"{prefix}.publish_gate.publishable must be exactly true"
                    )
                if publish_gate.get("reasons") != []:
                    raise ReplayValidationError(
                        f"{prefix}.publish_gate.reasons must be empty for target binding"
                    )
                _strict_fixed_number(
                    publish_gate.get("score"),
                    expected=1.0,
                    label=f"{prefix}.publish_gate.score",
                )
                _strict_fixed_number(
                    publish_gate.get("min_publish_score"),
                    expected=0.0,
                    label=f"{prefix}.publish_gate.min_publish_score",
                )
                if publish_gate.get("selection_strategy") != TARGET_BINDING_SELECTION_STRATEGY:
                    raise ReplayValidationError(
                        f"{prefix}.publish_gate.selection_strategy must be "
                        f"{TARGET_BINDING_SELECTION_STRATEGY}"
                    )
                if selected.get("partial") is not False:
                    raise ReplayValidationError(
                        f"{prefix}.partial must be exactly false for target binding"
                    )
                if selected.get("executable") is not True:
                    raise ReplayValidationError(
                        f"{prefix}.executable must be exactly true for target binding"
                    )
            if strategy in {"semantic_error", "semantic_unavailable"}:
                if selected.get("partial") is not True or selected.get("executable") is not False:
                    raise ReplayValidationError(
                        f"{prefix} from semantic_error must be partial and non-executable"
                    )
            elif selected.get("partial") is True or selected.get("executable") is False:
                raise ReplayValidationError(
                    f"{prefix} is non-executable outside a semantic_error trace"
                )


def _selection_projection(
    traces: Sequence[Mapping[str, Any]],
    *,
    candidate_source: str,
    include_candidate_binding: bool,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    target_mapping: List[Dict[str, Any]] = []
    selected_rule_count = 0
    rule_hit_sample_count = 0
    for trace in traces:
        projected_rules: List[Dict[str, Any]] = []
        for item in trace.get("retrieved_rules") or []:
            gate = item.get("publish_gate") if isinstance(item.get("publish_gate"), dict) else {}
            projected_rules.append(
                {
                    "rule_id": item.get("rule_id"),
                    "domain": item.get("domain"),
                    "topic_id": item.get("topic_id"),
                    "topic": item.get("topic"),
                    "cluster_id": item.get("cluster_id"),
                    "score": item.get("score"),
                    "score_kind": item.get("score_kind"),
                    "publishable": gate.get("publishable"),
                    "partial": item.get("partial") is True,
                    "executable": item.get("executable") is not False,
                }
            )
        selected_rule_count += len(projected_rules)
        if projected_rules:
            rule_hit_sample_count += 1
        records.append(
            {
                "id": trace.get("id"),
                "selection_strategy": trace.get("selection_strategy"),
                "semantic_selection_error": trace.get("semantic_selection_error") or "",
                "semantic_failed_stage": trace.get("semantic_failed_stage") or "",
                "empty_reason": trace.get("empty_reason") or "",
                "rules": projected_rules,
            }
        )
        if candidate_source == CANDIDATE_SOURCE_TARGET_BINDING:
            target_mapping.append(
                {
                    "id": trace.get("id"),
                    "target_rule_id": trace.get("target_rule_id"),
                    "selected_rule_id": (
                        projected_rules[0].get("rule_id")
                        if len(projected_rules) == 1
                        else None
                    ),
                }
            )
    projection = {
        "sample_count": len(records),
        "rule_hit_sample_count": rule_hit_sample_count,
        "selected_rule_count": selected_rule_count,
        "records_sha256": _object_sha256(records),
        "records": records,
    }
    if include_candidate_binding:
        projection["candidate_source"] = candidate_source
        projection["target_mapping_count"] = len(target_mapping)
        projection["target_mapping_sha256"] = _object_sha256(target_mapping)
        projection["target_mapping"] = target_mapping
    return projection


def _input_fingerprint(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ids, _ = _ordered_ids(rows, label=path.name)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "record_count": len(rows),
        "ordered_ids_sha256": _object_sha256(ids),
        "ordered_ids": ids,
    }


def _parse_metadata_json(value: str, *, label: str) -> Dict[str, Any]:
    raw = str(value or "").strip()
    if not raw:
        return {}
    if raw.startswith("@"):
        payload = _load_json(Path(raw[1:]), label=label)
    else:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ReplayValidationError(f"{label} must be a JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReplayValidationError(f"{label} must be a JSON object")
    return payload


def prepare_frozen_manifest(
    *,
    dataset_path: Path,
    frozen_retrieval_path: Path,
    catalog_path: Path,
    frozen_manifest_path: Path,
    prompt_metadata: Optional[Mapping[str, Any]] = None,
    retrieval_config_metadata: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    frozen_inputs = {
        dataset_path.resolve(),
        frozen_retrieval_path.resolve(),
        catalog_path.resolve(),
    }
    if frozen_manifest_path.resolve() in frozen_inputs:
        raise ReplayValidationError(
            "frozen manifest path must not overwrite dataset, retrieval trace, or catalog"
        )
    if frozen_manifest_path.exists() and not overwrite:
        raise ReplayValidationError(
            f"frozen manifest already exists (use --overwrite-frozen-manifest explicitly): "
            f"{frozen_manifest_path}"
        )
    dataset = _load_json_array(dataset_path, label="dataset")
    traces = _load_json_array(frozen_retrieval_path, label="frozen retrieval")
    catalog = _load_json(catalog_path, label="unified catalog")
    dataset_ids, dataset_keys = _ordered_ids(dataset, label="dataset")
    trace_ids, trace_keys = _ordered_ids(traces, label="frozen retrieval")
    if dataset_keys != trace_keys:
        raise ReplayValidationError(
            "dataset and frozen retrieval must contain exactly the same IDs in the same order"
        )
    rule_index = _catalog_index(catalog)
    candidate_source = _candidate_source_for_traces(traces)
    _validate_frozen_trace(traces, rule_index, dataset=dataset)
    projection = _selection_projection(
        traces,
        candidate_source=candidate_source,
        include_candidate_binding=True,
    )
    manifest_type = (
        TARGET_BINDING_MANIFEST_TYPE
        if candidate_source == CANDIDATE_SOURCE_TARGET_BINDING
        else FROZEN_MANIFEST_TYPE
    )
    manifest = {
        "schema_version": FROZEN_MANIFEST_SCHEMA_VERSION,
        "manifest_type": manifest_type,
        "candidate_source": candidate_source,
        "created_at_utc": _utc_now(),
        "inputs": {
            "dataset": _input_fingerprint(dataset_path, dataset),
            "retrieval_trace": _input_fingerprint(frozen_retrieval_path, traces),
            "unified_catalog": {
                "path": str(catalog_path),
                "size_bytes": catalog_path.stat().st_size,
                "sha256": sha256_file(catalog_path),
                "catalog_type": (catalog.get("metadata") or {}).get("catalog_type"),
                "rule_count": len(rule_index),
            },
        },
        "ordered_ids": dataset_ids,
        "ordered_ids_sha256": _object_sha256(dataset_ids),
        "selection_projection": projection,
        "prompt_metadata": dict(prompt_metadata or {}),
        "retrieval_config_metadata": dict(retrieval_config_metadata or {}),
    }
    # Keep both ordered-ID sources visible; equality above is an integrity check,
    # not an assumption hidden from the manifest reader.
    manifest["inputs"]["retrieval_trace"]["ordered_ids"] = trace_ids
    manifest["inputs"]["retrieval_trace"]["candidate_source"] = candidate_source
    _atomic_write_json(frozen_manifest_path, manifest)
    return manifest


def _validate_frozen_manifest(
    *,
    manifest: Any,
    manifest_path: Path,
    dataset_path: Path,
    frozen_retrieval_path: Path,
    catalog_path: Path,
    dataset: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    rule_index: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ReplayValidationError("frozen manifest must be a JSON object")
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version not in {
        LEGACY_FROZEN_MANIFEST_SCHEMA_VERSION,
        FROZEN_MANIFEST_SCHEMA_VERSION,
    }:
        raise ReplayValidationError("unsupported frozen manifest schema_version")
    candidate_source = _candidate_source_for_traces(traces)
    expected_manifest_type = (
        TARGET_BINDING_MANIFEST_TYPE
        if candidate_source == CANDIDATE_SOURCE_TARGET_BINDING
        else FROZEN_MANIFEST_TYPE
    )
    if manifest.get("manifest_type") != expected_manifest_type:
        raise ReplayValidationError("frozen manifest has the wrong manifest_type")
    if schema_version == LEGACY_FROZEN_MANIFEST_SCHEMA_VERSION:
        if candidate_source != CANDIDATE_SOURCE_SEMANTIC:
            raise ReplayValidationError(
                "legacy frozen manifests cannot describe target-binding candidates"
            )
        include_candidate_binding = False
    else:
        include_candidate_binding = True
        if manifest.get("candidate_source") != candidate_source:
            raise ReplayValidationError(
                "frozen manifest candidate_source does not match the candidate trace"
            )
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ReplayValidationError("frozen manifest.inputs must be an object")
    paths = {
        "dataset": dataset_path,
        "retrieval_trace": frozen_retrieval_path,
        "unified_catalog": catalog_path,
    }
    for name, path in paths.items():
        entry = inputs.get(name)
        if not isinstance(entry, dict):
            raise ReplayValidationError(f"frozen manifest is missing inputs.{name}")
        expected = str(entry.get("sha256") or "").lower()
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise ReplayValidationError(f"frozen manifest inputs.{name}.sha256 is invalid")
        actual = sha256_file(path)
        if actual != expected:
            raise ReplayValidationError(
                f"SHA256 mismatch for {name}: manifest={expected}, actual={actual}"
            )

    dataset_ids, dataset_keys = _ordered_ids(dataset, label="dataset")
    trace_ids, trace_keys = _ordered_ids(traces, label="frozen retrieval")
    if dataset_keys != trace_keys:
        raise ReplayValidationError(
            "dataset and frozen retrieval IDs differ or are in a different order"
        )
    if manifest.get("ordered_ids") != dataset_ids:
        raise ReplayValidationError("frozen manifest ordered_ids do not match the dataset")
    if manifest.get("ordered_ids_sha256") != _object_sha256(dataset_ids):
        raise ReplayValidationError("frozen manifest ordered_ids_sha256 does not match")
    for name, ids in (("dataset", dataset_ids), ("retrieval_trace", trace_ids)):
        entry = inputs[name]
        if entry.get("record_count") != len(ids):
            raise ReplayValidationError(f"frozen manifest inputs.{name}.record_count does not match")
        if entry.get("ordered_ids") != ids:
            raise ReplayValidationError(f"frozen manifest inputs.{name}.ordered_ids do not match")
        if entry.get("ordered_ids_sha256") != _object_sha256(ids):
            raise ReplayValidationError(
                f"frozen manifest inputs.{name}.ordered_ids_sha256 does not match"
            )

    _validate_frozen_trace(traces, rule_index, dataset=dataset)
    projection = _selection_projection(
        traces,
        candidate_source=candidate_source,
        include_candidate_binding=include_candidate_binding,
    )
    if manifest.get("selection_projection") != projection:
        raise ReplayValidationError(
            "frozen manifest selection_projection does not match the retrieval trace"
        )
    if not isinstance(manifest.get("prompt_metadata"), dict):
        raise ReplayValidationError("frozen manifest prompt_metadata must be an object")
    if not isinstance(manifest.get("retrieval_config_metadata"), dict):
        raise ReplayValidationError(
            "frozen manifest retrieval_config_metadata must be an object"
        )
    catalog_type = ((catalog.get("metadata") or {}).get("catalog_type"))
    if inputs["unified_catalog"].get("catalog_type") != catalog_type:
        raise ReplayValidationError("frozen manifest unified catalog type does not match")
    if inputs["unified_catalog"].get("rule_count") != len(rule_index):
        raise ReplayValidationError("frozen manifest unified catalog rule_count does not match")
    if include_candidate_binding and inputs["retrieval_trace"].get(
        "candidate_source"
    ) != candidate_source:
        raise ReplayValidationError(
            "frozen manifest retrieval trace candidate_source does not match"
        )
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "selection_projection_sha256": projection["records_sha256"],
        "candidate_source": candidate_source,
        "target_mapping_sha256": projection.get(
            "target_mapping_sha256", _object_sha256([])
        ),
    }


def _prepare_catalog_rule(rule: Mapping[str, Any]) -> Dict[str, Any]:
    prepared = dict(rule)
    rule_id = str(rule.get("rule_id") or rule.get("id") or "").strip()
    prepared["id"] = rule_id
    if not prepared.get("description"):
        parts: List[str] = []
        if rule.get("trigger"):
            parts.append(f"Trigger: {rule.get('trigger')}")
        if rule.get("check_logic"):
            parts.append(f"Check Logic: {rule.get('check_logic')}")
        prepared["description"] = "\n".join(parts)
    return prepared


def _topic_rank(trace: Mapping[str, Any], domain: str, topic_id: str, topic_name: str) -> Tuple[int, float]:
    topics = trace.get("retrieved_topics") or []
    if not isinstance(topics, list):
        return 0, 0.0
    top_score = 0.0
    if topics and isinstance(topics[0], dict):
        try:
            top_score = float(topics[0].get("score") or 0.0)
        except (TypeError, ValueError):
            top_score = 0.0
    for rank, item in enumerate(topics):
        if not isinstance(item, dict):
            continue
        if (
            str(item.get("domain") or "") == domain
            and str(item.get("topic_id") or "") == topic_id
            and str(item.get("topic") or "") == topic_name
        ):
            try:
                score = float(item.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            return rank, max(0.0, top_score - score)
    return 0, 0.0


def _rebuild_rule_records(
    trace: Mapping[str, Any],
    rule_index: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_records: List[Dict[str, Any]] = []
    executable_records: List[Dict[str, Any]] = []
    precheck_suppressed: List[Dict[str, Any]] = []
    for selected in trace.get("retrieved_rules") or []:
        rule_id = str(selected.get("rule_id") or "")
        owner = rule_index[rule_id]
        prepared_rule = _prepare_catalog_rule(owner["rule"])
        rank, topic_gap = _topic_rank(
            trace,
            str(owner["domain_name"]),
            str(owner["topic_id"]),
            str(owner["topic_name"]),
        )
        publish_gate = copy.deepcopy(selected.get("publish_gate") or {})
        score = float(selected.get("score") or 0.0)
        score_kind = str(selected.get("score_kind") or "semantic_0_1")
        retrieval_strategy = str(trace.get("selection_strategy") or "")
        target_binding = retrieval_strategy == TARGET_BINDING_SELECTION_STRATEGY
        record = {
            "domain": str(owner["domain_name"]),
            "topic_id": str(owner["topic_id"]),
            "topic_name": str(owner["topic_name"]),
            "topic": owner["topic"],
            "topic_rank": rank,
            "cluster_id": str(selected.get("cluster_id") or ""),
            "cluster": str(selected.get("cluster") or ""),
            "rule": prepared_rule,
            "score": score,
            "score_kind": score_kind,
            "semantic_score": (
                None
                if target_binding
                else float(selected.get("semantic_score") or score)
            ),
            "grounding_score": (
                None
                if target_binding
                else float(selected.get("grounding_score") or 0.0)
            ),
            "adjusted_score": score,
            "topic_gap": topic_gap,
            "min_score": float(publish_gate.get("min_publish_score") or 0.0),
            "scope": str(selected.get("scope") or prepared_rule.get("scope") or "domain"),
            "evidence": copy.deepcopy(selected.get("evidence") or {}),
            "publish_gate": publish_gate,
            "manual_override_reason": "",
            "retrieval_strategy": retrieval_strategy,
        }
        all_records.append(record)
        executable = (
            retrieval_strategy
            in {"semantic_tree_selection", TARGET_BINDING_SELECTION_STRATEGY}
            and selected.get("partial") is not True
            and selected.get("executable") is not False
            and publish_gate.get("publishable") is True
        )
        if executable:
            executable_records.append(record)
        elif retrieval_strategy in {
            "semantic_tree_selection",
            TARGET_BINDING_SELECTION_STRATEGY,
        }:
            precheck_suppressed.append(
                {
                    "reason": "rule_publish_gate_precheck",
                    "rule_id": rule_id,
                    "publish_gate": publish_gate,
                }
            )
    return all_records, executable_records, precheck_suppressed


def _checker_sample(sample: Mapping[str, Any]) -> Dict[str, Any]:
    # Evaluation labels and reference answers are intentionally excluded.  The
    # Checker implementation only consumes these four fields.
    return {
        key: sample.get(key)
        for key in ("id", "question", "context", "prediction")
        if key in sample
    }


def _safe_error(exc: BaseException) -> str:
    text = str(exc)
    for env_name in _SENSITIVE_ENV_NAMES:
        secret = os.getenv(env_name)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text[:1000]


def _source_identity() -> Dict[str, Any]:
    state = capture_source_state(PROJECT_ROOT)
    git = state.get("git") if isinstance(state.get("git"), dict) else {}
    source_tree = (
        state.get("source_tree") if isinstance(state.get("source_tree"), dict) else {}
    )
    return {
        "git_available": git.get("available") is True,
        "git_head": str(git.get("head") or ""),
        "git_branch": str(git.get("branch") or ""),
        "git_dirty": git.get("dirty") is True,
        "git_status_sha256": str(git.get("status_sha256") or ""),
        "git_tracked_diff_sha256": str(git.get("tracked_diff_sha256") or ""),
        "source_tree_sha256": str(source_tree.get("sha256") or ""),
        "source_file_count": int(source_tree.get("file_count") or 0),
    }


def _runtime_identity() -> Dict[str, Any]:
    runtime = probe_python_runtime(sys.executable, require_conda=True)
    return {
        "executable": str(runtime.get("executable") or ""),
        "python_version": str(runtime.get("python_version") or ""),
        "prefix": str(runtime.get("prefix") or ""),
        "is_conda": runtime.get("is_conda") is True,
        "conda_env": str(runtime.get("conda_env") or ""),
        "package_count": int(runtime.get("package_count") or 0),
        "package_set_sha256": str(runtime.get("package_set_sha256") or ""),
    }


def _api_transport_identity() -> Dict[str, Any]:
    endpoint_source = "default"
    endpoint = "https://api.openai.com/v1"
    for env_name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        raw = str(os.getenv(env_name) or "").strip()
        if raw:
            endpoint_source = env_name
            endpoint = raw.rstrip("/")
            break
    return {
        "endpoint_source": endpoint_source,
        "endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        "disable_thinking": str(os.getenv("OPENAI_DISABLE_THINKING") or "").strip().lower()
        in {"1", "true", "yes", "on"},
        "timeout_sec": str(os.getenv("PHYSICSVERIFIER_LLM_TIMEOUT_SEC") or "").strip(),
        "sdk_max_retries": str(os.getenv("PHYSICSVERIFIER_LLM_MAX_RETRIES") or "").strip(),
    }


class _JsonlTraceAudit:
    """Incrementally fingerprint the Checker JSONL trace without rescanning it."""

    def __init__(self, path: Path, *, reset: bool) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.write_bytes(b"")
        elif not self.path.exists():
            self.path.write_bytes(b"")
        self._hasher = hashlib.sha256()
        self._offset = 0
        self._record_count = 0
        self._raw_response_records = 0
        self._prompt_records = 0
        self._parse_status_counts: Counter[str] = Counter()
        self.refresh()

    def refresh(self) -> None:
        current_size = self.path.stat().st_size
        if current_size < self._offset:
            raise ReplayValidationError("Checker LLM trace was truncated during replay")
        if current_size == self._offset:
            return
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
        if not chunk.endswith(b"\n"):
            raise ReplayValidationError("Checker LLM trace ends with an incomplete JSONL record")
        self._hasher.update(chunk)
        for line_number, raw_line in enumerate(chunk.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ReplayValidationError(
                    f"Checker LLM trace contains invalid JSONL near appended line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ReplayValidationError("Checker LLM trace records must be JSON objects")
            parse_status = str(record.get("parse_status") or "").strip()
            if not parse_status:
                raise ReplayValidationError(
                    "Checker LLM trace record is missing parse_status"
                )
            self._record_count += 1
            self._parse_status_counts[parse_status] += 1
            if "raw_response" in record:
                self._raw_response_records += 1
            if "system_prompt" in record or "user_prompt" in record:
                self._prompt_records += 1
                raise ReplayValidationError(
                    "Checker LLM trace must not contain prompts during controlled replay"
                )
        self._offset = current_size

    def fingerprint(self) -> Dict[str, Any]:
        self.refresh()
        return {
            "path": str(self.path),
            "sha256": self._hasher.hexdigest(),
            "size_bytes": self._offset,
            "record_count": self._record_count,
            "raw_response_record_count": self._raw_response_records,
            "prompt_record_count": self._prompt_records,
            "prompts_included": self._prompt_records > 0,
            "parse_status_counts": dict(sorted(self._parse_status_counts.items())),
        }


def _checker_status_succeeded(status: str, failures: Sequence[Any]) -> bool:
    if failures:
        return False
    normalized = str(status or "").strip().lower()
    if normalized in _SUCCESS_CHECKER_STATUSES:
        return True
    return normalized.startswith("complete") and not any(
        token in normalized for token in ("fail", "error", "partial")
    )


def _validate_checker_result(
    result: Any,
    *,
    mode: str,
    rule_ids: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Any], List[Any], List[Any], str]:
    if not isinstance(result, dict):
        raise ReplayValidationError("checker analyze() must return a JSON object")
    checker_mode = str(result.get("checker_mode") or "").strip().lower()
    if checker_mode != mode:
        raise ReplayValidationError(
            f"checker returned mode {checker_mode!r}; expected {mode!r}"
        )
    checker_status = str(result.get("checker_status") or "").strip()
    if not checker_status:
        raise ReplayValidationError("checker result is missing checker_status")
    fields: Dict[str, List[Any]] = {}
    for key in ("diagnostics", "checker_decisions", "checker_failures", "checker_suppressed"):
        value = result.get(key) or []
        if not isinstance(value, list):
            raise ReplayValidationError(f"checker result {key} must be an array")
        fields[key] = value
    allowed = set(rule_ids)
    diagnostics: List[Dict[str, Any]] = []
    for index, diagnostic in enumerate(fields["diagnostics"]):
        if not isinstance(diagnostic, dict):
            raise ReplayValidationError(f"checker diagnostic[{index}] must be an object")
        rule_id = str(diagnostic.get("rule") or "").strip()
        if rule_id not in allowed:
            raise ReplayValidationError(
                f"checker diagnostic[{index}] references non-frozen rule {rule_id!r}"
            )
        diagnostics.append(diagnostic)
    for field_name in ("checker_decisions", "checker_failures"):
        for index, item in enumerate(fields[field_name]):
            if not isinstance(item, dict):
                raise ReplayValidationError(
                    f"checker result {field_name}[{index}] must be an object"
                )
            referenced_rule = str(item.get("rule_id") or item.get("rule") or "").strip()
            if referenced_rule and referenced_rule not in allowed:
                raise ReplayValidationError(
                    f"checker result {field_name}[{index}] references non-frozen rule "
                    f"{referenced_rule!r}"
                )
    decision_rule_ids = [
        str(item.get("rule_id") or item.get("rule") or "").strip()
        for item in fields["checker_decisions"]
    ]
    if decision_rule_ids != list(rule_ids):
        raise ReplayValidationError(
            "checker decisions must cover every frozen rule exactly once and in order"
        )
    for index, item in enumerate(fields["checker_suppressed"]):
        if not isinstance(item, dict):
            raise ReplayValidationError(
                f"checker result checker_suppressed[{index}] must be an object"
            )
        referenced_rule = str(item.get("rule_id") or item.get("rule") or "").strip()
        if referenced_rule and referenced_rule not in allowed:
            raise ReplayValidationError(
                f"checker result checker_suppressed[{index}] references non-frozen rule "
                f"{referenced_rule!r}"
            )
    return (
        diagnostics,
        fields["checker_decisions"],
        fields["checker_failures"],
        fields["checker_suppressed"],
        checker_status,
    )


def _base_output_row(
    sample: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    mode: str,
    config_sha256: str,
    candidate_source: str,
) -> Dict[str, Any]:
    target_binding = candidate_source == CANDIDATE_SOURCE_TARGET_BINDING
    return {
        "id": sample.get("id"),
        "topic": trace.get("topic"),
        "verifier": (
            "unified_v2_frozen_target_binding_checker_replay"
            if target_binding
            else "unified_v2_frozen_retrieval_checker_replay"
        ),
        "unified_mode": True,
        "candidate_source": candidate_source,
        "target_rule_id": trace.get("target_rule_id") if target_binding else None,
        "unified_retrieval_mode": (
            TARGET_BINDING_RETRIEVAL_MODE
            if target_binding
            else "frozen_semantic_trace"
        ),
        "selection_strategy": trace.get("selection_strategy"),
        "retrieval_score_kind": trace.get("retrieval_score_kind"),
        "semantic_selection_error": trace.get("semantic_selection_error") or "",
        "semantic_failed_stage": trace.get("semantic_failed_stage") or "",
        "terminal_stage": trace.get("terminal_stage") or "",
        "empty_reason": trace.get("empty_reason") or "",
        "retrieved_domains": copy.deepcopy(trace.get("retrieved_domains") or []),
        "retrieved_topics": copy.deepcopy(trace.get("retrieved_topics") or []),
        "retrieved_clusters": copy.deepcopy(trace.get("retrieved_clusters") or []),
        "retrieved_rules": copy.deepcopy(trace.get("retrieved_rules") or []),
        "checker_gate_mode": mode,
        "checker_min_confidence": FROZEN_CHECKER_MIN_CONFIDENCE,
        "checker_status": "not_run",
        "checker_failure_count": 0,
        "checker_decisions": [],
        "checker_failures": [],
        "checker_suppressed_diagnostics": [],
        "candidate_diagnostics": [],
        "diagnostics": [],
        "symbolic_post_diagnostics": [],
        "experience_post_diagnostics": [],
        "experience_code_post_diagnostics": [],
        "experience_symbolic_post_diagnostics": [],
        "symbolic_check": {
            "enabled": False,
            "actions": [],
            "suppressed_diagnostics": [],
        },
        "agentic": {"enabled": False, "actions": [], "suppressed_diagnostics": []},
        "experience_pipeline": {"enabled": False, "actions": []},
        "score": 0.0,
        "replay_config_sha256": config_sha256,
        "replay_completed": False,
        "replay": {
            "system_arm": mode,
            "checker_attempted": False,
            "checker_succeeded": False,
            "bottom_up_enabled": False,
        },
    }


def _failure_kind(failure: Any) -> str:
    if not isinstance(failure, dict):
        return "checker_failure"
    for key in ("kind", "failure_kind", "reason", "error_type", "stage", "status"):
        value = str(failure.get(key) or "").strip()
        if value:
            return value[:120]
    return "checker_failure"


def _summarize_rows(rows: Sequence[Mapping[str, Any]], *, total_samples: int) -> Dict[str, Any]:
    failure_counts: Counter[str] = Counter()
    suppression_counts: Counter[str] = Counter()
    checker_attempted = 0
    checker_success = 0
    no_rule_samples = 0
    rule_hit_samples = 0
    completed = 0
    diagnostics = 0
    diagnostic_samples = 0
    failed_samples = 0
    terminal_failure_samples = 0
    for row in rows:
        replay = row.get("replay") if isinstance(row.get("replay"), dict) else {}
        if row.get("replay_completed") is True:
            completed += 1
        if replay.get("checker_attempted") is True:
            checker_attempted += 1
        if replay.get("checker_succeeded") is True:
            checker_success += 1
        if replay.get("no_executable_rules") is True:
            no_rule_samples += 1
        if row.get("retrieved_rules"):
            rule_hit_samples += 1
        row_diagnostics = row.get("diagnostics") or []
        if isinstance(row_diagnostics, list):
            diagnostics += len(row_diagnostics)
            if row_diagnostics:
                diagnostic_samples += 1
        failures = row.get("checker_failures") or []
        row_has_failure = row.get("replay_completed") is not True
        if isinstance(failures, list):
            if failures:
                row_has_failure = True
            for failure in failures:
                failure_counts[_failure_kind(failure)] += 1
        if row_has_failure:
            failed_samples += 1
            if row.get("replay_completed") is True:
                terminal_failure_samples += 1
        symbolic = row.get("symbolic_check") if isinstance(row.get("symbolic_check"), dict) else {}
        suppressed = symbolic.get("suppressed_diagnostics") or []
        if isinstance(suppressed, list):
            for item in suppressed:
                if isinstance(item, dict):
                    suppression_counts[str(item.get("reason") or "unknown")] += 1

    processed = len(rows)
    retryable_failures = sum(row.get("replay_completed") is not True for row in rows)
    return {
        "total_samples": total_samples,
        "processed_samples": processed,
        "completed_samples": completed,
        "pending_samples": max(0, total_samples - processed),
        "retryable_failure_samples": retryable_failures,
        "failed_samples": failed_samples,
        "terminal_failure_samples": terminal_failure_samples,
        "retrieval_rule_hit_samples": rule_hit_samples,
        "no_executable_rule_samples": no_rule_samples,
        "checker_attempted_samples": checker_attempted,
        "checker_success_samples": checker_success,
        "checker_failure_samples": max(0, checker_attempted - checker_success),
        "diagnostic_samples": diagnostic_samples,
        "diagnostic_count": diagnostics,
        "processed_coverage": round(processed / total_samples, 6) if total_samples else 1.0,
        "completion_coverage": round(completed / total_samples, 6) if total_samples else 1.0,
        "retrieval_rule_coverage": round(rule_hit_samples / total_samples, 6) if total_samples else 0.0,
        "checker_coverage": (
            round(checker_success / checker_attempted, 6) if checker_attempted else 1.0
        ),
        "end_to_end_checker_coverage": (
            round(checker_success / total_samples, 6) if total_samples else 1.0
        ),
        "failure_counts": dict(sorted(failure_counts.items())),
        "suppression_counts": dict(sorted(suppression_counts.items())),
    }


def _report_payload(
    *,
    rows: Sequence[Mapping[str, Any]],
    total_samples: int,
    output_path: Path,
    artifacts: Mapping[str, Any],
    frozen_manifest_info: Mapping[str, Any],
    configuration: Mapping[str, Any],
    configuration_sha256: str,
    resume_requested: bool,
    resumed_completed_samples: int,
    orphan_trace_recovery: Mapping[str, Any],
    llm_trace: Mapping[str, Any],
) -> Dict[str, Any]:
    statistics = _summarize_rows(rows, total_samples=total_samples)
    finished = statistics["processed_samples"] == total_samples
    status = "running"
    if finished:
        if statistics["failed_samples"] == 0:
            status = "complete"
        elif statistics["retryable_failure_samples"]:
            status = "incomplete_failures"
        else:
            status = "complete_with_failures"
    candidate_source = str(
        configuration.get("candidate_source") or CANDIDATE_SOURCE_SEMANTIC
    )
    return {
        "schema_version": REPLAY_REPORT_SCHEMA_VERSION,
        "report_type": (
            "checker_only_frozen_target_binding_replay"
            if candidate_source == CANDIDATE_SOURCE_TARGET_BINDING
            else "checker_only_frozen_retrieval_replay"
        ),
        "candidate_source": candidate_source,
        "target_mapping_sha256": configuration.get("target_mapping_sha256"),
        "updated_at_utc": _utc_now(),
        "status": status,
        "system_arm_interpretation": (
            "checker_and_release_gate_system_arm; not a shared-candidate gate-only projection"
        ),
        "configuration_sha256": configuration_sha256,
        "configuration": dict(configuration),
        "inputs": dict(artifacts),
        "frozen_manifest": dict(frozen_manifest_info),
        "llm_trace": dict(llm_trace),
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path) if output_path.is_file() else "",
            "record_count": len(rows),
        },
        "resume": {
            "enabled": bool(resume_requested),
            "reused_completed_samples": resumed_completed_samples,
            "orphan_trace_recovery": dict(orphan_trace_recovery),
        },
        "statistics": statistics,
    }


def _load_resume_rows(
    output_path: Path,
    *,
    input_keys: Sequence[str],
    configuration_sha256: str,
) -> Dict[str, Dict[str, Any]]:
    if not output_path.exists():
        return {}
    rows = _load_json_array(output_path, label="replay checkpoint")
    allowed = set(input_keys)
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(rows):
        try:
            key = _id_key(row.get("id"))
        except ReplayValidationError as exc:
            raise ReplayValidationError(f"invalid replay checkpoint[{index}].id: {exc}") from exc
        if key not in allowed:
            raise ReplayValidationError(
                f"replay checkpoint contains ID absent from dataset: {row.get('id')!r}"
            )
        if key in by_id:
            raise ReplayValidationError(
                f"replay checkpoint contains duplicate ID: {row.get('id')!r}"
            )
        if row.get("replay_config_sha256") != configuration_sha256:
            raise ReplayValidationError(
                "replay checkpoint configuration hash does not match this run"
            )
        by_id[key] = row
    return by_id


def _default_verifier_factory(**kwargs: Any) -> PhysicsRuleVerifier:
    return PhysicsRuleVerifier(**kwargs)


def run_checker_replay(
    *,
    dataset_path: Path,
    frozen_retrieval_path: Path,
    catalog_path: Path,
    frozen_manifest_path: Path,
    output_path: Path,
    report_path: Path,
    model: str,
    mode: str = "legacy",
    checker_json_attempts: int = 1,
    enable_cache: bool = False,
    precision_mode: str = "strict",
    min_diagnostic_rule_score: Optional[float] = None,
    max_diagnostics_per_sample: int = 12,
    max_diagnostics_per_paragraph: int = 2,
    quote_required_symbol_ratio: float = 0.0,
    llm_temperature: float = 0.1,
    llm_max_output_tokens: int = 2048,
    checkpoint_every: int = 1,
    resume: bool = False,
    llm_trace_path: Optional[Path] = None,
    run_kind: str = "development",
    verifier_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    if mode not in CHECKER_GATE_MODES:
        raise ReplayValidationError(f"unsupported Checker mode: {mode}")
    if not 1 <= checker_json_attempts <= 5:
        raise ReplayValidationError("checker_json_attempts must be between 1 and 5")
    if checkpoint_every < 1:
        raise ReplayValidationError("checkpoint_every must be at least 1")
    if not str(model or "").strip():
        raise ReplayValidationError("model must be non-empty")
    if run_kind not in _RUN_KINDS:
        raise ReplayValidationError(
            "run_kind must be one of: " + ", ".join(_RUN_KINDS)
        )
    if run_kind in {"validation", "final"} and enable_cache:
        raise ReplayValidationError(
            f"{run_kind} replay forbids Checker cache; raw responses must be auditable"
        )
    if not math.isclose(
        float(getattr(PhysicsRuleVerifier, "CHECKER_MIN_CONFIDENCE", -1.0)),
        FROZEN_CHECKER_MIN_CONFIDENCE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ReplayValidationError(
            "PhysicsRuleVerifier CHECKER_MIN_CONFIDENCE drifted from the frozen replay value 0.8"
        )
    resolved_llm_trace_path = llm_trace_path or output_path.with_name(
        f"{output_path.name}.llm_trace.jsonl"
    )
    protected = {
        dataset_path.resolve(),
        frozen_retrieval_path.resolve(),
        catalog_path.resolve(),
        frozen_manifest_path.resolve(),
    }
    writable_paths = {
        output_path.resolve(),
        report_path.resolve(),
        resolved_llm_trace_path.resolve(),
    }
    if writable_paths & protected:
        raise ReplayValidationError(
            "output/report/LLM-trace paths must not overwrite frozen inputs"
        )
    if len(writable_paths) != 3:
        raise ReplayValidationError(
            "output, report, and Checker LLM-trace paths must be different"
        )

    dataset = _load_json_array(dataset_path, label="dataset")
    traces = _load_json_array(frozen_retrieval_path, label="frozen retrieval")
    catalog = _load_json(catalog_path, label="unified catalog")
    manifest = _load_json(frozen_manifest_path, label="frozen manifest")
    dataset_ids, dataset_keys = _ordered_ids(dataset, label="dataset")
    _, trace_keys = _ordered_ids(traces, label="frozen retrieval")
    if dataset_keys != trace_keys:
        raise ReplayValidationError(
            "dataset and frozen retrieval must contain exactly the same IDs in the same order"
        )
    rule_index = _catalog_index(catalog)
    frozen_manifest_info = _validate_frozen_manifest(
        manifest=manifest,
        manifest_path=frozen_manifest_path,
        dataset_path=dataset_path,
        frozen_retrieval_path=frozen_retrieval_path,
        catalog_path=catalog_path,
        dataset=dataset,
        traces=traces,
        catalog=catalog,
        rule_index=rule_index,
    )
    candidate_source = str(frozen_manifest_info["candidate_source"])
    source_identity = _source_identity()
    if run_kind in {"validation", "final"}:
        if source_identity["git_available"] is not True:
            raise ReplayValidationError(
                f"{run_kind} replay requires an available Git repository"
            )
        if source_identity["git_dirty"] is True:
            raise ReplayValidationError(
                f"{run_kind} replay requires a clean Git worktree"
            )
    runtime_identity = _runtime_identity()
    # The Checker loads `.env` during construction. Load it before recording
    # the endpoint/transport fingerprint so the report binds the endpoint that
    # will actually serve this replay without storing its raw URL or secrets.
    _load_env_file_fallback()
    api_transport_identity = _api_transport_identity()

    artifacts = {
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "record_count": len(dataset),
            "ordered_ids_sha256": _object_sha256(dataset_ids),
        },
        "retrieval_trace": {
            "path": str(frozen_retrieval_path),
            "sha256": sha256_file(frozen_retrieval_path),
            "record_count": len(traces),
            "ordered_ids_sha256": _object_sha256(dataset_ids),
            "candidate_source": candidate_source,
        },
        "unified_catalog": {
            "path": str(catalog_path),
            "sha256": sha256_file(catalog_path),
            "rule_count": len(rule_index),
        },
    }
    configuration = {
        "run_kind": run_kind,
        "system_arm": mode,
        "checker_model": str(model),
        "checker_prompt_version": SEMANTIC_RULE_CHECKER_PROMPT_VERSION,
        "checker_json_attempts": int(checker_json_attempts),
        "checker_min_confidence": FROZEN_CHECKER_MIN_CONFIDENCE,
        "checker_cache_enabled": bool(enable_cache),
        "llm_temperature": float(llm_temperature),
        "llm_max_output_tokens": int(llm_max_output_tokens),
        "precision_mode": str(precision_mode),
        "min_diagnostic_rule_score": min_diagnostic_rule_score,
        "max_diagnostics_per_sample": int(max_diagnostics_per_sample),
        "max_diagnostics_per_paragraph": int(max_diagnostics_per_paragraph),
        "quote_required_symbol_ratio": float(quote_required_symbol_ratio),
        "candidate_source": candidate_source,
        "target_mapping_sha256": frozen_manifest_info["target_mapping_sha256"],
        "retrieval_source": (
            "frozen_target_binding"
            if candidate_source == CANDIDATE_SOURCE_TARGET_BINDING
            else "frozen_semantic_trace"
        ),
        "retrieval_execution": False,
        "bottom_up_enabled": False,
        "llm_trace_path": str(resolved_llm_trace_path),
        "llm_trace_include_prompts": False,
        "dataset_sha256": artifacts["dataset"]["sha256"],
        "retrieval_trace_sha256": artifacts["retrieval_trace"]["sha256"],
        "unified_catalog_sha256": artifacts["unified_catalog"]["sha256"],
        "selection_projection_sha256": frozen_manifest_info[
            "selection_projection_sha256"
        ],
        "frozen_manifest_sha256": frozen_manifest_info["sha256"],
        "source_identity": source_identity,
        "runtime_identity": runtime_identity,
        "api_transport_identity": api_transport_identity,
    }
    configuration_sha256 = _object_sha256(configuration)

    existing: Dict[str, Dict[str, Any]] = {}
    previous_report: Optional[Dict[str, Any]] = None
    orphan_trace_recovery: Dict[str, Any] = {
        "detected": False,
        "extra_size_bytes": 0,
        "extra_record_count": 0,
        "extra_raw_response_record_count": 0,
    }
    checkpoint_exists = output_path.exists()
    if resume and checkpoint_exists:
        if not report_path.is_file():
            raise ReplayValidationError(
                "resume requires the existing replay sidecar report"
            )
        existing = _load_resume_rows(
            output_path,
            input_keys=dataset_keys,
            configuration_sha256=configuration_sha256,
        )
        loaded_report = _load_json(report_path, label="replay sidecar report")
        if not isinstance(loaded_report, dict) or loaded_report.get(
            "configuration_sha256"
        ) != configuration_sha256:
            raise ReplayValidationError(
                "replay sidecar report configuration does not match this run"
            )
        previous_report = loaded_report
        previous_output = previous_report.get("output")
        if not isinstance(previous_output, dict):
            raise ReplayValidationError("resume sidecar is missing output metadata")
        if previous_output.get("sha256") != sha256_file(output_path):
            raise ReplayValidationError(
                "replay checkpoint SHA256 does not match the resume sidecar"
            )
        if previous_output.get("record_count") != len(existing):
            raise ReplayValidationError(
                "replay checkpoint record_count does not match the resume sidecar"
            )
    elif resume and report_path.exists():
        raise ReplayValidationError(
            "resume sidecar exists but the replay checkpoint is missing"
        )

    trace_audit = _JsonlTraceAudit(
        resolved_llm_trace_path,
        reset=not (resume and checkpoint_exists),
    )
    if previous_report is not None:
        previous_trace = previous_report.get("llm_trace")
        current_trace = trace_audit.fingerprint()
        if not isinstance(previous_trace, dict):
            raise ReplayValidationError("resume sidecar is missing llm_trace metadata")
        if previous_trace.get("path") != current_trace.get("path"):
            raise ReplayValidationError(
                "Checker LLM trace path does not match the resume sidecar"
            )
        try:
            previous_size = int(previous_trace.get("size_bytes"))
            previous_records = int(previous_trace.get("record_count"))
            previous_raw_records = int(
                previous_trace.get("raw_response_record_count") or 0
            )
        except (TypeError, ValueError) as exc:
            raise ReplayValidationError(
                "resume sidecar has invalid Checker LLM trace counters"
            ) from exc
        if int(current_trace["size_bytes"]) < previous_size:
            raise ReplayValidationError(
                "Checker LLM trace was truncated after the resume sidecar checkpoint"
            )
        if _sha256_file_prefix(resolved_llm_trace_path, previous_size) != str(
            previous_trace.get("sha256") or ""
        ):
            raise ReplayValidationError(
                "Checker LLM trace checkpoint prefix does not match the resume sidecar"
            )
        if int(current_trace["record_count"]) < previous_records:
            raise ReplayValidationError(
                "Checker LLM trace record_count is below the resume sidecar"
            )
        orphan_trace_recovery = {
            "detected": int(current_trace["size_bytes"]) > previous_size,
            "checkpoint_prefix_sha256": str(previous_trace.get("sha256") or ""),
            "extra_size_bytes": int(current_trace["size_bytes"]) - previous_size,
            "extra_record_count": int(current_trace["record_count"]) - previous_records,
            "extra_raw_response_record_count": int(
                current_trace["raw_response_record_count"]
            )
            - previous_raw_records,
        }

    rows_by_id: Dict[str, Dict[str, Any]] = dict(existing)
    resumed_completed_samples = sum(
        row.get("replay_completed") is True for row in existing.values()
    )
    verifier: Any = None
    processed_since_start = 0

    def ordered_checkpoint_rows() -> List[Dict[str, Any]]:
        return [rows_by_id[key] for key in dataset_keys if key in rows_by_id]

    def write_checkpoint() -> Dict[str, Any]:
        rows = ordered_checkpoint_rows()
        _atomic_write_json(output_path, rows)
        report = _report_payload(
            rows=rows,
            total_samples=len(dataset),
            output_path=output_path,
            artifacts=artifacts,
            frozen_manifest_info=frozen_manifest_info,
            configuration=configuration,
            configuration_sha256=configuration_sha256,
            resume_requested=resume,
            resumed_completed_samples=resumed_completed_samples,
            orphan_trace_recovery=orphan_trace_recovery,
            llm_trace=trace_audit.fingerprint(),
        )
        _atomic_write_json(report_path, report)
        return report

    for sample, trace, key in zip(dataset, traces, dataset_keys):
        prior = rows_by_id.get(key)
        if prior is not None and prior.get("replay_completed") is True:
            continue
        row = _base_output_row(
            sample,
            trace,
            mode=mode,
            config_sha256=configuration_sha256,
            candidate_source=candidate_source,
        )
        _, executable_records, precheck_suppressed = _rebuild_rule_records(
            trace, rule_index
        )
        row["used_rules"] = [
            str((record.get("rule") or {}).get("id") or "")
            for record in executable_records
        ]

        strategy = str(trace.get("selection_strategy") or "")
        if strategy in {"semantic_error", "semantic_unavailable"}:
            failure = {
                "kind": "frozen_retrieval_error",
                "stage": str(trace.get("semantic_failed_stage") or trace.get("terminal_stage") or "retrieval"),
                "reason": str(trace.get("semantic_selection_error") or trace.get("empty_reason") or strategy),
            }
            row["checker_status"] = "not_run_frozen_retrieval_error"
            row["checker_failures"] = [failure]
            row["replay_completed"] = True
            row["replay"]["terminal_frozen_retrieval_error"] = True
        elif not executable_records:
            row["checker_status"] = "complete_no_rules"
            row["replay_completed"] = True
            row["replay"]["no_executable_rules"] = True
            row["symbolic_check"]["suppressed_diagnostics"] = precheck_suppressed
            row["agentic"]["suppressed_diagnostics"] = precheck_suppressed
        else:
            row["replay"]["checker_attempted"] = True
            try:
                if verifier is None:
                    factory = verifier_factory or _default_verifier_factory
                    disabled_manifest = output_path.parent / (
                        f".checker-replay-no-bottom-up-{configuration_sha256}.json"
                    )
                    verifier = factory(
                        llm_model=str(model),
                        log_dir=str(output_path.parent),
                        results_dir=str(output_path.parent),
                        enable_symbolic_check=False,
                        enable_llm_cache=bool(enable_cache),
                        unified_rules_path=str(catalog_path),
                        experience_code_manifest_path=str(disabled_manifest),
                        precision_mode=precision_mode,
                        min_diagnostic_rule_score=min_diagnostic_rule_score,
                        max_diagnostics_per_sample=max_diagnostics_per_sample,
                        max_diagnostics_per_paragraph=max_diagnostics_per_paragraph,
                        quote_required_symbol_ratio=quote_required_symbol_ratio,
                        unified_retrieval_mode="lexical",
                        checker_gate_mode=mode,
                        checker_json_attempts=checker_json_attempts,
                    )
                    if getattr(verifier, "semantic_matcher", None) is not None:
                        raise ReplayValidationError(
                            "Checker replay must not instantiate a semantic matcher"
                        )
                    if getattr(verifier, "enable_symbolic_check", False):
                        raise ReplayValidationError(
                            "Checker replay must keep bottom-up symbolic checks disabled"
                        )
                    if str(getattr(verifier, "checker_gate_mode", mode)) != mode:
                        raise ReplayValidationError("verifier Checker mode does not match replay arm")
                    actual_min_confidence = float(
                        getattr(
                            verifier.semantic_checker,
                            "checker_min_confidence",
                            getattr(verifier, "checker_min_confidence", -1.0),
                        )
                    )
                    if not math.isclose(
                        actual_min_confidence,
                        FROZEN_CHECKER_MIN_CONFIDENCE,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ReplayValidationError(
                            "semantic_checker checker_min_confidence does not match frozen value 0.8"
                        )
                    verifier.semantic_checker.llm_temperature = float(llm_temperature)
                    verifier.semantic_checker.llm_max_output_tokens = int(
                        llm_max_output_tokens
                    )
                    verifier.semantic_checker.llm_trace_path = str(
                        resolved_llm_trace_path
                    )
                    verifier.semantic_checker.llm_trace_include_prompts = False

                rule_ids = [
                    str((record.get("rule") or {}).get("id") or "")
                    for record in executable_records
                ]
                verifier.semantic_checker.rules_to_check = rule_ids
                verifier.semantic_checker.rule_translations = {
                    rule_id: {
                        "srd": PhysicsRuleVerifier._build_srd_for_rule(record["rule"])
                    }
                    for rule_id, record in zip(rule_ids, executable_records)
                }
                trace_before = trace_audit.fingerprint()
                checker_result = verifier.semantic_checker.analyze(_checker_sample(sample))
                (
                    candidate_diagnostics,
                    checker_decisions,
                    checker_failures,
                    checker_suppressed,
                    checker_status,
                ) = _validate_checker_result(
                    checker_result,
                    mode=mode,
                    rule_ids=rule_ids,
                )
                row["candidate_diagnostics"] = copy.deepcopy(candidate_diagnostics)
                row["checker_decisions"] = checker_decisions
                row["checker_failures"] = checker_failures
                row["checker_suppressed_diagnostics"] = checker_suppressed
                row["checker_status"] = checker_status
                suppressed = list(precheck_suppressed) + list(checker_suppressed)
                checker_succeeded = _checker_status_succeeded(
                    checker_status, checker_failures
                )
                if checker_succeeded:
                    trace_after = trace_audit.fingerprint()
                    raw_response_delta = (
                        trace_after["raw_response_record_count"]
                        - trace_before["raw_response_record_count"]
                    )
                    if not enable_cache and raw_response_delta < len(rule_ids):
                        raise ReplayValidationError(
                            "successful uncached Checker attempt did not append at least one "
                            "raw LLM response per frozen rule"
                        )
                    filtered, low_conf_suppressed = (
                        verifier._filter_low_confidence_unified_diagnostics(
                            candidate_diagnostics,
                            executable_records,
                        )
                    )
                    published, release_suppressed = (
                        verifier._apply_diagnostic_release_gate(
                            filtered,
                            executable_records,
                        )
                    )
                    suppressed.extend(low_conf_suppressed)
                    suppressed.extend(release_suppressed)
                    row["diagnostics"] = published
                    row["score"] = sum(
                        -1.0 if item.get("severity") == "error" else -0.5
                        for item in published
                        if item.get("severity") in {"error", "warning"}
                    )
                    row["replay_completed"] = True
                    row["replay"]["checker_succeeded"] = True
                else:
                    if not checker_failures:
                        checker_failures = [
                            {
                                "kind": "checker_status_failure",
                                "status": checker_status,
                            }
                        ]
                        row["checker_failures"] = checker_failures
                    suppressed.append(
                        {
                            "reason": "checker_sample_incomplete",
                            "checker_status": checker_status,
                            "candidate_count": len(candidate_diagnostics),
                        }
                    )
                row["symbolic_check"]["suppressed_diagnostics"] = suppressed
                row["agentic"]["suppressed_diagnostics"] = suppressed
            except ReplayValidationError:
                raise
            except Exception as exc:
                failure = {
                    "kind": "checker_replay_exception",
                    "error_type": type(exc).__name__,
                    "reason": _safe_error(exc),
                }
                row["checker_status"] = "checker_replay_exception"
                row["checker_failures"] = [failure]
                row["symbolic_check"]["suppressed_diagnostics"] = list(
                    precheck_suppressed
                )
                row["agentic"]["suppressed_diagnostics"] = list(
                    precheck_suppressed
                )

        row["checker_failure_count"] = len(row.get("checker_failures") or [])
        row["replay"]["llm_trace_path"] = str(resolved_llm_trace_path)
        rows_by_id[key] = row
        processed_since_start += 1
        if processed_since_start % checkpoint_every == 0:
            write_checkpoint()

    final_source_identity = _source_identity()
    if (
        final_source_identity["source_tree_sha256"]
        != source_identity["source_tree_sha256"]
    ):
        raise ReplayValidationError("runtime source tree changed during Checker replay")
    report = write_checkpoint()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled Checker+gate arms from a frozen semantic trace or an "
            "explicit target-rule binding; retrieval and bottom-up diagnostics are "
            "never executed."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--frozen-retrieval",
        required=True,
        help=(
            "Immutable candidate trace: genuine semantic retrieval, or "
            "candidate_source=frozen_target_binding."
        ),
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument(
        "--prepare-manifest-only",
        action="store_true",
        help="Validate and freeze hashes/ordered IDs/selection projection, then exit.",
    )
    parser.add_argument(
        "--overwrite-frozen-manifest",
        action="store_true",
        help="Explicitly replace an existing frozen manifest during preparation.",
    )
    parser.add_argument(
        "--prompt-metadata-json",
        default="{}",
        help="Known retrieval prompt metadata as a JSON object, or @path; unknown may remain {}.",
    )
    parser.add_argument(
        "--retrieval-config-metadata-json",
        default="{}",
        help="Known retrieval config metadata as a JSON object, or @path; unknown may remain {}.",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--report", default="")
    parser.add_argument(
        "--llm-trace",
        default="",
        help="Checker raw-response JSONL; defaults to <output>.llm_trace.jsonl. Prompts are excluded.",
    )
    parser.add_argument("--model", default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--mode", choices=CHECKER_GATE_MODES, default="legacy")
    parser.add_argument("--checker-json-attempts", type=int, default=1)
    parser.add_argument("--enable-cache", action="store_true")
    parser.add_argument("--precision-mode", choices=("strict", "balanced", "score_only"), default="strict")
    parser.add_argument("--min-diagnostic-rule-score", type=float, default=None)
    parser.add_argument("--max-per-sample", type=int, default=12)
    parser.add_argument("--max-per-paragraph", type=int, default=2)
    parser.add_argument("--quote-required-symbol-ratio", type=float, default=0.0)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--llm-max-output-tokens", type=int, default=2048)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-kind", choices=_RUN_KINDS, default="development")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset_path = Path(args.dataset)
    retrieval_path = Path(args.frozen_retrieval)
    catalog_path = Path(args.catalog)
    manifest_path = Path(args.frozen_manifest)
    try:
        if args.prepare_manifest_only:
            prompt_metadata = _parse_metadata_json(
                args.prompt_metadata_json,
                label="prompt metadata",
            )
            config_metadata = _parse_metadata_json(
                args.retrieval_config_metadata_json,
                label="retrieval config metadata",
            )
            manifest = prepare_frozen_manifest(
                dataset_path=dataset_path,
                frozen_retrieval_path=retrieval_path,
                catalog_path=catalog_path,
                frozen_manifest_path=manifest_path,
                prompt_metadata=prompt_metadata,
                retrieval_config_metadata=config_metadata,
                overwrite=bool(args.overwrite_frozen_manifest),
            )
            print(
                "Frozen checker-replay manifest written: "
                f"{manifest_path} ({len(manifest['ordered_ids'])} samples)"
            )
            return 0
        if args.overwrite_frozen_manifest:
            parser.error("--overwrite-frozen-manifest requires --prepare-manifest-only")
        if not args.output or not args.report:
            parser.error("normal replay requires both --output and --report")
        report = run_checker_replay(
            dataset_path=dataset_path,
            frozen_retrieval_path=retrieval_path,
            catalog_path=catalog_path,
            frozen_manifest_path=manifest_path,
            output_path=Path(args.output),
            report_path=Path(args.report),
            model=args.model,
            mode=args.mode,
            checker_json_attempts=args.checker_json_attempts,
            enable_cache=args.enable_cache,
            precision_mode=args.precision_mode,
            min_diagnostic_rule_score=args.min_diagnostic_rule_score,
            max_diagnostics_per_sample=args.max_per_sample,
            max_diagnostics_per_paragraph=args.max_per_paragraph,
            quote_required_symbol_ratio=args.quote_required_symbol_ratio,
            llm_temperature=args.llm_temperature,
            llm_max_output_tokens=args.llm_max_output_tokens,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            llm_trace_path=Path(args.llm_trace) if args.llm_trace else None,
            run_kind=args.run_kind,
        )
    except ReplayValidationError as exc:
        parser.error(str(exc))
    statistics = report["statistics"]
    print(
        "Checker replay "
        f"{report['status']}: {statistics['completed_samples']}/{statistics['total_samples']} "
        f"complete, diagnostics={statistics['diagnostic_count']}"
    )
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
