from __future__ import annotations

"""Audit a schema-compatible semantic retrieval trace on the P3 dataset.

This secondary audit must never be merged into the primary controlled
target-binding Checker gate.  Until a run sidecar binds source, runtime, model,
API transport, parameters, and output hashes, this script validates trace shape
and classifications only; it does not prove production-run provenance.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.unified_semantic_matcher import UnifiedSemanticMatcher
from scripts.evaluate_checker_mechanism_gate import (
    MECHANISMS,
    MechanismEvaluationError,
    _validate_dataset,
)
from scripts.run_checker_replay import (
    CANDIDATE_SOURCE_SEMANTIC,
    ReplayValidationError,
    _catalog_index,
    _validate_frozen_trace,
    prepare_frozen_manifest,
)


AUDIT_SCHEMA_VERSION = "p3_semantic_retrieval_schema_audit_v1"
AUDIT_TYPE = "checker_mechanism_semantic_retrieval_schema_audit"
SUBSET_SCOPE = "secondary_semantic_retrieval_target_hit_only"
CLAIM_BOUNDARY = (
    "Secondary schema-compatible semantic-retrieval trace audit only. Without an "
    "execution sidecar binding source, runtime, model, API transport, parameters, "
    "and output hashes, this report is not production provenance or external-validity "
    "evidence. Never merge its denominators or candidates into the primary controlled "
    "frozen-target-binding Checker gate."
)
SEMANTIC_STRATEGIES = {
    "semantic_tree_selection",
    "semantic_tree_empty",
    "semantic_error",
    "semantic_unavailable",
}
FAILURE_STRATEGIES = {"semantic_error", "semantic_unavailable"}
CATEGORIES = (
    "target_hit_executable",
    "target_present_suppressed",
    "wrong_only",
    "empty",
    "retrieval_failure",
)


class SemanticRetrievalAuditError(ValueError):
    """Raised when an input or output violates the frozen audit protocol."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise SemanticRetrievalAuditError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise SemanticRetrievalAuditError(f"duplicate JSON key is forbidden: {key}")
        output[key] = value
    return output


def _load_json(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise SemanticRetrievalAuditError(f"{label} does not exist: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except SemanticRetrievalAuditError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticRetrievalAuditError(f"{label} is not strict JSON: {path}: {exc}") from exc


def _load_array(path: Path, *, label: str) -> List[Dict[str, Any]]:
    payload = _load_json(path, label=label)
    if not isinstance(payload, list):
        raise SemanticRetrievalAuditError(f"{label} must be a JSON array")
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise SemanticRetrievalAuditError(f"{label}[{index}] must be an object")
        rows.append(row)
    return rows


def _typed_id(value: Any, *, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise SemanticRetrievalAuditError(f"{label} must be a non-empty string or integer")
    if isinstance(value, str) and not value.strip():
        raise SemanticRetrievalAuditError(f"{label} must not be empty")
    kind = "str" if isinstance(value, str) else "int"
    return f"{kind}:{json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"


def _ordered_typed_ids(rows: Sequence[Mapping[str, Any]], *, label: str) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if "id" not in row:
            raise SemanticRetrievalAuditError(f"{label}[{index}] is missing id")
        typed = _typed_id(row.get("id"), label=f"{label}[{index}].id")
        if typed in seen:
            raise SemanticRetrievalAuditError(
                f"duplicate typed ID in {label}: {row.get('id')!r}"
            )
        seen.add(typed)
        output.append(typed)
    return output


def _strict_string(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        suffix = "a string" if allow_empty else "a non-empty string"
        raise SemanticRetrievalAuditError(f"{label} must be {suffix}")
    return value.strip() if not allow_empty else value


def _strict_score(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise SemanticRetrievalAuditError(f"{label} must be a JSON number")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise SemanticRetrievalAuditError(f"{label} must be finite and between 0 and 1")
    return score


def _normalize_trigger(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _catalog_audit_index(
    catalog: Mapping[str, Any],
    replay_index: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Add catalog-derived P3 strata to the replay ownership index."""

    records: List[Dict[str, Any]] = []
    for rule_id, owner in replay_index.items():
        rule = owner.get("rule")
        if not isinstance(rule, Mapping):
            raise SemanticRetrievalAuditError(f"catalog rule {rule_id!r} is malformed")
        match = re.fullmatch(r"(gen|exp)_[0-9a-f]{16}", rule_id)
        if not match:
            raise SemanticRetrievalAuditError(
                f"catalog rule {rule_id!r} has no supported gen/exp origin"
            )
        trigger = _strict_string(rule.get("trigger"), label=f"catalog rule {rule_id}.trigger")
        normalized_trigger = _normalize_trigger(trigger)
        if not normalized_trigger:
            raise SemanticRetrievalAuditError(
                f"catalog rule {rule_id}.trigger normalizes to empty"
            )
        hint = rule.get("symbolic_hint")
        if not isinstance(hint, Mapping):
            raise SemanticRetrievalAuditError(
                f"catalog rule {rule_id}.symbolic_hint must be an object"
            )
        primitive = _strict_string(
            hint.get("primitive"), label=f"catalog rule {rule_id}.symbolic_hint.primitive"
        )
        domain_id = _strict_string(
            owner.get("domain_id"), label=f"catalog owner {rule_id}.domain_id"
        )
        records.append(
            {
                "rule_id": rule_id,
                "domain_id": domain_id,
                "domain": str(owner.get("domain_name") or ""),
                "topic_id": str(owner.get("topic_id") or ""),
                "topic": str(owner.get("topic_name") or ""),
                "cluster_ids": set(owner.get("cluster_ids") or set()),
                "cluster_names": dict(owner.get("cluster_names") or {}),
                "origin": match.group(1),
                "symbolic_primitive": primitive,
                "has_symbolic_primitive": primitive != "none",
                "normalized_trigger": normalized_trigger,
                "title": str(rule.get("title") or ""),
            }
        )

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["domain_id"], record["origin"])].append(record)
    output: Dict[str, Dict[str, Any]] = {}
    for cell in sorted(grouped):
        rows = sorted(
            grouped[cell],
            key=lambda row: (
                len(row["normalized_trigger"]),
                row["normalized_trigger"],
                row["rule_id"],
            ),
        )
        broad_count = (len(rows) + 1) // 2
        for index, row in enumerate(rows):
            normalized = dict(row)
            normalized["trigger_scope_proxy"] = (
                "broad_proxy" if index < broad_count else "narrow_proxy"
            )
            output[normalized["rule_id"]] = normalized
    if set(output) != set(replay_index):
        raise SemanticRetrievalAuditError("catalog audit index lost rule ownership")
    return output


def _validate_dataset_catalog_ownership(
    dataset: Sequence[Mapping[str, Any]],
    catalog_index: Mapping[str, Mapping[str, Any]],
) -> None:
    mechanisms_by_rule: Dict[str, List[str]] = defaultdict(list)
    for index, row in enumerate(dataset):
        label = f"dataset[{index}]"
        top_rule_id = _strict_string(row.get("target_rule_id"), label=f"{label}.target_rule_id")
        target = row.get("target_rule")
        if not isinstance(target, Mapping):
            raise SemanticRetrievalAuditError(f"{label}.target_rule must be an object")
        nested_rule_id = _strict_string(
            target.get("rule_id"), label=f"{label}.target_rule.rule_id"
        )
        if top_rule_id != nested_rule_id:
            raise SemanticRetrievalAuditError(
                f"{label} top-level and nested target rule IDs differ"
            )
        owner = catalog_index.get(top_rule_id)
        if not isinstance(owner, Mapping):
            raise SemanticRetrievalAuditError(
                f"{label} target_rule_id references unknown catalog rule {top_rule_id!r}"
            )
        expected_fields = {
            "domain_id": owner.get("domain_id"),
            "domain": owner.get("domain"),
            "topic_id": owner.get("topic_id"),
            "topic": owner.get("topic"),
            "origin": owner.get("origin"),
            "symbolic_primitive": owner.get("symbolic_primitive"),
            "has_symbolic_primitive": owner.get("has_symbolic_primitive"),
            "trigger_scope_proxy": owner.get("trigger_scope_proxy"),
        }
        for field, expected in expected_fields.items():
            if target.get(field) != expected:
                raise SemanticRetrievalAuditError(
                    f"{label}.target_rule.{field} does not match catalog-derived value "
                    f"{expected!r}"
                )
        cluster_id = _strict_string(
            target.get("cluster_id"), label=f"{label}.target_rule.cluster_id"
        )
        if cluster_id not in set(owner.get("cluster_ids") or set()):
            raise SemanticRetrievalAuditError(
                f"{label}.target_rule.cluster_id does not own target rule"
            )
        mechanisms_by_rule[top_rule_id].append(str(row.get("mechanism") or ""))

    expected_mechanisms = Counter(MECHANISMS)
    for rule_id, mechanisms in mechanisms_by_rule.items():
        if Counter(mechanisms) != expected_mechanisms:
            raise SemanticRetrievalAuditError(
                f"target rule {rule_id!r} must have exactly one row for each P3 mechanism"
            )


def _validate_semantic_publish_gate(gate: Any, *, label: str) -> None:
    if not isinstance(gate, Mapping):
        raise SemanticRetrievalAuditError(f"{label} must be an object")
    required = {
        "publishable",
        "reasons",
        "score",
        "semantic_score",
        "score_kind",
        "min_publish_score",
        "selection_strategy",
    }
    if not required.issubset(gate):
        raise SemanticRetrievalAuditError(
            f"{label} is missing required fields: {sorted(required - set(gate))}"
        )
    if not isinstance(gate.get("publishable"), bool):
        raise SemanticRetrievalAuditError(f"{label}.publishable must be boolean")
    reasons = gate.get("reasons")
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        raise SemanticRetrievalAuditError(f"{label}.reasons must be an array of strings")
    for field in ("score", "semantic_score", "min_publish_score"):
        _strict_score(gate.get(field), label=f"{label}.{field}")
    if gate.get("score_kind") != "semantic_0_1":
        raise SemanticRetrievalAuditError(f"{label}.score_kind must be 'semantic_0_1'")
    if gate.get("selection_strategy") != "semantic_tree_selection":
        raise SemanticRetrievalAuditError(
            f"{label}.selection_strategy must be 'semantic_tree_selection'"
        )
    if gate.get("publishable") is True and reasons:
        raise SemanticRetrievalAuditError(
            f"{label} cannot be publishable with suppression reasons"
        )


def _validate_semantic_trace_schema(
    traces: Sequence[Mapping[str, Any]],
    catalog_index: Mapping[str, Mapping[str, Any]],
) -> None:
    domain_owners = {
        (str(owner["domain_id"]), str(owner["domain"])) for owner in catalog_index.values()
    }
    topic_owners = {
        (str(owner["domain"]), str(owner["topic_id"]), str(owner["topic"]))
        for owner in catalog_index.values()
    }
    cluster_owners = {
        (
            str(owner["domain"]),
            str(owner["topic_id"]),
            str(owner["topic"]),
            str(cluster_id),
            str((owner.get("cluster_names") or {}).get(cluster_id) or ""),
        )
        for owner in catalog_index.values()
        for cluster_id in owner.get("cluster_ids") or set()
    }
    required_top = {
        "id",
        "topic",
        "verifier",
        "unified_mode",
        "unified_retrieval_mode",
        "selection_strategy",
        "retrieval_score_kind",
        "semantic_min_publish_score",
        "semantic_selection_error",
        "semantic_failed_stage",
        "semantic_input_policy",
        "background_analysis",
        "navigation_trace",
        "terminal_stage",
        "empty_reason",
        "retrieved_domains",
        "retrieved_topics",
        "retrieved_clusters",
        "retrieved_rules",
    }
    for sample_pos, trace in enumerate(traces):
        label = f"semantic trace[{sample_pos}]"
        if not required_top.issubset(trace):
            raise SemanticRetrievalAuditError(
                f"{label} is missing required fields: {sorted(required_top - set(trace))}"
            )
        if "target_rule_id" in trace:
            raise SemanticRetrievalAuditError(
                f"{label} contains target_rule_id and is not a raw semantic retrieval trace"
            )
        explicit_source = trace.get("candidate_source")
        if explicit_source not in {None, "", CANDIDATE_SOURCE_SEMANTIC}:
            raise SemanticRetrievalAuditError(
                f"{label}.candidate_source is not genuine semantic retrieval"
            )
        if trace.get("unified_retrieval_mode") != "semantic":
            raise SemanticRetrievalAuditError(
                f"{label}.unified_retrieval_mode must be exactly 'semantic'"
            )
        strategy = trace.get("selection_strategy")
        if strategy not in SEMANTIC_STRATEGIES:
            raise SemanticRetrievalAuditError(
                f"{label}.selection_strategy is not a supported semantic strategy"
            )
        if trace.get("retrieval_score_kind") != "semantic_0_1":
            raise SemanticRetrievalAuditError(
                f"{label}.retrieval_score_kind must be exactly 'semantic_0_1'"
            )
        if trace.get("verifier") != "unified_v2_semantic_retrieval_only":
            raise SemanticRetrievalAuditError(
                f"{label}.verifier does not identify run_verifier --retrieval-only"
            )
        if trace.get("unified_mode") is not True:
            raise SemanticRetrievalAuditError(f"{label}.unified_mode must be exactly true")
        if trace.get("semantic_input_policy") != UnifiedSemanticMatcher.INPUT_POLICY:
            raise SemanticRetrievalAuditError(
                f"{label}.semantic_input_policy does not match the production semantic policy"
            )
        _strict_score(
            trace.get("semantic_min_publish_score"),
            label=f"{label}.semantic_min_publish_score",
        )
        for field in (
            "semantic_selection_error",
            "semantic_failed_stage",
            "terminal_stage",
            "empty_reason",
        ):
            _strict_string(trace.get(field), label=f"{label}.{field}", allow_empty=True)
        for field in ("background_analysis", "navigation_trace"):
            if not isinstance(trace.get(field), Mapping):
                raise SemanticRetrievalAuditError(f"{label}.{field} must be an object")
        for field in (
            "retrieved_domains",
            "retrieved_topics",
            "retrieved_clusters",
            "retrieved_rules",
        ):
            value = trace.get(field)
            if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
                raise SemanticRetrievalAuditError(
                    f"{label}.{field} must be an array of objects"
                )

        error_text = str(trace.get("semantic_selection_error") or "")
        failed_stage = str(trace.get("semantic_failed_stage") or "")
        empty_reason = str(trace.get("empty_reason") or "")
        rules = list(trace.get("retrieved_rules") or [])
        if strategy in FAILURE_STRATEGIES:
            if not error_text or not failed_stage or not empty_reason:
                raise SemanticRetrievalAuditError(
                    f"{label} failure must record error, failed stage, and empty reason"
                )
        elif error_text or failed_stage:
            raise SemanticRetrievalAuditError(
                f"{label} non-failure strategy contains semantic failure metadata"
            )
        if strategy == "semantic_tree_selection" and not rules:
            raise SemanticRetrievalAuditError(f"{label} selection contains no candidates")
        if strategy == "semantic_tree_empty" and (rules or not empty_reason):
            raise SemanticRetrievalAuditError(
                f"{label} empty selection must have no candidates and a reason"
            )

        seen_domains: set[Tuple[str, str]] = set()
        for item_pos, item in enumerate(trace.get("retrieved_domains") or []):
            prefix = f"{label}.retrieved_domains[{item_pos}]"
            key = (
                _strict_string(item.get("domain_id"), label=f"{prefix}.domain_id"),
                _strict_string(item.get("domain"), label=f"{prefix}.domain"),
            )
            if key not in domain_owners or key in seen_domains:
                raise SemanticRetrievalAuditError(f"{prefix} has invalid/duplicate catalog ownership")
            seen_domains.add(key)
            _strict_score(item.get("score"), label=f"{prefix}.score")
            if item.get("score_kind") != "semantic_0_1":
                raise SemanticRetrievalAuditError(f"{prefix}.score_kind is invalid")

        seen_topics: set[Tuple[str, str, str]] = set()
        for item_pos, item in enumerate(trace.get("retrieved_topics") or []):
            prefix = f"{label}.retrieved_topics[{item_pos}]"
            key = (
                _strict_string(item.get("domain"), label=f"{prefix}.domain"),
                _strict_string(item.get("topic_id"), label=f"{prefix}.topic_id"),
                _strict_string(item.get("topic"), label=f"{prefix}.topic"),
            )
            if key not in topic_owners or key in seen_topics:
                raise SemanticRetrievalAuditError(f"{prefix} has invalid/duplicate catalog ownership")
            if not any(domain == key[0] for _, domain in seen_domains):
                raise SemanticRetrievalAuditError(f"{prefix} owner domain is absent")
            seen_topics.add(key)
            _strict_score(item.get("score"), label=f"{prefix}.score")
            if item.get("score_kind") != "semantic_0_1":
                raise SemanticRetrievalAuditError(f"{prefix}.score_kind is invalid")

        expected_topic = (
            str((trace.get("retrieved_topics") or [])[0].get("topic") or "")
            if trace.get("retrieved_topics")
            else None
        )
        if trace.get("topic") != expected_topic:
            raise SemanticRetrievalAuditError(f"{label}.topic is not the primary retrieved topic")

        seen_clusters: set[Tuple[str, str, str, str, str]] = set()
        for item_pos, item in enumerate(trace.get("retrieved_clusters") or []):
            prefix = f"{label}.retrieved_clusters[{item_pos}]"
            key = (
                _strict_string(item.get("domain"), label=f"{prefix}.domain"),
                _strict_string(item.get("topic_id"), label=f"{prefix}.topic_id"),
                _strict_string(item.get("topic"), label=f"{prefix}.topic"),
                _strict_string(item.get("cluster_id"), label=f"{prefix}.cluster_id"),
                _strict_string(item.get("cluster"), label=f"{prefix}.cluster"),
            )
            if key not in cluster_owners or key in seen_clusters:
                raise SemanticRetrievalAuditError(f"{prefix} has invalid/duplicate catalog ownership")
            if key[:3] not in seen_topics:
                raise SemanticRetrievalAuditError(f"{prefix} owner topic is absent")
            seen_clusters.add(key)
            _strict_score(item.get("score"), label=f"{prefix}.score")
            if item.get("score_kind") != "semantic_0_1":
                raise SemanticRetrievalAuditError(f"{prefix}.score_kind is invalid")

        for item_pos, item in enumerate(rules):
            prefix = f"{label}.retrieved_rules[{item_pos}]"
            required_rule = {
                "rule_id",
                "domain",
                "topic_id",
                "topic",
                "cluster_id",
                "cluster",
                "title",
                "scope",
                "score",
                "score_kind",
                "semantic_score",
                "grounding_score",
                "publish_gate",
                "manual_override_reason",
                "evidence",
            }
            if not required_rule.issubset(item):
                raise SemanticRetrievalAuditError(
                    f"{prefix} is missing required fields: {sorted(required_rule - set(item))}"
                )
            rule_id = _strict_string(item.get("rule_id"), label=f"{prefix}.rule_id")
            owner = catalog_index.get(rule_id)
            if not isinstance(owner, Mapping):
                raise SemanticRetrievalAuditError(f"{prefix} references an unknown catalog rule")
            cluster_id = _strict_string(item.get("cluster_id"), label=f"{prefix}.cluster_id")
            expected_cluster = str((owner.get("cluster_names") or {}).get(cluster_id) or "")
            for field, expected in (
                ("domain", owner.get("domain")),
                ("topic_id", owner.get("topic_id")),
                ("topic", owner.get("topic")),
                ("cluster", expected_cluster),
                ("title", owner.get("title")),
            ):
                if item.get(field) != expected:
                    raise SemanticRetrievalAuditError(
                        f"{prefix}.{field} does not match catalog ownership/content"
                    )
            _strict_string(item.get("scope"), label=f"{prefix}.scope")
            _strict_string(
                item.get("manual_override_reason"),
                label=f"{prefix}.manual_override_reason",
                allow_empty=True,
            )
            if not isinstance(item.get("evidence"), Mapping):
                raise SemanticRetrievalAuditError(f"{prefix}.evidence must be an object")
            for field in ("score", "semantic_score", "grounding_score"):
                _strict_score(item.get(field), label=f"{prefix}.{field}")
            if item.get("score_kind") != "semantic_0_1":
                raise SemanticRetrievalAuditError(f"{prefix}.score_kind is invalid")
            if float(item["score"]) != float(item["semantic_score"]):
                raise SemanticRetrievalAuditError(
                    f"{prefix}.score and semantic_score must be identical"
                )
            for optional_bool in ("partial", "executable"):
                if optional_bool in item and not isinstance(item.get(optional_bool), bool):
                    raise SemanticRetrievalAuditError(
                        f"{prefix}.{optional_bool} must be boolean when present"
                    )
            _validate_semantic_publish_gate(
                item.get("publish_gate"), label=f"{prefix}.publish_gate"
            )
            gate = item["publish_gate"]
            if float(gate["score"]) != round(float(item["semantic_score"]), 4):
                raise SemanticRetrievalAuditError(
                    f"{prefix}.publish_gate.score does not match the semantic candidate"
                )
            if float(gate["semantic_score"]) != round(
                float(item["semantic_score"]), 4
            ):
                raise SemanticRetrievalAuditError(
                    f"{prefix}.publish_gate.semantic_score does not match the candidate"
                )
            if float(gate["min_publish_score"]) != round(
                float(trace["semantic_min_publish_score"]), 4
            ):
                raise SemanticRetrievalAuditError(
                    f"{prefix}.publish_gate.min_publish_score does not match the trace"
                )


def _input_fingerprint(
    path: Path,
    rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if rows is not None:
        typed_ids = _ordered_typed_ids(rows, label=path.name)
        output.update(
            {
                "record_count": len(rows),
                "ordered_typed_ids": typed_ids,
                "ordered_typed_ids_sha256": _object_sha256(typed_ids),
            }
        )
    return output


def _classify_sample(
    dataset_row: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    index: int,
) -> Dict[str, Any]:
    target_rule_id = str(dataset_row.get("target_rule_id") or "")
    candidates = list(trace.get("retrieved_rules") or [])
    target_rank: Optional[int] = None
    target_candidate: Optional[Mapping[str, Any]] = None
    for rank, candidate in enumerate(candidates, start=1):
        if candidate.get("rule_id") == target_rule_id:
            target_rank = rank
            target_candidate = candidate
            break

    strategy = str(trace.get("selection_strategy") or "")
    suppression_reasons: List[str] = []
    if strategy in FAILURE_STRATEGIES:
        category = "retrieval_failure"
    elif strategy == "semantic_tree_empty":
        category = "empty"
    elif target_candidate is None:
        category = "wrong_only"
    else:
        gate = target_candidate.get("publish_gate")
        if target_candidate.get("partial") is True:
            suppression_reasons.append("partial_candidate")
        if target_candidate.get("executable") is False:
            suppression_reasons.append("explicitly_non_executable")
        if not isinstance(gate, Mapping) or gate.get("publishable") is not True:
            suppression_reasons.append("publish_gate_not_exact_true")
            if isinstance(gate, Mapping):
                suppression_reasons.extend(str(reason) for reason in gate.get("reasons") or [])
        category = (
            "target_present_suppressed" if suppression_reasons else "target_hit_executable"
        )

    target = dataset_row.get("target_rule") or {}
    return {
        "index": index,
        "id": dataset_row.get("id"),
        "typed_id": _typed_id(dataset_row.get("id"), label=f"dataset[{index}].id"),
        "target_rule_id": target_rule_id,
        "mechanism": dataset_row.get("mechanism"),
        "mechanism_subtype": dataset_row.get("mechanism_subtype"),
        "domain": target.get("domain"),
        "origin": target.get("origin"),
        "trigger_scope_proxy": target.get("trigger_scope_proxy"),
        "selection_strategy": strategy,
        "classification": category,
        "candidate_count": len(candidates),
        "target_rank": target_rank,
        "target_semantic_score": (
            float(target_candidate.get("semantic_score"))
            if target_candidate is not None
            else None
        ),
        "target_publishable": (
            (target_candidate.get("publish_gate") or {}).get("publishable")
            if target_candidate is not None
            else None
        ),
        "suppression_reasons": suppression_reasons,
        "failure_stage": str(trace.get("semantic_failed_stage") or ""),
        "failure_error": str(trace.get("semantic_selection_error") or ""),
        "empty_reason": str(trace.get("empty_reason") or ""),
    }


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 6) if denominator else None


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(sum(values) / len(values), 6) if values else None


def _summarize(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter(str(record.get("classification") or "") for record in records)
    total = len(records)
    failures = counts["retrieval_failure"]
    empty = counts["empty"]
    assessed = total - failures - empty
    hits = counts["target_hit_executable"]
    suppressed = counts["target_present_suppressed"]
    wrong = counts["wrong_only"]
    target_present = hits + suppressed
    ranks = [
        float(record["target_rank"])
        for record in records
        if record.get("target_rank") is not None
        and record.get("classification") not in {"retrieval_failure", "empty"}
    ]
    candidate_counts = [
        float(record.get("candidate_count") or 0)
        for record in records
        if record.get("classification") not in {"retrieval_failure", "empty"}
    ]
    return {
        "sample_count": total,
        "category_counts": {category: counts[category] for category in CATEGORIES},
        "retrieval_completed_count": total - failures,
        "retrieval_completed_coverage": _ratio(total - failures, total),
        "successful_nonempty_count": assessed,
        "successful_nonempty_coverage": _ratio(assessed, total),
        "retrieval_failure_count": failures,
        "empty_count": empty,
        "target_hit_executable_count": hits,
        "target_present_suppressed_count": suppressed,
        "wrong_only_count": wrong,
        "target_recall_denominator": assessed,
        "target_hit_executable_rate": _ratio(hits, assessed),
        "target_present_rate": _ratio(target_present, assessed),
        "target_miss_count": suppressed + wrong,
        "mean_target_rank_when_present": _mean(ranks),
        "min_target_rank_when_present": int(min(ranks)) if ranks else None,
        "max_target_rank_when_present": int(max(ranks)) if ranks else None,
        "mean_candidate_count_nonempty_success": _mean(candidate_counts),
        "denominator_policy": (
            "retrieval_failure and semantic_tree_empty are coverage outcomes, not target "
            "misses; target recall is conditional on a successful non-empty retrieval."
        ),
    }


def _stratify(
    records: Sequence[Mapping[str, Any]], field: str
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(field) or "")].append(record)
    return {key: _summarize(grouped[key]) for key in sorted(grouped)}


def _rule_level_summary(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    order: List[str] = []
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        rule_id = str(record.get("target_rule_id") or "")
        if rule_id not in grouped:
            order.append(rule_id)
        grouped[rule_id].append(record)
    output: List[Dict[str, Any]] = []
    for rule_id in order:
        rows = grouped[rule_id]
        summary = _summarize(rows)
        output.append(
            {
                "target_rule_id": rule_id,
                "domain": rows[0].get("domain"),
                "origin": rows[0].get("origin"),
                "trigger_scope_proxy": rows[0].get("trigger_scope_proxy"),
                "summary": summary,
                "mechanisms": [
                    {
                        "mechanism": row.get("mechanism"),
                        "mechanism_subtype": row.get("mechanism_subtype"),
                        "classification": row.get("classification"),
                        "target_rank": row.get("target_rank"),
                        "candidate_count": row.get("candidate_count"),
                    }
                    for row in rows
                ],
            }
        )
    return output


def _ensure_new_outputs(paths: Iterable[Optional[Path]], *, inputs: Sequence[Path]) -> None:
    resolved_inputs = {path.resolve() for path in inputs}
    resolved_outputs: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        resolved = path.resolve()
        if resolved in resolved_inputs:
            raise SemanticRetrievalAuditError(f"output must not overwrite an input: {path}")
        if resolved in resolved_outputs:
            raise SemanticRetrievalAuditError(f"output paths must be distinct: {path}")
        if path.exists():
            raise SemanticRetrievalAuditError(f"refusing to overwrite existing output: {path}")
        resolved_outputs.add(resolved)


def _write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise SemanticRetrievalAuditError(
            f"refusing to overwrite existing output: {path}"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def audit_semantic_retrieval(
    *,
    dataset_path: Path,
    semantic_trace_path: Path,
    catalog_path: Path,
    report_path: Optional[Path] = None,
    subset_dataset_path: Optional[Path] = None,
    subset_manifest_path: Optional[Path] = None,
    subset_trace_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate and classify a schema-compatible semantic-retrieval trace."""

    if (subset_dataset_path is None) != (subset_manifest_path is None):
        raise SemanticRetrievalAuditError(
            "subset dataset and subset manifest outputs must be requested together"
        )
    if subset_trace_path is not None and subset_dataset_path is None:
        raise SemanticRetrievalAuditError(
            "subset trace output requires subset dataset and manifest outputs"
        )
    if subset_dataset_path is not None and subset_trace_path is None:
        subset_trace_path = subset_dataset_path.with_name(
            f"{subset_dataset_path.stem}.semantic_trace.json"
        )
    _ensure_new_outputs(
        (report_path, subset_dataset_path, subset_trace_path, subset_manifest_path),
        inputs=(dataset_path, semantic_trace_path, catalog_path),
    )

    raw_dataset = _load_array(dataset_path, label="P3 dataset")
    traces = _load_array(semantic_trace_path, label="semantic retrieval trace")
    catalog = _load_json(catalog_path, label="unified catalog")
    if not isinstance(catalog, dict):
        raise SemanticRetrievalAuditError("unified catalog must be a JSON object")
    if not raw_dataset:
        raise SemanticRetrievalAuditError("P3 dataset must not be empty")
    try:
        dataset, dataset_typed_ids = _validate_dataset(raw_dataset)
    except MechanismEvaluationError as exc:
        raise SemanticRetrievalAuditError(f"invalid P3 dataset: {exc}") from exc
    trace_typed_ids = _ordered_typed_ids(traces, label="semantic retrieval trace")
    if dataset_typed_ids != trace_typed_ids:
        raise SemanticRetrievalAuditError(
            "dataset and semantic retrieval trace must have exactly the same typed IDs "
            "in the same order"
        )
    try:
        replay_index = _catalog_index(catalog)
    except ReplayValidationError as exc:
        raise SemanticRetrievalAuditError(f"invalid unified catalog: {exc}") from exc
    catalog_index = _catalog_audit_index(catalog, replay_index)
    _validate_dataset_catalog_ownership(dataset, catalog_index)
    _validate_semantic_trace_schema(traces, catalog_index)
    try:
        _validate_frozen_trace(traces, replay_index, dataset=dataset)
    except ReplayValidationError as exc:
        raise SemanticRetrievalAuditError(f"invalid semantic retrieval trace: {exc}") from exc

    records = [
        _classify_sample(dataset_row, trace, index=index)
        for index, (dataset_row, trace) in enumerate(zip(dataset, traces))
    ]
    inputs = {
        "dataset": _input_fingerprint(dataset_path, dataset),
        "semantic_retrieval_trace": _input_fingerprint(semantic_trace_path, traces),
        "unified_catalog": {
            **_input_fingerprint(catalog_path),
            "catalog_type": (catalog.get("metadata") or {}).get("catalog_type"),
            "rule_count": len(catalog_index),
        },
    }
    report: Dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_type": AUDIT_TYPE,
        "created_at_utc": _utc_now(),
        "candidate_source": CANDIDATE_SOURCE_SEMANTIC,
        "provenance_status": "unverified_without_execution_sidecar",
        "claim_boundary": CLAIM_BOUNDARY,
        "classification_policy": {
            "target_hit_executable": (
                "target present; partial is not true; executable is not false; "
                "publish_gate.publishable is exactly true"
            ),
            "target_present_suppressed": (
                "target present but at least one executable/publication condition fails"
            ),
            "wrong_only": "successful non-empty retrieval contains no target rule",
            "empty": "semantic_tree_empty; reported as coverage, never as a target miss",
            "retrieval_failure": (
                "semantic_error or semantic_unavailable; reported as coverage, never as a target miss"
            ),
            "target_rank_base": 1,
        },
        "inputs": inputs,
        "ordered_typed_ids": dataset_typed_ids,
        "ordered_typed_ids_sha256": _object_sha256(dataset_typed_ids),
        "overall": _summarize(records),
        "stratified": {
            "mechanism": _stratify(records, "mechanism"),
            "domain": _stratify(records, "domain"),
            "origin": _stratify(records, "origin"),
            "trigger_scope_proxy": _stratify(records, "trigger_scope_proxy"),
        },
        "rule_level_summary": _rule_level_summary(records),
        "samples": records,
        "target_hit_subset": None,
    }

    if subset_dataset_path is not None:
        assert subset_trace_path is not None
        assert subset_manifest_path is not None
        hit_positions = [
            index
            for index, record in enumerate(records)
            if record["classification"] == "target_hit_executable"
        ]
        subset_dataset = [dataset[index] for index in hit_positions]
        subset_traces = [traces[index] for index in hit_positions]
        subset_typed_ids = _ordered_typed_ids(subset_dataset, label="target-hit subset")
        _write_json_new(subset_dataset_path, subset_dataset)
        _write_json_new(subset_trace_path, subset_traces)
        try:
            manifest = prepare_frozen_manifest(
                dataset_path=subset_dataset_path,
                frozen_retrieval_path=subset_trace_path,
                catalog_path=catalog_path,
                frozen_manifest_path=subset_manifest_path,
                prompt_metadata={
                    "source": "run_verifier.py --retrieval-only",
                    "semantic_input_policy": UnifiedSemanticMatcher.INPUT_POLICY,
                },
                retrieval_config_metadata={
                    "scope": SUBSET_SCOPE,
                    "claim_boundary": CLAIM_BOUNDARY,
                    "selection_category": "target_hit_executable",
                    "source_input_fingerprints": inputs,
                    "source_ordered_typed_ids_sha256": _object_sha256(dataset_typed_ids),
                    "selected_source_positions": hit_positions,
                    "ordered_typed_ids": subset_typed_ids,
                    "ordered_typed_ids_sha256": _object_sha256(subset_typed_ids),
                },
            )
        except ReplayValidationError as exc:
            raise SemanticRetrievalAuditError(
                f"could not freeze target-hit replay subset: {exc}"
            ) from exc
        report["target_hit_subset"] = {
            "scope": SUBSET_SCOPE,
            "claim_boundary": CLAIM_BOUNDARY,
            "dataset": _input_fingerprint(subset_dataset_path, subset_dataset),
            "semantic_retrieval_trace": _input_fingerprint(subset_trace_path, subset_traces),
            "replay_manifest": {
                "path": str(subset_manifest_path.resolve()),
                "size_bytes": subset_manifest_path.stat().st_size,
                "sha256": _sha256_file(subset_manifest_path),
                "manifest_type": manifest.get("manifest_type"),
            },
            "ordered_typed_ids": subset_typed_ids,
            "ordered_typed_ids_sha256": _object_sha256(subset_typed_ids),
            "selected_source_positions": hit_positions,
        }

    if report_path is not None:
        _write_json_new(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a schema-compatible run_verifier --retrieval-only semantic trace on "
            "the P3 mechanism dataset. Without an execution sidecar this is not proof "
            "of production provenance, and it is never pooled with the primary "
            "controlled target-binding Checker gate."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--semantic-trace", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New audit report path")
    parser.add_argument(
        "--target-hit-subset-dataset",
        type=Path,
        help="Optional new dataset containing only executable target hits",
    )
    parser.add_argument(
        "--target-hit-subset-manifest",
        type=Path,
        help="Optional new replay-compatible frozen manifest (requires subset dataset)",
    )
    parser.add_argument(
        "--target-hit-subset-trace",
        type=Path,
        help=(
            "Optional new aligned semantic trace; defaults beside the subset dataset "
            "and is required internally for secondary Checker replay"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = audit_semantic_retrieval(
            dataset_path=args.dataset,
            semantic_trace_path=args.semantic_trace,
            catalog_path=args.catalog,
            report_path=args.output,
            subset_dataset_path=args.target_hit_subset_dataset,
            subset_manifest_path=args.target_hit_subset_manifest,
            subset_trace_path=args.target_hit_subset_trace,
        )
    except SemanticRetrievalAuditError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "sample_count": report["overall"]["sample_count"],
                "category_counts": report["overall"]["category_counts"],
                "target_hit_executable_rate": report["overall"][
                    "target_hit_executable_rate"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
