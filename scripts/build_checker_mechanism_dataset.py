from __future__ import annotations

"""Build the frozen P3 rule-mechanism dataset and honest target-binding trace.

The rule plan is derived only from the unified catalog.  In particular, this
script does not inspect evaluation data or previous results.  ``broad_proxy``
and ``narrow_proxy`` below are deterministic trigger-length strata; they are
not claims about semantic trigger breadth.
"""

import argparse
import hashlib
import ipaddress
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_manifest import capture_source_state, probe_python_runtime


EXPECTED_CATALOG_SHA256 = "838506bfa01b67cc0038ee05d5832dcaecbc55a260f736b9107bfbfd52612005"
PLAN_SCHEMA_VERSION = "p3_checker_rule_plan_v1"
PLAN_TYPE = "checker_mechanism_rule_plan"
DATASET_SCHEMA_VERSION = "p3_checker_mechanism_case_v1"
CANDIDATE_TRACE_SCHEMA_VERSION = "p3_target_candidate_trace_v1"
CHECKPOINT_SCHEMA_VERSION = "p3_checker_mechanism_checkpoint_v2"
RAW_TRACE_SCHEMA_VERSION = "p3_gt_raw_response_trace_v2"
GENERATION_MANIFEST_SCHEMA_VERSION = "p3_checker_mechanism_generation_manifest_v3"
GENERATION_MANIFEST_TYPE = "checker_mechanism_dataset_generation"
PROMPT_VERSION = "p3_checker_mechanism_gt_gemini3_flash_v3"
SELECTION_ALGORITHM_VERSION = "p3_catalog_hash_stratified_v1"
TRIGGER_SCOPE_PROXY_VERSION = "unicode_nfc_length_half_within_domain_origin_v1"
FIXED_SEED = "P3_RULE_SAMPLE_V1_20260808"
DEFAULT_MODEL = "gemini-3-flash-preview"
RUN_KINDS = ("development", "validation", "final")

MECHANISMS: Tuple[str, ...] = (
    "true_violation",
    "applicable_correct",
    "symbol_overlap_inapplicable",
    "equivalent_alternative",
    "insufficient_or_self_corrected",
)
CONSISTENCY_STATUSES = {
    "confirmed_violation",
    "equivalent_or_alternative",
    "self_corrected",
    "uncertain",
}

# Exact cell quotas.  Each tuple is
# (total, broad_proxy, narrow_proxy, primitive-none broad, primitive-none narrow).
CELL_QUOTAS: Dict[str, Dict[str, Tuple[int, int, int, int, int]]] = {
    "mechanics": {
        "gen": (6, 3, 3, 1, 1),
        "exp": (5, 2, 3, 1, 1),
    },
    "electromagnetism": {
        "gen": (6, 3, 3, 1, 1),
        "exp": (5, 2, 3, 1, 1),
    },
    "thermodynamics_statistical_physics": {
        "gen": (5, 2, 3, 1, 0),
        "exp": (5, 3, 2, 1, 1),
    },
    "optics": {
        "gen": (5, 2, 3, 0, 0),
        "exp": (5, 3, 2, 0, 0),
    },
    "modern_physics": {
        "gen": (5, 2, 3, 1, 0),
        "exp": (5, 3, 2, 1, 1),
    },
    "experimental_physics": {
        "gen": (3, 2, 1, 0, 0),
        "exp": (5, 3, 2, 0, 0),
    },
}


class MechanismDatasetError(ValueError):
    """Raised when a frozen input or generated artifact violates the protocol."""


ResponseProvider = Callable[[Mapping[str, Any], str, str, int], Any]


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


def _sha256_file_prefix(path: Path, size_bytes: int) -> str:
    if size_bytes < 0:
        raise MechanismDatasetError("raw trace checkpoint size must be non-negative")
    digest = hashlib.sha256()
    remaining = size_bytes
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise MechanismDatasetError("raw trace is shorter than checkpoint prefix")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reject_constant(value: str) -> None:
    raise MechanismDatasetError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MechanismDatasetError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _strict_json_loads(raw: str, *, label: str) -> Any:
    try:
        return json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except MechanismDatasetError:
        raise
    except json.JSONDecodeError as exc:
        raise MechanismDatasetError(f"{label} is not strict JSON: {exc}") from exc


def _load_json(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise MechanismDatasetError(f"{label} does not exist: {path}")
    return _strict_json_loads(path.read_text(encoding="utf-8"), label=label)


def _require_exact_keys(value: Mapping[str, Any], keys: Iterable[str], *, label: str) -> None:
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise MechanismDatasetError(
            f"{label} keys must be exactly {sorted(expected)}; got {sorted(actual)}"
        )


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MechanismDatasetError(f"{label} must be a non-empty string")
    return value.strip()


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MechanismDatasetError(f"{label} must be an integer")
    return value


def _normalize_trigger(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _origin(rule_id: str) -> str:
    match = re.fullmatch(r"(gen|exp)_[0-9a-f]{16}", rule_id)
    if not match:
        raise MechanismDatasetError(f"unsupported rule_id format: {rule_id!r}")
    return match.group(1)


def _selection_key(*, catalog_sha256: str, rule_id: str) -> str:
    payload = f"{FIXED_SEED}\0{catalog_sha256}\0{rule_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _catalog_records(catalog: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not isinstance(catalog, dict):
        raise MechanismDatasetError("catalog must be a JSON object")
    metadata = catalog.get("metadata")
    domains = catalog.get("domains")
    if not isinstance(metadata, dict) or metadata.get("catalog_type") != "unified_rules_v2":
        raise MechanismDatasetError("catalog_type must be unified_rules_v2")
    if not isinstance(domains, list):
        raise MechanismDatasetError("catalog.domains must be an array")
    domain_ids = [domain.get("id") for domain in domains if isinstance(domain, dict)]
    if set(domain_ids) != set(CELL_QUOTAS) or len(domain_ids) != len(CELL_QUOTAS):
        raise MechanismDatasetError("catalog must contain exactly the six pre-registered domains")

    records: List[Dict[str, Any]] = []
    rule_ids: set[str] = set()
    topic_count = 0
    topics_with_rules = 0
    cluster_count = 0
    for domain_pos, domain in enumerate(domains):
        if not isinstance(domain, dict):
            raise MechanismDatasetError(f"domains[{domain_pos}] must be an object")
        domain_id = _nonempty_string(domain.get("id"), label=f"domains[{domain_pos}].id")
        domain_name = _nonempty_string(
            domain.get("name"), label=f"domains[{domain_pos}].name"
        )
        topics = domain.get("topics")
        if not isinstance(topics, list):
            raise MechanismDatasetError(f"domain {domain_id} topics must be an array")
        topic_count += len(topics)
        for topic_pos, topic in enumerate(topics):
            if not isinstance(topic, dict):
                raise MechanismDatasetError(f"{domain_id}.topics[{topic_pos}] must be an object")
            topic_id = _nonempty_string(
                topic.get("id"), label=f"{domain_id}.topics[{topic_pos}].id"
            )
            topic_name = _nonempty_string(
                topic.get("name"), label=f"{domain_id}/{topic_id}.name"
            )
            if not topic_id.startswith(domain_id + "."):
                raise MechanismDatasetError(f"topic {topic_id!r} is outside domain {domain_id!r}")
            rules = topic.get("rules")
            clusters = topic.get("scenario_clusters")
            if not isinstance(rules, list) or not isinstance(clusters, list):
                raise MechanismDatasetError(f"topic {topic_id} rules/clusters must be arrays")
            if rules:
                topics_with_rules += 1
            topic_rules: Dict[str, Dict[str, Any]] = {}
            for rule_pos, rule in enumerate(rules):
                if not isinstance(rule, dict):
                    raise MechanismDatasetError(f"{topic_id}.rules[{rule_pos}] must be an object")
                for field in (
                    "rule_id",
                    "title",
                    "summary",
                    "trigger",
                    "check_logic",
                    "error_type",
                ):
                    _nonempty_string(rule.get(field), label=f"{topic_id}.rules[{rule_pos}].{field}")
                rule_id = str(rule["rule_id"])
                origin = _origin(rule_id)
                if rule_id in rule_ids:
                    raise MechanismDatasetError(f"duplicate catalog rule_id: {rule_id}")
                rule_ids.add(rule_id)
                hint = rule.get("symbolic_hint")
                if not isinstance(hint, dict):
                    raise MechanismDatasetError(f"rule {rule_id} symbolic_hint must be an object")
                _require_exact_keys(
                    hint,
                    ("primitive", "canonical", "required_symbols"),
                    label=f"rule {rule_id}.symbolic_hint",
                )
                primitive = _nonempty_string(
                    hint.get("primitive"), label=f"rule {rule_id}.symbolic_hint.primitive"
                )
                if not isinstance(hint.get("canonical"), str):
                    raise MechanismDatasetError(f"rule {rule_id} canonical must be a string")
                symbols = hint.get("required_symbols")
                if not isinstance(symbols, list) or not all(
                    isinstance(symbol, str) and symbol.strip() for symbol in symbols
                ):
                    raise MechanismDatasetError(
                        f"rule {rule_id} required_symbols must be non-empty strings"
                    )
                trigger = _normalize_trigger(str(rule["trigger"]))
                if not trigger:
                    raise MechanismDatasetError(f"rule {rule_id} has an empty normalized trigger")
                topic_rules[rule_id] = {
                    "rule_id": rule_id,
                    "origin": origin,
                    "domain_id": domain_id,
                    "domain": domain_name,
                    "topic_id": topic_id,
                    "topic": topic_name,
                    "title": str(rule["title"]),
                    "summary": str(rule["summary"]),
                    "trigger": str(rule["trigger"]),
                    "normalized_trigger": trigger,
                    "check_logic": str(rule["check_logic"]),
                    "error_type": str(rule["error_type"]),
                    "symbolic_hint": {
                        "primitive": primitive,
                        "canonical": hint["canonical"],
                        "required_symbols": list(symbols),
                    },
                    "symbolic_primitive": primitive,
                    "has_symbolic_primitive": primitive != "none",
                }

            memberships: Dict[str, Dict[str, str]] = {}
            seen_clusters: set[str] = set()
            cluster_count += len(clusters)
            for cluster_pos, cluster in enumerate(clusters):
                if not isinstance(cluster, dict):
                    raise MechanismDatasetError(
                        f"{topic_id}.scenario_clusters[{cluster_pos}] must be an object"
                    )
                cluster_id = _nonempty_string(
                    cluster.get("id") or cluster.get("cluster_id"),
                    label=f"{topic_id}.scenario_clusters[{cluster_pos}].id",
                )
                cluster_name = _nonempty_string(
                    cluster.get("name") or cluster_id,
                    label=f"{topic_id}/{cluster_id}.name",
                )
                if cluster_id in seen_clusters:
                    raise MechanismDatasetError(f"duplicate cluster {topic_id}/{cluster_id}")
                seen_clusters.add(cluster_id)
                member_ids = list(cluster.get("rule_ids") or [])
                groups = cluster.get("rule_groups") or []
                if not isinstance(groups, list):
                    raise MechanismDatasetError(f"cluster {topic_id}/{cluster_id} groups must be an array")
                for group in groups:
                    if not isinstance(group, dict):
                        raise MechanismDatasetError(f"cluster {topic_id}/{cluster_id} has invalid group")
                    member_ids.extend(group.get("rule_ids") or [])
                for raw_rule_id in member_ids:
                    member_id = str(raw_rule_id or "").strip()
                    if member_id not in topic_rules:
                        raise MechanismDatasetError(
                            f"cluster {topic_id}/{cluster_id} references unknown rule {member_id!r}"
                        )
                    previous = memberships.get(member_id)
                    owner = {"cluster_id": cluster_id, "cluster": cluster_name}
                    if previous is not None and previous != owner:
                        raise MechanismDatasetError(f"rule {member_id} belongs to multiple clusters")
                    memberships[member_id] = owner
            for rule_id, record in topic_rules.items():
                owner = memberships.get(rule_id)
                if owner is None:
                    raise MechanismDatasetError(f"rule {rule_id} has no cluster ownership")
                record.update(owner)
                records.append(record)

    if metadata.get("total_domains") is not None and metadata.get("total_domains") != len(domains):
        raise MechanismDatasetError("metadata.total_domains does not match catalog")
    if metadata.get("total_topics") is not None and metadata.get("total_topics") != topic_count:
        raise MechanismDatasetError("metadata.total_topics does not match catalog")
    if metadata.get("topics_with_rules") is not None and metadata.get("topics_with_rules") != topics_with_rules:
        raise MechanismDatasetError("metadata.topics_with_rules does not match catalog")
    if metadata.get("total_executable_rules") is not None and metadata.get("total_executable_rules") != len(records):
        raise MechanismDatasetError("metadata.total_executable_rules does not match catalog")
    if metadata.get("total_scenario_clusters") is not None and metadata.get("total_scenario_clusters") != cluster_count:
        raise MechanismDatasetError("metadata.total_scenario_clusters does not match catalog")
    return records, metadata


def _assign_trigger_scope_proxy(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for raw in records:
        record = dict(raw)
        grouped.setdefault((record["domain_id"], record["origin"]), []).append(record)
    output: List[Dict[str, Any]] = []
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
            row["trigger_scope_proxy"] = (
                "broad_proxy" if index < broad_count else "narrow_proxy"
            )
            output.append(row)
    return output


def _select_rules(records: Sequence[Mapping[str, Any]], *, catalog_sha256: str) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for domain_id, origins in CELL_QUOTAS.items():
        for origin, (total, broad, narrow, none_broad, none_narrow) in origins.items():
            if total != broad + narrow:
                raise MechanismDatasetError(f"invalid registered quota for {domain_id}/{origin}")
            for scope, scope_total, none_total in (
                ("broad_proxy", broad, none_broad),
                ("narrow_proxy", narrow, none_narrow),
            ):
                for has_symbolic, amount in (
                    (False, none_total),
                    (True, scope_total - none_total),
                ):
                    candidates = [
                        dict(row)
                        for row in records
                        if row["domain_id"] == domain_id
                        and row["origin"] == origin
                        and row["trigger_scope_proxy"] == scope
                        and row["has_symbolic_primitive"] is has_symbolic
                    ]
                    candidates.sort(
                        key=lambda row: _selection_key(
                            catalog_sha256=catalog_sha256, rule_id=row["rule_id"]
                        )
                    )
                    if len(candidates) < amount:
                        raise MechanismDatasetError(
                            "pre-registered stratum is insufficient; no fallback is allowed: "
                            f"{domain_id}/{origin}/{scope}/has_symbolic={has_symbolic} "
                            f"needs {amount}, found {len(candidates)}"
                        )
                    selected.extend(candidates[:amount])
    for row in selected:
        row["selection_key_sha256"] = _selection_key(
            catalog_sha256=catalog_sha256, rule_id=row["rule_id"]
        )
    selected.sort(key=lambda row: (row["selection_key_sha256"], row["rule_id"]))
    if len(selected) != 60 or len({row["rule_id"] for row in selected}) != 60:
        raise MechanismDatasetError("registered selection must contain exactly 60 unique rules")
    for index, row in enumerate(selected):
        row["selection_index"] = index
        row["fifth_mechanism_subtype"] = (
            "self_corrected" if index < 30 else "insufficient_information"
        )
    return selected


def _plan_without_sha(plan: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "plan_sha256"}


def _trigger_scope_proxy_metadata() -> Dict[str, Any]:
    return {
        "version": TRIGGER_SCOPE_PROXY_VERSION,
        "labels": ["broad_proxy", "narrow_proxy"],
        "definition": (
            "NFC-normalize and collapse whitespace; within each domain/origin sort by "
            "Unicode code-point length, normalized trigger, and rule_id; first ceil(n/2) "
            "is broad_proxy. This is a lexical length proxy, not semantic breadth."
        ),
        "semantic_breadth_claim": False,
    }


def _cell_quota_metadata() -> Dict[str, Any]:
    return {
        domain: {
            origin: {
                "total": values[0],
                "broad_proxy": values[1],
                "narrow_proxy": values[2],
                "primitive_none_broad_proxy": values[3],
                "primitive_none_narrow_proxy": values[4],
            }
            for origin, values in origins.items()
        }
        for domain, origins in CELL_QUOTAS.items()
    }


def _selection_summary(rules: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "rule_count": len(rules),
        "origin_counts": dict(sorted(Counter(row["origin"] for row in rules).items())),
        "trigger_scope_proxy_counts": dict(
            sorted(Counter(row["trigger_scope_proxy"] for row in rules).items())
        ),
        "has_symbolic_primitive_counts": {
            str(key).lower(): value
            for key, value in sorted(
                Counter(row["has_symbolic_primitive"] for row in rules).items()
            )
        },
        "fifth_mechanism_subtype_counts": dict(
            sorted(Counter(row["fifth_mechanism_subtype"] for row in rules).items())
        ),
        "domain_counts": dict(sorted(Counter(row["domain_id"] for row in rules).items())),
    }


def prepare_mechanism_plan(
    *,
    catalog_path: Path,
    plan_path: Path,
    expected_catalog_sha256: str = EXPECTED_CATALOG_SHA256,
) -> Dict[str, Any]:
    if plan_path.exists():
        raise MechanismDatasetError(f"rule plan is immutable and already exists: {plan_path}")
    catalog_sha256 = _sha256_file(catalog_path)
    if catalog_sha256 != expected_catalog_sha256:
        raise MechanismDatasetError(
            f"catalog SHA256 mismatch: expected {expected_catalog_sha256}, got {catalog_sha256}"
        )
    catalog = _load_json(catalog_path, label="unified catalog")
    records, metadata = _catalog_records(catalog)
    selected = _select_rules(
        _assign_trigger_scope_proxy(records), catalog_sha256=catalog_sha256
    )
    rules: List[Dict[str, Any]] = []
    for row in selected:
        rules.append(
            {
                key: row[key]
                for key in (
                    "selection_index",
                    "selection_key_sha256",
                    "rule_id",
                    "origin",
                    "domain_id",
                    "domain",
                    "topic_id",
                    "topic",
                    "cluster_id",
                    "cluster",
                    "title",
                    "summary",
                    "trigger",
                    "check_logic",
                    "error_type",
                    "symbolic_hint",
                    "symbolic_primitive",
                    "has_symbolic_primitive",
                    "trigger_scope_proxy",
                    "fifth_mechanism_subtype",
                )
            }
        )
    summary = _selection_summary(rules)
    plan: Dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_type": PLAN_TYPE,
        "seed": FIXED_SEED,
        "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
        "trigger_scope_proxy": _trigger_scope_proxy_metadata(),
        "catalog": {
            "sha256": catalog_sha256,
            "size_bytes": catalog_path.stat().st_size,
            "catalog_type": metadata.get("catalog_type"),
            "version": metadata.get("version"),
            "rule_count": len(records),
        },
        "cell_quotas": _cell_quota_metadata(),
        "ordered_rule_ids": [row["rule_id"] for row in rules],
        "rules": rules,
        "summary": summary,
    }
    plan["plan_sha256"] = _object_sha256(_plan_without_sha(plan))
    _atomic_write_json(plan_path, plan)
    return plan


def _validate_plan(plan: Any, *, catalog_path: Path) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        raise MechanismDatasetError("rule plan must be an object")
    _require_exact_keys(
        plan,
        (
            "schema_version",
            "plan_type",
            "seed",
            "selection_algorithm_version",
            "trigger_scope_proxy",
            "catalog",
            "cell_quotas",
            "ordered_rule_ids",
            "rules",
            "summary",
            "plan_sha256",
        ),
        label="rule plan",
    )
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION or plan.get("plan_type") != PLAN_TYPE:
        raise MechanismDatasetError("unsupported rule plan schema/type")
    if plan.get("seed") != FIXED_SEED:
        raise MechanismDatasetError("rule plan seed drifted")
    if plan.get("selection_algorithm_version") != SELECTION_ALGORITHM_VERSION:
        raise MechanismDatasetError("rule plan selection algorithm drifted")
    if plan.get("trigger_scope_proxy") != _trigger_scope_proxy_metadata():
        raise MechanismDatasetError("rule plan trigger_scope_proxy metadata drifted")
    if plan.get("cell_quotas") != _cell_quota_metadata():
        raise MechanismDatasetError("rule plan cell quotas drifted")
    expected_plan_sha = _object_sha256(_plan_without_sha(plan))
    if plan.get("plan_sha256") != expected_plan_sha:
        raise MechanismDatasetError("rule plan SHA256 is invalid")
    catalog_info = plan.get("catalog")
    if not isinstance(catalog_info, dict):
        raise MechanismDatasetError("rule plan catalog binding is missing")
    actual_catalog_sha = _sha256_file(catalog_path)
    if actual_catalog_sha != EXPECTED_CATALOG_SHA256:
        raise MechanismDatasetError(
            "current catalog does not match the pre-registered catalog SHA256"
        )
    if catalog_info.get("sha256") != actual_catalog_sha:
        raise MechanismDatasetError("current catalog does not match frozen rule plan")
    catalog = _load_json(catalog_path, label="unified catalog")
    records, metadata = _catalog_records(catalog)
    expected_catalog_info = {
        "sha256": actual_catalog_sha,
        "size_bytes": catalog_path.stat().st_size,
        "catalog_type": metadata.get("catalog_type"),
        "version": metadata.get("version"),
        "rule_count": len(records),
    }
    if catalog_info != expected_catalog_info:
        raise MechanismDatasetError("rule plan catalog metadata drifted")
    current = _select_rules(
        _assign_trigger_scope_proxy(records), catalog_sha256=actual_catalog_sha
    )
    rules = plan.get("rules")
    ordered_ids = plan.get("ordered_rule_ids")
    if not isinstance(rules, list) or not all(isinstance(row, dict) for row in rules):
        raise MechanismDatasetError("rule plan rules must be an array of objects")
    if ordered_ids != [row["rule_id"] for row in rules]:
        raise MechanismDatasetError("rule plan ordered IDs do not match rule records")
    if ordered_ids != [row["rule_id"] for row in current]:
        raise MechanismDatasetError("rule plan selection does not reproduce from catalog")
    # Bind every copied rule/ownership field, not merely IDs.
    current_by_id = {row["rule_id"]: row for row in current}
    expected_rule_keys = {
        "selection_index",
        "selection_key_sha256",
        "rule_id",
        "origin",
        "domain_id",
        "domain",
        "topic_id",
        "topic",
        "cluster_id",
        "cluster",
        "title",
        "summary",
        "trigger",
        "check_logic",
        "error_type",
        "symbolic_hint",
        "symbolic_primitive",
        "has_symbolic_primitive",
        "trigger_scope_proxy",
        "fifth_mechanism_subtype",
    }
    for row in rules:
        _require_exact_keys(row, expected_rule_keys, label="rule plan rule")
        if not isinstance(row.get("has_symbolic_primitive"), bool):
            raise MechanismDatasetError(
                f"rule plan has_symbolic_primitive is not boolean for {row.get('rule_id')}"
            )
        source = current_by_id.get(row.get("rule_id"))
        if source is None:
            raise MechanismDatasetError("rule plan references unknown catalog rule")
        for key in (
            "selection_index",
            "selection_key_sha256",
            "origin",
            "domain_id",
            "domain",
            "topic_id",
            "topic",
            "cluster_id",
            "cluster",
            "title",
            "summary",
            "trigger",
            "check_logic",
            "error_type",
            "symbolic_hint",
            "symbolic_primitive",
            "has_symbolic_primitive",
            "trigger_scope_proxy",
            "fifth_mechanism_subtype",
        ):
            if row.get(key) != source.get(key):
                raise MechanismDatasetError(f"rule plan field {key} drifted for {row.get('rule_id')}")
    if plan.get("summary") != _selection_summary(rules):
        raise MechanismDatasetError("rule plan summary drifted")
    return plan


def _expected_for(mechanism: str, subtype: str) -> Dict[str, Any]:
    if mechanism == "true_violation":
        return {
            "applicability": True,
            "current_violation": True,
            "publish": True,
            "consistency_status": "confirmed_violation",
        }
    if mechanism == "applicable_correct":
        return {
            "applicability": True,
            "current_violation": False,
            "publish": False,
            "consistency_status": None,
        }
    if mechanism == "symbol_overlap_inapplicable":
        return {
            "applicability": False,
            "current_violation": False,
            "publish": False,
            "consistency_status": None,
        }
    if mechanism == "equivalent_alternative":
        return {
            "applicability": True,
            "current_violation": False,
            "publish": False,
            "consistency_status": "equivalent_or_alternative",
        }
    if mechanism == "insufficient_or_self_corrected" and subtype == "self_corrected":
        return {
            "applicability": True,
            "current_violation": False,
            "publish": False,
            "consistency_status": "self_corrected",
        }
    if mechanism == "insufficient_or_self_corrected" and subtype == "insufficient_information":
        return {
            "applicability": True,
            "current_violation": False,
            "publish": False,
            "consistency_status": "uncertain",
        }
    raise MechanismDatasetError(f"invalid mechanism/subtype: {mechanism}/{subtype}")


def _prompt(rule: Mapping[str, Any]) -> Tuple[str, str]:
    subtype = rule["fifth_mechanism_subtype"]
    skeleton = {
        "cases": [
            {
                "mechanism": mechanism,
                "mechanism_subtype": subtype if index == 4 else "none",
                "question": "REPLACE_WITH_SYNTHETIC_QUESTION",
                "context": "REPLACE_WITH_SYNTHETIC_CONTEXT",
                "prediction": "REPLACE_WITH_SYNTHETIC_STUDENT_SOLUTION",
                "expected": _expected_for(
                    mechanism, subtype if index == 4 else "none"
                ),
                "gt_evidence": {
                    "applicability_spans": [],
                    "violation_spans": [],
                    "superseded_claim_spans": [],
                    "correction_spans": [],
                },
            }
            for index, mechanism in enumerate(MECHANISMS)
        ]
    }
    system = (
        "You construct synthetic physics-competition verifier cases from one supplied rule. "
        "Use no external benchmark examples. Return one strict JSON object only: the first output "
        "character must be { and the last must be }. Never use Markdown or ``` fences. Every "
        "evidence quote must be copied exactly from its declared source with exact 0-based "
        "half-open offsets."
    )
    user = (
        "Target catalog rule:\n"
        + json.dumps(
            {
                "rule_id": rule["rule_id"],
                "title": rule["title"],
                "summary": rule["summary"],
                "trigger": rule["trigger"],
                "check_logic": rule["check_logic"],
                "error_type": rule["error_type"],
                "symbolic_hint": rule["symbolic_hint"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n\nProduce exactly five cases in this exact order: "
        + ", ".join(MECHANISMS)
        + f". The last case subtype must be {subtype}. "
        "The mechanism field is the case-type enum shown below, NEVER the target rule_id. "
        "Use mechanism_subtype='none' for the first four. Include expected exactly as specified: "
        "true_violation=(applicable true,current violation true,publish true,confirmed_violation); "
        "applicable_correct=(true,false,false,null); symbol_overlap_inapplicable=(false,false,false,null); "
        "equivalent_alternative=(true,false,false,equivalent_or_alternative); last=(true,false,false,"
        + ("self_corrected" if subtype == "self_corrected" else "uncertain")
        + "). Preserve every key and every prefilled enum/expected value in this exact skeleton; "
        "replace only placeholder text and evidence arrays:\n"
        + json.dumps(
            skeleton,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        "A span is exactly {source,quote,start_char,end_char}; applicability source is question/context "
        "and all other sources are prediction. Do not add keys. Cases must be distinct. "
        "For true_violation include applicability and violation evidence. For applicable positive "
        "cases include applicability evidence. For symbol_overlap_inapplicable applicability evidence "
        "must be empty. For self_corrected include ordered superseded and correction spans; otherwise "
        "those two arrays are empty."
    )
    return system, user


def _validate_span(
    span: Any,
    *,
    sources: Mapping[str, str],
    allowed: set[str],
    label: str,
    repair_offsets_from_unique_quote: bool = False,
) -> Dict[str, Any]:
    if not isinstance(span, dict):
        raise MechanismDatasetError(f"{label} must be an object")
    _require_exact_keys(span, ("source", "quote", "start_char", "end_char"), label=label)
    source = span.get("source")
    quote = span.get("quote")
    if source not in allowed:
        raise MechanismDatasetError(f"{label}.source must be one of {sorted(allowed)}")
    if not isinstance(quote, str) or not quote:
        raise MechanismDatasetError(f"{label}.quote must be non-empty")
    start = _strict_int(span.get("start_char"), label=f"{label}.start_char")
    end = _strict_int(span.get("end_char"), label=f"{label}.end_char")
    text = sources[source]
    if text.count(quote) != 1:
        raise MechanismDatasetError(f"{label}.quote must occur exactly once in {source}")
    exact_start = text.index(quote)
    exact_end = exact_start + len(quote)
    if start != exact_start or end != exact_end:
        if not repair_offsets_from_unique_quote:
            raise MechanismDatasetError(
                f"{label} quote/offset does not exactly match {source}"
            )
        start, end = exact_start, exact_end
    return {"source": source, "quote": quote, "start_char": start, "end_char": end}


def _validate_generated_payload(
    payload: Any,
    *,
    rule: Mapping[str, Any],
    existing_content_hashes: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        raise MechanismDatasetError("generator response must be an object")
    _require_exact_keys(payload, ("cases",), label="generator response")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 5:
        raise MechanismDatasetError("generator response must contain exactly five cases")
    normalized: List[Dict[str, Any]] = []
    content_hashes = set(existing_content_hashes or set())
    within: set[str] = set()
    for index, case in enumerate(cases):
        label = f"cases[{index}]"
        if not isinstance(case, dict):
            raise MechanismDatasetError(f"{label} must be an object")
        _require_exact_keys(
            case,
            (
                "mechanism",
                "mechanism_subtype",
                "question",
                "context",
                "prediction",
                "expected",
                "gt_evidence",
            ),
            label=label,
        )
        mechanism = case.get("mechanism")
        if mechanism != MECHANISMS[index]:
            raise MechanismDatasetError(f"{label}.mechanism must be {MECHANISMS[index]}")
        subtype = case.get("mechanism_subtype")
        required_subtype = rule["fifth_mechanism_subtype"] if index == 4 else "none"
        if subtype != required_subtype:
            raise MechanismDatasetError(f"{label}.mechanism_subtype must be {required_subtype}")
        question = _nonempty_string(case.get("question"), label=f"{label}.question")
        context = _nonempty_string(case.get("context"), label=f"{label}.context")
        prediction = _nonempty_string(case.get("prediction"), label=f"{label}.prediction")
        if max(len(question), len(context), len(prediction)) > 8000:
            raise MechanismDatasetError(f"{label} text exceeds 8000 characters")
        content_sha = _object_sha256([question, context, prediction])
        if content_sha in within or content_sha in content_hashes:
            raise MechanismDatasetError(f"{label} duplicates another generated case")
        within.add(content_sha)
        expected = case.get("expected")
        if not isinstance(expected, dict):
            raise MechanismDatasetError(f"{label}.expected must be an object")
        _require_exact_keys(
            expected,
            ("applicability", "current_violation", "publish", "consistency_status"),
            label=f"{label}.expected",
        )
        for field in ("applicability", "current_violation", "publish"):
            if not isinstance(expected.get(field), bool):
                raise MechanismDatasetError(f"{label}.expected.{field} must be a boolean")
        consistency_status = expected.get("consistency_status")
        if consistency_status is not None and (
            not isinstance(consistency_status, str)
            or consistency_status not in CONSISTENCY_STATUSES
        ):
            raise MechanismDatasetError(
                f"{label}.expected.consistency_status is invalid"
            )
        required_expected = _expected_for(str(mechanism), str(subtype))
        if expected != required_expected:
            raise MechanismDatasetError(f"{label}.expected does not match the frozen mechanism target")
        evidence = case.get("gt_evidence")
        if not isinstance(evidence, dict):
            raise MechanismDatasetError(f"{label}.gt_evidence must be an object")
        evidence_keys = (
            "applicability_spans",
            "violation_spans",
            "superseded_claim_spans",
            "correction_spans",
        )
        _require_exact_keys(evidence, evidence_keys, label=f"{label}.gt_evidence")
        sources = {"question": question, "context": context, "prediction": prediction}
        normalized_evidence: Dict[str, List[Dict[str, Any]]] = {}
        for field in evidence_keys:
            spans = evidence.get(field)
            if not isinstance(spans, list):
                raise MechanismDatasetError(f"{label}.gt_evidence.{field} must be an array")
            allowed = {"question", "context"} if field == "applicability_spans" else {"prediction"}
            clean = [
                _validate_span(
                    span,
                    sources=sources,
                    allowed=allowed,
                    label=f"{label}.gt_evidence.{field}[{span_index}]",
                    repair_offsets_from_unique_quote=True,
                )
                for span_index, span in enumerate(spans)
            ]
            fingerprints = [_canonical_json(span) for span in clean]
            if len(fingerprints) != len(set(fingerprints)):
                raise MechanismDatasetError(f"{label}.gt_evidence.{field} contains duplicate spans")
            normalized_evidence[field] = clean
        if required_expected["applicability"] and not normalized_evidence["applicability_spans"]:
            raise MechanismDatasetError(f"{label} requires applicability evidence")
        if not required_expected["applicability"] and normalized_evidence["applicability_spans"]:
            raise MechanismDatasetError(f"{label} must not assert applicability evidence")
        if mechanism == "true_violation":
            if not normalized_evidence["violation_spans"]:
                raise MechanismDatasetError(f"{label} requires violation evidence")
        elif normalized_evidence["violation_spans"]:
            raise MechanismDatasetError(f"{label} must not contain violation evidence")
        is_self_corrected = subtype == "self_corrected"
        if is_self_corrected:
            superseded = normalized_evidence["superseded_claim_spans"]
            correction = normalized_evidence["correction_spans"]
            if not superseded or not correction:
                raise MechanismDatasetError(f"{label} self-correction evidence is incomplete")
            if max(span["end_char"] for span in superseded) > min(
                span["start_char"] for span in correction
            ):
                raise MechanismDatasetError(f"{label} correction must follow the superseded claim")
        elif normalized_evidence["superseded_claim_spans"] or normalized_evidence["correction_spans"]:
            raise MechanismDatasetError(f"{label} must not contain self-correction evidence")
        normalized.append(
            {
                "mechanism": mechanism,
                "mechanism_subtype": subtype,
                "question": question,
                "context": context,
                "prediction": prediction,
                "expected": required_expected,
                "gt_evidence": normalized_evidence,
                "case_content_sha256": content_sha,
            }
        )
    return normalized


def _target_rule_projection(rule: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: rule[key]
        for key in (
            "rule_id",
            "domain_id",
            "domain",
            "topic_id",
            "topic",
            "cluster_id",
            "origin",
            "symbolic_primitive",
            "has_symbolic_primitive",
            "trigger_scope_proxy",
        )
    }


def _case_id(rule: Mapping[str, Any], mechanism: str) -> str:
    return f"p3::{rule['rule_id']}::{mechanism}"


def _validate_dataset_row(row: Any, *, rule: Mapping[str, Any]) -> None:
    if not isinstance(row, dict):
        raise MechanismDatasetError("dataset row must be an object")
    _require_exact_keys(
        row,
        (
            "schema_version",
            "id",
            "question",
            "context",
            "prediction",
            "target_rule_id",
            "target_rule",
            "mechanism",
            "mechanism_subtype",
            "expected",
            "gt_evidence",
            "case_content_sha256",
            "gt_provenance",
        ),
        label="dataset row",
    )
    if row.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise MechanismDatasetError("dataset row schema_version is invalid")
    mechanism = row.get("mechanism")
    if mechanism not in MECHANISMS:
        raise MechanismDatasetError("dataset row mechanism is invalid")
    if row.get("id") != _case_id(rule, str(mechanism)):
        raise MechanismDatasetError("dataset row ID does not match target/mechanism")
    if row.get("target_rule_id") != rule.get("rule_id"):
        raise MechanismDatasetError("dataset row target_rule_id does not match plan")
    if row.get("target_rule") != _target_rule_projection(rule):
        raise MechanismDatasetError("dataset row target_rule does not match plan")
    target_rule = row.get("target_rule")
    if not isinstance(target_rule, dict):
        raise MechanismDatasetError("dataset row target_rule must be an object")
    _require_exact_keys(
        target_rule,
        (
            "rule_id",
            "domain_id",
            "domain",
            "topic_id",
            "topic",
            "cluster_id",
            "origin",
            "symbolic_primitive",
            "has_symbolic_primitive",
            "trigger_scope_proxy",
        ),
        label="dataset row.target_rule",
    )
    if not isinstance(target_rule.get("has_symbolic_primitive"), bool):
        raise MechanismDatasetError(
            "dataset row.target_rule.has_symbolic_primitive must be boolean"
        )
    for field in ("question", "context", "prediction"):
        _nonempty_string(row.get(field), label=f"dataset row.{field}")
    expected_subtype = (
        rule["fifth_mechanism_subtype"]
        if mechanism == "insufficient_or_self_corrected"
        else "none"
    )
    if row.get("mechanism_subtype") != expected_subtype:
        raise MechanismDatasetError("dataset row mechanism_subtype is invalid")
    expected = row.get("expected")
    if not isinstance(expected, dict):
        raise MechanismDatasetError("dataset row expected must be an object")
    _require_exact_keys(
        expected,
        ("applicability", "current_violation", "publish", "consistency_status"),
        label="dataset row.expected",
    )
    if not all(
        isinstance(expected.get(field), bool)
        for field in ("applicability", "current_violation", "publish")
    ):
        raise MechanismDatasetError("dataset row expected boolean fields are invalid")
    if expected != _expected_for(str(mechanism), expected_subtype):
        raise MechanismDatasetError("dataset row expected target is invalid")
    content_sha = _object_sha256(
        [row.get("question"), row.get("context"), row.get("prediction")]
    )
    if row.get("case_content_sha256") != content_sha:
        raise MechanismDatasetError("dataset row case_content_sha256 is invalid")
    evidence = row.get("gt_evidence")
    if not isinstance(evidence, dict):
        raise MechanismDatasetError("dataset row gt_evidence must be an object")
    evidence_keys = (
        "applicability_spans",
        "violation_spans",
        "superseded_claim_spans",
        "correction_spans",
    )
    _require_exact_keys(evidence, evidence_keys, label="dataset row.gt_evidence")
    sources = {
        "question": row["question"],
        "context": row["context"],
        "prediction": row["prediction"],
    }
    for field in evidence_keys:
        spans = evidence.get(field)
        if not isinstance(spans, list):
            raise MechanismDatasetError(f"dataset row.gt_evidence.{field} must be an array")
        allowed = {"question", "context"} if field == "applicability_spans" else {"prediction"}
        for index, span in enumerate(spans):
            _validate_span(
                span,
                sources=sources,
                allowed=allowed,
                label=f"dataset row.gt_evidence.{field}[{index}]",
            )
    provenance = row.get("gt_provenance")
    if not isinstance(provenance, dict):
        raise MechanismDatasetError("dataset row gt_provenance must be an object")
    _require_exact_keys(
        provenance,
        (
            "generator_model",
            "prompt_version",
            "plan_sha256",
            "raw_response_sha256",
            "validation_status",
        ),
        label="dataset row.gt_provenance",
    )
    if provenance.get("validation_status") != "schema_validated":
        raise MechanismDatasetError("dataset row validation_status is invalid")
    for field in ("generator_model", "prompt_version", "plan_sha256", "raw_response_sha256"):
        _nonempty_string(provenance.get(field), label=f"dataset row.gt_provenance.{field}")


def _dataset_rows(
    states: Mapping[str, Mapping[str, Any]],
    rules: Sequence[Mapping[str, Any]],
    *,
    model: str,
    plan_sha256: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rule in rules:
        state = states.get(str(rule["rule_id"]))
        if not isinstance(state, Mapping) or state.get("status") != "complete":
            continue
        cases = state.get("cases")
        if not isinstance(cases, list):
            raise MechanismDatasetError(f"checkpoint cases missing for {rule['rule_id']}")
        for case in cases:
            row = {
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "id": _case_id(rule, str(case["mechanism"])),
                    "question": case["question"],
                    "context": case["context"],
                    "prediction": case["prediction"],
                    "target_rule_id": rule["rule_id"],
                    "target_rule": _target_rule_projection(rule),
                    "mechanism": case["mechanism"],
                    "mechanism_subtype": case["mechanism_subtype"],
                    "expected": case["expected"],
                    "gt_evidence": case["gt_evidence"],
                    "case_content_sha256": case["case_content_sha256"],
                    "gt_provenance": {
                        "generator_model": model,
                        "prompt_version": PROMPT_VERSION,
                        "plan_sha256": plan_sha256,
                        "raw_response_sha256": state["raw_response_sha256"],
                        "validation_status": "schema_validated",
                    },
                }
            _validate_dataset_row(row, rule=rule)
            rows.append(row)
    return rows


def _candidate_trace_row(dataset_row: Mapping[str, Any], rule: Mapping[str, Any]) -> Dict[str, Any]:
    score_kind = "fixed_control_0_1"
    return {
        "schema_version": CANDIDATE_TRACE_SCHEMA_VERSION,
        "id": dataset_row["id"],
        "target_rule_id": rule["rule_id"],
        "candidate_source": "frozen_target_binding",
        "verifier": "unified_v2_frozen_target_binding",
        "topic": rule["topic"],
        "unified_retrieval_mode": "target_binding",
        "selection_strategy": "target_rule_binding",
        "retrieval_score_kind": score_kind,
        "semantic_selection_error": "",
        "semantic_failed_stage": "",
        "terminal_stage": "target_rule_binding",
        "empty_reason": "",
        "retrieved_domains": [
            {
                "domain_id": rule["domain_id"],
                "domain": rule["domain"],
                "score": 1.0,
                "score_kind": score_kind,
            }
        ],
        "retrieved_topics": [
            {
                "domain": rule["domain"],
                "topic_id": rule["topic_id"],
                "topic": rule["topic"],
                "score": 1.0,
                "score_kind": score_kind,
            }
        ],
        "retrieved_clusters": [
            {
                "domain": rule["domain"],
                "topic_id": rule["topic_id"],
                "topic": rule["topic"],
                "cluster_id": rule["cluster_id"],
                "cluster": rule["cluster"],
                "score": 1.0,
                "score_kind": score_kind,
            }
        ],
        "retrieved_rules": [
            {
                "rule_id": rule["rule_id"],
                "domain": rule["domain"],
                "topic_id": rule["topic_id"],
                "topic": rule["topic"],
                "cluster_id": rule["cluster_id"],
                "cluster": rule["cluster"],
                "title": rule["title"],
                "scope": "domain",
                "candidate_source": "frozen_target_binding",
                "score": 1.0,
                "score_kind": score_kind,
                "partial": False,
                "executable": True,
                "publish_gate": {
                    "publishable": True,
                    "reasons": [],
                    "score": 1.0,
                    "min_publish_score": 0.0,
                    "score_kind": score_kind,
                    "selection_strategy": "target_rule_binding",
                },
                "evidence": {
                    "binding_reason": "pre_registered_frozen_target_rule",
                    "target_rule_id": rule["rule_id"],
                },
            }
        ],
    }


def _candidate_trace_rows(
    dataset_rows: Sequence[Mapping[str, Any]], rules: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    by_id = {str(rule["rule_id"]): rule for rule in rules}
    output: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for row in dataset_rows:
        target = row.get("target_rule")
        if not isinstance(target, Mapping):
            raise MechanismDatasetError("dataset target_rule is missing")
        target_rule_id = row.get("target_rule_id")
        if not isinstance(target_rule_id, str) or not target_rule_id:
            raise MechanismDatasetError("dataset target_rule_id must be a non-empty string")
        if target.get("rule_id") != target_rule_id:
            raise MechanismDatasetError(
                "dataset target_rule_id does not match target_rule.rule_id"
            )
        rule = by_id.get(target_rule_id)
        if rule is None:
            raise MechanismDatasetError("dataset target rule is outside frozen plan")
        _validate_dataset_row(row, rule=rule)
        row_id = str(row["id"])
        content_sha = str(row["case_content_sha256"])
        if row_id in seen_ids:
            raise MechanismDatasetError(f"duplicate dataset ID: {row_id}")
        if content_sha in seen_content:
            raise MechanismDatasetError("duplicate dataset case content")
        seen_ids.add(row_id)
        seen_content.add(content_sha)
        trace_row = _candidate_trace_row(row, rule)
        if trace_row["target_rule_id"] != target_rule_id:
            raise MechanismDatasetError("candidate trace target does not match dataset target")
        output.append(trace_row)
    return output


def _load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _source_identity() -> Dict[str, Any]:
    state = capture_source_state(PROJECT_ROOT)
    git = state.get("git") if isinstance(state.get("git"), dict) else {}
    source_tree = (
        state.get("source_tree")
        if isinstance(state.get("source_tree"), dict)
        else {}
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


def _transport_settings() -> Dict[str, Any]:
    endpoint_source = "default"
    endpoint = "https://api.openai.com/v1"
    for env_name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        raw = str(os.getenv(env_name) or "").strip()
        if raw:
            endpoint_source = env_name
            endpoint = raw.rstrip("/")
            break
    parsed = urlsplit(endpoint)
    hostname = str(parsed.hostname or "").lower()
    is_loopback = hostname == "localhost" or hostname.endswith(".localhost")
    if hostname:
        try:
            is_loopback = is_loopback or ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            pass
    raw_timeout = str(os.getenv("PHYSICSVERIFIER_LLM_TIMEOUT_SEC") or "").strip()
    raw_retries = str(os.getenv("PHYSICSVERIFIER_LLM_MAX_RETRIES") or "").strip()
    try:
        timeout_sec = float(raw_timeout) if raw_timeout else 120.0
    except ValueError as exc:
        raise MechanismDatasetError(
            "PHYSICSVERIFIER_LLM_TIMEOUT_SEC must be a positive finite number"
        ) from exc
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise MechanismDatasetError(
            "PHYSICSVERIFIER_LLM_TIMEOUT_SEC must be a positive finite number"
        )
    try:
        sdk_max_retries = int(raw_retries) if raw_retries else 2
    except ValueError as exc:
        raise MechanismDatasetError(
            "PHYSICSVERIFIER_LLM_MAX_RETRIES must be a non-negative integer"
        ) from exc
    if sdk_max_retries < 0 or str(sdk_max_retries) != (raw_retries or str(sdk_max_retries)):
        raise MechanismDatasetError(
            "PHYSICSVERIFIER_LLM_MAX_RETRIES must be a canonical non-negative integer"
        )
    return {
        "endpoint": endpoint,
        "endpoint_source": endpoint_source,
        "endpoint_scheme": str(parsed.scheme or "").lower(),
        "endpoint_is_loopback": bool(is_loopback),
        "timeout_sec": float(timeout_sec),
        "sdk_max_retries": sdk_max_retries,
    }


def _api_transport_identity() -> Dict[str, Any]:
    settings = _transport_settings()
    return {
        "transport": "openai_compatible_chat_completions",
        "endpoint_source": settings["endpoint_source"],
        "endpoint_sha256": hashlib.sha256(
            str(settings["endpoint"]).encode("utf-8")
        ).hexdigest(),
        "endpoint_scheme": settings["endpoint_scheme"],
        "endpoint_is_loopback": settings["endpoint_is_loopback"],
        "response_format": "json_object",
        "timeout_sec": settings["timeout_sec"],
        "sdk_max_retries": settings["sdk_max_retries"],
    }


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise MechanismDatasetError(f"{label} must be a lowercase SHA256 hex string")
    return value


def _validate_source_identity(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise MechanismDatasetError("source_identity must be an object")
    _require_exact_keys(
        value,
        (
            "git_available",
            "git_head",
            "git_branch",
            "git_dirty",
            "git_status_sha256",
            "git_tracked_diff_sha256",
            "source_tree_sha256",
            "source_file_count",
        ),
        label="source_identity",
    )
    if not isinstance(value.get("git_available"), bool) or not isinstance(
        value.get("git_dirty"), bool
    ):
        raise MechanismDatasetError("source_identity Git flags must be boolean")
    source_file_count = _strict_int(
        value.get("source_file_count"), label="source_identity.source_file_count"
    )
    if source_file_count < 1:
        raise MechanismDatasetError("source_identity.source_file_count must be positive")
    _require_sha256(value.get("source_tree_sha256"), label="source_identity.source_tree_sha256")
    if value["git_available"]:
        git_head = value.get("git_head")
        if not isinstance(git_head, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", git_head):
            raise MechanismDatasetError("source_identity.git_head is invalid")
        _nonempty_string(value.get("git_branch"), label="source_identity.git_branch")
        _require_sha256(
            value.get("git_status_sha256"), label="source_identity.git_status_sha256"
        )
        _require_sha256(
            value.get("git_tracked_diff_sha256"),
            label="source_identity.git_tracked_diff_sha256",
        )
    else:
        for field in (
            "git_head",
            "git_branch",
            "git_status_sha256",
            "git_tracked_diff_sha256",
        ):
            if value.get(field) != "":
                raise MechanismDatasetError(
                    f"source_identity.{field} must be empty when Git is unavailable"
                )
    return value


def _validate_runtime_identity(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise MechanismDatasetError("runtime_identity must be an object")
    _require_exact_keys(
        value,
        (
            "executable",
            "python_version",
            "prefix",
            "is_conda",
            "conda_env",
            "package_count",
            "package_set_sha256",
        ),
        label="runtime_identity",
    )
    for field in ("executable", "python_version", "prefix", "conda_env"):
        _nonempty_string(value.get(field), label=f"runtime_identity.{field}")
    if not isinstance(value.get("is_conda"), bool):
        raise MechanismDatasetError("runtime_identity.is_conda must be boolean")
    package_count = _strict_int(
        value.get("package_count"), label="runtime_identity.package_count"
    )
    if package_count < 1:
        raise MechanismDatasetError("runtime_identity.package_count must be positive")
    _require_sha256(
        value.get("package_set_sha256"), label="runtime_identity.package_set_sha256"
    )
    return value


def _validate_api_transport_identity(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise MechanismDatasetError("api_transport_identity must be an object")
    _require_exact_keys(
        value,
        (
            "transport",
            "endpoint_source",
            "endpoint_sha256",
            "endpoint_scheme",
            "endpoint_is_loopback",
            "response_format",
            "timeout_sec",
            "sdk_max_retries",
        ),
        label="api_transport_identity",
    )
    if value.get("transport") != "openai_compatible_chat_completions":
        raise MechanismDatasetError("api_transport_identity.transport is invalid")
    if value.get("endpoint_source") not in {
        "default",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    }:
        raise MechanismDatasetError("api_transport_identity.endpoint_source is invalid")
    _require_sha256(
        value.get("endpoint_sha256"), label="api_transport_identity.endpoint_sha256"
    )
    if value.get("endpoint_scheme") not in {"http", "https"}:
        raise MechanismDatasetError("api_transport_identity.endpoint_scheme is invalid")
    if not isinstance(value.get("endpoint_is_loopback"), bool):
        raise MechanismDatasetError(
            "api_transport_identity.endpoint_is_loopback must be boolean"
        )
    if value.get("response_format") != "json_object":
        raise MechanismDatasetError("api_transport_identity.response_format is invalid")
    timeout_sec = value.get("timeout_sec")
    if (
        not isinstance(timeout_sec, float)
        or not math.isfinite(timeout_sec)
        or timeout_sec <= 0
    ):
        raise MechanismDatasetError(
            "api_transport_identity.timeout_sec must be a positive canonical float"
        )
    retries = _strict_int(
        value.get("sdk_max_retries"),
        label="api_transport_identity.sdk_max_retries",
    )
    if retries < 0:
        raise MechanismDatasetError(
            "api_transport_identity.sdk_max_retries must be non-negative"
        )
    return value


def _safe_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {str(exc)[:800]}"
    for name, value in os.environ.items():
        if value and any(token in name.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            text = text.replace(value, "[REDACTED]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", text)


def _openai_provider(*, model: str, temperature: float, max_output_tokens: int) -> ResponseProvider:
    _load_dotenv()
    try:
        import openai  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MechanismDatasetError("openai package is unavailable in the conda environment") from exc
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise MechanismDatasetError("OPENAI_API_KEY is not configured")
    settings = _transport_settings()
    base_url = (
        str(settings["endpoint"])
        if settings["endpoint_source"] != "default"
        else None
    )
    client_options: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": settings["timeout_sec"],
        "max_retries": settings["sdk_max_retries"],
    }
    if base_url:
        client_options["base_url"] = base_url
    client = openai.OpenAI(**client_options)

    def call(_rule: Mapping[str, Any], system: str, user: str, _attempt: int) -> Dict[str, str]:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )
        return {
            "raw_response": str(response.choices[0].message.content or "").strip(),
            "actual_model": str(getattr(response, "model", "") or "").strip(),
            "response_id": str(getattr(response, "id", "") or "").strip(),
        }

    return call


def _trace_record_count(path: Path, *, start: int = 0) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        handle.seek(start)
        return sum(1 for line in handle if line.strip())


def _validate_trace_records(path: Path) -> None:
    if not path.is_file():
        raise MechanismDatasetError("raw response trace is missing")
    for index, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not raw.strip():
            continue
        record = _strict_json_loads(raw, label=f"raw trace line {index + 1}")
        if not isinstance(record, dict):
            raise MechanismDatasetError(f"raw trace line {index + 1} must be an object")
        if _contains_forbidden_trace_key(record) or _contains_configured_secret(record):
            raise MechanismDatasetError(f"raw trace line {index + 1} contains prompt/secret fields")


def _raw_trace_record_hashes(path: Path) -> List[str]:
    _validate_trace_records(path)
    hashes: List[str] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        record = _strict_json_loads(raw, label=f"raw trace line {line_number}")
        if not isinstance(record, dict):
            raise MechanismDatasetError(
                f"raw trace line {line_number} must be an object"
            )
        hashes.append(_object_sha256(record))
    return hashes


def _unbound_trace_hashes(
    raw_hashes: Sequence[str],
    bound_hashes: Sequence[str],
) -> List[str]:
    remaining = Counter(bound_hashes)
    unbound: List[str] = []
    for record_hash in raw_hashes:
        if remaining[record_hash] > 0:
            remaining[record_hash] -= 1
        else:
            unbound.append(record_hash)
    missing = sorted(record_hash for record_hash, count in remaining.items() if count)
    if missing:
        raise MechanismDatasetError(
            "checkpoint attempt records are missing from the raw response trace"
        )
    return unbound


def _contains_forbidden_trace_key(value: Any) -> bool:
    forbidden = {
        "prompt",
        "prompts",
        "messages",
        "system_prompt",
        "user_prompt",
        "api_key",
        "token",
        "secret",
        "password",
    }
    if isinstance(value, dict):
        return any(
            str(key).lower() in forbidden or _contains_forbidden_trace_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_trace_key(item) for item in value)
    return False


def _contains_configured_secret(value: Mapping[str, Any]) -> bool:
    serialized = _canonical_json(value)
    for name, secret in os.environ.items():
        if (
            secret
            and len(secret) >= 8
            and any(token in name.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
            and secret in serialized
        ):
            return True
    return False


def _trace_fingerprint(path: Path) -> Dict[str, Any]:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "record_count": _trace_record_count(path),
    }


def _artifact_fingerprint(path: Path, *, record_count: Optional[int] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _append_trace(path: Path, record: Mapping[str, Any]) -> None:
    if _contains_forbidden_trace_key(record) or _contains_configured_secret(record):
        raise MechanismDatasetError("raw trace records must not contain prompts or keys")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_model(model: str) -> str:
    if not isinstance(model, str) or model != DEFAULT_MODEL:
        raise MechanismDatasetError(
            f"GT generation model must be exactly {DEFAULT_MODEL}"
        )
    return model


def _normalize_provider_response(
    value: Any,
    *,
    provider_kind: str,
) -> Tuple[str, str, str]:
    if isinstance(value, str):
        if provider_kind != "injected":
            raise MechanismDatasetError(
                "OpenAI-compatible provider did not return response metadata"
            )
        return value, "", ""
    if provider_kind == "injected":
        raise MechanismDatasetError(
            "injected response providers must return a raw response string"
        )
    if not isinstance(value, dict):
        raise MechanismDatasetError("generation provider returned an invalid response type")
    _require_exact_keys(
        value,
        ("raw_response", "actual_model", "response_id"),
        label="generation provider response",
    )
    for field in ("raw_response", "actual_model", "response_id"):
        if not isinstance(value.get(field), str):
            raise MechanismDatasetError(
                f"generation provider response.{field} must be a string"
            )
    return value["raw_response"], value["actual_model"], value["response_id"]


def _resume_configuration_compatible(
    stored: Any,
    current: Mapping[str, Any],
    *,
    run_kind: str,
) -> bool:
    if stored == current:
        return True
    if run_kind != "development" or not isinstance(stored, dict):
        return False
    if set(stored) != set(current):
        return False
    stored_source = stored.get("source_identity")
    current_source = current.get("source_identity")
    if not isinstance(stored_source, dict) or not isinstance(current_source, dict):
        return False
    mutable_worktree_fields = {
        "git_dirty",
        "git_status_sha256",
        "git_tracked_diff_sha256",
    }
    if {
        key: value
        for key, value in stored_source.items()
        if key not in mutable_worktree_fields
    } != {
        key: value
        for key, value in current_source.items()
        if key not in mutable_worktree_fields
    }:
        return False
    stored_without_source = {
        key: value for key, value in stored.items() if key != "source_identity"
    }
    current_without_source = {
        key: value for key, value in current.items() if key != "source_identity"
    }
    return stored_without_source == current_without_source


def build_mechanism_dataset(
    *,
    catalog_path: Path,
    plan_path: Path,
    dataset_path: Path,
    candidate_trace_path: Path,
    raw_trace_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    model: str = DEFAULT_MODEL,
    max_rules: int = 0,
    max_attempts: int = 3,
    temperature: float = 0.0,
    max_output_tokens: int = 6000,
    resume: bool = False,
    run_kind: str = "development",
    response_provider: Optional[ResponseProvider] = None,
) -> Dict[str, Any]:
    model = _validate_model(model)
    if isinstance(max_rules, bool) or not isinstance(max_rules, int) or max_rules < 0:
        raise MechanismDatasetError("max_rules must be a non-negative integer")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise MechanismDatasetError("max_attempts must be a positive integer")
    if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or max_output_tokens < 1:
        raise MechanismDatasetError("max_output_tokens must be a positive integer")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise MechanismDatasetError("temperature must be numeric")
    if not math.isfinite(float(temperature)) or float(temperature) < 0:
        raise MechanismDatasetError("temperature must be finite and non-negative")
    if run_kind not in RUN_KINDS:
        raise MechanismDatasetError(
            "run_kind must be one of: " + ", ".join(RUN_KINDS)
        )
    resolved = [
        catalog_path.resolve(),
        plan_path.resolve(),
        dataset_path.resolve(),
        candidate_trace_path.resolve(),
        raw_trace_path.resolve(),
        checkpoint_path.resolve(),
        manifest_path.resolve(),
    ]
    if len(set(resolved)) != len(resolved):
        raise MechanismDatasetError(
            "catalog, plan, outputs, raw trace, checkpoint, and manifest paths must differ"
        )
    if manifest_path.exists():
        raise MechanismDatasetError(
            f"generation manifest is immutable and already exists: {manifest_path}"
        )
    plan = _validate_plan(_load_json(plan_path, label="rule plan"), catalog_path=catalog_path)
    all_rules = list(plan["rules"])
    if max_rules > len(all_rules):
        raise MechanismDatasetError(
            f"max_rules cannot exceed the frozen plan size {len(all_rules)}"
        )
    target_rules = all_rules[:max_rules] if max_rules else all_rules
    if not target_rules:
        raise MechanismDatasetError("generation target contains no rules")
    target_ids = [rule["rule_id"] for rule in target_rules]
    provider_kind = (
        "openai_compatible" if response_provider is None else "injected"
    )
    _load_dotenv()
    source_identity = _source_identity()
    runtime_identity = _runtime_identity()
    api_transport_identity = _api_transport_identity()
    if len(target_rules) == 60 and run_kind not in {"validation", "final"}:
        raise MechanismDatasetError(
            "the full 60-rule cohort requires run_kind=validation or final"
        )
    if run_kind in {"validation", "final"}:
        if len(target_rules) != 60:
            raise MechanismDatasetError(
                f"{run_kind} generation requires the complete 60-rule cohort"
            )
        if provider_kind != "openai_compatible":
            raise MechanismDatasetError(
                f"{run_kind} generation forbids injected response providers"
            )
        if runtime_identity.get("is_conda") is not True:
            raise MechanismDatasetError(f"{run_kind} generation requires conda")
        if source_identity.get("git_available") is not True:
            raise MechanismDatasetError(
                f"{run_kind} generation requires an available Git repository"
            )
        if source_identity.get("git_dirty") is True:
            raise MechanismDatasetError(
                f"{run_kind} generation requires a clean Git worktree"
            )
        if (
            api_transport_identity.get("endpoint_scheme") != "https"
            or api_transport_identity.get("endpoint_is_loopback") is not False
        ):
            raise MechanismDatasetError(
                f"{run_kind} generation requires a non-loopback HTTPS endpoint"
            )
    configuration = {
        "plan_sha256": plan["plan_sha256"],
        "catalog_sha256": plan["catalog"]["sha256"],
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "target_rule_ids": target_ids,
        "max_attempts_per_invocation": max_attempts,
        "temperature": float(temperature),
        "max_output_tokens": max_output_tokens,
        "run_kind": run_kind,
        "provider_kind": provider_kind,
        "source_identity": source_identity,
        "runtime_identity": runtime_identity,
        "api_transport_identity": api_transport_identity,
    }
    configuration_sha = _object_sha256(configuration)
    orphan_records = 0

    if resume:
        checkpoint = _load_json(checkpoint_path, label="generation checkpoint")
        if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise MechanismDatasetError("unsupported generation checkpoint")
        stored_configuration = checkpoint.get("configuration")
        if not _resume_configuration_compatible(
            stored_configuration,
            configuration,
            run_kind=run_kind,
        ):
            raise MechanismDatasetError("generation checkpoint configuration payload drifted")
        if not isinstance(stored_configuration, dict):
            raise MechanismDatasetError("generation checkpoint configuration is invalid")
        stored_configuration_sha = checkpoint.get("configuration_sha256")
        if stored_configuration_sha != _object_sha256(stored_configuration):
            raise MechanismDatasetError("generation checkpoint configuration does not match")
        configuration = dict(stored_configuration)
        configuration_sha = str(stored_configuration_sha)
        source_identity = dict(configuration["source_identity"])
        runtime_identity = dict(configuration["runtime_identity"])
        api_transport_identity = dict(configuration["api_transport_identity"])
        trace_info = checkpoint.get("raw_trace")
        if not isinstance(trace_info, dict) or not raw_trace_path.is_file():
            raise MechanismDatasetError("generation checkpoint raw trace binding is missing")
        prefix_size = _strict_int(trace_info.get("size_bytes"), label="checkpoint raw trace size")
        if raw_trace_path.stat().st_size < prefix_size:
            raise MechanismDatasetError("raw trace is shorter than checkpoint")
        if _sha256_file_prefix(raw_trace_path, prefix_size) != trace_info.get("sha256"):
            raise MechanismDatasetError("raw trace checkpoint prefix was modified")
        _load_raw_trace_records(raw_trace_path)
        current_count = _trace_record_count(raw_trace_path)
        recorded_count = _strict_int(
            trace_info.get("record_count"), label="checkpoint raw trace record_count"
        )
        if current_count < recorded_count:
            raise MechanismDatasetError("raw trace lost checkpointed records")
        orphan_records = current_count - recorded_count
        states = checkpoint.get("states")
        if not isinstance(states, dict):
            raise MechanismDatasetError("checkpoint states must be an object")
        states = dict(states)
        foreign_state_ids = set(states) - set(target_ids)
        if foreign_state_ids:
            raise MechanismDatasetError(
                f"checkpoint contains foreign rule states: {sorted(foreign_state_ids)}"
            )
    else:
        for path in (dataset_path, candidate_trace_path, raw_trace_path, checkpoint_path):
            if path.exists():
                raise MechanismDatasetError(f"refusing to overwrite existing generation artifact: {path}")
        raw_trace_path.parent.mkdir(parents=True, exist_ok=True)
        raw_trace_path.touch(exist_ok=False)
        states: Dict[str, Any] = {}
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "configuration": configuration,
            "configuration_sha256": configuration_sha,
            "states": states,
            "raw_trace": _trace_fingerprint(raw_trace_path),
            "orphan_raw_record_sha256s": [],
        }
        _atomic_write_json(checkpoint_path, checkpoint)

    existing_hashes: set[str] = set()
    for rule in target_rules:
        state = states.get(rule["rule_id"])
        if isinstance(state, dict):
            attempts_total = _strict_int(
                state.get("attempts_total"),
                label=f"checkpoint attempts_total for {rule['rule_id']}",
            )
            attempt_hashes = state.get("attempt_record_sha256s")
            if (
                attempts_total < 1
                or not isinstance(attempt_hashes, list)
                or len(attempt_hashes) != attempts_total
                or not all(
                    isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item)
                    for item in attempt_hashes
                )
            ):
                raise MechanismDatasetError(
                    f"checkpoint attempt trace binding is invalid for {rule['rule_id']}"
                )
        if isinstance(state, dict) and state.get("status") == "complete":
            stored_cases = state.get("cases")
            if not isinstance(stored_cases, list) or not all(
                isinstance(case, dict) for case in stored_cases
            ):
                raise MechanismDatasetError(
                    f"checkpoint cases missing for {rule['rule_id']}"
                )
            model_cases = [
                {key: value for key, value in case.items() if key != "case_content_sha256"}
                for case in stored_cases
            ]
            cases = _validate_generated_payload(
                {"cases": model_cases},
                rule=rule,
                existing_content_hashes=existing_hashes,
            )
            # Provenance-only fields are produced by the validator; require exact stored form.
            if cases != state.get("cases"):
                raise MechanismDatasetError(f"checkpoint case content drifted for {rule['rule_id']}")
            existing_hashes.update(case["case_content_sha256"] for case in cases)

    provider = response_provider or _openai_provider(
        model=model, temperature=float(temperature), max_output_tokens=max_output_tokens
    )

    latest_orphan_hashes: List[str] = []

    def write_checkpoint() -> None:
        nonlocal latest_orphan_hashes
        bound_attempt_hashes = [
            record_hash
            for rule in target_rules
            for record_hash in (
                states.get(rule["rule_id"], {}).get("attempt_record_sha256s", [])
                if isinstance(states.get(rule["rule_id"]), dict)
                else []
            )
        ]
        raw_record_hashes = _raw_trace_record_hashes(raw_trace_path)
        latest_orphan_hashes = _unbound_trace_hashes(
            raw_record_hashes, bound_attempt_hashes
        )
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "configuration": configuration,
            "configuration_sha256": configuration_sha,
            "states": states,
            "raw_trace": _trace_fingerprint(raw_trace_path),
            "orphan_raw_record_sha256s": latest_orphan_hashes,
        }
        _atomic_write_json(checkpoint_path, payload)

    for rule in target_rules:
        rule_id = rule["rule_id"]
        previous = states.get(rule_id)
        if isinstance(previous, dict) and previous.get("status") == "complete":
            continue
        attempts_total = int(previous.get("attempts_total") or 0) if isinstance(previous, dict) else 0
        prior_attempt_hashes = (
            list(previous.get("attempt_record_sha256s") or [])
            if isinstance(previous, dict)
            else []
        )
        invocation_attempt_hashes: List[str] = []
        last_error = ""
        completed: Optional[List[Dict[str, Any]]] = None
        raw_response_sha = ""
        system_prompt, user_prompt = _prompt(rule)
        prompt_sha256 = hashlib.sha256(
            (system_prompt + "\0" + user_prompt).encode("utf-8")
        ).hexdigest()
        for local_attempt in range(1, max_attempts + 1):
            attempt_number = attempts_total + local_attempt
            raw = ""
            actual_model = ""
            response_id = ""
            try:
                provider_response = provider(
                    rule, system_prompt, user_prompt, attempt_number
                )
                raw, actual_model, response_id = _normalize_provider_response(
                    provider_response,
                    provider_kind=provider_kind,
                )
                if provider_kind == "openai_compatible":
                    if actual_model != model:
                        raise MechanismDatasetError(
                            "OpenAI-compatible response model does not match the exact GT model"
                        )
                    if not response_id:
                        raise MechanismDatasetError(
                            "OpenAI-compatible response is missing response_id"
                        )
                raw_response_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                payload = _strict_json_loads(raw, label=f"Gemini response for {rule_id}")
                completed = _validate_generated_payload(
                    payload, rule=rule, existing_content_hashes=existing_hashes
                )
                trace_record = {
                    "schema_version": RAW_TRACE_SCHEMA_VERSION,
                    "rule_id": rule_id,
                    "selection_index": rule["selection_index"],
                    "attempt": attempt_number,
                    "model": model,
                    "actual_model": actual_model,
                    "response_id": response_id,
                    "provider_kind": provider_kind,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_sha256": prompt_sha256,
                    "raw_response": raw,
                    "raw_response_sha256": raw_response_sha,
                    "parse_status": "valid",
                    "error": "",
                }
                _append_trace(raw_trace_path, trace_record)
                invocation_attempt_hashes.append(_object_sha256(trace_record))
                break
            except Exception as exc:
                last_error = _safe_error(exc)
                raw_response_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                trace_record = {
                        "schema_version": RAW_TRACE_SCHEMA_VERSION,
                        "rule_id": rule_id,
                        "selection_index": rule["selection_index"],
                        "attempt": attempt_number,
                        "model": model,
                        "actual_model": actual_model,
                        "response_id": response_id,
                        "provider_kind": provider_kind,
                        "prompt_version": PROMPT_VERSION,
                        "prompt_sha256": prompt_sha256,
                        "raw_response": raw,
                        "raw_response_sha256": raw_response_sha,
                        "parse_status": "invalid",
                        "error": last_error,
                    }
                _append_trace(raw_trace_path, trace_record)
                invocation_attempt_hashes.append(_object_sha256(trace_record))
        if completed is None:
            states[rule_id] = {
                "status": "failed",
                "attempts_total": attempts_total + max_attempts,
                "last_error": last_error or "no_valid_response",
                "attempt_record_sha256s": prior_attempt_hashes
                + invocation_attempt_hashes,
            }
        else:
            states[rule_id] = {
                "status": "complete",
                "attempts_total": attempts_total + local_attempt,
                "raw_response_sha256": raw_response_sha,
                "cases": completed,
                "attempt_record_sha256s": prior_attempt_hashes
                + invocation_attempt_hashes,
            }
            existing_hashes.update(case["case_content_sha256"] for case in completed)
        write_checkpoint()

    # Bind any orphan suffix even when every target state was already complete.
    write_checkpoint()

    final_source_identity = _source_identity()
    if run_kind in {"validation", "final"}:
        final_runtime_identity = _runtime_identity()
        final_api_transport_identity = _api_transport_identity()
        if final_source_identity != source_identity:
            raise MechanismDatasetError(
                f"source identity changed during {run_kind} generation"
            )
        if final_runtime_identity != runtime_identity:
            raise MechanismDatasetError(
                f"runtime identity changed during {run_kind} generation"
            )
        if final_api_transport_identity != api_transport_identity:
            raise MechanismDatasetError(
                f"API transport identity changed during {run_kind} generation"
            )

    dataset_rows = _dataset_rows(
        states, target_rules, model=model, plan_sha256=plan["plan_sha256"]
    )
    candidate_rows = _candidate_trace_rows(dataset_rows, target_rules)
    if len(dataset_rows) != len(candidate_rows):
        raise MechanismDatasetError("dataset and target-binding trace size differ")
    _atomic_write_json(dataset_path, dataset_rows)
    _atomic_write_json(candidate_trace_path, candidate_rows)
    completed_rules = sum(
        1
        for rule in target_rules
        if isinstance(states.get(rule["rule_id"]), dict)
        and states[rule["rule_id"]].get("status") == "complete"
    )
    failed_rule_ids = [
        rule["rule_id"]
        for rule in target_rules
        if not isinstance(states.get(rule["rule_id"]), dict)
        or states[rule["rule_id"]].get("status") != "complete"
    ]
    run_complete = not failed_rule_ids and len(dataset_rows) == 5 * len(target_rules)
    formal_complete = (
        run_complete and len(target_rules) == 60 and len(dataset_rows) == 300
    )
    report = {
        "plan_sha256": plan["plan_sha256"],
        "catalog_sha256": plan["catalog"]["sha256"],
        "model": model,
        "run_kind": run_kind,
        "provider_kind": provider_kind,
        "prompt_version": PROMPT_VERSION,
        "planned_rule_count": len(target_rules),
        "completed_rule_count": completed_rules,
        "failed_rule_ids": failed_rule_ids,
        "dataset_case_count": len(dataset_rows),
        "candidate_trace_count": len(candidate_rows),
        "resume": {
            "enabled": bool(resume),
            "orphan_raw_trace_records": orphan_records,
            "new_orphan_raw_trace_records": orphan_records,
            "total_orphan_raw_trace_records": len(latest_orphan_hashes),
        },
        "raw_trace": _trace_fingerprint(raw_trace_path),
        "run_complete": run_complete,
        "complete": formal_complete,
    }
    ordered_ids = [row["id"] for row in dataset_rows]
    generation_manifest: Dict[str, Any] = {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "manifest_type": GENERATION_MANIFEST_TYPE,
        "run_status": (
            "complete"
            if formal_complete
            else ("development_incomplete" if run_kind == "development" else "incomplete")
        ),
        "complete": formal_complete,
        "run_complete": run_complete,
        "failed_rule_ids": failed_rule_ids,
        "configuration": configuration,
        "configuration_sha256": configuration_sha,
        "environment": {
            "source": source_identity,
            "runtime": runtime_identity,
            "api_transport": api_transport_identity,
            "provider_kind": provider_kind,
        },
        "inputs": {
            "catalog": _artifact_fingerprint(
                catalog_path,
                record_count=int(plan["catalog"]["rule_count"]),
            ),
            "plan": {
                **_artifact_fingerprint(plan_path, record_count=len(all_rules)),
                "plan_sha256": plan["plan_sha256"],
            },
        },
        "artifacts": {
            "dataset": _artifact_fingerprint(
                dataset_path, record_count=len(dataset_rows)
            ),
            "candidate_trace": _artifact_fingerprint(
                candidate_trace_path, record_count=len(candidate_rows)
            ),
            "raw_response_trace": _artifact_fingerprint(
                raw_trace_path,
                record_count=_trace_record_count(raw_trace_path),
            ),
            "checkpoint": _artifact_fingerprint(
                checkpoint_path,
                record_count=len(states),
            ),
        },
        "ordered_ids": ordered_ids,
        "ordered_ids_sha256": _object_sha256(ordered_ids),
        "planned_rule_count": len(target_rules),
        "completed_rule_count": completed_rules,
        "dataset_case_count": len(dataset_rows),
        "candidate_trace_count": len(candidate_rows),
        "candidate_binding": {
            "candidate_source": "frozen_target_binding",
            "unified_retrieval_mode": "target_binding",
            "selection_strategy": "target_rule_binding",
            "retrieval_score_kind": "fixed_control_0_1",
        },
        "raw_trace_accounting": {
            "generation_attempt_record_count": sum(
                int(state.get("attempts_total") or 0)
                for state in states.values()
                if isinstance(state, dict)
            ),
            "orphan_record_count": len(latest_orphan_hashes),
            "orphan_record_sha256s": latest_orphan_hashes,
        },
    }
    generation_manifest["manifest_sha256"] = _object_sha256(
        {
            key: value
            for key, value in generation_manifest.items()
            if key != "manifest_sha256"
        }
    )
    _atomic_write_json(manifest_path, generation_manifest)
    report["manifest"] = {
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "manifest_sha256": generation_manifest["manifest_sha256"],
    }
    artifact_audit = audit_generation_artifacts(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
    )
    report["artifact_audit"] = {
        key: artifact_audit[key]
        for key in (
            "schema_version",
            "valid",
            "complete",
            "run_complete",
            "run_kind",
            "provider_kind",
            "planned_rule_count",
            "completed_rule_count",
            "dataset_case_count",
            "candidate_trace_count",
            "raw_trace_record_count",
            "orphan_raw_trace_record_count",
            "manifest_sha256",
            "dataset_sha256",
            "candidate_trace_sha256",
            "raw_response_trace_sha256",
            "artifact_paths",
        )
    }
    return report


def _resolve_manifest_path(
    record: Any,
    *,
    explicit: Optional[Path],
    label: str,
) -> Path:
    if not isinstance(record, dict):
        raise MechanismDatasetError(f"manifest {label} fingerprint must be an object")
    raw_path = _nonempty_string(record.get("path"), label=f"manifest {label}.path")
    bound = Path(raw_path)
    if not bound.is_absolute():
        bound = PROJECT_ROOT / bound
    bound = bound.resolve()
    if explicit is not None and explicit.resolve() != bound:
        raise MechanismDatasetError(
            f"explicit {label} path does not match generation manifest"
        )
    return bound


def _audit_fingerprint(
    record: Any,
    *,
    path: Path,
    label: str,
    record_count: Optional[int] = None,
) -> None:
    if not isinstance(record, dict):
        raise MechanismDatasetError(f"manifest {label} fingerprint must be an object")
    expected_keys = {"path", "size_bytes", "sha256"}
    if record_count is not None:
        expected_keys.add("record_count")
    if set(record) != expected_keys:
        raise MechanismDatasetError(f"manifest {label} fingerprint keys are invalid")
    if not path.is_file():
        raise MechanismDatasetError(f"manifest {label} artifact is missing: {path}")
    size_bytes = _strict_int(
        record.get("size_bytes"), label=f"manifest {label}.size_bytes"
    )
    if size_bytes < 0 or size_bytes != path.stat().st_size:
        raise MechanismDatasetError(f"manifest {label} size does not match")
    recorded_sha = _require_sha256(
        record.get("sha256"), label=f"manifest {label}.sha256"
    )
    if recorded_sha != _sha256_file(path):
        raise MechanismDatasetError(f"manifest {label} SHA256 does not match")
    if record_count is not None:
        recorded_count = _strict_int(
            record.get("record_count"), label=f"manifest {label}.record_count"
        )
        if recorded_count < 0 or recorded_count != record_count:
            raise MechanismDatasetError(f"manifest {label} record_count does not match")


def _load_json_object_array(path: Path, *, label: str) -> List[Dict[str, Any]]:
    payload = _load_json(path, label=label)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise MechanismDatasetError(f"{label} must be an array of objects")
    return list(payload)


def _load_raw_trace_records(path: Path) -> List[Dict[str, Any]]:
    _validate_trace_records(path)
    records: List[Dict[str, Any]] = []
    expected_keys = {
        "schema_version",
        "rule_id",
        "selection_index",
        "attempt",
        "model",
        "actual_model",
        "response_id",
        "provider_kind",
        "prompt_version",
        "prompt_sha256",
        "raw_response",
        "raw_response_sha256",
        "parse_status",
        "error",
    }
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        record = _strict_json_loads(raw_line, label=f"raw trace line {line_number}")
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise MechanismDatasetError(
                f"raw trace line {line_number} does not match the exact trace schema"
            )
        if record.get("schema_version") != RAW_TRACE_SCHEMA_VERSION:
            raise MechanismDatasetError(f"raw trace line {line_number} schema drifted")
        for field in (
            "rule_id",
            "model",
            "actual_model",
            "response_id",
            "provider_kind",
            "prompt_version",
            "prompt_sha256",
            "raw_response",
            "raw_response_sha256",
            "parse_status",
            "error",
        ):
            if not isinstance(record.get(field), str):
                raise MechanismDatasetError(
                    f"raw trace line {line_number}.{field} must be a string"
                )
        selection_index = _strict_int(
            record.get("selection_index"), label="raw trace selection_index"
        )
        if selection_index < 0:
            raise MechanismDatasetError("raw trace selection_index must be non-negative")
        attempt = _strict_int(record.get("attempt"), label="raw trace attempt")
        if attempt < 1:
            raise MechanismDatasetError("raw trace attempt must be positive")
        if record.get("parse_status") not in {"valid", "invalid"}:
            raise MechanismDatasetError("raw trace parse_status is invalid")
        _require_sha256(
            record.get("prompt_sha256"), label="raw trace prompt_sha256"
        )
        raw_response = str(record["raw_response"])
        _require_sha256(
            record.get("raw_response_sha256"), label="raw trace raw_response_sha256"
        )
        if record.get("raw_response_sha256") != hashlib.sha256(
            raw_response.encode("utf-8")
        ).hexdigest():
            raise MechanismDatasetError(
                f"raw trace line {line_number} raw_response SHA256 is invalid"
            )
        if record["parse_status"] == "valid" and record["error"] != "":
            raise MechanismDatasetError(
                f"raw trace line {line_number} valid response must have an empty error"
            )
        if record["parse_status"] == "invalid" and not record["error"].strip():
            raise MechanismDatasetError(
                f"raw trace line {line_number} invalid response must record an error"
            )
        records.append(record)
    return records


def audit_generation_artifacts(
    *,
    dataset_path: Path,
    manifest_path: Path,
    plan_path: Optional[Path] = None,
    catalog_path: Optional[Path] = None,
    candidate_trace_path: Optional[Path] = None,
    raw_trace_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fail closed unless a generation receipt reproduces every bound artifact."""

    manifest = _load_json(manifest_path, label="generation manifest")
    if not isinstance(manifest, dict):
        raise MechanismDatasetError("generation manifest must be an object")
    manifest_keys = {
        "schema_version",
        "manifest_type",
        "run_status",
        "complete",
        "run_complete",
        "failed_rule_ids",
        "configuration",
        "configuration_sha256",
        "environment",
        "inputs",
        "artifacts",
        "ordered_ids",
        "ordered_ids_sha256",
        "planned_rule_count",
        "completed_rule_count",
        "dataset_case_count",
        "candidate_trace_count",
        "candidate_binding",
        "raw_trace_accounting",
        "manifest_sha256",
    }
    _require_exact_keys(manifest, manifest_keys, label="generation manifest")
    if (
        manifest.get("schema_version") != GENERATION_MANIFEST_SCHEMA_VERSION
        or manifest.get("manifest_type") != GENERATION_MANIFEST_TYPE
    ):
        raise MechanismDatasetError("unsupported generation manifest schema/type")
    manifest_without_sha = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    _require_sha256(
        manifest.get("manifest_sha256"), label="generation manifest.manifest_sha256"
    )
    if manifest.get("manifest_sha256") != _object_sha256(manifest_without_sha):
        raise MechanismDatasetError("generation manifest self SHA256 is invalid")
    if not isinstance(manifest.get("complete"), bool) or not isinstance(
        manifest.get("run_complete"), bool
    ):
        raise MechanismDatasetError("generation manifest completion fields must be boolean")

    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise MechanismDatasetError("generation manifest configuration must be an object")
    _require_sha256(
        manifest.get("configuration_sha256"),
        label="generation manifest.configuration_sha256",
    )
    if manifest.get("configuration_sha256") != _object_sha256(configuration):
        raise MechanismDatasetError("generation configuration SHA256 is invalid")
    configuration_keys = {
        "plan_sha256",
        "catalog_sha256",
        "model",
        "prompt_version",
        "target_rule_ids",
        "max_attempts_per_invocation",
        "temperature",
        "max_output_tokens",
        "run_kind",
        "provider_kind",
        "source_identity",
        "runtime_identity",
        "api_transport_identity",
    }
    _require_exact_keys(configuration, configuration_keys, label="generation configuration")
    run_kind = configuration.get("run_kind")
    provider_kind = configuration.get("provider_kind")
    if run_kind not in RUN_KINDS:
        raise MechanismDatasetError("generation manifest run_kind is invalid")
    if provider_kind not in {"openai_compatible", "injected"}:
        raise MechanismDatasetError("generation manifest provider_kind is invalid")
    _validate_model(configuration.get("model"))
    if configuration.get("prompt_version") != PROMPT_VERSION:
        raise MechanismDatasetError("generation manifest prompt_version is invalid")
    _require_sha256(
        configuration.get("plan_sha256"), label="generation configuration.plan_sha256"
    )
    _require_sha256(
        configuration.get("catalog_sha256"),
        label="generation configuration.catalog_sha256",
    )
    max_attempts = _strict_int(
        configuration.get("max_attempts_per_invocation"),
        label="generation configuration.max_attempts_per_invocation",
    )
    max_output_tokens = _strict_int(
        configuration.get("max_output_tokens"),
        label="generation configuration.max_output_tokens",
    )
    if max_attempts < 1 or max_output_tokens < 1:
        raise MechanismDatasetError("generation attempt/token limits must be positive")
    temperature = configuration.get("temperature")
    if (
        not isinstance(temperature, float)
        or not math.isfinite(float(temperature))
        or float(temperature) < 0
    ):
        raise MechanismDatasetError("generation temperature is invalid")
    source_identity = _validate_source_identity(configuration.get("source_identity"))
    runtime_identity = _validate_runtime_identity(configuration.get("runtime_identity"))
    api_transport_identity = _validate_api_transport_identity(
        configuration.get("api_transport_identity")
    )
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        raise MechanismDatasetError("generation manifest environment must be an object")
    _require_exact_keys(
        environment,
        ("source", "runtime", "api_transport", "provider_kind"),
        label="generation manifest environment",
    )
    if environment != {
        "source": source_identity,
        "runtime": runtime_identity,
        "api_transport": api_transport_identity,
        "provider_kind": provider_kind,
    }:
        raise MechanismDatasetError("generation environment does not match configuration")

    inputs = manifest.get("inputs")
    artifacts = manifest.get("artifacts")
    if not isinstance(inputs, dict) or set(inputs) != {"catalog", "plan"}:
        raise MechanismDatasetError("generation manifest inputs are invalid")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "dataset",
        "candidate_trace",
        "raw_response_trace",
        "checkpoint",
    }:
        raise MechanismDatasetError("generation manifest artifacts are invalid")
    bound_catalog = _resolve_manifest_path(
        inputs["catalog"], explicit=catalog_path, label="catalog"
    )
    bound_plan = _resolve_manifest_path(inputs["plan"], explicit=plan_path, label="plan")
    bound_dataset = _resolve_manifest_path(
        artifacts["dataset"], explicit=dataset_path, label="dataset"
    )
    bound_candidate = _resolve_manifest_path(
        artifacts["candidate_trace"],
        explicit=candidate_trace_path,
        label="candidate trace",
    )
    bound_raw = _resolve_manifest_path(
        artifacts["raw_response_trace"],
        explicit=raw_trace_path,
        label="raw response trace",
    )
    bound_checkpoint = _resolve_manifest_path(
        artifacts["checkpoint"], explicit=checkpoint_path, label="checkpoint"
    )
    all_bound_paths = {
        bound_catalog,
        bound_plan,
        bound_dataset,
        bound_candidate,
        bound_raw,
        bound_checkpoint,
        manifest_path.resolve(),
    }
    if len(all_bound_paths) != 7:
        raise MechanismDatasetError(
            "generation manifest inputs and outputs must use seven distinct paths"
        )

    plan = _validate_plan(_load_json(bound_plan, label="rule plan"), catalog_path=bound_catalog)
    catalog = _load_json(bound_catalog, label="unified catalog")
    catalog_records, _ = _catalog_records(catalog)
    dataset = _load_json_object_array(bound_dataset, label="mechanism dataset")
    candidate_trace = _load_json_object_array(
        bound_candidate, label="target candidate trace"
    )
    checkpoint = _load_json(bound_checkpoint, label="generation checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise MechanismDatasetError("generation checkpoint schema is invalid")
    _require_exact_keys(
        checkpoint,
        (
            "schema_version",
            "configuration",
            "configuration_sha256",
            "states",
            "raw_trace",
            "orphan_raw_record_sha256s",
        ),
        label="generation checkpoint",
    )
    if checkpoint.get("configuration") != configuration or checkpoint.get(
        "configuration_sha256"
    ) != manifest.get("configuration_sha256"):
        raise MechanismDatasetError("generation checkpoint configuration drifted")
    states = checkpoint.get("states")
    if not isinstance(states, dict):
        raise MechanismDatasetError("generation checkpoint states must be an object")
    target_rule_ids = configuration.get("target_rule_ids")
    if not isinstance(target_rule_ids, list) or not all(
        isinstance(rule_id, str) and rule_id for rule_id in target_rule_ids
    ):
        raise MechanismDatasetError("generation target_rule_ids are invalid")
    if not target_rule_ids or len(target_rule_ids) > 60 or len(set(target_rule_ids)) != len(
        target_rule_ids
    ):
        raise MechanismDatasetError(
            "generation target_rule_ids must contain 1-60 unique frozen IDs"
        )
    all_rules = list(plan["rules"])
    target_rules = all_rules[: len(target_rule_ids)]
    if [rule["rule_id"] for rule in target_rules] != target_rule_ids:
        raise MechanismDatasetError("generation target rules are not a frozen plan prefix")
    if configuration.get("plan_sha256") != plan.get("plan_sha256"):
        raise MechanismDatasetError("generation configuration plan SHA256 drifted")
    if configuration.get("catalog_sha256") != plan["catalog"].get("sha256"):
        raise MechanismDatasetError("generation configuration catalog SHA256 drifted")
    if set(states) != set(target_rule_ids):
        raise MechanismDatasetError(
            "generation checkpoint states must exactly match target_rule_ids"
        )
    bound_attempt_hashes: List[str] = []
    for rule_id in target_rule_ids:
        state = states[rule_id]
        if not isinstance(state, dict):
            raise MechanismDatasetError(f"checkpoint state for {rule_id} must be an object")
        status = state.get("status")
        if status == "complete":
            _require_exact_keys(
                state,
                (
                    "status",
                    "attempts_total",
                    "raw_response_sha256",
                    "cases",
                    "attempt_record_sha256s",
                ),
                label=f"checkpoint state {rule_id}",
            )
            _require_sha256(
                state.get("raw_response_sha256"),
                label=f"checkpoint state {rule_id}.raw_response_sha256",
            )
            if not isinstance(state.get("cases"), list):
                raise MechanismDatasetError(
                    f"checkpoint state {rule_id}.cases must be an array"
                )
        elif status == "failed":
            _require_exact_keys(
                state,
                ("status", "attempts_total", "last_error", "attempt_record_sha256s"),
                label=f"checkpoint state {rule_id}",
            )
            _nonempty_string(
                state.get("last_error"), label=f"checkpoint state {rule_id}.last_error"
            )
        else:
            raise MechanismDatasetError(f"checkpoint state {rule_id}.status is invalid")
        attempts_total = _strict_int(
            state.get("attempts_total"),
            label=f"checkpoint state {rule_id}.attempts_total",
        )
        if attempts_total < 1:
            raise MechanismDatasetError(
                f"checkpoint state {rule_id}.attempts_total must be positive"
            )
        attempt_hashes = state.get("attempt_record_sha256s")
        if not isinstance(attempt_hashes, list) or len(attempt_hashes) != attempts_total:
            raise MechanismDatasetError(
                f"checkpoint state {rule_id} attempt trace count is invalid"
            )
        for index, record_hash in enumerate(attempt_hashes):
            _require_sha256(
                record_hash,
                label=f"checkpoint state {rule_id}.attempt_record_sha256s[{index}]",
            )
        bound_attempt_hashes.extend(attempt_hashes)
    if len(bound_attempt_hashes) != len(set(bound_attempt_hashes)):
        raise MechanismDatasetError("checkpoint attempt record hashes must be unique")

    expected_dataset = _dataset_rows(
        states,
        target_rules,
        model=str(configuration["model"]),
        plan_sha256=str(plan["plan_sha256"]),
    )
    if dataset != expected_dataset:
        raise MechanismDatasetError(
            "mechanism dataset does not reproduce from checkpoint and plan"
        )
    expected_candidate = _candidate_trace_rows(dataset, target_rules)
    if candidate_trace != expected_candidate:
        raise MechanismDatasetError(
            "target candidate trace does not reproduce from dataset and catalog ownership"
        )
    # Also apply the replay entry point's independent target-binding validator.
    from scripts.run_checker_replay import _catalog_index, _validate_frozen_trace

    try:
        _validate_frozen_trace(
            candidate_trace,
            _catalog_index(catalog),
            dataset=dataset,
        )
    except Exception as exc:
        raise MechanismDatasetError(
            f"replay target-binding validation failed: {exc}"
        ) from exc

    raw_records = _load_raw_trace_records(bound_raw)
    if checkpoint.get("raw_trace") != _trace_fingerprint(bound_raw):
        raise MechanismDatasetError(
            "generation checkpoint raw trace fingerprint does not match the final trace"
        )
    raw_record_hashes = [_object_sha256(record) for record in raw_records]
    if len(raw_record_hashes) != len(set(raw_record_hashes)):
        raise MechanismDatasetError("raw response trace contains duplicate records")
    checkpoint_orphans = checkpoint.get("orphan_raw_record_sha256s")
    if not isinstance(checkpoint_orphans, list):
        raise MechanismDatasetError(
            "generation checkpoint orphan trace binding must be an array"
        )
    for index, record_hash in enumerate(checkpoint_orphans):
        _require_sha256(
            record_hash,
            label=f"checkpoint orphan_raw_record_sha256s[{index}]",
        )
    expected_orphans = _unbound_trace_hashes(
        raw_record_hashes, bound_attempt_hashes
    )
    if checkpoint_orphans != expected_orphans:
        raise MechanismDatasetError(
            "generation checkpoint orphan trace accounting drifted"
        )
    accounting = manifest.get("raw_trace_accounting")
    if not isinstance(accounting, dict):
        raise MechanismDatasetError("generation raw_trace_accounting must be an object")
    _require_exact_keys(
        accounting,
        (
            "generation_attempt_record_count",
            "orphan_record_count",
            "orphan_record_sha256s",
        ),
        label="generation raw_trace_accounting",
    )
    generation_attempt_count = _strict_int(
        accounting.get("generation_attempt_record_count"),
        label="generation raw_trace_accounting.generation_attempt_record_count",
    )
    orphan_record_count = _strict_int(
        accounting.get("orphan_record_count"),
        label="generation raw_trace_accounting.orphan_record_count",
    )
    if (
        generation_attempt_count != len(bound_attempt_hashes)
        or orphan_record_count != len(expected_orphans)
        or accounting.get("orphan_record_sha256s") != expected_orphans
        or generation_attempt_count + orphan_record_count != len(raw_records)
    ):
        raise MechanismDatasetError("generation raw trace accounting is invalid")
    record_by_hash = dict(zip(raw_record_hashes, raw_records))
    for rule_id in target_rule_ids:
        state = states[rule_id]
        attempt_records = [
            record_by_hash[record_hash]
            for record_hash in state["attempt_record_sha256s"]
        ]
        if any(record.get("rule_id") != rule_id for record in attempt_records):
            raise MechanismDatasetError(
                f"checkpoint attempt trace ownership drifted for {rule_id}"
            )
        if [record.get("attempt") for record in attempt_records] != list(
            range(1, int(state["attempts_total"]) + 1)
        ):
            raise MechanismDatasetError(
                f"checkpoint attempt sequence drifted for {rule_id}"
            )
        if state["status"] == "complete":
            if (
                any(record.get("parse_status") != "invalid" for record in attempt_records[:-1])
                or attempt_records[-1].get("parse_status") != "valid"
                or attempt_records[-1].get("raw_response_sha256")
                != state.get("raw_response_sha256")
            ):
                raise MechanismDatasetError(
                    f"checkpoint completion attempt binding drifted for {rule_id}"
                )
        elif (
            any(record.get("parse_status") != "invalid" for record in attempt_records)
            or state.get("last_error") != attempt_records[-1].get("error")
        ):
            raise MechanismDatasetError(
                f"checkpoint failed attempt binding drifted for {rule_id}"
            )
    records_by_rule: Dict[str, List[Dict[str, Any]]] = {}
    rule_by_id = {str(rule["rule_id"]): rule for rule in target_rules}
    trace_content_hashes: set[str] = set()
    for record in raw_records:
        rule_id = str(record["rule_id"])
        rule = rule_by_id.get(rule_id)
        if rule is None:
            raise MechanismDatasetError(
                f"raw response trace references foreign rule {rule_id}"
            )
        if record.get("selection_index") != rule.get("selection_index"):
            raise MechanismDatasetError("raw response trace selection_index drifted")
        if record.get("model") != configuration.get("model") or record.get(
            "prompt_version"
        ) != configuration.get("prompt_version"):
            raise MechanismDatasetError("raw response trace model/prompt version drifted")
        if record.get("provider_kind") != provider_kind:
            raise MechanismDatasetError("raw response trace provider_kind drifted")
        if provider_kind == "injected":
            if record.get("actual_model") != "" or record.get("response_id") != "":
                raise MechanismDatasetError(
                    "injected raw response trace must not claim provider metadata"
                )
        elif record.get("parse_status") == "valid" and (
            record.get("actual_model") != DEFAULT_MODEL
            or not str(record.get("response_id") or "").strip()
        ):
            raise MechanismDatasetError(
                "valid OpenAI-compatible raw response lacks exact actual model/response ID"
            )
        system_prompt, user_prompt = _prompt(rule)
        expected_prompt_sha = hashlib.sha256(
            (system_prompt + "\0" + user_prompt).encode("utf-8")
        ).hexdigest()
        if record.get("prompt_sha256") != expected_prompt_sha:
            raise MechanismDatasetError("raw response trace prompt SHA256 drifted")
        parsed_cases: Optional[List[Dict[str, Any]]] = None
        parse_error: Optional[MechanismDatasetError] = None
        try:
            raw_payload = _strict_json_loads(
                str(record["raw_response"]),
                label=f"raw response for {rule_id} attempt {record['attempt']}",
            )
            parsed_cases = _validate_generated_payload(
                raw_payload,
                rule=rule,
                existing_content_hashes=trace_content_hashes,
            )
        except MechanismDatasetError as exc:
            parse_error = exc
        if record["parse_status"] == "valid":
            if parse_error is not None or parsed_cases is None:
                raise MechanismDatasetError(
                    f"raw response trace marks an invalid response valid for {rule_id}: "
                    f"{parse_error}"
                )
            trace_content_hashes.update(
                case["case_content_sha256"] for case in parsed_cases
            )
        elif parse_error is None:
            raise MechanismDatasetError(
                f"raw response trace marks a schema-valid response invalid for {rule_id}"
            )
        records_by_rule.setdefault(rule_id, []).append(record)

    existing_content_hashes: set[str] = set()
    completed_rule_ids: List[str] = []
    failed_rule_ids: List[str] = []
    for rule in target_rules:
        rule_id = str(rule["rule_id"])
        state = states.get(rule_id)
        rule_valid_records = [
            record
            for record in records_by_rule.get(rule_id, [])
            if record.get("parse_status") == "valid"
        ]
        if not isinstance(state, dict) or state.get("status") != "complete":
            if rule_valid_records:
                raise MechanismDatasetError(
                    f"failed rule {rule_id} must not have a valid raw response"
                )
            failed_rule_ids.append(rule_id)
            continue
        completed_rule_ids.append(rule_id)
        valid_records = [
            record
            for record in rule_valid_records
            if record.get("raw_response_sha256") == state.get("raw_response_sha256")
        ]
        if len(rule_valid_records) != 1 or len(valid_records) != 1:
            raise MechanismDatasetError(
                f"completed rule {rule_id} must bind exactly one valid raw response"
            )
        raw_payload = _strict_json_loads(
            str(valid_records[0]["raw_response"]),
            label=f"valid raw response for {rule_id}",
        )
        rebuilt_cases = _validate_generated_payload(
            raw_payload,
            rule=rule,
            existing_content_hashes=existing_content_hashes,
        )
        if rebuilt_cases != state.get("cases"):
            raise MechanismDatasetError(
                f"checkpoint cases for {rule_id} do not reproduce from raw response"
            )
        existing_content_hashes.update(
            case["case_content_sha256"] for case in rebuilt_cases
        )

    ordered_ids = [row["id"] for row in dataset]
    _require_sha256(
        manifest.get("ordered_ids_sha256"),
        label="generation manifest.ordered_ids_sha256",
    )
    if manifest.get("ordered_ids") != ordered_ids or manifest.get(
        "ordered_ids_sha256"
    ) != _object_sha256(ordered_ids):
        raise MechanismDatasetError("generation manifest ordered dataset IDs drifted")
    expected_counts = {
        "planned_rule_count": len(target_rules),
        "completed_rule_count": len(completed_rule_ids),
        "dataset_case_count": len(dataset),
        "candidate_trace_count": len(candidate_trace),
    }
    for field, expected in expected_counts.items():
        actual = _strict_int(manifest.get(field), label=f"generation manifest.{field}")
        if actual < 0 or actual != expected:
            raise MechanismDatasetError(f"generation manifest {field} drifted")
    if manifest.get("failed_rule_ids") != failed_rule_ids:
        raise MechanismDatasetError("generation manifest failed_rule_ids drifted")

    _audit_fingerprint(
        inputs["catalog"],
        path=bound_catalog,
        label="catalog",
        record_count=len(catalog_records),
    )
    if inputs["plan"].get("plan_sha256") != plan.get("plan_sha256"):
        raise MechanismDatasetError("manifest plan_sha256 drifted")
    # The plan fingerprint has one additional semantic self-hash field.
    if set(inputs["plan"]) != {
        "path",
        "size_bytes",
        "sha256",
        "record_count",
        "plan_sha256",
    }:
        raise MechanismDatasetError("manifest plan fingerprint keys are invalid")
    _audit_fingerprint(
        {key: value for key, value in inputs["plan"].items() if key != "plan_sha256"},
        path=bound_plan,
        label="plan",
        record_count=len(all_rules),
    )
    _audit_fingerprint(
        artifacts["dataset"],
        path=bound_dataset,
        label="dataset",
        record_count=len(dataset),
    )
    _audit_fingerprint(
        artifacts["candidate_trace"],
        path=bound_candidate,
        label="candidate trace",
        record_count=len(candidate_trace),
    )
    _audit_fingerprint(
        artifacts["raw_response_trace"],
        path=bound_raw,
        label="raw response trace",
        record_count=len(raw_records),
    )
    _audit_fingerprint(
        artifacts["checkpoint"],
        path=bound_checkpoint,
        label="checkpoint",
        record_count=len(states),
    )

    run_complete = not failed_rule_ids and len(dataset) == 5 * len(target_rules)
    formal_complete = run_complete and len(target_rules) == 60 and len(dataset) == 300
    if manifest.get("run_complete") is not run_complete:
        raise MechanismDatasetError("generation manifest run_complete is dishonest")
    if manifest.get("complete") is not formal_complete:
        raise MechanismDatasetError("generation manifest complete is dishonest")
    expected_status = (
        "complete"
        if formal_complete
        else ("development_incomplete" if run_kind == "development" else "incomplete")
    )
    if manifest.get("run_status") != expected_status:
        raise MechanismDatasetError("generation manifest run_status is dishonest")
    binding = manifest.get("candidate_binding")
    if binding != {
        "candidate_source": "frozen_target_binding",
        "unified_retrieval_mode": "target_binding",
        "selection_strategy": "target_rule_binding",
        "retrieval_score_kind": "fixed_control_0_1",
    }:
        raise MechanismDatasetError("generation candidate binding metadata drifted")
    if run_kind == "development" and len(target_rules) == 60:
        raise MechanismDatasetError(
            "development generation cannot claim the complete 60-rule cohort"
        )
    if run_kind in {"validation", "final"}:
        if len(target_rules) != 60:
            raise MechanismDatasetError(
                f"{run_kind} generation must bind the complete 60-rule cohort"
            )
        if provider_kind != "openai_compatible":
            raise MechanismDatasetError("formal generation used an injected provider")
        if source_identity.get("git_available") is not True or source_identity.get(
            "git_dirty"
        ) is not False:
            raise MechanismDatasetError("formal generation lacks clean Git identity")
        empty_sha256 = hashlib.sha256(b"").hexdigest()
        if (
            source_identity.get("git_status_sha256") != empty_sha256
            or source_identity.get("git_tracked_diff_sha256") != empty_sha256
        ):
            raise MechanismDatasetError("formal generation Git fingerprints are not clean")
        if runtime_identity.get("is_conda") is not True:
            raise MechanismDatasetError("formal generation lacks conda identity")
        if api_transport_identity.get("transport") != "openai_compatible_chat_completions":
            raise MechanismDatasetError("formal generation lacks API transport identity")
        if (
            api_transport_identity.get("endpoint_scheme") != "https"
            or api_transport_identity.get("endpoint_is_loopback") is not False
        ):
            raise MechanismDatasetError(
                "formal generation endpoint must be non-loopback HTTPS"
            )
        if _source_identity() != source_identity:
            raise MechanismDatasetError(
                "formal generation source identity no longer matches the recorded identity"
            )
        if _runtime_identity() != runtime_identity:
            raise MechanismDatasetError(
                "formal generation runtime identity no longer matches the recorded identity"
            )
        if _api_transport_identity() != api_transport_identity:
            raise MechanismDatasetError(
                "formal generation API transport no longer matches the recorded identity"
            )
    elif formal_complete:
        raise MechanismDatasetError("formal complete generation has invalid run_kind")

    return {
        "schema_version": "p3_generation_artifact_audit_v1",
        "valid": True,
        "complete": formal_complete,
        "run_complete": run_complete,
        "run_kind": run_kind,
        "provider_kind": provider_kind,
        "planned_rule_count": len(target_rules),
        "completed_rule_count": len(completed_rule_ids),
        "dataset_case_count": len(dataset),
        "candidate_trace_count": len(candidate_trace),
        "raw_trace_record_count": len(raw_records),
        "orphan_raw_trace_record_count": len(expected_orphans),
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset_sha256": artifacts["dataset"]["sha256"],
        "candidate_trace_sha256": artifacts["candidate_trace"]["sha256"],
        "raw_response_trace_sha256": artifacts["raw_response_trace"]["sha256"],
        "configuration": configuration,
        "manifest": manifest,
        "plan": plan,
        "artifact_paths": {
            "catalog": str(bound_catalog),
            "plan": str(bound_plan),
            "dataset": str(bound_dataset),
            "candidate_trace": str(bound_candidate),
            "raw_response_trace": str(bound_raw),
            "checkpoint": str(bound_checkpoint),
            "manifest": str(manifest_path.resolve()),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a catalog-only 60-rule plan or generate the strict five-mechanism "
            "Gemini 3 Flash dataset and honest target-binding candidate trace."
        )
    )
    parser.add_argument("--catalog", default=str(PROJECT_ROOT / "catalogs" / "rules_unified_3000.json"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--prepare-plan-only", action="store_true")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--candidate-trace", default="")
    parser.add_argument("--raw-trace", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-rules", type=int, default=0, help="Generate only the first N frozen rules for smoke testing.")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-kind", choices=RUN_KINDS, default="development")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    catalog_path = Path(args.catalog)
    plan_path = Path(args.plan)
    try:
        if args.prepare_plan_only:
            if args.resume or args.max_rules:
                parser.error("--prepare-plan-only cannot be combined with --resume/--max-rules")
            plan = prepare_mechanism_plan(catalog_path=catalog_path, plan_path=plan_path)
            print(
                json.dumps(
                    {
                        "plan": str(plan_path),
                        "plan_sha256": plan["plan_sha256"],
                        "rule_count": len(plan["rules"]),
                        "summary": plan["summary"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        missing = [
            flag
            for flag, value in (
                ("--dataset", args.dataset),
                ("--candidate-trace", args.candidate_trace),
                ("--raw-trace", args.raw_trace),
                ("--checkpoint", args.checkpoint),
                ("--manifest", args.manifest),
            )
            if not value
        ]
        if missing:
            parser.error("normal generation requires " + ", ".join(missing))
        report = build_mechanism_dataset(
            catalog_path=catalog_path,
            plan_path=plan_path,
            dataset_path=Path(args.dataset),
            candidate_trace_path=Path(args.candidate_trace),
            raw_trace_path=Path(args.raw_trace),
            checkpoint_path=Path(args.checkpoint),
            manifest_path=Path(args.manifest),
            model=args.model,
            max_rules=args.max_rules,
            max_attempts=args.max_attempts,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            resume=args.resume,
            run_kind=args.run_kind,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["run_complete"] else 4
    except MechanismDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
