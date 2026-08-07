"""Implementation module for deterministic pre-cluster rule preparation.

User-facing workflow commands should go through scripts/unified_rules_pipeline.py.
This module stays importable for focused tests and lower-level debugging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_unified_catalog import (
    DEFAULT_SCENARIO_CLUSTER_BLUEPRINTS_PATH,
    _load_scenario_cluster_blueprints,
    _resolve_distilled_topic,
    build_unified_catalog_from_data,
    merge_scenario_cluster_blueprints,
)
from scripts.compare_unified_catalogs import compare_catalogs
from rule_framework.incremental_validation import (
    incremental_artifact_binding,
    sha256_file,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _console_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True, indent=2)


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _key_text(value: Any) -> str:
    return _text(value).casefold()


def _ordered_unique(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _rules(payload: Any) -> List[Dict[str, Any]]:
    raw_rules = payload.get("rules") if isinstance(payload, dict) else payload
    if not isinstance(raw_rules, list):
        return []
    return [item for item in raw_rules if isinstance(item, dict)]


def _validate_generalized_support(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if metadata.get("generator") != "experience_candidate_generalizer_v1":
        return False
    if metadata.get("scope_mode") != "full" or metadata.get("complete") is not True:
        raise ValueError("Formal rule preparation requires complete full-scope generalization output.")

    minimum_candidates = max(2, int(metadata.get("min_source_candidates") or 2))
    minimum_samples = max(2, int(metadata.get("min_source_samples") or 2))
    mappings: Dict[str, List[str]] = {}
    assigned_candidates: set[str] = set()
    for result in payload.get("cluster_results", []) or []:
        if not isinstance(result, dict):
            continue
        input_candidate_ids = set(_ordered_unique(result.get("input_candidate_ids") or []))
        for mapping in result.get("mappings", []) or []:
            if not isinstance(mapping, dict):
                continue
            rule_id = _text(mapping.get("rule_id") or "")
            source_ids = _ordered_unique(mapping.get("source_candidate_ids") or [])
            if not rule_id or rule_id in mappings:
                raise ValueError("Generalization output contains missing or duplicate rule mappings.")
            if not set(source_ids).issubset(input_candidate_ids):
                raise ValueError("Formal rule mapping references candidates outside its input batch.")
            overlap = assigned_candidates.intersection(source_ids)
            if overlap:
                raise ValueError("A source candidate is assigned to multiple formal rules.")
            assigned_candidates.update(source_ids)
            mappings[rule_id] = source_ids

    for rule in _rules(payload):
        rule_id = _text(rule.get("rule_id") or "")
        sample_ids = _ordered_unique(rule.get("sample_ids") or [])
        if len(mappings.get(rule_id, [])) < minimum_candidates:
            raise ValueError(f"Formal rule {rule_id!r} lacks multi-candidate support.")
        if len(sample_ids) < minimum_samples:
            raise ValueError(f"Formal rule {rule_id!r} lacks multi-sample support.")
    return True


def _rules_from_baseline_catalog(payload: Any) -> List[Dict[str, Any]]:
    """Keep the previously validated executable catalog as seed coverage."""
    if not isinstance(payload, dict):
        return []
    out: List[Dict[str, Any]] = []
    for domain in payload.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = _text(domain.get("name") or "")
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            topic_name = _text(topic.get("name") or "")
            for rule in topic.get("rules", []) or []:
                if not isinstance(rule, dict):
                    continue
                summary = _text(rule.get("summary") or "")
                support = rule.get("support") if isinstance(rule.get("support"), dict) else {}
                out.append(
                    {
                        "rule_id": _text(rule.get("rule_id") or ""),
                        "domain": domain_name,
                        "topic": topic_name,
                        "title": _text(rule.get("title") or ""),
                        "summary": summary,
                        "trigger": _text(rule.get("trigger") or ""),
                        "check_logic": _text(rule.get("check_logic") or ""),
                        "error_type": _text(rule.get("error_type") or "logic") or "logic",
                        "symbolic_hint": rule.get("symbolic_hint") if isinstance(rule.get("symbolic_hint"), dict) else {},
                        "auxiliary": {
                            "node_summary": summary,
                            "scene_cues": [],
                            "boundary_cues": [],
                            "explore_cues": [],
                            "evidence_sample_ids": support.get("sample_ids") or [],
                        },
                        "count": int(support.get("count") or 1),
                        "sample_ids": support.get("sample_ids") or [],
                    }
                )
    return out


def _topic_key(domain: str, topic: str) -> str:
    return f"{domain}::{topic}"


def _safe_symbolic_hint(rule: Dict[str, Any]) -> Dict[str, Any]:
    raw = rule.get("symbolic_hint") if isinstance(rule.get("symbolic_hint"), dict) else {}
    return {
        "primitive": _text(raw.get("primitive") or "none") or "none",
        "canonical": _text(raw.get("canonical") or ""),
        "required_symbols": sorted(_ordered_unique(raw.get("required_symbols") or [])),
    }


def _symbolic_empty(rule: Dict[str, Any]) -> bool:
    hint = rule.get("symbolic_hint") if isinstance(rule.get("symbolic_hint"), dict) else {}
    primitive = _text(hint.get("primitive") or "")
    canonical = _text(hint.get("canonical") or "")
    symbols = _ordered_unique(hint.get("required_symbols") or [])
    return not primitive and not canonical and not symbols


def _auxiliary(rule: Dict[str, Any]) -> Dict[str, Any]:
    raw = rule.get("auxiliary") if isinstance(rule.get("auxiliary"), dict) else {}
    return {
        "node_summary": _text(raw.get("node_summary") or ""),
        "scene_cues": _ordered_unique(raw.get("scene_cues") or []),
        "boundary_cues": _ordered_unique(raw.get("boundary_cues") or []),
        "explore_cues": _ordered_unique(raw.get("explore_cues") or []),
        "evidence_sample_ids": _ordered_unique(raw.get("evidence_sample_ids") or []),
    }


def _has_text(aux: Dict[str, Any], key: str) -> bool:
    return bool(_text(aux.get(key)))


def _has_list(aux: Dict[str, Any], key: str) -> bool:
    return isinstance(aux.get(key), list) and any(_text(item) for item in aux.get(key) or [])


def _stable_rule_id(parts: Tuple[str, ...]) -> str:
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"norm_{digest}"


def _rule_summary(*, title: str, trigger: str, check_logic: str, auxiliary: Dict[str, Any]) -> str:
    """NaviRAG-style rule summary: short node-level meaning, not a static boundary list."""
    node_summary = _text(auxiliary.get("node_summary") or "")
    if node_summary:
        return node_summary[:160]
    if title and check_logic:
        return f"{title}: {check_logic}"[:160]
    if title and trigger:
        return f"{title}: {trigger}"[:160]
    return (title or trigger or check_logic)[:160]


def _canonical_rule(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw_domain = _text(raw.get("domain") or "Unknown")
    raw_topic = _text(raw.get("topic") or "Unknown")
    domain, topic = _resolve_distilled_topic(raw_domain, raw_topic)
    aux = _auxiliary(raw)
    sample_ids = _ordered_unique(raw.get("sample_ids") or [])
    aux["evidence_sample_ids"] = _ordered_unique(list(aux.get("evidence_sample_ids") or []) + sample_ids)
    return {
        "source_rule_id": _text(raw.get("rule_id") or ""),
        "domain": domain,
        "topic": topic,
        "title": _text(raw.get("title") or ""),
        "trigger": _text(raw.get("trigger") or ""),
        "check_logic": _text(raw.get("check_logic") or ""),
        "error_type": _text(raw.get("error_type") or "logic") or "logic",
        "symbolic_hint": _safe_symbolic_hint(raw),
        "auxiliary": aux,
        "count": int(raw.get("count") or 0),
        "sample_ids": sample_ids,
        "_raw_domain": raw_domain,
        "_raw_topic": raw_topic,
    }


def _merge_key(rule: Dict[str, Any]) -> Tuple[str, ...]:
    hint = rule["symbolic_hint"]
    return (
        _topic_key(rule["domain"], rule["topic"]),
        _key_text(rule["title"]),
        _key_text(rule["trigger"]),
        _key_text(rule["check_logic"]),
        _key_text(rule["error_type"]),
        _key_text(hint.get("primitive")),
        _key_text(hint.get("canonical")),
        ",".join(hint.get("required_symbols") or []),
    )


def _merge_auxiliary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    node_summary = next((_text(item.get("node_summary") or "") for item in items if _text(item.get("node_summary"))), "")
    return {
        "node_summary": node_summary,
        "scene_cues": _ordered_unique(cue for item in items for cue in item.get("scene_cues") or []),
        "boundary_cues": _ordered_unique(cue for item in items for cue in item.get("boundary_cues") or []),
        "explore_cues": _ordered_unique(cue for item in items for cue in item.get("explore_cues") or []),
        "evidence_sample_ids": _ordered_unique(sid for item in items for sid in item.get("evidence_sample_ids") or []),
    }


def _merge_group(key: Tuple[str, ...], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = items[0]
    sample_ids = _ordered_unique(sid for item in items for sid in item.get("sample_ids") or [])
    count = sum(int(item.get("count") or 0) for item in items)
    auxiliary = _merge_auxiliary([item["auxiliary"] for item in items])
    source_rule_ids = _ordered_unique(item.get("source_rule_id") for item in items)
    if len(source_rule_ids) == 1:
        rule_id = source_rule_ids[0]
    elif source_rule_ids:
        rule_id = _stable_rule_id(("merged_sources", *sorted(source_rule_ids)))
    else:
        rule_id = _stable_rule_id(key)
    return {
        "rule_id": rule_id,
        "domain": first["domain"],
        "topic": first["topic"],
        "title": first["title"],
        "summary": _rule_summary(
            title=first["title"],
            trigger=first["trigger"],
            check_logic=first["check_logic"],
            auxiliary=auxiliary,
        ),
        "trigger": first["trigger"],
        "check_logic": first["check_logic"],
        "error_type": first["error_type"],
        "symbolic_hint": first["symbolic_hint"],
        "auxiliary": auxiliary,
        "count": count if count > 0 else len(sample_ids),
        "sample_ids": sample_ids,
        "source_rule_ids": source_rule_ids,
    }


def _embedding_text(rule: Dict[str, Any]) -> str:
    hint = rule.get("symbolic_hint") if isinstance(rule.get("symbolic_hint"), dict) else {}
    aux = rule.get("auxiliary") if isinstance(rule.get("auxiliary"), dict) else {}
    parts = [
        f"summary: {_text(rule.get('summary') or '')}",
        f"title: {_text(rule.get('title') or '')}",
        f"trigger: {_text(rule.get('trigger') or '')}",
        f"check_logic: {_text(rule.get('check_logic') or '')}",
        f"error_type: {_text(rule.get('error_type') or '')}",
        f"symbolic: {_text(hint.get('canonical') or '')}",
        f"scene_cues: {'; '.join(_ordered_unique(aux.get('scene_cues') or []))}",
        f"explore_cues: {'; '.join(_ordered_unique(aux.get('explore_cues') or []))}",
    ]
    return "\n".join(part for part in parts if part.split(": ", 1)[-1].strip())


def _embedding_input_payload(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    skipped = 0
    for rule in rules:
        if not rule.get("rule_id") or not rule.get("summary"):
            skipped += 1
            continue
        topic_key = _topic_key(str(rule.get("domain") or ""), str(rule.get("topic") or ""))
        rows.append(
            {
                "rule_id": rule["rule_id"],
                "domain": rule["domain"],
                "topic": rule["topic"],
                "topic_key": topic_key,
                "title": rule["title"],
                "summary": rule["summary"],
                "trigger": rule["trigger"],
                "check_logic": rule["check_logic"],
                "error_type": rule["error_type"],
                "symbolic_hint": rule["symbolic_hint"],
                "auxiliary": rule["auxiliary"],
                "sample_ids": rule["sample_ids"],
                "source_rule_ids": rule["source_rule_ids"],
                "near_duplicate_key": f"{topic_key}::{rule['title']}",
                "embedding_text": _embedding_text(rule),
            }
        )
    return {
        "metadata": {
            "purpose": "topic_local_rule_embedding_clustering",
            "summary_source": "auxiliary.node_summary preferred; fallback to title/check_logic",
            "input_rule_count": len(rules),
            "rule_count": len(rows),
            "skipped_rule_count": skipped,
        },
        "rules": rows,
    }


def _catalog_topic_keys(catalog: Dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = _text(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            if isinstance(topic, dict):
                keys.add(_topic_key(domain_name, _text(topic.get("name") or "Unknown")))
    return keys


def _quality_report(
    *,
    canonical_rules: List[Dict[str, Any]],
    catalog_topic_keys: set[str],
    duplicate_groups: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    topic_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "rule_count": 0,
            "missing_symbolic_hint_count": 0,
            "rules_with_all_auxiliary_fields": 0,
            "error_types": Counter(),
        }
    )
    near_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    unmatched_topics: set[str] = set()
    normalization_changes = 0
    sample_ids: List[str] = []
    aux_counts = Counter()
    missing_symbolic = 0

    for rule in canonical_rules:
        topic_key = _topic_key(rule["domain"], rule["topic"])
        topic_stats[topic_key]["rule_count"] += 1
        topic_stats[topic_key]["error_types"][rule["error_type"]] += 1
        sample_ids.extend(rule.get("sample_ids") or [])
        if rule["_raw_domain"] != rule["domain"] or rule["_raw_topic"] != rule["topic"]:
            normalization_changes += 1
        if catalog_topic_keys and topic_key not in catalog_topic_keys:
            unmatched_topics.add(topic_key)
        if _symbolic_empty(rule):
            missing_symbolic += 1
            topic_stats[topic_key]["missing_symbolic_hint_count"] += 1

        aux = rule["auxiliary"]
        aux_fields = {
            "node_summary": _has_text(aux, "node_summary"),
            "scene_cues": _has_list(aux, "scene_cues"),
            "boundary_cues": _has_list(aux, "boundary_cues"),
            "explore_cues": _has_list(aux, "explore_cues"),
            "evidence_sample_ids": _has_list(aux, "evidence_sample_ids"),
        }
        for key, value in aux_fields.items():
            if value:
                aux_counts[f"rules_with_{key}"] += 1
        if all(aux_fields.values()):
            aux_counts["rules_with_all_auxiliary_fields"] += 1
            topic_stats[topic_key]["rules_with_all_auxiliary_fields"] += 1
        if rule["title"]:
            near_groups[(topic_key, _key_text(rule["title"]))].append(rule)

    near_duplicate_group_count = sum(1 for items in near_groups.values() if len(items) > 1)
    topics = {}
    for key, stat in topic_stats.items():
        topics[key] = {
            "rule_count": stat["rule_count"],
            "missing_symbolic_hint_count": stat["missing_symbolic_hint_count"],
            "rules_with_all_auxiliary_fields": stat["rules_with_all_auxiliary_fields"],
            "error_types": dict(sorted(stat["error_types"].items())),
        }
    topics = dict(sorted(topics.items(), key=lambda item: (-item[1]["rule_count"], item[0])))
    return {
        "total_rules": len(canonical_rules),
        "topic_bucket_count": len(topic_stats),
        "sample_id_count": len(set(sample_ids)),
        "topic_normalization_change_count": normalization_changes,
        "unmatched_topic_count": len(unmatched_topics),
        "unmatched_topics": sorted(unmatched_topics),
        "exact_duplicate_group_count": len(duplicate_groups),
        "near_duplicate_group_count": near_duplicate_group_count,
        "symbolic_hint": {
            "missing_or_empty_count": missing_symbolic,
            "coverage": ((len(canonical_rules) - missing_symbolic) / len(canonical_rules)) if canonical_rules else 0.0,
        },
        "auxiliary": {
            "rules_with_node_summary": int(aux_counts["rules_with_node_summary"]),
            "rules_with_scene_cues": int(aux_counts["rules_with_scene_cues"]),
            "rules_with_boundary_cues": int(aux_counts["rules_with_boundary_cues"]),
            "rules_with_explore_cues": int(aux_counts["rules_with_explore_cues"]),
            "rules_with_evidence_sample_ids": int(aux_counts["rules_with_evidence_sample_ids"]),
            "rules_with_all_auxiliary_fields": int(aux_counts["rules_with_all_auxiliary_fields"]),
        },
        "topics": topics,
    }


def _load_blueprints(paths: Sequence[Path] | None) -> Dict[str, List[Dict[str, Any]]]:
    if paths is None:
        paths = [DEFAULT_SCENARIO_CLUSTER_BLUEPRINTS_PATH]
    return merge_scenario_cluster_blueprints(*[_load_scenario_cluster_blueprints(path) for path in paths])


def prepare_rules_for_cluster(
    *,
    distilled_input: Path,
    knowledge_path: Path,
    tagged_path: Path,
    baseline_catalog_path: Path | None,
    distilled_output: Path,
    catalog_output: Path,
    report_output: Path,
    embedding_input_output: Path | None = None,
    scenario_cluster_blueprints_paths: Sequence[Path] | None = None,
    preserve_baseline_rule_ids: bool = False,
    incremental_manifest_path: Path | None = None,
) -> Dict[str, Any]:
    distilled_payload = _load_json(distilled_input)
    generalized_support_validated = _validate_generalized_support(distilled_payload)
    raw_rules = _rules(distilled_payload)
    baseline_rules = (
        _rules_from_baseline_catalog(_load_json(baseline_catalog_path))
        if baseline_catalog_path and baseline_catalog_path.exists()
        else []
    )
    all_raw_rules = raw_rules + baseline_rules
    canonical_distilled_rules = [_canonical_rule(rule) for rule in raw_rules]
    canonical_baseline_rules = [_canonical_rule(rule) for rule in baseline_rules]
    preserved_by_rule_id = 0
    covered_by_exact_content = 0
    if preserve_baseline_rule_ids:
        baseline_ids = {
            rule["source_rule_id"]
            for rule in canonical_baseline_rules
            if rule["source_rule_id"]
        }
        baseline_keys = {_merge_key(rule) for rule in canonical_baseline_rules}
        additions: List[Dict[str, Any]] = []
        for rule in canonical_distilled_rules:
            if rule["source_rule_id"] in baseline_ids:
                preserved_by_rule_id += 1
                continue
            if _merge_key(rule) in baseline_keys:
                covered_by_exact_content += 1
                continue
            additions.append(rule)
        canonical_rules = canonical_baseline_rules + additions
    else:
        canonical_rules = canonical_distilled_rules + canonical_baseline_rules
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for rule in canonical_rules:
        groups[_merge_key(rule)].append(rule)
    duplicate_groups = [items for items in groups.values() if len(items) > 1]
    normalized_rules = [_merge_group(key, items) for key, items in groups.items()]
    normalized_rules.sort(key=lambda item: (item["domain"], item["topic"], item["title"], item["rule_id"]))

    normalized_payload = {
        "metadata": (
            {
                **dict(distilled_payload.get("metadata") or {}),
                **(
                    {"formal_support_validated": True}
                    if generalized_support_validated
                    else {}
                ),
            }
            if isinstance(distilled_payload, dict)
            and isinstance(distilled_payload.get("metadata"), dict)
            else {}
        ),
        "summary": {
            "source": str(distilled_input),
            "input_rules": len(all_raw_rules),
            "distilled_input_rules": len(raw_rules),
            "baseline_seed_rules": len(baseline_rules),
            "output_rules": len(normalized_rules),
            "topic_bucket_count": len({_topic_key(rule["domain"], rule["topic"]) for rule in canonical_rules}),
            "topic_normalization_change_count": sum(
                1
                for rule in canonical_rules
                if rule["_raw_domain"] != rule["domain"] or rule["_raw_topic"] != rule["topic"]
            ),
            "merged_exact_duplicate_groups": len(duplicate_groups),
            "merged_exact_duplicate_rules": sum(len(items) for items in duplicate_groups),
            "preserve_baseline_rule_ids": bool(preserve_baseline_rule_ids),
            "preserved_by_rule_id": preserved_by_rule_id,
            "covered_by_exact_content": covered_by_exact_content,
        },
        "rules": normalized_rules,
    }
    if incremental_manifest_path is not None:
        formal_inputs = {"generalized": distilled_input}
        if baseline_catalog_path is not None:
            formal_inputs["base_catalog"] = baseline_catalog_path
        normalized_payload["metadata"].update(
            incremental_artifact_binding(
                incremental_manifest_path,
                stage="formal_preparation",
                input_paths=formal_inputs,
            )
        )
    _write_json(distilled_output, normalized_payload)
    embedding_payload = _embedding_input_payload(normalized_rules)
    if incremental_manifest_path is not None:
        embedding_payload["metadata"].update(
            incremental_artifact_binding(
                incremental_manifest_path,
                stage="formal_embedding_input",
                input_paths={"formal_rules": distilled_output},
            )
        )
    if embedding_input_output:
        _write_json(embedding_input_output, embedding_payload)

    knowledge_data = _load_json(knowledge_path)
    tagged_data = _load_json(tagged_path)
    blueprints = _load_blueprints(scenario_cluster_blueprints_paths)
    catalog = build_unified_catalog_from_data(knowledge_data, normalized_payload, tagged_data, blueprints)
    if incremental_manifest_path is not None:
        catalog_inputs = {
            "generalized": distilled_input,
            "formal_rules": distilled_output,
            "knowledge": knowledge_path,
            "tagged": tagged_path,
        }
        if baseline_catalog_path is not None:
            catalog_inputs["base_catalog"] = baseline_catalog_path
        catalog["metadata"].update(
            incremental_artifact_binding(
                incremental_manifest_path,
                stage="precluster_catalog",
                input_paths=catalog_inputs,
            )
        )
    _write_json(catalog_output, catalog)

    catalog_topics = _catalog_topic_keys(catalog)
    quality = _quality_report(
        canonical_rules=canonical_rules,
        catalog_topic_keys=catalog_topics,
        duplicate_groups=duplicate_groups,
    )
    comparison = (
        compare_catalogs(baseline_catalog_path, catalog_output, output_path=None)
        if baseline_catalog_path
        else {}
    )
    meta = catalog.get("metadata") if isinstance(catalog.get("metadata"), dict) else {}
    report = {
        "inputs": {
            "distilled": str(distilled_input),
            "knowledge": str(knowledge_path),
            "tagged": str(tagged_path),
            "baseline_catalog": str(baseline_catalog_path) if baseline_catalog_path else "",
        },
        "outputs": {
            "distilled_for_cluster": str(distilled_output),
            "catalog_for_cluster": str(catalog_output),
            "embedding_input": str(embedding_input_output) if embedding_input_output else "",
            "report": str(report_output),
        },
        "normalization": normalized_payload["summary"],
        "embedding_input": embedding_payload["metadata"],
        "quality": quality,
        "catalog": {
            "schema_profile": meta.get("schema_profile"),
            "topics_with_rules": meta.get("topics_with_rules"),
            "total_executable_rules": meta.get("total_executable_rules"),
            "total_scenario_clusters": meta.get("total_scenario_clusters"),
        },
        "comparison": comparison.get("summary", {}),
        "cluster_coverage": comparison.get("cluster_coverage", {}),
    }
    if incremental_manifest_path is not None:
        report["incremental_artifacts"] = {
            "configuration_sha256": normalized_payload["metadata"].get(
                "incremental_configuration_sha256"
            ),
            "sha256": {
                "formal_rules": sha256_file(distilled_output),
                "formal_embedding_input": (
                    sha256_file(embedding_input_output)
                    if embedding_input_output is not None
                    else ""
                ),
                "precluster_catalog": sha256_file(catalog_output),
            },
        }
    _write_json(report_output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the single deterministic rule set used before cluster completion.")
    parser.add_argument("--distilled-input", default="results/unified_rules_3000/semantic_experience_distilled.json")
    parser.add_argument("--knowledge", default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--tagged", default="catalogs/rules_300_tagged.json")
    parser.add_argument("--baseline-catalog", default="catalogs/rules_unified.json")
    parser.add_argument(
        "--preserve-baseline-rule-ids",
        action="store_true",
        help=(
            "Keep the baseline catalog authoritative and add only genuinely new "
            "generalized rules."
        ),
    )
    parser.add_argument("--distilled-output", default="results/unified_rules_3000/semantic_experience_distilled_for_cluster.json")
    parser.add_argument("--catalog-output", default="catalogs/rules_unified_3000.json")
    parser.add_argument("--report-output", default="results/unified_rules_3000/precluster_report.json")
    parser.add_argument("--embedding-input-output", default="results/unified_rules_3000/rule_embedding_input.json")
    parser.add_argument(
        "--scenario-cluster-blueprints",
        action="append",
        default=None,
        help="Repeat to merge blueprint sources. Defaults to catalogs/scenario_cluster_blueprints.json.",
    )
    parser.add_argument(
        "--no-scenario-cluster-blueprints",
        action="store_true",
        help="Build the precluster catalog without any scenario-cluster seed.",
    )
    parser.add_argument(
        "--incremental-manifest",
        default="",
        help="Optional schema-v2 manifest used to bind deterministic outputs.",
    )
    args = parser.parse_args()
    if args.no_scenario_cluster_blueprints and args.scenario_cluster_blueprints:
        parser.error(
            "--no-scenario-cluster-blueprints cannot be combined with "
            "--scenario-cluster-blueprints"
        )

    report = prepare_rules_for_cluster(
        distilled_input=Path(args.distilled_input),
        knowledge_path=Path(args.knowledge),
        tagged_path=Path(args.tagged),
        baseline_catalog_path=Path(args.baseline_catalog) if args.baseline_catalog else None,
        distilled_output=Path(args.distilled_output),
        catalog_output=Path(args.catalog_output),
        report_output=Path(args.report_output),
        embedding_input_output=Path(args.embedding_input_output) if args.embedding_input_output else None,
        scenario_cluster_blueprints_paths=(
            []
            if args.no_scenario_cluster_blueprints
            else (
                [Path(item) for item in args.scenario_cluster_blueprints]
                if args.scenario_cluster_blueprints is not None
                else None
            )
        ),
        preserve_baseline_rule_ids=bool(args.preserve_baseline_rule_ids),
        incremental_manifest_path=(
            Path(args.incremental_manifest) if args.incremental_manifest else None
        ),
    )
    print(_console_json({"normalization": report["normalization"], "catalog": report["catalog"]}))


if __name__ == "__main__":
    main()
