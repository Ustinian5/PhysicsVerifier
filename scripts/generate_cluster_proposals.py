from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rule_framework.incremental_validation import (
    incremental_artifact_binding,
    incremental_manifest_configuration_sha256,
)

try:
    import httpx
except ImportError:  # pragma: no cover - environment dependent
    httpx = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment dependent
    OpenAI = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - environment dependent
    load_dotenv = None


CLUSTER_LABEL_PROMPT_VERSION = "embedding-cluster-labeling-v1"
CLUSTER_LABEL_SYSTEM_PROMPT = (
    "You are labeling embedding-derived scenario clusters for a physics rule navigation tree. The cluster "
    "membership is fixed; do not add, remove, or move rules. Produce concise English names and summaries that "
    "help a top-down semantic navigator choose the right cluster. Use scene_cues, boundary_cues, and explore_cues "
    "as auxiliary navigation notes. Return JSON only."
)


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _ordered_unique(items: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        text = _norm_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _topic_key(domain: str, topic: str) -> str:
    return f"{_norm_text(domain).lower()}::{_norm_text(topic).lower()}"


def _embedding_topic_fingerprint(topic_item: Dict[str, Any]) -> str:
    membership = {
        "topic_key": _norm_text(topic_item.get("topic_key") or "").casefold(),
        "clusters": [
            {
                "cluster_id": _norm_text(cluster.get("cluster_id") or ""),
                "rule_ids": sorted(_ordered_unique(cluster.get("rule_ids") or [])),
            }
            for cluster in (topic_item.get("clusters") or [])
            if isinstance(cluster, dict)
        ],
        "residual_rule_ids": sorted(
            _ordered_unique(topic_item.get("residual_rule_ids") or [])
        ),
    }
    encoded = json.dumps(
        membership,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _incremental_configuration_binding(path_value: str) -> Dict[str, str]:
    raw = str(path_value or "").strip()
    if not raw:
        return {}
    path = Path(raw)
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Incremental manifest must contain a JSON object: {path}")
    configuration_sha256 = str(payload.get("configuration_sha256") or "").strip()
    if not configuration_sha256:
        raise ValueError(
            f"Incremental manifest is missing configuration_sha256: {path}"
        )
    actual_configuration_sha256 = incremental_manifest_configuration_sha256(
        payload
    )
    if configuration_sha256 != actual_configuration_sha256:
        raise ValueError(
            "Incremental manifest configuration_sha256 does not match its current "
            f"configuration: {path}"
        )
    return {
        "incremental_configuration_sha256": configuration_sha256,
        "incremental_manifest": str(path).replace("\\", "/"),
    }


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_client(*, api_key: str, base_url: str | None, trust_env: bool, request_timeout: float) -> Any:
    if not OpenAI:
        raise RuntimeError("OpenAI package is not available.")
    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "timeout": request_timeout,
        "max_retries": 0,
    }
    if httpx is not None:
        kwargs["http_client"] = httpx.Client(trust_env=trust_env, timeout=request_timeout)
    return OpenAI(**kwargs)


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = _norm_text(text)
    if not raw:
        raise RuntimeError("Cluster proposal model returned empty content.")

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S | re.I)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    loose = re.search(r"\{.*\}", raw, flags=re.S)
    if loose:
        try:
            data = json.loads(loose.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    preview = raw[:300]
    hint = ""
    if raw.startswith("{") and not raw.endswith("}"):
        hint = " The response looks truncated; increase --max-output-tokens or reduce the topic batch."
    raise RuntimeError(f"Cluster proposal model did not return a valid JSON object.{hint} Preview: {preview}")


def _contains_cjk(text: str) -> bool:
    src = _norm_text(text)
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", src))


def _find_cjk_proposal_fields(raw: Dict[str, Any]) -> List[str]:
    fields_to_check: List[str] = [
        _norm_text(raw.get("topic_summary") or ""),
        _norm_text(raw.get("rationale") or ""),
    ]
    for cluster in raw.get("clusters", []) or []:
        if not isinstance(cluster, dict):
            continue
        fields_to_check.extend(
            [
                _norm_text(cluster.get("cluster_id") or ""),
                _norm_text(cluster.get("name") or ""),
                _norm_text(cluster.get("summary") or ""),
                _norm_text(cluster.get("description") or ""),
                *[_norm_text(item) for item in (cluster.get("scene_cues") or cluster.get("includes") or [])],
                *[_norm_text(item) for item in (cluster.get("boundary_cues") or cluster.get("excludes") or [])],
                *[_norm_text(item) for item in (cluster.get("explore_cues") or cluster.get("entry_cues") or [])],
            ]
        )
    return [text for text in fields_to_check if _contains_cjk(text)]


def _assert_english_only_proposal(raw: Dict[str, Any]) -> None:
    offenders = _find_cjk_proposal_fields(raw)
    if offenders:
        raise RuntimeError(
            "Cluster proposal must be English-only for all generated semantic fields. "
            f"Offending preview: {offenders[0][:120]}"
        )


def _chat_json(
    client: Any,
    *,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int | None = None,
) -> Dict[str, Any]:
    request: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if max_output_tokens:
        request["max_tokens"] = int(max_output_tokens)
    response = client.chat.completions.create(**request)
    content = response.choices[0].message.content if response.choices else ""
    parsed = _extract_json_object(_norm_text(content))
    if not isinstance(parsed, dict):
        raise RuntimeError("Cluster proposal model must return a JSON object.")
    return parsed


def _parse_csv_values(raw_values: Sequence[str] | None) -> set[str]:
    out: set[str] = set()
    for raw in raw_values or []:
        for part in str(raw).split(","):
            text = _norm_text(part)
            if text:
                out.add(text)
    return out


def _build_distilled_auxiliary_index(distilled_payload: Dict[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    if not isinstance(distilled_payload, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for rule in distilled_payload.get("rules", []) or []:
        if not isinstance(rule, dict):
            continue
        rule_id = _norm_text(rule.get("rule_id") or "")
        aux = rule.get("auxiliary") if isinstance(rule.get("auxiliary"), dict) else {}
        if not rule_id:
            continue
        out[rule_id] = {
            "node_summary": _norm_text(aux.get("node_summary") or ""),
            "scene_cues": _ordered_unique(aux.get("scene_cues") or []),
            "boundary_cues": _ordered_unique(aux.get("boundary_cues") or []),
            "explore_cues": _ordered_unique(aux.get("explore_cues") or []),
            "evidence_sample_ids": _ordered_unique(aux.get("evidence_sample_ids") or []),
        }
    return out


def _build_rule_index(rule_input_payload: Dict[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    if not isinstance(rule_input_payload, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for rule in rule_input_payload.get("rules", []) or []:
        if not isinstance(rule, dict):
            continue
        rule_id = _norm_text(rule.get("rule_id") or "")
        if rule_id:
            out[rule_id] = rule
    return out


def _collect_topic_candidates(
    catalog: Dict[str, Any],
    *,
    only_missing_clusters: bool,
    domain_filters: set[str],
    topic_filters: set[str],
    max_topics: int,
    min_rule_count: int,
) -> List[Dict[str, Any]]:
    topics: List[Dict[str, Any]] = []
    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = _norm_text(domain.get("name") or "Unknown")
        if domain_filters and domain_name not in domain_filters:
            continue
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            topic_name = _norm_text(topic.get("name") or "Unknown")
            if topic_filters and topic_name not in topic_filters and _topic_key(domain_name, topic_name) not in topic_filters:
                continue
            rules = [item for item in (topic.get("rules") or []) if isinstance(item, dict)]
            scenario_clusters = topic.get("scenario_clusters") or []
            if only_missing_clusters and scenario_clusters:
                continue
            if len(rules) < min_rule_count:
                continue
            topics.append(
                {
                    "domain": domain_name,
                    "topic": topic_name,
                    "topic_obj": topic,
                    "rule_count": len(rules),
                    "has_clusters": bool(scenario_clusters),
                }
            )
    topics.sort(key=lambda item: (-int(item["rule_count"]), item["domain"], item["topic"]))
    if max_topics > 0:
        topics = topics[:max_topics]
    return topics


def _build_topic_prompt_payload(
    topic_match: Dict[str, Any],
    *,
    auxiliary_by_rule: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    topic = topic_match["topic_obj"]
    rules = [item for item in (topic.get("rules") or []) if isinstance(item, dict)]
    auxiliary_by_rule = auxiliary_by_rule or {}
    return {
        "domain": topic_match["domain"],
        "topic": topic_match["topic"],
        "topic_summary": _norm_text(topic.get("summary") or topic.get("description") or ""),
        "existing_scenario_clusters": [
            {
                "cluster_id": _norm_text(cluster.get("id") or cluster.get("cluster_id") or ""),
                "name": _norm_text(cluster.get("name") or ""),
                "summary": _norm_text(cluster.get("summary") or cluster.get("description") or ""),
                "rule_ids": _ordered_unique(cluster.get("rule_ids") or []),
            }
            for cluster in (topic.get("scenario_clusters") or [])
            if isinstance(cluster, dict)
        ],
        "rules": [
            {
                "rule_id": _norm_text(rule.get("rule_id") or ""),
                "title": _norm_text(rule.get("title") or ""),
                "summary": _norm_text(rule.get("summary") or ""),
                "trigger": _norm_text(rule.get("trigger") or ""),
                "check_logic": _norm_text(rule.get("check_logic") or ""),
                "error_type": _norm_text(rule.get("error_type") or ""),
                "symbolic_hint": rule.get("symbolic_hint") if isinstance(rule.get("symbolic_hint"), dict) else {},
                "auxiliary": auxiliary_by_rule.get(_norm_text(rule.get("rule_id") or ""), {}),
            }
            for rule in rules
        ],
        "output_schema": {
            "topic_summary": "string",
            "should_add_clusters": True,
            "rationale": "string",
            "clusters": [
                {
                    "cluster_id": "string",
                    "name": "string",
                    "summary": "string",
                    "description": "string",
                    "scene_cues": ["string"],
                    "boundary_cues": ["string"],
                    "explore_cues": ["string"],
                    "candidate_rule_ids": ["string"],
                }
            ],
            "residual_rule_ids": ["string"],
        },
    }


def _build_embedding_topic_prompt_payload(
    topic_item: Dict[str, Any],
    *,
    rule_index: Dict[str, Dict[str, Any]],
    max_rules_per_cluster: int,
) -> Dict[str, Any]:
    clusters = []
    for cluster in topic_item.get("clusters", []) or []:
        if not isinstance(cluster, dict):
            continue
        rule_ids = _ordered_unique(cluster.get("rule_ids") or [])
        sampled_rules = []
        for rule_id in rule_ids[:max_rules_per_cluster]:
            rule = rule_index.get(rule_id, {})
            sampled_rules.append(
                {
                    "rule_id": rule_id,
                    "title": _norm_text(rule.get("title") or ""),
                    "summary": _norm_text(rule.get("summary") or ""),
                    "trigger": _norm_text(rule.get("trigger") or ""),
                    "check_logic": _norm_text(rule.get("check_logic") or ""),
                    "error_type": _norm_text(rule.get("error_type") or ""),
                    "symbolic_hint": rule.get("symbolic_hint") if isinstance(rule.get("symbolic_hint"), dict) else {},
                    "auxiliary": rule.get("auxiliary") if isinstance(rule.get("auxiliary"), dict) else {},
                }
            )
        clusters.append(
            {
                "source_cluster_id": _norm_text(cluster.get("cluster_id") or ""),
                "size": int(cluster.get("size") or len(rule_ids)),
                "representative_rules": cluster.get("representative_rules") or [],
                "sampled_rules": sampled_rules,
            }
        )
    return {
        "domain": _norm_text(topic_item.get("domain") or ""),
        "topic": _norm_text(topic_item.get("topic") or ""),
        "topic_key": _norm_text(topic_item.get("topic_key") or ""),
        "rule_count": int(topic_item.get("rule_count") or 0),
        "embedding_clusters": clusters,
        "residual_rule_count": len(_ordered_unique(topic_item.get("residual_rule_ids") or [])),
        "task": (
            "Name and summarize each embedding cluster as a scenario cluster. Do not change rule assignments. "
            "Use concise NaviRAG-style summaries for tree navigation."
        ),
        "output_schema": {
            "topic_summary": "string",
            "rationale": "string",
            "clusters": [
                {
                    "source_cluster_id": "must match one input source_cluster_id",
                    "cluster_id": "stable snake_case id",
                    "name": "short scenario cluster name",
                    "summary": "1-2 sentence navigation summary",
                    "description": "slightly fuller scenario description",
                    "scene_cues": ["typical problem scenes"],
                    "boundary_cues": ["semantic boundaries or common confusions"],
                    "explore_cues": ["when navigation should explore this cluster"],
                }
            ],
        },
    }


def _cluster_label_resume_fingerprint(
    topic_item: Dict[str, Any],
    *,
    rule_index: Dict[str, Dict[str, Any]],
    system_prompt: str,
    model: str,
    temperature: float,
    max_topics: int,
    min_rule_count: int,
    max_rules_per_cluster: int,
    max_output_tokens: int | None,
    incremental_configuration_sha256: str = "",
    incremental_lineage: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "schema_version": 2,
        "prompt_version": CLUSTER_LABEL_PROMPT_VERSION,
        "full_membership_sha256": _embedding_topic_fingerprint(topic_item),
        "system_prompt": system_prompt,
        "user_payload": _build_embedding_topic_prompt_payload(
            topic_item,
            rule_index=rule_index,
            max_rules_per_cluster=max_rules_per_cluster,
        ),
        "model": model,
        "temperature": temperature,
        "max_topics": max_topics,
        "min_rule_count": min_rule_count,
        "max_rules_per_cluster": max_rules_per_cluster,
        "max_output_tokens": max_output_tokens,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_cluster_proposal(raw: Dict[str, Any], valid_rule_ids: set[str]) -> Dict[str, Any]:
    clusters: List[Dict[str, Any]] = []
    used_rule_ids: set[str] = set()
    for item in raw.get("clusters", []) or []:
        if not isinstance(item, dict):
            continue
        candidate_rule_ids = [rid for rid in _ordered_unique(item.get("candidate_rule_ids") or []) if rid in valid_rule_ids]
        used_rule_ids.update(candidate_rule_ids)
        clusters.append(
            {
                "cluster_id": _norm_text(item.get("cluster_id") or ""),
                "name": _norm_text(item.get("name") or ""),
                "summary": _norm_text(item.get("summary") or ""),
                "description": _norm_text(item.get("description") or ""),
                "scene_cues": _ordered_unique(item.get("scene_cues") or item.get("entry_cues") or []),
                "boundary_cues": _ordered_unique(item.get("boundary_cues") or item.get("excludes") or []),
                "explore_cues": _ordered_unique(item.get("explore_cues") or []),
                "candidate_rule_ids": candidate_rule_ids,
            }
        )
    residual_rule_ids = [rid for rid in _ordered_unique(raw.get("residual_rule_ids") or []) if rid in valid_rule_ids and rid not in used_rule_ids]
    return {
        "topic_summary": _norm_text(raw.get("topic_summary") or ""),
        "should_add_clusters": bool(raw.get("should_add_clusters")),
        "rationale": _norm_text(raw.get("rationale") or ""),
        "clusters": clusters,
        "residual_rule_ids": residual_rule_ids,
    }


def _normalize_embedding_cluster_labels(raw: Dict[str, Any], topic_item: Dict[str, Any]) -> Dict[str, Any]:
    source_clusters = [
        item for item in (topic_item.get("clusters") or []) if isinstance(item, dict)
    ]
    labels_by_source: Dict[str, Dict[str, Any]] = {}
    for item in raw.get("clusters", []) or []:
        if not isinstance(item, dict):
            continue
        source_id = _norm_text(item.get("source_cluster_id") or "")
        if source_id:
            labels_by_source[source_id] = item

    clusters: List[Dict[str, Any]] = []
    used_cluster_ids: set[str] = set()
    topic_label = re.sub(r"[^A-Za-z0-9 ]+", " ", _norm_text(topic_item.get("topic") or "")).strip()
    topic_label = re.sub(r"\s+", " ", topic_label)
    topic_label = " ".join(topic_label.split()[:5]).title()
    for index, source in enumerate(source_clusters, start=1):
        source_id = _norm_text(source.get("cluster_id") or f"embedding_cluster_{index:02d}")
        label = labels_by_source.get(source_id, {})
        fallback_name = f"{topic_label} Cluster {index:02d}" if topic_label else source_id.replace("_", " ").title()
        fallback_summary = f"Embedding-derived cluster for {topic_label or 'this topic'} rules."
        cluster_id = _norm_text(label.get("cluster_id") or source_id)
        normalized_cluster_id = re.sub(r"[^a-z0-9_]+", "_", cluster_id.lower()).strip("_") or source_id
        base_cluster_id = normalized_cluster_id
        suffix = 2
        while normalized_cluster_id in used_cluster_ids:
            normalized_cluster_id = f"{base_cluster_id}_{suffix:02d}"
            suffix += 1
        used_cluster_ids.add(normalized_cluster_id)
        clusters.append(
            {
                "cluster_id": normalized_cluster_id,
                "name": _norm_text(label.get("name") or fallback_name),
                "summary": _norm_text(label.get("summary") or fallback_summary),
                "description": _norm_text(label.get("description") or label.get("summary") or fallback_summary),
                "scene_cues": _ordered_unique(label.get("scene_cues") or []),
                "boundary_cues": _ordered_unique(label.get("boundary_cues") or []),
                "explore_cues": _ordered_unique(label.get("explore_cues") or []),
                "candidate_rule_ids": _ordered_unique(source.get("rule_ids") or []),
            }
        )
    return {
        "topic_summary": _norm_text(raw.get("topic_summary") or ""),
        "should_add_clusters": bool(clusters),
        "rationale": _norm_text(raw.get("rationale") or ""),
        "clusters": clusters,
        "residual_rule_ids": _ordered_unique(topic_item.get("residual_rule_ids") or []),
    }


def add_catalog_fallback_proposals(payload: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
    proposals = [item for item in (payload.get("proposals") or []) if isinstance(item, dict)]
    completed_topic_keys = {
        _norm_text(item.get("topic_key") or "").casefold()
        for item in proposals
    }
    added = 0
    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = _norm_text(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            topic_name = _norm_text(topic.get("name") or "Unknown")
            topic_key = _topic_key(domain_name, topic_name)
            if topic_key in completed_topic_keys:
                continue
            rule_ids = _ordered_unique(
                rule.get("rule_id")
                for rule in (topic.get("rules") or [])
                if isinstance(rule, dict)
            )
            if not rule_ids:
                continue
            proposals.append(
                {
                    "domain": domain_name,
                    "topic": topic_name,
                    "topic_key": topic_key,
                    "rule_count": len(rule_ids),
                    "existing_cluster_count": len(topic.get("scenario_clusters") or []),
                    "topic_summary": f"Catalog fallback proposal for {topic_name}.",
                    "should_add_clusters": True,
                    "rationale": "Generated locally from catalog rule membership because no labeled embedding proposal exists for this topic.",
                    "clusters": [],
                    "residual_rule_ids": rule_ids,
                    "label_source": "catalog_fallback",
                }
            )
            completed_topic_keys.add(topic_key)
            added += 1

    out = dict(payload)
    metadata = dict(out.get("metadata") or {})
    metadata["topic_count"] = len(proposals)
    metadata["target_topic_count"] = max(int(metadata.get("target_topic_count") or 0), len(proposals))
    metadata["failure_count"] = 0
    metadata["catalog_fallback_topic_count"] = int(metadata.get("catalog_fallback_topic_count") or 0) + added
    out["metadata"] = metadata
    out["proposals"] = proposals
    out["failures"] = [
        item for item in (payload.get("failures") or [])
        if isinstance(item, dict)
        and _norm_text(item.get("topic_key") or "").casefold() not in completed_topic_keys
    ]
    return out


def generate_cluster_proposals(
    *,
    catalog: Dict[str, Any],
    client: Any,
    model: str,
    temperature: float,
    only_missing_clusters: bool,
    domain_filters: set[str],
    topic_filters: set[str],
    max_topics: int,
    min_rule_count: int,
    auxiliary_by_rule: Dict[str, Dict[str, Any]] | None = None,
    max_output_tokens: int | None = None,
) -> Dict[str, Any]:
    topic_candidates = _collect_topic_candidates(
        catalog,
        only_missing_clusters=only_missing_clusters,
        domain_filters=domain_filters,
        topic_filters=topic_filters,
        max_topics=max_topics,
        min_rule_count=min_rule_count,
    )
    proposals: List[Dict[str, Any]] = []
    system_prompt = (
        "You are designing scenario clusters for a physics rule tree. Group rules by distinct audit scenario or "
        "failure mode, not by superficial keyword overlap and not by textbook chapter names. Prefer 2-4 clusters. "
        "Each cluster must represent a materially different reasoning scene, have clear semantic boundaries, and map "
        "to concrete rule_ids. Avoid over-fragmentation. If a topic is too small or too homogeneous to justify "
        "clusters, set should_add_clusters to false. Use English only for every generated field, including cluster "
        "names, summaries, descriptions, scene_cues, boundary_cues, explore_cues, and rationale. Do not output Chinese or "
        "mixed-language phrases. Return JSON only."
    )
    total_topics = len(topic_candidates)
    for index, topic_match in enumerate(topic_candidates, start=1):
        payload = _build_topic_prompt_payload(topic_match, auxiliary_by_rule=auxiliary_by_rule)
        print(
            f"[cluster-proposal] {index}/{total_topics} "
            f"{topic_match['domain']} / {topic_match['topic']} "
            f"rules={topic_match['rule_count']}",
            flush=True,
        )
        raw = _chat_json(
            client,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            max_output_tokens=max_output_tokens,
        )
        valid_rule_ids = {item["rule_id"] for item in payload["rules"]}
        normalized = _normalize_cluster_proposal(raw, valid_rule_ids)
        print(
            f"[cluster-proposal] done {index}/{total_topics} "
            f"clusters={len(normalized.get('clusters') or [])}",
            flush=True,
        )
        proposals.append(
            {
                "domain": topic_match["domain"],
                "topic": topic_match["topic"],
                "topic_key": _topic_key(topic_match["domain"], topic_match["topic"]),
                "rule_count": topic_match["rule_count"],
                "existing_cluster_count": len(topic_match["topic_obj"].get("scenario_clusters") or []),
                **normalized,
            }
        )
    return {
        "metadata": {
            "generator": "semantic_cluster_proposal_v1",
            "model": model,
            "only_missing_clusters": only_missing_clusters,
            "topic_count": len(proposals),
            "min_rule_count": min_rule_count,
        },
        "proposals": proposals,
    }


def generate_cluster_proposals_from_embedding_clusters(
    *,
    embedding_clusters: Dict[str, Any],
    rule_input: Dict[str, Any],
    client: Any,
    model: str,
    temperature: float,
    max_topics: int,
    min_rule_count: int,
    max_rules_per_cluster: int,
    max_output_tokens: int | None = None,
    output_path: Path | None = None,
    resume: bool = False,
    continue_on_error: bool = False,
    incremental_configuration_sha256: str = "",
    incremental_manifest: str = "",
    incremental_lineage: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    rule_index = _build_rule_index(rule_input)
    topics = [
        item
        for item in (embedding_clusters.get("topics") or [])
        if isinstance(item, dict)
        and int(item.get("rule_count") or 0) >= min_rule_count
        and item.get("clusters")
    ]
    topics.sort(key=lambda item: (-int(item.get("rule_count") or 0), _norm_text(item.get("topic_key") or "")))
    if max_topics > 0:
        topics = topics[:max_topics]

    system_prompt = CLUSTER_LABEL_SYSTEM_PROMPT
    proposals: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    if resume and output_path and output_path.exists():
        existing_payload = _load_json(output_path)
        proposals = [
            item for item in (existing_payload.get("proposals") or [])
            if isinstance(item, dict)
        ]
        failures = [
            item for item in (existing_payload.get("failures") or [])
            if isinstance(item, dict)
        ]
    expected_fingerprints = {
        _norm_text(item.get("topic_key") or "").casefold(): _cluster_label_resume_fingerprint(
            item,
            rule_index=rule_index,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_topics=max_topics,
            min_rule_count=min_rule_count,
            max_rules_per_cluster=max_rules_per_cluster,
            max_output_tokens=max_output_tokens,
            incremental_configuration_sha256=incremental_configuration_sha256,
            incremental_lineage=incremental_lineage,
        )
        for item in topics
    }
    proposals = [
        item
        for item in proposals
        if _norm_text(item.get("source_fingerprint") or "")
        == expected_fingerprints.get(_norm_text(item.get("topic_key") or "").casefold())
    ]
    for item in proposals:
        if not _norm_text(item.get("label_source") or ""):
            item["label_source"] = "model"
    # A deterministic embedding fallback is a recoverable checkpoint, not a
    # completed API label.  Resume must retry it after a transient model/network
    # failure; otherwise the strict finalizer can never reach a promotable state.
    proposals = [
        item
        for item in proposals
        if item.get("label_source") != "embedding_fallback"
    ]
    completed_topic_keys = {
        _norm_text(item.get("topic_key") or "").casefold() for item in proposals
    }
    original_failure_count = len(failures)
    failures = [
        item for item in failures
        if (
            _norm_text(item.get("topic_key") or "").casefold() in expected_fingerprints
            and _norm_text(item.get("topic_key") or "").casefold() not in completed_topic_keys
        )
    ]

    def _current_payload() -> Dict[str, Any]:
        metadata = {
            "generator": "embedding_cluster_labeling_v1",
            "model": model,
            "temperature": temperature,
            "max_topics": max_topics,
            "topic_count": len(proposals),
            "target_topic_count": total_topics,
            "failure_count": len(failures),
            "fallback_label_count": sum(
                1
                for item in proposals
                if item.get("label_source") == "embedding_fallback"
            ),
            "cjk_warning_count": sum(
                1 for item in proposals if item.get("contains_cjk_generated_text")
            ),
            "min_rule_count": min_rule_count,
            "max_rules_per_cluster": max_rules_per_cluster,
            "max_output_tokens": max_output_tokens,
            "prompt_version": CLUSTER_LABEL_PROMPT_VERSION,
            "resume_fingerprint_schema_version": 2,
        }
        if incremental_configuration_sha256:
            metadata["incremental_configuration_sha256"] = (
                incremental_configuration_sha256
            )
            metadata["incremental_manifest"] = incremental_manifest
            metadata["incremental_lineage"] = dict(incremental_lineage or {})
        return {
            "metadata": metadata,
            "proposals": proposals,
            "failures": failures,
        }

    total_topics = len(topics)
    if resume and output_path and output_path.exists() and len(failures) != original_failure_count:
        _dump_json(output_path, _current_payload())
    for index, topic_item in enumerate(topics, start=1):
        topic_key = _norm_text(topic_item.get("topic_key") or "").casefold()
        if topic_key in completed_topic_keys:
            print(f"[cluster-label] skip {index}/{total_topics} {topic_key}", flush=True)
            continue
        print(
            f"[cluster-label] {index}/{total_topics} "
            f"{topic_item.get('domain')} / {topic_item.get('topic')} "
            f"embedding_clusters={len(topic_item.get('clusters') or [])}",
            flush=True,
        )
        payload = _build_embedding_topic_prompt_payload(
            topic_item,
            rule_index=rule_index,
            max_rules_per_cluster=max_rules_per_cluster,
        )
        try:
            raw = _chat_json(
                client,
                model=model,
                temperature=temperature,
                system_prompt=system_prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            if continue_on_error:
                normalized = _normalize_embedding_cluster_labels(
                    {
                        "topic_summary": "Fallback labels generated from fixed embedding cluster membership.",
                        "rationale": f"Model labeling failed, so deterministic fallback labels were used: {exc}",
                    },
                    topic_item,
                )
                proposals.append(
                    {
                        "domain": _norm_text(topic_item.get("domain") or ""),
                        "topic": _norm_text(topic_item.get("topic") or ""),
                        "topic_key": topic_key,
                        "source_fingerprint": expected_fingerprints[topic_key],
                        "rule_count": int(topic_item.get("rule_count") or 0),
                        "existing_cluster_count": 0,
                        "contains_cjk_generated_text": False,
                        "cjk_generated_text_preview": "",
                        "label_source": "embedding_fallback",
                        **normalized,
                    }
                )
                failures = [
                    item for item in failures
                    if _norm_text(item.get("topic_key") or "").casefold() != topic_key
                ]
                completed_topic_keys.add(topic_key)
                if output_path:
                    _dump_json(output_path, _current_payload())
                    print(f"[cluster-label] fallback output saved to {output_path}", flush=True)
                print(f"[cluster-label] fallback {index}/{total_topics} {topic_key}: {exc}", flush=True)
                continue
            failures.append(
                {
                    "topic_key": topic_key,
                    "domain": _norm_text(topic_item.get("domain") or ""),
                    "topic": _norm_text(topic_item.get("topic") or ""),
                    "error": str(exc),
                }
            )
            if output_path:
                _dump_json(output_path, _current_payload())
                print(f"[cluster-label] partial output saved to {output_path}", flush=True)
            raise
        normalized = _normalize_embedding_cluster_labels(raw, topic_item)
        cjk_offenders = _find_cjk_proposal_fields(raw)
        failures = [
            item
            for item in failures
            if _norm_text(item.get("topic_key") or "").casefold() != topic_key
        ]
        proposals.append(
            {
                "domain": _norm_text(topic_item.get("domain") or ""),
                "topic": _norm_text(topic_item.get("topic") or ""),
                "topic_key": _norm_text(topic_item.get("topic_key") or "").casefold(),
                "source_fingerprint": expected_fingerprints[topic_key],
                "rule_count": int(topic_item.get("rule_count") or 0),
                "existing_cluster_count": 0,
                "contains_cjk_generated_text": bool(cjk_offenders),
                "cjk_generated_text_preview": cjk_offenders[0][:120] if cjk_offenders else "",
                "label_source": "model",
                **normalized,
            }
        )
        completed_topic_keys.add(topic_key)
        if output_path:
            _dump_json(output_path, _current_payload())
        print(
            f"[cluster-label] done {index}/{total_topics} "
            f"clusters={len(normalized.get('clusters') or [])}",
            flush=True,
        )
    return _current_payload()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scenario-cluster proposals from the current unified rules catalog.")
    parser.add_argument("--catalog", type=str, default="catalogs/rules_unified.json")
    parser.add_argument("--distilled-experience", type=str, default=None)
    parser.add_argument("--embedding-clusters", type=str, default=None)
    parser.add_argument("--rule-input", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/cluster_proposals.json")
    parser.add_argument("--model", type=str, default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--trust-env", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--domains", action="append", default=[], help="Repeat or comma-separate domain filters.")
    parser.add_argument("--topics", action="append", default=[], help="Repeat or comma-separate topic filters. Topic key format domain::topic is also accepted.")
    parser.add_argument("--max-topics", type=int, default=12)
    parser.add_argument("--min-rule-count", type=int, default=1)
    parser.add_argument("--all-topics", action="store_true", help="Include topics that already have scenario clusters.")
    parser.add_argument("--max-rules-per-cluster", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output file and skip completed topics.")
    parser.add_argument("--continue-on-error", action="store_true", help="Save failures and continue with remaining topics.")
    parser.add_argument(
        "--incremental-manifest",
        default="",
        help="Optional schema-v2 incremental manifest whose configuration hash binds this output.",
    )
    args = parser.parse_args()

    incremental_binding = _incremental_configuration_binding(
        args.incremental_manifest
    )
    if args.incremental_manifest:
        manifest_path = Path(args.incremental_manifest)
        manifest_payload = _load_json(manifest_path)
        base_proposal_record = (
            (manifest_payload.get("inputs") or {}).get("base_cluster_proposals")
            or {}
        )
        input_paths = {"precluster_catalog": Path(args.catalog)}
        if args.embedding_clusters:
            input_paths.update(
                {
                    "formal_clusters": Path(args.embedding_clusters),
                    "formal_rule_input": Path(args.rule_input),
                }
            )
        incremental_binding = incremental_artifact_binding(
            manifest_path,
            stage="cluster_labeling",
            input_paths=input_paths,
            input_sha256={
                "base_cluster_proposals": str(
                    base_proposal_record.get("sha256") or ""
                )
            },
        )

    if load_dotenv:
        load_dotenv()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    catalog = _load_json(Path(args.catalog))
    client = _build_client(
        api_key=api_key,
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or None,
        trust_env=bool(args.trust_env),
        request_timeout=float(args.request_timeout),
    )
    if args.embedding_clusters:
        rule_input_path = Path(args.rule_input) if args.rule_input else None
        if rule_input_path is None:
            raise RuntimeError("--rule-input is required when --embedding-clusters is used.")
        result = generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters=_load_json(Path(args.embedding_clusters)),
            rule_input=_load_json(rule_input_path),
            client=client,
            model=_norm_text(args.model),
            temperature=float(args.temperature),
            max_topics=int(args.max_topics),
            min_rule_count=int(args.min_rule_count),
            max_rules_per_cluster=int(args.max_rules_per_cluster),
            max_output_tokens=int(args.max_output_tokens),
            output_path=Path(args.output),
            resume=bool(args.resume),
            continue_on_error=bool(args.continue_on_error),
            incremental_configuration_sha256=incremental_binding.get(
                "incremental_configuration_sha256", ""
            ),
            incremental_manifest=incremental_binding.get(
                "incremental_manifest", ""
            ),
            incremental_lineage=incremental_binding.get(
                "incremental_lineage", {}
            ),
        )
        _dump_json(Path(args.output), result)
        print(f"Wrote cluster proposals to {args.output}")
        return
    distilled_payload = _load_json(Path(args.distilled_experience)) if args.distilled_experience else None
    result = generate_cluster_proposals(
        catalog=catalog,
        client=client,
        model=_norm_text(args.model),
        temperature=float(args.temperature),
        only_missing_clusters=not bool(args.all_topics),
        domain_filters=_parse_csv_values(args.domains),
        topic_filters=_parse_csv_values(args.topics),
        max_topics=int(args.max_topics),
        min_rule_count=int(args.min_rule_count),
        auxiliary_by_rule=_build_distilled_auxiliary_index(distilled_payload),
        max_output_tokens=int(args.max_output_tokens),
    )
    if incremental_binding:
        result.setdefault("metadata", {}).update(incremental_binding)
    _dump_json(Path(args.output), result)
    print(f"Wrote cluster proposals to {args.output}")


if __name__ == "__main__":
    main()
