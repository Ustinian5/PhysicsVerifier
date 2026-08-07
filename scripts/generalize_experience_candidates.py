from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    import openai
except ImportError as exc:  # pragma: no cover
    raise SystemExit("OpenAI package not found. Install project dependencies in the conda environment.") from exc

from rule_framework.incremental_validation import (
    incremental_artifact_binding,
    incremental_manifest_configuration_sha256,
)


ERROR_TYPES = {"concept", "logic", "calculation", "modeling", "units"}
GENERALIZATION_PROMPT_VERSION = "experience-candidate-generalization-v1"


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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _ordered_unique(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        item = _text(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _topic_key(domain: str, topic: str) -> str:
    return f"{_text(domain)}::{_text(topic)}"


def _stable_rule_id(domain: str, topic: str, source_candidate_ids: Iterable[str]) -> str:
    source = "\n".join([domain, topic, *sorted(_ordered_unique(source_candidate_ids))])
    return f"gen_{hashlib.sha1(source.encode('utf-8')).hexdigest()[:16]}"


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_symbolic_hint(value: Any) -> Dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    return {
        "primitive": _text(raw.get("primitive") or "none") or "none",
        "canonical": _text(raw.get("canonical") or ""),
        "required_symbols": _ordered_unique(raw.get("required_symbols") or []),
    }


def _extract_json_object(raw: Any) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty model response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*([\[{].*[\]}])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            candidate = fenced.group(1)
        else:
            object_start = text.find("{")
            array_start = text.find("[")
            starts = [item for item in (object_start, array_start) if item >= 0]
            if not starts:
                raise ValueError("model response does not contain JSON")
            start = min(starts)
            end = text.rfind("}" if text[start] == "{" else "]")
            candidate = text[start : end + 1]
        parsed = json.loads(candidate)
    if isinstance(parsed, list):
        return {"rules": parsed}
    if not isinstance(parsed, dict):
        raise ValueError(f"model response must be a JSON object or array, got {type(parsed).__name__}")
    return parsed


def _candidate_prompt_payload(rule: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": _text(rule.get("rule_id") or ""),
        "title": _text(rule.get("title") or ""),
        "trigger": _text(rule.get("trigger") or ""),
        "check_logic": _text(rule.get("check_logic") or ""),
        "error_type": _text(rule.get("error_type") or "logic"),
        "symbolic_hint": _normalize_symbolic_hint(rule.get("symbolic_hint")),
    }


def _build_prompt(domain: str, topic: str, candidates: List[Dict[str, Any]]) -> tuple[str, str]:
    system_prompt = (
        "你是物理经验候选概括器。输入内容只是单题候选经验，不是正式规则。"
        "你的任务是按物理定律、错误机制和适用条件进行概括；可以拆分，不得强行合并。"
        "只输出 JSON 对象。"
    )
    user_prompt = f"""
Domain: {domain}
Topic: {topic}

候选经验：
{json.dumps([_candidate_prompt_payload(item) for item in candidates], ensure_ascii=False, indent=2)}

输出格式：
{{
  "rules": [
    {{
      "source_candidate_ids": ["必须来自输入 candidate_id"],
      "title": "简洁的通用规则名称",
      "trigger": "物理状态或适用条件，不复述原题",
      "check_logic": "可执行的通用检查逻辑",
      "error_type": "concept|logic|calculation|modeling|units",
      "symbolic_hint": {{
        "primitive": "equation_equivalence|inequality_consistency|formula_pattern|power_law|none",
        "canonical": "必要时填写通用关系，否则为空字符串",
        "required_symbols": ["必要符号"]
      }}
    }}
  ]
}}

要求：
1. 只有控制定律、错误机制和适用条件一致的候选才能合并。
2. 同一组中存在不同机制时输出多条规则；无法可靠概括的候选可以不覆盖。
3. 删除单题数值、对象编号和原题特有措辞，但保留必要的通用公式与物理条件。
4. 合并规则只保留至少两条来源候选共同支持的信息；只出现在一条候选中的数值或限定必须删除。
5. 不要输出场景提示、背景摘要或其他附加字段。
"""
    return system_prompt, user_prompt


def _thinking_kwargs(
    *,
    model: str = "",
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    parameter = "thinking" if "deepseek" in model.casefold() else "enable_thinking"
    return {
        "extra_body": {
            "chat_template_kwargs": {
                parameter: bool(enable_thinking),
            }
        }
    }


def _retry_user_prompt(user_prompt: str) -> str:
    return (
        f"{user_prompt}\n\n"
        "响应修正：上一响应不是要求的 JSON 对象。"
        "本次只能返回一个包含 rules 数组的 JSON 对象；"
        "不得返回单个数字、字符串、分析文字或 Markdown。"
    )


def _is_route_unavailable(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "no available sub-groups" in message or (
        "503" in message and "bad_response_status_code" in message
    )


def _call_model(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    attempts: int,
    enable_thinking: bool = False,
    fallback_models: Sequence[str] = (),
) -> Dict[str, Any]:
    last_error: Exception | None = None
    models = _ordered_unique([model, *fallback_models])
    for model_index, active_model in enumerate(models, start=1):
        active_user_prompt = user_prompt
        for attempt in range(1, attempts + 1):
            started_at = time.monotonic()
            print(
                f"[generalize-api] model {model_index}/{len(models)} "
                f"attempt {attempt}/{attempts} model={active_model}",
                flush=True,
            )
            try:
                request: Dict[str, Any] = {
                    "model": active_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": active_user_prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                    **_thinking_kwargs(
                        model=active_model,
                        enable_thinking=enable_thinking,
                    ),
                }
                response = client.chat.completions.create(**request)
                message = response.choices[0].message
                payload = _extract_json_object(message.content)
                payload["_api_attempts"] = attempt
                payload["_model_used"] = active_model
                print(
                    f"[generalize-api] success model={active_model} attempt={attempt} "
                    f"elapsed={time.monotonic() - started_at:.1f}s",
                    flush=True,
                )
                return payload
            except Exception as exc:  # pragma: no cover - exercised against live APIs
                last_error = exc
                route_unavailable = _is_route_unavailable(exc)
                print(
                    f"[generalize-api] failed model={active_model} attempt={attempt} "
                    f"elapsed={time.monotonic() - started_at:.1f}s "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                if route_unavailable:
                    break
                if attempt < attempts:
                    active_user_prompt = _retry_user_prompt(user_prompt)
                    time.sleep(1.0)
    raise RuntimeError(
        f"candidate generalization failed across models {models}: {last_error}"
    )


def _materialize_cluster_result(
    *,
    domain: str,
    topic: str,
    cluster_id: str,
    candidates: List[Dict[str, Any]],
    model_payload: Dict[str, Any],
    min_source_candidates: int,
    min_source_samples: int,
) -> Dict[str, Any]:
    candidate_index = {
        _text(item.get("rule_id") or ""): item
        for item in candidates
        if _text(item.get("rule_id") or "")
    }
    generated_rules: List[Dict[str, Any]] = []
    mappings: List[Dict[str, Any]] = []
    covered: set[str] = set()
    seen_source_groups: set[tuple[str, ...]] = set()

    raw_rules = model_payload.get("rules") if isinstance(model_payload.get("rules"), list) else []
    for raw in raw_rules:
        if not isinstance(raw, dict):
            continue
        source_ids = [
            item
            for item in _ordered_unique(raw.get("source_candidate_ids") or [])
            if item in candidate_index and item not in covered
        ]
        title = _text(raw.get("title") or "")
        trigger = _text(raw.get("trigger") or "")
        check_logic = _text(raw.get("check_logic") or "")
        sample_ids = _ordered_unique(
            sample_id
            for source_id in source_ids
            for sample_id in (candidate_index[source_id].get("sample_ids") or [])
        )
        if (
            len(source_ids) < min_source_candidates
            or len(sample_ids) < min_source_samples
            or not title
            or not trigger
            or not check_logic
        ):
            continue
        source_group = tuple(sorted(source_ids))
        if source_group in seen_source_groups:
            continue
        seen_source_groups.add(source_group)
        error_type = _text(raw.get("error_type") or "logic")
        if error_type not in ERROR_TYPES:
            error_type = "logic"
        rule_id = _stable_rule_id(domain, topic, source_ids)
        generated_rules.append(
            {
                "rule_id": rule_id,
                "domain": domain,
                "topic": topic,
                "title": title,
                "trigger": trigger,
                "check_logic": check_logic,
                "error_type": error_type,
                "symbolic_hint": _normalize_symbolic_hint(raw.get("symbolic_hint")),
                "count": len(sample_ids),
                "sample_ids": sample_ids,
            }
        )
        mappings.append({"rule_id": rule_id, "source_candidate_ids": source_ids})
        covered.update(source_ids)

    input_ids = list(candidate_index)
    return {
        "domain": domain,
        "topic": topic,
        "cluster_id": cluster_id,
        "api_attempts": int(model_payload.get("_api_attempts") or 1),
        "model_used": _text(model_payload.get("_model_used") or ""),
        "input_candidate_ids": input_ids,
        "generated_rules": generated_rules,
        "mappings": mappings,
        "pending_candidate_ids": [item for item in input_ids if item not in covered],
    }


def _cluster_result_key(item: Dict[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
    return (
        _text(item.get("domain") or "").casefold(),
        _text(item.get("topic") or "").casefold(),
        _text(item.get("cluster_id") or ""),
        tuple(_ordered_unique(item.get("input_candidate_ids") or [])),
    )


def _batch_resume_fingerprint(
    batch: Dict[str, Any],
    *,
    min_source_candidates: int,
    min_source_samples: int,
    max_candidates_per_batch: int,
    resume_context: Dict[str, Any] | None,
) -> str:
    candidates = []
    for rule in batch.get("candidates", []) or []:
        if not isinstance(rule, dict):
            continue
        candidates.append(
            {
                "prompt_payload": _candidate_prompt_payload(rule),
                "sample_ids": _ordered_unique(rule.get("sample_ids") or []),
                "source_rule_ids": _ordered_unique(rule.get("source_rule_ids") or []),
                "count": int(rule.get("count") or 0),
            }
        )
    return _sha256_json(
        {
            "schema_version": 2,
            "prompt_version": GENERALIZATION_PROMPT_VERSION,
            "domain": batch.get("domain"),
            "topic": batch.get("topic"),
            "cluster_id": batch.get("cluster_id"),
            "source_cluster_id": batch.get("source_cluster_id"),
            "batch_index": batch.get("batch_index"),
            "batch_count": batch.get("batch_count"),
            "candidates": candidates,
            "min_source_candidates": min_source_candidates,
            "min_source_samples": min_source_samples,
            "max_candidates_per_batch": max_candidates_per_batch,
            "resume_context": resume_context or {},
        }
    )


def _build_result_payload(
    *,
    batches: List[Dict[str, Any]],
    cluster_results: List[Dict[str, Any]],
    scope_candidate_ids: List[str],
    residual_candidate_ids: List[str],
    unclustered_candidate_ids: List[str],
    missing_candidate_ids: List[str],
    selected_cluster_count: int,
    scope_mode: str,
    min_source_candidates: int,
    min_source_samples: int,
    output_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    formal_rules = [
        rule
        for result in cluster_results
        for rule in (result.get("generated_rules") or [])
        if isinstance(rule, dict)
    ]
    mapped_candidate_ids = _ordered_unique(
        candidate_id
        for result in cluster_results
        for mapping in (result.get("mappings") or [])
        if isinstance(mapping, dict)
        for candidate_id in (mapping.get("source_candidate_ids") or [])
    )
    mapped_set = set(mapped_candidate_ids)
    pending_candidate_ids = [
        candidate_id for candidate_id in scope_candidate_ids if candidate_id not in mapped_set
    ]
    failed_results = [item for item in cluster_results if _text(item.get("error") or "")]
    complete = (
        len(cluster_results) == len(batches)
        and not failed_results
        and not missing_candidate_ids
    )
    metadata = {
        "generator": "experience_candidate_generalizer_v1",
        "scope_mode": scope_mode,
        "complete": complete,
        "selected_cluster_count": selected_cluster_count,
        "selected_batch_count": len(batches),
        "processed_batch_count": len(cluster_results),
        "failed_batch_count": len(failed_results),
        "input_candidate_count": len(scope_candidate_ids),
        "residual_candidate_count": len(residual_candidate_ids),
        "unclustered_candidate_count": len(unclustered_candidate_ids),
        "missing_candidate_count": len(missing_candidate_ids),
        "generated_rule_count": len(formal_rules),
        "mapped_candidate_count": len(mapped_candidate_ids),
        "pending_candidate_count": len(pending_candidate_ids),
        "min_source_candidates": min_source_candidates,
        "min_source_samples": min_source_samples,
        "max_api_attempts_used": max(
            (int(item.get("api_attempts") or 0) for item in cluster_results),
            default=0,
        ),
        "models_used": _ordered_unique(
            item.get("model_used")
            for item in cluster_results
            if _text(item.get("model_used") or "")
        ),
    }
    metadata.update(output_metadata or {})
    return {
        "metadata": metadata,
        "rules": formal_rules,
        "cluster_results": cluster_results,
        "pending_candidate_ids": pending_candidate_ids,
        "residual_candidate_ids": residual_candidate_ids,
        "unclustered_candidate_ids": unclustered_candidate_ids,
        "missing_candidate_ids": missing_candidate_ids,
    }


def generalize_candidates(
    *,
    candidate_payload: Dict[str, Any],
    cluster_payload: Dict[str, Any],
    generate: Callable[[str, str, List[Dict[str, Any]]], Dict[str, Any]],
    domain_filter: str = "",
    topic_filter: str = "",
    cluster_filter: str = "",
    max_clusters: int = 0,
    min_source_candidates: int = 2,
    min_source_samples: int = 2,
    max_candidates_per_batch: int = 12,
    continue_on_error: bool = False,
    existing_payload: Dict[str, Any] | None = None,
    on_progress: Callable[[Dict[str, Any]], None] | None = None,
    resume_context: Dict[str, Any] | None = None,
    output_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    candidates = [
        item
        for item in (candidate_payload.get("rules") or [])
        if isinstance(item, dict) and _text(item.get("rule_id") or "")
    ]
    candidate_index = {_text(item["rule_id"]): item for item in candidates}
    batches: List[Dict[str, Any]] = []
    scope_candidate_ids: List[str] = []
    residual_candidate_ids: List[str] = []
    unclustered_candidate_ids: List[str] = []
    missing_candidate_ids: List[str] = []
    selected_cluster_count = 0
    assigned_candidate_ids: set[str] = set()
    assigned_residual_ids: set[str] = set()
    full_scope = not domain_filter and not topic_filter and not cluster_filter and max_clusters <= 0
    batch_size = max(2, int(max_candidates_per_batch))

    for topic_item in cluster_payload.get("topics", []) or []:
        if not isinstance(topic_item, dict):
            continue
        domain = _text(topic_item.get("domain") or "")
        topic = _text(topic_item.get("topic") or "")
        if domain_filter and domain.casefold() != domain_filter.casefold():
            continue
        if topic_filter and topic.casefold() != topic_filter.casefold():
            continue
        for cluster in topic_item.get("clusters", []) or []:
            if not isinstance(cluster, dict):
                continue
            cluster_id = _text(cluster.get("cluster_id") or "")
            if cluster_filter and cluster_id != cluster_filter:
                continue
            rule_ids = _ordered_unique(cluster.get("rule_ids") or [])
            duplicate_ids = [rule_id for rule_id in rule_ids if rule_id in assigned_candidate_ids]
            if duplicate_ids:
                raise ValueError(
                    f"candidate ids assigned to multiple selected clusters: {duplicate_ids[:5]}"
                )
            assigned_candidate_ids.update(rule_ids)
            missing_candidate_ids.extend(rule_id for rule_id in rule_ids if rule_id not in candidate_index)
            rows = [candidate_index[rule_id] for rule_id in rule_ids if rule_id in candidate_index]
            if rows:
                selected_cluster_count += 1
                scope_candidate_ids.extend(rule_ids)
                chunks = [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]
                for batch_index, chunk in enumerate(chunks, start=1):
                    batch_cluster_id = (
                        cluster_id
                        if len(chunks) == 1
                        else f"{cluster_id}__batch_{batch_index:03d}"
                    )
                    batches.append(
                        {
                            "domain": domain,
                            "topic": topic,
                            "cluster_id": batch_cluster_id,
                            "source_cluster_id": cluster_id,
                            "batch_index": batch_index,
                            "batch_count": len(chunks),
                            "candidates": chunk,
                        }
                    )
            if max_clusters > 0 and selected_cluster_count >= max_clusters:
                break
        if max_clusters > 0 and selected_cluster_count >= max_clusters:
            break
        if not cluster_filter and max_clusters <= 0:
            topic_residual_ids = _ordered_unique(topic_item.get("residual_rule_ids") or [])
            overlap = [
                rule_id
                for rule_id in topic_residual_ids
                if rule_id in assigned_candidate_ids or rule_id in assigned_residual_ids
            ]
            if overlap:
                raise ValueError(
                    f"candidate ids have duplicate cluster/residual assignments: {overlap[:5]}"
                )
            assigned_residual_ids.update(topic_residual_ids)
            missing_candidate_ids.extend(
                rule_id for rule_id in topic_residual_ids if rule_id not in candidate_index
            )
            residual_candidate_ids.extend(
                rule_id for rule_id in topic_residual_ids if rule_id in candidate_index
            )
            scope_candidate_ids.extend(topic_residual_ids)

    if full_scope:
        referenced_ids = assigned_candidate_ids | assigned_residual_ids
        unclustered_candidate_ids = [
            rule_id for rule_id in candidate_index if rule_id not in referenced_ids
        ]
        residual_candidate_ids.extend(unclustered_candidate_ids)
        scope_candidate_ids.extend(unclustered_candidate_ids)

    scope_candidate_ids = _ordered_unique(
        rule_id for rule_id in scope_candidate_ids if rule_id in candidate_index
    )
    residual_candidate_ids = _ordered_unique(residual_candidate_ids)
    missing_candidate_ids = _ordered_unique(missing_candidate_ids)
    if not batches and not residual_candidate_ids:
        raise ValueError("no candidate clusters or residual candidates matched the selected scope")

    existing_results = {
        _cluster_result_key(item): item
        for item in ((existing_payload or {}).get("cluster_results") or [])
        if isinstance(item, dict) and not _text(item.get("error") or "")
    }
    cluster_results: List[Dict[str, Any]] = []
    minimum_candidates = max(1, int(min_source_candidates))
    minimum_samples = max(1, int(min_source_samples))

    def current_payload() -> Dict[str, Any]:
        return _build_result_payload(
            batches=batches,
            cluster_results=cluster_results,
            scope_candidate_ids=scope_candidate_ids,
            residual_candidate_ids=residual_candidate_ids,
            unclustered_candidate_ids=unclustered_candidate_ids,
            missing_candidate_ids=missing_candidate_ids,
            selected_cluster_count=selected_cluster_count,
            scope_mode="full" if full_scope else "filtered",
            min_source_candidates=minimum_candidates,
            min_source_samples=minimum_samples,
            output_metadata=output_metadata,
        )

    for batch_number, batch in enumerate(batches, start=1):
        rows = batch["candidates"]
        input_ids = [_text(item.get("rule_id") or "") for item in rows]
        input_sample_ids = _ordered_unique(
            sample_id for item in rows for sample_id in (item.get("sample_ids") or [])
        )
        lookup_key = (
            batch["domain"].casefold(),
            batch["topic"].casefold(),
            batch["cluster_id"],
            tuple(input_ids),
        )
        resume_fingerprint = _batch_resume_fingerprint(
            batch,
            min_source_candidates=minimum_candidates,
            min_source_samples=minimum_samples,
            max_candidates_per_batch=batch_size,
            resume_context=resume_context,
        )
        existing_result = existing_results.get(lookup_key)
        if (
            existing_result is not None
            and _text(existing_result.get("resume_fingerprint_v2") or "")
            == resume_fingerprint
        ):
            result = dict(existing_result)
            result["reused"] = True
            cluster_results.append(result)
            if on_progress:
                on_progress(current_payload())
            continue
        if len(rows) < minimum_candidates or len(input_sample_ids) < minimum_samples:
            result = {
                "domain": batch["domain"],
                "topic": batch["topic"],
                "cluster_id": batch["cluster_id"],
                "source_cluster_id": batch["source_cluster_id"],
                "batch_index": batch["batch_index"],
                "batch_count": batch["batch_count"],
                "api_attempts": 0,
                "input_candidate_ids": input_ids,
                "generated_rules": [],
                "mappings": [],
                "pending_candidate_ids": input_ids,
                "skipped_reason": "insufficient_candidate_or_sample_support",
                "resume_fingerprint_v2": resume_fingerprint,
            }
            cluster_results.append(result)
            if on_progress:
                on_progress(current_payload())
            continue
        print(
            f"[generalize] batch {batch_number}/{len(batches)} "
            f"{batch['domain']} / {batch['topic']} / {batch['cluster_id']} "
            f"candidates={len(rows)}",
            flush=True,
        )
        try:
            model_payload = generate(batch["domain"], batch["topic"], rows)
            result = _materialize_cluster_result(
                domain=batch["domain"],
                topic=batch["topic"],
                cluster_id=batch["cluster_id"],
                candidates=rows,
                model_payload=model_payload,
                min_source_candidates=minimum_candidates,
                min_source_samples=minimum_samples,
            )
            result["source_cluster_id"] = batch["source_cluster_id"]
            result["batch_index"] = batch["batch_index"]
            result["batch_count"] = batch["batch_count"]
            result["resume_fingerprint_v2"] = resume_fingerprint
        except Exception as exc:
            result = {
                "domain": batch["domain"],
                "topic": batch["topic"],
                "cluster_id": batch["cluster_id"],
                "source_cluster_id": batch["source_cluster_id"],
                "batch_index": batch["batch_index"],
                "batch_count": batch["batch_count"],
                "api_attempts": 0,
                "input_candidate_ids": input_ids,
                "generated_rules": [],
                "mappings": [],
                "pending_candidate_ids": input_ids,
                "error": f"{type(exc).__name__}: {exc}",
                "resume_fingerprint_v2": resume_fingerprint,
            }
            cluster_results.append(result)
            if on_progress:
                on_progress(current_payload())
            if not continue_on_error:
                raise
            continue
        cluster_results.append(result)
        if on_progress:
            on_progress(current_payload())

    return current_payload()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalize clustered single-sample experience candidates.")
    parser.add_argument(
        "--candidates",
        default="results/unified_rules_3000/semantic_experience_distilled_for_cluster.json",
    )
    parser.add_argument(
        "--clusters",
        default="results/unified_rules_3000/rule_embedding_clusters.json",
    )
    parser.add_argument(
        "--output",
        default="results/unified_rules_3000/semantic_experience_generalized_pilot.json",
    )
    parser.add_argument("--model", default="gemini-3-flash-preview-nothinking")
    parser.add_argument(
        "--fallback-model",
        action="append",
        default=["gemini-2.5-flash-nothinking"],
        help="Repeat to add fallback models. Route-unavailable 503 errors switch immediately.",
    )
    parser.add_argument("--domain", default="")
    parser.add_argument("--topic", default="")
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--max-clusters", type=int, default=1)
    parser.add_argument("--min-source-candidates", type=int, default=2)
    parser.add_argument("--min-source-samples", type=int, default=2)
    parser.add_argument("--max-candidates-per-batch", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable model thinking. Disabled by default for stable structured output.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
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
        base_generalized_record = (
            (manifest_payload.get("inputs") or {}).get("base_generalized") or {}
            if isinstance(manifest_payload, dict)
            else {}
        )
        incremental_binding = incremental_artifact_binding(
            manifest_path,
            stage="candidate_generalization",
            input_paths={
                "candidate_rules": Path(args.candidates),
                "candidate_clusters": Path(args.clusters),
            },
            input_sha256={
                "base_generalized": str(
                    base_generalized_record.get("sha256") or ""
                )
            },
        )
        incremental_binding["incremental_behavior_configuration"] = {
            "model_chain": _ordered_unique([args.model, *args.fallback_model]),
            "temperature": 0.0,
            "max_clusters": max(0, int(args.max_clusters)),
            "max_candidates_per_batch": max(
                2, int(args.max_candidates_per_batch)
            ),
            "min_source_candidates": max(1, int(args.min_source_candidates)),
            "min_source_samples": max(1, int(args.min_source_samples)),
            "max_tokens": max(256, int(args.max_tokens)),
            "request_timeout_seconds": max(1.0, float(args.request_timeout)),
            "attempts": max(1, int(args.attempts)),
            "thinking_enabled": bool(args.enable_thinking),
        }

    if load_dotenv:
        load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    request_timeout = max(1.0, float(args.request_timeout))
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=request_timeout,
        max_retries=0,
    )

    def generate(domain: str, topic: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        system_prompt, user_prompt = _build_prompt(domain, topic, candidates)
        return _call_model(
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max(256, int(args.max_tokens)),
            attempts=max(1, int(args.attempts)),
            enable_thinking=bool(args.enable_thinking),
            fallback_models=args.fallback_model,
        )

    output_path = Path(args.output)
    existing_payload = (
        _load_json(output_path)
        if args.resume and output_path.exists()
        else None
    )

    def save_progress(payload: Dict[str, Any]) -> None:
        payload["metadata"]["model"] = args.model
        payload["metadata"]["fallback_models"] = _ordered_unique(args.fallback_model)
        payload["metadata"]["candidate_source"] = args.candidates
        payload["metadata"]["cluster_source"] = args.clusters
        payload["metadata"]["output_mode"] = "prompt_json"
        payload["metadata"]["request_timeout_seconds"] = request_timeout
        payload["metadata"]["thinking_enabled"] = bool(args.enable_thinking)
        payload["metadata"]["prompt_version"] = GENERALIZATION_PROMPT_VERSION
        payload["metadata"]["resume_fingerprint_schema_version"] = 2
        _write_json(output_path, payload)

    result = generalize_candidates(
        candidate_payload=_load_json(Path(args.candidates)),
        cluster_payload=_load_json(Path(args.clusters)),
        generate=generate,
        domain_filter=args.domain,
        topic_filter=args.topic,
        cluster_filter=args.cluster_id,
        max_clusters=max(0, int(args.max_clusters)),
        min_source_candidates=max(1, int(args.min_source_candidates)),
        min_source_samples=max(1, int(args.min_source_samples)),
        max_candidates_per_batch=max(2, int(args.max_candidates_per_batch)),
        continue_on_error=bool(args.continue_on_error),
        existing_payload=existing_payload,
        on_progress=save_progress,
        resume_context={
            "model_chain": _ordered_unique([args.model, *args.fallback_model]),
            "temperature": 0.0,
            "max_tokens": max(256, int(args.max_tokens)),
            "attempts": max(1, int(args.attempts)),
            "request_timeout_seconds": request_timeout,
            "thinking_enabled": bool(args.enable_thinking),
        },
        output_metadata=incremental_binding,
    )
    save_progress(result)
    print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))
    if not result["metadata"]["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
