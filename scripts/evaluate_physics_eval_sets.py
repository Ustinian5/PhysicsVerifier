from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _typed_id_key(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("sample IDs must be non-empty strings or integers")
    if isinstance(value, str) and not value.strip():
        raise ValueError("sample IDs must not be empty")
    return f"{type(value).__name__}:{json.dumps(value, ensure_ascii=False, sort_keys=True)}"


def _index_by_id(
    items: List[Dict[str, Any]],
    *,
    label: str,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(items):
        if not isinstance(row, dict):
            raise ValueError(f"{label}[{index}] must be a JSON object")
        if "id" not in row:
            raise ValueError(f"{label}[{index}] is missing id")
        try:
            sid = _typed_id_key(row.get("id"))
        except ValueError as exc:
            raise ValueError(f"invalid {label}[{index}].id: {exc}") from exc
        if sid in out:
            raise ValueError(f"duplicate typed sample ID in {label}: {row.get('id')!r}")
        out[sid] = row
    return out


def _result_identity(items: List[Dict[str, Any]]) -> Dict[str, str]:
    identity: Dict[str, str] = {}
    for field in ("checker_gate_mode", "replay_config_sha256"):
        present = [str(row.get(field) or "").strip() for row in items]
        nonempty = {value for value in present if value}
        if len(nonempty) > 1:
            raise ValueError(f"results mix multiple {field} values")
        if nonempty and any(not value for value in present):
            raise ValueError(f"results mix missing and populated {field} values")
        identity[field] = next(iter(nonempty), "")
    return identity


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _collapse_text_for_match(text: str) -> tuple[str, List[int]]:
    src = str(text or "")
    out: List[str] = []
    mapping: List[int] = []
    prev_space = False
    for i, ch in enumerate(src):
        c = ch
        if c in "{}[]()$`":
            continue
        if c == "\\":
            continue
        if c.isspace():
            if out and (not prev_space):
                out.append(" ")
                mapping.append(i)
                prev_space = True
            continue
        out.append(c.lower())
        mapping.append(i)
        prev_space = False

    while out and out[0] == " ":
        out.pop(0)
        mapping.pop(0)
    while out and out[-1] == " ":
        out.pop()
        mapping.pop()
    return "".join(out), mapping


def _locate_quote_span(text: str, quote: str) -> Dict[str, Any]:
    src = str(text or "")
    q = str(quote or "").strip()
    if not src or not q:
        return {
            "start_char": -1,
            "end_char": -1,
            "line_index": -1,
            "span_valid": False,
            "locate_method": "missing_quote",
            "span_ambiguous": False,
        }

    def _pack(start: int, end: int, method: str, ambiguous: bool) -> Dict[str, Any]:
        return {
            "start_char": int(start),
            "end_char": int(end),
            "line_index": int(src.count("\n", 0, start) + 1),
            "span_valid": True,
            "locate_method": method,
            "span_ambiguous": bool(ambiguous),
        }

    exact = list(re.finditer(re.escape(q), src))
    if exact:
        m0 = exact[0]
        return _pack(m0.start(), m0.end(), "exact", len(exact) > 1)

    ci = list(re.finditer(re.escape(q), src, flags=re.I))
    if ci:
        m0 = ci[0]
        return _pack(m0.start(), m0.end(), "case_insensitive", len(ci) > 1)

    parts = [re.escape(x) for x in re.split(r"\s+", q) if x]
    if parts:
        pat = r"\s+".join(parts)
        ws = list(re.finditer(pat, src, flags=re.I))
        if ws:
            m0 = ws[0]
            return _pack(m0.start(), m0.end(), "whitespace_fuzzy", len(ws) > 1)

    src_norm, src_map = _collapse_text_for_match(src)
    q_norm, _ = _collapse_text_for_match(q)
    if src_norm and q_norm:
        k = src_norm.find(q_norm)
        if k >= 0:
            s = src_map[k]
            e = src_map[min(len(src_map) - 1, k + len(q_norm) - 1)] + 1
            return _pack(s, e, "normalized_substring", False)

    return {
        "start_char": -1,
        "end_char": -1,
        "line_index": -1,
        "span_valid": False,
        "locate_method": "not_found",
        "span_ambiguous": False,
    }


def _paragraph_ranges(text: str) -> List[Dict[str, Any]]:
    src = str(text or "")
    if not src:
        return []

    target_len = 220
    min_len = 120
    max_len = 360
    n = len(src)
    boundary_set = {0, n}
    for m in re.finditer(r"[。！？!?；;](?:\s+|$)|\n+", src):
        boundary_set.add(m.end())
    boundaries = sorted(boundary_set)

    out: List[Dict[str, Any]] = []
    start = 0
    para_idx = 0
    while start < n:
        if n - start <= max_len:
            end = n
        else:
            low = min(n, start + min_len)
            high = min(n, start + max_len)
            desired = min(n, start + target_len)
            candidates = [b for b in boundaries if low <= b <= high]
            if candidates:
                end = min(candidates, key=lambda b: abs(b - desired))
            else:
                end = high

        s = start
        e = max(start, end)
        while s < e and src[s].isspace():
            s += 1
        while e > s and src[e - 1].isspace():
            e -= 1
        if e > s:
            para_idx += 1
            out.append({"paragraph_index": para_idx, "start_char": s, "end_char": e, "text": src[s:e]})
        start = end if end > start else start + 1

    if not out and src.strip():
        out.append({"paragraph_index": 1, "start_char": 0, "end_char": len(src), "text": src})
    return out


def _expand_span_to_context_window(
    text: str,
    start_char: int,
    end_char: int,
    *,
    left_context: int = 90,
    right_context: int = 120,
    max_window: int = 320,
) -> Dict[str, int]:
    src = str(text or "")
    n = len(src)
    if n <= 0 or start_char < 0 or end_char <= start_char:
        return {"start_char": -1, "end_char": -1}

    s = max(0, int(start_char) - left_context)
    e = min(n, int(end_char) + right_context)
    while s > 0 and (not src[s - 1].isspace()) and (int(start_char) - s) < (left_context + 50):
        s -= 1
    while e < n and (not src[e].isspace()) and (e - int(end_char)) < (right_context + 50):
        e += 1

    if e - s > max_window:
        mid = (int(start_char) + int(end_char)) // 2
        half = max_window // 2
        s = max(0, mid - half)
        e = min(n, s + max_window)

    while s < e and src[s].isspace():
        s += 1
    while e > s and src[e - 1].isspace():
        e -= 1
    return {"start_char": s if e > s else -1, "end_char": e if e > s else -1}


def _paragraph_from_offset(paragraphs: List[Dict[str, Any]], offset: int) -> Optional[Dict[str, Any]]:
    if offset < 0:
        return None
    for p in paragraphs:
        s = int(p.get("start_char") or -1)
        e = int(p.get("end_char") or -1)
        if s <= offset < e:
            return p
    return None


def _paragraph_by_index(paragraphs: List[Dict[str, Any]], paragraph_index: int) -> Optional[Dict[str, Any]]:
    if paragraph_index <= 0:
        return None
    for p in paragraphs:
        if int(p.get("paragraph_index") or -1) == paragraph_index:
            return p
    return None


def _extract_gt_entries(row: Dict[str, Any], sample_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    gt_loc = row.get("physics_error_gt")

    if isinstance(gt_loc, list):
        for i, item in enumerate(gt_loc):
            if not isinstance(item, dict):
                continue
            error_text = str(item.get("error_text") or item.get("error") or "").strip()
            if not error_text:
                continue
            start = _safe_int(item.get("start_char"))
            end = _safe_int(item.get("end_char"))
            paragraph_index = _safe_int(item.get("paragraph_index"))
            span_valid = bool(item.get("span_valid"))
            if start is not None and end is not None and end > start and start >= 0:
                span_valid = True

            paragraph_valid = bool(item.get("paragraph_valid"))
            if paragraph_index is not None and paragraph_index >= 1:
                paragraph_valid = True

            out.append(
                {
                    "error_id": str(item.get("error_id") or f"{sample_id}_e{i + 1}"),
                    "error_text": error_text,
                    "answer_quote": str(item.get("answer_quote") or item.get("quote") or "").strip(),
                    "start_char": int(start) if start is not None else -1,
                    "end_char": int(end) if end is not None else -1,
                    "line_index": int(_safe_int(item.get("line_index")) or -1),
                    "span_valid": span_valid,
                    "span_source": str(item.get("span_source") or ""),
                    "paragraph_index": int(paragraph_index) if paragraph_index is not None else -1,
                    "paragraph_start_char": int(_safe_int(item.get("paragraph_start_char")) or -1),
                    "paragraph_end_char": int(_safe_int(item.get("paragraph_end_char")) or -1),
                    "paragraph_valid": paragraph_valid,
                    "paragraph_source": str(item.get("paragraph_source") or ""),
                    "locatable_valid": bool(span_valid or paragraph_valid),
                }
            )
        return out

    # Legacy fallback: old datasets only have physics_error_examples without location.
    gt_items = row.get("physics_error_examples")
    gt_items = gt_items if isinstance(gt_items, list) else []
    for i, x in enumerate(gt_items):
        if not isinstance(x, dict):
            continue
        err = str(x.get("error") or "").strip()
        if not err:
            continue
        out.append(
            {
                "error_id": f"{sample_id}_legacy_e{i + 1}",
                "error_text": err,
                "answer_quote": "",
                "start_char": -1,
                "end_char": -1,
                "line_index": -1,
                "span_valid": False,
                "span_source": "legacy_no_location",
                "paragraph_index": -1,
                "paragraph_start_char": -1,
                "paragraph_end_char": -1,
                "paragraph_valid": False,
                "paragraph_source": "legacy_no_location",
                "locatable_valid": False,
            }
        )
    return out


def _collect_pred_findings(pred_item: Dict[str, Any], audit_item: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    diagnostics = pred_item.get("diagnostics") if isinstance(pred_item, dict) else []
    diagnostics = diagnostics if isinstance(diagnostics, list) else []
    for d in diagnostics:
        if not isinstance(d, dict):
            continue
        rule = str(d.get("rule") or "").strip()
        message = str(d.get("message") or "").strip()

        evidence = d.get("evidence")
        quote = ""
        loc = {}
        if isinstance(evidence, dict):
            quote = str(evidence.get("quote") or "").strip()
            loc = evidence.get("location") if isinstance(evidence.get("location"), dict) else {}
        elif isinstance(evidence, str):
            quote = evidence.strip()

        start = _safe_int(loc.get("start_char"))
        end = _safe_int(loc.get("end_char"))
        line_index = _safe_int(loc.get("line_index"))
        paragraph_index = _safe_int(loc.get("paragraph_index"))
        span_valid = bool(start is not None and end is not None and end > start and start >= 0)
        paragraph_valid = bool(paragraph_index is not None and paragraph_index >= 1)

        text_parts = [rule, message, quote]
        text = " | ".join([p for p in text_parts if p])
        if text:
            findings.append(
                {
                    "source": "diagnostic",
                    "rule": rule,
                    "message": message,
                    "quote": quote,
                    "text": text,
                    "rule_match": d.get("rule_match") if isinstance(d.get("rule_match"), dict) else {},
                    "release_gate": d.get("release_gate") if isinstance(d.get("release_gate"), dict) else {},
                    "start_char": int(start) if start is not None else -1,
                    "end_char": int(end) if end is not None else -1,
                    "line_index": int(line_index) if line_index is not None else -1,
                    "span_valid": span_valid,
                    "locate_method": str(loc.get("locate_method") or "existing"),
                    "paragraph_index": int(paragraph_index) if paragraph_index is not None else -1,
                    "paragraph_start_char": int(_safe_int(loc.get("paragraph_start_char")) or -1),
                    "paragraph_end_char": int(_safe_int(loc.get("paragraph_end_char")) or -1),
                    "paragraph_valid": paragraph_valid,
                    "paragraph_source": str(loc.get("paragraph_source") or "existing"),
                    "locatable_valid": bool(span_valid or paragraph_valid),
                }
            )

    checker_mode = str(pred_item.get("checker_gate_mode") or "legacy").strip().lower()
    if isinstance(audit_item, dict) and checker_mode == "legacy":
        checks = audit_item.get("experience_code_checks")
        checks = checks if isinstance(checks, list) else []
        for c in checks:
            if not isinstance(c, dict):
                continue
            res = str(c.get("result") or "").strip().lower()
            if res != "fail" or str(c.get("publish_skipped") or "").strip():
                continue
            rule = str(c.get("rule") or "").strip()
            message = str(c.get("message") or "").strip()
            evidence = str(c.get("evidence") or "").strip()
            text = " | ".join([p for p in [rule, message, evidence] if p])
            if text:
                findings.append(
                    {
                        "source": "experience_code",
                        "rule": rule,
                        "message": message,
                        "quote": "",
                        "text": text,
                        "start_char": -1,
                        "end_char": -1,
                        "line_index": -1,
                        "span_valid": False,
                        "locate_method": "not_available",
                        "paragraph_index": -1,
                        "paragraph_start_char": -1,
                        "paragraph_end_char": -1,
                        "paragraph_valid": False,
                        "paragraph_source": "not_available",
                        "locatable_valid": False,
                    }
                )

    # Ordered unique
    out: List[Dict[str, Any]] = []
    seen = set()
    for f in findings:
        key = (
            str(f.get("source") or ""),
            str(f.get("rule") or ""),
            str(f.get("message") or ""),
            str(f.get("quote") or ""),
            int(f.get("start_char") or -1),
            int(f.get("end_char") or -1),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def _execution_failure_reason(pred_item: Any) -> str:
    if not isinstance(pred_item, dict):
        return "missing_result"
    selection = str(pred_item.get("selection_strategy") or "").strip().lower()
    if selection in {"semantic_error", "semantic_unavailable"} or str(
        pred_item.get("semantic_selection_error") or ""
    ).strip():
        return "semantic_retrieval_failure"
    checker_status = str(pred_item.get("checker_status") or "").strip().lower()
    try:
        checker_failure_count = int(pred_item.get("checker_failure_count") or 0)
    except (TypeError, ValueError):
        checker_failure_count = 1
    raw_checker_failures = pred_item.get("checker_failures")
    if isinstance(raw_checker_failures, list):
        checker_failure_count = max(checker_failure_count, len(raw_checker_failures))
    if checker_failure_count > 0 or checker_status in {
        "failed",
        "partial_failure",
        "not_run",
    }:
        return "checker_failure"
    return ""


def _fill_missing_pred_locations(findings: List[Dict[str, Any]], answer_text: str) -> List[Dict[str, Any]]:
    paragraphs = _paragraph_ranges(answer_text)
    out: List[Dict[str, Any]] = []
    for f in findings:
        item = dict(f)
        span_valid = bool(item.get("span_valid"))
        quote = str(item.get("quote") or "").strip()

        # Character offsets are often generated by the verifier model and can be
        # syntactically valid while pointing at unrelated text.  Validate them
        # against the quoted evidence before using them for location metrics.
        if span_valid and quote:
            start = int(item.get("start_char") or -1)
            end = int(item.get("end_char") or -1)
            quoted_norm, _ = _collapse_text_for_match(quote)
            sliced_norm, _ = _collapse_text_for_match(answer_text[start:end] if start >= 0 else "")
            if not quoted_norm or sliced_norm != quoted_norm:
                span_valid = False
                item["span_valid"] = False

        if not span_valid and quote:
            loc = _locate_quote_span(answer_text, quote)
            if bool(loc.get("span_valid")):
                item["start_char"] = int(loc.get("start_char", -1))
                item["end_char"] = int(loc.get("end_char", -1))
                item["line_index"] = int(loc.get("line_index", -1))
                item["span_valid"] = True
                item["locate_method"] = f"fallback_{loc.get('locate_method') or 'quote_match'}"
                span_valid = True

        # A trustworthy span is the source of truth for its paragraph.  Rebuild
        # paragraph metadata even when the model supplied a paragraph index.
        paragraph_valid = False
        if span_valid:
            start = int(item.get("start_char") or -1)
            end = int(item.get("end_char") or -1)
            p = _paragraph_from_offset(paragraphs, start)
            if p is not None:
                ctx = _expand_span_to_context_window(answer_text, start, end)
                item["paragraph_index"] = int(p.get("paragraph_index") or -1)
                item["paragraph_start_char"] = int(ctx.get("start_char") or p.get("start_char") or -1)
                item["paragraph_end_char"] = int(ctx.get("end_char") or p.get("end_char") or -1)
                item["paragraph_valid"] = True
                item["paragraph_source"] = "from_quote_context" if quote else "from_span_context"
                paragraph_valid = True
        else:
            paragraph_valid = bool(item.get("paragraph_valid"))

        if not paragraph_valid:
            pidx = int(_safe_int(item.get("paragraph_index")) or -1)
            p = _paragraph_by_index(paragraphs, pidx)
            if p is not None:
                item["paragraph_valid"] = True
                item["paragraph_start_char"] = int(p.get("start_char") or -1)
                item["paragraph_end_char"] = int(p.get("end_char") or -1)
                item["paragraph_source"] = item.get("paragraph_source") or "model_declared"
                paragraph_valid = True

        item["locatable_valid"] = bool(span_valid or paragraph_valid)
        out.append(item)
    return out


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if len(t) >= 3]


def _keyword_match(gt_error: str, findings_text: List[str], min_overlap: int = 3) -> bool:
    gt_tokens = set(_tokenize(gt_error))
    if not gt_tokens:
        return False
    for f in findings_text:
        f_tokens = set(_tokenize(f))
        if len(gt_tokens & f_tokens) >= min_overlap:
            return True
    return False


def _llm_match_coverage(model: str, gt_errors: List[str], findings_text: List[str]) -> Optional[List[bool]]:
    try:
        import os
        import openai  # type: ignore
    except Exception:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    client = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)

    system_prompt = (
        "You are an evaluator for physics-error detection recall. "
        "Given ground-truth error statements and predicted findings, decide whether each GT error is covered. "
        "Return JSON only: {\"covered\": [true/false, ...]} with the same length as GT list."
    )
    user_prompt = (
        "Ground-truth errors:\n"
        + json.dumps(gt_errors, ensure_ascii=False)
        + "\n\nPredicted findings:\n"
        + json.dumps(findings_text, ensure_ascii=False)
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        data = json.loads(raw)
        covered = data.get("covered") if isinstance(data, dict) else None
        if not isinstance(covered, list):
            return None
        out: List[bool] = []
        for i in range(len(gt_errors)):
            v = covered[i] if i < len(covered) else False
            out.append(bool(v))
        return out
    except Exception:
        return None


def _span_metrics(a_start: int, a_end: int, b_start: int, b_end: int) -> Tuple[float, float, float, int]:
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    if overlap <= 0:
        return 0.0, 0.0, 0.0, 0
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0, 0.0, 0.0, overlap
    iou = overlap / union
    gt_cov = overlap / max(1, a_end - a_start)
    pred_cov = overlap / max(1, b_end - b_start)
    return iou, gt_cov, pred_cov, overlap


def _region_metrics(a_start: int, a_end: int, b_start: int, b_end: int) -> Tuple[float, float, float, int]:
    return _span_metrics(a_start, a_end, b_start, b_end)


def _match_by_location(
    gt_entries: List[Dict[str, Any]],
    pred_findings: List[Dict[str, Any]],
    iou_threshold: float,
    coverage_threshold: float,
) -> Tuple[Set[str], List[Dict[str, Any]], Set[int]]:
    matched_gt_ids: Set[str] = set()
    matched_details: List[Dict[str, Any]] = []
    used_pred_idx: Set[int] = set()

    for gt in gt_entries:
        g_start = int(gt.get("start_char") or -1)
        g_end = int(gt.get("end_char") or -1)
        g_span_valid = bool(gt.get("span_valid")) and g_start >= 0 and g_end > g_start
        g_para_idx = int(gt.get("paragraph_index") or -1)
        g_para_valid = bool(gt.get("paragraph_valid")) and g_para_idx >= 1
        if not g_span_valid and not g_para_valid:
            continue

        best_idx = -1
        best_payload: Dict[str, Any] = {}
        best_score = -1.0

        if g_span_valid:
            for idx, pred in enumerate(pred_findings):
                if idx in used_pred_idx:
                    continue
                if not bool(pred.get("span_valid")):
                    continue
                p_start = int(pred.get("start_char") or -1)
                p_end = int(pred.get("end_char") or -1)
                if p_start < 0 or p_end <= p_start:
                    continue

                iou, gt_cov, pred_cov, overlap = _span_metrics(g_start, g_end, p_start, p_end)
                if iou <= 0 and max(gt_cov, pred_cov) <= 0:
                    continue
                score = iou + 0.01 * max(gt_cov, pred_cov)
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_payload = {
                        "iou": iou,
                        "gt_coverage": gt_cov,
                        "pred_coverage": pred_cov,
                        "overlap": overlap,
                        "match_type": "span",
                    }

        if best_idx < 0 and g_para_valid:
            g_ps = int(gt.get("paragraph_start_char") or -1)
            g_pe = int(gt.get("paragraph_end_char") or -1)
            gt_text = str(gt.get("error_text") or "")
            gt_tokens = set(_tokenize(gt_text))
            for idx, pred in enumerate(pred_findings):
                if idx in used_pred_idx:
                    continue
                p_para_idx = int(pred.get("paragraph_index") or -1)
                if not bool(pred.get("paragraph_valid")):
                    continue
                p_ps = int(pred.get("paragraph_start_char") or -1)
                p_pe = int(pred.get("paragraph_end_char") or -1)

                region_overlap_ok = False
                region_score = 0.0
                if g_ps >= 0 and g_pe > g_ps and p_ps >= 0 and p_pe > p_ps:
                    r_iou, r_gc, r_pc, _ = _region_metrics(g_ps, g_pe, p_ps, p_pe)
                    if r_iou > 0 or max(r_gc, r_pc) > 0:
                        region_overlap_ok = True
                        region_score = r_iou + 0.05 * max(r_gc, r_pc)

                if (not region_overlap_ok) and p_para_idx != g_para_idx:
                    continue

                pred_tokens = set(_tokenize(str(pred.get("text") or "")))
                overlap = len(gt_tokens & pred_tokens)
                score = float(overlap) + region_score
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_payload = {
                        "iou": 0.0,
                        "gt_coverage": 0.0,
                        "pred_coverage": 0.0,
                        "overlap": 0,
                        "match_type": "paragraph_region" if region_overlap_ok else "paragraph",
                    }

        if best_idx < 0:
            continue

        iou = float(best_payload.get("iou") or 0.0)
        gt_cov = float(best_payload.get("gt_coverage") or 0.0)
        pred_cov = float(best_payload.get("pred_coverage") or 0.0)
        match_type = str(best_payload.get("match_type") or "span")
        if match_type in {"paragraph", "paragraph_region"} or iou >= iou_threshold or max(gt_cov, pred_cov) >= coverage_threshold:
            pred = pred_findings[best_idx]
            used_pred_idx.add(best_idx)
            gt_id = str(gt.get("error_id") or "")
            if gt_id:
                matched_gt_ids.add(gt_id)
            matched_details.append(
                {
                    "gt_error_id": gt_id,
                    "gt_span": [g_start, g_end],
                    "pred_span": [int(pred.get("start_char") or -1), int(pred.get("end_char") or -1)],
                    "iou": iou,
                    "gt_coverage": gt_cov,
                    "pred_coverage": pred_cov,
                    "gt_error_text": str(gt.get("error_text") or ""),
                    "pred_text": str(pred.get("text") or ""),
                    "match_type": match_type,
                    "gt_paragraph_index": int(gt.get("paragraph_index") or -1),
                    "pred_paragraph_index": int(pred.get("paragraph_index") or -1),
                }
            )

    return matched_gt_ids, matched_details, used_pred_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate error-level metrics on GT-annotated wrong-answer dataset.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--audit", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--match-mode", type=str, default="location", choices=["location"])
    parser.add_argument("--location-iou-threshold", type=float, default=0.5)
    parser.add_argument("--location-coverage-threshold", type=float, default=0.6)
    args = parser.parse_args()

    ds = _load_json(args.dataset)
    pred = _load_json(args.results)
    audit = _load_json(args.audit)

    if not isinstance(ds, list):
        raise SystemExit("Dataset file must be a JSON array.")
    if not isinstance(pred, list):
        raise SystemExit("Results file must be a JSON array.")
    if not isinstance(audit, list):
        raise SystemExit("Audit file must be a JSON array.")

    try:
        _index_by_id(ds, label="dataset")
        pred_idx = _index_by_id(pred, label="results")
        audit_idx = _index_by_id(audit, label="audit")
        result_identity = _result_identity(pred)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    detail_rows: List[Dict[str, Any]] = []
    total_gt_errors = 0
    total_gt_locatable = 0
    matched_gt_errors = 0
    total_pred_findings = 0
    total_pred_locatable = 0
    total_location_matches = 0
    total_paragraph_matches = 0
    total_unmatched_gt_locatable = 0
    total_unmatched_pred_locatable = 0
    total_matched_pred_locatable = 0
    iou_values: List[float] = []
    false_positive_replay: List[Dict[str, Any]] = []
    failure_by_stage: Dict[str, int] = {}

    for row in ds:
        if not isinstance(row, dict):
            continue
        raw_id = row.get("id")
        sid_key = _typed_id_key(raw_id)
        sid = str(raw_id)

        gt_entries = _extract_gt_entries(row, sid)
        loc_gt_entries = [x for x in gt_entries if bool(x.get("locatable_valid"))]

        pred_item = pred_idx.get(sid_key)
        failure_reason = _execution_failure_reason(pred_item)
        if failure_reason:
            failure_by_stage[failure_reason] = failure_by_stage.get(failure_reason, 0) + 1
            detail_rows.append(
                {
                    "id": sid,
                    "gt_error_count": len(gt_entries),
                    "gt_locatable_error_count": len(loc_gt_entries),
                    "scored": False,
                    "execution_failure": failure_reason,
                }
            )
            continue
        assert isinstance(pred_item, dict)
        audit_item = audit_idx.get(sid_key, {})
        findings = _collect_pred_findings(pred_item, audit_item)
        findings = _fill_missing_pred_locations(findings, answer_text=str(row.get("prediction") or ""))

        pred_locatable_count = len([x for x in findings if bool(x.get("locatable_valid"))])

        total_gt_errors += len(gt_entries)
        total_gt_locatable += len(loc_gt_entries)
        total_pred_findings += len(findings)
        total_pred_locatable += pred_locatable_count

        matched_ids_location: Set[str] = set()
        location_matches: List[Dict[str, Any]] = []
        used_pred_loc_idx: Set[int] = set()
        if loc_gt_entries:
            matched_ids_location, location_matches, used_pred_loc_idx = _match_by_location(
                loc_gt_entries,
                findings,
                iou_threshold=float(args.location_iou_threshold),
                coverage_threshold=float(args.location_coverage_threshold),
            )

        matched_ids_total = set(matched_ids_location)
        matched_total = len(matched_ids_total)
        matched_loc = matched_total

        matched_gt_errors += matched_total
        total_matched_pred_locatable += len(used_pred_loc_idx)
        total_location_matches += len(location_matches)
        total_paragraph_matches += len(
            [m for m in location_matches if str(m.get("match_type") or "") in {"paragraph", "paragraph_region"}]
        )
        iou_values.extend([float(x.get("iou") or 0.0) for x in location_matches])

        loc_gt_ids = {str(x.get("error_id") or "") for x in loc_gt_entries if str(x.get("error_id") or "")}
        unmatched_gt_ids = sorted([gid for gid in loc_gt_ids if gid not in matched_ids_location])
        pred_locatable_indices = [idx for idx, x in enumerate(findings) if bool(x.get("locatable_valid"))]
        unmatched_pred_loc_indices = sorted([idx for idx in pred_locatable_indices if idx not in used_pred_loc_idx])
        unmatched_pred_loc_items = [findings[idx] for idx in unmatched_pred_loc_indices]
        for x in unmatched_pred_loc_items:
            rule_match = x.get("rule_match") if isinstance(x.get("rule_match"), dict) else {}
            publish_gate = rule_match.get("publish_gate") if isinstance(rule_match.get("publish_gate"), dict) else {}
            release_gate = x.get("release_gate") if isinstance(x.get("release_gate"), dict) else {}
            false_positive_replay.append(
                {
                    "id": sid,
                    "rule": str(x.get("rule") or ""),
                    "message": str(x.get("message") or ""),
                    "quote": str(x.get("quote") or ""),
                    "paragraph_index": int(x.get("paragraph_index") or -1),
                    "rule_score": float(rule_match.get("score") or release_gate.get("rule_score") or 0.0),
                    "score_kind": str(
                        rule_match.get("score_kind")
                        or release_gate.get("score_kind")
                        or "lexical"
                    ),
                    "min_score": float(rule_match.get("min_score") or 0.0),
                    "topic_rank": int(rule_match.get("topic_rank") or -1),
                    "topic_gap": float(rule_match.get("topic_gap") or 0.0),
                    "publish_gate": publish_gate,
                    "release_gate": release_gate,
                }
            )

        total_unmatched_gt_locatable += len(unmatched_gt_ids)
        total_unmatched_pred_locatable += len(unmatched_pred_loc_items)

        detail_rows.append(
            {
                "id": sid,
                "gt_error_count": len(gt_entries),
                "gt_locatable_error_count": len(loc_gt_entries),
                "pred_finding_count": len(findings),
                "pred_locatable_finding_count": pred_locatable_count,
                "matched_error_count": matched_total,
                "matched_location_error_count": matched_loc,
                "matched_paragraph_error_count": len(
                    [m for m in location_matches if str(m.get("match_type") or "") in {"paragraph", "paragraph_region"}]
                ),
                "sample_error_recall": (matched_total / len(gt_entries)) if gt_entries else 0.0,
                "sample_location_recall": (matched_loc / len(loc_gt_entries)) if loc_gt_entries else 0.0,
                "pred_has_diagnostic": bool((pred_item or {}).get("diagnostics")),
                "pred_has_error": bool(findings),
                "location_matches": location_matches,
                "unmatched_gt_error_ids": unmatched_gt_ids,
                "unmatched_pred_locatable_count": len(unmatched_pred_loc_items),
                "unmatched_pred_locatable_preview": [
                    {
                        "rule": str(x.get("rule") or ""),
                        "quote": str(x.get("quote") or ""),
                        "start_char": int(x.get("start_char") or -1),
                        "end_char": int(x.get("end_char") or -1),
                        "paragraph_index": int(x.get("paragraph_index") or -1),
                        "rule_score": float((x.get("rule_match") or {}).get("score") or 0.0)
                        if isinstance(x.get("rule_match"), dict)
                        else 0.0,
                        "score_kind": str((x.get("rule_match") or {}).get("score_kind") or "lexical")
                        if isinstance(x.get("rule_match"), dict)
                        else "lexical",
                        "topic_rank": int((x.get("rule_match") or {}).get("topic_rank") or -1)
                        if isinstance(x.get("rule_match"), dict)
                        else -1,
                        "release_reasons": list(((x.get("release_gate") or {}).get("reasons") or []))
                        if isinstance(x.get("release_gate"), dict)
                        else [],
                    }
                    for x in unmatched_pred_loc_items[:3]
                ],
                "scored": True,
                "execution_failure": "",
            }
        )

    total_samples = len(detail_rows)
    scored_samples = len([r for r in detail_rows if r.get("scored") is True])
    sample_triggered = sum(1 for r in detail_rows if r.get("scored") is True and r.get("pred_has_error") is True)
    sample_location_hit = sum(
        1
        for r in detail_rows
        if r.get("scored") is True and int(r.get("matched_location_error_count") or 0) > 0
    )

    recall = (matched_gt_errors / total_gt_errors) if total_gt_errors else 0.0
    precision = (total_matched_pred_locatable / total_pred_locatable) if total_pred_locatable else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    output = {
        "summary": {
            "level": "error",
            "match_mode": args.match_mode,
            "dataset_size": total_samples,
            "scored_size": scored_samples,
            "failed_size": sum(failure_by_stage.values()),
            "coverage": (scored_samples / total_samples) if total_samples else 0.0,
            "failure_by_stage": failure_by_stage,
            "checker_gate_mode": result_identity["checker_gate_mode"],
            "replay_config_sha256": result_identity["replay_config_sha256"],
            "total_gt_errors": total_gt_errors,
            "total_gt_locatable_errors": total_gt_locatable,
            "matched_gt_errors": matched_gt_errors,
            "matched_pred_locatable": total_matched_pred_locatable,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "recall_location_only": (matched_gt_errors / total_gt_locatable) if total_gt_locatable else 0.0,
            "gt_location_valid_ratio": (total_gt_locatable / total_gt_errors) if total_gt_errors else 0.0,
            "sample_trigger_ratio": (sample_triggered / scored_samples) if scored_samples else 0.0,
            "sample_location_hit_ratio": (sample_location_hit / scored_samples) if scored_samples else 0.0,
            "pred_location_coverage": (total_pred_locatable / total_pred_findings) if total_pred_findings else 0.0,
            "location_match_pairs": total_location_matches,
            "location_paragraph_match_pairs": total_paragraph_matches,
            "location_unmatched_gt_errors": total_unmatched_gt_locatable,
            "location_unmatched_pred_findings": total_unmatched_pred_locatable,
            "mean_iou_matched": (sum(iou_values) / len(iou_values)) if iou_values else 0.0,
        },
        "details": detail_rows,
        "false_positive_replay": false_positive_replay,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
