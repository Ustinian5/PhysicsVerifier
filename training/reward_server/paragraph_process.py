"""Paragraph-level process reward helpers for 8B GRPO.

One verifier call on the full (short) completion; diagnostics are mapped onto
auto-chunked paragraphs and collapsed to a scalar with clean / first-error /
density terms. Shared by the reward server and the offline discriminability sim.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


PARA_MIN_LEN = _env_int("PHYSICS_REWARD_PARA_MIN", 150)
PARA_TARGET_LEN = _env_int("PHYSICS_REWARD_PARA_TARGET", 220)
PARA_MAX_LEN = _env_int("PHYSICS_REWARD_PARA_MAX", 280)

W_CLEAN = _env_float("PHYSICS_REWARD_W_CLEAN", 0.5)
W_FIRST = _env_float("PHYSICS_REWARD_W_FIRST", 0.3)
W_DENSE = _env_float("PHYSICS_REWARD_W_DENSE", 0.2)
# process_paragraph is process-only: final-answer / format terms stay off.
W_ANSWER = _env_float("PHYSICS_REWARD_W_ANSWER", 0.0)
W_FORMAT = _env_float("PHYSICS_REWARD_W_FORMAT", 0.0)


@dataclass(frozen=True)
class ProcessParagraphWeights:
    clean: float = W_CLEAN
    first: float = W_FIRST
    dense: float = W_DENSE
    answer: float = W_ANSWER
    format: float = W_FORMAT


DEFAULT_WEIGHTS = ProcessParagraphWeights()


def paragraph_ranges(
    text: str,
    *,
    min_len: int = PARA_MIN_LEN,
    target_len: int = PARA_TARGET_LEN,
    max_len: int = PARA_MAX_LEN,
) -> List[Dict[str, Any]]:
    """Chunk answer text into ~target_len character paragraphs.

    Same boundary heuristic as SemanticRuleChecker._paragraph_ranges, with
    tunable lengths for the short-sample process-reward path.
    """
    src = str(text or "")
    if not src:
        return []
    min_len = max(1, int(min_len))
    max_len = max(min_len, int(max_len))
    target_len = min(max(int(target_len), min_len), max_len)
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
            out.append(
                {
                    "paragraph_index": para_idx,
                    "start_char": s,
                    "end_char": e,
                    "text": src[s:e],
                }
            )
        start = end if end > start else start + 1

    if not out and src.strip():
        out.append({"paragraph_index": 1, "start_char": 0, "end_char": len(src), "text": src})
    return out


def truncate_to_n_paragraphs(text: str, n_keep: int, **chunk_kwargs: Any) -> str:
    paras = paragraph_ranges(text, **chunk_kwargs)
    if not paras or n_keep <= 0:
        return ""
    last = paras[min(n_keep, len(paras)) - 1]
    return str(text or "")[: int(last["end_char"])]


def diagnostic_weight(diag: Dict[str, Any]) -> float:
    """Return scoring weight for a diagnostic severity (error=1.0, warning=0.5)."""
    sev = str(diag.get("severity") or "error").strip().lower()
    if sev == "error":
        return 1.0
    if sev == "warning":
        return 0.5
    return 0.0


def diagnostic_span(diag: Dict[str, Any]) -> Tuple[int, int, int]:
    """Return (start_char, end_char, paragraph_index) with -1 if unknown."""
    loc = diag.get("location") if isinstance(diag.get("location"), dict) else {}
    start = diag.get("start_char", loc.get("start_char", -1))
    end = diag.get("end_char", loc.get("end_char", -1))
    pidx = diag.get("paragraph_index", loc.get("paragraph_index", -1))
    try:
        start_i = int(start) if start is not None else -1
    except (TypeError, ValueError):
        start_i = -1
    try:
        end_i = int(end) if end is not None else -1
    except (TypeError, ValueError):
        end_i = -1
    try:
        pidx_i = int(pidx) if pidx is not None else -1
    except (TypeError, ValueError):
        pidx_i = -1
    return start_i, end_i, pidx_i


def _overlaps(start: int, end: int, para: Dict[str, Any]) -> bool:
    ps = int(para.get("start_char") or 0)
    pe = int(para.get("end_char") or 0)
    if start < 0 or end <= start:
        return False
    return start < pe and end > ps


def map_errors_to_paragraphs(
    paragraphs: Sequence[Dict[str, Any]],
    diagnostics: Iterable[Dict[str, Any]],
) -> Dict[int, float]:
    """Sum severity-weighted diagnostics overlapping each paragraph_index."""
    counts: Dict[int, float] = {int(p["paragraph_index"]): 0.0 for p in paragraphs}
    by_idx = {int(p["paragraph_index"]): p for p in paragraphs}
    for diag in diagnostics:
        weight = diagnostic_weight(diag)
        if weight <= 0.0:
            continue
        start, end, pidx = diagnostic_span(diag)
        hit = False
        if start >= 0:
            for p in paragraphs:
                if _overlaps(start, end if end > start else start + 1, p):
                    counts[int(p["paragraph_index"])] += weight
                    hit = True
                    break
        if not hit and pidx in by_idx:
            counts[pidx] += weight
            hit = True
        if not hit and paragraphs:
            # Unlocated error still counts against the last paragraph (conservative).
            counts[int(paragraphs[-1]["paragraph_index"])] += weight
    return counts


def aggregate_process_components(
    paragraphs: Sequence[Dict[str, Any]],
    diagnostics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    n_paras = len(paragraphs)
    n_errors = sum(diagnostic_weight(d) for d in diagnostics if diagnostic_weight(d) > 0.0)
    if n_paras <= 0:
        return {
            "n_paragraphs": 0,
            "n_errors": n_errors,
            "n_bad_paragraphs": 0,
            "first_bad_paragraph_index": None,
            "errors_per_paragraph": {},
            "r_clean": 0.0,
            "r_first": 0.0,
            "r_dense": 0.0,
        }
    counts = map_errors_to_paragraphs(paragraphs, diagnostics)
    para_badness = {idx: min(1.0, float(w)) for idx, w in counts.items()}
    bad = [idx for idx, w in sorted(para_badness.items()) if w > 0]
    n_bad = len(bad)
    first_bad = bad[0] if bad else None
    weighted_bad_sum = sum(para_badness.values())
    r_clean = 1.0 - (weighted_bad_sum / n_paras)
    if first_bad is None:
        r_first = 1.0
    else:
        r_first = max(0.0, (int(first_bad) - 1) / n_paras)
    n_errors = sum(diagnostic_weight(d) for d in diagnostics if diagnostic_weight(d) > 0.0)
    r_dense = 1.0 / (1.0 + float(n_errors))
    return {
        "n_paragraphs": n_paras,
        "n_errors": n_errors,
        "n_bad_paragraphs": n_bad,
        "first_bad_paragraph_index": first_bad,
        "errors_per_paragraph": counts,
        "r_clean": float(r_clean),
        "r_first": float(r_first),
        "r_dense": float(r_dense),
    }


def combine_process_paragraph_score(
    *,
    acc: bool,
    boxed: bool,
    components: Dict[str, Any],
    weights: ProcessParagraphWeights = DEFAULT_WEIGHTS,
    process_only: bool = True,
) -> float:
    w_answer = 0.0 if process_only else weights.answer
    w_format = 0.0 if process_only else weights.format
    score = (
        weights.clean * float(components.get("r_clean") or 0.0)
        + weights.first * float(components.get("r_first") or 0.0)
        + weights.dense * float(components.get("r_dense") or 0.0)
        + w_answer * (1.0 if acc else 0.0)
        + w_format * (1.0 if boxed else 0.0)
    )
    return float(score)


def score_text_with_diagnostics(
    text: str,
    diagnostics: Sequence[Dict[str, Any]],
    *,
    acc: bool = False,
    boxed: bool = False,
    weights: ProcessParagraphWeights = DEFAULT_WEIGHTS,
    min_len: int = PARA_MIN_LEN,
    target_len: int = PARA_TARGET_LEN,
    max_len: int = PARA_MAX_LEN,
    process_only: bool = True,
) -> Dict[str, Any]:
    paras = paragraph_ranges(text, min_len=min_len, target_len=target_len, max_len=max_len)
    comps = aggregate_process_components(paras, diagnostics)
    score = combine_process_paragraph_score(
        acc=acc,
        boxed=boxed,
        components=comps,
        weights=weights,
        process_only=process_only,
    )
    out = dict(comps)
    out["score"] = score
    out["n_tokens_est"] = max(1, int(math.ceil(len(text) / 3.0))) if text else 0
    return out


def group_has_variance(rewards: Sequence[float], min_spread: float = 1e-6) -> bool:
    if not rewards:
        return False
    return (max(rewards) - min(rewards)) >= min_spread
