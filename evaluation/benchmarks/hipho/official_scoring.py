#!/usr/bin/env python3
"""HiPhO paper scoring: answer-level + marking-scheme step-level + exam/MNS."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from training.compat.math_grading import extract_boxed_answer, grade_answer_verl, remove_boxed

OFFICIAL_GRADER_MODEL = "gemini-2.5-flash"

# Official IPhO 2025 theory thresholds quoted in the HiPhO paper / dataset card.
DEFAULT_MEDAL_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "IPhO_2025": {"gold": 19.7, "silver": 12.1, "bronze": 7.2, "full": 30.0},
}


class GraderValidationError(ValueError):
    """Raised when a step-level grader payload fails schema/range checks."""


def _boxed_span(src: str, start: int) -> Optional[Tuple[int, int, str]]:
    boxed_at = src.find("\\boxed", start)
    fbox_at = src.find("\\fbox", start)
    candidates = [i for i in (boxed_at, fbox_at) if i >= 0]
    if not candidates:
        return None
    idx = min(candidates)
    i = idx
    right = None
    depth = 0
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    if right is None:
        return None
    return idx, right + 1, src[idx : right + 1]


def extract_all_boxed(text: str) -> List[str]:
    """Extract boxed answers in document order (not only the last one)."""
    src = str(text or "")
    answers: List[str] = []
    start = 0
    while True:
        span = _boxed_span(src, start)
        if span is None:
            break
        _lo, hi, chunk = span
        inner = remove_boxed(chunk)
        answers.append(inner if inner is not None else chunk)
        start = hi
    return answers


def extract_subquestion_answers(prediction: str, n: int) -> List[str]:
    boxed = extract_all_boxed(prediction)
    if not boxed:
        last = extract_boxed_answer(prediction)
        boxed = [last] if last else [""]
    if n <= 1:
        return [boxed[-1] if boxed else ""]
    if len(boxed) >= n:
        return boxed[:n]
    out = list(boxed) + [""] * (n - len(boxed))
    return out


def rule_based_equivalent(prediction_or_answer: str, gold: str) -> bool:
    """Kydlíček-style math verifier used by the paper, implemented via local mathd+sympy."""
    if not gold:
        return False
    pred = str(prediction_or_answer or "")
    if "\\boxed" not in pred and pred.strip():
        pred = "\\boxed{" + pred.strip() + "}"
    gold_s = str(gold)
    if "\\boxed" not in gold_s and gold_s.strip():
        gold_s = "\\boxed{" + gold_s.strip() + "}"
    if grade_answer_verl(pred, gold_s):
        return True
    if grade_answer_verl("\\boxed{" + pred.strip() + "}", gold_s):
        return True
    return False


def answer_level_score(
    prediction: str,
    gold_answers: Sequence[str],
    full_marks: Sequence[float],
    *,
    llm_equivalent: Optional[Callable[[str, str], bool]] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    n = max(len(gold_answers), len(full_marks), 1)
    golds = list(gold_answers) + [""] * max(0, n - len(gold_answers))
    marks = list(full_marks) + ([full_marks[-1] if full_marks else 1.0] * max(0, n - len(full_marks)))
    preds = extract_subquestion_answers(prediction, n)
    details: List[Dict[str, Any]] = []
    total = 0.0
    for i in range(n):
        gold = golds[i]
        mark = float(marks[i])
        pred_i = preds[i] if i < len(preds) else ""
        method = "none"
        correct = False
        if gold:
            if rule_based_equivalent(pred_i, gold) or rule_based_equivalent(prediction, gold):
                correct = True
                method = "rule"
            elif llm_equivalent is not None:
                try:
                    correct = bool(llm_equivalent(pred_i or prediction, gold))
                    method = "llm" if correct or pred_i else "llm"
                except Exception as exc:  # noqa: BLE001
                    method = f"llm_error:{type(exc).__name__}"
                    correct = False
        awarded = mark if correct else 0.0
        total += awarded
        details.append(
            {
                "subquestion": i,
                "gold": gold,
                "predicted": pred_i,
                "full_mark": mark,
                "awarded": awarded,
                "correct": correct,
                "method": method,
            }
        )
    return total, details


def step_score_from_criteria(
    criterion_scores: Sequence[Dict[str, Any]],
    criteria: Sequence[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    by_id = {str(item.get("id")): item for item in criterion_scores if isinstance(item, dict)}
    if len(by_id) != len(criteria):
        raise GraderValidationError(
            f"grader returned {len(by_id)} criteria, expected {len(criteria)}"
        )
    audited: List[Dict[str, Any]] = []
    total = 0.0
    for crit in criteria:
        cid = str(crit["id"])
        if cid not in by_id:
            raise GraderValidationError(f"missing criterion {cid}")
        item = by_id[cid]
        weight = float(crit["weight"])
        if "s" in item and item.get("s") is not None:
            s_ij = float(item["s"])
        elif "completion" in item:
            s_ij = float(item["completion"])
        elif weight:
            s_ij = float(item.get("awarded_points", 0.0)) / weight
        else:
            s_ij = 0.0
        if not math.isfinite(s_ij) or s_ij < 0.0 - 1e-9 or s_ij > 1.0 + 1e-9:
            raise GraderValidationError(f"criterion {cid} completion {s_ij} not in [0, 1]")
        s_ij = min(1.0, max(0.0, s_ij))
        awarded = weight * s_ij
        reported = item.get("awarded_points")
        if reported is not None and abs(float(reported) - awarded) > 1e-4:
            raise GraderValidationError(
                f"criterion {cid} awarded_points {reported} != weight*s {awarded}"
            )
        if awarded < -1e-9 or awarded > weight + 1e-9:
            raise GraderValidationError(f"criterion {cid} awarded {awarded} outside [0, {weight}]")
        total += awarded
        audited.append(
            {
                "id": cid,
                "description": crit.get("description", ""),
                "weight": weight,
                "s": s_ij,
                "awarded_points": awarded,
                "evidence": item.get("evidence") or item.get("quote") or "",
                "reason": item.get("reason") or item.get("brief_reason") or "",
            }
        )
    return total, audited


def best_step_score(
    schemes: Sequence[Dict[str, Any]],
    score_scheme: Callable[[Dict[str, Any]], Tuple[float, List[Dict[str, Any]]]],
) -> Tuple[float, Dict[str, Any]]:
    if not schemes:
        return 0.0, {"scheme": None, "criteria": []}
    best_score = float("-inf")
    best_payload: Dict[str, Any] = {}
    for scheme in schemes:
        score, audited = score_scheme(scheme)
        if score > best_score:
            best_score = score
            best_payload = {"scheme": scheme.get("name"), "criteria": audited, "step_score": score}
    return float(best_score), best_payload


def problem_score(answer_level: float, step_level: float) -> float:
    return max(float(answer_level), float(step_level))


def exam_totals(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    exams: Dict[str, Dict[str, float]] = {}
    for row in rows:
        exam = str(row.get("exam") or "unknown")
        bucket = exams.setdefault(exam, {"points": 0.0, "full_marks": 0.0, "n": 0.0})
        bucket["points"] += float(row.get("final_score") or 0.0)
        bucket["full_marks"] += float(row.get("full_mark") or 0.0)
        bucket["n"] += 1.0
    for bucket in exams.values():
        full = bucket["full_marks"]
        bucket["normalized"] = (bucket["points"] / full) if full else 0.0
    return exams


def mean_normalized_score(exam_stats: Dict[str, Dict[str, float]]) -> float:
    if not exam_stats:
        return 0.0
    return sum(v.get("normalized", 0.0) for v in exam_stats.values()) / len(exam_stats)


def medal_for_points(exam: str, points: float, thresholds: Optional[Dict[str, Dict[str, float]]] = None) -> str:
    table = (thresholds or DEFAULT_MEDAL_THRESHOLDS).get(exam) or {}
    if not table:
        return "unknown"
    if points >= float(table.get("gold", math.inf)):
        return "gold"
    if points >= float(table.get("silver", math.inf)):
        return "silver"
    if points >= float(table.get("bronze", math.inf)):
        return "bronze"
    return "none"


def score_problem_record(
    *,
    prediction: str,
    gold_answers: Sequence[str],
    full_marks: Sequence[float],
    marking_schemes: Sequence[Dict[str, Any]],
    llm_equivalent: Optional[Callable[[str, str], bool]] = None,
    step_grader: Optional[Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    answer_score, answer_details = answer_level_score(
        prediction,
        gold_answers,
        full_marks,
        llm_equivalent=llm_equivalent,
    )
    step_details: Dict[str, Any] = {"scheme": None, "criteria": []}
    if marking_schemes and step_grader is not None:
        def _score_scheme(scheme: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
            raw = step_grader(prediction, scheme)
            return step_score_from_criteria(raw, scheme.get("criteria") or [])

        step_score, step_details = best_step_score(marking_schemes, _score_scheme)
    elif marking_schemes and step_grader is None:
        step_score = 0.0
        step_details = {"scheme": None, "criteria": [], "skipped": "no_step_grader"}
    else:
        step_score = 0.0
    final = problem_score(answer_score, step_score)
    return {
        "answer_score": answer_score,
        "step_score": step_score,
        "final_score": final,
        "answer_details": answer_details,
        "step_details": step_details,
        "full_mark": float(sum(full_marks)) if full_marks else 0.0,
    }
