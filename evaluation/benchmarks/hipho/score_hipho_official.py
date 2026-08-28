#!/usr/bin/env python3
"""Official HiPhO-TO scorer: answer-level + marking-scheme step-level + exam/MNS.

Predictions must not include gold answers or marking schemes. Gold is loaded
from a separate official jsonl in this process only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmarks.hipho.hipho_contract import (
    OfficialHiPhOError,
    is_internal_expansion_row,
    is_official_text_only,
    load_jsonl,
    load_manifest,
)
from evaluation.benchmarks.hipho.official_scoring import (
    DEFAULT_MEDAL_THRESHOLDS,
    OFFICIAL_GRADER_MODEL,
    exam_totals,
    mean_normalized_score,
    medal_for_points,
    score_problem_record,
)

GOLD_KEYS = {
    "answer",
    "answers",
    "label",
    "labels",
    "marking",
    "marking_scheme",
    "marking_schemes",
    "ground_truth",
}


class OfficialGraderUnavailable(RuntimeError):
    pass


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def list_models(base_url: str, api_key: str) -> List[str]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key or "EMPTY")
    return [m.id for m in client.models.list().data]


def _index_gold(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if is_internal_expansion_row(row):
            raise OfficialHiPhOError("gold file contains internal expansion rows")
        if not is_official_text_only(row):
            continue
        key = str(row.get("id") or "")
        if not key:
            raise OfficialHiPhOError("gold row missing id")
        out[key] = row
    return out


def _prediction_id(row: Dict[str, Any]) -> str:
    return str(row.get("id") or row.get("problem_id") or row.get("sample_id") or "")


def _strip_gold(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k not in GOLD_KEYS}


def _make_openai():
    from openai import OpenAI

    return OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "").rstrip("/"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        timeout=float(os.environ.get("HIPHO_GRADER_TIMEOUT", "180")),
    )


def _llm_equivalent_factory(model: str):
    client = _make_openai()

    def _equiv(predicted: str, gold: str) -> bool:
        messages = [
            {
                "role": "system",
                "content": (
                    "You judge whether two physics answers are mathematically/physically equivalent. "
                    "Reply JSON {\"equivalent\": true|false, \"reason\": \"...\"} only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"predicted": predicted, "gold": gold}, ensure_ascii=False),
            },
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        data = json.loads(text)
        return bool(data.get("equivalent"))

    return _equiv


def _step_grader_factory(model: str):
    client = _make_openai()

    def _grade(prediction: str, scheme: Dict[str, Any]) -> List[Dict[str, Any]]:
        criteria = scheme.get("criteria") or []
        payload = {
            "task": "score_official_marking_scheme",
            "solution": prediction,
            "criteria": criteria,
            "output_schema": {
                "candidates": [
                    {
                        "id": "criterion_id",
                        "s": 0.0,
                        "awarded_points": 0.0,
                        "evidence": "quote",
                        "reason": "short",
                    }
                ]
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the official HiPhO step grader. Score ONLY the provided marking-scheme criteria. "
                    "Do not invent extra criteria. s is completion in [0,1] using the official partial-credit rule. "
                    "awarded_points must equal weight * s. Quote brief evidence from the solution. JSON only."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        items = data.get("criteria") or data.get("candidates") or data.get("scores") or []
        return list(items)

    return _grade


def score_rows(
    predictions: List[Dict[str, Any]],
    gold_by_id: Dict[str, Dict[str, Any]],
    *,
    llm_equivalent=None,
    step_grader=None,
    official_reproduction: bool,
    grader_model: str,
) -> Dict[str, Any]:
    per_row: List[Dict[str, Any]] = []
    missing = 0
    for pred_row in predictions:
        pid = _prediction_id(pred_row)
        gold = gold_by_id.get(pid)
        if gold is None:
            missing += 1
            continue
        result = score_problem_record(
            prediction=str(pred_row.get("prediction") or ""),
            gold_answers=list(gold.get("answer") or []),
            full_marks=list(gold.get("points") or [gold.get("full_mark") or 0.0]),
            marking_schemes=list(gold.get("marking_schemes") or []),
            llm_equivalent=llm_equivalent,
            step_grader=step_grader,
        )
        rec = {
            "id": pid,
            "exam": gold.get("exam"),
            "field": gold.get("field"),
            "answer_type": gold.get("answer_type"),
            "modality": gold.get("modality"),
            "full_mark": gold.get("full_mark"),
            **result,
        }
        per_row.append(rec)

    exams = exam_totals(per_row)
    medals = {
        exam: medal_for_points(exam, stats["points"])
        for exam, stats in exams.items()
    }
    boxed_hits = 0
    for rec in per_row:
        boxed_hits += int(any(d.get("correct") for d in rec.get("answer_details") or []))
    summary = {
        "benchmark": "HiPhO-TO",
        "official_reproduction": official_reproduction,
        "grader_model": grader_model,
        "grader_status": "official" if official_reproduction else "non_official_grader",
        "n_predictions": len(predictions),
        "n_scored": len(per_row),
        "n_missing_gold": missing,
        "exam_points": {k: v["points"] for k, v in exams.items()},
        "exam_full_marks": {k: v["full_marks"] for k, v in exams.items()},
        "exam_normalized": {k: v["normalized"] for k, v in exams.items()},
        "exam_medals": medals,
        "mns": mean_normalized_score(exams),
        "total_points": sum(v["points"] for v in exams.values()),
        "total_full_marks": sum(v["full_marks"] for v in exams.values()),
        "boxed_acc": boxed_hits / max(len(per_row), 1),
        "rows": per_row,
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit-jsonl", type=Path, default=None)
    p.add_argument("--grader-model", default=os.environ.get("HIPHO_GRADER_MODEL", OFFICIAL_GRADER_MODEL))
    p.add_argument("--allow-non-official", action="store_true")
    args = p.parse_args()

    _load_env(ROOT / ".env")
    if args.manifest:
        load_manifest(args.manifest)

    gold_rows = load_jsonl(args.gold)
    if any(is_internal_expansion_row(r) for r in gold_rows):
        raise OfficialHiPhOError("refusing to score against internal expansion gold")
    gold_by_id = _index_gold(gold_rows)
    predictions = [_strip_gold(r) for r in load_jsonl(args.predictions)]

    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    official = False
    llm_equivalent = None
    step_grader = None
    grader_model = args.grader_model
    try:
        if not base_url:
            raise OfficialGraderUnavailable("OPENAI_BASE_URL missing")
        models = list_models(base_url, api_key)
        if grader_model not in models:
            raise OfficialGraderUnavailable(
                f"{grader_model} not in remote /v1/models (have {models[:8]})"
            )
        if grader_model != OFFICIAL_GRADER_MODEL:
            raise OfficialGraderUnavailable(
                f"grader {grader_model} is not the paper model {OFFICIAL_GRADER_MODEL}"
            )
        llm_equivalent = _llm_equivalent_factory(grader_model)
        step_grader = _step_grader_factory(grader_model)
        official = True
    except OfficialGraderUnavailable as exc:
        if not args.allow_non_official:
            print(f"[error] official HiPhO grader unavailable: {exc}", file=sys.stderr)
            print("[error] refusing to publish paper-style HiPhO scores", file=sys.stderr)
            return 2
        print(f"[warn] non_official_grader: {exc}", file=sys.stderr)
        grader_model = f"non_official:{grader_model}"

    started = time.time()
    summary = score_rows(
        predictions,
        gold_by_id,
        llm_equivalent=llm_equivalent,
        step_grader=step_grader,
        official_reproduction=official,
        grader_model=grader_model,
    )
    summary["elapsed_sec"] = time.time() - started
    summary["predictions"] = str(args.predictions)
    summary["gold"] = str(args.gold)
    summary["medal_thresholds"] = DEFAULT_MEDAL_THRESHOLDS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.audit_jsonl:
        args.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.audit_jsonl.open("w", encoding="utf-8") as f:
            for row in summary.get("rows") or []:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    public = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(public, ensure_ascii=False))
    return 0 if official or args.allow_non_official else 2


if __name__ == "__main__":
    raise SystemExit(main())
