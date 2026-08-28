#!/usr/bin/env python3
"""Score HiPhO predictions with answer-level + verifier process metrics."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from training.compat.math_grading import grade_answer_verl

DEFAULT_UNIFIED_RULES = ROOT / "catalogs/rules_unified_3000_runtime_backfilled.json"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _labels(answer: Any) -> List[str]:
    if answer is None:
        return []
    if isinstance(answer, list):
        return [str(x) for x in answer]
    return [str(answer)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=None, help="Optional gold jsonl if predictions were stripped")
    parser.add_argument("--use-verifier", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    rows = _load_jsonl(args.predictions)
    gold_by_id = {}
    if args.gold:
        for grow in _load_jsonl(args.gold):
            gid = str(grow.get("id") or grow.get("sample_id") or (grow.get("metadata") or {}).get("sample_id") or "")
            if gid:
                gold_by_id[gid] = grow
    verifier = None
    if args.use_verifier:
        from core.physics_rule_verifier import PhysicsRuleVerifier

        symbolic_enabled = os.environ.get("PHYSICSVERIFIER_SYMBOLIC_ENABLED", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        verifier = PhysicsRuleVerifier(
            llm_model=os.environ.get(
                "PHYSICSVERIFIER_LLM_MODEL",
                "qwen3-30b-a3b-instruct-2507",
            ),
            unified_rules_path=os.environ.get(
                "PHYSICSVERIFIER_UNIFIED_RULES",
                str(DEFAULT_UNIFIED_RULES),
            ),
            enable_symbolic_check=symbolic_enabled,
            unified_retrieval_mode=os.environ.get(
                "PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE",
                "semantic",
            ),
            semantic_output_adapter=os.environ.get("PHYSICSVERIFIER_SEMANTIC_OUTPUT_ADAPTER") or None,
        )

    acc_hits = 0
    total_errors = 0
    per_exam: Dict[str, Dict[str, float]] = {}

    for row in rows:
        pred = row.get("prediction", "")
        labels = _labels(row.get("answer") or row.get("label"))
        if not labels:
            gid = str(row.get("id") or row.get("sample_id") or (row.get("metadata") or {}).get("sample_id") or "")
            gold = gold_by_id.get(gid, {})
            labels = _labels(gold.get("answer") or gold.get("label"))
        acc = any(grade_answer_verl(pred, gt) for gt in labels) if labels else False
        acc_hits += int(acc)

        n_errors = 0
        if verifier is not None:
            result = verifier.verify({"question": row.get("question", ""), "prediction": pred})
            n_errors = sum(
                1 for d in (result.get("diagnostics") or []) if str(d.get("severity", "")).lower() == "error"
            )
        total_errors += n_errors

        exam = str((row.get("metadata") or {}).get("exam") or "unknown")
        bucket = per_exam.setdefault(exam, {"n": 0, "acc": 0, "errors": 0})
        bucket["n"] += 1
        bucket["acc"] += int(acc)
        bucket["errors"] += n_errors

    n = max(len(rows), 1)
    summary = {
        "n_samples": len(rows),
        "boxed_acc": acc_hits / n,
        "answer_acc": acc_hits / n,
        "metric_note": "boxed_acc is a diagnostic binary match, not official HiPhO.",
        "avg_process_errors": total_errors / n,
        "per_exam": {
            k: {
                "n": int(v["n"]),
                "boxed_acc": v["acc"] / max(v["n"], 1),
                "answer_acc": v["acc"] / max(v["n"], 1),
                "avg_process_errors": v["errors"] / max(v["n"], 1),
            }
            for k, v in per_exam.items()
        },
        "predictions": str(args.predictions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
