#!/usr/bin/env python3
"""Offline sim: short truncation + paragraph process reward vs answer_only.

Uses gold process errors from the error-level eval set (no GPU / judge).
Writes results/paragraph_process_offline_audit.json and recommended env defaults.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.reward_server.paragraph_process import (
    ProcessParagraphWeights,
    group_has_variance,
    paragraph_ranges,
    score_text_with_diagnostics,
    truncate_to_n_paragraphs,
)

LENGTH_GRIDS = (
    {"name": "120-180", "min_len": 90, "target_len": 150, "max_len": 180},
    {"name": "180-280", "min_len": 150, "target_len": 220, "max_len": 280},
    {"name": "280-400", "min_len": 220, "target_len": 320, "max_len": 400},
)
N_SAMPLES = 8
MIN_SPREAD = 1e-6


def _load_eval(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("samples") or data.get("items") or data.get("data") or []
    return [r for r in data if isinstance(r, dict) and str(r.get("prediction") or "").strip()]


def _gold_errors_in_span(row: Dict[str, Any], end_char: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for err in row.get("physics_error_gt") or []:
        if not isinstance(err, dict):
            continue
        start = int(err.get("start_char") if err.get("start_char") is not None else -1)
        if start < 0:
            # keep unlocated errors only for full-text variants
            if end_char >= len(str(row.get("prediction") or "")):
                e = dict(err)
                e.setdefault("severity", "error")
                out.append(e)
            continue
        if start < end_char:
            e = dict(err)
            e.setdefault("severity", "error")
            out.append(e)
    return out


def _first_error_offset(row: Dict[str, Any]) -> int | None:
    starts = []
    for err in row.get("physics_error_gt") or []:
        if not isinstance(err, dict):
            continue
        start = err.get("start_char")
        try:
            s = int(start)
        except (TypeError, ValueError):
            continue
        if s >= 0:
            starts.append(s)
    return min(starts) if starts else None


def _answer_only_score(text: str, label: str) -> float:
    # Cheap proxy: truncated prefixes almost never keep a matching boxed answer.
    # Import grading only when label is present.
    if not label or not text:
        return 0.0
    try:
        from training.compat.math_grading import grade_answer_verl
    except Exception:
        return 0.0
    return 1.0 if grade_answer_verl(text, label) else 0.0


def _variants_for_row(
    row: Dict[str, Any],
    *,
    min_len: int,
    target_len: int,
    max_len: int,
    max_chars: int,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    pred = str(row.get("prediction") or "")
    paras = paragraph_ranges(pred, min_len=min_len, target_len=target_len, max_len=max_len)
    variants: List[Tuple[str, List[Dict[str, Any]]]] = []

    def add(text: str) -> None:
        text = text or ""
        if max_chars > 0:
            text = text[:max_chars]
        variants.append((text, _gold_errors_in_span(row, len(text))))

    for k in (1, 2, 3):
        add(truncate_to_n_paragraphs(pred, k, min_len=min_len, target_len=target_len, max_len=max_len))
    add(pred)
    first = _first_error_offset(row)
    if first is not None and first > 0:
        add(pred[:first])
        add(pred[: min(len(pred), first + max(40, max_len))])
    else:
        add(pred[: max(1, len(pred) // 3)])
        add(pred[: max(1, len(pred) // 2)])
    if paras:
        last2 = paras[-2:] if len(paras) >= 2 else paras
        add(pred[int(last2[0]["start_char"]) : int(last2[-1]["end_char"])])
    else:
        add(pred[-max_len:] if pred else "")

    # Pad / trim to n_samples.
    if not variants:
        variants = [("", [])]
    while len(variants) < N_SAMPLES:
        variants.append(variants[len(variants) % max(1, len(variants))])
    return variants[:N_SAMPLES]


def _summarize_groups(group_rewards: List[List[float]]) -> Dict[str, Any]:
    stds = [statistics.pstdev(g) if len(g) > 1 else 0.0 for g in group_rewards]
    spreads = [(max(g) - min(g)) if g else 0.0 for g in group_rewards]
    n_eff = sum(1 for g in group_rewards if group_has_variance(g, MIN_SPREAD))
    n = max(len(group_rewards), 1)
    return {
        "n_groups": len(group_rewards),
        "effective_groups": n_eff,
        "effective_group_rate": 100.0 * n_eff / n,
        "mean_within_group_std": sum(stds) / n,
        "mean_within_group_spread": sum(spreads) / n,
        "mean_reward": sum(sum(g) / max(len(g), 1) for g in group_rewards) / n,
    }


def simulate_grid(rows: Sequence[Dict[str, Any]], max_chars: int) -> Dict[str, Any]:
    results = []
    for cfg in LENGTH_GRIDS:
        proc_groups: List[List[float]] = []
        ans_groups: List[List[float]] = []
        para_counts: List[int] = []
        judge_calls = 0
        for row in rows:
            variants = _variants_for_row(
                row,
                min_len=cfg["min_len"],
                target_len=cfg["target_len"],
                max_len=cfg["max_len"],
                max_chars=max_chars,
            )
            label = str(row.get("answer") or "")
            proc_rewards = []
            ans_rewards = []
            for text, diags in variants:
                scored = score_text_with_diagnostics(
                    text,
                    diags,
                    acc=False,
                    boxed=False,
                    weights=ProcessParagraphWeights(answer=0.0, format=0.0),
                    min_len=cfg["min_len"],
                    target_len=cfg["target_len"],
                    max_len=cfg["max_len"],
                    process_only=True,
                )
                proc_rewards.append(float(scored["score"]))
                ans_rewards.append(_answer_only_score(text, label))
                para_counts.append(int(scored["n_paragraphs"]))
                judge_calls += 1  # one verify per short completion
            proc_groups.append(proc_rewards)
            ans_groups.append(ans_rewards)
        results.append(
            {
                **cfg,
                "process_paragraph": _summarize_groups(proc_groups),
                "answer_only": _summarize_groups(ans_groups),
                "mean_paragraphs_per_variant": (sum(para_counts) / max(len(para_counts), 1)),
                "estimated_judge_calls": judge_calls,
            }
        )
    best = max(results, key=lambda r: (r["process_paragraph"]["effective_group_rate"], r["process_paragraph"]["mean_within_group_std"]))
    return {
        "n_eval_rows": len(rows),
        "n_samples_per_prompt": N_SAMPLES,
        "max_chars": max_chars,
        "min_spread": MIN_SPREAD,
        "process_only_reward": True,
        "grids": results,
        "recommended": {
            "name": best["name"],
            "min_len": best["min_len"],
            "target_len": best["target_len"],
            "max_len": best["max_len"],
            "effective_group_rate": best["process_paragraph"]["effective_group_rate"],
            "answer_only_effective_group_rate": best["answer_only"]["effective_group_rate"],
            "mean_within_group_std": best["process_paragraph"]["mean_within_group_std"],
        },
        "gate": {
            "process_better_than_answer_only": best["process_paragraph"]["effective_group_rate"]
            > best["answer_only"]["effective_group_rate"] + 1.0,
            "effective_group_rate_ge_5": best["process_paragraph"]["effective_group_rate"] >= 5.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/derived/expansion3000_scale_seed20260508/error_eval_dataset_100.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results/paragraph_process_offline_audit.json")
    parser.add_argument("--env-out", type=Path, default=ROOT / "training/openrlhf/paragraph_process_defaults.env")
    parser.add_argument("--max-chars", type=int, default=1536, help="~512 tokens at ~3 chars/token")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    rows = _load_eval(args.dataset)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        print(f"[error] no eval rows in {args.dataset}", file=sys.stderr)
        return 2
    payload = simulate_grid(rows, max_chars=args.max_chars)
    payload["dataset"] = str(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rec = payload["recommended"]
    args.env_out.parent.mkdir(parents=True, exist_ok=True)
    args.env_out.write_text(
        "\n".join(
            [
                f"export PHYSICS_REWARD_PARA_MIN={rec['min_len']}",
                f"export PHYSICS_REWARD_PARA_TARGET={rec['target_len']}",
                f"export PHYSICS_REWARD_PARA_MAX={rec['max_len']}",
                "export PHYSICS_REWARD_MODE=process_paragraph",
                "export PHYSICS_REWARD_VERIFIER_ON_WRONG=1",
                "export PHYSICS_REWARD_W_CLEAN=0.5",
                "export PHYSICS_REWARD_W_FIRST=0.3",
                "export PHYSICS_REWARD_W_DENSE=0.2",
                "export PHYSICS_REWARD_W_ANSWER=0",
                "export PHYSICS_REWARD_W_FORMAT=0",
                "export GENERATE_MAX_LEN=512",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "env_out": str(args.env_out),
                "recommended": rec,
                "gate": payload["gate"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
