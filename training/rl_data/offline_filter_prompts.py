#!/usr/bin/env python3
"""Offline filter RL prompts using reward server scores from baseline rollouts.

Also builds a bootstrap GRPO curriculum: single parseable answer, no multi-question
single-label items, length-capped prompts, optional pass-rate banding, and an
offline legacy-vs-variance filter audit on the same rewards.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]

NUMBERED_RE = re.compile(
    r"(?:(?:^|\n)\s*(?:\(?[1-9]\)|[1-9]\.|[A-Da-d][\.\)])\s+\S)"
    r"|(?:\b(?:Part|Task|Question)\s+[A-D1-9]\b)"
    r"|(?:\([1-9]\)\s)",
    re.IGNORECASE,
)
MULTI_ASK_RE = re.compile(
    r"\b(?:find|determine|calculate|what are|evaluate)\b",
    re.IGNORECASE,
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _question_from_row(row: Dict[str, Any]) -> str:
    meta_q = str((row.get("metadata") or {}).get("question") or "")
    if meta_q.strip():
        return meta_q
    inp = row.get("input")
    if isinstance(inp, list):
        parts = []
        for msg in inp:
            if isinstance(msg, dict) and msg.get("role") == "user":
                parts.append(str(msg.get("content") or ""))
        return "\n".join(parts)
    return str(inp or "")


def _label_from_row(row: Dict[str, Any]) -> List[str]:
    label = row.get("label")
    if isinstance(label, list):
        return [str(x) for x in label if x is not None and str(x).strip()]
    if label is None:
        return []
    text = str(label).strip()
    return [text] if text else []


def extract_boxed_contents(text: str) -> List[str]:
    """Return inner contents of each \\boxed{...} using brace matching."""
    contents: List[str] = []
    needle = "\\boxed"
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        j = idx + len(needle)
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text) or text[j] != "{":
            start = j
            continue
        depth = 0
        k = j
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    contents.append(text[j + 1 : k].strip())
                    break
            k += 1
        start = k + 1 if k < len(text) else j + 1
    return contents


def parseable_single_answer(labels: Sequence[str]) -> Tuple[bool, str]:
    if not labels:
        return False, "empty_label"
    joined = "\n".join(labels)
    boxed = extract_boxed_contents(joined)
    if not boxed:
        # Accept a non-empty raw label as a single answer if there is only one label.
        if len(labels) == 1 and labels[0].strip():
            return True, "raw_label"
        return False, "unparseable_label"
    nonempty = [b for b in boxed if b]
    if not nonempty:
        return False, "empty_boxed"
    if len(nonempty) > 1 or len(labels) > 1:
        return False, "multiple_final_answers"
    return True, "ok"


def looks_multi_question(question: str, n_labels: int) -> bool:
    markers = NUMBERED_RE.findall(question or "")
    unique_markers = {m.strip().lower() for m in markers}
    if len(unique_markers) >= 2 and n_labels <= 1:
        return True
    asks = MULTI_ASK_RE.findall(question or "")
    if len(asks) >= 3 and n_labels <= 1 and len(question) > 1200:
        return True
    return False


def quality_drop_reason(
    row: Dict[str, Any],
    *,
    max_prompt_chars: int,
) -> Optional[str]:
    question = _question_from_row(row)
    labels = _label_from_row(row)
    if not question.strip():
        return "empty_prompt"
    if max_prompt_chars > 0 and len(question) > max_prompt_chars:
        return "prompt_too_long"
    ok, reason = parseable_single_answer(labels)
    if not ok:
        return reason
    if looks_multi_question(question, len(labels)):
        return "multi_question_single_label"
    return None


def _pass_rate_from_group(accs: Sequence[bool], scores: Sequence[float]) -> float:
    if accs:
        return sum(1 for a in accs if a) / len(accs)
    if scores:
        return sum(1 for s in scores if s >= 0.5) / len(scores)
    return float("nan")


def simulate_group_rewards(
    pass_rate: float,
    n_samples: int,
    rng: random.Random,
    *,
    format_bonus: float = 0.0,
) -> List[float]:
    rewards = []
    for _ in range(n_samples):
        correct = rng.random() < max(0.0, min(1.0, pass_rate))
        if correct:
            rewards.append(1.0)
        else:
            rewards.append(format_bonus)
    return rewards


def _try_import_filter():
    import importlib.util

    path = Path("/slow_share/jinjianhan/workspace/openrlhf_rl/OpenRLHF/openrlhf/trainer/ppo_utils/dynamic_filter.py")
    spec = importlib.util.spec_from_file_location("openrlhf_dynamic_filter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load dynamic_filter from {path}")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["openrlhf_dynamic_filter"] = mod
    spec.loader.exec_module(mod)
    return mod.MODE_MEAN_RANGE, mod.MODE_REWARD_VARIANCE, mod.FilterConfig, mod.simulate_filter_rates


def audit_filters(
    rows: Sequence[Dict[str, Any]],
    *,
    n_samples: int,
    seed: int,
    format_bonus: float,
) -> Dict[str, Any]:
    _, mode_var, FilterConfig, simulate_filter_rates = _try_import_filter()
    rng = random.Random(seed)
    groups: List[List[float]] = []
    for row in rows:
        pr = (row.get("metadata") or {}).get("pass_rate")
        if pr is None or (isinstance(pr, float) and math.isnan(pr)):
            pr = 0.15
        groups.append(simulate_group_rewards(float(pr), n_samples, rng, format_bonus=format_bonus))
    variance_cfg = FilterConfig(
        mode=mode_var,
        n_samples_per_prompt=n_samples,
        rollout_batch_size=1,
    )
    mixed = simulate_filter_rates(groups, variance_cfg)
    all_wrong_bonus = simulate_filter_rates([[format_bonus] * n_samples for _ in range(max(len(rows), 1))], variance_cfg)
    all_wrong_zero = simulate_filter_rates([[0.0] * n_samples for _ in range(max(len(rows), 1))], variance_cfg)
    return {
        "n_groups": len(groups),
        "n_samples_per_prompt": n_samples,
        "format_bonus": format_bonus,
        "simulated_from_pass_rate": mixed,
        "all_wrong_format_bonus": all_wrong_bonus,
        "all_wrong_zero": all_wrong_zero,
    }


def build_curriculum(
    rows: Sequence[Dict[str, Any]],
    *,
    max_prompt_chars: int,
    min_pass_rate: Optional[float],
    max_pass_rate: Optional[float],
    max_keep: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    drop_reasons: Counter[str] = Counter()
    source_kept: Counter[str] = Counter()
    pass_buckets: Counter[str] = Counter()
    kept: List[Dict[str, Any]] = []
    for row in rows:
        reason = quality_drop_reason(row, max_prompt_chars=max_prompt_chars)
        if reason:
            drop_reasons[reason] += 1
            continue
        meta = dict(row.get("metadata") or {})
        pr = meta.get("pass_rate")
        if pr is not None and min_pass_rate is not None and max_pass_rate is not None:
            try:
                prf = float(pr)
            except (TypeError, ValueError):
                drop_reasons["bad_pass_rate"] += 1
                continue
            if not (min_pass_rate <= prf <= max_pass_rate):
                drop_reasons["pass_rate_out_of_band"] += 1
                continue
            if prf < 0.05:
                pass_buckets["<0.05"] += 1
            elif prf <= 0.40:
                pass_buckets["0.05-0.40"] += 1
            elif prf < 0.95:
                pass_buckets["0.40-0.95"] += 1
            else:
                pass_buckets[">=0.95"] += 1
        else:
            pass_buckets["unknown"] += 1
        out = dict(row)
        out["metadata"] = meta
        kept.append(out)
        source_kept[str(meta.get("source") or "unknown")] += 1

    rng = random.Random(seed)
    rng.shuffle(kept)
    if max_keep > 0:
        kept = kept[:max_keep]
    audit = {
        "input_rows": len(rows),
        "kept": len(kept),
        "drop_reasons": dict(drop_reasons),
        "source_distribution": dict(source_kept),
        "pass_rate_buckets": dict(pass_buckets),
        "max_prompt_chars": max_prompt_chars,
        "min_pass_rate": min_pass_rate,
        "max_pass_rate": max_pass_rate,
        "max_keep": max_keep,
        "seed": seed,
    }
    return kept, audit


def _attach_pass_rates(
    prompts: List[Dict[str, Any]],
    rollout_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"scores": [], "accs": []})
    for rec in rollout_rows:
        key = _question_from_row(rec) or json.dumps(rec.get("input"), ensure_ascii=False)
        reward = rec.get("reward") or {}
        if isinstance(reward, dict):
            groups[key]["scores"].append(float(reward.get("score", 0.0)))
            groups[key]["accs"].append(bool(reward.get("acc", False)))
        else:
            groups[key]["scores"].append(float(reward))
            groups[key]["accs"].append(float(reward) > 0.5)
    out = []
    for row in prompts:
        key = _question_from_row(row) or json.dumps(row.get("input"), ensure_ascii=False)
        g = groups.get(key)
        row = dict(row)
        meta = dict(row.get("metadata") or {})
        if g and g["accs"]:
            meta["pass_rate"] = _pass_rate_from_group(g["accs"], g["scores"])
            meta["n_rollouts"] = len(g["accs"])
            meta["avg_score"] = sum(g["scores"]) / len(g["scores"])
        row["metadata"] = meta
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["legacy", "curriculum"], default="legacy")
    parser.add_argument("--rollout-scores", type=Path, default=None, help="jsonl: prompt row + rollout responses + reward")
    parser.add_argument("--prompts", type=Path, default=ROOT / "data/rl/openrlhf_prompts.jsonl")
    parser.add_argument("--output-rl", type=Path, default=ROOT / "data/rl/rl_prompts.jsonl")
    parser.add_argument("--output-heldout", type=Path, default=ROOT / "data/rl/heldout_eval.jsonl")
    parser.add_argument("--output-sft", type=Path, default=ROOT / "data/rl/sft_data.jsonl")
    parser.add_argument("--output-audit", type=Path, default=None)
    parser.add_argument("--heldout-size", type=int, default=150)
    parser.add_argument("--min-pass-rate", type=float, default=0.05)
    parser.add_argument("--max-pass-rate", type=float, default=0.95)
    parser.add_argument("--max-prompt-chars", type=int, default=2500)
    parser.add_argument("--max-keep", type=int, default=0)
    parser.add_argument("--n-samples-per-prompt", type=int, default=8)
    parser.add_argument("--format-bonus", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "curriculum":
        rows = _load_jsonl(args.prompts)
        if args.rollout_scores and args.rollout_scores.is_file():
            rows = _attach_pass_rates(rows, _load_jsonl(args.rollout_scores))
            min_pr, max_pr = args.min_pass_rate, args.max_pass_rate
        else:
            min_pr, max_pr = None, None
        kept, audit = build_curriculum(
            rows,
            max_prompt_chars=args.max_prompt_chars,
            min_pass_rate=min_pr,
            max_pass_rate=max_pr,
            max_keep=args.max_keep,
            seed=args.seed,
        )
        audit["filter_simulation"] = audit_filters(
            kept,
            n_samples=args.n_samples_per_prompt,
            seed=args.seed,
            format_bonus=args.format_bonus,
        )
        if rows:
            _, full_audit = build_curriculum(
                rows,
                max_prompt_chars=10**9,
                min_pass_rate=None,
                max_pass_rate=None,
                max_keep=0,
                seed=args.seed,
            )
            # Compare simulated effective-group rate: curriculum vs unfiltered quality-pass-through.
            audit["unfiltered_quality_only_keep"] = full_audit["kept"]
            audit["filter_simulation_unfiltered"] = audit_filters(
                rows[: min(len(rows), max(len(kept), 1))],
                n_samples=args.n_samples_per_prompt,
                seed=args.seed,
                format_bonus=args.format_bonus,
            )
        _write_jsonl(args.output_rl, kept)
        audit_path = args.output_audit or args.output_rl.with_suffix(".audit.json")
        audit["output_rl"] = str(args.output_rl)
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"kept": len(kept), "audit": str(audit_path)}, ensure_ascii=False))
        return

    if args.rollout_scores is None:
        parser.error("--rollout-scores is required in legacy mode")

    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"scores": [], "accs": [], "row": None})
    for rec in _load_jsonl(args.rollout_scores):
        key = _question_from_row(rec) or json.dumps(rec.get("input"), ensure_ascii=False)
        groups[key]["row"] = rec
        reward = rec.get("reward") or {}
        if isinstance(reward, dict):
            groups[key]["scores"].append(float(reward.get("score", 0.0)))
            groups[key]["accs"].append(bool(reward.get("acc", False)))
        else:
            groups[key]["scores"].append(float(reward))
            groups[key]["accs"].append(float(reward) > 0.5)

    candidates: List[Dict[str, Any]] = []
    sft_rows: List[Dict[str, Any]] = []

    for _, g in groups.items():
        row = g["row"]
        if row is None:
            continue
        n = len(g["accs"])
        if n == 0:
            continue
        pass_rate = sum(1 for a in g["accs"] if a) / n
        avg_score = sum(g["scores"]) / n

        meta = dict(row.get("metadata") or {})
        meta.update(
            {
                "pass_rate": pass_rate,
                "avg_score": avg_score,
                "n_rollouts": n,
            }
        )
        row = dict(row)
        row["metadata"] = meta

        if args.min_pass_rate < pass_rate < args.max_pass_rate:
            candidates.append(row)

        # Reject-sampling SFT: correct + zero process errors
        if pass_rate > 0 and all(s >= 0.99 for s in g["scores"]):
            sft_rows.append(
                {
                    "input": row["input"],
                    "output": row.get("best_response", ""),
                    "metadata": meta,
                }
            )

    random.seed(args.seed)
    random.shuffle(candidates)
    heldout = candidates[: args.heldout_size]
    train_pool = candidates[args.heldout_size :]

    _write_jsonl(args.output_rl, train_pool)
    _write_jsonl(args.output_heldout, heldout)
    if sft_rows:
        _write_jsonl(args.output_sft, sft_rows)

    stats = {
        "total_groups": len(groups),
        "train_pool": len(train_pool),
        "heldout": len(heldout),
        "sft_candidates": len(sft_rows),
        "output_rl": str(args.output_rl),
        "output_heldout": str(args.output_heldout),
        "output_sft": str(args.output_sft),
    }
    stats_path = args.output_rl.with_suffix(".filter_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
