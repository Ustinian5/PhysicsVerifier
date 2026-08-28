#!/usr/bin/env python3
"""Analyze step 0/20/40/60/80/100 onset: heldout exact-acc + official HiPhO-TO MNS."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def bootstrap_mean_ci(values: Sequence[float], n: int = 1000, seed: int = 0) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[min(len(means) - 1, int(0.975 * n))]
    return sum(values) / len(values), lo, hi


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def earliest_onset(
    checkpoints: Sequence[Dict[str, Any]],
    *,
    grader_noise: float,
) -> Dict[str, Any]:
    """Earliest ckpt with heldout +2 correct (no next drop) AND HiPhO MNS > step0 + noise.

    If only step100 improves, both heldout and HiPhO-TO must be positive.
    """
    if not checkpoints:
        return {"onset_step": None, "conclusion": "100 步内未观察到稳定、可外部验证的提升", "reason": "no_checkpoints"}
    by_step = {int(r["step"]): r for r in checkpoints}
    steps = sorted(by_step)
    if 0 not in by_step:
        return {"onset_step": None, "conclusion": "100 步内未观察到稳定、可外部验证的提升", "reason": "missing_step0"}
    base = by_step[0]
    base_heldout = int(base.get("heldout_correct") or 0)
    base_mns = float(base.get("hipho_mns") or 0.0)
    candidates: List[int] = [s for s in steps if s > 0]
    for i, step in enumerate(candidates):
        cur = by_step[step]
        heldout = int(cur.get("heldout_correct") or 0)
        mns = float(cur.get("hipho_mns") or 0.0)
        heldout_gain = heldout - base_heldout
        mns_gain = mns - base_mns
        next_ok = True
        if i + 1 < len(candidates):
            nxt = by_step[candidates[i + 1]]
            next_ok = int(nxt.get("heldout_correct") or 0) >= heldout
        else:
            # step 100 first-time improvement requires both metrics positive
            if heldout_gain < 2 or mns_gain <= grader_noise:
                continue
            return {
                "onset_step": step,
                "heldout_gain": heldout_gain,
                "mns_gain": mns_gain,
                "conclusion": f"最早在 step {step} 同时出现 heldout 与 HiPhO-TO 正向提升",
            }
        if heldout_gain >= 2 and next_ok and mns_gain > grader_noise:
            return {
                "onset_step": step,
                "heldout_gain": heldout_gain,
                "mns_gain": mns_gain,
                "conclusion": f"最早在 step {step} 看到稳定、可外部验证的提升",
            }
    return {
        "onset_step": None,
        "conclusion": "100 步内未观察到稳定、可外部验证的提升",
        "reason": "gates_not_met",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="JSON list of per-checkpoint scores")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--grader-noise", type=float, default=0.01)
    args = p.parse_args()
    payload = _load(args.input)
    rows = payload.get("checkpoints") or payload
    if isinstance(rows, dict):
        rows = list(rows.values())
    onset = earliest_onset(rows, grader_noise=args.grader_noise)
    # paired deltas vs step0
    by_step = {int(r["step"]): r for r in rows}
    base = by_step.get(0, {})
    deltas = []
    for step, row in sorted(by_step.items()):
        if step == 0:
            continue
        heldout_delta = int(row.get("heldout_correct") or 0) - int(base.get("heldout_correct") or 0)
        mns_delta = float(row.get("hipho_mns") or 0.0) - float(base.get("hipho_mns") or 0.0)
        deltas.append({"step": step, "heldout_correct_delta": heldout_delta, "hipho_mns_delta": mns_delta})
    out = {
        "onset": onset,
        "checkpoints": rows,
        "deltas_vs_step0": deltas,
        "grader_noise": args.grader_noise,
        "note": "Train reward increase is not treated as success; only heldout + HiPhO-TO count.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(onset, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
