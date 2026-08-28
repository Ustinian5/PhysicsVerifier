#!/usr/bin/env python3
"""Watch GRPO logs vs 30B process-score recap and heldout acc; flag reward hacking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _step_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "log_history" in obj:
            continue
        if "reward" in obj:
            rows.append(obj)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logging-jsonl", type=Path, required=True)
    p.add_argument("--heldout-acc-json", type=Path, default=None)
    p.add_argument("--recap-json", type=Path, default=None, help="optional 30B smoke_self_judge.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-std", type=float, default=0.12)
    args = p.parse_args()

    steps = _step_rows(args.logging_jsonl)
    rewards = [float(s["reward"]) for s in steps]
    stds = [float(s.get("reward_std") or 0.0) for s in steps]
    clipped = [float(s.get("completions/clipped_ratio") or 0.0) for s in steps]
    report: Dict[str, Any] = {
        "n_steps": len(steps),
        "reward_mean": sum(rewards) / max(len(rewards), 1),
        "reward_std_mean": sum(stds) / max(len(stds), 1),
        "clipped_ratio_mean": sum(clipped) / max(len(clipped), 1),
        "flags": [],
    }
    if stds and (sum(stds) / len(stds)) < args.min_std:
        report["flags"].append("low_ingroup_std")
    if clipped and (sum(clipped) / len(clipped)) > 0.5:
        report["flags"].append("high_clip_ratio")
    if args.heldout_acc_json and args.heldout_acc_json.is_file():
        acc = json.loads(args.heldout_acc_json.read_text(encoding="utf-8"))
        report["heldout_answer_acc"] = acc.get("answer_acc")
    if args.recap_json and args.recap_json.is_file():
        recap = json.loads(args.recap_json.read_text(encoding="utf-8"))
        report["recap_30b_mean"] = recap.get("mean_b") or recap.get("mean_a")
        report["spearman_vs_30b"] = recap.get("spearman")
        if recap.get("spearman") is not None and recap["spearman"] < 0.2:
            report["flags"].append("self_judge_decoupled_from_30b")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
