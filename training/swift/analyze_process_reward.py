#!/usr/bin/env python3
"""Phase 5: compare baseline vs SFT vs process-RL heldout/HiPhO acc and reward stats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _acc(summary: Dict[str, Any], label: str, bench: str) -> Optional[float]:
    for row in summary.get("entries") or []:
        if row.get("label") == label:
            val = row.get(f"{bench}_answer_acc")
            return float(val) if val is not None else None
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--summary",
        type=Path,
        default=Path("/home/jinjianhan/PhysicsVerifier/results/hipho_baseline_matrix_8b/summary_all.json"),
    )
    p.add_argument(
        "--logging-jsonl",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--recap-json",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--smoke-json",
        type=Path,
        default=Path("/slow_share/jinjianhan/ckpt/qwen3-8b-physics-swift/self_judge_smoke.json"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/home/jinjianhan/PhysicsVerifier/results/hipho_baseline_matrix_8b/process_reward_effectiveness.json"),
    )
    args = p.parse_args()

    summary = _load(args.summary) or {"entries": [], "comparisons_vs_base": []}
    report: Dict[str, Any] = {"summary": summary, "verdict": []}

    def delta(label: str, bench: str) -> Optional[float]:
        cur = _acc(summary, label, bench)
        base = _acc(summary, "base_8b", bench)
        if cur is None or base is None:
            return None
        return cur - base

    sft_h = delta("sft_8b", "heldout")
    sft_p = delta("sft_8b", "hipho")
    rl_labels = [e["label"] for e in summary.get("entries") or [] if str(e.get("label", "")).startswith("swift_procrl")]
    rl_h = max((delta(x, "heldout") or -1.0) for x in rl_labels) if rl_labels else None
    rl_p = max((delta(x, "hipho") or -1.0) for x in rl_labels) if rl_labels else None
    report["sft_vs_base"] = {"heldout": sft_h, "hipho": sft_p}
    report["best_procrl_vs_base"] = {"heldout": rl_h, "hipho": rl_p, "labels": rl_labels}

    smoke = _load(args.smoke_json)
    if smoke:
        report["self_judge_smoke"] = smoke
    recap = _load(args.recap_json) if args.recap_json else None
    if recap:
        report["recap_30b"] = {k: recap[k] for k in recap if not str(k).startswith("scores")}

    if args.logging_jsonl and args.logging_jsonl.is_file():
        rewards: List[float] = []
        stds: List[float] = []
        for line in args.logging_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if "reward" in obj:
                rewards.append(float(obj["reward"]))
                stds.append(float(obj.get("reward_std") or 0.0))
        report["rl_reward"] = {
            "n": len(rewards),
            "mean": sum(rewards) / max(len(rewards), 1),
            "std_mean": sum(stds) / max(len(stds), 1),
        }

    if sft_h is not None and sft_h > 0:
        report["verdict"].append("sft_beats_base_heldout")
    if rl_h is not None and sft_h is not None and rl_h > 0:
        report["verdict"].append("procrl_beats_base_heldout")
        if rl_h > sft_h:
            report["verdict"].append("procrl_beats_sft_heldout")
            report["process_reward_effective"] = True
        else:
            report["process_reward_effective"] = False
    else:
        report["process_reward_effective"] = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
