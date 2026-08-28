#!/usr/bin/env python3
"""Monitor Swift GRPO + llm_step_score metrics; fail-closed stop conditions."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List


STOP_ZERO_STD_STEPS = 3
STOP_FAIL_RATE = 0.01
STOP_SATURATION = 0.50
LENGTH_EXPLODE_RATIO = 1.8


def _tail_jsonl(path: Path, n: int = 200) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    rows: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _extract_train_rows(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]:
        if "{" not in line:
            continue
        start = line.find("{")
        try:
            obj = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (
            "loss" in obj or "reward" in obj or "grad_norm" in obj or "rewards/mean" in obj
        ):
            rows.append(obj)
    return rows


HARD_STOP = {
    "zero_std_three_steps",
    "reward_saturation",
    "reward_up_length_explosion",
    "nan_loss",
    "bad_grad_norm",
}


def evaluate(metrics_rows: List[Dict[str, Any]], train_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    reasons: List[str] = []
    warnings: List[str] = []
    zero_run = 0
    for row in metrics_rows[-STOP_ZERO_STD_STEPS:]:
        std = float(row.get("physics_llm_step_group_std_mean") or row.get("group_std") or 1.0)
        if std <= 1e-12:
            zero_run += 1
        else:
            zero_run = 0
    if len(metrics_rows) >= STOP_ZERO_STD_STEPS and zero_run >= STOP_ZERO_STD_STEPS:
        reasons.append("zero_std_three_steps")
    if metrics_rows and train_rows:
        last = metrics_rows[-1]
        fail = float(last.get("llm_step_failures") or 0.0)
        calls = float(last.get("llm_step_api_calls") or last.get("physics_reward_batch_unique_scored") or 0.0)
        if calls and (fail / calls) > STOP_FAIL_RATE:
            warnings.append("api_or_schema_fail_rate")
        if float(last.get("physics_llm_step_sat01_rate") or 0.0) > STOP_SATURATION:
            reasons.append("reward_saturation")
    lengths: List[float] = []
    rewards: List[float] = []
    for row in train_rows:
        if row.get("completion_length") is not None:
            lengths.append(float(row["completion_length"]))
        if row.get("reward") is not None:
            rewards.append(float(row["reward"]))
        loss = row.get("loss")
        if loss is not None:
            try:
                if not (float("-inf") < float(loss) < float("inf")):
                    reasons.append("nan_loss")
            except (TypeError, ValueError):
                reasons.append("nan_loss")
        gn = row.get("grad_norm")
        if gn is not None:
            try:
                gnf = float(gn)
                if not (gnf == gnf) or gnf > 1e4:
                    reasons.append("bad_grad_norm")
            except (TypeError, ValueError):
                reasons.append("bad_grad_norm")
    if len(lengths) >= 8 and len(rewards) >= 8:
        early_len = sum(lengths[:4]) / 4.0
        late_len = sum(lengths[-4:]) / 4.0
        early_r = sum(rewards[:4]) / 4.0
        late_r = sum(rewards[-4:]) / 4.0
        if late_r > early_r + 0.05 and early_len > 0 and late_len / early_len >= LENGTH_EXPLODE_RATIO:
            reasons.append("reward_up_length_explosion")
    hard = sorted(set(reasons) & HARD_STOP)
    return {
        "stop": bool(hard),
        "reasons": hard,
        "warnings": sorted(set(warnings)),
        "n_metrics": len(metrics_rows),
        "n_train": len(train_rows),
    }


def maybe_stop_training(pid_file: Path, report: Dict[str, Any]) -> None:
    if not report.get("stop"):
        return
    if not pid_file.is_file():
        return
    pid = pid_file.read_text(encoding="utf-8").strip()
    if not pid.isdigit():
        return
    Path(str(pid_file) + ".stop_reason.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    try:
        os.kill(int(pid), 15)
    except OSError:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, default=Path("/home/jinjianhan/PhysicsVerifier/logs/physics_reward_metrics.jsonl"))
    p.add_argument("--train-log", type=Path, required=True)
    p.add_argument("--pid-file", type=Path, default=None)
    p.add_argument("--poll-sec", type=float, default=30)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    while True:
        metrics = _tail_jsonl(args.metrics)
        train = _extract_train_rows(args.train_log)
        report = evaluate(metrics, train)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if args.pid_file and report.get("n_train", 0) >= 1:
            maybe_stop_training(args.pid_file, report)
        if args.once or report.get("stop"):
            return 2 if report.get("stop") else 0
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
