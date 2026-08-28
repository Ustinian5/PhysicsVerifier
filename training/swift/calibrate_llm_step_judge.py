#!/usr/bin/env python3
"""Read-only calibration gate for the DeepSeek llm_step_score judge."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.reward_server.llm_step_judge import LLMStepJudge, LLMStepJudgeError, build_messages
from training.swift.smoke_self_judge import spearman

ANTI_HACK: List[Tuple[str, str, float]] = [
    ("empty", "", 0.25),
    ("copy_question", "__COPY__", 0.35),
    ("formula_dump", "F=ma\nE=mc^2\nV=IR\nP=IV\nK=1/2 mv^2\nU=mgh", 0.4),
    ("self_praise", "The answer is correct. I am confident. \\boxed{42}", 0.35),
    ("unit_conflict", "Let mass m=2 s. Then F=ma=2 s * 3 m/s^2 = 6 s*m/s^2. \\boxed{6 N}", 0.5),
]


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _question(row: Dict[str, Any]) -> str:
    if row.get("question"):
        return str(row["question"])
    for msg in reversed(row.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content") or "")
    return str(row.get("prompt") or "")


def _candidate_group(question: str) -> List[str]:
    q = str(question or "").strip()
    snippet = q[:320]
    return [
        "",
        q[:500],
        "The answer is correct. I am confident. \\boxed{42}",
        f"From the problem: {snippet[:160]}\nI guess without deriving. \\boxed{{0}}",
        (
            f"Physical system: {snippet[:220]}\n"
            "Apply Newton's laws or energy conservation as appropriate. "
            "Introduce the unknown x, write one governing equation, and stop before algebra. "
            "The boxed value is not fully justified. \\boxed{x}"
        ),
        (
            f"Restate the asked unknown from: {snippet[:180]}\n\n"
            "Model the bodies, forces, and constraints given in the text. "
            "Write the relevant equations, solve symbolically, check dimensions, "
            "and avoid inventing data that the problem did not give. "
            "A self-contained derivation isolates the requested quantity.\n"
            "\\boxed{result}"
        ),
    ]


CONTROL_GROUPS: List[Tuple[str, List[str]]] = [
    (
        "A mass m is at rest on a frictionless horizontal table. What is its acceleration?",
        [
            "",
            "A mass m is at rest on a frictionless horizontal table. What is its acceleration?",
            "The answer is obviously correct. \\boxed{9.8}",
            "F=ma so maybe a=g. \\boxed{g}",
            "No horizontal force acts, but I did not check friction carefully. a might be 0.",
            "The table is frictionless and the mass is at rest, so the net force is zero. "
            "Newton's second law then gives a = F/m = 0. \\boxed{0}",
        ],
    ),
    (
        "A stone is dropped from rest. After 2 s, what is its speed? Take g=10 m/s^2 and neglect air resistance.",
        [
            "",
            "The answer is correct. \\boxed{2}",
            "s=vt so v=2 m/s. \\boxed{2}",
            "Use v=v0+at with a=g but I am not sure about the sign. \\boxed{10}",
            "Dropped from rest so v0=0. Then v=gt. Numerically v=10*2=20 but units might be wrong.",
            "Take downward positive. v_0=0, a=g=10 m/s^2, t=2 s. v=v_0+at=20 m/s downward. \\boxed{20 m/s}",
        ],
    ),
    (
        "A 2 ohm resistor has 4 A through it. What is the voltage drop?",
        [
            "",
            "I copied the question. A 2 ohm resistor has 4 A through it.",
            "Power is IV so V=8. Wait maybe not. \\boxed{8}",
            "Ohm's law is V=IR. I forget which symbol is which.",
            "V=IR=2*4=8 but I did not write the unit.",
            "Ohm's law: V=IR. I=4 A, R=2 ohm, so V=8 V. \\boxed{8 V}",
        ],
    ),
    (
        "A 3 kg object moves at 4 m/s. What is its kinetic energy?",
        [
            "",
            "KE=mgh. \\boxed{12}",
            "The answer is correct. \\boxed{12}",
            "K=mv^2=3*16=48. I missed the 1/2. \\boxed{48}",
            "K=1/2 mv^2. m=3, v=4, so 1/2*3*16=24 but maybe relativistic?",
            "Non-relativistic kinetic energy K=\\frac12 mv^2=\\frac12(3)(16)=24 J. \\boxed{24 J}",
        ],
    ),
    (
        "Two charges +q and +q are a distance r apart in vacuum. What is the Coulomb force magnitude?",
        [
            "",
            "They attract. \\boxed{0}",
            "F=qE. \\boxed{q}",
            "F=kq/r^2. I dropped one charge. \\boxed{kq/r^2}",
            "Coulomb's law F=k q1 q2 / r^2. Same sign so repulsive. Magnitude k q^2/r^2.",
            "Coulomb's law in vacuum: F=\\frac{1}{4\\pi\\epsilon_0}\\frac{q^2}{r^2}, repulsive. \\boxed{\\frac{1}{4\\pi\\epsilon_0}\\frac{q^2}{r^2}}",
        ],
    ),
]


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return statistics.pstdev([float(x) for x in xs])


def evaluate_gate(report: Dict[str, Any]) -> List[str]:
    fails: List[str] = []
    if report.get("json_valid_rate", 0) < 0.99:
        fails.append("json_valid_rate < 0.99")
    if report.get("api_fail_rate", 1) > 0:
        fails.append("api_fail_rate > 0")
    if report.get("mean_group_std", 0) < 0.08:
        fails.append("mean_group_std < 0.08")
    if report.get("zero_std_rate", 1) > 0.05:
        fails.append("zero_std_rate > 0.05")
    if report.get("extreme_rate", 1) > 0.95 and report.get("mean_group_std", 0) < 0.08:
        fails.append("extreme_rate all saturated")
    if report.get("repeat_spearman", 0) < 0.8:
        fails.append("repeat_spearman < 0.8")
    if report.get("repeat_mad", 1) > 0.1:
        fails.append("repeat_mad > 0.1")
    if report.get("anti_hack_fail"):
        fails.append("anti_hack_fail: " + ",".join(report["anti_hack_fail"]))
    gold_in_prompt = report.get("gold_in_prompt")
    if gold_in_prompt:
        fails.append("gold leaked into judge prompt")
    return fails


def calibrate(judge: LLMStepJudge, groups: List[Tuple[str, List[str]]], n_repeat: int, gate_n: int = 0) -> Dict[str, Any]:
    scored: List[List[float]] = []
    repeats: List[List[float]] = []
    json_ok = 0
    api_fail = 0
    n = 0
    gold_in_prompt = False
    for question, cands in groups:
        messages = build_messages(question, cands)
        if "\\boxed{GOLD_LEAK_TOKEN}" in json.dumps(messages):
            gold_in_prompt = True
        n += 1
        print(f"[calib] group {n}/{len(groups)} chars_q={len(question)} n_cand={len(cands)}", flush=True)
        try:
            payload = judge.score_group(question, cands)
            scores = [float(x["score"]) for x in payload]
            scored.append(scores)
            json_ok += 1
        except LLMStepJudgeError:
            api_fail += 1
            continue
    for i, (question, cands) in enumerate(groups[:n_repeat], 1):
        print(f"[calib] repeat {i}/{n_repeat}", flush=True)
        try:
            a = [float(x["score"]) for x in judge.score_group(question, cands)]
            b = [float(x["score"]) for x in judge.score_group(question, cands)]
            repeats.append([*a, *b])
            json_ok += 2
            n += 2
        except LLMStepJudgeError:
            api_fail += 1
            n += 1
    stds = [_std(s) for s in scored]
    gate_n = min(gate_n or len(CONTROL_GROUPS), len(stds))
    gate_stds = stds[:gate_n]
    gate_flat = [x for row in scored[:gate_n] for x in row]
    extreme = sum(1 for x in gate_flat if x <= 1e-8 or x >= 1.0 - 1e-8) / max(len(gate_flat), 1)
    spearmans: List[float] = []
    mads: List[float] = []
    for row in repeats:
        half = len(row) // 2
        a, b = row[:half], row[half:]
        spearmans.append(spearman(a, b))
        mads.append(sum(abs(x - y) for x, y in zip(a, b)) / max(len(a), 1))
    anti_fail: List[str] = []
    if scored:
        q0, _ = groups[0]
        for name, text, cap in ANTI_HACK:
            sol = q0 if text == "__COPY__" else text
            try:
                payload = judge.score_group(q0, [sol, "A complete Newton's-law derivation concluding a=0. \\boxed{0}"])
                if payload[0]["score"] > cap:
                    anti_fail.append(name)
            except LLMStepJudgeError:
                anti_fail.append(f"{name}:api")
    mean_lat = statistics.mean(judge.latencies_ms) if judge.latencies_ms else 0.0
    calls = max(judge.calls, 1)
    return {
        "n_groups": len(groups),
        "n_gate_groups": gate_n,
        "json_valid_rate": json_ok / max(n, 1),
        "api_fail_rate": api_fail / max(n, 1),
        "mean_group_std": statistics.mean(gate_stds) if gate_stds else 0.0,
        "zero_std_rate": sum(1 for s in gate_stds if s <= 1e-12) / max(len(gate_stds), 1),
        "extreme_rate": extreme,
        "olympiad_stub_mean_std": statistics.mean(stds[gate_n:]) if len(stds) > gate_n else None,
        "all_group_stds": [round(s, 4) for s in stds],
        "repeat_spearman": statistics.mean(spearmans) if spearmans else 0.0,
        "repeat_mad": statistics.mean(mads) if mads else 1.0,
        "anti_hack_fail": anti_fail,
        "gold_in_prompt": gold_in_prompt,
        "mean_latency_ms": mean_lat,
        "api_calls": judge.calls,
        "prompt_tokens": judge.prompt_tokens,
        "completion_tokens": judge.completion_tokens,
        "eta_100_steps_hours": (800 * (mean_lat / 1000.0) / 3600.0) if mean_lat else None,
        "call_budget_100_steps": 800,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=ROOT / "data/rl/swift_prompts_max2048.jsonl")
    p.add_argument("--n-groups", type=int, default=30)
    p.add_argument("--n-repeat", type=int, default=10)
    p.add_argument("--report", type=Path, default=ROOT / "logs/llm_step_judge_calibration.json")
    p.add_argument("--dry-run", action="store_true", help="Use a fake judge (unit tests / syntax)")
    args = p.parse_args()
    _load_env(ROOT / ".env")

    rows = _load_jsonl(args.prompts)
    groups: List[Tuple[str, List[str]]] = list(CONTROL_GROUPS)
    for row in rows:
        q = _question(row)
        if not q:
            continue
        groups.append((q, _candidate_group(q)))
        if len(groups) >= args.n_groups:
            break
    if len(groups) < 3:
        print("[error] not enough prompts for calibration", file=sys.stderr)
        return 2

    if args.dry_run:
        def fake_complete(messages, extra_user=None):
            n = json.loads(messages[1]["content"])["candidates"]
            cands = []
            for i, item in enumerate(n):
                sol = item["solution"]
                score = 1.0 if not sol else (2.0 if len(sol) < 40 else (5.0 + (i % 4)))
                cands.append({"id": item["id"], "score": score, "fatal_error": False, "answer_only": not sol})
            return json.dumps({"candidates": cands})

        judge = LLMStepJudge(complete_fn=fake_complete, sleep_fn=lambda _s: None)
    else:
        judge = LLMStepJudge.from_env()

    report = calibrate(judge, groups, min(args.n_repeat, len(CONTROL_GROUPS)), gate_n=len(CONTROL_GROUPS))
    fails = evaluate_gate(report)
    report["ok"] = not fails
    report["failures"] = fails
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
