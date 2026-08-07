#!/usr/bin/env python3
"""Emit Markdown summary tables for dual-chain experiment dirs (stdlib only).

Usage:
  STAMP_BATCH=20260510_044549 python scripts/legacy/emit_dual_chain_results_md.py
  STAMP_BATCH=... STAMP_4B=... python scripts/legacy/emit_dual_chain_results_md.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_summary(metrics_path: Path) -> dict:
    if not metrics_path.is_file():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        return data.get("summary") if isinstance(data.get("summary"), dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def fmt_row_err(label: str, d: Path) -> str:
    s = load_summary(d / "error_metrics.json")
    if not s:
        return f"| {label} | (missing error_metrics.json) | | |"
    return f"| {label} | {s.get('precision', 'n/a')} | {s.get('recall', 'n/a')} | {s.get('f1', 'n/a')} |"


def fmt_row_q(label: str, d: Path) -> str:
    s = load_summary(d / "question_metrics.json")
    if not s:
        return f"| {label} | (missing question_metrics.json) | | |"
    return f"| {label} | {s.get('precision', 'n/a')} | {s.get('recall', 'n/a')} | {s.get('f1', 'n/a')} |"


def triple_prf(s: dict) -> str:
    if not s:
        return "n/a"
    p, r, f = s.get("precision"), s.get("recall"), s.get("f1")
    return f"{p} / {r} / {f}"


def main() -> None:
    stamp_batch = os.environ.get("STAMP_BATCH", "").strip()
    if not stamp_batch:
        bf = ROOT / "results" / "_batch_baseline_ablations_stamp.txt"
        if bf.is_file():
            stamp_batch = bf.read_text(encoding="utf-8").strip()
    if not stamp_batch:
        print("Set STAMP_BATCH or ensure results/_batch_baseline_ablations_stamp.txt exists", file=sys.stderr)
        sys.exit(1)

    stamp_4b = os.environ.get("STAMP_4B", "").strip()

    print("")
    print(f"### Auto-generated ({stamp_batch}) — error-level location")
    print("")
    print("| 运行 | Precision | Recall | F1 |")
    print("|------|-----------|--------|-----|")
    print(fmt_row_err("baseline_qwen3_same", ROOT / f"results/baseline_qwen3_same_{stamp_batch}"))
    print(fmt_row_err("e2e_ablation_ruletop6", ROOT / f"results/e2e_ablation_ruletop6_{stamp_batch}"))
    print(fmt_row_err("e2e_ablation_score4", ROOT / f"results/e2e_ablation_score4_{stamp_batch}"))
    print("")
    print(f"### Auto-generated ({stamp_batch}) — question-level")
    print("")
    print("| 运行 | Precision | Recall | F1 |")
    print("|------|-----------|--------|-----|")
    print(fmt_row_q("baseline_qwen3_same", ROOT / f"results/baseline_qwen3_same_{stamp_batch}"))
    print(fmt_row_q("e2e_ablation_ruletop6", ROOT / f"results/e2e_ablation_ruletop6_{stamp_batch}"))
    print(fmt_row_q("e2e_ablation_score4", ROOT / f"results/e2e_ablation_score4_{stamp_batch}"))

    if stamp_4b:
        d1 = ROOT / f"results/e2e_main_check_4b_{stamp_4b}"
        d2 = ROOT / f"results/baseline_check_4b_{stamp_4b}"
        s1e = load_summary(d1 / "error_metrics.json")
        s1q = load_summary(d1 / "question_metrics.json")
        s2e = load_summary(d2 / "error_metrics.json")
        s2q = load_summary(d2 / "question_metrics.json")
        print("")
        print(f"### Auto-generated — check model qwen3-4b-instruct-2507 (STAMP={stamp_4b})")
        print("")
        print("| 运行 | 错误级 P/R/F1 | 题目级 P/R/F1 |")
        print("|------|----------------|----------------|")
        print(f"| e2e_main_check_4b | {triple_prf(s1e)} | {triple_prf(s1q)} |")
        print(f"| baseline_check_4b | {triple_prf(s2e)} | {triple_prf(s2q)} |")


if __name__ == "__main__":
    main()
