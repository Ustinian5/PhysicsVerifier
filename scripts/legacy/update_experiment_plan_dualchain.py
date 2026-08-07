#!/usr/bin/env python3
"""Refresh auto-generated tables inside results/dual_chain_experiment_tracking.md (stdlib only).

Markers in the doc (do not remove):
  <!-- AUTO:DUALCHAIN_BATCH_TABLES --> ... <!-- END:DUALCHAIN_BATCH_TABLES -->
  <!-- AUTO:DUALCHAIN_4B_TABLES --> ... <!-- END:DUALCHAIN_4B_TABLES -->

Usage:
  python scripts/legacy/update_experiment_plan_dualchain.py --phase batch
  STAMP_4B=... python scripts/legacy/update_experiment_plan_dualchain.py --phase fourb
  python scripts/legacy/update_experiment_plan_dualchain.py --phase all
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "results" / "dual_chain_experiment_tracking.md"
STAMP_FILE = ROOT / "results" / "_batch_baseline_ablations_stamp.txt"
STAMP_4B_FILE = ROOT / "results" / "_dualchain_check4b_stamp.txt"
UTC = datetime.timezone.utc

START_BATCH = "<!-- AUTO:DUALCHAIN_BATCH_TABLES -->"
END_BATCH = "<!-- END:DUALCHAIN_BATCH_TABLES -->"
START_4B = "<!-- AUTO:DUALCHAIN_4B_TABLES -->"
END_4B = "<!-- END:DUALCHAIN_4B_TABLES -->"


def _load_summary(p: Path) -> dict:
    if not p.is_file():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        s = d.get("summary")
        return s if isinstance(s, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _r4(x) -> str:
    if x is None:
        return "n/a"
    try:
        return str(round(float(x), 4))
    except (TypeError, ValueError):
        return str(x)


def resolve_stamp_batch() -> str:
    if STAMP_FILE.is_file():
        return STAMP_FILE.read_text(encoding="utf-8").strip()
    return ""


def resolve_stamp_4b() -> str:
    if STAMP_4B_FILE.is_file():
        return STAMP_4B_FILE.read_text(encoding="utf-8").strip()
    return ""


def build_batch_tables(stamp_batch: str) -> str:
    b = ROOT / f"results/baseline_qwen3_same_{stamp_batch}"
    r6 = ROOT / f"results/e2e_ablation_ruletop6_{stamp_batch}"
    s4 = ROOT / f"results/e2e_ablation_score4_{stamp_batch}"

    def row(name: str, d: Path) -> str:
        e = _load_summary(d / "error_metrics.json")
        q = _load_summary(d / "question_metrics.json")
        if not e and not q:
            return (
                f"| {name} "
                f"| — | — | — "
                f"| — | — | — "
                f"| 指标文件缺失 `results/.../{d.name}/` |"
            )
        return (
            f"| {name} "
            f"| {_r4(e.get('precision'))} | {_r4(e.get('recall'))} | {_r4(e.get('f1'))} "
            f"| {_r4(q.get('precision'))} | {_r4(q.get('recall'))} | {_r4(q.get('f1'))} "
            f"| 检查模型 `qwen3-30b-a3b-instruct-2507` |"
        )

    now = datetime.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "",
        f"_本段由 `scripts/legacy/update_experiment_plan_dualchain.py` 于 {now} 依据磁盘上的 `*_metrics.json` 刷新。_",
        "",
        f"批次 `STAMP={stamp_batch}`：**语义 baseline**（无 `rule` 字段）+ **消融** `unified-rule-top-n 6` + **消融** `min-diagnostic-rule-score 4.0`。错误级匹配 **location**；题目级同 §11 上文。",
        "",
        "| 运行 | 错误级 P | R | F1 | 题目级 P | R | F1 | 说明 |",
        "|------|----------|--------|-----|----------|--------|-----|------|",
        row("baseline_qwen3_same", b),
        row("e2e_ablation_ruletop6", r6),
        row("e2e_ablation_score4", s4),
        "",
    ]
    return "\n".join(lines)


def build_fourb_tables(stamp_4b: str) -> str:
    m = ROOT / f"results/e2e_main_check_4b_{stamp_4b}"
    bl = ROOT / f"results/baseline_check_4b_{stamp_4b}"

    def pack(d: Path, label: str) -> str:
        e = _load_summary(d / "error_metrics.json")
        q = _load_summary(d / "question_metrics.json")
        if not e and not q:
            return f"| {label} | — | — | 指标缺失 `{d.name}` |"
        return (
            f"| {label} "
            f"| {_r4(e.get('precision'))} / {_r4(e.get('recall'))} / {_r4(e.get('f1'))} "
            f"| {_r4(q.get('precision'))} / {_r4(q.get('recall'))} / {_r4(q.get('f1'))} "
            f"| `results/{d.name}/` |"
        )

    now = datetime.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "",
        f"_本段由 `scripts/legacy/update_experiment_plan_dualchain.py` 于 {now} 刷新。_",
        "",
        f"检查模型 **`qwen3-4b-instruct-2507`**（`STAMP={stamp_4b}`）：主流程 `e2e_main_check_4b_*` 与语义 baseline `baseline_check_4b_*`。",
        "",
        "| 运行 | 错误级 P/R/F1 | 题目级 P/R/F1 | 目录 |",
        "|------|----------------|----------------|------|",
        pack(m, "e2e_main_check_4b"),
        pack(bl, "baseline_check_4b"),
        "",
    ]
    return "\n".join(lines)


def _replace_region(text: str, start: str, end: str, body: str) -> str:
    if start not in text or end not in text:
        print(f"Missing markers {start!r} or {end!r} in {PLAN}", file=sys.stderr)
        return text
    pattern = re.compile(
        re.escape(start) + r".*?" + re.escape(end),
        flags=re.DOTALL,
    )
    new_block = start + "\n" + body.rstrip() + "\n" + end
    return pattern.sub(new_block, text, count=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("batch", "fourb", "all"), default="all")
    args = parser.parse_args()

    if not PLAN.is_file():
        print(f"Missing {PLAN}", file=sys.stderr)
        sys.exit(2)

    text = PLAN.read_text(encoding="utf-8")
    stamp_batch = resolve_stamp_batch()
    stamp_4b = resolve_stamp_4b()

    if args.phase in ("batch", "all"):
        if not stamp_batch:
            print("No STAMP in results/_batch_baseline_ablations_stamp.txt", file=sys.stderr)
            if args.phase == "batch":
                sys.exit(1)
        else:
            body = build_batch_tables(stamp_batch)
            text = _replace_region(text, START_BATCH, END_BATCH, body)

    if args.phase in ("fourb", "all"):
        if not stamp_4b:
            print("No results/_dualchain_check4b_stamp.txt yet (4B batch not started).", file=sys.stderr)
            if args.phase == "fourb":
                sys.exit(1)
        else:
            body = build_fourb_tables(stamp_4b)
            text = _replace_region(text, START_4B, END_4B, body)

    PLAN.write_text(text, encoding="utf-8")
    print(f"Updated {PLAN}")


if __name__ == "__main__":
    main()
