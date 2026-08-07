#!/usr/bin/env python3
"""Aggregate metrics from parallel four-exp runs (three baselines + no-symbolic ablation)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    s = data.get("summary")
    return s if isinstance(s, dict) else None


def _fmt_prf(s: Optional[Dict[str, Any]]) -> str:
    if not s:
        return "—"
    p = s.get("precision")
    r = s.get("recall")
    f = s.get("f1")
    if p is None or r is None or f is None:
        return "—"
    return f"{float(p):.3f} / {float(r):.3f} / {float(f):.3f}"


def _tn(s: Optional[Dict[str, Any]]) -> str:
    if not s:
        return "—"
    tn = s.get("tn")
    return str(int(tn)) if tn is not None else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True, help="Timestamp suffix used by run_four_qwen_*.sh")
    ap.add_argument("--write-md", type=str, default="", help="Optional markdown path to write summary table")
    ap.add_argument("--root", type=str, default=".")
    args = ap.parse_args()
    root = Path(args.root)

    rows: List[Tuple[str, str, Optional[Path], Optional[Path]]] = [
        ("B1", "基线：同模型 qwen3-30b-a3b-instruct-2507", root / f"results/baseline_qwen3_same_{args.stamp}", None),
        ("B2", "基线：qwen3-next-80b-a3b-instruct", root / f"results/baseline_qwen3_next80_{args.stamp}", None),
        ("B3", "基线：qwen3-235b-a22b-instruct-2507", root / f"results/baseline_qwen3_mo235_{args.stamp}", None),
        ("B4", "消融：关闭符号核查（语义+规则，无经验代码）", root / f"results/e2e_no_symbolic_ablation_{args.stamp}", None),
    ]

    ref_v4_e = root / "results/e2e_exp_sym_v4_full/error_metrics.json"
    ref_v4_q = root / "results/e2e_exp_sym_v4_full/question_metrics.json"

    lines: List[str] = []
    lines.append("### 四实验汇总（与本脚本同批次 STAMP）")
    lines.append("")
    lines.append("| 实验 | 说明 | 错误级 P/R/F1 | 题目级 P/R/F1 | TN | 目录 |")
    lines.append("|------|------|----------------|----------------|----|------|")

    for tag, desc, bdir, _ in rows:
        em = bdir / "error_metrics.json" if bdir else None
        qm = bdir / "question_metrics.json" if bdir else None
        se = _load_summary(em) if em else None
        sq = _load_summary(qm) if qm else None
        err = _fmt_prf(se)
        qu = _fmt_prf(sq)
        tn_s = _tn(sq)
        dstr = str(bdir.relative_to(root)) if bdir and bdir.exists() else str(bdir)
        lines.append(f"| {tag} | {desc} | {err} | {qu} | {tn_s} | `{dstr}` |")

    if ref_v4_e.is_file() and ref_v4_q.is_file():
        se = _load_summary(ref_v4_e)
        sq = _load_summary(ref_v4_q)
        lines.append(
            f"| 参考 | 主实验 v4（全量经验代码+符号，同测评集） | {_fmt_prf(se)} | {_fmt_prf(sq)} | {_tn(sq)} | `results/e2e_exp_sym_v4_full` |"
        )

    text = "\n".join(lines) + "\n"
    print(text)
    if args.write_md:
        out = Path(args.write_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
