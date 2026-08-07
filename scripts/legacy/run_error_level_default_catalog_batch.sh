#!/usr/bin/env bash
# Sequential error-level batch (default catalog, API):
#   [1] 235B end-to-end main (rules + semantic + symbolic)
#   [2] 30B ablation: no programmatic symbolic checks
#
# Usage:
#   nohup bash scripts/legacy/run_error_level_default_catalog_batch.sh > results/_default_catalog_batch_nohup.log 2>&1 &
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
export STAMP
echo "$STAMP" > "$ROOT/results/_default_catalog_batch_stamp.txt"

echo "[batch] STAMP=$STAMP starting 235B main..."
RUN_235B_ONLY=1 bash "$ROOT/scripts/legacy/run_error_level_30b_no_sym_and_235b_e2e.sh"

echo "[batch] STAMP=$STAMP starting 30B no-symbolic ablation..."
RUN_30B_NO_SYM_ONLY=1 bash "$ROOT/scripts/legacy/run_error_level_30b_no_sym_and_235b_e2e.sh"

SUMMARY="$ROOT/results/default_catalog_error_batch_${STAMP}.md"
export ROOT STAMP SUMMARY
"$ROOT/.venv/bin/python" <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
stamp = os.environ["STAMP"]
summary_path = Path(os.environ["SUMMARY"])

rows = [
    ("30B 主流程（默认目录，STAMP=20260519_145935）", root / "results/e2e_no_metadata_enhance_30b_error_20260519_145935"),
    ("235B 主流程（默认目录）", root / f"results/e2e_main_235b_error_{stamp}"),
    ("30B 无符号消融（默认目录）", root / f"results/e2e_no_symbolic_30b_error_{stamp}"),
]

lines = [
    "# 错误级实验汇总（默认规则库，无 LLM 元数据增强）",
    "",
    f"**STAMP:** `{stamp}`",
    "**规则库:** `catalogs/legacy/unified_rule_library_v2_distilled300_20260503.json`",
    "**符号预算:** 40 | **top_n:** 6 | **score≥4.0**",
    "",
    "| 配置 | Recall | Precision | F1 | 命中 GT | 预测诊断 | 样本触发率 |",
    "|------|--------|-----------|-----|---------|----------|------------|",
]
for label, d in rows:
    p = d / "error_metrics.json"
    if not p.exists():
        lines.append(f"| {label} | — | — | — | — | — | — |")
        continue
    s = json.loads(p.read_text())["summary"]
    vr = json.loads((d / "error_verifier_results.json").read_text())
    nd = sum(len(x.get("diagnostics") or []) for x in vr)
    lines.append(
        f"| {label} | {s['recall']:.3f} | {s['precision']:.3f} | {s['f1']:.3f} | "
        f"{s['matched_gt_errors']}/{s['total_gt_errors']} | {nd} | {s['sample_trigger_ratio']:.2f} |"
    )
lines += ["", "## 对比说明", ""]
main = json.loads((rows[0][1] / "error_metrics.json").read_text())["summary"]
nosym = json.loads((rows[2][1] / "error_metrics.json").read_text())["summary"]
b235 = json.loads((rows[1][1] / "error_metrics.json").read_text())["summary"]
lines.append(
    f"- **30B 无符号 vs 30B 主流程（+符号）**：Recall {nosym['recall']:.3f} vs {main['recall']:.3f} "
    f"（Δ {(nosym['recall']-main['recall'])*100:+.2f} pp），关闭符号后命中 {nosym['matched_gt_errors']} vs {main['matched_gt_errors']} 条。"
)
lines.append(
    f"- **235B vs 30B 主流程**：Recall {b235['recall']:.3f} vs {main['recall']:.3f}；235B 预测诊断仅 {sum(len(x.get('diagnostics') or []) for x in json.loads((rows[1][1]/'error_verifier_results.json').read_text()))} 条，输出更保守。"
)
lines += ["", "## 目录", ""]
for label, d in rows:
    lines.append(f"- **{label}:** `{d.relative_to(root)}`")
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[ok] wrote {summary_path}")
PY

echo "[ok] batch complete STAMP=$STAMP summary=$SUMMARY"
