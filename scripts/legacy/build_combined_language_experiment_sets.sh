#!/usr/bin/env bash
set -euo pipefail

# One-command dataset builder for end-to-end experiments from data/combined_language_only.json.
#
# Default split sizes:
#   smoke:           20  (quick verifier sanity checks)
#   rule_expansion: 600  (experience mining / rule library growth; never used for final reporting)
#   main_test:      100  (held-out main evaluation set)
#   val_ablation:    80  (retrieval / threshold tuning and ablations)
#
# Usage in screen:
#   screen -S pv_build_sets
#   bash scripts/legacy/build_combined_language_experiment_sets.sh
#
# Useful overrides:
#   MAX_ROLLOUTS=50 TEST_N=100 EXPANSION_N=600 VAL_N=80 bash scripts/legacy/build_combined_language_experiment_sets.sh
#   MAX_ROLLOUTS=0  bash scripts/legacy/build_combined_language_experiment_sets.sh   # full file

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT="${INPUT:-data/combined_language_only.json}"
SEED="${SEED:-20260508}"
MAX_ROLLOUTS="${MAX_ROLLOUTS:-0}"
SMOKE_N="${SMOKE_N:-20}"
EXPANSION_N="${EXPANSION_N:-600}"
TEST_N="${TEST_N:-100}"
VAL_N="${VAL_N:-80}"
RUN_TAG="${RUN_TAG:-combined_language_e2e_seed${SEED}_test${TEST_N}}"
OUTDIR="${OUTDIR:-data/derived/${RUN_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${RUN_TAG}}"

mkdir -p "$OUTDIR" "$RESULTS_DIR"

RUN_CONFIG="$OUTDIR/run_config.txt"
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root=$ROOT_DIR"
  echo "input=$INPUT"
  echo "seed=$SEED"
  echo "max_rollouts=$MAX_ROLLOUTS"
  echo "smoke_n=$SMOKE_N"
  echo "expansion_n=$EXPANSION_N"
  echo "test_n=$TEST_N"
  echo "val_n=$VAL_N"
  echo "outdir=$OUTDIR"
  echo "results_dir=$RESULTS_DIR"
} | tee "$RUN_CONFIG"

echo "[1/4] Auditing source rollout file..."
python3 scripts/audit_combined_language_dataset.py \
  --input "$INPUT" \
  --max-rollouts "$MAX_ROLLOUTS" \
  --out-json "$OUTDIR/source_audit.json" \
  --out-md "$OUTDIR/source_audit.md" \
  2>&1 | tee "$RESULTS_DIR/source_audit.log"

echo "[2/4] Exporting mutually exclusive splits..."
python3 scripts/legacy/export_combined_language_eval_slices.py \
  --input "$INPUT" \
  --outdir "$OUTDIR" \
  --seed "$SEED" \
  --max-rollouts "$MAX_ROLLOUTS" \
  --smoke-n "$SMOKE_N" \
  --expansion-n "$EXPANSION_N" \
  --test-n "$TEST_N" \
  --val-n "$VAL_N" \
  2>&1 | tee "$RESULTS_DIR/export.log"

echo "[3/4] Running split quality and leakage audit..."
python3 - "$OUTDIR" <<'PY'
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

outdir = Path(sys.argv[1])
split_files = {
    "smoke": outdir / "combined_language_smoke.json",
    "rule_expansion": outdir / "combined_language_rule_expansion.json",
    "main_test": outdir / "combined_language_main_test.json",
    "val_ablation": outdir / "combined_language_val_ablation.json",
}

def load(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing split file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"split is not a JSON list: {path}")
    return [x for x in data if isinstance(x, dict)]

def p95(values: List[int]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    return float(xs[min(len(xs) - 1, round(0.95 * (len(xs) - 1)))])

rows_by_split = {name: load(path) for name, path in split_files.items()}
all_ids: Dict[str, str] = {}
overlaps = []
per_split = {}
for name, rows in rows_by_split.items():
    ids = []
    q_keys = set()
    q_lens = []
    pred_lens = []
    empty_q = empty_pred = empty_answer = 0
    for row in rows:
        sid = str(row.get("id") or "")
        ids.append(sid)
        if sid in all_ids:
            overlaps.append({"id": sid, "left": all_ids[sid], "right": name})
        elif sid:
            all_ids[sid] = name
        q = str(row.get("question") or "")
        pred = str(row.get("prediction") or "")
        answer = str(row.get("answer") or "")
        q_lens.append(len(q))
        pred_lens.append(len(pred))
        if not q.strip():
            empty_q += 1
        if not pred.strip():
            empty_pred += 1
        if not answer.strip() or answer.strip() in {"[]", "null"}:
            empty_answer += 1
        q_keys.add(" ".join(q.split()).casefold()[:300])
    per_split[name] = {
        "count": len(rows),
        "unique_id_count": len(set(ids)),
        "duplicate_id_count": len(ids) - len(set(ids)),
        "unique_question_prefix_count": len(q_keys),
        "empty_question_count": empty_q,
        "empty_prediction_count": empty_pred,
        "empty_answer_count": empty_answer,
        "question_length_mean": round(statistics.mean(q_lens), 2) if q_lens else 0.0,
        "prediction_length_mean": round(statistics.mean(pred_lens), 2) if pred_lens else 0.0,
        "prediction_length_p95": round(p95(pred_lens), 2),
    }

audit = {
    "outdir": str(outdir),
    "split_files": {k: str(v) for k, v in split_files.items()},
    "per_split": per_split,
    "cross_split_id_overlaps": overlaps,
    "passes_basic_quality": not overlaps
    and all(x["empty_question_count"] == 0 for x in per_split.values())
    and all(x["empty_prediction_count"] == 0 for x in per_split.values())
    and all(x["duplicate_id_count"] == 0 for x in per_split.values()),
    "note": (
        "This structural audit checks extraction integrity and split leakage. "
        "It does not replace rubric-level physics_error_gt audit for error-level metrics."
    ),
}
(outdir / "split_quality_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
if not audit["passes_basic_quality"]:
    raise SystemExit("split quality audit failed")
PY

echo "[4/4] Writing README..."
cat > "$OUTDIR/README.md" <<EOF
# Combined-language E2E dataset splits

- input: \`$INPUT\`
- seed: \`$SEED\`
- max_rollouts: \`$MAX_ROLLOUTS\`
- rule expansion: \`combined_language_rule_expansion.json\` ($EXPANSION_N requested)
- main test: \`combined_language_main_test.json\` ($TEST_N requested)
- validation / ablation: \`combined_language_val_ablation.json\` ($VAL_N requested)
- smoke: \`combined_language_smoke.json\` ($SMOKE_N requested)

Important:

- Splits are sampled from one shuffled index list and are mutually exclusive by source sample id.
- Use \`rule_expansion\` only for experience mining / rule-library growth.
- Use \`val_ablation\` for retrieval tuning, threshold sweeps, and ablations.
- Keep \`main_test\` locked for final reporting.
- These files contain question + raw model answer + reference answer labels. They do not contain \`physics_error_gt\`; use rubric-derived datasets for error-level P/R/F1.

Next recommended commands:

\`\`\`bash
# Smoke verifier run
python3 scripts/run_verifier.py \\
  --unified-catalog catalogs/unified_rule_library.json \\
  --input "$OUTDIR/combined_language_smoke.json" \\
  --output "$RESULTS_DIR/smoke_verify.json" \\
  --symbolic-output "$RESULTS_DIR/smoke_symbolic_audit.json" \\
  --full-output "$RESULTS_DIR/smoke_full.json"

# Retrieval tuning on validation set
python3 scripts/analyze_rule_matching.py \\
  --catalog catalogs/unified_rule_library.json \\
  --input "$OUTDIR/combined_language_val_ablation.json" \\
  --outdir "$RESULTS_DIR/matching_val" \\
  --ab-compare-topic-text \\
  --sweep-tuning
\`\`\`
EOF

echo "[ok] Dataset build complete."
echo "OUTDIR=$OUTDIR"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "Manifest: $OUTDIR/combined_language_export_manifest.json"
echo "Split audit: $OUTDIR/split_quality_audit.json"
