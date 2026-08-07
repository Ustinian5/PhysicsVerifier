#!/usr/bin/env bash
set -euo pipefail

# Full dataset-construction pipeline for the end-to-end PhysicsVerifier experiment.
#
# Final products:
#   1) raw_splits/:
#      Raw prompt/response/label/reward/metadata samples split into
#      rule_expansion / main_test / val_ablation / smoke.
#
#   2) qa_chain/:
#      Verifier-compatible normalized question+raw-answer datasets directly
#      extracted from combined_language_only.json.
#
#   3) annotated_chain/:
#      Built and LLM-annotated evaluation datasets:
#        - error_eval_dataset_<RECALL_SIZE>.json       (error-level, with physics_error_gt)
#        - question_eval_dataset_<RECALL_SIZE>_<PRECISION_SIZE>.json
#        - question_right_only_<PRECISION_SIZE>.json
#        - error_quality_audit.json
#
# Run in screen:
#   screen -S pv_e2e_sets
#   cd /home/jinjianhan/PhysicsVerifier
#   bash scripts/legacy/build_e2e_dual_chain_datasets.sh
#
# Fast dry run:
#   MAX_ROLLOUTS=2 TEST_N=20 TEST_RIGHT_N=5 TEST_WRONG_N=15 EXPANSION_N=20 VAL_N=10 RECALL_SIZE=8 QUESTION_RECALL_SIZE=4 PRECISION_SIZE=4 bash scripts/legacy/build_e2e_dual_chain_datasets.sh
#
# Skip expensive LLM annotation while testing file plumbing:
#   SKIP_ANNOTATION=1 MAX_ROLLOUTS=2 bash scripts/legacy/build_e2e_dual_chain_datasets.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "[error] Python interpreter not found or not executable: $PYTHON" >&2
  echo "        Create/activate .venv first, or pass PYTHON=/path/to/python." >&2
  exit 2
fi

INPUT="${INPUT:-data/combined_language_only.json}"
SEED="${SEED:-20260508}"
MAX_ROLLOUTS="${MAX_ROLLOUTS:-0}"

# Raw split sizes.
SMOKE_N="${SMOKE_N:-20}"
EXPANSION_N="${EXPANSION_N:-600}"
TEST_N="${TEST_N:-200}"
TEST_RIGHT_N="${TEST_RIGHT_N:-50}"
TEST_WRONG_N="${TEST_WRONG_N:-150}"
VAL_N="${VAL_N:-80}"

# Annotated dual-chain evaluation sizes, built from the held-out main_test split.
# Defaults:
#   - error-level dataset: 100 error rows with physics_error_gt
#   - question-level dataset: 50 error rows + 50 right-only rows
RECALL_SIZE="${RECALL_SIZE:-100}"
QUESTION_RECALL_SIZE="${QUESTION_RECALL_SIZE:-50}"
PRECISION_SIZE="${PRECISION_SIZE:-50}"
MAX_RECALL_SCAN="${MAX_RECALL_SCAN:-$TEST_N}"
MIN_VALID_GT_PER_SAMPLE="${MIN_VALID_GT_PER_SAMPLE:-1}"
MAX_ERRORS="${MAX_ERRORS:-0}"
STRONG_MODEL="${STRONG_MODEL:-qwen3-30b-a3b-instruct-2507}"

SKIP_ANNOTATION="${SKIP_ANNOTATION:-0}"
RUN_TAG="${RUN_TAG:-combined_language_dual_chain_seed${SEED}_test${TEST_N}}"
OUTDIR="${OUTDIR:-data/derived/${RUN_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${RUN_TAG}}"
QA_DIR="$OUTDIR/qa_chain"
RAW_DIR="$OUTDIR/raw_splits"
ANNOTATED_DIR="$OUTDIR/annotated_chain"

mkdir -p "$OUTDIR" "$RESULTS_DIR" "$QA_DIR" "$RAW_DIR" "$ANNOTATED_DIR"

RUN_CONFIG="$OUTDIR/run_config.txt"
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root=$ROOT_DIR"
  echo "python=$PYTHON"
  echo "input=$INPUT"
  echo "seed=$SEED"
  echo "max_rollouts=$MAX_ROLLOUTS"
  echo "smoke_n=$SMOKE_N"
  echo "expansion_n=$EXPANSION_N"
  echo "test_n=$TEST_N"
  echo "test_right_n=$TEST_RIGHT_N"
  echo "test_wrong_n=$TEST_WRONG_N"
  echo "val_n=$VAL_N"
  echo "recall_size=$RECALL_SIZE"
  echo "question_recall_size=$QUESTION_RECALL_SIZE"
  echo "precision_size=$PRECISION_SIZE"
  echo "max_recall_scan=$MAX_RECALL_SCAN"
  echo "min_valid_gt_per_sample=$MIN_VALID_GT_PER_SAMPLE"
  echo "max_errors=$MAX_ERRORS"
  echo "strong_model=$STRONG_MODEL"
  echo "skip_annotation=$SKIP_ANNOTATION"
  echo "outdir=$OUTDIR"
  echo "results_dir=$RESULTS_DIR"
} | tee "$RUN_CONFIG"

echo "[1/6] Compiling dataset scripts with .venv..."
"$PYTHON" -m py_compile \
  scripts/combined_language_io.py \
  scripts/combined_language_samples.py \
  scripts/audit_combined_language_dataset.py \
  scripts/legacy/export_combined_language_eval_slices.py \
  scripts/legacy/export_combined_language_raw_splits.py \
  scripts/build_physics_eval_sets.py \
  scripts/audit_eval_set_quality.py

echo "[2/6] Auditing source combined-language file..."
"$PYTHON" scripts/audit_combined_language_dataset.py \
  --input "$INPUT" \
  --max-rollouts "$MAX_ROLLOUTS" \
  --out-json "$OUTDIR/source_audit.json" \
  --out-md "$OUTDIR/source_audit.md" \
  2>&1 | tee "$RESULTS_DIR/source_audit.log"

echo "[3/6] Exporting normalized QA-chain splits..."
"$PYTHON" scripts/legacy/export_combined_language_eval_slices.py \
  --input "$INPUT" \
  --outdir "$QA_DIR" \
  --seed "$SEED" \
  --max-rollouts "$MAX_ROLLOUTS" \
  --smoke-n "$SMOKE_N" \
  --expansion-n "$EXPANSION_N" \
  --test-n "$TEST_N" \
  --test-right-n "$TEST_RIGHT_N" \
  --test-wrong-n "$TEST_WRONG_N" \
  --val-n "$VAL_N" \
  2>&1 | tee "$RESULTS_DIR/export_qa_chain.log"

echo "[4/6] Exporting matching raw source splits..."
"$PYTHON" scripts/legacy/export_combined_language_raw_splits.py \
  --input "$INPUT" \
  --manifest "$QA_DIR/combined_language_export_manifest.json" \
  --outdir "$RAW_DIR" \
  --max-rollouts "$MAX_ROLLOUTS" \
  2>&1 | tee "$RESULTS_DIR/export_raw_splits.log"

echo "[5/6] Auditing QA split quality and leakage..."
"$PYTHON" - "$QA_DIR" <<'PY'
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

qa_dir = Path(sys.argv[1])
split_files = {
    "smoke": qa_dir / "combined_language_smoke.json",
    "rule_expansion": qa_dir / "combined_language_rule_expansion.json",
    "main_test": qa_dir / "combined_language_main_test.json",
    "val_ablation": qa_dir / "combined_language_val_ablation.json",
}

def load(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"not a JSON list: {path}")
    return [x for x in data if isinstance(x, dict)]

def p95(values: List[int]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    return float(xs[min(len(xs) - 1, round(0.95 * (len(xs) - 1)))])

all_ids: Dict[str, str] = {}
overlaps = []
per_split = {}
for split, path in split_files.items():
    rows = load(path)
    ids = []
    q_prefixes = set()
    q_lens = []
    pred_lens = []
    empty_q = empty_pred = empty_answer = 0
    for row in rows:
        sid = str(row.get("id") or "")
        ids.append(sid)
        if sid in all_ids:
            overlaps.append({"id": sid, "left": all_ids[sid], "right": split})
        elif sid:
            all_ids[sid] = split
        q = str(row.get("question") or "")
        pred = str(row.get("prediction") or "")
        ans = str(row.get("answer") or "")
        q_lens.append(len(q))
        pred_lens.append(len(pred))
        q_prefixes.add(" ".join(q.split()).casefold()[:300])
        empty_q += 0 if q.strip() else 1
        empty_pred += 0 if pred.strip() else 1
        empty_answer += 0 if ans.strip() and ans.strip() not in {"[]", "null"} else 1
    per_split[split] = {
        "count": len(rows),
        "unique_id_count": len(set(ids)),
        "duplicate_id_count": len(ids) - len(set(ids)),
        "unique_question_prefix_count": len(q_prefixes),
        "empty_question_count": empty_q,
        "empty_prediction_count": empty_pred,
        "empty_answer_count": empty_answer,
        "question_length_mean": round(statistics.mean(q_lens), 2) if q_lens else 0.0,
        "prediction_length_mean": round(statistics.mean(pred_lens), 2) if pred_lens else 0.0,
        "prediction_length_p95": round(p95(pred_lens), 2),
    }

audit = {
    "qa_dir": str(qa_dir),
    "per_split": per_split,
    "cross_split_id_overlaps": overlaps,
    "passes_basic_quality": not overlaps
    and all(x["empty_question_count"] == 0 for x in per_split.values())
    and all(x["empty_prediction_count"] == 0 for x in per_split.values())
    and all(x["duplicate_id_count"] == 0 for x in per_split.values()),
    "note": "Structural QA-chain audit only; annotated-chain error-level audit is separate.",
}
(qa_dir / "split_quality_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
if not audit["passes_basic_quality"]:
    raise SystemExit("QA split quality audit failed")
PY

if [[ "$SKIP_ANNOTATION" == "1" ]]; then
  echo "[6/6] Skipping annotated-chain construction because SKIP_ANNOTATION=1."
else
  echo "[6/6] Building annotated dual-chain evaluation datasets from held-out main_test..."
  "$PYTHON" scripts/build_physics_eval_sets.py \
    --input "$QA_DIR/combined_language_main_test.json" \
    --recall-input "$QA_DIR/combined_language_main_test.json" \
    --precision-input "$QA_DIR/combined_language_main_test.json" \
    --error-output "$ANNOTATED_DIR/error_eval_dataset_${RECALL_SIZE}.json" \
    --question-output "$ANNOTATED_DIR/question_eval_dataset_${QUESTION_RECALL_SIZE}_${PRECISION_SIZE}.json" \
    --precision-output "$ANNOTATED_DIR/question_right_only_${PRECISION_SIZE}.json" \
    --recall-size "$RECALL_SIZE" \
    --question-recall-size "$QUESTION_RECALL_SIZE" \
    --precision-size "$PRECISION_SIZE" \
    --seed "$SEED" \
    --strong-model "$STRONG_MODEL" \
    --max-errors "$MAX_ERRORS" \
    --max-recall-scan "$MAX_RECALL_SCAN" \
    --min-valid-gt-per-sample "$MIN_VALID_GT_PER_SAMPLE" \
    2>&1 | tee "$RESULTS_DIR/build_annotated_chain.log"

  "$PYTHON" scripts/audit_eval_set_quality.py \
    --recall-dataset "$ANNOTATED_DIR/error_eval_dataset_${RECALL_SIZE}.json" \
    --output "$ANNOTATED_DIR/error_quality_audit.json" \
    2>&1 | tee "$RESULTS_DIR/error_quality_audit.log"
fi

cat > "$OUTDIR/README.md" <<EOF
# E2E dual-chain dataset build

Generated by \`scripts/legacy/build_e2e_dual_chain_datasets.sh\`.

## Source and split policy

- source: \`$INPUT\`
- seed: \`$SEED\`
- max_rollouts: \`$MAX_ROLLOUTS\`
- raw splits: \`raw_splits/\`
- normalized QA-chain splits: \`qa_chain/\`
- annotated-chain datasets: \`annotated_chain/\`

The splits are mutually exclusive by source sample id:

- \`rule_expansion\`: use only for experience mining / rule-library growth.
- \`val_ablation\`: use for retrieval tuning and ablations.
- \`main_test\`: locked held-out set for final reporting.
- \`smoke\`: tiny sanity checks.

## Main outputs

- Raw rule expansion data: \`raw_splits/raw_rule_expansion.json\`
- Raw main test data: \`raw_splits/raw_main_test.json\`
- QA main test: \`qa_chain/combined_language_main_test.json\`
- QA validation: \`qa_chain/combined_language_val_ablation.json\`
- QA split audit: \`qa_chain/split_quality_audit.json\`
- Annotated error-level set: \`annotated_chain/error_eval_dataset_${RECALL_SIZE}.json\`
- Annotated question-level set: \`annotated_chain/question_eval_dataset_${QUESTION_RECALL_SIZE}_${PRECISION_SIZE}.json\`
- Annotated quality audit: \`annotated_chain/error_quality_audit.json\`

## Next steps

1. Use \`raw_splits/raw_rule_expansion.json\` / \`qa_chain/combined_language_rule_expansion.json\` for rule expansion only.
2. Tune retrieval on \`qa_chain/combined_language_val_ablation.json\`.
3. Run final verifier/evaluation on annotated-chain datasets after quality audit passes.

Example final run:

\`\`\`bash
$PYTHON scripts/run_physics_eval_pipeline.py \\
  --python "$PYTHON" \\
  --skip-build \\
  --output-dir "$RESULTS_DIR/final_eval" \\
  --recall-size "$RECALL_SIZE" \\
  --precision-size "$PRECISION_SIZE" \\
  --unified-catalog catalogs/unified_rule_library.json \\
  --check-model qwen3-30b-a3b-instruct-2507 \\
  --run-quality-audit
\`\`\`

Note: for \`run_physics_eval_pipeline.py --skip-build\`, copy or symlink annotated-chain files to the expected names under the target output dir, or pass through the direct scripts manually.
EOF

echo "[ok] Full dataset construction pipeline completed."
echo "OUTDIR=$OUTDIR"
echo "RAW_DIR=$RAW_DIR"
echo "QA_DIR=$QA_DIR"
echo "ANNOTATED_DIR=$ANNOTATED_DIR"
echo "RESULTS_DIR=$RESULTS_DIR"
