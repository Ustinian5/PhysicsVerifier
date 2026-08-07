#!/usr/bin/env bash
# Sequential batch (recommended for API stability):
#   1) Semantic baseline (same model as verifier) with updated prompt (no rule field)
#   2) Ablation: unified v2 rule pool width = 6
#   3) Ablation: min diagnostic rule score = 4.0
#
# Usage:
#   nohup bash scripts/legacy/run_baseline_and_precision_ablations.sh > results/_batch_baseline_ablations_nohup.log 2>&1 &
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "[error] PYTHON not executable: $PYTHON" >&2
  exit 2
fi

DATASET_DIR="${DATASET_DIR:-data/derived/combined_language_dual_chain_seed20260508_test200/annotated_chain}"
ERROR_DATASET="${ERROR_DATASET:-$DATASET_DIR/error_eval_dataset_100.json}"
QUESTION_DATASET="${QUESTION_DATASET:-$DATASET_DIR/question_eval_dataset_50_50.json}"
PRECISION_DATASET="${PRECISION_DATASET:-$DATASET_DIR/question_right_only_50.json}"
UNIFIED_CATALOG="${UNIFIED_CATALOG:-catalogs/legacy/unified_rule_library_v2_distilled300_20260503.json}"
MANIFEST="${EXPERIENCE_CODE_MANIFEST:-results/experience_symbolic_program_manifest_v2_unified.json}"
MODULE="${EXPERIENCE_CODE_MODULE:-symbolic.generated_experience_checks_v2_unified}"
CHECK_MODEL="${CHECK_MODEL:-qwen3-30b-a3b-instruct-2507}"
STRONG_MODEL="${STRONG_MODEL:-qwen3-30b-a3b-instruct-2507}"
SYMBOLIC_TOPIC_CHECK_LIMIT="${SYMBOLIC_TOPIC_CHECK_LIMIT:-32}"

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
echo "$STAMP" > "$ROOT/results/_batch_baseline_ablations_stamp.txt"

EMPTY_AUDIT="${EMPTY_AUDIT:-$ROOT/results/_empty_symbolic_audit.json}"
echo '[]' > "$EMPTY_AUDIT"

RECALL_SIZE="${RECALL_SIZE:-100}"
PRECISION_SIZE="${PRECISION_SIZE:-50}"

echo "[batch] STAMP=$STAMP start $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- 1) Baseline (semantic-only, same model) ---
BASE_OUT="$ROOT/results/baseline_qwen3_same_${STAMP}"
mkdir -p "$BASE_OUT"
echo "$CHECK_MODEL" > "$BASE_OUT/model.txt"
{
  echo "=== Baseline same (rule-free diagnostics) start ==="
  "$PYTHON" scripts/run_llm_checker_baseline.py \
    --input "$ERROR_DATASET" --model "$CHECK_MODEL" \
    --out_json "$BASE_OUT/error_verifier_results.json"
  "$PYTHON" scripts/run_llm_checker_baseline.py \
    --input "$QUESTION_DATASET" --model "$CHECK_MODEL" \
    --out_json "$BASE_OUT/question_verifier_results.json"
  "$PYTHON" scripts/evaluate_physics_eval_sets.py \
    --dataset "$ERROR_DATASET" \
    --results "$BASE_OUT/error_verifier_results.json" \
    --audit "$EMPTY_AUDIT" \
    --output "$BASE_OUT/error_metrics.json" \
    --match-mode location
  "$PYTHON" scripts/evaluate_question_level_sets.py \
    --dataset "$QUESTION_DATASET" \
    --results "$BASE_OUT/question_verifier_results.json" \
    --audit "$EMPTY_AUDIT" \
    --output "$BASE_OUT/question_metrics.json"
  echo "=== Baseline same done ==="
} | tee "$BASE_OUT/run.log"

# --- 2) Ablation rule top-n = 6 ---
AB6_OUT="$ROOT/results/e2e_ablation_ruletop6_${STAMP}"
mkdir -p "$AB6_OUT"
cp -f "$ERROR_DATASET" "$AB6_OUT/error_eval_dataset_${RECALL_SIZE}.json"
cp -f "$QUESTION_DATASET" "$AB6_OUT/question_eval_dataset_${RECALL_SIZE}_${PRECISION_SIZE}.json"
if [[ -f "$PRECISION_DATASET" ]]; then
  cp -f "$PRECISION_DATASET" "$AB6_OUT/question_right_only_${PRECISION_SIZE}.json"
fi
echo "[run] e2e_ablation_ruletop6 -> $AB6_OUT"
"$PYTHON" scripts/run_physics_eval_pipeline.py \
  --python "$PYTHON" \
  --output-dir "$AB6_OUT" \
  --recall-size "$RECALL_SIZE" \
  --precision-size "$PRECISION_SIZE" \
  --strong-model "$STRONG_MODEL" \
  --check-model "$CHECK_MODEL" \
  --unified-catalog "$UNIFIED_CATALOG" \
  --experience-code-manifest "$MANIFEST" \
  --experience-code-module "$MODULE" \
  --symbolic-topic-check-limit "$SYMBOLIC_TOPIC_CHECK_LIMIT" \
  --max-per-sample "${MAX_PER_SAMPLE:-12}" \
  --max-per-paragraph "${MAX_PER_PARAGRAPH:-2}" \
  --skip-build \
  --unified-rule-top-n 6 \
  2>&1 | tee "$AB6_OUT/pipeline.log"

# --- 3) Ablation min diagnostic score = 4.0 ---
AB4_OUT="$ROOT/results/e2e_ablation_score4_${STAMP}"
mkdir -p "$AB4_OUT"
cp -f "$ERROR_DATASET" "$AB4_OUT/error_eval_dataset_${RECALL_SIZE}.json"
cp -f "$QUESTION_DATASET" "$AB4_OUT/question_eval_dataset_${RECALL_SIZE}_${PRECISION_SIZE}.json"
if [[ -f "$PRECISION_DATASET" ]]; then
  cp -f "$PRECISION_DATASET" "$AB4_OUT/question_right_only_${PRECISION_SIZE}.json"
fi
echo "[run] e2e_ablation_score4 -> $AB4_OUT"
"$PYTHON" scripts/run_physics_eval_pipeline.py \
  --python "$PYTHON" \
  --output-dir "$AB4_OUT" \
  --recall-size "$RECALL_SIZE" \
  --precision-size "$PRECISION_SIZE" \
  --strong-model "$STRONG_MODEL" \
  --check-model "$CHECK_MODEL" \
  --unified-catalog "$UNIFIED_CATALOG" \
  --experience-code-manifest "$MANIFEST" \
  --experience-code-module "$MODULE" \
  --symbolic-topic-check-limit "$SYMBOLIC_TOPIC_CHECK_LIMIT" \
  --max-per-sample "${MAX_PER_SAMPLE:-12}" \
  --max-per-paragraph "${MAX_PER_PARAGRAPH:-2}" \
  --skip-build \
  --min-diagnostic-rule-score 4.0 \
  2>&1 | tee "$AB4_OUT/pipeline.log"

echo "[ok] Batch finished STAMP=$STAMP $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  baseline: $BASE_OUT"
echo "  ablation ruletop6: $AB6_OUT"
echo "  ablation score4: $AB4_OUT"

if [[ -x "$PYTHON" ]]; then
  {
    echo ""
    echo "---"
    echo "<!-- auto tables batch-only $(date -u +%Y-%m-%dT%H:%M:%SZ) STAMP=$STAMP -->"
    STAMP_BATCH="$STAMP" "$PYTHON" "$ROOT/scripts/legacy/emit_dual_chain_results_md.py"
  } >> "$ROOT/results/dual_chain_experiment_tracking.md" || true
fi
