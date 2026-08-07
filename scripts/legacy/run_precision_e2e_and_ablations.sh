#!/usr/bin/env bash
# Triple PhysicsVerifier run on the dual-chain staged datasets (no semantic baseline):
#   1) e2e_precision_opt: no --unified-rule-top-n / --min-diagnostic-rule-score
#      -> PhysicsRuleVerifier defaults: unified_rule_top_n=6, min_diagnostic_rule_score=4.0
#   2) Ablation A: explicit --unified-rule-top-n 6 (same numeric default as run 1)
#   3) Ablation B: explicit --min-diagnostic-rule-score 4.0 (same numeric default as run 1)
# For a strict top_n ablation, add a run with e.g. --unified-rule-top-n 4.
#
# See also: scripts/legacy/run_baseline_and_precision_ablations.sh (baseline + both ablations).
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

DATASET_DIR="${DATASET_DIR:-data/derived/combined_language_dual_chain_seed20260508_test200/annotated_chain}"
ERROR_DATASET="${ERROR_DATASET:-$DATASET_DIR/error_eval_dataset_100.json}"
QUESTION_DATASET="${QUESTION_DATASET:-$DATASET_DIR/question_eval_dataset_50_50.json}"
PRECISION_DATASET="${PRECISION_DATASET:-$DATASET_DIR/question_right_only_50.json}"
UNIFIED_CATALOG="${UNIFIED_CATALOG:-catalogs/legacy/unified_rule_library_v2_distilled300_20260503.json}"
MANIFEST="${EXPERIENCE_CODE_MANIFEST:-results/experience_symbolic_program_manifest_v2_unified.json}"
MODULE="${EXPERIENCE_CODE_MODULE:-symbolic.generated_experience_checks_v2_unified}"
CHECK_MODEL="${CHECK_MODEL:-qwen3-30b-a3b-instruct-2507}"
STRONG_MODEL="${STRONG_MODEL:-qwen3-30b-a3b-instruct-2507}"

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
echo "$STAMP" > "$ROOT/results/_precision_run_stamp.txt"

RECALL_SIZE="${RECALL_SIZE:-100}"
PRECISION_SIZE="${PRECISION_SIZE:-50}"

run_one () {
  local tag="$1"
  shift
  local out="$ROOT/results/${tag}_${STAMP}"
  mkdir -p "$out"
  cp -f "$ERROR_DATASET" "$out/error_eval_dataset_${RECALL_SIZE}.json"
  cp -f "$QUESTION_DATASET" "$out/question_eval_dataset_${RECALL_SIZE}_${PRECISION_SIZE}.json"
  if [[ -f "$PRECISION_DATASET" ]]; then
    cp -f "$PRECISION_DATASET" "$out/question_right_only_${PRECISION_SIZE}.json"
  fi
  echo "[run] $tag -> $out"
  "$PYTHON" scripts/run_physics_eval_pipeline.py \
    --python "$PYTHON" \
    --output-dir "$out" \
    --recall-size "$RECALL_SIZE" \
    --precision-size "$PRECISION_SIZE" \
    --strong-model "$STRONG_MODEL" \
    --check-model "$CHECK_MODEL" \
    --unified-catalog "$UNIFIED_CATALOG" \
    --experience-code-manifest "$MANIFEST" \
    --experience-code-module "$MODULE" \
    --symbolic-topic-check-limit "${SYMBOLIC_TOPIC_CHECK_LIMIT:-32}" \
    --max-per-sample "${MAX_PER_SAMPLE:-12}" \
    --max-per-paragraph "${MAX_PER_PARAGRAPH:-2}" \
    --skip-build \
    "$@" \
    2>&1 | tee "$out/pipeline.log"
}

run_one "e2e_precision_opt"
run_one "e2e_ablation_ruletop6" --unified-rule-top-n 6
run_one "e2e_ablation_score4" --min-diagnostic-rule-score 4.0

echo "[ok] All runs finished. STAMP=$STAMP"
