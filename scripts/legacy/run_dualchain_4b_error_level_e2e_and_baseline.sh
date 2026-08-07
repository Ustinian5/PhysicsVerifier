#!/usr/bin/env bash
# Dual-chain **error-level only** (100-sample recall set): PhysicsVerifier end-to-end vs semantic baseline,
# both using check model **qwen3-4b-instruct-2507**. Intended for `screen` / `tmux` / `nohup`.
#
# Usage (from repo root):
#   bash scripts/legacy/run_dualchain_4b_error_level_e2e_and_baseline.sh
#   PROGRESS_EVERY=10 CHECK_MODEL=qwen3-4b-instruct-2507 bash scripts/legacy/run_dualchain_4b_error_level_e2e_and_baseline.sh
#   nohup bash scripts/legacy/run_dualchain_4b_error_level_e2e_and_baseline.sh > results/_dualchain_4b_error_only_nohup.log 2>&1 &
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
if [[ ! -f "$ERROR_DATASET" ]]; then
  echo "[error] Missing error dataset: $ERROR_DATASET" >&2
  exit 2
fi

CHECK_MODEL="${CHECK_MODEL:-qwen3-4b-instruct-2507}"
STRONG_MODEL="${STRONG_MODEL:-qwen3-30b-a3b-instruct-2507}"
UNIFIED_CATALOG="${UNIFIED_CATALOG:-catalogs/legacy/unified_rule_library_v2_distilled300_20260503.json}"
MANIFEST="${EXPERIENCE_CODE_MANIFEST:-results/experience_symbolic_program_manifest_v2_unified.json}"
MODULE="${EXPERIENCE_CODE_MODULE:-symbolic.generated_experience_checks_v2_unified}"
SYMBOLIC_TOPIC_CHECK_LIMIT="${SYMBOLIC_TOPIC_CHECK_LIMIT:-32}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
MAX_PER_SAMPLE="${MAX_PER_SAMPLE:-12}"
MAX_PER_PARAGRAPH="${MAX_PER_PARAGRAPH:-2}"
BASELINE_TIMEOUT="${BASELINE_TIMEOUT:-180}"

EMPTY_AUDIT="${EMPTY_AUDIT:-$ROOT/results/_empty_symbolic_audit.json}"
[[ -s "$EMPTY_AUDIT" ]] || echo '[]' > "$EMPTY_AUDIT"

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
echo "$STAMP" > "$ROOT/results/_dualchain_4b_error_only_stamp.txt"

TAG_MAIN="${CHECK4B_ERROR_MAIN_TAG:-e2e_main_error_4b}"
TAG_BASE="${CHECK4B_ERROR_BASE_TAG:-baseline_error_4b}"
MAIN_OUT="$ROOT/results/${TAG_MAIN}_${STAMP}"
BASE_OUT="$ROOT/results/${TAG_BASE}_${STAMP}"

mkdir -p "$MAIN_OUT" "$BASE_OUT"
echo "$CHECK_MODEL" > "$MAIN_OUT/check_model.txt"
echo "$CHECK_MODEL" > "$BASE_OUT/model.txt"
cp -f "$ERROR_DATASET" "$MAIN_OUT/error_eval_dataset_100.json"
cp -f "$ERROR_DATASET" "$BASE_OUT/error_eval_dataset_100.json"

SCRIPT_START=$(date -u +%s)
echo "================================================================"
echo "[dualchain-4b-error-only] STAMP=$STAMP"
echo "  check model: $CHECK_MODEL"
echo "  strong model (unused in skip-build path): $STRONG_MODEL"
echo "  error dataset: $ERROR_DATASET"
echo "  main out:  $MAIN_OUT"
echo "  baseline:  $BASE_OUT"
echo "  progress every N samples: $PROGRESS_EVERY"
echo "================================================================"

echo ""
echo "--- [1/3] PhysicsVerifier end-to-end (error-level, 100 samples) ---"
"$PYTHON" scripts/run_verifier.py \
  --input "$MAIN_OUT/error_eval_dataset_100.json" \
  --output "$MAIN_OUT/error_verifier_results.json" \
  --symbolic-output "$MAIN_OUT/error_symbolic_audit.json" \
  --model "$CHECK_MODEL" \
  --unified-catalog "$UNIFIED_CATALOG" \
  --experience-code-manifest "$MANIFEST" \
  --experience-code-module "$MODULE" \
  --symbolic-topic-check-limit "$SYMBOLIC_TOPIC_CHECK_LIMIT" \
  --max-per-sample "$MAX_PER_SAMPLE" \
  --max-per-paragraph "$MAX_PER_PARAGRAPH" \
  --progress-interval "$PROGRESS_EVERY" \
  2>&1 | tee "$MAIN_OUT/run_e2e_error.log"

"$PYTHON" scripts/evaluate_physics_eval_sets.py \
  --dataset "$MAIN_OUT/error_eval_dataset_100.json" \
  --results "$MAIN_OUT/error_verifier_results.json" \
  --audit "$MAIN_OUT/error_symbolic_audit.json" \
  --output "$MAIN_OUT/error_metrics.json" \
  --match-mode location

echo ""
echo "--- [2/3] Semantic baseline (same 100 samples, same check model) ---"
"$PYTHON" scripts/run_llm_checker_baseline.py \
  --input "$BASE_OUT/error_eval_dataset_100.json" \
  --model "$CHECK_MODEL" \
  --out_json "$BASE_OUT/error_verifier_results.json" \
  --timeout "$BASELINE_TIMEOUT" \
  --progress-interval "$PROGRESS_EVERY" \
  --no-tqdm \
  2>&1 | tee "$BASE_OUT/run_baseline.log"

"$PYTHON" scripts/evaluate_physics_eval_sets.py \
  --dataset "$BASE_OUT/error_eval_dataset_100.json" \
  --results "$BASE_OUT/error_verifier_results.json" \
  --audit "$EMPTY_AUDIT" \
  --output "$BASE_OUT/error_metrics.json" \
  --match-mode location

echo ""
echo "--- [3/3] Summary paths ---"
echo "  E2E metrics:     $MAIN_OUT/error_metrics.json"
echo "  Baseline metrics: $BASE_OUT/error_metrics.json"

SCRIPT_END=$(date -u +%s)
WALL=$((SCRIPT_END - SCRIPT_START))
echo ""
echo "[ok] dualchain 4b error-only batch finished STAMP=$STAMP wall_s=${WALL}s ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
