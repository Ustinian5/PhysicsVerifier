#!/usr/bin/env bash
set -euo pipefail

# End-to-end PhysicsVerifier evaluation with the refactored symbolic check
# pipeline (deterministic experience-code only, no primitive+spec / agentic
# LLM path). Symbolic verification is on by default.
#
# Run inside screen, e.g.:
#
#   screen -S pv_e2e_exp_sym
#   cd /home/jinjianhan/PhysicsVerifier
#   bash scripts/legacy/run_e2e_with_experience_symbolic.sh
#
# Override any of the following with environment variables before invoking:
#
#   PYTHON                          path to interpreter (default: ./.venv/bin/python)
#   DATASET_DIR                     directory containing annotated_chain/* files
#   ERROR_DATASET                   error-level dataset path
#   QUESTION_DATASET                question-level dataset path
#   PRECISION_DATASET               question-level right-only dataset path
#   UNIFIED_CATALOG                 unified rule catalog JSON
#   EXPERIENCE_CODE_MANIFEST        manifest of generated experience-code checks
#   EXPERIENCE_CODE_MODULE          Python module path with the generated checks
#   SYMBOLIC_TOPIC_CHECK_LIMIT      max bottom-up checks per retrieved topic
#   CHECK_MODEL                     LLM used by the verifier
#   STRONG_MODEL                    LLM used for ground-truth annotation (data build only)
#   MAX_PER_SAMPLE / MAX_PER_PARAGRAPH  diagnostic precision caps
#   PRECISION_MODE                  strict|balanced|score_only
#   RUN_TAG                         output sub-directory tag
#   SKIP_BUILD                      1 to skip dataset (re)build
#   ENABLE_QUALITY_AUDIT            1 to run quality audit during build
#   NO_SYMBOLIC_CHECK               1 to pass --no-symbolic-check (semantic-only ablation)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "[error] python interpreter not found or not executable: $PYTHON" >&2
  echo "        create/activate .venv first or pass PYTHON=/path/to/python." >&2
  exit 2
fi

DATASET_DIR="${DATASET_DIR:-data/derived/combined_language_dual_chain_seed20260508_test200/annotated_chain}"
ERROR_DATASET="${ERROR_DATASET:-$DATASET_DIR/error_eval_dataset_100.json}"
QUESTION_DATASET="${QUESTION_DATASET:-$DATASET_DIR/question_eval_dataset_50_50.json}"
PRECISION_DATASET="${PRECISION_DATASET:-$DATASET_DIR/question_right_only_50.json}"

UNIFIED_CATALOG="${UNIFIED_CATALOG:-catalogs/legacy/unified_rule_library_v2_distilled300_20260503.json}"
EXPERIENCE_CODE_MANIFEST="${EXPERIENCE_CODE_MANIFEST:-results/experience_symbolic_program_manifest_v2_unified.json}"
EXPERIENCE_CODE_MODULE="${EXPERIENCE_CODE_MODULE:-symbolic.generated_experience_checks_v2_unified}"
SYMBOLIC_TOPIC_CHECK_LIMIT="${SYMBOLIC_TOPIC_CHECK_LIMIT:-40}"

CHECK_MODEL="${CHECK_MODEL:-qwen3-30b-a3b-instruct-2507}"
STRONG_MODEL="${STRONG_MODEL:-qwen3-30b-a3b-instruct-2507}"
PRECISION_MODE="${PRECISION_MODE:-strict}"
MAX_PER_SAMPLE="${MAX_PER_SAMPLE:-12}"
MAX_PER_PARAGRAPH="${MAX_PER_PARAGRAPH:-2}"

RUN_TAG="${RUN_TAG:-e2e_experience_symbolic_$(date -u +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-results/$RUN_TAG}"
mkdir -p "$OUTDIR"

SKIP_BUILD="${SKIP_BUILD:-1}"
ENABLE_QUALITY_AUDIT="${ENABLE_QUALITY_AUDIT:-0}"
NO_SYMBOLIC_CHECK="${NO_SYMBOLIC_CHECK:-0}"

if [[ ! -f "$UNIFIED_CATALOG" ]]; then
  echo "[error] unified catalog not found: $UNIFIED_CATALOG" >&2
  exit 3
fi
if [[ ! -f "$EXPERIENCE_CODE_MANIFEST" ]]; then
  echo "[error] experience-code manifest not found: $EXPERIENCE_CODE_MANIFEST" >&2
  exit 4
fi
if [[ "$SKIP_BUILD" == "1" ]]; then
  for f in "$ERROR_DATASET" "$QUESTION_DATASET"; do
    if [[ ! -f "$f" ]]; then
      echo "[error] dataset missing (set SKIP_BUILD=0 to rebuild or provide correct path): $f" >&2
      exit 5
    fi
  done
fi

cat > "$OUTDIR/run_config.txt" <<EOF
timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
root=$ROOT_DIR
python=$PYTHON
dataset_dir=$DATASET_DIR
error_dataset=$ERROR_DATASET
question_dataset=$QUESTION_DATASET
precision_dataset=$PRECISION_DATASET
unified_catalog=$UNIFIED_CATALOG
experience_code_manifest=$EXPERIENCE_CODE_MANIFEST
experience_code_module=$EXPERIENCE_CODE_MODULE
symbolic_topic_check_limit=$SYMBOLIC_TOPIC_CHECK_LIMIT
check_model=$CHECK_MODEL
strong_model=$STRONG_MODEL
precision_mode=$PRECISION_MODE
max_per_sample=$MAX_PER_SAMPLE
max_per_paragraph=$MAX_PER_PARAGRAPH
run_tag=$RUN_TAG
outdir=$OUTDIR
skip_build=$SKIP_BUILD
enable_quality_audit=$ENABLE_QUALITY_AUDIT
no_symbolic_check=$NO_SYMBOLIC_CHECK
EOF
echo "[ok] config written to $OUTDIR/run_config.txt"

EXTRA_PIPELINE_FLAGS=()
if [[ "$SKIP_BUILD" == "1" ]]; then
  EXTRA_PIPELINE_FLAGS+=("--skip-build")
fi
if [[ "$ENABLE_QUALITY_AUDIT" == "1" ]]; then
  EXTRA_PIPELINE_FLAGS+=("--run-quality-audit")
fi
if [[ "$NO_SYMBOLIC_CHECK" == "1" ]]; then
  EXTRA_PIPELINE_FLAGS+=("--no-symbolic-check")
fi

if [[ "$SKIP_BUILD" == "1" ]]; then
  RECALL_SIZE_OVERRIDE="$(basename "$ERROR_DATASET" | sed -E 's/^error_eval_dataset_([0-9]+)\.json$/\1/')"
  Q_SIZES="$(basename "$QUESTION_DATASET" | sed -E 's/^question_eval_dataset_([0-9]+)_([0-9]+)\.json$/\1 \2/')"
  read -r QUESTION_RECALL_SIZE QUESTION_PRECISION_SIZE <<< "$Q_SIZES"
  if [[ -z "${RECALL_SIZE_OVERRIDE:-}" || "$RECALL_SIZE_OVERRIDE" == "$(basename "$ERROR_DATASET")" ]]; then
    RECALL_SIZE_OVERRIDE=100
  fi
  if [[ -z "${QUESTION_RECALL_SIZE:-}" ]]; then QUESTION_RECALL_SIZE=50; fi
  if [[ -z "${QUESTION_PRECISION_SIZE:-}" ]]; then QUESTION_PRECISION_SIZE=50; fi

  EXPECTED_ERROR_NAME="error_eval_dataset_${RECALL_SIZE_OVERRIDE}.json"
  EXPECTED_QUESTION_NAME="question_eval_dataset_${RECALL_SIZE_OVERRIDE}_${QUESTION_PRECISION_SIZE}.json"
  EXPECTED_RIGHT_NAME="question_right_only_${QUESTION_PRECISION_SIZE}.json"

  STAGED_ERROR="$OUTDIR/$EXPECTED_ERROR_NAME"
  STAGED_QUESTION="$OUTDIR/$EXPECTED_QUESTION_NAME"
  STAGED_RIGHT="$OUTDIR/$EXPECTED_RIGHT_NAME"

  cp -f "$ERROR_DATASET" "$STAGED_ERROR"
  cp -f "$QUESTION_DATASET" "$STAGED_QUESTION"
  if [[ -f "$PRECISION_DATASET" ]]; then
    cp -f "$PRECISION_DATASET" "$STAGED_RIGHT"
  fi

  RECALL_SIZE_FOR_PIPELINE="$RECALL_SIZE_OVERRIDE"
  PRECISION_SIZE_FOR_PIPELINE="$QUESTION_PRECISION_SIZE"
else
  RECALL_SIZE_FOR_PIPELINE="${RECALL_SIZE_FOR_PIPELINE:-100}"
  PRECISION_SIZE_FOR_PIPELINE="${PRECISION_SIZE_FOR_PIPELINE:-50}"
fi

set -x
"$PYTHON" scripts/run_physics_eval_pipeline.py \
  --python "$PYTHON" \
  --output-dir "$OUTDIR" \
  --recall-size "$RECALL_SIZE_FOR_PIPELINE" \
  --precision-size "$PRECISION_SIZE_FOR_PIPELINE" \
  --strong-model "$STRONG_MODEL" \
  --check-model "$CHECK_MODEL" \
  --unified-catalog "$UNIFIED_CATALOG" \
  --experience-code-manifest "$EXPERIENCE_CODE_MANIFEST" \
  --experience-code-module "$EXPERIENCE_CODE_MODULE" \
  --symbolic-topic-check-limit "$SYMBOLIC_TOPIC_CHECK_LIMIT" \
  --max-per-sample "$MAX_PER_SAMPLE" \
  --max-per-paragraph "$MAX_PER_PARAGRAPH" \
  "${EXTRA_PIPELINE_FLAGS[@]}" \
  2>&1 | tee "$OUTDIR/pipeline.log"
set +x

echo "[ok] e2e pipeline finished. results in $OUTDIR"
