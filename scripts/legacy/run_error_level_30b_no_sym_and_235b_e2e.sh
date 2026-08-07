#!/usr/bin/env bash
# Error-level only (100-sample recall set), sequential API batch for screen/tmux:
#   [1] 30B end-to-end ablation: rules + semantic, NO programmatic symbolic checks
#   [2] 235B end-to-end main pipeline: rules + semantic + experience-code symbolic checks
#
# Prerequisites (from repo root):
#   - .venv with deps; API keys in repo-root .env (OPENAI_API_KEY, OPENAI_BASE_URL)
#   - data/derived/.../error_eval_dataset_100.json
#   - results/experience_symbolic_program_manifest_v2_unified.json (required for [2])
#
# Usage:
#   screen -S pv_err_30b_235b
#   cd /home/jinjianhan/PhysicsVerifier
#   bash scripts/legacy/run_error_level_30b_no_sym_and_235b_e2e.sh
#   ENV_FILE=/path/to/.env bash scripts/legacy/run_error_level_30b_no_sym_and_235b_e2e.sh
#
#   PROGRESS_EVERY=10 SYMBOLIC_TOPIC_CHECK_LIMIT=40 bash scripts/legacy/run_error_level_30b_no_sym_and_235b_e2e.sh
#   RUN_235B_ONLY=1 bash scripts/legacy/run_error_level_30b_no_sym_and_235b_e2e.sh
#   RUN_30B_NO_SYM_ONLY=1 bash scripts/legacy/run_error_level_30b_no_sym_and_235b_e2e.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "[error] PYTHON not executable: $PYTHON" >&2
  exit 2
fi

ENV_FILE="${ENV_FILE:-$ROOT/.env}"

# Load OPENAI_* from .env into this shell (and child Python processes).
# Does not override variables already set in the environment.
load_repo_dotenv() {
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "[warn] .env not found: $ENV_FILE (using existing shell env only)" >&2
    return 0
  fi
  local loaded
  loaded="$("$PYTHON" - "$ENV_FILE" <<'PY'
import os
import shlex
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
keys = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE")

try:
    from dotenv import load_dotenv

    load_dotenv(env_path, override=False)
except ImportError:
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key not in keys or key in os.environ:
                continue
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value

for key in keys:
    val = os.environ.get(key)
    if val:
        print(f"export {key}={shlex.quote(val)}")
PY
)" || true
  if [[ -n "$loaded" ]]; then
    # shellcheck disable=SC1090
    eval "$loaded"
    echo "[ok] loaded API env from $ENV_FILE"
  else
    echo "[warn] $ENV_FILE present but no OPENAI_* keys exported" >&2
  fi
}

load_repo_dotenv

DATASET_DIR="${DATASET_DIR:-data/derived/combined_language_dual_chain_seed20260508_test200/annotated_chain}"
ERROR_DATASET="${ERROR_DATASET:-$DATASET_DIR/error_eval_dataset_100.json}"
UNIFIED_CATALOG="${UNIFIED_CATALOG:-catalogs/legacy/unified_rule_library_v2_distilled300_20260503.json}"
MANIFEST="${EXPERIENCE_CODE_MANIFEST:-results/experience_symbolic_program_manifest_v2_unified.json}"
MODULE="${EXPERIENCE_CODE_MODULE:-symbolic.generated_experience_checks_v2_unified}"

MODEL_30B="${MODEL_30B:-qwen3-30b-a3b-instruct-2507}"
MODEL_235B="${MODEL_235B:-qwen3-235b-a22b-instruct-2507}"
SYMBOLIC_TOPIC_CHECK_LIMIT="${SYMBOLIC_TOPIC_CHECK_LIMIT:-40}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
MAX_PER_SAMPLE="${MAX_PER_SAMPLE:-12}"
MAX_PER_PARAGRAPH="${MAX_PER_PARAGRAPH:-2}"

TAG_NO_SYM="${TAG_NO_SYM:-e2e_no_symbolic_30b_error}"
TAG_235B="${TAG_235B:-e2e_main_235b_error}"
TAG_MAIN_30B="${TAG_MAIN_30B:-e2e_main_30b_error}"

RUN_30B_NO_SYM_ONLY="${RUN_30B_NO_SYM_ONLY:-0}"
RUN_235B_ONLY="${RUN_235B_ONLY:-0}"
RUN_MAIN_30B_ONLY="${RUN_MAIN_30B_ONLY:-0}"
active=$((RUN_30B_NO_SYM_ONLY + RUN_235B_ONLY + RUN_MAIN_30B_ONLY))
if [[ "$active" -gt 1 ]]; then
  echo "[error] set at most one of RUN_30B_NO_SYM_ONLY, RUN_235B_ONLY, RUN_MAIN_30B_ONLY" >&2
  exit 2
fi

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
echo "$STAMP" > "$ROOT/results/_error_level_30b_235b_stamp.txt"

preflight() {
  if [[ ! -f "$ERROR_DATASET" ]]; then
    echo "[error] Missing error dataset: $ERROR_DATASET" >&2
    exit 3
  fi
  if [[ ! -f "$UNIFIED_CATALOG" ]]; then
    echo "[error] Missing unified catalog: $UNIFIED_CATALOG" >&2
    exit 3
  fi
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[error] OPENAI_API_KEY missing (set in $ENV_FILE or shell env)" >&2
    exit 3
  fi
  if [[ -z "${OPENAI_BASE_URL:-}${OPENAI_API_BASE:-}" ]]; then
    echo "[warn] OPENAI_BASE_URL / OPENAI_API_BASE not set; using default OpenAI endpoint." >&2
  fi
}

run_error_e2e() {
  local tag="$1"
  local model="$2"
  local no_sym_flag="$3"  # "1" or "0"
  local out="$ROOT/results/${tag}_${STAMP}"
  mkdir -p "$out"
  cp -f "$ERROR_DATASET" "$out/error_eval_dataset_100.json"
  echo "$model" > "$out/check_model.txt"
  echo "$no_sym_flag" > "$out/no_symbolic_check.txt"
  echo "$SYMBOLIC_TOPIC_CHECK_LIMIT" > "$out/symbolic_topic_check_limit.txt"

  cat > "$out/run_config.txt" <<EOF
timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
stamp=$STAMP
tag=$tag
check_model=$model
no_symbolic_check=$no_sym_flag
symbolic_topic_check_limit=$SYMBOLIC_TOPIC_CHECK_LIMIT
error_dataset=$ERROR_DATASET
unified_catalog=$UNIFIED_CATALOG
experience_code_manifest=$MANIFEST
experience_code_module=$MODULE
max_per_sample=$MAX_PER_SAMPLE
max_per_paragraph=$MAX_PER_PARAGRAPH
progress_every=$PROGRESS_EVERY
EOF

  local sym_extra=()
  if [[ "$no_sym_flag" == "1" ]]; then
    sym_extra+=(--no-symbolic-check)
  else
    if [[ ! -f "$MANIFEST" ]]; then
      echo "[error] Manifest required for symbolic e2e ($tag): $MANIFEST" >&2
      exit 4
    fi
  fi

  echo ""
  echo "================================================================"
  echo "[$tag] model=$model no_symbolic=$no_sym_flag"
  echo "  out: $out"
  echo "================================================================"

  local t0
  t0=$(date -u +%s)

  "$PYTHON" scripts/run_verifier.py \
    --input "$out/error_eval_dataset_100.json" \
    --output "$out/error_verifier_results.json" \
    --symbolic-output "$out/error_symbolic_audit.json" \
    --model "$model" \
    --unified-catalog "$UNIFIED_CATALOG" \
    --experience-code-manifest "$MANIFEST" \
    --experience-code-module "$MODULE" \
    --symbolic-topic-check-limit "$SYMBOLIC_TOPIC_CHECK_LIMIT" \
    --max-per-sample "$MAX_PER_SAMPLE" \
    --max-per-paragraph "$MAX_PER_PARAGRAPH" \
    --progress-interval "$PROGRESS_EVERY" \
    "${sym_extra[@]}" \
    2>&1 | tee "$out/run_verifier.log"

  "$PYTHON" scripts/evaluate_physics_eval_sets.py \
    --dataset "$out/error_eval_dataset_100.json" \
    --results "$out/error_verifier_results.json" \
    --audit "$out/error_symbolic_audit.json" \
    --output "$out/error_metrics.json" \
    --match-mode location

  local t1 wall
  t1=$(date -u +%s)
  wall=$((t1 - t0))
  echo "[ok] $tag finished wall_s=${wall}s metrics=$out/error_metrics.json"
}

preflight

SCRIPT_START=$(date -u +%s)
echo "================================================================"
echo "[batch] error-level 30B no-symbolic + 235B e2e"
echo "  STAMP=$STAMP"
echo "  ENV_FILE=$ENV_FILE"
echo "  OPENAI_BASE_URL=${OPENAI_BASE_URL:-${OPENAI_API_BASE:-<default>}}"
echo "  ERROR_DATASET=$ERROR_DATASET"
echo "  SYMBOLIC_TOPIC_CHECK_LIMIT=$SYMBOLIC_TOPIC_CHECK_LIMIT"
echo "  RUN_30B_NO_SYM_ONLY=$RUN_30B_NO_SYM_ONLY RUN_235B_ONLY=$RUN_235B_ONLY RUN_MAIN_30B_ONLY=$RUN_MAIN_30B_ONLY"
echo "  UNIFIED_CATALOG=$UNIFIED_CATALOG"
echo "================================================================"

if [[ "$RUN_235B_ONLY" != "1" && "$RUN_MAIN_30B_ONLY" != "1" ]]; then
  run_error_e2e "$TAG_NO_SYM" "$MODEL_30B" "1"
fi

if [[ "$RUN_30B_NO_SYM_ONLY" != "1" && "$RUN_MAIN_30B_ONLY" != "1" ]]; then
  run_error_e2e "$TAG_235B" "$MODEL_235B" "0"
fi

if [[ "$RUN_MAIN_30B_ONLY" == "1" ]]; then
  run_error_e2e "$TAG_MAIN_30B" "$MODEL_30B" "0"
fi

SCRIPT_END=$(date -u +%s)
WALL=$((SCRIPT_END - SCRIPT_START))
echo ""
echo "[ok] batch complete STAMP=$STAMP wall_s=${WALL}s ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
echo "  [30B no-symbolic] results/${TAG_NO_SYM}_${STAMP}/error_metrics.json"
echo "  [235B e2e main]     results/${TAG_235B}_${STAMP}/error_metrics.json"
echo "  [30B main]          results/${TAG_MAIN_30B}_${STAMP}/error_metrics.json"
