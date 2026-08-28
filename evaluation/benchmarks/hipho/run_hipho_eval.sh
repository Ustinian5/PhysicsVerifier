#!/usr/bin/env bash
# Run HiPhO text-only evaluation for a model checkpoint served via vLLM.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
VENV="${VENV:-${ROOT}/.venv}"
BENCH_ROOT="${BENCH_ROOT:-/slow_share/jinjianhan/workspace/benchmarks/hipho}"
HIPHO_JSONL="${HIPHO_JSONL:-${BENCH_ROOT}/hipho_text_only.jsonl}"
HIPHO_MANIFEST="${HIPHO_MANIFEST:-${BENCH_ROOT}/hipho_official_manifest.json}"
MODEL_DIR="${MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-30B-A3B-Instruct-2507}"
MODEL_NAME="${MODEL_NAME:-qwen3-30b-a3b-instruct-2507}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8766/v1}"
OUT_DIR="${OUT_DIR:-${ROOT}/results/hipho_eval}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TEMPERATURE="${TEMPERATURE:-0.2}"
MAX_TOKENS="${MAX_TOKENS:-8192}"

OUT_DIR="${OUT_DIR:-${ROOT}/results/hipho_eval}"
RUN_LABEL="${RUN_LABEL:-}"
TS="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${RUN_LABEL}" ]]; then
  OUT_SUB="${OUT_DIR}/${RUN_LABEL}"
else
  OUT_SUB="${OUT_DIR}/${TS}"
fi
mkdir -p "${OUT_SUB}"
PRED_OUT="${PRED_OUT:-${OUT_SUB}/predictions.jsonl}"
SCORE_OUT="${SCORE_OUT:-${OUT_SUB}/scores.json}"

"${VENV}/bin/python" "${ROOT}/evaluation/benchmarks/hipho/generate_hipho_predictions.py" \
  --input "${HIPHO_JSONL}" \
  --output "${PRED_OUT}" \
  --base-url "${BASE_URL}" \
  --model "${MODEL_NAME}" \
  --max-samples "${MAX_SAMPLES}" \
  --temperature "${TEMPERATURE:-0.2}" \
  --max-tokens "${MAX_TOKENS:-8192}"

"${VENV}/bin/python" "${ROOT}/evaluation/benchmarks/hipho/score_hipho_predictions.py" \
  --predictions "${PRED_OUT}" \
  --output "${SCORE_OUT}"

echo "[ok] HiPhO eval done"
echo "predictions: ${PRED_OUT}"
echo "scores: ${SCORE_OUT}"
