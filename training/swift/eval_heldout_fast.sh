#!/usr/bin/env bash
# Fast heldout/HiPhO answer-acc probe for a single HF checkpoint. Uses GPU 7 by default.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
SCRIPT_DIR="${ROOT}/evaluation/benchmarks/hipho"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
VENV="${VENV:-${ROOT}/.venv}"
MODEL_DIR="${1:?usage: eval_heldout_fast.sh MODEL_DIR [out_dir]}"
OUT="${2:-${MODEL_DIR}/heldout_fast_eval}"
PORT="${PORT:-8766}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
HELDOUT="${HELDOUT_JSONL:-${ROOT}/data/rl/heldout_eval.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
mkdir -p "${OUT}"
RUN_ID="heldout_fast" MODEL_DIR="${MODEL_DIR}" PORT="${PORT}" CUDA_DEVICE="${CUDA_DEVICE}" \
  MAX_LEN=8192 GPU_UTIL=0.45 SERVED_NAME=qwen3-8b \
  bash "${SCRIPT_DIR}/manage_eval_vllm.sh" start
"${PYTHON}" "${SCRIPT_DIR}/generate_hipho_predictions.py" \
  --input "${HELDOUT}" \
  --output "${OUT}/heldout_predictions.jsonl" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model qwen3-8b \
  --max-samples "${MAX_SAMPLES}" \
  --max-tokens 4096 \
  --concurrency 16 \
  --resume
"${VENV}/bin/python" "${SCRIPT_DIR}/score_hipho_predictions.py" \
  --predictions "${OUT}/heldout_predictions.jsonl" \
  --output "${OUT}/heldout_scores.json" \
  --no-use-verifier
cat "${OUT}/heldout_scores.json"
RUN_ID="heldout_fast" PORT="${PORT}" bash "${SCRIPT_DIR}/manage_eval_vllm.sh" stop || true
