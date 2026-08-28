#!/usr/bin/env bash
# Rolling-restart 4 self-judge vLLM replicas onto a new 8B checkpoint.
# Does not stop the load balancer; backends come back one at a time.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
MODEL_DIR="${JUDGE_MODEL_DIR:-${1:-}}"
SERVED_NAME="${JUDGE_SERVED_NAME:-qwen3-8b-self-judge}"
PREFER_JUDGE="${PREFER_JUDGE:-4,7,6,5}"
JUDGE_PORTS=(8766 8767 8768 8769)
JUDGE_RUN_IDS=(local_judge local_judge2 local_judge3 local_judge4)
GPU_UTIL="${JUDGE_GPU_UTIL:-0.45}"
MAX_LEN="${JUDGE_MAX_LEN:-8192}"

[[ -n "${MODEL_DIR}" && -f "${MODEL_DIR}/config.json" ]] || {
  echo "[error] JUDGE_MODEL_DIR missing/invalid: ${MODEL_DIR}" >&2
  exit 2
}

IFS=',' read -ra GPUS <<< "${PREFER_JUDGE}"
for i in 0 1 2 3; do
  gpu="${GPUS[$i]}"
  port="${JUDGE_PORTS[$i]}"
  run_id="${JUDGE_RUN_IDS[$i]}"
  echo "[refresh] gpu=${gpu} port=${port} model=${MODEL_DIR}"
  RUN_ID="${run_id}" PORT="${port}" PID_FILE="${LOG_DIR}/${run_id}_vllm.pid" \
    bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" stop || true
  sleep 2
  ENABLE_PREFIX_CACHING=1 \
  JUDGE_MODEL_DIR="${MODEL_DIR}" \
  JUDGE_SERVED_NAME="${SERVED_NAME}" \
  JUDGE_CUDA_DEVICE="${gpu}" \
  JUDGE_PORT="${port}" \
  JUDGE_RUN_ID="${run_id}" \
  JUDGE_GPU_UTIL="${GPU_UTIL}" \
  JUDGE_MAX_LEN="${MAX_LEN}" \
  JUDGE_LOG="${LOG_DIR}/${run_id}_vllm.log" \
  JUDGE_PID_FILE="${LOG_DIR}/${run_id}_vllm.pid" \
  VLLM_READY_SECS=600 \
    bash "${ROOT}/training/openrlhf/start_local_judge_if_needed.sh"
  curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null \
    || { echo "[error] judge :${port} not ready after refresh" >&2; exit 1; }
done
echo "[ok] self-judge refreshed to ${MODEL_DIR}"
