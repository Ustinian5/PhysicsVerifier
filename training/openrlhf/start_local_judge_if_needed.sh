#!/usr/bin/env bash
# Start a local PhysicsVerifier vLLM judge on a spare GPU when reward mode needs it.
# If an existing judge is healthy but on the wrong GPU, migrate precisely.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
MODE="${PHYSICS_REWARD_MODE:-answer_low_verifier}"

# External API configured: no local judge.
if [[ -n "${PHYSICSVERIFIER_OPENAI_BASE_URL:-}" ]]; then
  echo "[judge] external API set; skip local judge"
  exit 0
fi

# answer_only does not need a judge.
if [[ "${MODE}" == "answer_only" ]]; then
  echo "[judge] answer_only; skip local judge"
  exit 0
fi

export MODEL_DIR="${JUDGE_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-30B-A3B-Instruct-2507}"
export SERVED_NAME="${JUDGE_SERVED_NAME:-qwen3-30b-a3b}"
export PORT="${JUDGE_PORT:-8766}"
export CUDA_DEVICE="${JUDGE_CUDA_DEVICE:-7}"
export GPU_UTIL="${JUDGE_GPU_UTIL:-0.90}"
export MAX_LEN="${JUDGE_MAX_LEN:-8192}"
export RUN_ID="${JUDGE_RUN_ID:-local_judge}"
export LOG="${JUDGE_LOG:-${ROOT}/logs/local_judge_vllm.log}"
export PID_FILE="${JUDGE_PID_FILE:-${ROOT}/logs/local_judge_vllm.pid}"
export PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
export VLLM="${VLLM:-/data1/jinjianhan/venv/openrlhf_train/bin/vllm}"

judge_gpu_of_pid() {
  local pid="$1"
  # Best effort: parse CUDA_VISIBLE_DEVICES from /proc/<pid>/environ
  local env_cvd
  env_cvd="$(tr '\0' '\n' <"/proc/${pid}/environ" 2>/dev/null | awk -F= '/^CUDA_VISIBLE_DEVICES=/{print $2; exit}')"
  if [[ -n "${env_cvd}" ]]; then
    echo "${env_cvd%%,*}"
    return 0
  fi
  return 1
}

needs_migrate=0
if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      cur_gpu="$(judge_gpu_of_pid "${old_pid}" || true)"
      if [[ -n "${cur_gpu}" && "${cur_gpu}" != "${CUDA_DEVICE}" ]]; then
        echo "[judge] healthy on GPU${cur_gpu} but want GPU${CUDA_DEVICE}; migrating"
        needs_migrate=1
      else
        echo "[judge] already ready at http://127.0.0.1:${PORT}/v1 (gpu=${cur_gpu:-unknown})"
        exit 0
      fi
    else
      echo "[judge] port healthy but pid file stale; restarting on GPU${CUDA_DEVICE}"
      needs_migrate=1
    fi
  else
    echo "[judge] port healthy without pid file; restarting on GPU${CUDA_DEVICE}"
    needs_migrate=1
  fi
fi

if [[ "${needs_migrate}" -eq 1 ]]; then
  bash "${ROOT}/training/openrlhf/stop_local_judge.sh" "${CUDA_DEVICE}" || true
  sleep 2
fi

# Fail fast if CUDA/fabricmanager is down (A800 NVSwitch hosts).
if ! TRY_RESTART_FABRICMANAGER="${TRY_RESTART_FABRICMANAGER:-1}" \
  bash "${ROOT}/training/openrlhf/ensure_cuda_ready.sh"; then
  echo "[error] cannot start local judge: CUDA not ready" >&2
  exit 2
fi

echo "[judge] starting local vLLM on GPU ${CUDA_DEVICE} port ${PORT}"
bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" start
