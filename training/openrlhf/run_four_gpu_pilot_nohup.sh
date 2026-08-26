#!/usr/bin/env bash
# Launch 10-step 4-GPU pilot under nohup (survives SSH disconnect).
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
PID_FILE="${PID_FILE:-${LOG_DIR}/four_gpu_pilot10.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/four_gpu_pilot10.log}"
CKPT="${QWEN8B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-bootstrap10}"

mkdir -p "${LOG_DIR}" "${CKPT}/plots"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    if pgrep -P "${old_pid}" >/dev/null 2>&1 || pgrep -af 'run_four_gpu_pilot.sh' | grep -v grep >/dev/null 2>&1; then
      echo "[ok] pilot already running pid=${old_pid} log=${LOG_FILE}"
      exit 0
    fi
  fi
fi

# Truncate previous log for this fresh run.
: >"${LOG_FILE}"

nohup bash -c "
  set -euo pipefail
  source '${WORKSPACE}/openrlhf_rl/env.sh'
  export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-0,1,2,3}'
  export TRAIN_STAGE='${TRAIN_STAGE:-bootstrap}'
  export TRAIN_TOPOLOGY='${TRAIN_TOPOLOGY:-colocate}'
  export ACTOR_GPUS='${ACTOR_GPUS:-}'
  export VLLM_ENGINES='${VLLM_ENGINES:-}'
  export PHYSICS_REWARD_MODE='${PHYSICS_REWARD_MODE:-answer_only}'
  export PHYSICS_REWARD_W_FORMAT='${PHYSICS_REWARD_W_FORMAT:-0}'
  export QWEN8B_RL_CKPT='${CKPT}'
  export PILOT_MAX_STEPS='${PILOT_MAX_STEPS:-10}'
  export GENERATE_MAX_LEN='${GENERATE_MAX_LEN:-1024}'
  export ROLLOUT_BATCH_SIZE='${ROLLOUT_BATCH_SIZE:-3}'
  export N_SAMPLES_PER_PROMPT='${N_SAMPLES_PER_PROMPT:-8}'
  export TRAIN_BATCH_SIZE='${TRAIN_BATCH_SIZE:-24}'
  export MICRO_ROLLOUT_BATCH_SIZE='${MICRO_ROLLOUT_BATCH_SIZE:-2}'
  export VLLM_GPU_MEMORY_UTILIZATION='${VLLM_GPU_MEMORY_UTILIZATION:-0.55}'
  export MAX_SAMPLES='${MAX_SAMPLES:-2048}'
  export DYNAMIC_FILTER_MIN='${DYNAMIC_FILTER_MIN:-0.0}'
  export DYNAMIC_FILTER_MAX='${DYNAMIC_FILTER_MAX:-1.0}'
  export DYNAMIC_FILTER_MODE='${DYNAMIC_FILTER_MODE:-reward_variance}'
  export DYNAMIC_FILTER_MAX_GEN_BATCHES='${DYNAMIC_FILTER_MAX_GEN_BATCHES:-32}'
  export PROMPT_DATA='${PROMPT_DATA:-}'
  export JUDGE_CUDA_DEVICE='${JUDGE_CUDA_DEVICE:-3}'
  export ALLOW_RAY_JOBS='${ALLOW_RAY_JOBS:-0}'
  export RAY_JOB_SUBMIT_ATTEMPTS='${RAY_JOB_SUBMIT_ATTEMPTS:-1}'
  export ALLOW_DIRECT_LAUNCH='${ALLOW_DIRECT_LAUNCH:-1}'
  export RAY_GCS_PORT='${RAY_GCS_PORT:-26379}'
  export RAY_DASHBOARD_PORT='${RAY_DASHBOARD_PORT:-28265}'
  bash '${ROOT}/training/openrlhf/run_four_gpu_pilot.sh'
" >>"${LOG_FILE}" 2>&1 &
echo $! >"${PID_FILE}"
echo "[launch] four_gpu_pilot10 pid=$(cat "${PID_FILE}") log=${LOG_FILE} ckpt=${CKPT}"
echo "[launch] stage=${TRAIN_STAGE:-bootstrap} topology=${TRAIN_TOPOLOGY:-colocate} reward_mode=${PHYSICS_REWARD_MODE:-answer_only} train_gpus=${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
