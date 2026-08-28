#!/usr/bin/env bash
# Start 3x Qwen3-30B-A3B on GPUs 4,5,6 and generate SFT solutions via rejection sampling.
# Uses ports 8780–8783 so HiPhO eval on :8766 is undisturbed.
set -euo pipefail
ulimit -f unlimited 2>/dev/null || true
export TMPDIR="${TMPDIR:-/slow_share/jinjianhan/tmp/swift}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
mkdir -p "${TMPDIR}"

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
MODEL_DIR="${JUDGE_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-30B-A3B-Instruct-2507}"
SERVED_NAME="${SFT_GEN_MODEL:-qwen3-30b-a3b}"
LB_PORT="${SFT_GEN_LB_PORT:-8780}"
PORTS=(8781 8782 8783)
GPUS=(4 5 6)
RUN_IDS=(sft_gen0 sft_gen1 sft_gen2)
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
CKPT_LOG="${CKPT_LOG:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-sft/sft_datagen.log}"
mkdir -p "${LOG_DIR}" "$(dirname "${CKPT_LOG}")"

start_one() {
  local gpu="$1" port="$2" run_id="$3"
  if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "[sft-gen] already ready gpu=${gpu} port=${port}"
    return 0
  fi
  RUN_ID="${run_id}" MODEL_DIR="${MODEL_DIR}" PORT="${port}" CUDA_DEVICE="${gpu}" \
    MAX_LEN=8192 GPU_UTIL="${GPU_UTIL:-0.88}" SERVED_NAME="${SERVED_NAME}" \
    LOG="${LOG_DIR}/${run_id}_vllm.log" PID_FILE="${LOG_DIR}/${run_id}_vllm.pid" \
    VLLM_READY_SECS=7200 ENABLE_PREFIX_CACHING=1 \
    bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" start
}

start_pids=()
for i in 0 1 2; do
  start_one "${GPUS[$i]}" "${PORTS[$i]}" "${RUN_IDS[$i]}" &
  start_pids+=($!)
done
fail=0
for p in "${start_pids[@]}"; do
  wait "${p}" || fail=1
done
[[ "${fail}" -eq 0 ]] || { echo "[error] one or more 30B generators failed to start" >&2; exit 2; }

if [[ -f "${LOG_DIR}/sft_gen_lb.pid" ]]; then
  old="$(cat "${LOG_DIR}/sft_gen_lb.pid" 2>/dev/null || true)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    kill -TERM "${old}" 2>/dev/null || true
    sleep 1
    kill -9 "${old}" 2>/dev/null || true
  fi
fi
nohup "${PYTHON}" "${ROOT}/training/openrlhf/judge_lb_proxy.py" \
  --host 127.0.0.1 --port "${LB_PORT}" \
  --backends "127.0.0.1:${PORTS[0]},127.0.0.1:${PORTS[1]},127.0.0.1:${PORTS[2]}" \
  >>"${LOG_DIR}/sft_gen_lb.log" 2>&1 &
echo $! >"${LOG_DIR}/sft_gen_lb.pid"
for _ in $(seq 1 40); do
  curl -sf "http://127.0.0.1:${LB_PORT}/health" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -sf "http://127.0.0.1:${LB_PORT}/v1/models" >/dev/null \
  || { echo "[error] SFT gen LB not ready on :${LB_PORT}" >&2; exit 2; }

echo "[sft-gen] generating solutions via ${LB_PORT}"
"${PYTHON}" "${ROOT}/training/rl_data/generate_sft_solutions.py" \
  --base-url "http://127.0.0.1:${LB_PORT}/v1" \
  --model "${SERVED_NAME}" \
  --k "${SFT_GEN_K:-4}" \
  --concurrency "${SFT_GEN_CONCURRENCY:-18}" \
  --local-only \
  "$@"
echo "[ok] local SFT generation finished; unsolved rows in data/rl/sft_unsolved.jsonl"
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
if [[ -n "${OPENAI_API_KEY:-}" && "${OPENAI_API_KEY}" != "EMPTY" && -n "${OPENAI_BASE_URL:-}" && "${OPENAI_BASE_URL}" != *"127.0.0.1"* ]]; then
  echo "[sft-gen] filling unsolved via API model=${SFT_API_MODEL:-qwen3-30b-a3b-instruct-2507}"
  SFT_API_MODEL="${SFT_API_MODEL:-qwen3-30b-a3b-instruct-2507}" \
    "${PYTHON}" "${ROOT}/training/rl_data/generate_sft_solutions.py" \
      --api-only \
      --api-base-url "${OPENAI_BASE_URL}" \
      --api-key "${OPENAI_API_KEY}" \
      --api-model "${SFT_API_MODEL}" \
      --concurrency "${SFT_API_CONCURRENCY:-8}" \
      "$@"
else
  echo "[hint] skip API fill (no remote OPENAI_*); set SFT_API_MODEL and keys in .env"
fi
for i in 0 1 2; do
  RUN_ID="${RUN_IDS[$i]}" PORT="${PORTS[$i]}" PID_FILE="${LOG_DIR}/${RUN_IDS[$i]}_vllm.pid" \
    bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" stop || true
done
if [[ -f "${LOG_DIR}/sft_gen_lb.pid" ]]; then
  old="$(cat "${LOG_DIR}/sft_gen_lb.pid" 2>/dev/null || true)"
  if [[ -n "${old}" ]]; then
    kill -TERM "${old}" 2>/dev/null || true
  fi
  rm -f "${LOG_DIR}/sft_gen_lb.pid"
fi
