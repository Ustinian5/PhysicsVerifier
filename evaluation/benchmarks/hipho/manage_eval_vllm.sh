#!/usr/bin/env bash
# Start/stop a temporary vLLM server for benchmark evaluation.
set -euo pipefail

ACTION="${1:-status}"
MODEL_DIR="${MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-30B-A3B-Instruct-2507}"
SERVED_NAME="${SERVED_NAME:-qwen3-30b-a3b-instruct-2507}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8766}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_LEN="${MAX_LEN:-32768}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
VLLM="${VLLM:-/data1/jinjianhan/venv/openrlhf_train/bin/vllm}"
RUN_ID="${RUN_ID:-default}"
LOG="${LOG:-/home/jinjianhan/PhysicsVerifier/results/hipho_eval/vllm_${RUN_ID}.log}"
PID_FILE="${PID_FILE:-/home/jinjianhan/PhysicsVerifier/results/hipho_eval/vllm_${RUN_ID}.pid}"

mkdir -p "$(dirname "${LOG}")" "$(dirname "${PID_FILE}")"

_stop_pid() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    pkill -TERM -P "${pid}" 2>/dev/null || true
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 3
    if kill -0 "${pid}" 2>/dev/null; then
      pkill -9 -P "${pid}" 2>/dev/null || true
      kill -9 "${pid}" 2>/dev/null || true
    fi
  fi
}

case "${ACTION}" in
  start)
    if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
      echo "[error] invalid model dir: ${MODEL_DIR}" >&2
      exit 2
    fi
    if [[ -f "${PID_FILE}" ]]; then
      old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
      if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
        if curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
          echo "[ok] vLLM already running pid=${old_pid} url=http://${HOST}:${PORT}/v1"
          exit 0
        fi
        _stop_pid "${old_pid}"
      fi
    fi
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    extra_args=()
    if [[ "${ENABLE_PREFIX_CACHING:-0}" == "1" ]]; then
      extra_args+=(--enable-prefix-caching)
    fi
    if [[ -n "${TOKENIZER:-}" ]]; then
      extra_args+=(--tokenizer "${TOKENIZER}")
    fi
    nohup "${VLLM}" serve "${MODEL_DIR}" \
      --host "${HOST}" \
      --port "${PORT}" \
      --served-model-name "${SERVED_NAME}" \
      --dtype auto \
      --max-model-len "${MAX_LEN}" \
      --gpu-memory-utilization "${GPU_UTIL}" \
      --enforce-eager \
      --trust-remote-code \
      --disable-log-requests \
      "${extra_args[@]}" \
      >>"${LOG}" 2>&1 &
    echo $! >"${PID_FILE}"
    ready_secs="${VLLM_READY_SECS:-600}"
    polls=$(( ready_secs / 5 ))
    for _ in $(seq 1 "${polls}"); do
      if curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[ok] started vLLM pid=$(cat "${PID_FILE}") model=${SERVED_NAME} dir=${MODEL_DIR}"
        exit 0
      fi
      sleep 5
    done
    echo "[error] vLLM failed to become ready; see ${LOG}" >&2
    exit 1
    ;;
  stop)
    if [[ -f "${PID_FILE}" ]]; then
      _stop_pid "$(cat "${PID_FILE}" 2>/dev/null || true)"
      rm -f "${PID_FILE}"
    fi
    pkill -f "vllm serve.*--port ${PORT}" 2>/dev/null || true
    echo "[ok] stopped vLLM on port ${PORT}"
    ;;
  status)
    if curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[ok] vLLM ready on http://${HOST}:${PORT}/v1"
      exit 0
    fi
    echo "[warn] vLLM not ready on http://${HOST}:${PORT}/v1"
    exit 1
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 1
    ;;
esac
