#!/usr/bin/env bash
# Stop the local PhysicsVerifier vLLM judge without touching other users' jobs.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
PORT="${JUDGE_PORT:-8766}"
PID_FILE="${PID_FILE:-${ROOT}/logs/local_judge_vllm.pid}"
EXPECTED_GPU="${1:-${JUDGE_CUDA_DEVICE:-}}"

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    # Kill process group / children first, then parent.
    pkill -TERM -P "${pid}" 2>/dev/null || true
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 3
    pkill -9 -P "${pid}" 2>/dev/null || true
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
fi

# Only kill vLLM serving this exact port if still present.
while read -r pid; do
  [[ -z "${pid}" ]] && continue
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  if [[ "${cmdline}" == *"--port ${PORT}"* ]] || [[ "${cmdline}" == *"--port=${PORT}"* ]]; then
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${pid}" 2>/dev/null || true
  fi
done < <(pgrep -f "vllm serve" || true)

# Also use manage_eval_vllm stop for bookkeeping.
PORT="${PORT}" RUN_ID="${JUDGE_RUN_ID:-local_judge}" \
  bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" stop >/dev/null 2>&1 || true

for _ in $(seq 1 30); do
  if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[error] judge still reachable on :${PORT}" >&2
  exit 1
fi

if [[ -n "${EXPECTED_GPU}" ]]; then
  free_mib="$(nvidia-smi -i "${EXPECTED_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  echo "[ok] stopped local judge on port ${PORT}; GPU${EXPECTED_GPU} free_mib=${free_mib}"
else
  echo "[ok] stopped local judge on port ${PORT}"
fi
