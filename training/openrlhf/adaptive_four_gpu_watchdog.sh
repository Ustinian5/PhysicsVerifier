#!/usr/bin/env bash
# Launch the adaptive four-GPU acquire + Hybrid pilot worker in the background.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
LOG="${ADAPTIVE_LOG:-${LOG_DIR}/adaptive_four_gpu_watchdog.log}"
PID_FILE="${ADAPTIVE_PID_FILE:-${LOG_DIR}/adaptive_four_gpu_watchdog.pid}"
CKPT="${ADAPTIVE_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-pilot10}"
STATUS="${ADAPTIVE_STATUS:-${CKPT}/adaptive_acquire_status.json}"
WORKER="${ROOT}/training/openrlhf/adaptive_four_gpu_worker.sh"

STABLE_SECS="${STABLE_SECS:-600}"
FREE_MIB="${FREE_MIB:-75000}"
UTIL_MAX="${UTIL_MAX:-5}"

mkdir -p "${LOG_DIR}" "${CKPT}"

if [[ -f "${PID_FILE}" ]]; then
  old="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[ok] adaptive watchdog already running pid=${old} log=${LOG}"
    exit 0
  fi
fi

: >"${LOG}"
nohup env \
  ADAPTIVE_CKPT="${CKPT}" \
  ADAPTIVE_STATUS="${STATUS}" \
  FREE_MIB="${FREE_MIB}" \
  UTIL_MAX="${UTIL_MAX}" \
  STABLE_SECS="${STABLE_SECS}" \
  POLL_SECS="${POLL_SECS:-15}" \
  MAX_WAIT_SECS="${MAX_WAIT_SECS:-86400}" \
  RESERVE_MIB="${RESERVE_MIB:-512}" \
  MAX_ACQUIRE_RETRIES="${MAX_ACQUIRE_RETRIES:-5}" \
  BACKOFF_SECS="${BACKOFF_SECS:-120}" \
  RAY_GCS_PORT="${RAY_GCS_PORT:-26379}" \
  RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-28265}" \
  bash "${WORKER}" >>"${LOG}" 2>&1 &

echo $! >"${PID_FILE}"
echo "[launch] adaptive four-GPU watchdog pid=$(cat "${PID_FILE}") log=${LOG} status=${STATUS}"
echo "[hint] waits for any 4 GPUs idle ${STABLE_SECS}s (free>=${FREE_MIB}MiB, util<=${UTIL_MAX}%), then Hybrid pilot"
