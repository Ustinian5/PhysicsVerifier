#!/usr/bin/env bash
# Precise cleanup for the 4-GPU 8B pilot. Never global pkill / ray stop --force.
# Order: wrapper -> direct train -> isolated Ray -> judge -> reward -> reservation -> curve watcher.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
CKPT="${QWEN8B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-bootstrap10}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
RAY_GCS_PORT="${RAY_GCS_PORT:-26379}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/orhf8b_ray_${RAY_GCS_PORT}}"
DELETE_RAY_TMP="${DELETE_RAY_TMP:-0}"

stop_pid_file() {
  local file="$1"
  local label="${2:-pid}"
  [[ -f "${file}" ]] || return 0
  local pid
  pid="$(cat "${file}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    pkill -TERM -P "${pid}" 2>/dev/null || true
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 2
    pkill -9 -P "${pid}" 2>/dev/null || true
    kill -9 "${pid}" 2>/dev/null || true
    echo "[stop] ${label} pid=${pid}"
  else
    echo "[stop] ${label} stale pid file ${file}"
  fi
  rm -f "${file}"
}

gcs_alive() {
  ss -ltn 2>/dev/null | grep -q ":${RAY_GCS_PORT} " || \
    curl -sf "http://127.0.0.1:${RAY_GCS_PORT}" >/dev/null 2>&1
}

# 1) nohup / wrapper
stop_pid_file "${LOG_DIR}/four_gpu_pilot10.pid" "pilot_wrapper"
stop_pid_file "${LOG_DIR}/bootstrap_pilot10.pid" "bootstrap_wrapper"

# 2) direct train (current + legacy ckpt)
stop_pid_file "${CKPT}/direct_train.pid" "direct_train"
stop_pid_file "/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-pilot10/direct_train.pid" "legacy_direct_train"

# 3) isolated Ray (PID file first, then GCS-port-bound processes only)
RAY_HEAD_PID_FILE="${CKPT}/ray/ray_head.pid"
stop_pid_file "${RAY_HEAD_PID_FILE}" "ray_head"
stop_pid_file "/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-pilot10/ray/ray_head.pid" "legacy_ray_head"
while read -r rp; do
  [[ -z "${rp}" ]] && continue
  cmdline="$(tr '\0' ' ' <"/proc/${rp}/cmdline" 2>/dev/null || true)"
  if [[ "${cmdline}" == *"--port=${RAY_GCS_PORT}"* ]] || [[ "${cmdline}" == *"--port ${RAY_GCS_PORT}"* ]] || \
     [[ "${cmdline}" == *"gcs_server"* && "${cmdline}" == *"${RAY_GCS_PORT}"* ]]; then
    kill -TERM "${rp}" 2>/dev/null || true
    sleep 1
    kill -9 "${rp}" 2>/dev/null || true
    echo "[stop] ray pid=${rp} port=${RAY_GCS_PORT}"
  fi
done < <(pgrep -f "ray.*${RAY_GCS_PORT}|gcs_server.*${RAY_GCS_PORT}" || true)
while read -r rp; do
  [[ -z "${rp}" ]] && continue
  cmdline="$(tr '\0' ' ' <"/proc/${rp}/cmdline" 2>/dev/null || true)"
  if [[ "${cmdline}" == *"${RAY_TMPDIR}"* ]]; then
    kill -TERM "${rp}" 2>/dev/null || true
    sleep 1
    kill -9 "${rp}" 2>/dev/null || true
    echo "[stop] ray-tmpdir pid=${rp}"
  fi
done < <(pgrep -f "${RAY_TMPDIR}" || true)
rm -f "${CKPT}/ray/ray_address.txt"

# Never delete a live Ray session directory.
if gcs_alive; then
  echo "[warn] GCS :${RAY_GCS_PORT} still listening; leaving ${RAY_TMPDIR} intact" >&2
else
  if [[ "${DELETE_RAY_TMP}" == "1" && -d "${RAY_TMPDIR}" ]]; then
    rm -rf "${RAY_TMPDIR}"
    echo "[stop] removed idle Ray tmp ${RAY_TMPDIR}"
  fi
fi

# 4) local judge
bash "${ROOT}/training/openrlhf/stop_local_judge.sh" >/dev/null 2>&1 || true

# 5) reward server
stop_pid_file "${LOG_DIR}/physics_reward_server.pid" "reward_server"
rm -f "${LOG_DIR}/physics_reward_server.mode"

# 6) reservation holder
stop_pid_file "${LOG_DIR}/gpu_reservation.pid" "gpu_reservation"

# 7) curve watcher (current ckpt and legacy pilot10 ckpt)
stop_pid_file "${CKPT}/plots/curve_watcher.pid" "curve_watcher"
stop_pid_file "/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-pilot10/plots/curve_watcher.pid" "legacy_curve_watcher"

echo "[ok] stop_four_gpu_pilot finished ckpt=${CKPT}"
