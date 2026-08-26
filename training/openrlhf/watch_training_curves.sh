#!/usr/bin/env bash
# Daemon: periodically refresh GRPO training curve plots from train_launch.log.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
ENV_FILE="${ENV_FILE:-/slow_share/jinjianhan/workspace/openrlhf_rl/env.sh}"
SAVE_PATH="${QWEN8B_RL_CKPT:-${QWEN30B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-30b-physics-openrlhf}}"
OUT_DIR="${PLOT_OUT_DIR:-${SAVE_PATH}/plots}"
INTERVAL="${PLOT_INTERVAL_SEC:-120}"
PID_FILE="${OUT_DIR}/curve_watcher.pid"
WATCHER_LOG="${OUT_DIR}/curve_watcher.log"

PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
PLOT_SCRIPT="${ROOT}/training/openrlhf/plot_training_curves.py"

mkdir -p "${OUT_DIR}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  PYTHON="${PYTHON:-${TRAIN_VENV}/bin/python}"
fi

run_plot() {
  if [[ "${1:-}" == "--quiet" ]]; then
    "${PYTHON}" "${PLOT_SCRIPT}" \
      --save-path "${SAVE_PATH}" \
      --out-dir "${OUT_DIR}" >>"${WATCHER_LOG}" 2>&1 || true
  else
    "${PYTHON}" "${PLOT_SCRIPT}" \
      --save-path "${SAVE_PATH}" \
      --out-dir "${OUT_DIR}"
  fi
}

watch_loop() {
  echo "[watcher] started at $(date -Iseconds) interval=${INTERVAL}s save_path=${SAVE_PATH}" >>"${WATCHER_LOG}"
  while true; do
    run_plot --quiet
    sleep "${INTERVAL}"
  done
}

case "${1:-start}" in
  start)
    if [[ -f "${PID_FILE}" ]]; then
      old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
      if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
        echo "[ok] curve watcher already running pid=${old_pid}"
        exit 0
      fi
    fi
    nohup bash "${BASH_SOURCE[0]}" run >>"${WATCHER_LOG}" 2>&1 &
    echo $! >"${PID_FILE}"
    disown || true
    echo "[ok] curve watcher started pid=$(cat "${PID_FILE}") log=${WATCHER_LOG}"
    ;;
  run)
    watch_loop
    ;;
  stop)
    if [[ -f "${PID_FILE}" ]]; then
      kill "$(cat "${PID_FILE}")" 2>/dev/null || true
      rm -f "${PID_FILE}"
      echo "[ok] curve watcher stopped"
    fi
    ;;
  once)
    run_plot
    ;;
  *)
    echo "usage: $0 {start|stop|once}" >&2
    exit 1
    ;;
esac
