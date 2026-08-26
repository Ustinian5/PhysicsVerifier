#!/usr/bin/env bash
# Disconnect-safe launcher: reward server + training (Ray --no-wait) + curve watcher.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
SAVE_PATH="${QWEN30B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-30b-physics-openrlhf}"
LOG="${TRAIN_LOG:-${SAVE_PATH}/train_launch.log}"
PID_FILE="${SAVE_PATH}/training_launcher.pid"

mkdir -p "${SAVE_PATH}" "$(dirname "${LOG}")"

# Reward server is already nohup-managed; ensure it is up.
bash "${ROOT}/training/reward_server/start_reward_server.sh"

# Background curve watcher (nohup + PID file).
bash "${ROOT}/training/openrlhf/watch_training_curves.sh" start

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "[warn] training launcher pid=${old_pid} still running; skip duplicate submit" >&2
    exit 0
  fi
fi

{
  echo ""
  echo "===== DAEMON LAUNCH $(date -Iseconds) ====="
  cd "${ROOT}"
  bash training/openrlhf/launch_training.sh
} >>"${LOG}" 2>&1 &

launcher_pid=$!
echo "${launcher_pid}" >"${PID_FILE}"
disown "${launcher_pid}" 2>/dev/null || true
echo "[ok] training launcher pid=${launcher_pid} log=${LOG}"
echo "[ok] plots -> ${SAVE_PATH}/plots/training_curves.png"
