#!/usr/bin/env bash
# Stop ms-swift GRPO only. Judges / reward / LB are skipped unless SKIP_STOP_*=0.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
CKPT="${QWEN8B_SWIFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-swift}"

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

stop_pid_file "${LOG_DIR}/swift_grpo.pid" "swift_wrapper"
stop_pid_file "${CKPT}/swift_train.pid" "swift_train"

while read -r rp; do
  [[ -z "${rp}" ]] && continue
  cmdline="$(tr '\0' ' ' <"/proc/${rp}/cmdline" 2>/dev/null || true)"
  if [[ "${cmdline}" == *"swift"* && "${cmdline}" == *"rlhf"* ]]; then
    kill -TERM "${rp}" 2>/dev/null || true
    sleep 1
    kill -9 "${rp}" 2>/dev/null || true
    echo "[stop] swift rlhf pid=${rp}"
  fi
done < <(pgrep -f "swift rlhf|swift.cli.rlhf" || true)

if [[ "${SKIP_STOP_LB:-1}" != "1" ]]; then
  stop_pid_file "${LOG_DIR}/judge_lb_proxy.pid" "judge_lb"
else
  echo "[stop] skip judge LB (SKIP_STOP_LB=${SKIP_STOP_LB:-1})"
fi

if [[ "${SKIP_STOP_JUDGE:-1}" != "1" ]]; then
  bash "${ROOT}/training/openrlhf/stop_local_judge.sh" >/dev/null 2>&1 || true
  PID_FILE="${LOG_DIR}/local_judge2_vllm.pid" JUDGE_PORT=8767 JUDGE_RUN_ID=local_judge2 \
    bash "${ROOT}/training/openrlhf/stop_local_judge.sh" >/dev/null 2>&1 || true
  PID_FILE="${LOG_DIR}/local_judge3_vllm.pid" JUDGE_PORT=8768 JUDGE_RUN_ID=local_judge3 \
    bash "${ROOT}/training/openrlhf/stop_local_judge.sh" >/dev/null 2>&1 || true
  PID_FILE="${LOG_DIR}/local_judge4_vllm.pid" JUDGE_PORT=8769 JUDGE_RUN_ID=local_judge4 \
    bash "${ROOT}/training/openrlhf/stop_local_judge.sh" >/dev/null 2>&1 || true
else
  echo "[stop] skip judge (SKIP_STOP_JUDGE=${SKIP_STOP_JUDGE:-1})"
fi

if [[ "${SKIP_STOP_REWARD:-1}" != "1" ]]; then
  stop_pid_file "${LOG_DIR}/physics_reward_server.pid" "reward_server"
  rm -f "${LOG_DIR}/physics_reward_server.mode"
else
  echo "[stop] skip reward server (SKIP_STOP_REWARD=${SKIP_STOP_REWARD:-1})"
fi

echo "[ok] stop_swift_training finished ckpt=${CKPT}"
