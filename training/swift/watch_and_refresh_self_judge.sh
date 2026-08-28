#!/usr/bin/env bash
# Poll for new GRPO checkpoints and rolling-refresh 8B self-judge replicas.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
CKPT="${QWEN8B_SWIFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-swift}"
STAMP="${CKPT}/.self_judge_loaded"
POLL_SECS="${JUDGE_REFRESH_POLL_SECS:-60}"
TRAIN_PID_FILE="${LOG_DIR:-${ROOT}/logs}/swift_grpo.pid"

latest_ckpt() {
  ls -dt "${CKPT}"/v*-*/checkpoint-* 2>/dev/null | head -1 || true
}

while true; do
  if [[ -f "${TRAIN_PID_FILE}" ]]; then
    tpid="$(cat "${TRAIN_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${tpid}" ]] && ! kill -0 "${tpid}" 2>/dev/null; then
      echo "[watch] train pid ${tpid} gone; exit"
      exit 0
    fi
  fi
  latest="$(latest_ckpt)"
  if [[ -n "${latest}" && -f "${latest}/config.json" ]]; then
    prev="$(cat "${STAMP}" 2>/dev/null || true)"
    if [[ "${latest}" != "${prev}" ]]; then
      echo "[watch] refreshing judges to ${latest}"
      JUDGE_MODEL_DIR="${latest}" bash "${ROOT}/training/swift/refresh_self_judge.sh" "${latest}" \
        && echo "${latest}" >"${STAMP}" \
        || echo "[watch] refresh failed for ${latest}"
    fi
  fi
  sleep "${POLL_SECS}"
done
