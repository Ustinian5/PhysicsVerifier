#!/usr/bin/env bash
# Compatibility wrapper: delegates to adaptive_four_gpu_watchdog.sh.
# Legacy TRAIN_GPUS/FREE_MIB env vars are accepted but adaptive selection is preferred.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
echo "[compat] wait_gpus_and_launch_pilot.sh -> adaptive_four_gpu_watchdog.sh"
if [[ -n "${TRAIN_GPUS:-}" ]]; then
  echo "[compat] ignoring fixed TRAIN_GPUS=${TRAIN_GPUS}; adaptive watchdog picks any 4 idle GPUs"
fi
if [[ -n "${FREE_MIB:-}" ]]; then
  export FREE_MIB
fi
exec bash "${ROOT}/training/openrlhf/adaptive_four_gpu_watchdog.sh"
