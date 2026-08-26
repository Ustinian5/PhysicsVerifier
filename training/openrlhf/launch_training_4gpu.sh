#!/usr/bin/env bash
# Launch 4-GPU Qwen3-8B OpenRLHF GRPO (optional local judge on a spare GPU).
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ -z "${PHYSICSVERIFIER_OPENAI_BASE_URL:-}" && -z "${PHYSICS_REWARD_MODE:-}" ]]; then
  # Default for production 4-GPU path remains answer_only unless caller overrides.
  export PHYSICS_REWARD_MODE="answer_only"
fi

bash "${ROOT}/training/openrlhf/download_qwen3_8b.sh"
bash "${ROOT}/training/openrlhf/start_local_judge_if_needed.sh"
bash "${ROOT}/training/reward_server/start_reward_server.sh"
bash "${ROOT}/training/openrlhf/check_prerequisites_4gpu.sh"
bash "${ROOT}/training/openrlhf/watch_training_curves.sh" start

export QWEN8B_RL_CKPT="${QWEN8B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf}"
export PLOT_OUT_DIR="${PLOT_OUT_DIR:-${QWEN8B_RL_CKPT}/plots}"
bash "${ROOT}/training/openrlhf/run-qwen3-8b-physics-4gpu-openrlhf.sh"
