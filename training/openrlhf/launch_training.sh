#!/usr/bin/env bash
# Launch OpenRLHF GRPO after prerequisite checks.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"

# Default: 4-GPU Qwen3-8B + external API reward (see launch_training_4gpu.sh).
if [[ "${OPENRLHF_TRAINING_MODE:-4gpu}" == "4gpu" ]]; then
  exec bash "${ROOT}/training/openrlhf/launch_training_4gpu.sh" "$@"
fi

bash "${ROOT}/training/openrlhf/check_prerequisites.sh"

# Ensure OpenRLHF-format data exists
if [[ ! -s "${ROOT}/data/rl/openrlhf_prompts.jsonl" ]]; then
  bash "${ROOT}/training/openrlhf/prepare_openrlhf_data.sh"
fi
if [[ -s "${ROOT}/data/rl/heldout_eval.jsonl" && ! -s "${ROOT}/data/rl/openrlhf_heldout.jsonl" ]]; then
  bash "${ROOT}/training/openrlhf/prepare_openrlhf_data.sh" \
    "${ROOT}/data/rl/heldout_eval.jsonl" \
    "${ROOT}/data/rl/openrlhf_heldout.jsonl"
fi

# Reward server and curve watcher are nohup daemons (survive SSH disconnect).
bash "${ROOT}/training/reward_server/start_reward_server.sh"
bash "${ROOT}/training/openrlhf/watch_training_curves.sh" start

echo "[launch] Starting OpenRLHF GRPO training..."
# Use GPUs 0-5 for training; GPU6 reserved for PhysicsVerifier judge.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
bash "${ROOT}/training/openrlhf/run-qwen3-30b-physics-6gpu-openrlhf.sh"
