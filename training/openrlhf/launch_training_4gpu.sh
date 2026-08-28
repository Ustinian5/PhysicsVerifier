#!/usr/bin/env bash
# Launch 3-train + 1-judge Qwen3-8B OpenRLHF GRPO with paragraph process reward.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
if [[ -f "${ROOT}/training/openrlhf/paragraph_process_defaults.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/training/openrlhf/paragraph_process_defaults.env"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export JUDGE_CUDA_DEVICE="${JUDGE_CUDA_DEVICE:-3}"
export QWEN8B_RL_CKPT="${QWEN8B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf}"
export PLOT_OUT_DIR="${PLOT_OUT_DIR:-${QWEN8B_RL_CKPT}/plots}"
export PHYSICS_REWARD_METRICS_LOG="${PHYSICS_REWARD_METRICS_LOG:-${PLOT_OUT_DIR}/physics_reward_metrics.jsonl}"
export TRAIN_STAGE="${TRAIN_STAGE:-bootstrap}"
export TRAIN_TOPOLOGY="${TRAIN_TOPOLOGY:-colocate}"
export ACTOR_GPUS="${ACTOR_GPUS:-3}"
export VLLM_ENGINES="${VLLM_ENGINES:-3}"
export PHYSICS_REWARD_MODE="${PHYSICS_REWARD_MODE:-process_paragraph}"
export PHYSICS_REWARD_VERIFIER_ON_WRONG="${PHYSICS_REWARD_VERIFIER_ON_WRONG:-1}"
export PHYSICS_REWARD_W_ANSWER=0
export PHYSICS_REWARD_W_FORMAT=0
export GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-512}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_BIND_IP="${RAY_BIND_IP:-127.0.0.1}"
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER="${RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER:-0}"
export ALLOW_RAY_JOBS="${ALLOW_RAY_JOBS:-0}"
export ALLOW_DIRECT_LAUNCH="${ALLOW_DIRECT_LAUNCH:-1}"
export PHYSICSVERIFIER_UNIFIED_RULE_TOP_N="${PHYSICSVERIFIER_UNIFIED_RULE_TOP_N:-4}"
export RAY_GCS_PORT="${RAY_GCS_PORT:-26379}"
export MIN_SLOW_TMP_GB="${MIN_SLOW_TMP_GB:-20}"
# shellcheck disable=SC1091
source "${ROOT}/training/openrlhf/setup_slow_share_tmp.sh"

bash "${ROOT}/training/openrlhf/download_qwen3_8b.sh"
bash "${ROOT}/training/openrlhf/start_local_judge_if_needed.sh"
bash "${ROOT}/training/reward_server/start_reward_server.sh"
bash "${ROOT}/training/openrlhf/check_prerequisites_4gpu.sh"
bash "${ROOT}/training/openrlhf/watch_training_curves.sh" start
bash "${ROOT}/training/openrlhf/run-qwen3-8b-physics-4gpu-openrlhf.sh"
