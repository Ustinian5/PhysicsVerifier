#!/usr/bin/env bash
# Launch 4-train + 4-judge Qwen3-8B OpenRLHF GRPO with paragraph process reward.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
if [[ -f "${ROOT}/training/openrlhf/paragraph_process_defaults.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/training/openrlhf/paragraph_process_defaults.env"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export JUDGE_CUDA_DEVICE="${JUDGE_CUDA_DEVICE:-4}"
export TRAIN_STAGE="${TRAIN_STAGE:-full}"
export TRAIN_TOPOLOGY="${TRAIN_TOPOLOGY:-colocate}"
export ACTOR_GPUS="${ACTOR_GPUS:-4}"
export VLLM_ENGINES="${VLLM_ENGINES:-4}"
export PHYSICS_REWARD_MODE="${PHYSICS_REWARD_MODE:-process_paragraph}"
export PHYSICS_REWARD_VERIFIER_ON_WRONG="${PHYSICS_REWARD_VERIFIER_ON_WRONG:-1}"
export PHYSICS_REWARD_W_ANSWER=0
export PHYSICS_REWARD_W_FORMAT=0
export PILOT_MAX_STEPS=0
export MAX_SAMPLES="${MAX_SAMPLES:-100000}"
export SAVE_STEPS="${SAVE_STEPS:-20}"
# Force 8-GPU numbers after sourcing paragraph_process_defaults.env (4-GPU values).
# ${VAR:-default} cannot override a value already exported by that file.
export GENERATE_MAX_LEN=512
export ROLLOUT_BATCH_SIZE=4
export N_SAMPLES_PER_PROMPT=6
export TRAIN_BATCH_SIZE=24
export PHYSICS_REWARD_CONCURRENCY=24
export PHYSICS_REWARD_MAX_RESPONSE_CHARS=2048
export PHYSICSVERIFIER_UNIFIED_RULE_TOP_N=2
export PHYSICS_REWARD_TIMEOUT=3600
export ENABLE_EVAL=0
export EVAL_STEPS="${EVAL_STEPS:-50}"
export EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"
export PROMPT_DATA="${PROMPT_DATA:-${ROOT}/data/rl/openrlhf_prompts.jsonl}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_BIND_IP="${RAY_BIND_IP:-127.0.0.1}"
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER="${RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER:-0}"
export ALLOW_RAY_JOBS="${ALLOW_RAY_JOBS:-0}"
export ALLOW_DIRECT_LAUNCH="${ALLOW_DIRECT_LAUNCH:-1}"
export PHYSICSVERIFIER_PRECISION_MODE="${PHYSICSVERIFIER_PRECISION_MODE:-balanced}"
export PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE="${PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE:-lexical}"
export RAY_GCS_PORT="${RAY_GCS_PORT:-26379}"
export MIN_SLOW_TMP_GB="${MIN_SLOW_TMP_GB:-20}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8765/v1}"
# shellcheck disable=SC1091
source "${ROOT}/training/openrlhf/setup_slow_share_tmp.sh"

bash "${ROOT}/training/openrlhf/download_qwen3_8b.sh"
bash "${ROOT}/training/openrlhf/start_local_judge_if_needed.sh"
bash "${ROOT}/training/reward_server/start_reward_server.sh"
bash "${ROOT}/training/openrlhf/check_prerequisites_8gpu.sh"
bash "${ROOT}/training/openrlhf/watch_training_curves.sh" start

export QWEN8B_RL_CKPT="${QWEN8B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf}"
export PLOT_OUT_DIR="${PLOT_OUT_DIR:-${QWEN8B_RL_CKPT}/plots}"
bash "${ROOT}/training/openrlhf/run-qwen3-8b-physics-4gpu-openrlhf.sh"
