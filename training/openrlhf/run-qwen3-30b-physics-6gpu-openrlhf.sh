#!/usr/bin/env bash
# GRPO training for Qwen3-30B-A3B physics RL via OpenRLHF 0.8.x (6 GPUs).
# GPU6 reserved for PhysicsVerifier vLLM judge; GPU7 optional spare.
#
# Training goals (unchanged from slime plan):
#   - GRPO (advantage_estimator=group_norm)
#   - n_samples_per_prompt=8
#   - remote PhysicsVerifier reward
#   - lr=1e-6, adam offload, bf16, ZeRO-3
#   - response max len 4096 (override with GENERATE_MAX_LEN=8192 if needed)
set -ex

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
ENV_FILE="${ENV_FILE:-${WORKSPACE}/openrlhf_rl/env.sh}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
else
  echo "[error] Missing ${ENV_FILE}; run training/openrlhf/setup_openrlhf_env.sh first" >&2
  exit 1
fi

PYTHON="${PYTHON:-${TRAIN_VENV}/bin/python}"
PROMPT_DATA="${PROMPT_DATA:-${ROOT}/data/rl/openrlhf_prompts.jsonl}"
HELDOUT_DATA="${HELDOUT_DATA:-${ROOT}/data/rl/openrlhf_heldout.jsonl}"
REWARD_FUNC="${REWARD_FUNC:-${ROOT}/training/openrlhf/physics_reward_func.py}"
RM_URL="${RM_URL:-http://127.0.0.1:8770/get_reward}"
MODEL_PATH="${QWEN30B_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-30B-A3B-Instruct-2507}"
SAVE_PATH="${QWEN30B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-30b-physics-openrlhf}"

# Prepare OpenRLHF-format data if missing
if [[ ! -s "${PROMPT_DATA}" ]]; then
  bash "${ROOT}/training/openrlhf/prepare_openrlhf_data.sh" \
    "${ROOT}/data/rl/rl_prompts.jsonl" "${PROMPT_DATA}"
fi
if [[ -s "${ROOT}/data/rl/heldout_eval.jsonl" && ! -s "${HELDOUT_DATA}" ]]; then
  bash "${ROOT}/training/openrlhf/prepare_openrlhf_data.sh" \
    "${ROOT}/data/rl/heldout_eval.jsonl" "${HELDOUT_DATA}"
fi

# Health checks
curl -sf http://127.0.0.1:8770/health >/dev/null
curl -sf http://127.0.0.1:8766/v1/models >/dev/null
export PHYSICS_REWARD_URL="${RM_URL}"

# Stop stale Ray / vLLM engines from previous OpenRLHF runs (do NOT kill GPU6 judge)
ray stop --force || true
pkill -9 -f 'openrlhf.cli.train_ppo_ray' || true
sleep 2

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export PYTHONUNBUFFERED=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# Derive GPU layout. TP>=2 NCCL is broken on this host (driver 550); default TP=1
# and split GPUs: half actor ZeRO-3, half dedicated vLLM engines (non-colocate).
IFS=',' read -ra _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-${#_GPU_ARR[@]}}"
VLLM_TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
COLLOCATE_ARGS=(--colocate_all_models --vllm_enable_sleep --deepspeed_enable_sleep)
VLLM_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
ACTOR_GPUS="${NUM_TRAIN_GPUS}"
VLLM_ENGINES="${VLLM_ENGINES:-}"

if [[ "${VLLM_TP}" -eq 1 ]]; then
  ACTOR_GPUS=$(( NUM_TRAIN_GPUS / 2 ))
  VLLM_ENGINES="${VLLM_ENGINES:-${ACTOR_GPUS}}"
  COLLOCATE_ARGS=()
  VLLM_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
elif [[ -z "${VLLM_ENGINES}" ]]; then
  VLLM_ENGINES=$(( NUM_TRAIN_GPUS / VLLM_TP ))
fi

if [[ "${VLLM_ENGINES}" -lt 1 ]]; then
  echo "[error] Invalid VLLM_ENGINES=${VLLM_ENGINES}" >&2
  exit 1
fi
if [[ "${COLLOCATE_ARGS[*]}" == *colocate_all_models* ]]; then
  if [[ $(( VLLM_ENGINES * VLLM_TP )) -ne "${NUM_TRAIN_GPUS}" ]]; then
    echo "[error] colocate mode needs NUM_TRAIN_GPUS == VLLM_ENGINES * VLLM_TP" >&2
    exit 1
  fi
else
  if [[ $(( ACTOR_GPUS + VLLM_ENGINES * VLLM_TP )) -ne "${NUM_TRAIN_GPUS}" ]]; then
    echo "[error] split mode needs ACTOR_GPUS + VLLM_ENGINES*VLLM_TP == NUM_TRAIN_GPUS; got ${ACTOR_GPUS}+${VLLM_ENGINES}*${VLLM_TP} vs ${NUM_TRAIN_GPUS}" >&2
    exit 1
  fi
fi
echo "[launch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS} ACTOR_GPUS=${ACTOR_GPUS} VLLM_ENGINES=${VLLM_ENGINES} VLLM_TP=${VLLM_TP} VLLM_MEM=${VLLM_MEM_UTIL}"

mkdir -p "${SAVE_PATH}" "${SAVE_PATH}/ckpt" "${SAVE_PATH}/runs"

# Attention backend:
# - Prefer real flash-attn CUDA package when present (--flash_attn + packing).
# - Otherwise use PyTorch SDPA (flash_sdp / mem_efficient) via OPENRLHF_ATTN_IMPL=sdpa.
#   Host has no nvcc; only a bert_padding shim is installed, which previously forced eager
#   attention and OOM'd on long PPO forwards.
FLASH_ARGS=()
ATTN_IMPL="${OPENRLHF_ATTN_IMPL:-sdpa}"
if "${PYTHON}" -c "from flash_attn.flash_attn_interface import flash_attn_func" 2>/dev/null; then
  FLASH_ARGS=(--flash_attn --packing_samples)
  ATTN_IMPL="flash_attention_2"
  echo "[launch] Using real flash-attn CUDA ops"
else
  echo "[warn] Real flash-attn unavailable; using PyTorch ${ATTN_IMPL} (flash_sdp preferred)"
fi

GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-4096}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-12}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-96}"
SAVE_STEPS="${SAVE_STEPS:-20}"
MICRO_ROLLOUT_BS="${MICRO_ROLLOUT_BATCH_SIZE:-1}"

# Auto-resume when a prior actor checkpoint exists.
LOAD_CKPT_ARGS=()
if [[ -d "${SAVE_PATH}/ckpt/_actor" ]]; then
  if [[ -s "${SAVE_PATH}/ckpt/_actor/latest" ]]; then
    LATEST_TAG="$(tr -d '[:space:]' < "${SAVE_PATH}/ckpt/_actor/latest")"
    echo "[launch] Resume from ${LATEST_TAG} (latest pointer)"
  else
    LATEST_STEP="$(find "${SAVE_PATH}/ckpt/_actor" -maxdepth 1 -type d -name 'global_step*' \
      | sed 's|.*/global_step||' | sort -n | tail -1)"
    LATEST_TAG="global_step${LATEST_STEP}"
    printf '%s' "${LATEST_TAG}" > "${SAVE_PATH}/ckpt/_actor/latest"
    echo "[launch] Resume from ${LATEST_TAG}"
  fi
  LOAD_CKPT_ARGS=(--load_checkpoint)
  echo "[launch] Found ${SAVE_PATH}/ckpt/_actor; enabling --load_checkpoint"
fi

# Optional held-out eval (requires remote_rm_url, already set)
EVAL_ARGS=()
if [[ -s "${HELDOUT_DATA}" ]]; then
  EVAL_ARGS=(--eval_dataset "${HELDOUT_DATA}" --eval_n_samples_per_prompt 4)
fi

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_TRAIN_GPUS}" --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port=8265

# Ray dashboard may lag gcs_server; wait before job submit.
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:8265/api/version" >/dev/null; then
    break
  fi
  sleep 2
done
if ! curl -sf "http://127.0.0.1:8265/api/version" >/dev/null; then
  echo "[error] Ray dashboard not ready at http://127.0.0.1:8265" >&2
  exit 1
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PHYSICS_REWARD_URL\": \"${RM_URL}\",
    \"PHYSICS_ROOT\": \"${ROOT}\",
    \"PYTHONPATH\": \"${ROOT}:${OPENRLHF_ROOT:-}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES\": \"1\",
    \"VLLM_USE_V1\": \"0\",
    \"VLLM_ENABLE_V1_MULTIPROCESSING\": \"0\",
    \"NCCL_CUMEM_ENABLE\": \"0\",
    \"CUDA_HOME\": \"${CUDA_HOME:-/slow_share/jinjianhan/workspace/openrlhf_rl/cuda_stub}\",
    \"PATH\": \"${CUDA_HOME:-/slow_share/jinjianhan/workspace/openrlhf_rl/cuda_stub}/bin:${PATH}\",
    \"PYDEV_INCLUDE\": \"${WORKSPACE_ROOT}/openrlhf_rl/pydev/include/python3.10\",
    \"CPATH\": \"${WORKSPACE_ROOT}/openrlhf_rl/pydev/include/python3.10\",
    \"CPLUS_INCLUDE_PATH\": \"${WORKSPACE_ROOT}/openrlhf_rl/pydev/include/python3.10\",
    \"OPENRLHF_USE_TORCH_ADAM\": \"1\",
    \"OPENRLHF_ATTN_IMPL\": \"${ATTN_IMPL}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"TORCH_NCCL_TRACE_BUFFER_SIZE\": \"16777216\",
    \"TORCH_NCCL_DUMP_ON_TIMEOUT\": \"1\"
  }
}"

echo "[launch] ATTN_IMPL=${ATTN_IMPL} GENERATE_MAX_LEN=${GENERATE_MAX_LEN} ROLLOUT_BS=${ROLLOUT_BATCH_SIZE} TRAIN_BS=${TRAIN_BATCH_SIZE} SAVE_STEPS=${SAVE_STEPS} MICRO_ROLLOUT_BS=${MICRO_ROLLOUT_BS} adam_offload=on save_hf_ckpt=off"

# GPU layout: split 3 actor + 3 vLLM (TP1) when NCCL blocks TP2; else hybrid colocate.
# --no-wait: submit and return immediately so SSH/Cursor disconnect does not block the job.
SUBMIT_LOG="${SAVE_PATH}/ray_job_submit.log"
ray job submit --address="http://127.0.0.1:8265" --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${PYTHON}" -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node "${ACTOR_GPUS}" \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node "${ACTOR_GPUS}" \
  --vllm_num_engines "${VLLM_ENGINES}" \
  --vllm_tensor_parallel_size "${VLLM_TP}" \
  "${COLLOCATE_ARGS[@]}" \
  --vllm_gpu_memory_utilization "${VLLM_MEM_UTIL}" \
  --vllm_sync_backend nccl \
  --enforce_eager \
  --pretrain "${MODEL_PATH}" \
  --remote_rm_url "${REWARD_FUNC}" \
  --save_path "${SAVE_PATH}" \
  --ckpt_path "${SAVE_PATH}/ckpt" \
  --max_ckpt_num 3 \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps 1 \
  --eval_steps 20 \
  --micro_train_batch_size 1 \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --micro_rollout_batch_size "${MICRO_ROLLOUT_BS}" \
  --rollout_batch_size "${ROLLOUT_BATCH_SIZE}" \
  --n_samples_per_prompt 8 \
  --max_epochs 1 \
  --num_episodes 1 \
  --prompt_max_len 2048 \
  --generate_max_len "${GENERATE_MAX_LEN}" \
  --max_samples 100000 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 1e-6 \
  --init_kl_coef 0.0 \
  --use_kl_loss \
  --kl_estimator k3 \
  --advantage_estimator group_norm \
  --eps_clip 0.2 \
  --entropy_loss_coef 0.0 \
  --aux_loss_coef 0.001 \
  --gradient_checkpointing \
  --adam_offload \
  --prompt_data "${PROMPT_DATA}" \
  --input_key input \
  --label_key label \
  --apply_chat_template \
  --dynamic_filtering \
  --dynamic_filtering_reward_range 0.01 0.99 \
  --use_tensorboard "${SAVE_PATH}/runs" \
  "${LOAD_CKPT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${FLASH_ARGS[@]}" \
  2>&1 | tee "${SUBMIT_LOG}"
JOB_ID="$(grep -oE 'raysubmit_[A-Za-z0-9]+' "${SUBMIT_LOG}" | tail -1)"
printf '%s\n' "${JOB_ID}" > "${SAVE_PATH}/ray_job_id.txt"
echo "[launch] Ray job submitted: ${JOB_ID} (disconnect-safe; monitor via train_launch.log or 'ray job logs ${JOB_ID}')"
