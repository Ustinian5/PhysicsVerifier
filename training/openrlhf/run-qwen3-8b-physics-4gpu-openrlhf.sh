#!/usr/bin/env bash
# GRPO training for Qwen3-8B physics RL via OpenRLHF on exactly 3 train GPUs
# (+ 1 external judge GPU managed separately).
#
# TRAIN_TOPOLOGY=colocate (default): 3 Actor + 3 vLLM TP1 Hybrid Engine
# TRAIN_TOPOLOGY=split: 2 Actor + 1 vLLM TP1 (OOM/IPC fallback)
#
# Uses an isolated Ray head (own GCS/dashboard/session). Never attaches to
# another user's :8265 cluster and never runs global `ray stop --force`.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
ENV_FILE="${ENV_FILE:-${WORKSPACE}/openrlhf_rl/env.sh}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

PYTHON="${PYTHON:-${TRAIN_VENV}/bin/python}"
TRAIN_STAGE="${TRAIN_STAGE:-verifier}"
if [[ "${TRAIN_STAGE}" == "bootstrap" ]]; then
  PROMPT_DATA="${PROMPT_DATA:-${ROOT}/data/rl/bootstrap_curriculum.jsonl}"
  export PHYSICS_REWARD_MODE="${PHYSICS_REWARD_MODE:-answer_only}"
  export PHYSICS_REWARD_W_FORMAT="${PHYSICS_REWARD_W_FORMAT:-0}"
else
  PROMPT_DATA="${PROMPT_DATA:-${ROOT}/data/rl/openrlhf_prompts.jsonl}"
fi
HELDOUT_DATA="${HELDOUT_DATA:-${ROOT}/data/rl/openrlhf_heldout.jsonl}"
REWARD_FUNC="${REWARD_FUNC:-${ROOT}/training/openrlhf/physics_reward_func.py}"
RM_URL="${RM_URL:-http://127.0.0.1:8770/get_reward}"
MODEL_PATH="${QWEN8B_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-8B}"
SAVE_PATH="${QWEN8B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf}"

curl -sf http://127.0.0.1:8770/health >/dev/null
export PHYSICS_REWARD_URL="${RM_URL}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTHONUNBUFFERED=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

IFS=',' read -ra _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-${#_GPU_ARR[@]}}"
VLLM_TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
TRAIN_TOPOLOGY="${TRAIN_TOPOLOGY:-colocate}"

case "${TRAIN_TOPOLOGY}" in
  colocate)
    ACTOR_GPUS="${ACTOR_GPUS:-3}"
    VLLM_ENGINES="${VLLM_ENGINES:-3}"
    VLLM_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
    COLLOCATE_ARGS=(--colocate_all_models --vllm_enable_sleep --deepspeed_enable_sleep)
    if [[ "${ACTOR_GPUS}" -ne $(( VLLM_ENGINES * VLLM_TP )) ]]; then
      echo "[error] colocate requires actor_gpus == engines*TP (got ${ACTOR_GPUS} vs $((VLLM_ENGINES*VLLM_TP)))" >&2
      exit 1
    fi
    if [[ "${ACTOR_GPUS}" -ne "${NUM_TRAIN_GPUS}" ]]; then
      echo "[error] colocate requires actor_gpus == num train GPUs (${ACTOR_GPUS} vs ${NUM_TRAIN_GPUS})" >&2
      exit 1
    fi
    ;;
  split)
    ACTOR_GPUS="${ACTOR_GPUS:-2}"
    VLLM_ENGINES="${VLLM_ENGINES:-1}"
    VLLM_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
    COLLOCATE_ARGS=()
    if [[ $(( ACTOR_GPUS + VLLM_ENGINES * VLLM_TP )) -ne "${NUM_TRAIN_GPUS}" ]]; then
      echo "[error] split requires ACTOR_GPUS + VLLM_ENGINES*TP == ${NUM_TRAIN_GPUS}" >&2
      exit 1
    fi
    ;;
  *)
    echo "[error] unknown TRAIN_TOPOLOGY=${TRAIN_TOPOLOGY} (use colocate|split)" >&2
    exit 1
    ;;
esac

mkdir -p "${SAVE_PATH}" "${SAVE_PATH}/ckpt" "${SAVE_PATH}/runs" "${SAVE_PATH}/plots" "${SAVE_PATH}/ray"

FLASH_ARGS=()
ATTN_IMPL="${OPENRLHF_ATTN_IMPL:-sdpa}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-2048}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
SAVE_STEPS="${SAVE_STEPS:-20}"
MICRO_ROLLOUT_BS="${MICRO_ROLLOUT_BATCH_SIZE:-1}"
FILTER_MIN="${DYNAMIC_FILTER_MIN:-0.0}"
FILTER_MAX="${DYNAMIC_FILTER_MAX:-1.0}"
FILTER_MODE="${DYNAMIC_FILTER_MODE:-reward_variance}"
FILTER_MIN_SPREAD="${DYNAMIC_FILTER_MIN_SPREAD:-1e-6}"
FILTER_MIN_STD="${DYNAMIC_FILTER_MIN_STD:-0.0}"
MAX_GEN_BATCHES="${DYNAMIC_FILTER_MAX_GEN_BATCHES:-0}"
MAX_CANDIDATE_SAMPLES="${DYNAMIC_FILTER_MAX_CANDIDATE_SAMPLES:-0}"
FILTER_BUDGET_ACTION="${DYNAMIC_FILTER_BUDGET_EXHAUSTED:-skip}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"
export PHYSICS_REWARD_METRICS_LOG="${PHYSICS_REWARD_METRICS_LOG:-${SAVE_PATH}/plots/physics_reward_metrics.jsonl}"
PILOT_MAX_STEPS="${PILOT_MAX_STEPS:-0}"
# Prefer direct launch on isolated Ray; Jobs API is optional / best-effort.
ALLOW_RAY_JOBS="${ALLOW_RAY_JOBS:-0}"
RAY_JOB_SUBMIT_ATTEMPTS="${RAY_JOB_SUBMIT_ATTEMPTS:-1}"
ALLOW_DIRECT_LAUNCH="${ALLOW_DIRECT_LAUNCH:-1}"

# Isolated Ray ports / session (do not collide with shared :8265 or default worker 10002-19999).
RAY_GCS_PORT="${RAY_GCS_PORT:-26379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-28265}"
RAY_CLIENT_PORT="${RAY_CLIENT_PORT:-26380}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-26381}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-27380}"
# AF_UNIX socket paths must be <=107 bytes; keep Ray tempdir short.
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/orhf8b_ray_${RAY_GCS_PORT}}"
RAY_HEAD_PID_FILE="${RAY_HEAD_PID_FILE:-${SAVE_PATH}/ray/ray_head.pid}"
RAY_ADDRESS_FILE="${RAY_ADDRESS_FILE:-${SAVE_PATH}/ray/ray_address.txt}"
mkdir -p "${RAY_TMPDIR}" "${SAVE_PATH}/ray"

LOAD_CKPT_ARGS=()
if [[ "${PILOT_MAX_STEPS}" -le 0 && -d "${SAVE_PATH}/ckpt/_actor" && -s "${SAVE_PATH}/ckpt/_actor/latest" ]]; then
  LOAD_CKPT_ARGS=(--load_checkpoint)
fi

EVAL_ARGS=()
if [[ -s "${HELDOUT_DATA}" && "${PILOT_MAX_STEPS}" -le 0 ]]; then
  EVAL_ARGS=(--eval_dataset "${HELDOUT_DATA}" --eval_n_samples_per_prompt 2 --eval_steps 10)
fi

MANIFEST="${SAVE_PATH}/run_manifest.json"
python3 - <<PY >"${MANIFEST}"
import json, os, datetime
print(json.dumps({
  "created_at": datetime.datetime.utcnow().isoformat() + "Z",
  "save_path": "${SAVE_PATH}",
  "model_path": "${MODEL_PATH}",
  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
  "train_topology": "${TRAIN_TOPOLOGY}",
  "actor_gpus": int("${ACTOR_GPUS}"),
  "vllm_engines": int("${VLLM_ENGINES}"),
  "vllm_tp": int("${VLLM_TP}"),
  "vllm_gpu_memory_utilization": float("${VLLM_MEM_UTIL}"),
  "generate_max_len": int("${GENERATE_MAX_LEN}"),
  "rollout_batch_size": int("${ROLLOUT_BATCH_SIZE}"),
  "train_batch_size": int("${TRAIN_BATCH_SIZE}"),
  "micro_rollout_batch_size": int("${MICRO_ROLLOUT_BS}"),
  "max_samples": int("${MAX_SAMPLES}"),
  "dynamic_filter_min": float("${FILTER_MIN}"),
  "dynamic_filter_max": float("${FILTER_MAX}"),
  "dynamic_filtering_mode": "${FILTER_MODE}",
  "dynamic_filtering_min_spread": float("${FILTER_MIN_SPREAD}"),
  "dynamic_filtering_max_gen_batches": int("${MAX_GEN_BATCHES}"),
  "n_samples_per_prompt": int("${N_SAMPLES_PER_PROMPT}"),
  "pilot_max_steps": int("${PILOT_MAX_STEPS}"),
  "train_stage": "${TRAIN_STAGE}",
  "prompt_data": "${PROMPT_DATA}",
  "reward_url": "${RM_URL}",
  "reward_mode": os.environ.get("PHYSICS_REWARD_MODE", ""),
  "ray_gcs_port": int("${RAY_GCS_PORT}"),
  "ray_dashboard_port": int("${RAY_DASHBOARD_PORT}"),
}, ensure_ascii=False, indent=2))
PY

export PHYSICS_REWARD_URL="${RM_URL}"
export PHYSICS_ROOT="${ROOT}"
export PYTHONPATH="${ROOT}:${OPENRLHF_ROOT:-}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export NCCL_CUMEM_ENABLE=0
export CUDA_HOME="${CUDA_HOME:-/slow_share/jinjianhan/workspace/openrlhf_rl/cuda_stub}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export OPENRLHF_USE_TORCH_ADAM=1
export OPENRLHF_ATTN_IMPL="${ATTN_IMPL}"
# vLLM sleep / CuMemAllocator (colocate) is incompatible with expandable_segments.
if [[ "${TRAIN_TOPOLOGY}" == "colocate" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}"
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi
ALLOC_CONF_JSON="${PYTORCH_CUDA_ALLOC_CONF:-}"

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
    \"CUDA_HOME\": \"${CUDA_HOME}\",
    \"PATH\": \"${PATH}\",
    \"OPENRLHF_USE_TORCH_ADAM\": \"1\",
    \"OPENRLHF_ATTN_IMPL\": \"${ATTN_IMPL}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${ALLOC_CONF_JSON}\",
    \"PHYSICS_REWARD_MODE\": \"${PHYSICS_REWARD_MODE:-}\",
    \"PHYSICS_REWARD_W_FORMAT\": \"${PHYSICS_REWARD_W_FORMAT:-}\",
    \"PHYSICS_REWARD_METRICS_LOG\": \"${PHYSICS_REWARD_METRICS_LOG:-}\"
  }
}"

TRAIN_ARGS=(
  -m openrlhf.cli.train_ppo_ray
  --ref_num_nodes 1
  --ref_num_gpus_per_node "${ACTOR_GPUS}"
  --actor_num_nodes 1
  --actor_num_gpus_per_node "${ACTOR_GPUS}"
  --vllm_num_engines "${VLLM_ENGINES}"
  --vllm_tensor_parallel_size "${VLLM_TP}"
  --vllm_gpu_memory_utilization "${VLLM_MEM_UTIL}"
  --vllm_sync_backend nccl
  --enforce_eager
  --pretrain "${MODEL_PATH}"
  --remote_rm_url "${REWARD_FUNC}"
  --save_path "${SAVE_PATH}"
  --ckpt_path "${SAVE_PATH}/ckpt"
  --max_ckpt_num 3
  --save_steps "${SAVE_STEPS}"
  --logging_steps 1
  --micro_train_batch_size 1
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --micro_rollout_batch_size "${MICRO_ROLLOUT_BS}"
  --rollout_batch_size "${ROLLOUT_BATCH_SIZE}"
  --n_samples_per_prompt "${N_SAMPLES_PER_PROMPT}"
  --max_epochs 1
  --num_episodes 1
  --prompt_max_len 2048
  --generate_max_len "${GENERATE_MAX_LEN}"
  --max_samples "${MAX_SAMPLES}"
  --zero_stage 3
  --bf16
  --actor_learning_rate 1e-6
  --init_kl_coef 0.0
  --use_kl_loss
  --kl_estimator k3
  --advantage_estimator group_norm
  --eps_clip 0.2
  --entropy_loss_coef 0.0
  --aux_loss_coef 0.0
  --gradient_checkpointing
  --adam_offload
  --prompt_data "${PROMPT_DATA}"
  --input_key input
  --label_key label
  --apply_chat_template
  --dynamic_filtering
  --dynamic_filtering_reward_range "${FILTER_MIN}" "${FILTER_MAX}"
  --dynamic_filtering_mode "${FILTER_MODE}"
  --dynamic_filtering_min_spread "${FILTER_MIN_SPREAD}"
  --dynamic_filtering_min_std "${FILTER_MIN_STD}"
  --dynamic_filtering_max_gen_batches "${MAX_GEN_BATCHES}"
  --dynamic_filtering_max_candidate_samples "${MAX_CANDIDATE_SAMPLES}"
  --dynamic_filtering_budget_exhausted "${FILTER_BUDGET_ACTION}"
  --use_tensorboard "${SAVE_PATH}/runs"
  "${COLLOCATE_ARGS[@]}"
  "${LOAD_CKPT_ARGS[@]}"
  "${EVAL_ARGS[@]}"
  "${FLASH_ARGS[@]}"
)

stop_own_ray() {
  local pid=""
  if [[ -f "${RAY_HEAD_PID_FILE}" ]]; then
    pid="$(cat "${RAY_HEAD_PID_FILE}" 2>/dev/null || true)"
  fi
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    # Stop only this head and its children; never global ray stop.
    pkill -TERM -P "${pid}" 2>/dev/null || true
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 3
    pkill -9 -P "${pid}" 2>/dev/null || true
    kill -9 "${pid}" 2>/dev/null || true
  fi
  # Also stop raylets bound to our GCS port if still present.
  while read -r rp; do
    [[ -z "${rp}" ]] && continue
    cmdline="$(tr '\0' ' ' <"/proc/${rp}/cmdline" 2>/dev/null || true)"
    if [[ "${cmdline}" == *"--port=${RAY_GCS_PORT}"* ]] || [[ "${cmdline}" == *"--port ${RAY_GCS_PORT}"* ]]; then
      kill -TERM "${rp}" 2>/dev/null || true
      sleep 1
      kill -9 "${rp}" 2>/dev/null || true
    fi
  done < <(pgrep -f "ray.*${RAY_GCS_PORT}" || true)
  rm -f "${RAY_HEAD_PID_FILE}" "${RAY_ADDRESS_FILE}"
}

start_isolated_ray_head() {
  stop_own_ray
  mkdir -p "${RAY_TMPDIR}"
  echo "[ray] starting isolated head gcs=${RAY_GCS_PORT} dashboard=${RAY_DASHBOARD_PORT}"
  # ray start --head forks a daemon; capture parent pid of the CLI invocation.
  RAY_TMPDIR="${RAY_TMPDIR}" nohup ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --port "${RAY_GCS_PORT}" \
    --dashboard-host=127.0.0.1 \
    --dashboard-port "${RAY_DASHBOARD_PORT}" \
    --ray-client-server-port "${RAY_CLIENT_PORT}" \
    --min-worker-port "${RAY_MIN_WORKER_PORT}" \
    --max-worker-port "${RAY_MAX_WORKER_PORT}" \
    --num-gpus "${NUM_TRAIN_GPUS}" \
    --disable-usage-stats \
    --temp-dir "${RAY_TMPDIR}" \
    >"${SAVE_PATH}/ray/ray_start.log" 2>&1 &
  local starter_pid=$!
  echo "${starter_pid}" >"${RAY_HEAD_PID_FILE}"
  # Prefer the long-lived gcs_server / raylet PID if available.
  for _ in $(seq 1 40); do
    if curl -sf "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/version" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if ! curl -sf "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/version" >/dev/null 2>&1; then
    echo "[error] isolated Ray dashboard not ready on :${RAY_DASHBOARD_PORT}" >&2
    tail -n 80 "${SAVE_PATH}/ray/ray_start.log" || true
    return 1
  fi
  # Record durable PIDs for precise cleanup.
  local gcs_pid
  gcs_pid="$(pgrep -f "gcs_server.*--port=${RAY_GCS_PORT}" | head -1 || true)"
  if [[ -n "${gcs_pid}" ]]; then
    echo "${gcs_pid}" >"${RAY_HEAD_PID_FILE}"
  fi
  printf '127.0.0.1:%s\n' "${RAY_GCS_PORT}" >"${RAY_ADDRESS_FILE}"
  export RAY_ADDRESS="127.0.0.1:${RAY_GCS_PORT}"
  echo "[ray] ready RAY_ADDRESS=${RAY_ADDRESS}"
}

launch_mode=""
JOB_ID=""
TRAIN_PID=""
SUBMIT_LOG="${SAVE_PATH}/ray_job_submit.log"
DIRECT_LOG="${SAVE_PATH}/direct_train.log"
STATUS_FILE="${SAVE_PATH}/launch_status.json"
: >"${SUBMIT_LOG}"

submit_ray_job() {
  ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" --no-wait \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${PYTHON}" "${TRAIN_ARGS[@]}"
}

start_isolated_ray_head

if [[ "${ALLOW_RAY_JOBS}" == "1" ]]; then
  for attempt in $(seq 1 "${RAY_JOB_SUBMIT_ATTEMPTS}"); do
    if submit_ray_job 2>&1 | tee -a "${SUBMIT_LOG}"; then
      JOB_ID="$(grep -oE 'raysubmit_[A-Za-z0-9]+' "${SUBMIT_LOG}" | tail -1 || true)"
      if [[ -n "${JOB_ID}" ]]; then
        launch_mode="ray_job"
        break
      fi
    fi
    echo "[warn] ray job submit attempt ${attempt}/${RAY_JOB_SUBMIT_ATTEMPTS} failed" | tee -a "${SUBMIT_LOG}"
    sleep 5
  done
fi

if [[ -z "${JOB_ID}" ]]; then
  if [[ "${ALLOW_DIRECT_LAUNCH}" != "1" ]]; then
    echo "[error] ray job submit failed and direct launch disabled" | tee -a "${SUBMIT_LOG}"
    stop_own_ray
    exit 1
  fi
  echo "[launch] direct train_ppo_ray on isolated GCS 127.0.0.1:${RAY_GCS_PORT}" | tee -a "${SUBMIT_LOG}"
  export RAY_ADDRESS="127.0.0.1:${RAY_GCS_PORT}"
  nohup env \
    RAY_ADDRESS="${RAY_ADDRESS}" \
    PHYSICS_REWARD_URL="${RM_URL}" \
    PHYSICS_ROOT="${ROOT}" \
    PYTHONPATH="${ROOT}:${OPENRLHF_ROOT:-}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
    VLLM_USE_V1=0 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    NCCL_CUMEM_ENABLE=0 \
    OPENRLHF_USE_TORCH_ADAM=1 \
    OPENRLHF_ATTN_IMPL="${ATTN_IMPL}" \
    PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF_JSON}" \
    PHYSICS_REWARD_MODE="${PHYSICS_REWARD_MODE:-}" \
    PHYSICS_REWARD_W_FORMAT="${PHYSICS_REWARD_W_FORMAT:-}" \
    PHYSICS_REWARD_METRICS_LOG="${PHYSICS_REWARD_METRICS_LOG:-}" \
    "${PYTHON}" "${TRAIN_ARGS[@]}" \
    >"${DIRECT_LOG}" 2>&1 &
  TRAIN_PID=$!
  echo "${TRAIN_PID}" >"${SAVE_PATH}/direct_train.pid"
  launch_mode="direct"
  sleep 15
  if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "[error] direct train_ppo_ray exited early; see ${DIRECT_LOG}" >&2
    tail -n 80 "${DIRECT_LOG}" || true
    stop_own_ray
    exit 1
  fi
  echo "[launch] direct train pid=${TRAIN_PID} topology=${TRAIN_TOPOLOGY} log=${DIRECT_LOG}"
else
  printf '%s\n' "${JOB_ID}" > "${SAVE_PATH}/ray_job_id.txt"
  echo "[launch] Ray job submitted: ${JOB_ID} topology=${TRAIN_TOPOLOGY}"
fi

python3 - <<PY >"${STATUS_FILE}"
import json, datetime
print(json.dumps({
  "created_at": datetime.datetime.utcnow().isoformat() + "Z",
  "launch_mode": "${launch_mode}",
  "train_topology": "${TRAIN_TOPOLOGY}",
  "job_id": "${JOB_ID}",
  "direct_pid": "${TRAIN_PID}",
  "ray_address": "127.0.0.1:${RAY_GCS_PORT}",
  "ray_dashboard": "http://127.0.0.1:${RAY_DASHBOARD_PORT}",
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
  "actor_gpus": int("${ACTOR_GPUS}"),
  "vllm_engines": int("${VLLM_ENGINES}"),
  "train_stage": "${TRAIN_STAGE}",
  "save_path": "${SAVE_PATH}",
  "submit_log": "${SUBMIT_LOG}",
  "direct_log": "${DIRECT_LOG}",
}, ensure_ascii=False, indent=2))
PY

monitor_reached_step() {
  local target="$1"
  local metrics_csv="${SAVE_PATH}/plots/training_metrics.csv"
  local step_reached=0
  local last_step=0
  if [[ -s "${metrics_csv}" ]]; then
    last_step="$(awk -F, 'NR>1 {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); s=$1} END {print s+0}' "${metrics_csv}")"
    if [[ "${last_step}" -ge "${target}" ]]; then
      step_reached=1
    fi
  fi
  if [[ "${step_reached}" -eq 0 ]]; then
    if [[ -n "${JOB_ID}" ]] && RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}" ray job logs "${JOB_ID}" 2>/dev/null | grep -qE "Global step[[:space:]]+${target}:"; then
      step_reached=1
    elif [[ -s "${DIRECT_LOG}" ]] && grep -qE "Global step[[:space:]]+${target}:" "${DIRECT_LOG}"; then
      step_reached=1
    fi
  fi
  echo "${step_reached}"
}

detect_colocate_failure() {
  # Returns 0 if logs show OOM / CUDA IPC / sleep errors warranting split fallback.
  local log="${DIRECT_LOG}"
  [[ -s "${log}" ]] || return 1
  grep -qiE 'CUDA out of memory|OutOfMemoryError|cuda ipc|CUDAIPC|enable_sleep_mode|Failed to allocate|NCCL error|ncclUnhandledCudaError' "${log}"
}

stop_training() {
  if [[ -n "${JOB_ID}" ]]; then
    RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}" ray job stop "${JOB_ID}" || true
  fi
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    sleep 8
    kill -9 "${TRAIN_PID}" 2>/dev/null || true
  fi
  # Precise: only kill our direct train pid tree, not arbitrary train_ppo_ray.
  if [[ -f "${SAVE_PATH}/direct_train.pid" ]]; then
    local dpid
    dpid="$(cat "${SAVE_PATH}/direct_train.pid" 2>/dev/null || true)"
    if [[ -n "${dpid}" ]] && kill -0 "${dpid}" 2>/dev/null; then
      pkill -TERM -P "${dpid}" 2>/dev/null || true
      kill -TERM "${dpid}" 2>/dev/null || true
      sleep 3
      pkill -9 -P "${dpid}" 2>/dev/null || true
      kill -9 "${dpid}" 2>/dev/null || true
    fi
  fi
  stop_own_ray
}

cleanup_on_exit() {
  # Keep Ray alive during pilot monitoring; only stop when this script exits after stop_training.
  :
}
trap cleanup_on_exit EXIT

if [[ "${PILOT_MAX_STEPS}" -gt 0 ]]; then
  echo "[pilot] waiting for global step ${PILOT_MAX_STEPS} (mode=${launch_mode} topology=${TRAIN_TOPOLOGY})"
  for _ in $(seq 1 2880); do
    if [[ "$(monitor_reached_step "${PILOT_MAX_STEPS}")" -eq 1 ]]; then
      echo "[pilot] reached step ${PILOT_MAX_STEPS}; stopping job"
      stop_training
      sleep 10
      break
    fi
    if [[ "${launch_mode}" == "direct" && -n "${TRAIN_PID}" ]] && ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
      if [[ "$(monitor_reached_step "${PILOT_MAX_STEPS}")" -eq 1 ]]; then
        echo "[pilot] process exited after reaching target"
        stop_own_ray
        break
      fi
      if [[ "${TRAIN_TOPOLOGY}" == "colocate" ]] && detect_colocate_failure; then
        echo "[warn] colocate failure detected; signaling fallback to split" | tee -a "${SUBMIT_LOG}"
        python3 - <<PY >"${SAVE_PATH}/fallback_request.json"
import json, datetime
print(json.dumps({
  "requested_topology": "split",
  "reason": "colocate_oom_or_ipc",
  "at": datetime.datetime.utcnow().isoformat() + "Z",
  "direct_log": "${DIRECT_LOG}",
}, ensure_ascii=False, indent=2))
PY
        stop_own_ray
        exit 42
      fi
      echo "[error] direct training process exited before reaching step ${PILOT_MAX_STEPS}" >&2
      tail -n 120 "${DIRECT_LOG}" || true
      stop_own_ray
      exit 1
    fi
    sleep 30
  done
fi
