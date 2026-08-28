#!/usr/bin/env bash
# Launch 4-train + 4-judge Qwen3-8B GRPO with ms-swift.
# Reuses PhysicsVerifier judges / LB / reward server. Does not touch OpenRLHF ckpts.
set -euo pipefail
# Root /tmp is often 100% full on this node; never put bash heredocs or train logs there.
ulimit -f unlimited 2>/dev/null || true
SLOW_TMP_ROOT="${SLOW_TMP_ROOT:-/slow_share/jinjianhan/tmp}"
export TMPDIR="${TMPDIR:-${SLOW_TMP_ROOT}/swift}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
mkdir -p "${TMPDIR}"

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
# An empty CUDA_VISIBLE_DEVICES (e.g. leftover from --help) makes torch.cuda fail.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  unset CUDA_VISIBLE_DEVICES
fi
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
CKPT="${QWEN8B_SWIFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-swift}"
SWIFT_VENV="${SWIFT_VENV:-/data1/jinjianhan/venv/swift_train}"
ORHF_PYTHON="${ORHF_PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
SWIFT_PYTHON="${SWIFT_PYTHON:-${SWIFT_VENV}/bin/python}"
PID_FILE="${PID_FILE:-${LOG_DIR}/swift_grpo.pid}"
# Train stdout on slow_share: a full root disk previously truncated this log at 40KiB and killed the job.
LOG_FILE="${LOG_FILE:-${CKPT}/swift_grpo.log}"
REPORT="${REPORT:-${CKPT}/swift_launch_report.json}"
MODEL_DIR="${QWEN8B_MODEL_DIR:-}"
SFT_CKPT="${QWEN8B_SFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-sft}"
if [[ -z "${MODEL_DIR}" ]]; then
  if [[ -f "${SFT_CKPT}/config.json" ]]; then
    MODEL_DIR="${SFT_CKPT}"
  else
    MODEL_DIR="$(ls -d "${SFT_CKPT}"/v*-*/checkpoint-* 2>/dev/null | tail -1 || true)"
  fi
  if [[ -z "${MODEL_DIR}" || ! -f "${MODEL_DIR}/config.json" ]]; then
    MODEL_DIR="/slow_share/jinjianhan/models/Qwen3-8B"
  fi
fi
PROMPT_DATA="${PROMPT_DATA:-${ROOT}/data/rl/swift_prompts_max2048.jsonl}"
PLUGIN="${PLUGIN:-${ROOT}/training/swift/physics_reward_plugin.py}"
FREE_MIB="${FREE_MIB:-75000}"
UTIL_MAX="${UTIL_MAX:-5}"
PREFER_JUDGE="${PREFER_JUDGE:-4,7,6,5}"
JUDGE_LB_PORT="${JUDGE_LB_PORT:-8765}"
JUDGE_PORTS=(8766 8767 8768 8769)
JUDGE_RUN_IDS=(local_judge local_judge2 local_judge3 local_judge4)
JUDGE_MODEL_DIR="${JUDGE_MODEL_DIR:-${MODEL_DIR}}"
if [[ -z "${JUDGE_SERVED_NAME:-}" ]]; then
  if [[ "${JUDGE_MODEL_DIR}" == *"30B"* ]]; then
    JUDGE_SERVED_NAME="qwen3-30b-a3b"
  else
    JUDGE_SERVED_NAME="qwen3-8b-self-judge"
  fi
fi
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.45}"
JUDGE_REFRESH="${JUDGE_REFRESH:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
NUM_GENERATIONS="${NUM_GENERATIONS:-6}"
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-2}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-1536}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
SAVE_STEPS="${SAVE_STEPS:-20}"
NPROC="${NPROC_PER_NODE:-4}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.20}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
MAX_RESAMPLE_TIMES="${MAX_RESAMPLE_TIMES:-2}"
RESUME_FROM="${RESUME_FROM:-}"

mkdir -p "${LOG_DIR}" "${CKPT}/plots" "${CKPT}/runs"

refuse() {
  local reason="$1"
  mkdir -p "$(dirname "${REPORT}")"
  python3 -c 'import json,datetime,os,sys; print(json.dumps({"ok":False,"phase":"refused","reason":sys.argv[1],"at":datetime.datetime.utcnow().isoformat()+"Z","ckpt":os.environ.get("CKPT","")},ensure_ascii=False,indent=2))' "${reason}" >"${REPORT}"
  echo "[refuse] ${reason}" >&2
  exit 2
}

alive_pid_file() {
  local file="$1"
  [[ -f "${file}" ]] || return 1
  local pid
  pid="$(cat "${file}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

check_loopback_ports() {
  python3 - <<'PY'
import re, sys
raw = sys.stdin.read()
ports = {8765, 8766, 8767, 8768, 8769, 8770}
bad = []
for line in raw.splitlines():
    if "LISTEN" not in line and not line.strip().startswith("tcp"):
        # ss -ltn still has LISTEN
        pass
    toks = [t for t in line.split() if ":" in t]
    if not toks:
        continue
    local = toks[0]
    if local.startswith("[") and "]:" in local:
        host, _, port_s = local[1:].partition("]:")
    else:
        host, _, port_s = local.rpartition(":")
    if not port_s.isdigit():
        continue
    port = int(port_s)
    if port not in ports:
        continue
    host = host.strip() or "*"
    if host.startswith("::ffff:"):
        host = host.split("::ffff:")[-1]
    if host in {"*", "::", "[::]"}:
        host = "0.0.0.0"
    if host not in {"127.0.0.1", "::1"}:
        bad.append(f"{host}:{port}")
if bad:
    print("NON_LOOPBACK " + " ".join(bad), file=sys.stderr)
    sys.exit(1)
PY
}

start_judge_if_needed() {
  local gpu="$1"
  local port="$2"
  local run_id="$3"
  local log="${LOG_DIR}/${run_id}_vllm.log"
  local pid_file="${LOG_DIR}/${run_id}_vllm.pid"

  if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "[judge] already ready gpu=${gpu} port=${port}"
    return 0
  fi

  echo "[judge] starting gpu=${gpu} port=${port} run_id=${run_id} prefix_cache=1 model=${JUDGE_MODEL_DIR}"
  ENABLE_PREFIX_CACHING=1 \
  JUDGE_MODEL_DIR="${JUDGE_MODEL_DIR}" \
  JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME}" \
  JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL}" \
  JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-8192}" \
  JUDGE_CUDA_DEVICE="${gpu}" \
  JUDGE_PORT="${port}" \
  JUDGE_RUN_ID="${run_id}" \
  JUDGE_LOG="${log}" \
  JUDGE_PID_FILE="${pid_file}" \
  VLLM_READY_SECS=900 \
    bash "${ROOT}/training/openrlhf/start_local_judge_if_needed.sh" || true

  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && return 0
    sleep 5
  done
  refuse "judge not ready gpu=${gpu} port=${port}"
}

restart_judge_lb() {
  local backends="$1"
  if [[ -f "${LOG_DIR}/judge_lb_proxy.pid" ]]; then
    local old_pid
    old_pid="$(cat "${LOG_DIR}/judge_lb_proxy.pid" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      kill -TERM "${old_pid}" 2>/dev/null || true
      sleep 1
      kill -9 "${old_pid}" 2>/dev/null || true
      echo "[lb] stopped old judge_lb pid=${old_pid}"
    fi
    rm -f "${LOG_DIR}/judge_lb_proxy.pid"
  fi
  nohup "${ORHF_PYTHON}" "${ROOT}/training/openrlhf/judge_lb_proxy.py" \
    --host 127.0.0.1 --port "${JUDGE_LB_PORT}" \
    --backends "${backends}" \
    >>"${LOG_DIR}/judge_lb_proxy.log" 2>&1 &
  echo $! >"${LOG_DIR}/judge_lb_proxy.pid"
  for _ in $(seq 1 40); do
    curl -sf "http://127.0.0.1:${JUDGE_LB_PORT}/health" >/dev/null 2>&1 && break
    sleep 0.5
  done
  curl -sf "http://127.0.0.1:${JUDGE_LB_PORT}/v1/models" >/dev/null 2>&1 \
    || refuse "judge load balancer not ready on :${JUDGE_LB_PORT}"
  echo "[lb] ready backends=${backends}"
}

[[ -x "${SWIFT_PYTHON}" ]] || refuse "swift venv missing; run training/swift/setup_swift_env.sh"
if [[ ! -s "${PROMPT_DATA}" && "${PROMPT_DATA}" == *swift_prompts_max2048.jsonl ]]; then
  echo "[launch] filtering long prompts -> ${PROMPT_DATA}"
  "${ORHF_PYTHON}" "${ROOT}/training/rl_data/filter_swift_prompts.py" \
    --src "${ROOT}/data/rl/swift_prompts.jsonl" \
    --dst "${PROMPT_DATA}" \
    --tokenizer "${MODEL_DIR}" \
    --max-tokens 2048 || refuse "prompt filter failed"
fi
[[ -s "${PROMPT_DATA}" ]] || refuse "missing swift prompts ${PROMPT_DATA}"
[[ -f "${PLUGIN}" ]] || refuse "missing reward plugin ${PLUGIN}"
[[ -f "${MODEL_DIR}/config.json" ]] || refuse "missing 8B model ${MODEL_DIR}"

SKIP_STOP_JUDGE="${SKIP_STOP_JUDGE:-1}" SKIP_STOP_REWARD="${SKIP_STOP_REWARD:-1}" SKIP_STOP_LB="${SKIP_STOP_LB:-1}" \
  bash "${ROOT}/training/swift/stop_swift_training.sh" || true

if alive_pid_file "${PID_FILE}" || alive_pid_file "${CKPT}/swift_train.pid"; then
  refuse "stale_or_live_pid: swift training still running"
fi

bash "${ROOT}/training/openrlhf/ensure_cuda_ready.sh" || refuse "CUDA not ready"

allow_pids=""
IFS=',' read -ra _pref <<< "${PREFER_JUDGE}"
for g in "${_pref[@]}"; do
  while read -r pid; do
    [[ -z "${pid}" || "${pid}" == "N/A" ]] && continue
    allow_pids="${allow_pids:+${allow_pids},}${pid}"
  done < <(nvidia-smi --id="${g}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
done
echo "[probe] prefer_judge=${PREFER_JUDGE} allow_pids=${allow_pids:-none}"

probe_args=(probe --bundle8 --n-train 4 --n-judge 4 --prefer-judge "${PREFER_JUDGE}" --free-mib "${FREE_MIB}" --util-max "${UTIL_MAX}")
if [[ -n "${allow_pids}" ]]; then
  probe_args+=(--allow-pids "${allow_pids}")
fi
probe="$("${ORHF_PYTHON}" "${ROOT}/training/openrlhf/gpu_bundle_utils.py" "${probe_args[@]}")"
echo "${probe}" >"${CKPT}/gpu_selection.json"
ok="$(python3 -c 'import json,sys; print(int(json.loads(sys.stdin.read()).get("ok", False)))' <<<"${probe}")"
if [[ "${ok}" != "1" ]]; then
  reason="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("reason",""))' <<<"${probe}")"
  refuse "need_8gpu_bundle: ${reason}"
fi

train_gpus="$(python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(",".join(str(x) for x in d["train_gpus"]))' <<<"${probe}")"
judge_gpus="$(python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(",".join(str(x) for x in d["judge_gpus"]))' <<<"${probe}")"
mapfile -t judge_gpu_arr < <(python3 -c 'import json,sys; print("\n".join(str(x) for x in json.loads(sys.stdin.read())["judge_gpus"]))' <<<"${probe}")

echo "[launch] swift train=${train_gpus} judges=${judge_gpus} ckpt=${CKPT} max_steps=${MAX_STEPS}"

export PHYSICS_REWARD_MODE=process_paragraph
export PHYSICS_REWARD_W_ANSWER=0
export PHYSICS_REWARD_W_FORMAT=0
export ENABLE_PREFIX_CACHING=1
judge_pids=()
for i in 0 1 2 3; do
  start_judge_if_needed "${judge_gpu_arr[$i]}" "${JUDGE_PORTS[$i]}" "${JUDGE_RUN_IDS[$i]}" &
  judge_pids+=($!)
done
for jp in "${judge_pids[@]}"; do
  wait "${jp}" || refuse "a judge replica failed to start"
done

lb_backends="127.0.0.1:${JUDGE_PORTS[0]},127.0.0.1:${JUDGE_PORTS[1]},127.0.0.1:${JUDGE_PORTS[2]},127.0.0.1:${JUDGE_PORTS[3]}"
restart_judge_lb "${lb_backends}"

export OPENAI_BASE_URL="http://127.0.0.1:${JUDGE_LB_PORT}/v1"
export PHYSICS_REWARD_CONCURRENCY=24
export PHYSICS_REWARD_MAX_RESPONSE_CHARS="${PHYSICS_REWARD_MAX_RESPONSE_CHARS:-6144}"
export PHYSICSVERIFIER_UNIFIED_RULE_TOP_N="${PHYSICSVERIFIER_UNIFIED_RULE_TOP_N:-4}"
export PHYSICSVERIFIER_PRECISION_MODE="${PHYSICSVERIFIER_PRECISION_MODE:-strict}"
export PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE=lexical
export PHYSICSVERIFIER_LLM_MODEL="${PHYSICSVERIFIER_LLM_MODEL:-${JUDGE_SERVED_NAME}}"
export PHYSICS_REWARD_CACHE_SIZE="${PHYSICS_REWARD_CACHE_SIZE:-4096}"
export VENV="${VENV:-/data1/jinjianhan/venv/openrlhf_train}"
bash "${ROOT}/training/reward_server/start_reward_server.sh" || refuse "reward server failed"

ss -ltn 2>/dev/null | check_loopback_ports || refuse "non-loopback listener on judge/reward ports"

curl -sf http://127.0.0.1:8770/health >/dev/null || refuse "reward /health failed"
for port in "${JUDGE_PORTS[@]}"; do
  curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null || refuse "judge :${port} not ready"
done
n_be="$(curl -sf http://127.0.0.1:${JUDGE_LB_PORT}/health | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("backends",[])))')"
[[ "${n_be}" -ge 4 ]] || refuse "LB backends=${n_be} expected 4"

MAX_STEPS_ARGS=()
if [[ "${MAX_STEPS}" != "0" ]]; then
  MAX_STEPS_ARGS+=(--max_steps "${MAX_STEPS}")
fi
RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  [[ -f "${RESUME_FROM}/config.json" ]] || refuse "resume checkpoint missing ${RESUME_FROM}"
  RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM}" --resume_only_model true --load_args false)
  echo "[launch] resume_only_model from ${RESUME_FROM}"
fi

export PHYSICS_REWARD_URL="http://127.0.0.1:8770/get_reward"
export PHYSICS_REWARD_TIMEOUT="${PHYSICS_REWARD_TIMEOUT:-3600}"
export MASTER_ADDR=127.0.0.1
export CUDA_VISIBLE_DEVICES="${train_gpus}"
export NPROC_PER_NODE="${NPROC}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_HOME:-${WORKSPACE}/openrlhf_rl/cuda_stub}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export TRL_EXPERIMENTAL_SILENCE=1
# shellcheck disable=SC1091
source "${ROOT}/training/openrlhf/setup_slow_share_tmp.sh" || true
# Keep Swift/HF temp on /slow_share. Do not use /data1 (full) or /tmp (root is full).
# /data1/jinjianhan/tmp is a symlink onto NFS anyway, so a "local" TMPDIR there is not local.
SWIFT_TMP="${SWIFT_TMP:-${SLOW_TMP_ROOT:-/slow_share/jinjianhan/tmp}/swift}"
mkdir -p "${SWIFT_TMP}" "${SWIFT_TMP}/hf_datasets" "${SWIFT_TMP}/hf_home"
export TMPDIR="${SWIFT_TMP}"
export TEMP="${SWIFT_TMP}"
export TMP="${SWIFT_TMP}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SWIFT_TMP}/hf_datasets}"
export HF_HOME="${HF_HOME:-${SWIFT_TMP}/hf_home}"

echo "[launch] starting swift rlhf on GPUs ${train_gpus}"
# Driver 550 cannot use vLLM sleep. Keep colocate util low so train backward has headroom.
nohup env \
  CUDA_VISIBLE_DEVICES="${train_gpus}" \
  NPROC_PER_NODE="${NPROC}" \
  MASTER_ADDR=127.0.0.1 \
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" \
  PHYSICS_REWARD_URL="${PHYSICS_REWARD_URL}" \
  PHYSICS_REWARD_TIMEOUT="${PHYSICS_REWARD_TIMEOUT}" \
  CUDA_HOME="${CUDA_HOME}" \
  PATH="${PATH}" \
  DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK}" \
  TRL_EXPERIMENTAL_SILENCE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONFAULTHANDLER=1 \
  TOKENIZERS_PARALLELISM=false \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  NCCL_CUMEM_ENABLE=0 \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8" \
  TMPDIR="${TMPDIR}" \
  TEMP="${TEMP}" \
  TMP="${TMP}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  HF_HOME="${HF_HOME}" \
  "${SWIFT_VENV}/bin/swift" rlhf \
    --rlhf_type grpo \
    --model "${MODEL_DIR}" \
    --external_plugins "${PLUGIN}" \
    --reward_funcs physics_verifier \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_GPU_UTIL}" \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len "${MAX_LENGTH}" \
    --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS}" \
    --vllm_enforce_eager true \
    --vllm_enable_prefix_caching true \
    --sleep_level 0 \
    --offload_model true \
    --offload_optimizer true \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --dataset "${PROMPT_DATA}" \
    --max_completion_length "${MAX_COMPLETION_LEN}" \
    --max_length "${MAX_LENGTH}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BS}" \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --epsilon 0.2 \
    --beta 0.0 \
    --temperature 1.0 \
    --num_generations "${NUM_GENERATIONS}" \
    --dynamic_sample true \
    --max_resample_times "${MAX_RESAMPLE_TIMES}" \
    --eval_strategy no \
    --save_steps "${SAVE_STEPS}" \
    --save_only_model true \
    --save_total_limit 3 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed zero3 \
    --report_to tensorboard \
    --logging_dir "${CKPT}/runs" \
    --output_dir "${CKPT}" \
    --log_completions true \
    --dataloader_num_workers 0 \
    --use_hf true \
    --overlong_filter false \
    "${RESUME_ARGS[@]}" \
    "${MAX_STEPS_ARGS[@]}" \
  >>"${LOG_FILE}" 2>&1 &
echo $! >"${PID_FILE}"
echo $! >"${CKPT}/swift_train.pid"

export CKPT TRAIN_GPUS="${train_gpus}" JUDGE_GPUS="${judge_gpus}" PID_FILE LOG_FILE REPORT MAX_STEPS
export RESUME_FROM VLLM_GPU_UTIL MAX_COMPLETION_LEN JUDGE_MODEL_DIR
python3 - <<'PY' >"${REPORT}"
import json, datetime, os
print(json.dumps({
  "ok": True,
  "phase": "launched",
  "reason": "swift_grpo_4plus4",
  "at": datetime.datetime.utcnow().isoformat() + "Z",
  "ckpt": os.environ["CKPT"],
  "pid": int(open(os.environ["PID_FILE"]).read().strip()),
  "log": os.environ["LOG_FILE"],
  "cuda_visible_devices": os.environ["TRAIN_GPUS"],
  "judge_gpus": [int(x) for x in os.environ["JUDGE_GPUS"].split(",") if x],
  "max_steps": int(os.environ["MAX_STEPS"]),
  "resume_from": os.environ.get("RESUME_FROM", ""),
  "vllm_gpu_memory_utilization": float(os.environ.get("VLLM_GPU_UTIL", "0.22")),
  "generate_max_len": int(os.environ.get("MAX_COMPLETION_LEN", "1536")),
  "num_generations": 6,
  "reward_max_response_chars": int(os.environ.get("PHYSICS_REWARD_MAX_RESPONSE_CHARS", "6144")),
  "unified_rule_top_n": int(os.environ.get("PHYSICSVERIFIER_UNIFIED_RULE_TOP_N", "4")),
  "precision_mode": os.environ.get("PHYSICSVERIFIER_PRECISION_MODE", "strict"),
  "judge_model_dir": os.environ.get("JUDGE_MODEL_DIR", ""),
  "prefix_caching": True,
}, ensure_ascii=False, indent=2))
PY

echo "[launch] swift_grpo pid=$(cat "${PID_FILE}") log=${LOG_FILE} train=${train_gpus} judges=${judge_gpus}"
echo "[launch] report=${REPORT}"
if [[ "${JUDGE_REFRESH}" == "1" ]]; then
  nohup bash "${ROOT}/training/swift/watch_and_refresh_self_judge.sh" \
    >>"${CKPT}/self_judge_refresh.log" 2>&1 &
  echo $! >"${LOG_DIR}/self_judge_refresh.pid"
  echo "[launch] self-judge refresh watcher pid=$(cat "${LOG_DIR}/self_judge_refresh.pid")"
fi
