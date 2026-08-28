#!/usr/bin/env bash
# One-shot 8-GPU full training: 4 colocate train GPUs + 4 dedicated judges.
# Never抢卡. Keep already-loaded GPU4/7 judges; add replicas on freed GPUs.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
CKPT="${QWEN8B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf}"
PID_FILE="${PID_FILE:-${LOG_DIR}/full_training_8gpu.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/full_training_8gpu.log}"
RAY_GCS_PORT="${RAY_GCS_PORT:-26379}"
export RAY_GCS_PORT
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
FREE_MIB="${FREE_MIB:-75000}"
UTIL_MAX="${UTIL_MAX:-5}"
REPORT="${REPORT:-${CKPT}/full_8gpu_launch_report.json}"
export MIN_SLOW_TMP_GB="${MIN_SLOW_TMP_GB:-20}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/orhf8b_ray_${RAY_GCS_PORT}}"
PREFER_JUDGE="${PREFER_JUDGE:-4,7}"
JUDGE_LB_PORT="${JUDGE_LB_PORT:-8765}"
JUDGE_PORTS=(8766 8767 8768 8769)
JUDGE_RUN_IDS=(local_judge local_judge2 local_judge3 local_judge4)

mkdir -p "${LOG_DIR}" "${CKPT}/plots" "${CKPT}/ray"

refuse() {
  local reason="$1"
  mkdir -p "$(dirname "${REPORT}")"
  python3 -c 'import json,datetime,os,sys; print(json.dumps({"ok":False,"phase":"refused","reason":sys.argv[1],"at":datetime.datetime.utcnow().isoformat()+"Z","ckpt":os.environ.get("CKPT","")},ensure_ascii=False,indent=2))' "${reason}" >"${REPORT}"
  echo "[refuse] ${reason}" >&2
  echo "[refuse] report=${REPORT}" >&2
  exit 2
}

alive_pid_file() {
  local file="$1"
  [[ -f "${file}" ]] || return 1
  local pid
  pid="$(cat "${file}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
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

  echo "[judge] starting gpu=${gpu} port=${port} run_id=${run_id}"
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

  nohup "${PYTHON}" "${ROOT}/training/openrlhf/judge_lb_proxy.py" \
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

export CKPT
QWEN8B_RL_CKPT="${CKPT}" SKIP_STOP_JUDGE=1 SKIP_STOP_REWARD=1 SKIP_STOP_LB=1 \
  bash "${ROOT}/training/openrlhf/stop_four_gpu_pilot.sh" || true
QWEN8B_RL_CKPT="/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-bootstrap10" \
  SKIP_STOP_JUDGE=1 SKIP_STOP_REWARD=1 SKIP_STOP_LB=1 \
  bash "${ROOT}/training/openrlhf/stop_four_gpu_pilot.sh" || true
sleep 2

if alive_pid_file "${PID_FILE}" || alive_pid_file "${CKPT}/direct_train.pid"; then
  refuse "stale_or_live_pid: a previous full-train pid is still alive"
fi

if ss -ltn 2>/dev/null | grep -q ":${RAY_GCS_PORT} "; then
  refuse "residual_gcs_${RAY_GCS_PORT}: isolated Ray still listening after cleanup"
fi

# shellcheck disable=SC1091
source "${ROOT}/training/openrlhf/setup_slow_share_tmp.sh" || refuse "slow_share_tmp unavailable"

allow_pids=""
IFS=',' read -ra _pref <<< "${PREFER_JUDGE}"
for g in "${_pref[@]}"; do
  while read -r pid; do
    [[ -z "${pid}" || "${pid}" == "N/A" ]] && continue
    allow_pids="${allow_pids:+${allow_pids},}${pid}"
  done < <(nvidia-smi --id="${g}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
done

probe_args=(probe --bundle8 --n-train 4 --n-judge 4 --prefer-judge "${PREFER_JUDGE}" --free-mib "${FREE_MIB}" --util-max "${UTIL_MAX}")
if [[ -n "${allow_pids}" ]]; then
  probe_args+=(--allow-pids "${allow_pids}")
fi
probe="$("${PYTHON}" "${ROOT}/training/openrlhf/gpu_bundle_utils.py" "${probe_args[@]}")"
echo "${probe}" >"${CKPT}/gpu_selection.json"
ok="$(python3 -c 'import json,sys; print(int(json.loads(sys.stdin.read()).get("ok", False)))' <<<"${probe}")"
if [[ "${ok}" != "1" ]]; then
  reason="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("reason",""))' <<<"${probe}")"
  refuse "need_8gpu_bundle: ${reason}"
fi

train_gpus="$(python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(",".join(str(x) for x in d["train_gpus"]))' <<<"${probe}")"
judge_gpus="$(python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(",".join(str(x) for x in d["judge_gpus"]))' <<<"${probe}")"
mapfile -t judge_gpu_arr < <(python3 -c 'import json,sys; print("\n".join(str(x) for x in json.loads(sys.stdin.read())["judge_gpus"]))' <<<"${probe}")

echo "[launch] 8gpu train=${train_gpus} judges=${judge_gpus} ckpt=${CKPT}"

export PHYSICS_REWARD_MODE=process_paragraph
for i in 0 1 2 3; do
  start_judge_if_needed "${judge_gpu_arr[$i]}" "${JUDGE_PORTS[$i]}" "${JUDGE_RUN_IDS[$i]}"
done

lb_backends="127.0.0.1:${JUDGE_PORTS[0]},127.0.0.1:${JUDGE_PORTS[1]},127.0.0.1:${JUDGE_PORTS[2]},127.0.0.1:${JUDGE_PORTS[3]}"
restart_judge_lb "${lb_backends}"

export OPENAI_BASE_URL="http://127.0.0.1:${JUDGE_LB_PORT}/v1"
export PHYSICS_REWARD_CONCURRENCY=24
export PHYSICS_REWARD_MODE=process_paragraph
export PHYSICS_REWARD_MAX_RESPONSE_CHARS=2048
export PHYSICSVERIFIER_UNIFIED_RULE_TOP_N=2
export VENV="${VENV:-/data1/jinjianhan/venv/openrlhf_train}"
bash "${ROOT}/training/reward_server/start_reward_server.sh" || refuse "reward server failed after judge LB switch"

echo "[launch] starting full training train=${train_gpus} judges=${judge_gpus}"

nohup bash -c "
  set -euo pipefail
  source '${WORKSPACE}/openrlhf_rl/env.sh'
  if [[ -f '${ROOT}/training/openrlhf/paragraph_process_defaults.env' ]]; then
    source '${ROOT}/training/openrlhf/paragraph_process_defaults.env'
  fi
  export ROLLOUT_BATCH_SIZE=4
  export N_SAMPLES_PER_PROMPT=6
  export TRAIN_BATCH_SIZE=24
  export PHYSICS_REWARD_CONCURRENCY=24
  export PILOT_MAX_STEPS=0
  export ACTOR_GPUS=4
  export VLLM_ENGINES=4
  export TRAIN_STAGE=full
  export TRAIN_TOPOLOGY=colocate
  export CUDA_VISIBLE_DEVICES='${train_gpus}'
  export JUDGE_CUDA_DEVICE='${judge_gpu_arr[0]}'
  export PHYSICS_REWARD_MODE=process_paragraph
  export PHYSICS_REWARD_VERIFIER_ON_WRONG=1
  export PHYSICS_REWARD_W_FORMAT=0
  export PHYSICS_REWARD_W_ANSWER=0
  export OPENAI_BASE_URL='http://127.0.0.1:${JUDGE_LB_PORT}/v1'
  export RAY_TMPDIR='${RAY_TMPDIR}'
  export RAY_TMPDIR_REAL='${RAY_TMPDIR_REAL:-}'
  export TMPDIR='${TMPDIR}'
  export TEMP='${TEMP:-${TMPDIR}}'
  export TMP='${TMP:-${TMPDIR}}'
  export QWEN8B_RL_CKPT='${CKPT}'
  export GENERATE_MAX_LEN=512
  export PHYSICS_REWARD_MAX_RESPONSE_CHARS=2048
  export PHYSICS_REWARD_TIMEOUT=3600
  export PHYSICSVERIFIER_UNIFIED_RULE_TOP_N=2
  export PHYSICSVERIFIER_PRECISION_MODE='${PHYSICSVERIFIER_PRECISION_MODE:-balanced}'
  export PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE='${PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE:-lexical}'
  export DYNAMIC_FILTER_MAX_GEN_BATCHES='${DYNAMIC_FILTER_MAX_GEN_BATCHES:-8}'
  export VLLM_GPU_MEMORY_UTILIZATION='${VLLM_GPU_MEMORY_UTILIZATION:-0.55}'
  export MAX_SAMPLES='${MAX_SAMPLES:-100000}'
  export SAVE_STEPS='${SAVE_STEPS:-20}'
  export PROMPT_DATA='${ROOT}/data/rl/openrlhf_prompts.jsonl'
  export DYNAMIC_FILTER_MODE=reward_variance
  export DYNAMIC_FILTER_MIN=0.0
  export DYNAMIC_FILTER_MAX=1.0
  export ENABLE_EVAL=0
  export ALLOW_RAY_JOBS=0
  export ALLOW_DIRECT_LAUNCH=1
  export MASTER_ADDR=127.0.0.1
  export RAY_BIND_IP=127.0.0.1
  export RAY_GCS_PORT='${RAY_GCS_PORT}'
  export RAY_DASHBOARD_PORT='${RAY_DASHBOARD_PORT:-28265}'
  export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0
  source '${ROOT}/training/openrlhf/setup_slow_share_tmp.sh'
  export PHYSICS_REWARD_METRICS_LOG='${CKPT}/plots/physics_reward_metrics.jsonl'
  bash '${ROOT}/training/openrlhf/launch_training_8gpu.sh'
" >>"${LOG_FILE}" 2>&1 &
echo $! >"${PID_FILE}"

export TRAIN_GPUS="${train_gpus}" JUDGE_GPUS="${judge_gpus}" PID_FILE LOG_FILE REPORT JUDGE_LB_PORT
python3 - <<'PY' >"${REPORT}"
import json, datetime, os
print(json.dumps({
  "ok": True,
  "phase": "launched",
  "reason": "eight_gpu_full_training_4plus4",
  "at": datetime.datetime.utcnow().isoformat() + "Z",
  "ckpt": os.environ["CKPT"],
  "pid": int(open(os.environ["PID_FILE"]).read().strip()),
  "log": os.environ["LOG_FILE"],
  "cuda_visible_devices": os.environ["TRAIN_GPUS"],
  "judge_gpus": [int(x) for x in os.environ["JUDGE_GPUS"].split(",") if x],
  "judge_lb": "http://127.0.0.1:%s/v1" % os.environ["JUDGE_LB_PORT"],
  "train_stage": "full",
  "reward_mode": "process_paragraph",
  "actor_gpus": 4,
  "generate_max_len": 512,
  "reward_max_response_chars": 2048,
  "unified_rule_top_n": 2,
  "rollout_batch_size": 4,
  "n_samples_per_prompt": 6,
  "train_batch_size": 24,
  "reward_concurrency": 24,
  "pilot_max_steps": 0,
}, ensure_ascii=False, indent=2))
PY

echo "[launch] full_training_8gpu pid=$(cat "${PID_FILE}") log=${LOG_FILE} train=${train_gpus} judges=${judge_gpus}"
echo "[launch] report=${REPORT}"
