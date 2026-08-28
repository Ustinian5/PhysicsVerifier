#!/usr/bin/env bash
# One-shot 10-step bootstrap pilot launcher.
# Probe idle GPUs once; never wait, never start the adaptive watchdog, never抢卡.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
CKPT="${QWEN8B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-bootstrap10}"
PID_FILE="${PID_FILE:-${LOG_DIR}/bootstrap_pilot10.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/bootstrap_pilot10.log}"
RAY_GCS_PORT="${RAY_GCS_PORT:-26379}"
export RAY_GCS_PORT
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
FREE_MIB="${FREE_MIB:-75000}"
UTIL_MAX="${UTIL_MAX:-5}"
REPORT="${REPORT:-${CKPT}/oneshot_launch_report.json}"
export MIN_SLOW_TMP_GB="${MIN_SLOW_TMP_GB:-20}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/orhf8b_ray_${RAY_GCS_PORT}}"

mkdir -p "${LOG_DIR}" "${CKPT}/plots" "${CKPT}/ray"

refuse() {
  local reason="$1"
  mkdir -p "$(dirname "${REPORT}")"
  python3 - <<PY >"${REPORT}"
import json, datetime
print(json.dumps({
  "ok": False,
  "phase": "refused",
  "reason": """${reason}""",
  "at": datetime.datetime.utcnow().isoformat() + "Z",
  "ckpt": "${CKPT}",
  "watchdog_started": False,
}, ensure_ascii=False, indent=2))
PY
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

# Clean *our* failed-run leftovers, then probe once.
QWEN8B_RL_CKPT="${CKPT}" bash "${ROOT}/training/openrlhf/stop_four_gpu_pilot.sh" || true
sleep 2

if alive_pid_file "${PID_FILE}" || alive_pid_file "${LOG_DIR}/four_gpu_pilot10.pid" || alive_pid_file "${CKPT}/direct_train.pid"; then
  refuse "stale_or_live_pid: a previous pilot/train pid is still alive"
fi

# Half-dead isolated Ray: GCS up but session metadata missing.
if ss -ltn 2>/dev/null | grep -q ":${RAY_GCS_PORT} "; then
  meta=""
  if [[ -d "${RAY_TMPDIR}" ]]; then
    meta="$(find "${RAY_TMPDIR}" -name node_ip_address.json 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "${meta}" ]]; then
    refuse "half_dead_ray: GCS ${RAY_GCS_PORT} is up but session metadata is missing; not deleting a live session"
  fi
  refuse "residual_gcs_${RAY_GCS_PORT}: isolated Ray still listening after cleanup"
fi

# After our isolated Ray is gone, put this job's temp files on slow_share.
# shellcheck disable=SC1091
source "${ROOT}/training/openrlhf/setup_slow_share_tmp.sh" || refuse "slow_share_tmp: cannot prepare /slow_share/jinjianhan/tmp"

probe="$("${PYTHON}" "${ROOT}/training/openrlhf/gpu_bundle_utils.py" probe --bundle --free-mib "${FREE_MIB}" --util-max "${UTIL_MAX}")"
echo "${probe}" >"${CKPT}/gpu_selection.json"
ok="$(python3 -c "import json,sys; print(int(json.loads(sys.stdin.read()).get('ok', False)))" <<<"${probe}")"
if [[ "${ok}" != "1" ]]; then
  idle="$(python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('idle_indices',[]))" <<<"${probe}")"
  refuse "need_four_idle_gpus (idle=${idle}); not waiting, not starting watchdog"
fi

gpus="$(python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(','.join(str(x) for x in d['gpus']))" <<<"${probe}")"
train_gpus="$(python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(','.join(str(x) for x in d['train_gpus']))" <<<"${probe}")"
judge_gpu="$(python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['judge_gpu'])" <<<"${probe}")"
echo "[launch] one-shot bootstrap train=${train_gpus} judge=${judge_gpu} bundle=${gpus} ckpt=${CKPT}"

nohup bash -c "
  set -euo pipefail
  source '${WORKSPACE}/openrlhf_rl/env.sh'
  if [[ -f '${ROOT}/training/openrlhf/paragraph_process_defaults.env' ]]; then
    source '${ROOT}/training/openrlhf/paragraph_process_defaults.env'
  fi
  export TRAIN_STAGE=bootstrap
  export TRAIN_TOPOLOGY=colocate
  export CUDA_VISIBLE_DEVICES='${train_gpus}'
  export JUDGE_CUDA_DEVICE='${judge_gpu}'
  export ACTOR_GPUS=3
  export VLLM_ENGINES=3
  export PHYSICS_REWARD_MODE=process_paragraph
  export PHYSICS_REWARD_VERIFIER_ON_WRONG=1
  export PHYSICS_REWARD_W_FORMAT=0
  export PHYSICS_REWARD_W_ANSWER=0
  export RAY_TMPDIR='${RAY_TMPDIR}'
  export RAY_TMPDIR_REAL='${RAY_TMPDIR_REAL:-}'
  export TMPDIR='${TMPDIR}'
  export TEMP='${TEMP:-${TMPDIR}}'
  export TMP='${TMP:-${TMPDIR}}'
  export QWEN8B_RL_CKPT='${CKPT}'
  export PILOT_MAX_STEPS='${PILOT_MAX_STEPS:-10}'
  export GENERATE_MAX_LEN='${GENERATE_MAX_LEN:-768}'
  export ROLLOUT_BATCH_SIZE='${ROLLOUT_BATCH_SIZE:-2}'
  export N_SAMPLES_PER_PROMPT='${N_SAMPLES_PER_PROMPT:-6}'
  export TRAIN_BATCH_SIZE='${TRAIN_BATCH_SIZE:-12}'
  export MICRO_ROLLOUT_BATCH_SIZE='${MICRO_ROLLOUT_BATCH_SIZE:-1}'
  export PHYSICS_REWARD_CONCURRENCY='${PHYSICS_REWARD_CONCURRENCY:-12}'
  export PHYSICS_REWARD_MAX_RESPONSE_CHARS='${PHYSICS_REWARD_MAX_RESPONSE_CHARS:-3072}'
  export PHYSICS_REWARD_TIMEOUT='${PHYSICS_REWARD_TIMEOUT:-1800}'
  export PHYSICSVERIFIER_UNIFIED_RULE_TOP_N='${PHYSICSVERIFIER_UNIFIED_RULE_TOP_N:-4}'
  export PHYSICSVERIFIER_PRECISION_MODE='${PHYSICSVERIFIER_PRECISION_MODE:-balanced}'
  export PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE='${PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE:-lexical}'
  export DYNAMIC_FILTER_MAX_GEN_BATCHES='${DYNAMIC_FILTER_MAX_GEN_BATCHES:-8}'
  export VLLM_GPU_MEMORY_UTILIZATION='${VLLM_GPU_MEMORY_UTILIZATION:-0.55}'
  export MAX_SAMPLES='${MAX_SAMPLES:-2048}'
  export DYNAMIC_FILTER_MODE=reward_variance
  export DYNAMIC_FILTER_MAX_GEN_BATCHES=32
  export DYNAMIC_FILTER_MIN=0.0
  export DYNAMIC_FILTER_MAX=1.0
  export PROMPT_DATA='${ROOT}/data/rl/bootstrap_curriculum.jsonl'
  export ALLOW_RAY_JOBS=0
  export ALLOW_DIRECT_LAUNCH=1
  export MASTER_ADDR=127.0.0.1
  export RAY_BIND_IP=127.0.0.1
  export RAY_GCS_PORT='${RAY_GCS_PORT}'
  export RAY_DASHBOARD_PORT='${RAY_DASHBOARD_PORT:-28265}'
  export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0
  source '${ROOT}/training/openrlhf/setup_slow_share_tmp.sh'
  export PHYSICS_REWARD_METRICS_LOG='${CKPT}/plots/physics_reward_metrics.jsonl'
  bash '${ROOT}/training/openrlhf/run_four_gpu_pilot.sh'
" >>"${LOG_FILE}" 2>&1 &
echo $! >"${PID_FILE}"

python3 - <<PY >"${REPORT}"
import json, datetime
print(json.dumps({
  "ok": True,
  "phase": "launched",
  "reason": "four_idle_gpus",
  "at": datetime.datetime.utcnow().isoformat() + "Z",
  "ckpt": "${CKPT}",
  "pid": int(open("${PID_FILE}").read().strip()),
  "log": "${LOG_FILE}",
  "cuda_visible_devices": "${train_gpus}",
  "judge_cuda_device": int("${judge_gpu}"),
  "train_stage": "bootstrap",
  "reward_mode": "process_paragraph",
  "process_only_reward": True,
  "ray_tmpdir": "${RAY_TMPDIR}",
  "tmpdir": "${TMPDIR}",
  "watchdog_started": False,
  "pilot_max_steps": int("${PILOT_MAX_STEPS:-10}"),
}, ensure_ascii=False, indent=2))
PY

echo "[launch] bootstrap_pilot pid=$(cat "${PID_FILE}") log=${LOG_FILE} train=${train_gpus} judge=${judge_gpu}"
echo "[launch] watchdog_started=0 report=${REPORT}"
