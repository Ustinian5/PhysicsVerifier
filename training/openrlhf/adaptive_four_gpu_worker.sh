#!/usr/bin/env bash
# Worker for adaptive_four_gpu_watchdog.sh (runs under nohup).
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
ENV_FILE="${WORKSPACE}/openrlhf_rl/env.sh"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

PYTHON="${PYTHON:-${TRAIN_VENV:-/data1/jinjianhan/venv/openrlhf_train}/bin/python}"
UTILS="${ROOT}/training/openrlhf/gpu_bundle_utils.py"
# Force pilot ckpt unless explicitly overridden for this adaptive path.
CKPT="${ADAPTIVE_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-pilot10}"
export QWEN8B_RL_CKPT="${CKPT}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
STATUS="${ADAPTIVE_STATUS:-${CKPT}/adaptive_acquire_status.json}"
LOCK_FILE="${ADAPTIVE_LOCK:-${LOG_DIR}/adaptive_four_gpu.lock}"
RESERVE_PID_FILE="${RESERVE_PID_FILE:-${LOG_DIR}/gpu_reservation.pid}"
RESERVE_STATUS="${RESERVE_STATUS:-${CKPT}/gpu_reservation_status.json}"

FREE_MIB="${FREE_MIB:-75000}"
UTIL_MAX="${UTIL_MAX:-5}"
STABLE_SECS="${STABLE_SECS:-600}"
POLL_SECS="${POLL_SECS:-15}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-86400}"
RESERVE_MIB="${RESERVE_MIB:-512}"
MAX_ACQUIRE_RETRIES="${MAX_ACQUIRE_RETRIES:-5}"
BACKOFF_SECS="${BACKOFF_SECS:-120}"
RAY_GCS_PORT="${RAY_GCS_PORT:-26379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-28265}"

mkdir -p "${LOG_DIR}" "${CKPT}" "$(dirname "${STATUS}")"

# write_status PHASE [KEY VALUE ...]
# Optional: write_status_json PHASE JSON_FILE
write_status() {
  local phase="$1"
  shift
  python3 - "${STATUS}" "${phase}" "$@" <<'PY'
import json, datetime, sys
status_path = sys.argv[1]
phase = sys.argv[2]
args = sys.argv[3:]
extra = {}
i = 0
while i < len(args):
    key = args[i]
    if i + 1 >= len(args):
        break
    val = args[i + 1]
    i += 2
    # best-effort type coercion
    if val.lower() in ("true", "false"):
        coerced = val.lower() == "true"
    else:
        try:
            coerced = int(val)
        except ValueError:
            try:
                coerced = float(val)
            except ValueError:
                coerced = val
    extra[key] = coerced
payload = {"phase": phase, "updated_at": datetime.datetime.now().isoformat()}
payload.update(extra)
with open(status_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
}

write_status_json_file() {
  local phase="$1"
  local json_file="$2"
  python3 - "${STATUS}" "${phase}" "${json_file}" <<'PY'
import json, datetime, sys
from json import JSONDecoder
status_path, phase, json_file = sys.argv[1], sys.argv[2], sys.argv[3]
extra = {}
with open(json_file, "r", encoding="utf-8") as f:
    raw = f.read().strip()
    if raw:
        extra, _ = JSONDecoder().raw_decode(raw)
payload = {"phase": phase, "updated_at": datetime.datetime.now().isoformat()}
if isinstance(extra, dict):
    payload.update(extra)
with open(status_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
}

stop_reservation() {
  if [[ -f "${RESERVE_PID_FILE}" ]]; then
    local rpid
    rpid="$(cat "${RESERVE_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${rpid}" ]] && kill -0 "${rpid}" 2>/dev/null; then
      kill -TERM "${rpid}" 2>/dev/null || true
      sleep 2
      kill -9 "${rpid}" 2>/dev/null || true
    fi
    rm -f "${RESERVE_PID_FILE}"
  fi
}

acquire_stable_bundle() {
  local waited=0
  local stable_for=0
  local last_key=""
  local probe=""
  local probe_file
  probe_file="$(mktemp)"
  write_status "waiting_stable_bundle" stable_secs_required "${STABLE_SECS}" free_mib "${FREE_MIB}"
  while true; do
    if ! TRY_RESTART_FABRICMANAGER=0 bash "${ROOT}/training/openrlhf/ensure_cuda_ready.sh" >/dev/null 2>&1; then
      sudo -n systemctl restart nvidia-fabricmanager >/dev/null 2>&1 || true
      stable_for=0
      last_key=""
      write_status "waiting_cuda" waited_secs "${waited}"
      sleep "${POLL_SECS}"
      waited=$((waited + POLL_SECS))
      continue
    fi

    set +e
    "${PYTHON}" "${UTILS}" probe --bundle --free-mib "${FREE_MIB}" --util-max "${UTIL_MAX}" >"${probe_file}" 2>/tmp/adaptive_probe_err.$$
    probe_rc=$?
    set -e
    probe_ok="$(python3 - "${probe_file}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    d = json.loads(p.read_text(encoding="utf-8"))
except Exception as e:
    print("false")
    raise SystemExit(0)
print("true" if d.get("ok") else "false")
PY
)"
    if [[ "${probe_ok}" == "true" ]]; then
      # Key only on the 4-GPU set so tiny free-memory jitter (judge flip) does not reset the window.
      key="$(python3 - "${probe_file}" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(",".join(map(str, sorted(d["gpus"]))))
PY
)"
      if [[ "${key}" == "${last_key}" ]]; then
        stable_for=$((stable_for + POLL_SECS))
      else
        last_key="${key}"
        stable_for=0
        echo "[acquire] new candidate bundle: ${key}" >&2
      fi
      python3 - "${probe_file}" "${waited}" "${stable_for}" "${STABLE_SECS}" "${probe_file}.extra" <<'PY'
import json, sys
from json import JSONDecoder
raw = open(sys.argv[1], encoding="utf-8").read().strip()
d, _ = JSONDecoder().raw_decode(raw)
extra = {
  "waited_secs": int(sys.argv[2]),
  "stable_for": int(sys.argv[3]),
  "stable_secs_required": int(sys.argv[4]),
  "candidate": d,
}
open(sys.argv[5], "w", encoding="utf-8").write(json.dumps(extra, ensure_ascii=False))
PY
      write_status_json_file "waiting_stable_bundle" "${probe_file}.extra" || true
      rm -f "${probe_file}.extra"
      if [[ "${stable_for}" -ge "${STABLE_SECS}" ]]; then
        cat "${probe_file}"
        rm -f "${probe_file}" /tmp/adaptive_probe_err.$$
        return 0
      fi
    else
      last_key=""
      stable_for=0
      write_status "waiting_idle_gpus" waited_secs "${waited}"
      # Keep probe stderr out of the main log unless debugging.
      rm -f /tmp/adaptive_probe_err.$$
    fi

    sleep "${POLL_SECS}"
    waited=$((waited + POLL_SECS))
    if [[ "${waited}" -ge "${MAX_WAIT_SECS}" ]]; then
      write_status "timeout" waited_secs "${waited}"
      rm -f "${probe_file}"
      return 3
    fi
  done
}

reserve_bundle() {
  local gpus_csv="$1"
  local reprobe
  reprobe="$(mktemp)"
  stop_reservation
  (
    flock -w 30 9 || exit 9
    if ! "${PYTHON}" "${UTILS}" probe --bundle --free-mib "${FREE_MIB}" --util-max "${UTIL_MAX}" >"${reprobe}"; then
      echo "[error] revalidation failed under flock" >&2
      exit 4
    fi
    python3 - <<PY
import json, sys
want = set(int(x) for x in "${gpus_csv}".split(",") if x.strip())
got = set(json.load(open("${reprobe}"))["gpus"])
sys.exit(0 if want == got else 5)
PY
    nohup "${PYTHON}" "${UTILS}" reserve \
      --gpus "${gpus_csv}" \
      --mib "${RESERVE_MIB}" \
      --pid-file "${RESERVE_PID_FILE}" \
      --status-file "${RESERVE_STATUS}" \
      --timeout-secs 7200 \
      >>"${LOG_DIR}/gpu_reservation.log" 2>&1 &
    for _ in $(seq 1 30); do
      if [[ -f "${RESERVE_PID_FILE}" ]] && kill -0 "$(cat "${RESERVE_PID_FILE}")" 2>/dev/null; then
        exit 0
      fi
      sleep 1
    done
    exit 6
  ) 9>"${LOCK_FILE}"
  local rc=$?
  rm -f "${reprobe}"
  return "${rc}"
}

release_reservation_keep() {
  local keep_csv="$1"
  stop_reservation
  if [[ -n "${keep_csv}" ]]; then
    nohup "${PYTHON}" "${UTILS}" reserve \
      --gpus "${keep_csv}" \
      --mib "${RESERVE_MIB}" \
      --pid-file "${RESERVE_PID_FILE}" \
      --status-file "${RESERVE_STATUS}" \
      --timeout-secs 7200 \
      >>"${LOG_DIR}/gpu_reservation.log" 2>&1 &
    sleep 2
  fi
}

stop_train_only() {
  if [[ -f "${CKPT}/direct_train.pid" ]]; then
    local dpid
    dpid="$(cat "${CKPT}/direct_train.pid" || true)"
    if [[ -n "${dpid}" ]] && kill -0 "${dpid}" 2>/dev/null; then
      pkill -TERM -P "${dpid}" 2>/dev/null || true
      kill -TERM "${dpid}" 2>/dev/null || true
      sleep 3
      pkill -9 -P "${dpid}" 2>/dev/null || true
      kill -9 "${dpid}" 2>/dev/null || true
    fi
  fi
  if [[ -f "${CKPT}/ray/ray_head.pid" ]]; then
    local rpid
    rpid="$(cat "${CKPT}/ray/ray_head.pid" || true)"
    if [[ -n "${rpid}" ]] && kill -0 "${rpid}" 2>/dev/null; then
      pkill -TERM -P "${rpid}" 2>/dev/null || true
      kill -TERM "${rpid}" 2>/dev/null || true
      sleep 2
      kill -9 "${rpid}" 2>/dev/null || true
    fi
  fi
  if [[ -f "${LOG_DIR}/four_gpu_pilot10.pid" ]]; then
    local ppid
    ppid="$(cat "${LOG_DIR}/four_gpu_pilot10.pid" || true)"
    if [[ -n "${ppid}" ]] && kill -0 "${ppid}" 2>/dev/null; then
      pkill -TERM -P "${ppid}" 2>/dev/null || true
      kill -TERM "${ppid}" 2>/dev/null || true
      sleep 2
      kill -9 "${ppid}" 2>/dev/null || true
    fi
  fi
}

run_pilot_once() {
  local topology="$1"
  local train_gpus="$2"
  local judge_gpu="$3"

  write_status "handing_off_judge" judge_gpu "${judge_gpu}" train_gpus "${train_gpus}" topology "${topology}"
  release_reservation_keep "${train_gpus}"

  env -u PHYSICSVERIFIER_OPENAI_BASE_URL -u PHYSICSVERIFIER_OPENAI_API_KEY \
    PHYSICS_REWARD_MODE=answer_low_verifier \
    JUDGE_CUDA_DEVICE="${judge_gpu}" \
    OPENAI_BASE_URL=http://127.0.0.1:8766/v1 \
    OPENAI_API_KEY=EMPTY \
    PHYSICSVERIFIER_LLM_MODEL=qwen3-30b-a3b \
    bash "${ROOT}/training/openrlhf/start_local_judge_if_needed.sh"

  # Force local judge path; shell often has PHYSICSVERIFIER_OPENAI_* from prior sessions.
  if [[ -f "${LOG_DIR}/physics_reward_server.pid" ]]; then
    rwp="$(cat "${LOG_DIR}/physics_reward_server.pid" 2>/dev/null || true)"
    if [[ -n "${rwp}" ]] && kill -0 "${rwp}" 2>/dev/null; then
      kill -TERM "${rwp}" 2>/dev/null || true
      sleep 2
      kill -9 "${rwp}" 2>/dev/null || true
    fi
    rm -f "${LOG_DIR}/physics_reward_server.pid" "${LOG_DIR}/physics_reward_server.mode"
  fi
  env -u PHYSICSVERIFIER_OPENAI_BASE_URL -u PHYSICSVERIFIER_OPENAI_API_KEY \
    PHYSICS_REWARD_MODE=answer_low_verifier \
    OPENAI_BASE_URL=http://127.0.0.1:8766/v1 \
    OPENAI_API_KEY=EMPTY \
    PHYSICSVERIFIER_LLM_MODEL=qwen3-30b-a3b \
    bash "${ROOT}/training/reward_server/start_reward_server.sh"
  stop_reservation
  write_status "launching_pilot" topology "${topology}" train_gpus "${train_gpus}" judge_gpu "${judge_gpu}"

  local vllm_util=0.55 actor_gpus=3 vllm_engines=3 micro_rollout=2 rollout_bs=12 n_samples=2 train_bs=24
  if [[ "${topology}" == "split" ]]; then
    vllm_util=0.70; actor_gpus=2; vllm_engines=1; micro_rollout=1; rollout_bs=8; n_samples=2; train_bs=16
  fi

  env -u PHYSICSVERIFIER_OPENAI_BASE_URL -u PHYSICSVERIFIER_OPENAI_API_KEY \
    PHYSICS_REWARD_MODE=answer_low_verifier \
    JUDGE_CUDA_DEVICE="${judge_gpu}" \
    OPENAI_BASE_URL=http://127.0.0.1:8766/v1 \
    OPENAI_API_KEY=EMPTY \
    PHYSICSVERIFIER_LLM_MODEL=qwen3-30b-a3b \
    CUDA_VISIBLE_DEVICES="${train_gpus}" \
    TRAIN_TOPOLOGY="${topology}" \
    ACTOR_GPUS="${actor_gpus}" \
    VLLM_ENGINES="${vllm_engines}" \
    VLLM_GPU_MEMORY_UTILIZATION="${vllm_util}" \
    GENERATE_MAX_LEN=1536 \
    ROLLOUT_BATCH_SIZE="${rollout_bs}" \
    N_SAMPLES_PER_PROMPT="${n_samples}" \
    TRAIN_BATCH_SIZE="${train_bs}" \
    MICRO_ROLLOUT_BATCH_SIZE="${micro_rollout}" \
    MAX_SAMPLES=512 \
    PILOT_MAX_STEPS=10 \
    DYNAMIC_FILTER_MIN=0.0 \
    DYNAMIC_FILTER_MAX=1.0 \
    ALLOW_RAY_JOBS=0 \
    ALLOW_DIRECT_LAUNCH=1 \
    RAY_GCS_PORT="${RAY_GCS_PORT}" \
    RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT}" \
    QWEN8B_RL_CKPT="${CKPT}" \
    bash "${ROOT}/training/openrlhf/run_four_gpu_pilot_nohup.sh"

  local pilot_pid
  pilot_pid="$(cat "${LOG_DIR}/four_gpu_pilot10.pid" 2>/dev/null || true)"
  write_status "pilot_running" topology "${topology}" pilot_pid "${pilot_pid}" train_gpus "${train_gpus}" judge_gpu "${judge_gpu}"

  while [[ -n "${pilot_pid}" ]] && kill -0 "${pilot_pid}" 2>/dev/null; do
    if [[ -s "${CKPT}/plots/training_metrics.csv" ]]; then
      local steps
      steps="$(awk -F, 'NR>1 {c++} END {print c+0}' "${CKPT}/plots/training_metrics.csv")"
      write_status "pilot_running" topology "${topology}" pilot_pid "${pilot_pid}" global_steps "${steps}" train_gpus "${train_gpus}" judge_gpu "${judge_gpu}"
    fi
    sleep 30
  done

  if [[ -f "${CKPT}/fallback_request.json" && "${topology}" == "colocate" ]]; then
    return 42
  fi
  local steps_done=0
  if [[ -s "${CKPT}/plots/training_metrics.csv" ]]; then
    steps_done="$(awk -F, 'NR>1 {c++} END {print c+0}' "${CKPT}/plots/training_metrics.csv")"
  fi
  if [[ "${steps_done}" -ge 1 ]]; then
    return 0
  fi
  if [[ "${topology}" == "colocate" && -s "${CKPT}/direct_train.log" ]] \
    && grep -qiE 'CUDA out of memory|OutOfMemoryError|cuda ipc|enable_sleep_mode|ncclUnhandledCudaError' "${CKPT}/direct_train.log"; then
    return 42
  fi
  return 1
}

acquire_attempt=0
BUNDLE_FILE="${CKPT}/selected_bundle.json"
while [[ "${acquire_attempt}" -lt "${MAX_ACQUIRE_RETRIES}" ]]; do
  acquire_attempt=$((acquire_attempt + 1))
  # Avoid command-substitution subshell so status updates stay in this process.
  set +e
  acquire_stable_bundle >"${BUNDLE_FILE}"
  acq_rc=$?
  set -e
  if [[ "${acq_rc}" -ne 0 ]]; then
    exit "${acq_rc}"
  fi
  train_gpus="$(python3 - "${BUNDLE_FILE}" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(",".join(map(str, d["train_gpus"])))
PY
)"
  judge_gpu="$(python3 - "${BUNDLE_FILE}" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(d["judge_gpu"])
PY
)"
  all_gpus="$(python3 - "${BUNDLE_FILE}" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(",".join(map(str, d["gpus"])))
PY
)"
  echo "[acquire] stable bundle all=${all_gpus} train=${train_gpus} judge=${judge_gpu}"

  t0=$(date +%s)
  reserve_bundle "${all_gpus}"
  t1=$(date +%s)
  echo "[acquire] reservation ready in $((t1 - t0))s"
  write_status "reserved" gpus "${all_gpus}" train_gpus "${train_gpus}" judge_gpu "${judge_gpu}" reserve_secs "$((t1 - t0))"

  python3 - <<PY >"${CKPT}/gpu_selection.json"
import json, datetime
print(json.dumps({
  "selected_at": datetime.datetime.now().isoformat(),
  "all_gpus": [int(x) for x in "${all_gpus}".split(",") if x.strip()],
  "train_gpus": [int(x) for x in "${train_gpus}".split(",") if x.strip()],
  "judge_gpu": int("${judge_gpu}"),
  "stable_secs": int("${STABLE_SECS}"),
  "free_mib": int("${FREE_MIB}"),
}, ensure_ascii=False, indent=2))
PY

  topology="colocate"
  rm -f "${CKPT}/fallback_request.json"
  set +e
  run_pilot_once "${topology}" "${train_gpus}" "${judge_gpu}"
  rc=$?
  set -e

  if [[ "${rc}" -eq 42 ]]; then
    echo "[fallback] colocate failed; retrying split on same bundle"
    write_status "fallback_split" train_gpus "${train_gpus}" judge_gpu "${judge_gpu}"
    stop_train_only
    topology="split"
    set +e
    run_pilot_once "split" "${train_gpus}" "${judge_gpu}"
    rc=$?
    set -e
  fi

  bash "${ROOT}/training/openrlhf/generate_four_gpu_admission.sh" || true

  steps_done=0
  if [[ -s "${CKPT}/plots/training_metrics.csv" ]]; then
    steps_done="$(awk -F, 'NR>1 {c++} END {print c+0}' "${CKPT}/plots/training_metrics.csv")"
  fi
  if [[ "${steps_done}" -ge 1 ]]; then
    write_status "finished" global_steps "${steps_done}" topology_final "${topology}" train_gpus "${train_gpus}" judge_gpu "${judge_gpu}"
    stop_reservation
    exit 0
  fi

  echo "[retry] pilot failed without steps; backoff ${BACKOFF_SECS}s then re-acquire"
  write_status "retry_backoff" attempt "${acquire_attempt}" backoff_secs "${BACKOFF_SECS}"
  JUDGE_CUDA_DEVICE="${judge_gpu}" bash "${ROOT}/training/openrlhf/stop_local_judge.sh" "${judge_gpu}" >/dev/null 2>&1 || true
  if [[ -f "${LOG_DIR}/physics_reward_server.pid" ]]; then
    rwp="$(cat "${LOG_DIR}/physics_reward_server.pid" || true)"
    if [[ -n "${rwp}" ]] && kill -0 "${rwp}" 2>/dev/null; then
      kill -TERM "${rwp}" 2>/dev/null || true
      sleep 1
      kill -9 "${rwp}" 2>/dev/null || true
    fi
  fi
  stop_train_only
  stop_reservation
  sleep "${BACKOFF_SECS}"
done

write_status "failed" attempts "${acquire_attempt}"
exit 1
