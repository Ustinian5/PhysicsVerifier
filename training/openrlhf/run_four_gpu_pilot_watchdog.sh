#!/usr/bin/env bash
# Wait until CUDA/fabricmanager is healthy, then launch the 4-GPU 8B pilot in background.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/four_gpu_pilot_watchdog.log}"
PID_FILE="${WATCH_PID_FILE:-${LOG_DIR}/four_gpu_pilot_watchdog.pid}"
STATUS_FILE="${STATUS_FILE:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-pilot10/watchdog_status.json}"
POLL_SECS="${POLL_SECS:-30}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-86400}"

mkdir -p "${LOG_DIR}" "$(dirname "${STATUS_FILE}")"

if [[ -f "${PID_FILE}" ]]; then
  old="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[ok] watchdog already running pid=${old} log=${WATCH_LOG}"
    exit 0
  fi
fi

: >"${WATCH_LOG}"

nohup bash -c "
set -euo pipefail
ROOT='${ROOT}'
STATUS_FILE='${STATUS_FILE}'
POLL_SECS='${POLL_SECS}'
MAX_WAIT_SECS='${MAX_WAIT_SECS}'
start_ts=\$(date -Iseconds)
python3 - <<PY >\"\${STATUS_FILE}\"
import json
print(json.dumps({
  'phase': 'waiting_cuda',
  'started_at': '\${start_ts}',
  'reason': 'nvidia-fabricmanager/CUDA not ready',
  'hint': 'sudo systemctl restart nvidia-fabricmanager',
}, ensure_ascii=False, indent=2))
PY
elapsed=0
while true; do
  if TRY_RESTART_FABRICMANAGER=0 bash \"\${ROOT}/training/openrlhf/ensure_cuda_ready.sh\" >>'${WATCH_LOG}' 2>&1; then
    break
  fi
  # opportunistic passwordless restart each cycle
  sudo -n systemctl restart nvidia-fabricmanager >>'${WATCH_LOG}' 2>&1 || true
  sleep \"\${POLL_SECS}\"
  elapsed=\$((elapsed + POLL_SECS))
  if [[ \"\${elapsed}\" -ge \"\${MAX_WAIT_SECS}\" ]]; then
    python3 - <<PY >\"\${STATUS_FILE}\"
import json, datetime
print(json.dumps({
  'phase': 'timeout',
  'ended_at': datetime.datetime.now().isoformat(),
  'waited_secs': \${elapsed},
}, ensure_ascii=False, indent=2))
PY
    exit 3
  fi
  python3 - <<PY >\"\${STATUS_FILE}\"
import json, datetime
print(json.dumps({
  'phase': 'waiting_cuda',
  'updated_at': datetime.datetime.now().isoformat(),
  'waited_secs': \${elapsed},
  'hint': 'sudo systemctl restart nvidia-fabricmanager',
}, ensure_ascii=False, indent=2))
PY
done

python3 - <<PY >\"\${STATUS_FILE}\"
import json, datetime
print(json.dumps({
  'phase': 'launching_pilot',
  'cuda_ready_at': datetime.datetime.now().isoformat(),
}, ensure_ascii=False, indent=2))
PY

export PHYSICS_REWARD_MODE='${PHYSICS_REWARD_MODE:-answer_low_verifier}'
export JUDGE_CUDA_DEVICE='${JUDGE_CUDA_DEVICE:-7}'
export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-0,1,2,3}'
export GENERATE_MAX_LEN='${GENERATE_MAX_LEN:-1536}'
export ROLLOUT_BATCH_SIZE='${ROLLOUT_BATCH_SIZE:-8}'
export N_SAMPLES_PER_PROMPT='${N_SAMPLES_PER_PROMPT:-2}'
export TRAIN_BATCH_SIZE='${TRAIN_BATCH_SIZE:-16}'
export MAX_SAMPLES='${MAX_SAMPLES:-512}'
export PILOT_MAX_STEPS='${PILOT_MAX_STEPS:-10}'
export DYNAMIC_FILTER_MIN='${DYNAMIC_FILTER_MIN:-0.0}'
export DYNAMIC_FILTER_MAX='${DYNAMIC_FILTER_MAX:-1.0}'
export QWEN8B_RL_CKPT='${QWEN8B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-pilot10}'
export RAY_JOB_SUBMIT_ATTEMPTS='${RAY_JOB_SUBMIT_ATTEMPTS:-5}'
export ALLOW_DIRECT_LAUNCH='${ALLOW_DIRECT_LAUNCH:-1}'

bash \"\${ROOT}/training/openrlhf/run_four_gpu_pilot_nohup.sh\" >>'${WATCH_LOG}' 2>&1
pilot_pid=\$(cat \"\${ROOT}/logs/four_gpu_pilot10.pid\" 2>/dev/null || true)
python3 - <<PY >\"\${STATUS_FILE}\"
import json, datetime
print(json.dumps({
  'phase': 'pilot_launched',
  'launched_at': datetime.datetime.now().isoformat(),
  'pilot_pid': '\${pilot_pid}',
  'pilot_log': '\${ROOT}/logs/four_gpu_pilot10.log',
}, ensure_ascii=False, indent=2))
PY

# Wait for pilot process tree to finish, then refresh admission.
while [[ -n \"\${pilot_pid}\" ]] && kill -0 \"\${pilot_pid}\" 2>/dev/null; do
  sleep 60
done
bash \"\${ROOT}/training/openrlhf/generate_four_gpu_admission.sh\" >>'${WATCH_LOG}' 2>&1 || true
python3 - <<PY >\"\${STATUS_FILE}\"
import json, datetime
from pathlib import Path
adm = Path('\${ROOT}/results/four_gpu_pilot_admission.json')
payload = {
  'phase': 'finished',
  'ended_at': datetime.datetime.now().isoformat(),
  'admission_report': str(adm),
}
if adm.is_file():
  try:
    payload['admission_pass'] = json.loads(adm.read_text(encoding='utf-8')).get('admission_pass')
    payload['global_steps'] = json.loads(adm.read_text(encoding='utf-8')).get('global_steps')
  except Exception:
    pass
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
" >>"${WATCH_LOG}" 2>&1 &

echo $! >"${PID_FILE}"
echo "[launch] cuda watchdog pid=$(cat "${PID_FILE}") log=${WATCH_LOG} status=${STATUS_FILE}"
echo "[hint] CUDA currently blocked; once fabricmanager is restarted, pilot auto-starts."
