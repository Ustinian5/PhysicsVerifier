#!/usr/bin/env bash
# Orchestrate remaining SFT -> smoke -> calib -> full process-RL -> final eval.
# Assumes eval matrix and/or SFT datagen may already be running.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
SFT_DATA="${SFT_DATA:-${ROOT}/data/rl/sft_solutions.jsonl}"
SFT_CKPT="${QWEN8B_SFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-sft}"
SWIFT_CKPT="${QWEN8B_SWIFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-swift}"
DATAGEN_PID_FILE="${DATAGEN_PID_FILE:-${ROOT}/logs/sft_datagen.pid}"
EVAL_PID_FILE="${EVAL_PID_FILE:-${ROOT}/logs/hipho_matrix_8b.pid}"
PIPE_LOG="${PIPE_LOG:-${SFT_CKPT}/sft_selfjudge_pipeline.log}"
MIN_SFT_ROWS="${MIN_SFT_ROWS:-200}"
TARGET_SFT_ROWS="${TARGET_SFT_ROWS:-1400}"
mkdir -p "${SFT_CKPT}" "${SWIFT_CKPT}" "$(dirname "${PIPE_LOG}")"

log() { echo "[pipeline $(date -Iseconds)] $*" | tee -a "${PIPE_LOG}"; }

alive() {
  local f="$1"
  [[ -f "${f}" ]] || return 1
  local pid
  pid="$(cat "${f}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

wait_pidfile() {
  local f="$1" name="$2"
  if ! alive "${f}"; then
    return 0
  fi
  log "waiting for ${name} pid=$(cat "${f}")"
  while alive "${f}"; do
    sleep 60
  done
  log "${name} finished"
}

wait_gpus_free() {
  local ids="$1"
  log "waiting for GPUs ${ids} to be free (<2GiB used)"
  while true; do
    busy=0
    IFS=',' read -ra gs <<< "${ids}"
    for g in "${gs[@]}"; do
      mem="$(nvidia-smi --id="${g}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
      if [[ "${mem}" =~ ^[0-9]+$ ]] && [[ "${mem}" -gt 2000 ]]; then
        busy=1
        break
      fi
    done
    if [[ "${busy}" -eq 0 ]]; then
      log "GPUs ${ids} free"
      return 0
    fi
    sleep 120
  done
}

n_sft() { [[ -f "${SFT_DATA}" ]] && wc -l < "${SFT_DATA}" | tr -d ' ' || echo 0; }

log "waiting for local SFT datagen (>=${TARGET_SFT_ROWS} rows or process exit, min ${MIN_SFT_ROWS})"
while true; do
  n="$(n_sft)"
  log "sft_solutions.jsonl rows=${n}"
  if ! alive "${DATAGEN_PID_FILE}"; then
    break
  fi
  if [[ "${n}" -ge "${TARGET_SFT_ROWS}" ]]; then
    log "reached target rows; still waiting for datagen/API fill to finish"
    wait_pidfile "${DATAGEN_PID_FILE}" sft_datagen
    break
  fi
  sleep 120
done
n="$(n_sft)"
if [[ "${n}" -lt "${MIN_SFT_ROWS}" ]]; then
  log "ERROR SFT data too small: ${n}"
  exit 2
fi

if [[ ! -f "${SFT_CKPT}/config.json" ]] && ! ls -d "${SFT_CKPT}"/v*-*/checkpoint-* >/dev/null 2>&1; then
  wait_gpus_free "${SFT_GPUS:-0,1,2,3}"
  log "starting SFT on GPUs ${SFT_GPUS:-0,1,2,3}"
  bash "${ROOT}/training/swift/run_swift_sft_8b.sh"
  sft_pid="$(cat "${SFT_CKPT}/swift_sft.pid")"
  while kill -0 "${sft_pid}" 2>/dev/null; do
    sleep 60
  done
  log "SFT process exited"
else
  log "SFT ckpt already present; skip train"
fi

wait_pidfile "${EVAL_PID_FILE}" baseline_eval
# GPU 7 must be free before gate eval.
log "SFT heldout gate"
if ! CUDA_DEVICE=7 PORT=8766 bash "${ROOT}/training/swift/run_sft_gate_eval.sh"; then
  log "ERROR SFT failed heldout gate; stop before RL"
  exit 3
fi

log "self-judge smoke (8B local vs 30B API)"
SMOKE_JUDGE_GPU=0 bash "${ROOT}/training/swift/run_self_judge_smoke.sh" || log "smoke finished with warning"

SMOKE="${SWIFT_CKPT}/self_judge_smoke.json"
if [[ -f "${SMOKE}" ]]; then
  pass="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pass", False))' "${SMOKE}")"
  if [[ "${pass}" != "True" ]]; then
    export JUDGE_REFRESH=0
    log "low 8B/30B agreement; freeze judge at SFT ckpt (JUDGE_REFRESH=0)"
  fi
fi

log "reward-calib 10-step pilot"
wait_gpus_free "0,1,2,3,4,5,6,7"
bash "${ROOT}/training/swift/run_reward_calib_pilot.sh"
pilot_pid="$(cat "${ROOT}/logs/swift_grpo.pid" 2>/dev/null || true)"
if [[ -n "${pilot_pid}" ]]; then
  while kill -0 "${pilot_pid}" 2>/dev/null; do sleep 60; done
fi
sleep 10
log_jsonl="$(ls -t "${SWIFT_CKPT}"/v*-*/logging.jsonl 2>/dev/null | head -1 || true)"
if [[ -n "${log_jsonl}" ]]; then
  "${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}" \
    "${ROOT}/training/swift/monitor_process_reward.py" \
    --logging-jsonl "${log_jsonl}" \
    --output "${SWIFT_CKPT}/calib_monitor.json" || true
fi

log "full GRPO from SFT (process reward only)"
wait_gpus_free "0,1,2,3,4,5,6,7"
export MAX_STEPS="${MAX_STEPS:-0}"
export JUDGE_REFRESH="${JUDGE_REFRESH:-1}"
bash "${ROOT}/training/swift/launch_swift_grpo_8gpu.sh"
train_pid="$(cat "${ROOT}/logs/swift_grpo.pid")"
last_recap=""
while kill -0 "${train_pid}" 2>/dev/null; do
  latest="$(ls -d "${SWIFT_CKPT}"/v*-*/checkpoint-* 2>/dev/null | tail -1 || true)"
  if [[ -n "${latest}" && "${latest}" != "${last_recap}" ]]; then
    comps="$(dirname "${latest}")/completions.jsonl"
    if [[ -s "${comps}" ]]; then
      bash "${ROOT}/training/swift/recap_30b_api.sh" "${comps}" "${SWIFT_CKPT}/recap_30b.json" || true
      last_recap="${latest}"
    fi
  fi
  sleep 300
done
log "full GRPO exited"

log "final eval matrix + effectiveness analysis"
bash "${ROOT}/training/swift/run_final_eval_matrix.sh"
log "pipeline complete"
