#!/usr/bin/env bash
# 10-step pilot for 4-GPU Qwen3-8B GRPO with admission gate report.
# TRAIN_STAGE=bootstrap (default): 4-GPU Hybrid colocate, answer_only, no 30B judge.
# TRAIN_STAGE=verifier: 3 train GPUs + 1 local judge GPU.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
ENV_FILE="${WORKSPACE}/openrlhf_rl/env.sh"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

export TRAIN_STAGE="${TRAIN_STAGE:-bootstrap}"
export TRAIN_TOPOLOGY="${TRAIN_TOPOLOGY:-colocate}"

if [[ "${TRAIN_STAGE}" == "bootstrap" ]]; then
  export PHYSICS_REWARD_MODE="${PHYSICS_REWARD_MODE:-answer_only}"
  export PHYSICS_REWARD_W_FORMAT="${PHYSICS_REWARD_W_FORMAT:-0}"
  export QWEN8B_RL_CKPT="${QWEN8B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-bootstrap10}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export ACTOR_GPUS="${ACTOR_GPUS:-4}"
  export VLLM_ENGINES="${VLLM_ENGINES:-4}"
  export GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-1024}"
  export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-3}"
  export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
  export MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-2}"
  export DYNAMIC_FILTER_MODE="${DYNAMIC_FILTER_MODE:-reward_variance}"
  export DYNAMIC_FILTER_MAX_GEN_BATCHES="${DYNAMIC_FILTER_MAX_GEN_BATCHES:-32}"
  export PROMPT_DATA="${PROMPT_DATA:-${ROOT}/data/rl/bootstrap_curriculum.jsonl}"
  export MAX_SAMPLES="${MAX_SAMPLES:-2048}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
  if [[ -z "${PHYSICSVERIFIER_OPENAI_BASE_URL:-}" && -z "${PHYSICS_REWARD_MODE:-}" ]]; then
    export PHYSICS_REWARD_MODE="answer_low_verifier"
  fi
  export PHYSICS_REWARD_MODE="${PHYSICS_REWARD_MODE:-answer_low_verifier}"
  export QWEN8B_RL_CKPT="${QWEN8B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-pilot10}"
  export GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-1536}"
  export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-12}"
  export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
  export MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-2}"
  export DYNAMIC_FILTER_MODE="${DYNAMIC_FILTER_MODE:-reward_variance}"
  export DYNAMIC_FILTER_MAX_GEN_BATCHES="${DYNAMIC_FILTER_MAX_GEN_BATCHES:-32}"
  if [[ "${TRAIN_TOPOLOGY}" == "colocate" ]]; then
    export ACTOR_GPUS="${ACTOR_GPUS:-3}"
    export VLLM_ENGINES="${VLLM_ENGINES:-3}"
  else
    export ACTOR_GPUS="${ACTOR_GPUS:-2}"
    export VLLM_ENGINES="${VLLM_ENGINES:-1}"
  fi
fi

export PLOT_OUT_DIR="${PLOT_OUT_DIR:-${QWEN8B_RL_CKPT}/plots}"
export SAVE_STEPS="${SAVE_STEPS:-999}"
export PILOT_MAX_STEPS="${PILOT_MAX_STEPS:-10}"
export MAX_SAMPLES="${MAX_SAMPLES:-512}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
# Align global train batch with rollout_batch_size * n_samples_per_prompt.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
export DYNAMIC_FILTER_MIN="${DYNAMIC_FILTER_MIN:-0.0}"
export DYNAMIC_FILTER_MAX="${DYNAMIC_FILTER_MAX:-1.0}"
export JUDGE_CUDA_DEVICE="${JUDGE_CUDA_DEVICE:-3}"
export ALLOW_RAY_JOBS="${ALLOW_RAY_JOBS:-0}"
export RAY_JOB_SUBMIT_ATTEMPTS="${RAY_JOB_SUBMIT_ATTEMPTS:-1}"
export ALLOW_DIRECT_LAUNCH="${ALLOW_DIRECT_LAUNCH:-1}"
export RAY_GCS_PORT="${RAY_GCS_PORT:-26379}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-28265}"
export PHYSICS_REWARD_METRICS_LOG="${PHYSICS_REWARD_METRICS_LOG:-${QWEN8B_RL_CKPT}/plots/physics_reward_metrics.jsonl}"

OUT="${OUT:-${ROOT}/results/four_gpu_pilot_admission.json}"
STATUS_OUT="${STATUS_OUT:-${QWEN8B_RL_CKPT}/pilot_status.json}"
mkdir -p "$(dirname "${OUT}")" "${PLOT_OUT_DIR}" "${QWEN8B_RL_CKPT}"

PYTHON="${PYTHON:-${TRAIN_VENV:-/data1/jinjianhan/venv/openrlhf_train}/bin/python}"

bash "${ROOT}/training/openrlhf/download_qwen3_8b.sh"
bash "${ROOT}/training/openrlhf/ensure_cuda_ready.sh"
bash "${ROOT}/training/openrlhf/start_local_judge_if_needed.sh"
# 4-GPU pilot defaults to local judge reward; ignore inherited external API env.
if [[ "${PHYSICS_REWARD_MODE}" != "answer_only" ]]; then
  # Restart reward if an old external-mode server is still up.
  if [[ -f "${ROOT}/logs/physics_reward_server.mode" ]] && grep -q external "${ROOT}/logs/physics_reward_server.mode" 2>/dev/null; then
    if [[ -f "${ROOT}/logs/physics_reward_server.pid" ]]; then
      _rp="$(cat "${ROOT}/logs/physics_reward_server.pid" 2>/dev/null || true)"
      if [[ -n "${_rp}" ]] && kill -0 "${_rp}" 2>/dev/null; then
        kill -TERM "${_rp}" 2>/dev/null || true
        sleep 2
        kill -9 "${_rp}" 2>/dev/null || true
      fi
      rm -f "${ROOT}/logs/physics_reward_server.pid" "${ROOT}/logs/physics_reward_server.mode"
    fi
  fi
  env -u PHYSICSVERIFIER_OPENAI_BASE_URL -u PHYSICSVERIFIER_OPENAI_API_KEY \
    PHYSICS_REWARD_MODE="${PHYSICS_REWARD_MODE}" \
    OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8766/v1}" \
    OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" \
    PHYSICSVERIFIER_LLM_MODEL="${PHYSICSVERIFIER_LLM_MODEL:-qwen3-30b-a3b}" \
    bash "${ROOT}/training/reward_server/start_reward_server.sh"
else
  bash "${ROOT}/training/reward_server/start_reward_server.sh"
fi
bash "${ROOT}/training/openrlhf/check_prerequisites_4gpu.sh"
bash "${ROOT}/training/openrlhf/watch_training_curves.sh" stop 2>/dev/null || true
export QWEN8B_RL_CKPT="${QWEN8B_RL_CKPT}"
export PLOT_OUT_DIR="${PLOT_OUT_DIR}"
bash "${ROOT}/training/openrlhf/watch_training_curves.sh" start

START_TS="$(date -Iseconds)"
python3 - <<PY >"${STATUS_OUT}"
import json, datetime, os
print(json.dumps({
  "phase": "training",
  "started_at": "${START_TS}",
  "ckpt": "${QWEN8B_RL_CKPT}",
  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
  "judge_cuda_device": os.environ.get("JUDGE_CUDA_DEVICE", ""),
  "train_topology": os.environ.get("TRAIN_TOPOLOGY", ""),
  "actor_gpus": int(os.environ.get("ACTOR_GPUS", "0") or 0),
  "vllm_engines": int(os.environ.get("VLLM_ENGINES", "0") or 0),
  "reward_mode": os.environ.get("PHYSICS_REWARD_MODE", ""),
  "pilot_max_steps": int("${PILOT_MAX_STEPS}"),
  "generate_max_len": int("${GENERATE_MAX_LEN}"),
  "rollout_batch_size": int("${ROLLOUT_BATCH_SIZE}"),
  "n_samples_per_prompt": int("${N_SAMPLES_PER_PROMPT}"),
  "train_batch_size": int("${TRAIN_BATCH_SIZE}"),
  "train_stage": os.environ.get("TRAIN_STAGE", ""),
  "dynamic_filtering_mode": os.environ.get("DYNAMIC_FILTER_MODE", ""),
  "vllm_gpu_memory_utilization": float("${VLLM_GPU_MEMORY_UTILIZATION}"),
}, ensure_ascii=False, indent=2))
PY

run_train() {
  bash "${ROOT}/training/openrlhf/run-qwen3-8b-physics-4gpu-openrlhf.sh"
}

set +e
run_train
TRAIN_RC=$?
set -e

FALLBACK_USED=0
FALLBACK_REASON=""
if [[ "${TRAIN_RC}" -eq 42 && "${TRAIN_TOPOLOGY}" == "colocate" ]]; then
  FALLBACK_USED=1
  FALLBACK_REASON="colocate_oom_or_ipc"
  echo "[fallback] colocate -> split on same CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  export TRAIN_TOPOLOGY=split
  IFS=',' read -ra _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  if [[ "${#_GPU_ARR[@]}" -ge 4 ]]; then
    export ACTOR_GPUS=3
    export VLLM_ENGINES=1
  else
    export ACTOR_GPUS=2
    export VLLM_ENGINES=1
  fi
  export VLLM_GPU_MEMORY_UTILIZATION="${SPLIT_VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
  export ROLLOUT_BATCH_SIZE="${SPLIT_ROLLOUT_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
  export N_SAMPLES_PER_PROMPT="${SPLIT_N_SAMPLES_PER_PROMPT:-${N_SAMPLES_PER_PROMPT}}"
  export TRAIN_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
  export MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-1}"
  rm -f "${QWEN8B_RL_CKPT}/fallback_request.json"
  python3 - <<PY >"${QWEN8B_RL_CKPT}/fallback_status.json"
import json, datetime, os
print(json.dumps({
  "fallback_used": True,
  "from": "colocate",
  "to": "split",
  "reason": "${FALLBACK_REASON}",
  "at": datetime.datetime.now().isoformat(),
  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
  "judge_cuda_device": os.environ.get("JUDGE_CUDA_DEVICE", ""),
}, ensure_ascii=False, indent=2))
PY
  set +e
  run_train
  TRAIN_RC=$?
  set -e
fi

END_TS="$(date -Iseconds)"

"${PYTHON}" "${ROOT}/training/openrlhf/plot_training_curves.py" \
  --save-path "${QWEN8B_RL_CKPT}" \
  --out-dir "${PLOT_OUT_DIR}" \
  --no-sync-ray || true

python3 - <<PY
import json, subprocess
from pathlib import Path
path = Path("${QWEN8B_RL_CKPT}") / "gpu_util_snapshot.json"
rows = []
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in out.strip().splitlines():
        i, used, free, util = [x.strip() for x in line.split(",")]
        rows.append({"index": int(i), "mem_used_mib": int(used), "mem_free_mib": int(free), "util_pct": float(util)})
except Exception:
    rows = []
path.write_text(json.dumps(rows), encoding="utf-8")
PY

cuda_ok=0
if TRY_RESTART_FABRICMANAGER=0 bash "${ROOT}/training/openrlhf/ensure_cuda_ready.sh" >/dev/null 2>&1; then
  cuda_ok=1
fi
fm_active=0
if systemctl is-active nvidia-fabricmanager >/dev/null 2>&1; then
  fm_active=1
fi

"${PYTHON}" "${ROOT}/training/openrlhf/admission_report.py" \
  --ckpt "${QWEN8B_RL_CKPT}" \
  --out "${OUT}" \
  --target-steps "${PILOT_MAX_STEPS}" \
  --train-rc "${TRAIN_RC}" \
  --cuda-ok "${cuda_ok}" \
  --fm-active "${fm_active}" \
  --train-stage "${TRAIN_STAGE}" \
  --train-topology "${TRAIN_TOPOLOGY}" \
  --start "${START_TS}" \
  --end "${END_TS}" \
  --fallback-used "${FALLBACK_USED}" \
  --fallback-reason "${FALLBACK_REASON}"

python3 - <<PY >"${STATUS_OUT}"
import json
from pathlib import Path
adm = json.loads(Path("${OUT}").read_text(encoding="utf-8"))
print(json.dumps({
  "phase": "finished",
  "ended_at": "${END_TS}",
  "train_rc": int("${TRAIN_RC}"),
  "train_topology": "${TRAIN_TOPOLOGY}",
  "train_stage": "${TRAIN_STAGE}",
  "fallback_used": bool(int("${FALLBACK_USED}")),
  "admission_pass": adm.get("admission_pass"),
  "global_steps": adm.get("global_steps"),
  "last_step_num": adm.get("last_step_num"),
  "verifier_stage_ready": (adm.get("verifier_stage_gate") or {}).get("ready"),
  "admission_report": "${OUT}",
}, ensure_ascii=False, indent=2))
PY

echo "[ok] pilot admission report -> ${OUT}"
exit "${TRAIN_RC}"
