#!/usr/bin/env bash
# 4-GPU Swift GRPO with DeepSeek-only llm_step_score rewards. Does not start local judges.
set -euo pipefail
ulimit -f unlimited 2>/dev/null || true
SLOW_TMP_ROOT="${SLOW_TMP_ROOT:-/slow_share/jinjianhan/tmp}"
export TMPDIR="${TMPDIR:-${SLOW_TMP_ROOT}/swift}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
mkdir -p "${TMPDIR}"

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
export PHYSICS_ROOT="${ROOT}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  unset CUDA_VISIBLE_DEVICES
fi
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
MODE="${MODE:-full}"  # smoke | full
if [[ "${MODE}" == "smoke" ]]; then
  CKPT="${QWEN8B_LLM_VERIFIER_SMOKE_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-deepseek-v4-flash-grpo-onset-smoke}"
  MAX_STEPS="${MAX_STEPS:-2}"
  SAVE_STEPS="${SAVE_STEPS:-2}"
else
  CKPT="${QWEN8B_LLM_VERIFIER_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-deepseek-v4-flash-grpo-onset}"
  MAX_STEPS="${MAX_STEPS:-100}"
  SAVE_STEPS="${SAVE_STEPS:-10}"
fi
SWIFT_VENV="${SWIFT_VENV:-/data1/jinjianhan/venv/swift_train}"
ORHF_PYTHON="${ORHF_PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
SWIFT_PID_FILE="${SWIFT_PID_FILE:-${LOG_DIR}/swift_llm_verifier_grpo.pid}"
REWARD_PID_FILE="${REWARD_PID_FILE:-${LOG_DIR}/physics_reward_server_llm_step.pid}"
PID_FILE="${SWIFT_PID_FILE}"
LOG_FILE="${LOG_FILE:-${CKPT}/swift_grpo.log}"
REPORT="${REPORT:-${CKPT}/swift_launch_report.json}"
SMOKE_CKPT="${QWEN8B_LLM_VERIFIER_SMOKE_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-deepseek-v4-flash-grpo-onset-smoke}"
MODEL_DIR="${QWEN8B_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-8B}"
PROMPT_DATA="${PROMPT_DATA:-${ROOT}/data/rl/swift_prompts_max2048.jsonl}"
PLUGIN="${PLUGIN:-${ROOT}/training/swift/llm_step_reward_plugin.py}"
FREE_MIB="${FREE_MIB:-75000}"
UTIL_MAX="${UTIL_MAX:-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-6}"
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-3}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-1536}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
NPROC="${NPROC_PER_NODE:-4}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.30}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
MAX_RESAMPLE_TIMES="${MAX_RESAMPLE_TIMES:-2}"
SEED="${SEED:-42}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-12}"
REWARD_PORT="${REWARD_PORT:-8771}"
CALIB_REPORT="${CALIB_REPORT:-${ROOT}/logs/llm_step_judge_calibration.json}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${ROOT}/data/rl/train_manifest.json}"
HIPHO_JSONL="${HIPHO_JSONL:-/slow_share/jinjianhan/workspace/benchmarks/hipho/hipho_text_only.jsonl}"
HELDOUT="${HELDOUT:-${ROOT}/data/rl/heldout_eval.jsonl}"
SKIP_CALIBRATION="${SKIP_CALIBRATION:-0}"

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

[[ -x "${SWIFT_VENV}/bin/swift" ]] || refuse "missing swift binary"
[[ -f "${MODEL_DIR}/config.json" ]] || refuse "missing base model ${MODEL_DIR}"
[[ -s "${PROMPT_DATA}" ]] || refuse "missing prompt data ${PROMPT_DATA}"
[[ -f "${PLUGIN}" ]] || refuse "missing plugin ${PLUGIN}"

if [[ "${MODEL_DIR}" != *"/Qwen3-8B" && "${MODEL_DIR}" != *"/Qwen3-8B/" ]]; then
  echo "[warn] MODEL_DIR is ${MODEL_DIR}; this experiment should start from base Qwen3-8B"
fi

avail_kb="$(df -Pk "${CKPT}" | awk 'NR==2{print $4}')"
if [[ -n "${avail_kb}" && "${avail_kb}" -lt 50000000 ]]; then
  refuse "insufficient disk on $(dirname "${CKPT}"): ${avail_kb} KiB"
fi

if alive_pid_file "${SWIFT_PID_FILE}" || alive_pid_file "${CKPT}/swift_train.pid"; then
  refuse "stale_or_live_pid: llm verifier training still running"
fi

LOCK_FILE="${CKPT}/launch.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  refuse "another launch holds ${LOCK_FILE}"
fi
if pgrep -f "${CKPT}/run_swift.sh" >/dev/null 2>&1; then
  refuse "leftover run_swift.sh is still running under ${CKPT}; not starting a second copy"
fi

pick_master_port() {
  local p
  for p in 29511 29512 29513 29514 29515 29516 29517 29518 29611; do
    if ! ss -ltn 2>/dev/null | grep -qE ":${p}[[:space:]]"; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}
MASTER_PORT="${MASTER_PORT:-$(pick_master_port)}" || refuse "no free MASTER_PORT"
export MASTER_PORT

bash "${ROOT}/training/openrlhf/ensure_cuda_ready.sh" || refuse "CUDA not ready"

python3 "${ROOT}/training/rl_data/audit_eval_leakage.py" \
  --train "${PROMPT_DATA}" \
  --heldout "${HELDOUT}" \
  --hipho "${HIPHO_JSONL}" \
  --manifest "${TRAIN_MANIFEST}" \
  --fail-on-exact || refuse "eval leakage audit failed"

if [[ "${SKIP_CALIBRATION}" != "1" && "${MODE}" == "full" ]]; then
  if [[ ! -f "${CALIB_REPORT}" ]]; then
    refuse "calibration report missing; run training/swift/calibrate_llm_step_judge.py first"
  fi
  python3 - "${CALIB_REPORT}" <<'PY' || refuse "calibration gate failed"
import json, sys
rep = json.loads(open(sys.argv[1], encoding="utf-8").read())
if not rep.get("ok"):
    sys.exit(2)
PY
fi

if [[ "${MODE}" == "full" && "${SKIP_SMOKE_GATE:-0}" != "1" ]]; then
  SMOKE_CKPT="${SMOKE_CKPT}" python3 - <<'PY' || refuse "2-step smoke has not passed; rerun MODE=smoke first"
import json, os, sys
from pathlib import Path
root = Path(os.environ["SMOKE_CKPT"])
logs = sorted(root.glob("v*/logging.jsonl"), key=lambda p: p.stat().st_mtime)
if not logs:
    sys.exit(2)
steps = set()
for line in logs[-1].read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    gs = str(obj.get("global_step/max_steps") or "")
    if "/" in gs:
        steps.add(int(gs.split("/")[0]))
if 2 not in steps:
    sys.exit(2)
PY
fi

probe="$("${ORHF_PYTHON}" "${ROOT}/training/openrlhf/gpu_bundle_utils.py" probe --train-only --n-train 4 --free-mib "${FREE_MIB}" --util-max "${UTIL_MAX}")"
echo "${probe}" >"${CKPT}/gpu_selection.json"
ok="$(python3 -c 'import json,sys; print(int(json.loads(sys.stdin.read()).get("ok", False)))' <<<"${probe}")"
if [[ "${ok}" != "1" ]]; then
  reason="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("reason",""))' <<<"${probe}")"
  refuse "need_4_idle_train_gpus: ${reason}"
fi
train_gpus="$(python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(",".join(str(x) for x in d["train_gpus"]))' <<<"${probe}")"

export PHYSICS_REWARD_MODE=llm_step_score
export PHYSICSVERIFIER_LLM_MODEL=deepseek-v4-flash
export LLM_STEP_JUDGE_TIMEOUT=300
export LLM_STEP_JUDGE_MAX_TOKENS="${LLM_STEP_JUDGE_MAX_TOKENS:-4096}"
export LLM_STEP_JUDGE_MAX_RETRIES="${LLM_STEP_JUDGE_MAX_RETRIES:-6}"
export LLM_STEP_JUDGE_CONCURRENCY="${LLM_STEP_JUDGE_CONCURRENCY:-32}"
export PHYSICS_REWARD_HTTP_RETRIES="${PHYSICS_REWARD_HTTP_RETRIES:-5}"
export PHYSICS_REWARD_CONCURRENCY="${PHYSICS_REWARD_CONCURRENCY:-32}"
export PHYSICS_REWARD_CACHE_SIZE="${PHYSICS_REWARD_CACHE_SIZE:-4096}"
export HOST=127.0.0.1
export PORT="${REWARD_PORT}"
export PID_FILE="${REWARD_PID_FILE}"
export LOG="${CKPT}/physics_reward_server.log"
export VENV="${VENV:-${ROOT}/.venv}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  export VENV="/data1/jinjianhan/venv/openrlhf_train"
fi
bash "${ROOT}/training/reward_server/start_reward_server.sh" || refuse "reward server failed"

export PHYSICS_REWARD_URL="http://127.0.0.1:${REWARD_PORT}/get_reward"
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
SWIFT_TMP="${SWIFT_TMP:-${SLOW_TMP_ROOT}/swift}"
mkdir -p "${SWIFT_TMP}" "${SWIFT_TMP}/hf_datasets" "${SWIFT_TMP}/hf_home"
export TMPDIR="${SWIFT_TMP}" TEMP="${SWIFT_TMP}" TMP="${SWIFT_TMP}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SWIFT_TMP}/hf_datasets}"
export HF_HOME="${HF_HOME:-${SWIFT_TMP}/hf_home}"

echo "[launch] llm_step GRPO mode=${MODE} gpus=${train_gpus} steps=${MAX_STEPS} ckpt=${CKPT}"
echo "[wrapper] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting swift rlhf mode=${MODE} gpus=${train_gpus} steps=${MAX_STEPS}" >>"${LOG_FILE}"
export LOG_FILE
export RESUME_FROM="${RESUME_FROM:-}"
export PHYSICS_REWARD_HTTP_RETRIES="${PHYSICS_REWARD_HTTP_RETRIES:-5}"
RUN_SH="${CKPT}/run_swift.sh"
python3 - "${RUN_SH}" "${SWIFT_PID_FILE}" "${CKPT}/swift_train.pid" "${train_gpus}" "${NPROC}" "${ROOT}" "${PHYSICS_REWARD_URL}" "${PHYSICS_REWARD_TIMEOUT}" "${CUDA_HOME}" "${PATH}" "${DS_SKIP_CUDA_CHECK}" "${TMPDIR}" "${HF_DATASETS_CACHE}" "${HF_HOME}" "${SWIFT_VENV}" "${MODEL_DIR}" "${PLUGIN}" "${VLLM_GPU_UTIL}" "${MAX_LENGTH}" "${VLLM_MAX_NUM_SEQS}" "${PROMPT_DATA}" "${MAX_COMPLETION_LEN}" "${PER_DEVICE_TRAIN_BS}" "${GRAD_ACCUM}" "${NUM_GENERATIONS}" "${SEED}" "${MAX_RESAMPLE_TIMES}" "${SAVE_STEPS}" "${SAVE_TOTAL_LIMIT}" "${CKPT}" "${MAX_STEPS}" <<'PY'
import os, sys, textwrap, pathlib
out = pathlib.Path(sys.argv[1])
pid_file, train_pid = sys.argv[2], sys.argv[3]
vals = sys.argv[4:]
keys = [
    "train_gpus","nproc","root","reward_url","reward_timeout","cuda_home","path",
    "ds_skip","tmpdir","hf_datasets","hf_home","swift_venv","model_dir","plugin",
    "vllm_util","max_length","vllm_seqs","prompt_data","max_comp","per_device_bs",
    "grad_accum","num_gen","seed","max_resample","save_steps","save_total","ckpt","max_steps",
]
env = dict(zip(keys, vals))
log_file = os.environ["LOG_FILE"]
http_retries = os.environ.get("PHYSICS_REWARD_HTTP_RETRIES", "5")
master_port = os.environ.get("MASTER_PORT", "29511")
resume_from = os.environ.get("RESUME_FROM", "").strip()
resume_flags = ""
if resume_from:
    resume_flags = (
        f'    --resume_from_checkpoint "{resume_from}" \\\n'
        "    --resume_only_model true \\\n"
        "    --load_args false \\\n"
    )
script = f'''#!/usr/bin/env bash
exec >>"{log_file}" 2>&1
if [[ -s "{pid_file}" ]]; then
  old="$(cat "{pid_file}" 2>/dev/null || true)"
  if [[ -n "${{old}}" && "${{old}}" != "$$" ]] && kill -0 "${{old}}" 2>/dev/null; then
    echo "[wrapper] pid=$$ refusing to start; live wrapper ${{old}} owns {pid_file}"
    exit 0
  fi
fi
echo $$ >"{pid_file}"
echo $$ >"{train_pid}"
echo "[wrapper] pid=$$ starting /usr/bin/env swift $(date -u +%Y-%m-%dT%H:%M:%SZ)"
trap '' HUP
/usr/bin/env \\
  CUDA_VISIBLE_DEVICES="{env["train_gpus"]}" \\
  NPROC_PER_NODE="{env["nproc"]}" \\
  MASTER_ADDR=127.0.0.1 \\
  MASTER_PORT="{master_port}" \\
  PYTHONPATH="{env["root"]}:${{PYTHONPATH:-}}" \\
  PHYSICS_REWARD_URL="{env["reward_url"]}" \\
  PHYSICS_REWARD_TIMEOUT="{env["reward_timeout"]}" \\
  PHYSICS_REWARD_HTTP_RETRIES="{http_retries}" \\
  CUDA_HOME="{env["cuda_home"]}" \\
  PATH="{env["path"]}" \\
  DS_SKIP_CUDA_CHECK="{env["ds_skip"]}" \\
  TRL_EXPERIMENTAL_SILENCE=1 \\
  PYTHONUNBUFFERED=1 \\
  PYTHONFAULTHANDLER=1 \\
  TOKENIZERS_PARALLELISM=false \\
  CUDA_DEVICE_MAX_CONNECTIONS=1 \\
  NCCL_CUMEM_ENABLE=0 \\
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8" \\
  TMPDIR="{env["tmpdir"]}" TEMP="{env["tmpdir"]}" TMP="{env["tmpdir"]}" \\
  HF_DATASETS_CACHE="{env["hf_datasets"]}" HF_HOME="{env["hf_home"]}" \\
  "{env["swift_venv"]}/bin/swift" rlhf \\
    --rlhf_type grpo \\
    --model "{env["model_dir"]}" \\
    --external_plugins "{env["plugin"]}" \\
    --reward_funcs llm_step_verifier \\
    --use_vllm true \\
    --vllm_mode colocate \\
    --vllm_gpu_memory_utilization "{env["vllm_util"]}" \\
    --vllm_tensor_parallel_size 1 \\
    --vllm_max_model_len "{env["max_length"]}" \\
    --vllm_max_num_seqs "{env["vllm_seqs"]}" \\
    --vllm_enforce_eager true \\
    --vllm_enable_prefix_caching true \\
    --sleep_level 0 \\
    --offload_model true \\
    --offload_optimizer true \\
    --tuner_type full \\
    --torch_dtype bfloat16 \\
    --attn_impl sdpa \\
    --dataset "{env["prompt_data"]}" \\
    --max_completion_length "{env["max_comp"]}" \\
    --max_length "{env["max_length"]}" \\
    --num_train_epochs 1 \\
    --per_device_train_batch_size "{env["per_device_bs"]}" \\
    --gradient_accumulation_steps "{env["grad_accum"]}" \\
    --learning_rate 1e-6 \\
    --epsilon 0.2 \\
    --beta 0.01 \\
    --temperature 1.0 \\
    --num_generations "{env["num_gen"]}" \\
    --seed "{env["seed"]}" \\
    --dynamic_sample true \\
    --max_resample_times "{env["max_resample"]}" \\
    --eval_strategy no \\
    --save_steps "{env["save_steps"]}" \\
    --save_only_model true \\
    --save_total_limit "{env["save_total"]}" \\
    --logging_steps 1 \\
    --gradient_checkpointing true \\
    --deepspeed zero3 \\
    --report_to tensorboard \\
    --logging_dir "{env["ckpt"]}/runs" \\
    --output_dir "{env["ckpt"]}" \\
    --log_completions true \\
    --dataloader_num_workers 0 \\
    --use_hf true \\
    --overlong_filter false \\
{resume_flags}    --max_steps "{env["max_steps"]}"
status=$?
echo "[wrapper] swift exited status=${{status}} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit ${{status}}
'''
out.write_text(script)
os.chmod(out, 0o755)
print(f"[ok] wrote {out}")
PY
UNIT_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
mkdir -p "${UNIT_DIR}"
UNIT_NAME="llm-verifier-grpo-${MODE}.service"
cat >"${UNIT_DIR}/${UNIT_NAME}" <<EOF
[Unit]
Description=PhysicsVerifier llm_step GRPO (${MODE})
After=default.target

[Service]
Type=simple
KillMode=none
RemainAfterExit=yes
TasksMax=infinity
LimitNOFILE=1048576
WorkingDirectory=${ROOT}
Environment=HOME=${HOME}
Environment=USER=${USER}
Environment=LANG=C.UTF-8
ExecStart=/bin/bash ${RUN_SH}
Restart=no
EOF
systemctl --user daemon-reload
if systemctl --user is-active --quiet "${UNIT_NAME}"; then
  refuse "systemd unit ${UNIT_NAME} is already active"
fi
systemctl --user reset-failed "${UNIT_NAME}" 2>/dev/null || true
systemctl --user start "${UNIT_NAME}"
echo "[launch] detached via systemctl --user start ${UNIT_NAME} master_port=${MASTER_PORT}"
for _ in $(seq 1 40); do
  if [[ -s "${SWIFT_PID_FILE}" ]] && kill -0 "$(cat "${SWIFT_PID_FILE}" 2>/dev/null)" 2>/dev/null; then
    break
  fi
  sleep 0.25
done
sleep 8
if [[ ! -s "${SWIFT_PID_FILE}" ]] || ! kill -0 "$(cat "${SWIFT_PID_FILE}" 2>/dev/null)" 2>/dev/null; then
  refuse "swift died during startup; see ${LOG_FILE}"
fi

export CKPT TRAIN_GPUS="${train_gpus}" SWIFT_PID_FILE LOG_FILE REPORT MAX_STEPS MODE MODEL_DIR PROMPT_DATA SEED PER_DEVICE_TRAIN_BS GRAD_ACCUM SAVE_STEPS SAVE_TOTAL_LIMIT
python3 - <<'PY' >"${REPORT}"
import hashlib, json, datetime, os, subprocess
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

git = ""
try:
    git = subprocess.check_output(["git", "-C", os.environ.get("PHYSICS_ROOT", "."), "rev-parse", "HEAD"], text=True).strip()
except Exception:
    pass
print(json.dumps({
  "ok": True,
  "phase": "launched",
  "reason": "llm_step_verifier_grpo_4gpu",
  "mode": os.environ.get("MODE"),
  "at": datetime.datetime.utcnow().isoformat() + "Z",
  "ckpt": os.environ["CKPT"],
  "pid": int(open(os.environ["SWIFT_PID_FILE"]).read().strip()),
  "log": os.environ["LOG_FILE"],
  "cuda_visible_devices": os.environ["TRAIN_GPUS"],
  "judge_gpus": [],
  "max_steps": int(os.environ["MAX_STEPS"]),
  "save_steps": int(os.environ.get("SAVE_STEPS", "10")),
  "save_total_limit": int(os.environ.get("SAVE_TOTAL_LIMIT", "12")),
  "model_dir": os.environ.get("MODEL_DIR"),
  "prompt_data": os.environ.get("PROMPT_DATA"),
  "prompt_sha256": sha256(os.environ["PROMPT_DATA"]) if Path(os.environ["PROMPT_DATA"]).is_file() else "",
  "git_commit": git,
  "seed": int(os.environ.get("SEED", "42")),
  "num_generations": 6,
  "max_completion_length": 1536,
  "max_length": 4096,
  "per_device_train_batch_size": int(os.environ.get("PER_DEVICE_TRAIN_BS", "2")),
  "gradient_accumulation_steps": int(os.environ.get("GRAD_ACCUM", "3")),
  "learning_rate": 1e-6,
  "epsilon": 0.2,
  "beta": 0.01,
  "reward_mode": "llm_step_score",
  "judge_model": "deepseek-v4-flash",
  "prompt_version": "llm_step_v1",
}, ensure_ascii=False, indent=2))
PY

nohup python3 "${ROOT}/training/swift/monitor_llm_step_reward.py" \
  --metrics "${ROOT}/logs/physics_reward_metrics.jsonl" \
  --train-log "${LOG_FILE}" \
  --pid-file "${SWIFT_PID_FILE}" \
  >>"${CKPT}/monitor.log" 2>&1 &
echo $! >"${CKPT}/monitor.pid"

echo "[launch] pid=$(cat "${SWIFT_PID_FILE}") log=${LOG_FILE} train=${train_gpus} report=${REPORT}"
