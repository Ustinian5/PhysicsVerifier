#!/usr/bin/env bash
# Full-parameter SFT of Qwen3-8B on rejection-sampled solutions. Isolated from OpenRLHF venv.
set -euo pipefail
ulimit -f unlimited 2>/dev/null || true
SLOW_TMP_ROOT="${SLOW_TMP_ROOT:-/slow_share/jinjianhan/tmp}"
export TMPDIR="${TMPDIR:-${SLOW_TMP_ROOT}/swift}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
mkdir -p "${TMPDIR}"

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
SWIFT_VENV="${SWIFT_VENV:-/data1/jinjianhan/venv/swift_train}"
MODEL_DIR="${QWEN8B_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-8B}"
CKPT="${QWEN8B_SFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-sft}"
DATA="${SFT_DATA:-${ROOT}/data/rl/sft_solutions.jsonl}"
LOG_FILE="${LOG_FILE:-${CKPT}/swift_sft.log}"
NPROC="${NPROC_PER_NODE:-4}"
TRAIN_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

[[ -x "${SWIFT_VENV}/bin/swift" ]] || { echo "[error] missing ${SWIFT_VENV}/bin/swift" >&2; exit 2; }
[[ -s "${DATA}" ]] || { echo "[error] missing SFT data ${DATA}" >&2; exit 2; }
[[ -f "${MODEL_DIR}/config.json" ]] || { echo "[error] missing model ${MODEL_DIR}" >&2; exit 2; }
n="$(wc -l < "${DATA}")"
if [[ "${n}" -lt 200 ]]; then
  echo "[error] SFT data too small: ${n} rows (need >=200)" >&2
  exit 2
fi

mkdir -p "${CKPT}" "${TMPDIR}/hf_datasets" "${TMPDIR}/hf_home"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NPROC_PER_NODE="${NPROC}"
export MASTER_ADDR=127.0.0.1
export CUDA_HOME="${CUDA_HOME:-${WORKSPACE}/openrlhf_rl/cuda_stub}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${TMPDIR}/hf_datasets}"
export HF_HOME="${HF_HOME:-${TMPDIR}/hf_home}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[sft] n_rows=${n} gpus=${TRAIN_GPUS} out=${CKPT}"
nohup env \
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
  NPROC_PER_NODE="${NPROC}" \
  MASTER_ADDR=127.0.0.1 \
  CUDA_HOME="${CUDA_HOME}" \
  PATH="${PATH}" \
  DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK}" \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  TMPDIR="${TMPDIR}" TEMP="${TEMP}" TMP="${TMP}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" HF_HOME="${HF_HOME}" \
  "${SWIFT_VENV}/bin/swift" sft \
    --model "${MODEL_DIR}" \
    --dataset "${DATA}" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --num_train_epochs "${SFT_EPOCHS:-2}" \
    --per_device_train_batch_size "${SFT_BS:-1}" \
    --gradient_accumulation_steps "${SFT_GAS:-8}" \
    --learning_rate "${SFT_LR:-1e-5}" \
    --max_length "${SFT_MAX_LEN:-4096}" \
    --gradient_checkpointing true \
    --deepspeed zero2 \
    --logging_steps 1 \
    --save_steps "${SFT_SAVE_STEPS:-200}" \
    --save_only_model true \
    --save_total_limit 2 \
    --eval_strategy no \
    --dataloader_num_workers 0 \
    --use_hf true \
    --output_dir "${CKPT}" \
    --logging_dir "${CKPT}/runs" \
    --report_to tensorboard \
  >>"${LOG_FILE}" 2>&1 &
echo $! >"${CKPT}/swift_sft.pid"
echo "[sft] pid=$(cat "${CKPT}/swift_sft.pid") log=${LOG_FILE}"
