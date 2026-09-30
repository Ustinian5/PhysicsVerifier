#!/usr/bin/env bash
# Isolated launcher; explicitly choose a personal workspace and an idle GPU.
# Shared packages and weights remain read-only; this does not stop other jobs.
set -euo pipefail
: "${GPTOSS_PERSONAL_ROOT:?Set your personal workspace absolute path}"
: "${GPTOSS_GPU:?Set an explicitly checked idle GPU index}"
: "${GPTOSS_PYTHON:?Set the Python executable in your isolated conda environment}"
PERSONAL="$GPTOSS_PERSONAL_ROOT"
mkdir -p "$PERSONAL"
exec 9>"$PERSONAL/gptoss-vllm.lock"
flock -n 9 || { echo "Personal vLLM service already running" >&2; exit 1; }
nvidia-smi -i "$GPTOSS_GPU" --query-gpu=memory.used --format=csv,noheader,nounits | \
  awk '$1 > 100 {exit 1}' || { echo "Selected GPU is occupied; inspect processes before starting" >&2; exit 1; }
MODEL="${GPTOSS_MODEL_PATH:-/slow_share/jinjianhan/models/gpt-oss-120b}"
# Optional: reuse a compatible shared runtime without writing bytecode there.
if [[ -n "${GPTOSS_SHARED_SITE_PACKAGES:-}" ]]; then
  export PYTHONPATH="$GPTOSS_SHARED_SITE_PACKAGES"
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="$GPTOSS_GPU"
export HF_HOME="$PERSONAL/.cache/gptoss120b"
export HF_HUB_OFFLINE=1
export TIKTOKEN_ENCODINGS_BASE="$PERSONAL/.cache/gptoss120b/harmony"
test -s "$TIKTOKEN_ENCODINGS_BASE/o200k_base.tiktoken"
export TMPDIR="$PERSONAL/tmp"
export VLLM_CACHE_ROOT="$PERSONAL/.cache/gptoss120b/vllm"
export TORCHINDUCTOR_CACHE_DIR="$PERSONAL/.cache/gptoss120b/torch"
export TRITON_CACHE_DIR="$PERSONAL/.cache/gptoss120b/triton"
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
exec "$GPTOSS_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "${GPTOSS_PORT:-18083}" --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 --max-model-len 131072 --max-num-seqs 1 \
  --enforce-eager --safetensors-load-strategy eager --seed 42 \
  --generation-config vllm --disable-log-requests
