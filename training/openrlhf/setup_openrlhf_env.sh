#!/usr/bin/env bash
# Install OpenRLHF 0.8.2 stack compatible with driver 550 / CUDA 12.4 / torch 2.6 / vLLM 0.8.5.
# Creates a dedicated training venv so the GPU6 judge venv is not disturbed.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=../rl_train/pip_mirror.sh
source "${ROOT}/training/compat/pip_mirror.sh"

TRAIN_VENV="${TRAIN_VENV:-/data1/jinjianhan/venv/openrlhf_train}"
OPENRLHF_SRC="${OPENRLHF_SRC:-${WORKSPACE}/openrlhf_rl/OpenRLHF}"
LOG_DIR="${LOG_DIR:-${WORKSPACE}/openrlhf_rl}"
mkdir -p "${LOG_DIR}" "${PIP_CACHE_DIR}" "${TMPDIR}"

echo "[deps] Target training venv: ${TRAIN_VENV}"

if [[ ! -x "${TRAIN_VENV}/bin/python" ]]; then
  echo "[deps] Creating training venv by cloning PhysicsVerifier .venv ..."
  if [[ -d /data1/jinjianhan/venv/PhysicsVerifier ]]; then
    rsync -a /data1/jinjianhan/venv/PhysicsVerifier/ "${TRAIN_VENV}/"
    # Fix shebangs that still point at the judge venv
    if grep -rl '/home/jinjianhan/PhysicsVerifier/.venv\|/data1/jinjianhan/venv/PhysicsVerifier' "${TRAIN_VENV}/bin" >/dev/null 2>&1; then
      grep -rl '/home/jinjianhan/PhysicsVerifier/.venv\|/data1/jinjianhan/venv/PhysicsVerifier' "${TRAIN_VENV}/bin" | \
        xargs sed -i \
          -e "s|/home/jinjianhan/PhysicsVerifier/.venv|${TRAIN_VENV}|g" \
          -e "s|/data1/jinjianhan/venv/PhysicsVerifier|${TRAIN_VENV}|g"
    fi
  else
    echo "[error] Source venv missing at /data1/jinjianhan/venv/PhysicsVerifier" >&2
    exit 1
  fi
fi

PIP="${TRAIN_VENV}/bin/pip"
PY="${TRAIN_VENV}/bin/python"

echo "[deps] Pinning torch 2.6.0+cu124 (driver 550 compatible) ..."
pip_install "${PIP}" \
  "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "torchaudio==2.6.0+cu124" \
  --index-url https://download.pytorch.org/whl/cu124 \
  2>&1 | tee "${LOG_DIR}/pip_torch.log" | tail -15

echo "[deps] Installing OpenRLHF 0.8.2 (matches vLLM 0.8.5) ..."
# Prefer local git checkout at v0.8.2 if present
if [[ -d "${OPENRLHF_SRC}/.git" ]]; then
  (cd "${OPENRLHF_SRC}" && git checkout -q v0.8.2 2>/dev/null || true)
  pip_install "${PIP}" -e "${OPENRLHF_SRC}" --no-deps 2>&1 | tee "${LOG_DIR}/pip_openrlhf.log" | tail -10
else
  pip_install "${PIP}" "openrlhf==0.8.2" --no-deps 2>&1 | tee "${LOG_DIR}/pip_openrlhf.log" | tail -10
fi

echo "[deps] Installing OpenRLHF runtime deps (skip flash-attn if already present) ..."
pip_install "${PIP}" \
  "deepspeed==0.16.9" "transformers==4.52.3" "ray[default]==2.43.0" "click==8.2.1" \
  "triton==3.2.0" accelerate bitsandbytes datasets einops isort jsonlines loralib optimum \
  "optree>=0.13.0" packaging peft "pynvml>=12.0.0" tensorboard torchdata \
  torchmetrics tqdm transformers-stream-generator wandb wheel \
  2>&1 | tee "${LOG_DIR}/pip_orhf_deps.log" | tail -20

# Keep existing vLLM 0.8.5 from cloned venv; do not force-upgrade to 0.8.5.post1
if ! "${PY}" -c "import vllm" 2>/dev/null; then
  echo "[deps] Installing vllm==0.8.5 ..."
  pip_install "${PIP}" "vllm==0.8.5" 2>&1 | tee -a "${LOG_DIR}/pip_orhf_deps.log" | tail -10
fi

# vLLM may pull nvidia-nccl-cu13 (CUDA 13) which breaks NCCL on driver 550 / CUDA 12.4.
echo "[deps] Pinning NCCL to cu12 for driver 550 compatibility ..."
pip uninstall -y nvidia-nccl-cu13 2>/dev/null || true
pip_install "${PIP}" "nvidia-nccl-cu12==2.21.5" 2>&1 | tee -a "${LOG_DIR}/pip_orhf_deps.log" | tail -5

# flash-attn: host has no nvcc. Install a pure-Python shim (bert_padding + distributed)
# sufficient for OpenRLHF when ring_attn_size=1 and --flash_attn is not used.
FA_DIR="${TRAIN_VENV}/lib/python3.10/site-packages/flash_attn"
if ! "${PY}" -c "from flash_attn.bert_padding import unpad_input" 2>/dev/null; then
  echo "[deps] Installing flash_attn Python shim (no CUDA kernels) ..."
  mkdir -p "${FA_DIR}/utils"
  TMPFA="${TMPDIR}/flash_attn_shim"
  mkdir -p "${TMPFA}"
  curl -sfL --max-time 60 \
    "https://cdn.jsdelivr.net/gh/Dao-AILab/flash-attention@v2.7.4.post1/flash_attn/bert_padding.py" \
    -o "${FA_DIR}/bert_padding.py"
  curl -sfL --max-time 60 \
    "https://cdn.jsdelivr.net/gh/Dao-AILab/flash-attention@v2.7.4.post1/flash_attn/utils/distributed.py" \
    -o "${FA_DIR}/utils/distributed.py"
  printf '%s\n' \
    '# Minimal flash_attn shim for OpenRLHF without CUDA flash-attn wheels.' \
    > "${FA_DIR}/__init__.py"
  touch "${FA_DIR}/utils/__init__.py"
fi

cat > "${WORKSPACE}/openrlhf_rl/env.sh" <<EOF
export WORKSPACE_ROOT="${WORKSPACE}"
export PHYSICS_ROOT="${ROOT}"
export OPENRLHF_ROOT="${OPENRLHF_SRC}"
export TRAIN_VENV="${TRAIN_VENV}"
export PYTHON="${TRAIN_VENV}/bin/python"
export PIP_INDEX_URL="\${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-${WORKSPACE}/.cache/pip}"
export TMPDIR="\${TMPDIR:-${WORKSPACE}/tmp}"
export PHYSICS_REWARD_URL="\${PHYSICS_REWARD_URL:-http://127.0.0.1:8770/get_reward}"
export QWEN30B_MODEL_DIR="\${QWEN30B_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-30B-A3B-Instruct-2507}"
export QWEN30B_RL_CKPT="\${QWEN30B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-30b-physics-openrlhf}"
export QWEN8B_MODEL_DIR="\${QWEN8B_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-8B}"
export QWEN8B_RL_CKPT="\${QWEN8B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
# Host has no system CUDA toolkit; DeepSpeed only needs CUDA_HOME for version probe.
export CUDA_HOME="\${CUDA_HOME:-${WORKSPACE}/openrlhf_rl/cuda_stub}"
export PATH="\${CUDA_HOME}/bin:\${PATH}"
EOF

# Ensure CUDA stub exists for DeepSpeed import (no real nvcc on host)
STUB="${WORKSPACE}/openrlhf_rl/cuda_stub"
mkdir -p "${STUB}/bin"
if [[ ! -x "${STUB}/bin/nvcc" ]]; then
  cat > "${STUB}/bin/nvcc" <<'NVCC'
#!/bin/bash
echo "Cuda compilation tools, release 12.4, V12.4.131"
echo "Build cuda_12.4.r12.4/compiler.stub"
NVCC
  chmod +x "${STUB}/bin/nvcc"
  echo "12.4" > "${STUB}/version.txt"
fi

echo "[deps] Wrote ${WORKSPACE}/openrlhf_rl/env.sh"
echo "[deps] Verifying imports ..."
source "${WORKSPACE}/openrlhf_rl/env.sh"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("torch", torch.__version__, "cuda", torch.version.cuda)
import vllm, openrlhf, deepspeed, transformers, ray
print("vllm", vllm.__version__)
print("openrlhf ok")
print("deepspeed", deepspeed.__version__)
print("transformers", transformers.__version__)
print("ray", ray.__version__)
PY
echo "[ok] OpenRLHF training env ready"
