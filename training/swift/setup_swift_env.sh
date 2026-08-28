#!/usr/bin/env bash
# Create an isolated ms-swift GRPO venv. Does not modify openrlhf_train.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
# shellcheck source=../compat/pip_mirror.sh
source "${ROOT}/training/compat/pip_mirror.sh"

TRAIN_VENV="${SWIFT_VENV:-/data1/jinjianhan/venv/swift_train}"
SRC_VENV="${OPENRLHF_VENV:-/data1/jinjianhan/venv/openrlhf_train}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
mkdir -p "${LOG_DIR}" "${PIP_CACHE_DIR}" "${TMPDIR}"

echo "[swift-env] Target venv: ${TRAIN_VENV}"

if [[ ! -x "${TRAIN_VENV}/bin/python" ]]; then
  echo "[swift-env] Cloning ${SRC_VENV} -> ${TRAIN_VENV} (keeps torch 2.6 / vLLM 0.8.5)"
  if [[ ! -x "${SRC_VENV}/bin/python" ]]; then
    echo "[error] source venv missing: ${SRC_VENV}" >&2
    exit 1
  fi
  rsync -a "${SRC_VENV}/" "${TRAIN_VENV}/"
  if grep -rl "${SRC_VENV}" "${TRAIN_VENV}/bin" >/dev/null 2>&1; then
    grep -rl "${SRC_VENV}" "${TRAIN_VENV}/bin" | \
      xargs sed -i -e "s|${SRC_VENV}|${TRAIN_VENV}|g"
  fi
fi

PIP="${TRAIN_VENV}/bin/pip"
PY="${TRAIN_VENV}/bin/python"

echo "[swift-env] Installing ms-swift, trl, aiohttp ..."
pip_install "${PIP}" --upgrade pip wheel setuptools \
  2>&1 | tee "${LOG_DIR}/pip_swift_env.log" | tail -5
pip_install "${PIP}" "ms-swift" "trl>=0.17.0" "aiohttp" "datasets==4.8.4" \
  2>&1 | tee -a "${LOG_DIR}/pip_swift_env.log" | tail -20

if ! "${PY}" -c "import vllm" 2>/dev/null; then
  echo "[swift-env] Installing vllm==0.8.5 ..."
  pip_install "${PIP}" "vllm==0.8.5" 2>&1 | tee -a "${LOG_DIR}/pip_swift_env.log" | tail -10
fi

pip uninstall -y nvidia-nccl-cu13 2>/dev/null || true
pip_install "${PIP}" "nvidia-nccl-cu12==2.21.5" 2>&1 | tee -a "${LOG_DIR}/pip_swift_env.log" | tail -3
pip_install "${PIP}" "datasets==4.8.4" 2>&1 | tee -a "${LOG_DIR}/pip_swift_env.log" | tail -3

echo "[swift-env] Patching vLLM 0.8.5 compatibility (reset_mm_cache optional) ..."
"${PY}" - <<'PY'
from pathlib import Path
import swift, os
root = Path(swift.__file__).resolve().parent
old = "self.engine.engine.reset_mm_cache()"
new = "hasattr(self.engine.engine, 'reset_mm_cache') and self.engine.engine.reset_mm_cache()"
for rel in ("rlhf_trainers/rollout_mixin.py", "megatron/trainers/rollout_mixin.py"):
    path = root / rel
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8")
    if old in text and "hasattr(self.engine.engine, 'reset_mm_cache')" not in text:
        path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"[ok] patched {path}")
    else:
        print(f"[skip] {path}")
PY

ENV_FILE="${WORKSPACE}/swift_rl/env.sh"
mkdir -p "$(dirname "${ENV_FILE}")"
cat > "${ENV_FILE}" <<EOF
export WORKSPACE_ROOT="${WORKSPACE}"
export PHYSICS_ROOT="${ROOT}"
export SWIFT_VENV="${TRAIN_VENV}"
export PYTHON="${TRAIN_VENV}/bin/python"
export PIP_INDEX_URL="\${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-${WORKSPACE}/.cache/pip}"
export TMPDIR="\${TMPDIR:-/slow_share/jinjianhan/tmp/swift}"
export PHYSICS_REWARD_URL="\${PHYSICS_REWARD_URL:-http://127.0.0.1:8770/get_reward}"
export QWEN8B_MODEL_DIR="\${QWEN8B_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-8B}"
export QWEN8B_SWIFT_CKPT="\${QWEN8B_SWIFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-swift}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_HOME="\${CUDA_HOME:-${WORKSPACE}/openrlhf_rl/cuda_stub}"
export PATH="\${CUDA_HOME}/bin:\${PATH}"
EOF

echo "[swift-env] Verifying imports ..."
# shellcheck disable=SC1090
source "${ENV_FILE}"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("torch", torch.__version__, "cuda", torch.version.cuda)
import vllm, trl, aiohttp
print("vllm", vllm.__version__)
print("trl", trl.__version__)
import swift
print("swift", getattr(swift, "__version__", "ok"))
from swift.rewards import AsyncORM, orms
print("swift.rewards.AsyncORM ok")
PY
echo "[ok] swift training env ready venv=${TRAIN_VENV}"
echo "[ok] env file ${ENV_FILE}"
