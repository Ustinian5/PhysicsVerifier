#!/usr/bin/env bash
# Verify CUDA is usable on this NVSwitch host (requires nvidia-fabricmanager).
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
TRY_RESTART="${TRY_RESTART_FABRICMANAGER:-1}"

fm_active() {
  systemctl is-active nvidia-fabricmanager >/dev/null 2>&1
}

cuda_ok() {
  "${PYTHON}" - <<'PY' >/dev/null 2>&1
import torch
x = torch.zeros(1, device="cuda:0")
assert x.is_cuda
print(torch.cuda.get_device_name(0))
PY
}

if cuda_ok; then
  echo "[ok] CUDA ready"
  exit 0
fi

echo "[fail] CUDA not usable (likely nvidia-fabricmanager down)"
systemctl is-active nvidia-fabricmanager >/dev/null 2>&1 || \
  echo "[fail] nvidia-fabricmanager.service is not active"

if [[ "${TRY_RESTART}" == "1" ]]; then
  if sudo -n systemctl restart nvidia-fabricmanager >/dev/null 2>&1; then
    sleep 3
    if cuda_ok; then
      echo "[ok] CUDA ready after fabricmanager restart"
      exit 0
    fi
  else
    echo "[hint] passwordless sudo unavailable; ask an admin to run:"
    echo "  sudo systemctl restart nvidia-fabricmanager"
  fi
fi

exit 2
