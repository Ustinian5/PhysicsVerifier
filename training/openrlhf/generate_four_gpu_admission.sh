#!/usr/bin/env bash
# Generate four-GPU pilot admission report (works with partial or complete pilot runs).
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
CKPT="${ADMISSION_CKPT:-${ADAPTIVE_CKPT:-${QWEN8B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-bootstrap10}}}"
OUT="${OUT:-${ROOT}/results/four_gpu_pilot_admission.json}"
TARGET_STEPS="${PILOT_MAX_STEPS:-10}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"

cuda_ok=0
if TRY_RESTART_FABRICMANAGER=0 bash "${ROOT}/training/openrlhf/ensure_cuda_ready.sh" >/dev/null 2>&1; then
  cuda_ok=1
fi
fm_active=0
if systemctl is-active nvidia-fabricmanager >/dev/null 2>&1; then
  fm_active=1
fi

GPU_SNAP_FILE="${CKPT}/gpu_util_snapshot.json"
mkdir -p "${CKPT}"
python3 - <<PY
import json, subprocess
from pathlib import Path
path = Path("${GPU_SNAP_FILE}")
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

"${PYTHON}" "${ROOT}/training/openrlhf/admission_report.py" \
  --ckpt "${CKPT}" \
  --out "${OUT}" \
  --target-steps "${TARGET_STEPS}" \
  --cuda-ok "${cuda_ok}" \
  --fm-active "${fm_active}" \
  --train-stage "${TRAIN_STAGE:-}" \
  --train-topology "${TRAIN_TOPOLOGY:-}"

echo "[ok] admission report -> ${OUT}"
python3 -c "import json; d=json.load(open('${OUT}')); print('steps', d.get('last_step_num'), 'topology', d.get('train_topology'), 'stage', d.get('train_stage'), 'admission_pass', d.get('admission_pass'), 'verifier_ready', (d.get('verifier_stage_gate') or {}).get('ready'))"
