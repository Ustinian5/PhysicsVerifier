#!/usr/bin/env bash
# One-shot: launch 10-step process_paragraph pilot if 4 GPUs + slow_share tmp are free.
# Start full training only when an existing admission report already passes.
# Never watchdog, never抢卡, never ray stop --force.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
CKPT="${QWEN8B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf-bootstrap10}"
ADM="${ADM:-${ROOT}/results/four_gpu_pilot_admission.json}"
FULL_CKPT="${FULL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf}"

pass="$(python3 - "${ADM}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print("0")
    raise SystemExit
try:
    d = json.loads(p.read_text())
except Exception:
    print("0")
    raise SystemExit
adm = d.get("admission") or {}
ok = bool(d.get("admission_pass")) and int(d.get("last_step_num") or 0) >= 1
mode = str((d.get("run_manifest") or {}).get("reward_mode") or "")
if ok and mode in {"process_paragraph", ""}:
    print("1")
else:
    print("0")
PY
)"

if [[ "${pass}" == "1" ]]; then
  echo "[full] admission already passed; launching full process_paragraph training ckpt=${FULL_CKPT}"
  export QWEN8B_RL_CKPT="${FULL_CKPT}"
  exec bash "${ROOT}/training/openrlhf/launch_training_4gpu.sh"
fi

echo "[pilot] admission not yet passed; one-shot 10-step bootstrap (no watchdog)"
exec bash "${ROOT}/training/openrlhf/launch_bootstrap_pilot_once.sh"
