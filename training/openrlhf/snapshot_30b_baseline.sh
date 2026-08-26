#!/usr/bin/env bash
# Snapshot immutable metadata for the paused 30B OpenRLHF run.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
SAVE_PATH="${QWEN30B_RL_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-30b-physics-openrlhf}"
OUT_DIR="${OUT_DIR:-${SAVE_PATH}/baseline_snapshot}"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUT_DIR}"
cp -a "${ROOT}/training/openrlhf/run-qwen3-30b-physics-6gpu-openrlhf.sh" "${OUT_DIR}/run-qwen3-30b-physics-6gpu-openrlhf.sh"
if [[ -f /slow_share/jinjianhan/workspace/openrlhf_rl/env.sh ]]; then
  cp -a /slow_share/jinjianhan/workspace/openrlhf_rl/env.sh "${OUT_DIR}/env.sh"
fi
cp -a "${SAVE_PATH}/plots/training_metrics.csv" "${OUT_DIR}/training_metrics.csv" 2>/dev/null || true
cp -a "${SAVE_PATH}/ckpt/_actor/latest" "${OUT_DIR}/latest_pointer.txt" 2>/dev/null || true
cp -a "${SAVE_PATH}/ray_job_id.txt" "${OUT_DIR}/ray_job_id.txt" 2>/dev/null || true

python3 - <<PY >"${OUT_DIR}/summary_${TS}.json"
import json
from pathlib import Path
save = Path("${SAVE_PATH}")
metrics = Path("${OUT_DIR}/training_metrics.csv")
rows = []
if metrics.is_file():
    lines = metrics.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) > 1:
        header = lines[0].split(",")
        for line in lines[1:]:
            vals = line.split(",")
            rows.append(dict(zip(header, vals)))
latest = (save / "ckpt/_actor/latest").read_text(encoding="utf-8").strip() if (save / "ckpt/_actor/latest").is_file() else ""
print(json.dumps({
  "snapshot_ts": "${TS}",
  "save_path": str(save),
  "latest_checkpoint": latest,
  "global_steps_recorded": len(rows),
  "last_step": rows[-1] if rows else {},
  "resume_entrypoint": "bash training/openrlhf/run-qwen3-30b-physics-6gpu-openrlhf.sh",
  "note": "6 training GPUs + separate judge required for 30B",
}, ensure_ascii=False, indent=2))
PY

echo "[ok] baseline snapshot written to ${OUT_DIR}"
