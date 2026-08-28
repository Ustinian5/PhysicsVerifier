#!/usr/bin/env bash
# Heldout answer-acc gate: SFT ckpt must beat base_8b from the baseline matrix.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
SFT_CKPT="${1:-${QWEN8B_SFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-sft}}"
if [[ ! -f "${SFT_CKPT}/config.json" ]]; then
  SFT_CKPT="$(ls -d "${SFT_CKPT}"/v*-*/checkpoint-* 2>/dev/null | tail -1 || true)"
fi
[[ -f "${SFT_CKPT}/config.json" ]] || { echo "[error] missing SFT ckpt" >&2; exit 2; }
BASE_SCORES="${BASE_SCORES:-${ROOT}/results/hipho_baseline_matrix_8b/base_8b/heldout_scores.json}"
OUT="${OUT:-${SFT_CKPT}/heldout_fast_eval}"
MAX_SAMPLES="${MAX_SAMPLES:-50}" CUDA_DEVICE="${CUDA_DEVICE:-7}" PORT="${PORT:-8766}" \
  bash "${ROOT}/training/swift/eval_heldout_fast.sh" "${SFT_CKPT}" "${OUT}"
python3 - <<PY
import json, sys
from pathlib import Path
sft = json.loads(Path("${OUT}/heldout_scores.json").read_text())
base_path = Path("${BASE_SCORES}")
sft_acc = float(sft.get("answer_acc") or 0.0)
base_acc = 0.0
if base_path.is_file():
    base_acc = float(json.loads(base_path.read_text()).get("answer_acc") or 0.0)
report = {"sft_acc": sft_acc, "base_acc": base_acc, "delta": sft_acc - base_acc, "pass": sft_acc > base_acc}
Path("${OUT}/gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
sys.exit(0 if report["pass"] else 3)
PY
