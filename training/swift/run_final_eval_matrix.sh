#!/usr/bin/env bash
# Phase 5: add SFT + process-RL checkpoints onto the HiPhO/heldout matrix, then analyze.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
SWIFT_CKPT="${QWEN8B_SWIFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-swift}"
MATRIX_DIR="${MATRIX_DIR:-${ROOT}/results/hipho_baseline_matrix_8b}"

extras=()
labels=()
while IFS= read -r ckpt; do
  [[ -f "${ckpt}/config.json" ]] || continue
  case "${ckpt}" in
    *"/v6-20260825-032454/"*|*"/v7-20260825-040313/"*|*"/v9-20260825-095707/"*) continue ;;
  esac
  step="$(basename "${ckpt}")"
  step="${step#checkpoint-}"
  tag="swift_procrl_step${step}"
  extras+=("${tag}=${ckpt}")
  labels+=("${tag}")
done < <(ls -d "${SWIFT_CKPT}"/v*-*/checkpoint-* 2>/dev/null | sort || true)

export MATRIX_DIR
export SKIP_DONE="${SKIP_DONE:-1}"
if [[ ${#extras[@]} -gt 0 ]]; then
  export EXTRA_MODELS="$(IFS=','; echo "${extras[*]}")"
  export EXTRA_LABELS="$(IFS=','; echo "${labels[*]}")"
fi
bash "${ROOT}/evaluation/benchmarks/hipho/run_hipho_baseline_matrix_8b.sh"

latest_log="$(ls -t "${SWIFT_CKPT}"/v*-*/logging.jsonl 2>/dev/null | head -1 || true)"
recap="${SWIFT_CKPT}/recap_30b.json"
args=(--summary "${MATRIX_DIR}/summary_all.json" --output "${MATRIX_DIR}/process_reward_effectiveness.json")
[[ -n "${latest_log}" ]] && args+=(--logging-jsonl "${latest_log}")
[[ -f "${recap}" ]] && args+=(--recap-json "${recap}")
/data1/jinjianhan/venv/openrlhf_train/bin/python "${ROOT}/training/swift/analyze_process_reward.py" "${args[@]}"
echo "[ok] final matrix ${MATRIX_DIR}/summary_all.json"
