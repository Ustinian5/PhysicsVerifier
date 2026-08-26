#!/usr/bin/env bash
# Check OpenRLHF physics RL prerequisites.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
ENV_FILE="${WORKSPACE}/openrlhf_rl/env.sh"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
PYTHON="${PYTHON:-${TRAIN_VENV:-${ROOT}/.venv}/bin/python}"

errors=0
warns=0

check() {
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then
    echo "[ok] $name"
  else
    echo "[fail] $name"
    errors=$((errors + 1))
  fi
}

warn() {
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then
    echo "[ok] $name"
  else
    echo "[warn] $name"
    warns=$((warns + 1))
  fi
}

check "reward server /health" curl -sf http://127.0.0.1:8770/health
check "reward server /get_reward" bash -c 'curl -sf -X POST http://127.0.0.1:8770/get_reward -H "Content-Type: application/json" -d "{\"query\":[\"a\"],\"prompts\":[\"\"],\"labels\":[\"b\"]}" | grep -q rewards'
if [[ -z "${PHYSICSVERIFIER_OPENAI_BASE_URL:-}" && "${PHYSICS_REWARD_MODE:-}" != "answer_only" ]]; then
  check "vllm judge" curl -sf http://127.0.0.1:8766/v1/models
else
  warn "vllm judge" true
fi
check "train prompts (openrlhf format)" test -s "${ROOT}/data/rl/openrlhf_prompts.jsonl"
warn "heldout (openrlhf format)" test -s "${ROOT}/data/rl/openrlhf_heldout.jsonl"
warn "openrlhf import" "${PYTHON}" -c "import openrlhf"
warn "vllm import" "${PYTHON}" -c "import vllm"
warn "deepspeed import" "${PYTHON}" -c "import deepspeed"
warn "hf model dir" test -d "${QWEN30B_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-30B-A3B-Instruct-2507}"
warn "flash_attn" "${PYTHON}" -c "import flash_attn"

echo "---"
echo "errors=$errors warns=$warns"
if [[ "$errors" -gt 0 ]]; then
  exit 1
fi
