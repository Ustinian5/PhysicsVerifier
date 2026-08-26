#!/usr/bin/env bash
# Check prerequisites for 4-GPU Qwen3-8B OpenRLHF training.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
ENV_FILE="${WORKSPACE}/openrlhf_rl/env.sh"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
PYTHON="${PYTHON:-${TRAIN_VENV:-/data1/jinjianhan/venv/openrlhf_train}/bin/python}"
QWEN8B_MODEL_DIR="${QWEN8B_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-8B}"
MODE="${PHYSICS_REWARD_MODE:-answer_only}"

errors=0
warns=0
check() { local n="$1"; shift; if "$@" >/dev/null 2>&1; then echo "[ok] $n"; else echo "[fail] $n"; errors=$((errors+1)); fi; }
warn() { local n="$1"; shift; if "$@" >/dev/null 2>&1; then echo "[ok] $n"; else echo "[warn] $n"; warns=$((warns+1)); fi; }

check "CUDA usable" bash "${ROOT}/training/openrlhf/ensure_cuda_ready.sh"
check "reward server /health" curl -sf http://127.0.0.1:8770/health
check "reward server /get_reward" bash -c 'curl -sf -X POST http://127.0.0.1:8770/get_reward -H "Content-Type: application/json" -d "{\"query\":[\"a\"],\"prompts\":[\"\"],\"labels\":[\"b\"]}" | grep -q rewards'
check "train prompts" test -s "${ROOT}/data/rl/openrlhf_prompts.jsonl"
check "4 visible GPUs" bash -c '[[ $(nvidia-smi -L | wc -l) -ge 4 ]]'
check "8B model dir" test -d "${QWEN8B_MODEL_DIR}"
check "8B config" test -f "${QWEN8B_MODEL_DIR}/config.json"
warn "openrlhf import" "${PYTHON}" -c "import openrlhf"
warn "vllm import" "${PYTHON}" -c "import vllm"
warn "fabricmanager active" systemctl is-active nvidia-fabricmanager

if [[ -n "${PHYSICSVERIFIER_OPENAI_BASE_URL:-}" ]]; then
  check "external verifier API env" bash -c '[[ -n "${PHYSICSVERIFIER_OPENAI_BASE_URL}" ]]'
elif [[ "${MODE}" != "answer_only" ]]; then
  check "local judge /v1/models" curl -sf http://127.0.0.1:8766/v1/models
else
  warn "external verifier API" bash -c '[[ -n "${PHYSICSVERIFIER_OPENAI_BASE_URL:-}" ]]'
fi

echo "---"
echo "errors=$errors warns=$warns mode=${MODE}"
[[ "$errors" -eq 0 ]]
