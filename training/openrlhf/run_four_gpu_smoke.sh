#!/usr/bin/env bash
# Quick 4-GPU smoke validation: reward API + vLLM import + Ray head on 4 GPUs.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
ENV_FILE="${WORKSPACE}/openrlhf_rl/env.sh"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
PYTHON="${PYTHON:-${TRAIN_VENV}/bin/python}"

if [[ -z "${PHYSICSVERIFIER_OPENAI_BASE_URL:-}" ]]; then
  export PHYSICS_REWARD_MODE="${PHYSICS_REWARD_MODE:-answer_only}"
fi

bash "${ROOT}/training/reward_server/start_reward_server.sh"
curl -sf -X POST http://127.0.0.1:8770/get_reward \
  -H "Content-Type: application/json" \
  -d '{"query":["prompt\\nassistant\\n\\\\boxed{1}"],"prompts":["prompt"],"labels":["1"]}' \
  | grep -q rewards

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
ray stop --force || true
ray start --head --num-gpus 4 --disable-usage-stats
"${PYTHON}" - <<'PY'
import torch
assert torch.cuda.device_count() >= 4
print("cuda_devices", torch.cuda.device_count())
import openrlhf, vllm
print("imports_ok")
PY
ray stop --force || true
echo "[ok] four-gpu smoke passed"
