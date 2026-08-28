#!/usr/bin/env bash
# Offline 30B process-score recap of recent GRPO completions via remote API (no extra GPU).
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CKPT="${QWEN8B_SWIFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-swift}"
ROLLOUTS="${1:-}"
if [[ -z "${ROLLOUTS}" ]]; then
  ROLLOUTS="$(ls -t "${CKPT}"/v*-*/completions.jsonl 2>/dev/null | head -1 || true)"
fi
[[ -s "${ROLLOUTS}" ]] || { echo "[error] no completions jsonl" >&2; exit 2; }
OUT="${2:-${CKPT}/recap_30b.json}"
LIMIT="${RECAP_LIMIT:-20}"

REMOTE_BASE="$(python3 - <<PY
from pathlib import Path
for line in Path("${ROOT}/.env").read_text().splitlines():
    if line.startswith("OPENAI_BASE_URL="):
        print(line.split("=",1)[1].strip().strip('"').strip("'"))
        break
PY
)"
REMOTE_KEY="$(python3 - <<PY
from pathlib import Path
for line in Path("${ROOT}/.env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        print(line.split("=",1)[1].strip().strip('"').strip("'"))
        break
PY
)"
export PHYSICS_REWARD_MODE=process_paragraph
export PHYSICS_REWARD_W_ANSWER=0
export PHYSICS_REWARD_W_FORMAT=0
export PHYSICSVERIFIER_UNIFIED_RULE_TOP_N="${PHYSICSVERIFIER_UNIFIED_RULE_TOP_N:-4}"
export PHYSICSVERIFIER_PRECISION_MODE="${PHYSICSVERIFIER_PRECISION_MODE:-strict}"
export PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE=lexical
export PHYSICSVERIFIER_OPENAI_BASE_URL="${REMOTE_BASE}"
export PHYSICSVERIFIER_OPENAI_API_KEY="${REMOTE_KEY}"
export PHYSICSVERIFIER_LLM_MODEL="${SFT_API_MODEL:-qwen3-30b-a3b-instruct-2507}"
PORT=8771 PID_FILE="${ROOT}/logs/physics_reward_server_30b_api.pid" \
  LOG="${ROOT}/logs/physics_reward_server_30b_api.log" \
  bash "${ROOT}/training/reward_server/start_reward_server.sh"

"${PYTHON}" "${ROOT}/training/swift/smoke_self_judge.py" \
  --rollouts "${ROLLOUTS}" \
  --url-a "http://127.0.0.1:8771/get_reward" \
  --limit "${LIMIT}" \
  --output "${OUT}"
echo "[ok] 30B recap ${OUT}"
