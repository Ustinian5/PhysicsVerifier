#!/usr/bin/env bash
# Start PhysicsVerifier reward server with local judge or external OpenAI-compatible API.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
VENV="${VENV:-${ROOT}/.venv}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8770}"
LOG="${LOG:-${ROOT}/logs/physics_reward_server.log}"
PID_FILE="${PID_FILE:-${ROOT}/logs/physics_reward_server.pid}"

mkdir -p "$(dirname "$LOG")"

cd "${ROOT}" || exit 1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PHYSICS_REWARD_LAMBDA="${PHYSICS_REWARD_LAMBDA:-0.3}"
export PHYSICS_REWARD_ERROR_CAP="${PHYSICS_REWARD_ERROR_CAP:-3}"
export PHYSICS_REWARD_CONCURRENCY="${PHYSICS_REWARD_CONCURRENCY:-8}"
export PHYSICS_REWARD_MODE="${PHYSICS_REWARD_MODE:-answer_low_verifier}"
export PHYSICS_REWARD_W_ANSWER="${PHYSICS_REWARD_W_ANSWER:-1.0}"
export PHYSICS_REWARD_W_FORMAT="${PHYSICS_REWARD_W_FORMAT:-0.05}"
export PHYSICS_REWARD_W_VERIFIER="${PHYSICS_REWARD_W_VERIFIER:-0.1}"
export PHYSICS_VERIFIER_SAMPLE_RATE="${PHYSICS_VERIFIER_SAMPLE_RATE:-1.0}"

CONFIGURED_OPENAI_BASE_URL="${PHYSICSVERIFIER_OPENAI_BASE_URL:-${OPENAI_BASE_URL:-}}"
if [[ -n "${PHYSICSVERIFIER_OPENAI_BASE_URL:-}" ]]; then
  export OPENAI_BASE_URL="${PHYSICSVERIFIER_OPENAI_BASE_URL}"
fi
if [[ -n "${PHYSICSVERIFIER_OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${PHYSICSVERIFIER_OPENAI_API_KEY}"
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8766/v1}"
export PHYSICSVERIFIER_LLM_MODEL="${PHYSICSVERIFIER_LLM_MODEL:-qwen3-30b-a3b}"
export PHYSICSVERIFIER_UNIFIED_RULES="${PHYSICSVERIFIER_UNIFIED_RULES:-${ROOT}/catalogs/rules_unified_3000_runtime_backfilled.json}"
export PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE="${PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE:-semantic}"
# The norm_* runtime catalog has no matching exp_* symbolic manifest.
export PHYSICSVERIFIER_SYMBOLIC_ENABLED="${PHYSICSVERIFIER_SYMBOLIC_ENABLED:-0}"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      mode_file="$(dirname "$PID_FILE")/physics_reward_server.mode"
      old_mode="$(cat "${mode_file}" 2>/dev/null || true)"
      if [[ "${old_mode}" == "${PHYSICS_REWARD_MODE}" ]]; then
        echo "[ok] reward server already running pid=$old_pid mode=${PHYSICS_REWARD_MODE}"
        exit 0
      fi
      echo "[reward] mode changed (${old_mode} -> ${PHYSICS_REWARD_MODE}); restarting"
      kill -TERM "$old_pid" 2>/dev/null || true
      sleep 2
      kill -9 "$old_pid" 2>/dev/null || true
    fi
  fi
fi

if [[ -n "${CONFIGURED_OPENAI_BASE_URL}" ]]; then
  echo "[reward] using external verifier API at ${OPENAI_BASE_URL}"
  "${VENV}/bin/python" - <<'PY'
import os, sys
from openai import OpenAI
base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
key = os.environ.get("OPENAI_API_KEY", "EMPTY")
model = os.environ.get("PHYSICSVERIFIER_LLM_MODEL", "")
client = OpenAI(base_url=base, api_key=key)
models = [m.id for m in client.models.list().data]
if model and model not in models:
    print(f"[warn] configured model {model} not in remote list; available={models[:5]}", file=sys.stderr)
print("[ok] external verifier API reachable")
PY
elif [[ "${PHYSICS_REWARD_MODE}" == "answer_only" ]]; then
  echo "[reward] answer_only mode; skipping local judge/API warmup"
else
  curl -sf "${OPENAI_BASE_URL%/}/models" >/dev/null || {
    echo "[error] local judge unavailable at ${OPENAI_BASE_URL}; set PHYSICSVERIFIER_OPENAI_BASE_URL or PHYSICS_REWARD_MODE=answer_only" >&2
    exit 2
  }
fi

nohup "${VENV}/bin/python" "${ROOT}/training/reward_server/physics_reward_server.py" \
  --host "$HOST" --port "$PORT" \
  --reward-mode "${PHYSICS_REWARD_MODE}" \
  >"$LOG" 2>&1 &
echo $! >"$PID_FILE"
printf '%s' "${PHYSICS_REWARD_MODE}" >"$(dirname "$PID_FILE")/physics_reward_server.mode"
ready=0
for _ in $(seq 1 20); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" -ne 1 ]]; then
  echo "[error] reward server failed to become healthy on ${HOST}:${PORT}; see ${LOG}" >&2
  tail -n 40 "${LOG}" >&2 || true
  exit 2
fi
echo "[ok] reward server started on ${HOST}:${PORT} mode=${PHYSICS_REWARD_MODE}"
