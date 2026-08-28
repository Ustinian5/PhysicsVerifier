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
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # Project-owned dotenv files in this workflow use shell-compatible KEY=VALUE syntax.
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PHYSICS_ROOT="${ROOT}"
export PHYSICS_REWARD_MODE="${PHYSICS_REWARD_MODE:-answer_low_verifier}"
case "${PHYSICS_REWARD_MODE}" in
  process_paragraph)
    PV_REWARD_DEFAULT_CONCURRENCY=12
    PV_REWARD_DEFAULT_MAX_RESPONSE_CHARS=3072
    PV_REWARD_DEFAULT_RULE_TOP_N=4
    PV_REWARD_DEFAULT_PRECISION=balanced
    PV_REWARD_DEFAULT_RETRIEVAL=lexical
    ;;
  llm_step_score)
    PV_REWARD_DEFAULT_CONCURRENCY=32
    PV_REWARD_DEFAULT_MAX_RESPONSE_CHARS=12000
    PV_REWARD_DEFAULT_RULE_TOP_N=6
    PV_REWARD_DEFAULT_PRECISION=strict
    PV_REWARD_DEFAULT_RETRIEVAL=semantic
    ;;
  *)
    PV_REWARD_DEFAULT_CONCURRENCY=8
    PV_REWARD_DEFAULT_MAX_RESPONSE_CHARS=12000
    PV_REWARD_DEFAULT_RULE_TOP_N=6
    PV_REWARD_DEFAULT_PRECISION=strict
    PV_REWARD_DEFAULT_RETRIEVAL=semantic
    ;;
esac
export PHYSICS_REWARD_LAMBDA="${PHYSICS_REWARD_LAMBDA:-0.3}"
export PHYSICS_REWARD_ERROR_CAP="${PHYSICS_REWARD_ERROR_CAP:-3}"
export PHYSICS_REWARD_CONCURRENCY="${PHYSICS_REWARD_CONCURRENCY:-${PV_REWARD_DEFAULT_CONCURRENCY}}"
export PHYSICS_REWARD_MAX_RESPONSE_CHARS="${PHYSICS_REWARD_MAX_RESPONSE_CHARS:-${PV_REWARD_DEFAULT_MAX_RESPONSE_CHARS}}"
export PHYSICS_REWARD_W_ANSWER="${PHYSICS_REWARD_W_ANSWER:-1.0}"
export PHYSICS_REWARD_W_FORMAT="${PHYSICS_REWARD_W_FORMAT:-0.05}"
export PHYSICS_REWARD_W_VERIFIER="${PHYSICS_REWARD_W_VERIFIER:-0.1}"
export PHYSICS_VERIFIER_SAMPLE_RATE="${PHYSICS_VERIFIER_SAMPLE_RATE:-1.0}"
export PHYSICS_REWARD_VERIFIER_FAILURE_POLICY="${PHYSICS_REWARD_VERIFIER_FAILURE_POLICY:-raise}"
export PHYSICSVERIFIER_CHECKER_GATE_MODE="${PHYSICSVERIFIER_CHECKER_GATE_MODE:-legacy}"
export PHYSICSVERIFIER_CHECKER_JSON_ATTEMPTS="${PHYSICSVERIFIER_CHECKER_JSON_ATTEMPTS:-3}"
export PHYSICSVERIFIER_SEMANTIC_JSON_ATTEMPTS="${PHYSICSVERIFIER_SEMANTIC_JSON_ATTEMPTS:-3}"
export PHYSICSVERIFIER_UNIFIED_RULE_TOP_N="${PHYSICSVERIFIER_UNIFIED_RULE_TOP_N:-${PV_REWARD_DEFAULT_RULE_TOP_N}}"
export PHYSICSVERIFIER_PRECISION_MODE="${PHYSICSVERIFIER_PRECISION_MODE:-${PV_REWARD_DEFAULT_PRECISION}}"
export PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE="${PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE:-${PV_REWARD_DEFAULT_RETRIEVAL}}"
export PHYSICSVERIFIER_MAX_DIAGNOSTICS_PER_SAMPLE="${PHYSICSVERIFIER_MAX_DIAGNOSTICS_PER_SAMPLE:-12}"
export PHYSICSVERIFIER_MAX_DIAGNOSTICS_PER_PARAGRAPH="${PHYSICSVERIFIER_MAX_DIAGNOSTICS_PER_PARAGRAPH:-2}"
export PHYSICSVERIFIER_ENABLE_LLM_CACHE="${PHYSICSVERIFIER_ENABLE_LLM_CACHE:-0}"
export PHYSICSVERIFIER_REQUIRE_PROVIDER_IDENTITY="${PHYSICSVERIFIER_REQUIRE_PROVIDER_IDENTITY:-1}"
if [[ "${PHYSICS_REWARD_MODE}" == "process_paragraph" ]]; then
  export PHYSICS_REWARD_VERIFIER_ON_WRONG="${PHYSICS_REWARD_VERIFIER_ON_WRONG:-1}"
  export PHYSICS_REWARD_W_ANSWER=0
  export PHYSICS_REWARD_W_FORMAT=0
  export PHYSICS_REWARD_W_CLEAN="${PHYSICS_REWARD_W_CLEAN:-0.5}"
  export PHYSICS_REWARD_W_FIRST="${PHYSICS_REWARD_W_FIRST:-0.3}"
  export PHYSICS_REWARD_W_DENSE="${PHYSICS_REWARD_W_DENSE:-0.2}"
  export PHYSICS_REWARD_CONCURRENCY="${PHYSICS_REWARD_CONCURRENCY:-12}"
  export PHYSICS_REWARD_MAX_RESPONSE_CHARS="${PHYSICS_REWARD_MAX_RESPONSE_CHARS:-3072}"
  export PHYSICSVERIFIER_UNIFIED_RULE_TOP_N="${PHYSICSVERIFIER_UNIFIED_RULE_TOP_N:-4}"
  export PHYSICSVERIFIER_PRECISION_MODE="${PHYSICSVERIFIER_PRECISION_MODE:-balanced}"
  export PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE="${PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE:-lexical}"
fi
if [[ "${PHYSICS_REWARD_MODE}" == "llm_step_score" ]]; then
  export PHYSICSVERIFIER_LLM_MODEL="${PHYSICSVERIFIER_LLM_MODEL:-deepseek-v4-flash}"
  export LLM_STEP_JUDGE_TIMEOUT="${LLM_STEP_JUDGE_TIMEOUT:-300}"
  export LLM_STEP_JUDGE_CONCURRENCY="${LLM_STEP_JUDGE_CONCURRENCY:-32}"
  export PHYSICS_REWARD_CONCURRENCY="${PHYSICS_REWARD_CONCURRENCY:-32}"
  export PHYSICS_REWARD_W_ANSWER=0
  export PHYSICS_REWARD_W_FORMAT=0
  export PHYSICS_REWARD_W_VERIFIER=0
  if [[ -z "${OPENAI_BASE_URL:-}" || "${OPENAI_BASE_URL}" == *"127.0.0.1"* ]]; then
    echo "[error] llm_step_score requires remote OPENAI_BASE_URL from .env (not a local judge)" >&2
    exit 2
  fi
  if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" ]]; then
    echo "[error] llm_step_score requires OPENAI_API_KEY from .env" >&2
    exit 2
  fi
fi

CONFIGURED_OPENAI_BASE_URL="${PHYSICSVERIFIER_OPENAI_BASE_URL:-${OPENAI_BASE_URL:-}}"
if [[ "${PHYSICS_REWARD_MODE}" != "llm_step_score" && -n "${PHYSICSVERIFIER_OPENAI_BASE_URL:-}" ]]; then
  export OPENAI_BASE_URL="${PHYSICSVERIFIER_OPENAI_BASE_URL}"
fi
if [[ "${PHYSICS_REWARD_MODE}" != "llm_step_score" && -n "${PHYSICSVERIFIER_OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${PHYSICSVERIFIER_OPENAI_API_KEY}"
fi
if [[ "${PHYSICS_REWARD_MODE}" == "llm_step_score" ]]; then
  if [[ -z "${OPENAI_BASE_URL:-}" || "${OPENAI_BASE_URL}" == *"127.0.0.1"* ]]; then
    echo "[error] llm_step_score requires remote OPENAI_BASE_URL from .env (not a local judge)" >&2
    exit 2
  fi
  export PHYSICSVERIFIER_LLM_MODEL="${PHYSICSVERIFIER_LLM_MODEL:-deepseek-v4-flash}"
else
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8766/v1}"
  export PHYSICSVERIFIER_LLM_MODEL="${PHYSICSVERIFIER_LLM_MODEL:-qwen3-30b-a3b}"
fi
export PHYSICSVERIFIER_UNIFIED_RULES="${PHYSICSVERIFIER_UNIFIED_RULES:-}"
# The norm_* runtime catalog has no matching exp_* symbolic manifest.
export PHYSICSVERIFIER_SYMBOLIC_ENABLED="${PHYSICSVERIFIER_SYMBOLIC_ENABLED:-0}"
if [[ "${PHYSICS_REWARD_MODE}" != "answer_only" && "${PHYSICS_REWARD_MODE}" != "llm_step_score" && -z "${PHYSICSVERIFIER_UNIFIED_RULES}" ]]; then
  echo "[error] set PHYSICSVERIFIER_UNIFIED_RULES explicitly for verifier reward" >&2
  exit 2
fi

port_pid() {
  ss -lptn "sport = :${PORT}" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n1
}

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      if curl -sf "http://${HOST}:${PORT}/health" | "${VENV}/bin/python" -c '
import json, os, sys

actual = (json.load(sys.stdin) or {}).get("config") or {}
truthy = {"1", "true", "yes", "on"}
expected = {
    "reward_mode": os.environ["PHYSICS_REWARD_MODE"],
    "lambda": float(os.environ["PHYSICS_REWARD_LAMBDA"]),
    "error_cap": int(os.environ["PHYSICS_REWARD_ERROR_CAP"]),
    "concurrency": int(os.environ["PHYSICS_REWARD_CONCURRENCY"]),
    "w_answer": float(os.environ["PHYSICS_REWARD_W_ANSWER"]),
    "w_format": float(os.environ["PHYSICS_REWARD_W_FORMAT"]),
    "w_verifier": float(os.environ["PHYSICS_REWARD_W_VERIFIER"]),
    "w_length": float(os.environ.get("PHYSICS_REWARD_W_LENGTH", "0")),
    "w_clean": float(os.environ.get("PHYSICS_REWARD_W_CLEAN", "0.5")),
    "w_first": float(os.environ.get("PHYSICS_REWARD_W_FIRST", "0.3")),
    "w_dense": float(os.environ.get("PHYSICS_REWARD_W_DENSE", "0.2")),
    "verifier_sample_rate": float(os.environ["PHYSICS_VERIFIER_SAMPLE_RATE"]),
    "verifier_on_wrong": os.environ.get("PHYSICS_REWARD_VERIFIER_ON_WRONG", "").lower() in truthy,
    "verifier_failure_policy": os.environ["PHYSICS_REWARD_VERIFIER_FAILURE_POLICY"],
    "unified_rules": os.environ["PHYSICSVERIFIER_UNIFIED_RULES"],
    "retrieval_mode": os.environ["PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE"],
    "semantic_output_adapter": os.environ.get("PHYSICSVERIFIER_SEMANTIC_OUTPUT_ADAPTER", ""),
    "llm_model": os.environ["PHYSICSVERIFIER_LLM_MODEL"],
    "openai_base_url": os.environ["OPENAI_BASE_URL"],
    "symbolic_enabled": os.environ.get("PHYSICSVERIFIER_SYMBOLIC_ENABLED", "0").lower() in truthy,
    "checker_gate_mode": os.environ["PHYSICSVERIFIER_CHECKER_GATE_MODE"],
    "checker_json_attempts": int(os.environ["PHYSICSVERIFIER_CHECKER_JSON_ATTEMPTS"]),
    "semantic_json_attempts": int(os.environ["PHYSICSVERIFIER_SEMANTIC_JSON_ATTEMPTS"]),
    "unified_rule_top_n": int(os.environ["PHYSICSVERIFIER_UNIFIED_RULE_TOP_N"]),
    "precision_mode": os.environ["PHYSICSVERIFIER_PRECISION_MODE"],
    "max_diagnostics_per_sample": int(os.environ["PHYSICSVERIFIER_MAX_DIAGNOSTICS_PER_SAMPLE"]),
    "max_diagnostics_per_paragraph": int(os.environ["PHYSICSVERIFIER_MAX_DIAGNOSTICS_PER_PARAGRAPH"]),
    "llm_cache_enabled": os.environ["PHYSICSVERIFIER_ENABLE_LLM_CACHE"].lower() in truthy,
    "require_provider_identity": os.environ["PHYSICSVERIFIER_REQUIRE_PROVIDER_IDENTITY"].lower() in truthy,
    "max_response_chars": int(os.environ.get("PHYSICS_REWARD_MAX_RESPONSE_CHARS", "12000")),
    "paragraph_min_chars": int(os.environ.get("PHYSICS_REWARD_PARA_MIN", "150")),
    "paragraph_target_chars": int(os.environ.get("PHYSICS_REWARD_PARA_TARGET", "220")),
    "paragraph_max_chars": int(os.environ.get("PHYSICS_REWARD_PARA_MAX", "280")),
    "reward_cache_size": int(os.environ.get("PHYSICS_REWARD_CACHE_SIZE", "4096")),
    "llm_step_prompt_version": "llm_step_v1",
    "llm_step_model": "deepseek-v4-flash",
    "llm_step_timeout": float(os.environ.get("LLM_STEP_JUDGE_TIMEOUT", "300")),
    "llm_step_max_tokens": int(os.environ.get("LLM_STEP_JUDGE_MAX_TOKENS", "4096")),
    "llm_step_max_retries": int(os.environ.get("LLM_STEP_JUDGE_MAX_RETRIES", "6")),
    "llm_step_concurrency": int(os.environ.get("LLM_STEP_JUDGE_CONCURRENCY", "32")),
    "metrics_log": os.environ.get(
        "PHYSICS_REWARD_METRICS_LOG",
        os.path.join(os.environ["PHYSICS_ROOT"], "logs/physics_reward_metrics.jsonl"),
    ),
}
raise SystemExit(0 if actual == expected else 1)
'; then
        echo "[ok] reward server already running pid=$old_pid with matching config"
        exit 0
      fi
      echo "[reward] running server config differs; restarting"
      kill -TERM "$old_pid" 2>/dev/null || true
      sleep 2
      kill -9 "$old_pid" 2>/dev/null || true
    else
      echo "[reward] stale pid=${old_pid} is alive but health check failed; restarting"
      kill -TERM "$old_pid" 2>/dev/null || true
      sleep 2
      kill -9 "$old_pid" 2>/dev/null || true
    fi
  fi
fi

if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  occupant="$(port_pid)"
  echo "[reward] ${HOST}:${PORT} is occupied by pid=${occupant:-unknown} without a trusted matching PID; restarting"
  if [[ -n "${occupant}" ]]; then
    kill -TERM "${occupant}" 2>/dev/null || true
    sleep 2
    kill -9 "${occupant}" 2>/dev/null || true
  fi
fi

if [[ "${PHYSICS_REWARD_MODE}" == "answer_only" ]]; then
  echo "[reward] answer_only mode; skipping judge/API warmup"
elif [[ "${PHYSICS_REWARD_MODE}" == "llm_step_score" && "${SKIP_LLM_PREFLIGHT:-0}" != "1" ]]; then
  echo "[reward] llm_step_score remote API at ${OPENAI_BASE_URL} model=${PHYSICSVERIFIER_LLM_MODEL}"
  "${VENV}/bin/python" - <<'PY'
import os, sys
sys.path.insert(0, os.environ.get("PHYSICS_ROOT", "/home/jinjianhan/PhysicsVerifier"))
from training.reward_server.llm_step_judge import DEFAULT_MODEL, LLMStepJudge, require_remote_model
model = os.environ.get("PHYSICSVERIFIER_LLM_MODEL", DEFAULT_MODEL)
if model != DEFAULT_MODEL:
    print(f"[error] refusing model fallback: {model} (required {DEFAULT_MODEL})", file=sys.stderr)
    sys.exit(2)
require_remote_model(model)
judge = LLMStepJudge.from_env()
judge.score_group(
    "A mass m is at rest on a frictionless table. What is its acceleration?",
    ["Net force is zero so a=0.", "The answer is 42 without derivation."],
)
print("[ok] llm_step_score preflight passed")
PY
elif [[ "${PHYSICS_REWARD_MODE}" == "llm_step_score" ]]; then
  echo "[reward] llm_step_score skipping preflight; remote API at ${OPENAI_BASE_URL} model=${PHYSICSVERIFIER_LLM_MODEL}"
else
  if [[ -n "${CONFIGURED_OPENAI_BASE_URL}" ]]; then
    echo "[reward] using external verifier API at ${OPENAI_BASE_URL}"
  else
    echo "[reward] using local verifier API at ${OPENAI_BASE_URL}"
  fi
  "${VENV}/bin/python" - <<'PY'
import os, sys
from openai import OpenAI
base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
key = os.environ.get("OPENAI_API_KEY", "EMPTY")
model = os.environ.get("PHYSICSVERIFIER_LLM_MODEL", "")
client = OpenAI(base_url=base, api_key=key)
models = [m.id for m in client.models.list().data]
if model and model not in models:
    print(f"[error] configured model {model} not in provider list; available={models[:5]}", file=sys.stderr)
    raise SystemExit(2)
print("[ok] verifier API and configured model are available")
PY
fi

nohup "${VENV}/bin/python" "${ROOT}/training/reward_server/physics_reward_server.py" \
  --host "$HOST" --port "$PORT" \
  --reward-mode "${PHYSICS_REWARD_MODE}" \
  --concurrency "${PHYSICS_REWARD_CONCURRENCY}" \
  >"$LOG" 2>&1 &
echo $! >"$PID_FILE"
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
