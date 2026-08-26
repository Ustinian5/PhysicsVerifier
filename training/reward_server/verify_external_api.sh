#!/usr/bin/env bash
# Verify external OpenAI-compatible API reward backend (or answer_only fallback).
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
OUT="${OUT:-${ROOT}/results/api_reward_verify.json}"

mkdir -p "$(dirname "${OUT}")"

if [[ -n "${PHYSICSVERIFIER_OPENAI_BASE_URL:-}" ]]; then
  echo "[verify] external API mode: ${PHYSICSVERIFIER_OPENAI_BASE_URL}"
  export PHYSICS_REWARD_MODE="${PHYSICS_REWARD_MODE:-answer_low_verifier}"
else
  echo "[verify] no PHYSICSVERIFIER_OPENAI_BASE_URL; using answer_only fallback"
  export PHYSICS_REWARD_MODE=answer_only
fi

bash "${ROOT}/training/reward_server/start_reward_server.sh"

python3 - <<'PY' >"${OUT}"
import json, os, time, requests

url = "http://127.0.0.1:8770/get_reward"
payload = {
    "query": ["Solve.\\nassistant\\n\\\\boxed{42}"],
    "prompts": ["Solve."],
    "labels": ["42"],
}
t0 = time.time()
resp = requests.post(url, json=payload, timeout=120)
resp.raise_for_status()
data = resp.json()
summary = {
    "mode": os.environ.get("PHYSICS_REWARD_MODE", ""),
    "external_api": bool(os.environ.get("PHYSICSVERIFIER_OPENAI_BASE_URL")),
    "openai_base_url": os.environ.get("OPENAI_BASE_URL", ""),
    "latency_s": time.time() - t0,
    "reward": data.get("rewards", [None])[0],
    "extra_logs": data.get("extra_logs", {}),
    "status": "ok",
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[ok] API reward verify -> ${OUT}"
