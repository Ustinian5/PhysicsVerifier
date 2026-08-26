#!/usr/bin/env bash
# Fixed-prompt profile for reward latency and filtering diagnostics.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
OUT="${OUT:-${ROOT}/results/openrlhf_dryrun_profile.json}"
INPUT="${INPUT:-${ROOT}/data/rl/heldout_eval.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-16}"

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
"${PYTHON}" - <<PY >"${OUT}"
import json, time, requests, statistics
from pathlib import Path

def prompt_text(row):
    prompt = row.get("input") or row.get("prompt") or ""
    if isinstance(prompt, list):
        return " ".join(str(m.get("content", "")) for m in prompt if isinstance(m, dict))
    return str(prompt)

rows = []
with Path("${INPUT}").open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        if i >= int("${MAX_SAMPLES}"):
            break
        row = json.loads(line)
        prompt = prompt_text(row)
        label = row.get("label") or ""
        payload = {"query": [prompt + "\\nassistant\\n\\\\boxed{1}"], "prompts": [prompt], "labels": [label]}
        t0 = time.time()
        resp = requests.post("http://127.0.0.1:8770/get_reward", json=payload, timeout=180)
        resp.raise_for_status()
        dt = time.time() - t0
        data = resp.json()
        rows.append({"reward": data["rewards"][0], "latency_s": dt, "extra": data.get("extra_logs", {})})

latencies = sorted(r["latency_s"] for r in rows)
summary = {
    "n": len(rows),
    "reward_mean": sum(r["reward"] for r in rows) / max(len(rows), 1),
    "latency_p50": latencies[len(latencies) // 2] if latencies else 0.0,
    "latency_p95": latencies[int(len(latencies) * 0.95)] if latencies else 0.0,
    "rows": rows,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
echo "[ok] dryrun profile -> ${OUT}"
