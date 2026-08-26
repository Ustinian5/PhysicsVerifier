#!/usr/bin/env bash
# Compare reward modes on a fixed heldout prompt slice.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
VENV="${VENV:-${ROOT}/.venv}"
INPUT="${INPUT:-${ROOT}/data/rl/heldout_eval.jsonl}"
OUT_DIR="${OUT_DIR:-${ROOT}/results/reward_ablation}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8770}"

mkdir -p "${OUT_DIR}"
MODES=(answer_only)
if [[ -n "${PHYSICSVERIFIER_OPENAI_BASE_URL:-}" ]] || curl -sf http://127.0.0.1:8766/v1/models >/dev/null 2>&1; then
  MODES+=(answer_low_verifier answer_full_verifier)
fi

for mode in "${MODES[@]}"; do
  export PHYSICS_REWARD_MODE="${mode}"
  if [[ "${mode}" == "answer_low_verifier" ]]; then
    export PHYSICS_REWARD_W_VERIFIER="0.1"
  elif [[ "${mode}" == "answer_full_verifier" ]]; then
    export PHYSICS_REWARD_W_VERIFIER="${PHYSICS_REWARD_LAMBDA:-0.3}"
  fi
  bash "${ROOT}/training/reward_server/start_reward_server.sh"
  OUT_JSON="${OUT_DIR}/${mode}.json"
  "${VENV}/bin/python" - <<PY
import json, os, requests
from pathlib import Path

root = Path("${ROOT}")
rows = []
with Path("${INPUT}").open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        if int("${MAX_SAMPLES}") and i >= int("${MAX_SAMPLES}"):
            break
        row = json.loads(line)
        prompt = row.get("input") or row.get("prompt") or ""
        if isinstance(prompt, list):
            prompt = " ".join(
                str(m.get("content", "")) for m in prompt if isinstance(m, dict)
            )
        label = row.get("label") or ""
        query = f"{prompt}\\nassistant\\nplaceholder"
        payload = {"query": [query], "prompts": [prompt], "labels": [label]}
        resp = requests.post("http://${HOST}:${PORT}/get_reward", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        rows.append({
            "id": row.get("id"),
            "reward": data["rewards"][0],
            "extra_logs": data.get("extra_logs", {}),
        })
summary = {
    "mode": "${mode}",
    "n": len(rows),
    "reward_mean": sum(r["reward"] for r in rows) / max(len(rows), 1),
    "rows": rows,
}
Path("${OUT_JSON}").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"mode": "${mode}", "reward_mean": summary["reward_mean"], "n": summary["n"]}, ensure_ascii=False))
PY
done

echo "[ok] reward ablation outputs in ${OUT_DIR}"
