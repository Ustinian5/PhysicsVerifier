#!/usr/bin/env bash
# Re-score one frozen rollout set under each reward mode.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
VENV="${VENV:-${ROOT}/.venv}"
PYTHON="${PYTHON:-${VENV}/bin/python}"
INPUT="${INPUT:-${ROOT}/data/rl/baseline_rollout_scores.jsonl}"
OUT_DIR="${OUT_DIR:-${ROOT}/results/reward_ablation}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8770}"

mkdir -p "${OUT_DIR}"
if [[ ! -s "${INPUT}" ]]; then
  echo "[error] frozen rollout file is missing or empty: ${INPUT}" >&2
  echo "[error] reward ablation requires stored responses, not prompt-only heldout data" >&2
  exit 2
fi
MODES=(answer_only answer_low_verifier answer_full_verifier)

for mode in "${MODES[@]}"; do
  export PHYSICS_REWARD_MODE="${mode}"
  if [[ "${mode}" == "answer_low_verifier" ]]; then
    export PHYSICS_REWARD_W_VERIFIER="0.1"
  elif [[ "${mode}" == "answer_full_verifier" ]]; then
    export PHYSICS_REWARD_W_VERIFIER="${PHYSICS_REWARD_LAMBDA:-0.3}"
  fi
  bash "${ROOT}/training/reward_server/start_reward_server.sh"
  OUT_JSON="${OUT_DIR}/${mode}.json"
  "${PYTHON}" - <<PY
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
        prompt = row.get("input") if "input" in row else row.get("prompt")
        if isinstance(prompt, list):
            prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in prompt if isinstance(m, dict)
            )
        prompt = str(prompt or "")
        response = row.get("response")
        label = row.get("label")
        if response is None or not str(response).strip():
            raise ValueError(f"row {i} has no stored response")
        if label is None or (isinstance(label, str) and not label.strip()):
            raise ValueError(f"row {i} has no ground-truth label")
        query = prompt + str(response)
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
