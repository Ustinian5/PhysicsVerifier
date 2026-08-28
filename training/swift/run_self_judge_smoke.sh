#!/usr/bin/env bash
# Score ~50 v9 rollouts with 8B self-judge (local) vs 30B (remote API).
# Does not require a local 30B GPU.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
SFT_CKPT="${QWEN8B_SFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-sft}"
if [[ ! -f "${SFT_CKPT}/config.json" ]]; then
  SFT_CKPT="$(ls -d "${SFT_CKPT}"/v*-*/checkpoint-* 2>/dev/null | tail -1 || true)"
fi
[[ -f "${SFT_CKPT}/config.json" ]] || { echo "[error] missing SFT ckpt" >&2; exit 2; }
ROLLOUTS="${ROLLOUTS:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-swift/v9-20260825-095707/completions.jsonl}"
OUT_DIR="${OUT_DIR:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-swift}"
SMOKE_PORT="${SMOKE_JUDGE_PORT:-8784}"
GPU="${SMOKE_JUDGE_GPU:-0}"
LIMIT="${SMOKE_LIMIT:-50}"

REMOTE_BASE=""
REMOTE_KEY=""
if [[ -f "${ROOT}/.env" ]]; then
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
fi

RUN_ID=self_judge_smoke MODEL_DIR="${SFT_CKPT}" PORT="${SMOKE_PORT}" CUDA_DEVICE="${GPU}" \
  MAX_LEN=8192 GPU_UTIL=0.45 SERVED_NAME=qwen3-8b-self-judge \
  LOG="${ROOT}/logs/self_judge_smoke_vllm.log" PID_FILE="${ROOT}/logs/self_judge_smoke_vllm.pid" \
  ENABLE_PREFIX_CACHING=1 VLLM_READY_SECS=600 \
  bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" start

export PHYSICS_REWARD_MODE=process_paragraph
export PHYSICS_REWARD_W_ANSWER=0
export PHYSICS_REWARD_W_FORMAT=0
export PHYSICSVERIFIER_UNIFIED_RULE_TOP_N="${PHYSICSVERIFIER_UNIFIED_RULE_TOP_N:-4}"
export PHYSICSVERIFIER_PRECISION_MODE="${PHYSICSVERIFIER_PRECISION_MODE:-strict}"
export PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE=lexical
export PHYSICSVERIFIER_LLM_MODEL=qwen3-8b-self-judge
export OPENAI_BASE_URL="http://127.0.0.1:${SMOKE_PORT}/v1"
export OPENAI_API_KEY=EMPTY
unset PHYSICSVERIFIER_OPENAI_BASE_URL PHYSICSVERIFIER_OPENAI_API_KEY
PORT=8770 PID_FILE="${ROOT}/logs/physics_reward_server.pid" \
  LOG="${ROOT}/logs/physics_reward_server.log" \
  bash "${ROOT}/training/reward_server/start_reward_server.sh"

"${PYTHON}" "${ROOT}/training/swift/smoke_self_judge.py" \
  --rollouts "${ROLLOUTS}" \
  --url-a "http://127.0.0.1:8770/get_reward" \
  --limit "${LIMIT}" \
  --output "${OUT_DIR}/self_judge_smoke_8b.json"

if [[ -z "${REMOTE_BASE}" || "${REMOTE_BASE}" == *"127.0.0.1"* || -z "${REMOTE_KEY}" || "${REMOTE_KEY}" == "EMPTY" ]]; then
  echo "[error] remote 30B API not configured in .env" >&2
  exit 2
fi
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
  --output "${OUT_DIR}/self_judge_smoke_30b.json"

"${PYTHON}" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "${ROOT}")
from training.swift.smoke_self_judge import spearman
out = Path("${OUT_DIR}")
a = json.loads((out / "self_judge_smoke_8b.json").read_text())
b = json.loads((out / "self_judge_smoke_30b.json").read_text())
sa, sb = a["scores_a"], b["scores_a"]
n = min(len(sa), len(sb))
sa, sb = sa[:n], sb[:n]
rho = spearman(sa, sb) if n else 0.0
agree = sum(1 for x, y in zip(sa, sb) if abs(x - y) <= 0.1) / max(n, 1)
report = {
    "n": n,
    "mean_8b": a.get("mean_a"),
    "mean_30b": b.get("mean_a"),
    "std_8b": a.get("std_a"),
    "std_30b": b.get("std_a"),
    "spearman": rho,
    "agree_within_0.1": agree,
    "pass": bool(n and rho >= 0.3),
}
(out / "self_judge_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if not report["pass"]:
    print("[warn] 8B vs 30B Spearman < 0.3; prefer JUDGE_REFRESH=0 (fixed SFT judge)", flush=True)
PY

RUN_ID=self_judge_smoke PORT="${SMOKE_PORT}" bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" stop || true
echo "[ok] smoke report ${OUT_DIR}/self_judge_smoke.json"
