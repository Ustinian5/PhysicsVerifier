#!/usr/bin/env bash
# Check prerequisites for 8-GPU (4 train + 4 judge) Qwen3-8B OpenRLHF training.
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
MIN_TRAIN="${MIN_TRAIN_GPUS:-4}"
JUDGE_LB="${JUDGE_LB_URL:-http://127.0.0.1:8765/v1}"
RAY_GCS_PORT="${RAY_GCS_PORT:-26379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-28265}"
RAY_CLIENT_PORT="${RAY_CLIENT_PORT:-26380}"

errors=0
warns=0
check() { local n="$1"; shift; if "$@" >/dev/null 2>&1; then echo "[ok] $n"; else echo "[fail] $n"; errors=$((errors+1)); fi; }
warn() { local n="$1"; shift; if "$@" >/dev/null 2>&1; then echo "[ok] $n"; else echo "[warn] $n"; warns=$((warns+1)); fi; }

check "CUDA usable" bash "${ROOT}/training/openrlhf/ensure_cuda_ready.sh"
check "reward server /health" curl -sf http://127.0.0.1:8770/health
warn "reward server /get_reward (bounded)" bash -c 'timeout 15 curl -sf -X POST http://127.0.0.1:8770/get_reward -H "Content-Type: application/json" -d "{\"query\":[\"a\"],\"prompts\":[\"\"],\"labels\":[\"b\"]}" | grep -q rewards'
check "train prompts" test -s "${PROMPT_DATA:-${ROOT}/data/rl/openrlhf_prompts.jsonl}"
nvis="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:-}")"
check "${MIN_TRAIN} visible train GPUs" bash -c "[[ ${nvis} -ge ${MIN_TRAIN} ]]"
check "8B model dir" test -d "${QWEN8B_MODEL_DIR}"
check "8B config" test -f "${QWEN8B_MODEL_DIR}/config.json"
warn "openrlhf import" "${PYTHON}" -c "import openrlhf"
warn "vllm import" "${PYTHON}" -c "import vllm"
warn "fabricmanager active" systemctl is-active nvidia-fabricmanager

if [[ -n "${PHYSICSVERIFIER_OPENAI_BASE_URL:-}" ]]; then
  check "external verifier API env" bash -c '[[ -n "${PHYSICSVERIFIER_OPENAI_BASE_URL}" ]]'
elif [[ "${MODE}" != "answer_only" ]]; then
  check "judge-1 /v1/models" curl -sf http://127.0.0.1:8766/v1/models
  check "judge-2 /v1/models" curl -sf http://127.0.0.1:8767/v1/models
  check "judge-3 /v1/models" curl -sf http://127.0.0.1:8768/v1/models
  check "judge-4 /v1/models" curl -sf http://127.0.0.1:8769/v1/models
  check "judge LB /v1/models" curl -sf "${JUDGE_LB}/models"
  check "judge LB 4 backends" bash -c '
    n=$(curl -sf http://127.0.0.1:8765/health | python3 -c "import json,sys; print(len(json.load(sys.stdin).get(\"backends\",[])))")
    [[ "${n}" -ge 4 ]]
  '
fi

# Fail closed if Ray / judge / reward listen on non-loopback addresses.
check "loopback listeners" bash -c "
  ss -ltn 2>/dev/null | python3 - <<'PY'
import re, sys
bad = []
pat = re.compile(r':(%s)\s' % '|'.join([
    '${RAY_GCS_PORT}', '${RAY_DASHBOARD_PORT}', '${RAY_CLIENT_PORT}',
    '8765', '8766', '8767', '8768', '8769', '8770',
]))
for line in sys.stdin:
    line = line.strip()
    if not pat.search(line):
        continue
    # ss: LISTEN 0 128 127.0.0.1:26379 ...
    parts = line.split()
    for tok in parts:
        if ':' not in tok:
            continue
        host = tok.rsplit(':', 1)[0]
        host = host.strip('[]')
        if host.startswith('::ffff:'):
            host = host.split('::ffff:')[-1]
        if host in ('127.0.0.1', '::1', '[::1]'):
            continue
        if host in ('0.0.0.0', '*', '[::]', '::'):
            bad.append(line)
            break
        # Any other host (LAN IP) is also bad.
        if host and host not in ('127.0.0.1', '::1'):
            bad.append(line)
            break
if bad:
    for b in bad:
        print('NON_LOOPBACK', b, file=sys.stderr)
    sys.exit(1)
PY
"

echo "---"
echo "errors=$errors warns=$warns mode=${MODE} train_gpus=${CUDA_VISIBLE_DEVICES:-}"
[[ "$errors" -eq 0 ]]
