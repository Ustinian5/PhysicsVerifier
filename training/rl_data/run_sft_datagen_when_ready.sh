#!/usr/bin/env bash
# Wait for 3 local 30B generators, then rejection-sample SFT solutions + API fill.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
LB_PORT="${SFT_GEN_LB_PORT:-8780}"
PORTS=(8781 8782 8783)
SERVED_NAME="${SFT_GEN_MODEL:-qwen3-30b-a3b}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
mkdir -p "${LOG_DIR}"

echo "[sft-gen] waiting for backends ${PORTS[*]}"
for port in "${PORTS[@]}"; do
  ready=0
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[sft-gen] ready :${port}"
      ready=1
      break
    fi
    sleep 30
  done
  [[ "${ready}" -eq 1 ]] || { echo "[error] :${port} not ready after long wait" >&2; exit 2; }
done

if [[ -f "${LOG_DIR}/sft_gen_lb.pid" ]]; then
  old="$(cat "${LOG_DIR}/sft_gen_lb.pid" 2>/dev/null || true)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    kill -TERM "${old}" 2>/dev/null || true
    sleep 1
    kill -9 "${old}" 2>/dev/null || true
  fi
fi
nohup "${PYTHON}" "${ROOT}/training/openrlhf/judge_lb_proxy.py" \
  --host 127.0.0.1 --port "${LB_PORT}" \
  --backends "127.0.0.1:${PORTS[0]},127.0.0.1:${PORTS[1]},127.0.0.1:${PORTS[2]}" \
  >>"${LOG_DIR}/sft_gen_lb.log" 2>&1 &
echo $! >"${LOG_DIR}/sft_gen_lb.pid"
for _ in $(seq 1 40); do
  curl -sf "http://127.0.0.1:${LB_PORT}/health" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -sf "http://127.0.0.1:${LB_PORT}/v1/models" >/dev/null \
  || { echo "[error] SFT gen LB not ready on :${LB_PORT}" >&2; exit 2; }

echo "[sft-gen] local rejection sampling"
"${PYTHON}" "${ROOT}/training/rl_data/generate_sft_solutions.py" \
  --base-url "http://127.0.0.1:${LB_PORT}/v1" \
  --model "${SERVED_NAME}" \
  --k "${SFT_GEN_K:-4}" \
  --concurrency "${SFT_GEN_CONCURRENCY:-18}" \
  --local-only \
  "$@"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
if [[ -n "${OPENAI_API_KEY:-}" && "${OPENAI_API_KEY}" != "EMPTY" && -n "${OPENAI_BASE_URL:-}" && "${OPENAI_BASE_URL}" != *"127.0.0.1"* ]]; then
  echo "[sft-gen] filling unsolved via API"
  SFT_API_MODEL="${SFT_API_MODEL:-qwen3-30b-a3b-instruct-2507}" \
    "${PYTHON}" "${ROOT}/training/rl_data/generate_sft_solutions.py" \
      --api-only \
      --api-base-url "${OPENAI_BASE_URL}" \
      --api-key "${OPENAI_API_KEY}" \
      --api-model "${SFT_API_MODEL}" \
      --concurrency "${SFT_API_CONCURRENCY:-8}"
fi
echo "[ok] SFT datagen finished"
# Free GPUs 4–6 for later self-judge / GRPO. Stop by pid file + exact port only.
for i in 0 1 2; do
  RUN_ID="sft_gen${i}" PORT="${PORTS[$i]}" PID_FILE="${LOG_DIR}/sft_gen${i}_vllm.pid" \
    bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" stop || true
done
if [[ -f "${LOG_DIR}/sft_gen_lb.pid" ]]; then
  old="$(cat "${LOG_DIR}/sft_gen_lb.pid" 2>/dev/null || true)"
  if [[ -n "${old}" ]]; then
    kill -TERM "${old}" 2>/dev/null || true
    sleep 1
    kill -9 "${old}" 2>/dev/null || true
  fi
  rm -f "${LOG_DIR}/sft_gen_lb.pid"
fi
echo "[ok] stopped SFT-gen 30B replicas"
