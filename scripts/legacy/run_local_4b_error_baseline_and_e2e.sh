#!/usr/bin/env bash
# Error-level: local Qwen3-4B-AWQ (vLLM) semantic baseline + PhysicsVerifier e2e.
#
# Usage (screen recommended; total wall time often 1–3h):
#   bash scripts/legacy/run_local_4b_error_baseline_and_e2e.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
START_VLLM="${START_VLLM:-/home/jinjianhan/deploy/qwen3_q4/start_vllm_4b.sh}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8765}"
LOCAL_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
CHECK_MODEL="${CHECK_MODEL:-qwen3-4b-instruct-2507}"
WAIT_VLLM_SEC="${WAIT_VLLM_SEC:-600}"

ENV_FILE="${ENV_FILE:-$ROOT/.env}"

load_repo_dotenv() {
  if [[ ! -f "$ENV_FILE" ]]; then
    return 0
  fi
  local loaded
  loaded="$("$PYTHON" - "$ENV_FILE" <<'PY'
import os, shlex, sys
from pathlib import Path
env_path = Path(sys.argv[1])
keys = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE")
try:
    from dotenv import load_dotenv
    load_dotenv(env_path, override=False)
except ImportError:
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key not in keys or key in os.environ:
                continue
            os.environ[key] = value.strip().strip('"').strip("'")
for key in keys:
    val = os.environ.get(key)
    if val:
        print(f"export {key}={shlex.quote(val)}")
PY
)" || true
  if [[ -n "$loaded" ]]; then
    # shellcheck disable=SC1090
    eval "$loaded"
  fi
}

load_repo_dotenv
export OPENAI_BASE_URL="$LOCAL_BASE_URL"
export OPENAI_API_BASE="$LOCAL_BASE_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-dummy-key}"
export OPENAI_DISABLE_THINKING="${OPENAI_DISABLE_THINKING:-1}"

echo "[local-4b] starting / waiting for vLLM at $LOCAL_BASE_URL"
bash "$START_VLLM"

deadline=$(( $(date +%s) + WAIT_VLLM_SEC ))
until curl -sf "${LOCAL_BASE_URL}/models" >/dev/null 2>&1; do
  if [[ $(date +%s) -ge $deadline ]]; then
    echo "[error] vLLM not ready after ${WAIT_VLLM_SEC}s; see /home/jinjianhan/deploy/qwen3_q4/vllm_4b.log" >&2
    exit 2
  fi
  sleep 5
done
echo "[ok] vLLM ready; model list:"
curl -s "${LOCAL_BASE_URL}/models" | "$PYTHON" -c "import json,sys; d=json.load(sys.stdin); print([m.get('id') for m in d.get('data',[])])" 2>/dev/null || true

# Quick smoke call
"$PYTHON" - <<PY
import os
from openai import OpenAI
c = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
r = c.chat.completions.create(
    model="${CHECK_MODEL}",
    messages=[{"role": "user", "content": "Reply with OK only."}],
    max_tokens=8,
    temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print("[smoke]", (r.choices[0].message.content or "")[:80])
PY

export CHECK_MODEL
export PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
export SYMBOLIC_TOPIC_CHECK_LIMIT="${SYMBOLIC_TOPIC_CHECK_LIMIT:-32}"
export TAG_MAIN="${TAG_MAIN:-e2e_main_error_4b_local}"
export TAG_BASE="${TAG_BASE:-baseline_error_4b_local}"

bash "$ROOT/scripts/legacy/run_dualchain_4b_error_level_e2e_and_baseline.sh"

STAMP="$(cat "$ROOT/results/_dualchain_4b_error_only_stamp.txt")"
MAIN_OUT="$ROOT/results/${TAG_MAIN}_${STAMP}"
BASE_OUT="$ROOT/results/${TAG_BASE}_${STAMP}"

echo ""
echo "================================================================"
echo "[summary] STAMP=$STAMP local 4B error-level"
"$PYTHON" - <<PY
import json
from pathlib import Path

def row(label, path):
    p = Path(path) / "error_metrics.json"
    if not p.exists():
        print(f"  {label}: (missing metrics)")
        return
    s = json.loads(p.read_text())["summary"]
    print(
        f"  {label}: recall={s['recall']:.4f} precision={s['precision']:.4f} f1={s['f1']:.4f} "
        f"matched={s['matched_gt_errors']}/{s['total_gt_errors']}"
    )

root = Path("${ROOT}")
stamp = "${STAMP}"
row("E2E", root / f"results/e2e_main_error_4b_{stamp}")
row("Baseline", root / f"results/baseline_error_4b_{stamp}")
PY
echo "  E2E dir:      $MAIN_OUT"
echo "  Baseline dir: $BASE_OUT"
echo "================================================================"
