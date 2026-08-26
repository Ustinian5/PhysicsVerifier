#!/usr/bin/env bash
# Download Qwen3-8B BF16 weights for 4-GPU OpenRLHF training.
set -u

DEST="${DEST:-/slow_share/jinjianhan/models/Qwen3-8B}"
REPO_ID="${REPO_ID:-Qwen/Qwen3-8B}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
ENV_FILE="${WORKSPACE}/openrlhf_rl/env.sh"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
PYTHON="${PYTHON:-${TRAIN_VENV:-/data1/jinjianhan/venv/openrlhf_train}/bin/python}"
LOG="${LOG:-/slow_share/jinjianhan/workspace/openrlhf_rl/download_qwen3_8b.log}"
MIN_BYTES="${MIN_BYTES:-14000000000}"
SLEEP_SEC="${SLEEP_SEC:-30}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"

export HF_HOME="${HF_HOME:-/slow_share/jinjianhan/models/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-900}"

mkdir -p "$DEST" "$(dirname "$LOG")" "$HF_HOME"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

is_complete() {
  local sz
  sz=$(du -sb "$DEST" 2>/dev/null | awk '{print $1}')
  [[ "${sz:-0}" -ge "$MIN_BYTES" ]] || return 1
  [[ -f "${DEST}/config.json" ]] || return 1
  [[ -f "${DEST}/model.safetensors" || -f "${DEST}/model.safetensors.index.json" ]] || return 1
}

attempt=0
log "start 8B download repo=${REPO_ID} dest=${DEST}"
while ! is_complete; do
  attempt=$((attempt + 1))
  if [[ "$MAX_ATTEMPTS" -gt 0 && "$attempt" -gt "$MAX_ATTEMPTS" ]]; then
    log "ERROR: exceeded MAX_ATTEMPTS=${MAX_ATTEMPTS}"
    exit 1
  fi
  log "attempt=${attempt}"
  if "$PYTHON" - <<PY >>"$LOG" 2>&1
import os
repo = "${REPO_ID}"
dest = "${DEST}"
try:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=dest, local_dir_use_symlinks=False)
    print("hf download pass returned")
except Exception as exc:
    print(f"hf download failed: {exc}")
    try:
        from modelscope import snapshot_download as ms_download
        ms_download(repo, local_dir=dest)
        print("modelscope download pass returned")
    except Exception as exc2:
        print(f"modelscope download failed: {exc2}")
        raise
PY
  then
    if is_complete; then
      log "SUCCESS: $(du -sh "$DEST" | awk '{print $1}') at ${DEST}"
      exit 0
    fi
    log "WARN: incomplete after pass; retrying"
  else
    log "WARN: download failed; retrying in ${SLEEP_SEC}s"
  fi
  sleep "$SLEEP_SEC"
done
