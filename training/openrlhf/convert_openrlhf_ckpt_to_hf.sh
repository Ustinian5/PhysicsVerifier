#!/usr/bin/env bash
# Convert an OpenRLHF DeepSpeed ZeRO-3 actor checkpoint to HuggingFace safetensors.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
ENV_FILE="${ENV_FILE:-${WORKSPACE}/openrlhf_rl/env.sh}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

PYTHON="${PYTHON:-${TRAIN_VENV:-/data1/jinjianhan/venv/openrlhf_train}/bin/python}"
CKPT_ROOT="${CKPT_ROOT:-/slow_share/jinjianhan/ckpt/qwen3-30b-physics-openrlhf/ckpt}"
ACTOR_DIR="${ACTOR_DIR:-${CKPT_ROOT}/_actor}"
TAG="${TAG:?set TAG=global_stepN}"
BASE_MODEL="${BASE_MODEL:-/slow_share/jinjianhan/models/Qwen3-30B-A3B-Instruct-2507}"
OUT_DIR="${OUT_DIR:-${CKPT_ROOT}/${TAG}_hf}"
ZERO_SCRIPT="${ZERO_SCRIPT:-${ACTOR_DIR}/zero_to_fp32.py}"
MAX_SHARD="${MAX_SHARD:-5GB}"

if [[ ! -d "${ACTOR_DIR}/${TAG}" ]]; then
  echo "[error] missing actor checkpoint: ${ACTOR_DIR}/${TAG}" >&2
  exit 2
fi
if [[ ! -f "${ZERO_SCRIPT}" ]]; then
  echo "[error] missing zero_to_fp32.py at ${ZERO_SCRIPT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUT_DIR}")"
if [[ -f "${OUT_DIR}/model.safetensors.index.json" || -f "${OUT_DIR}/model.safetensors" ]]; then
  echo "[ok] HF export already exists: ${OUT_DIR}"
  exit 0
fi

TMP_OUT="${OUT_DIR}.tmp.$$"
rm -rf "${TMP_OUT}"
mkdir -p "${TMP_OUT}"

echo "[convert] ZeRO -> fp32 safetensors tag=${TAG}"
"${PYTHON}" "${ZERO_SCRIPT}" "${ACTOR_DIR}" "${TMP_OUT}" \
  --tag "${TAG}" \
  --safe_serialization \
  --max_shard_size "${MAX_SHARD}"

echo "[convert] copy tokenizer/config from ${BASE_MODEL}"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json vocab.json merges.txt added_tokens.json chat_template.jinja; do
  if [[ -f "${BASE_MODEL}/${f}" ]]; then
    cp -a "${BASE_MODEL}/${f}" "${TMP_OUT}/"
  fi
done

"${PYTHON}" - <<PY
from pathlib import Path
import json

out = Path("${TMP_OUT}")
idx = out / "model.safetensors.index.json"
single = out / "model.safetensors"
if not idx.exists() and not single.exists():
    raise SystemExit(f"missing weight shards under {out}")
if idx.exists():
    meta = json.loads(idx.read_text(encoding="utf-8"))
    shards = meta.get("weight_map", {})
    missing = [name for name in shards.values() if not (out / name).exists()]
    if missing:
        raise SystemExit(f"missing shard files: {missing[:5]}")
print(json.dumps({"ok": True, "out": str(out), "has_index": idx.exists()}, ensure_ascii=False))
PY

rm -rf "${OUT_DIR}"
mv "${TMP_OUT}" "${OUT_DIR}"
printf '%s\n' "${TAG}" > "${OUT_DIR}/.openrlhf_tag"
echo "[ok] exported HF checkpoint -> ${OUT_DIR}"
