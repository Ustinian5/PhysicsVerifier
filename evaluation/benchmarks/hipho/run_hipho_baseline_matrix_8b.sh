#!/usr/bin/env bash
# HiPhO + heldout answer-accuracy matrix for Qwen3-8B base / OpenRLHF / ms-swift checkpoints.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VENV="${VENV:-${ROOT}/.venv}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
BASE_MODEL="${BASE_MODEL:-/slow_share/jinjianhan/models/Qwen3-8B}"
ORHF_CKPT_ROOT="${ORHF_CKPT_ROOT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-openrlhf/ckpt}"
SWIFT_CKPT_ROOT="${SWIFT_CKPT_ROOT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-swift}"
MATRIX_DIR="${MATRIX_DIR:-${ROOT}/results/hipho_baseline_matrix_8b}"
BENCH_ROOT="${BENCH_ROOT:-/slow_share/jinjianhan/workspace/benchmarks/hipho}"
HIPHO_JSONL="${HIPHO_JSONL:-${BENCH_ROOT}/hipho_text_only.jsonl}"
HELDOUT_JSONL="${HELDOUT_JSONL:-${ROOT}/data/rl/heldout_eval.jsonl}"
PORT="${PORT:-8766}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TEMPERATURE="${TEMPERATURE:-0.2}"
# Leave headroom for system/user prompt under max_model_len.
MAX_LEN="${MAX_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
GPU_UTIL="${GPU_UTIL:-0.45}"
SERVED_NAME="${SERVED_NAME:-qwen3-8b}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-24}"

mkdir -p "${MATRIX_DIR}"
if [[ ! -f "${HIPHO_JSONL}" ]]; then
  bash "${ROOT}/evaluation/benchmarks/hipho/setup_hipho.sh"
fi

declare -A MODEL_DIRS=(
  [base_8b]="${BASE_MODEL}"
  [openrlhf_step20]="${ORHF_CKPT_ROOT}/global_step20_hf"
  [openrlhf_step40]="${ORHF_CKPT_ROOT}/global_step40_hf"
  [swift_step10]="${SWIFT_CKPT_ROOT}/v6-20260825-032454/checkpoint-10"
  [swift_step20]="${SWIFT_CKPT_ROOT}/v7-20260825-040313/checkpoint-20"
  [swift_step40]="${SWIFT_CKPT_ROOT}/v9-20260825-095707/checkpoint-40"
)
SFT_CKPT="${QWEN8B_SFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-sft}"
if [[ -f "${SFT_CKPT}/config.json" ]]; then
  MODEL_DIRS[sft_8b]="${SFT_CKPT}"
else
  sft_latest="$(ls -d "${SFT_CKPT}"/v*-*/checkpoint-* 2>/dev/null | tail -1 || true)"
  if [[ -n "${sft_latest}" && -f "${sft_latest}/config.json" ]]; then
    MODEL_DIRS[sft_8b]="${sft_latest}"
  fi
fi
if [[ -n "${EXTRA_MODELS:-}" ]]; then
  IFS=',' read -ra _extras <<< "${EXTRA_MODELS}"
  for item in "${_extras[@]}"; do
    [[ "${item}" == *"="* ]] || continue
    MODEL_DIRS["${item%%=*}"]="${item#*=}"
  done
fi
LABELS=(base_8b openrlhf_step20 openrlhf_step40 swift_step10 swift_step20 swift_step40)
[[ -n "${MODEL_DIRS[sft_8b]+x}" ]] && LABELS+=(sft_8b)
if [[ -n "${EXTRA_LABELS:-}" ]]; then
  IFS=',' read -ra _el <<< "${EXTRA_LABELS}"
  LABELS+=("${_el[@]}")
fi

for tag in openrlhf_step20 openrlhf_step40; do
  step="${tag#openrlhf_step}"
  if [[ ! -f "${MODEL_DIRS[${tag}]}/model.safetensors.index.json" && ! -f "${MODEL_DIRS[${tag}]}/model.safetensors" ]]; then
    CKPT_ROOT="${ORHF_CKPT_ROOT}" BASE_MODEL="${BASE_MODEL}" TAG="global_${step}" \
      bash "${ROOT}/training/openrlhf/convert_openrlhf_ckpt_to_hf.sh"
  fi
done

sft_path=""
if [[ -n "${MODEL_DIRS[sft_8b]+x}" ]]; then
  sft_path="${MODEL_DIRS[sft_8b]}"
fi
python3 - <<PY >"${MATRIX_DIR}/manifest.json"
import json, os
models = {
  "base_8b": "${BASE_MODEL}",
  "openrlhf_step20": "${ORHF_CKPT_ROOT}/global_step20_hf",
  "openrlhf_step40": "${ORHF_CKPT_ROOT}/global_step40_hf",
  "swift_step10": "${SWIFT_CKPT_ROOT}/v6-20260825-032454/checkpoint-10",
  "swift_step20": "${SWIFT_CKPT_ROOT}/v7-20260825-040313/checkpoint-20",
  "swift_step40": "${SWIFT_CKPT_ROOT}/v9-20260825-095707/checkpoint-40",
}
sft = "${sft_path}"
if sft:
    models["sft_8b"] = sft
for item in os.environ.get("EXTRA_MODELS", "").split(","):
    if "=" in item:
        k, v = item.split("=", 1)
        models[k] = v
print(json.dumps({
  "temperature": float("${TEMPERATURE}"),
  "max_tokens": int("${MAX_TOKENS}"),
  "max_model_len": int("${MAX_LEN}"),
  "max_samples": int("${MAX_SAMPLES}"),
  "served_name": "${SERVED_NAME}",
  "hipho_jsonl": "${HIPHO_JSONL}",
  "heldout_jsonl": "${HELDOUT_JSONL}",
  "models": models,
}, ensure_ascii=False, indent=2))
PY

_eval_one_dataset() {
  local dataset="$1"
  local input_jsonl="$2"
  local out_sub="$3"
  local pred_out="${out_sub}/$(basename "${dataset}")_predictions.jsonl"
  local score_out="${out_sub}/$(basename "${dataset}")_scores.json"
  "${PYTHON}" "${ROOT}/evaluation/benchmarks/hipho/generate_hipho_predictions.py" \
    --input "${input_jsonl}" \
    --output "${pred_out}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "${SERVED_NAME}" \
    --max-samples "${MAX_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --max-tokens "${MAX_TOKENS}" \
    --concurrency "${GEN_CONCURRENCY}" \
    --resume
  "${VENV}/bin/python" "${ROOT}/evaluation/benchmarks/hipho/score_hipho_predictions.py" \
    --predictions "${pred_out}" \
    --output "${score_out}" \
    --no-use-verifier
}

for label in "${LABELS[@]}"; do
  if [[ -n "${ONLY_LABELS:-}" ]]; then
    case ",${ONLY_LABELS}," in
      *",${label},"*) ;;
      *) continue ;;
    esac
  fi
  OUT_SUB="${MATRIX_DIR}/${label}"
  mkdir -p "${OUT_SUB}"
  MODEL_DIR="${MODEL_DIRS[${label}]}"
  [[ -f "${MODEL_DIR}/config.json" ]] || { echo "[skip] missing ${label} ${MODEL_DIR}" >&2; continue; }
  if [[ "${SKIP_DONE:-1}" == "1" && -f "${OUT_SUB}/hipho_scores.json" ]]; then
    if [[ ! -f "${HELDOUT_JSONL}" || -f "${OUT_SUB}/heldout_scores.json" ]]; then
      echo "[matrix] skip done ${label}"
      continue
    fi
  fi
  echo "[matrix] evaluating ${label} from ${MODEL_DIR}"

  RUN_ID="${label}" MODEL_DIR="${MODEL_DIR}" PORT="${PORT}" CUDA_DEVICE="${CUDA_DEVICE}" \
    MAX_LEN="${MAX_LEN}" GPU_UTIL="${GPU_UTIL}" SERVED_NAME="${SERVED_NAME}" \
    bash "${SCRIPT_DIR}/manage_eval_vllm.sh" stop || true
  sleep 3

  TOKENIZER=""
  if python3 - <<PY
import json,sys
from pathlib import Path
p=Path("${MODEL_DIR}")/"tokenizer_config.json"
if not p.is_file():
    sys.exit(1)
est=json.loads(p.read_text()).get("extra_special_tokens")
sys.exit(0 if isinstance(est, list) else 1)
PY
  then
    TOKENIZER="${BASE_MODEL}"
    echo "[matrix] using base tokenizer for ${label}"
  fi
  START_TS="$(date -Iseconds)"
  RUN_ID="${label}" MODEL_DIR="${MODEL_DIR}" PORT="${PORT}" CUDA_DEVICE="${CUDA_DEVICE}" \
    MAX_LEN="${MAX_LEN}" GPU_UTIL="${GPU_UTIL}" SERVED_NAME="${SERVED_NAME}" \
    VLLM_READY_SECS="${VLLM_READY_SECS:-1800}" TOKENIZER="${TOKENIZER}" \
    bash "${SCRIPT_DIR}/manage_eval_vllm.sh" start

  _eval_one_dataset hipho "${HIPHO_JSONL}" "${OUT_SUB}"
  if [[ -f "${HELDOUT_JSONL}" ]]; then
    _eval_one_dataset heldout "${HELDOUT_JSONL}" "${OUT_SUB}"
  fi

  END_TS="$(date -Iseconds)"
  python3 - <<PY >"${OUT_SUB}/run_meta.json"
import json
print(json.dumps({
  "label": "${label}",
  "model_dir": "${MODEL_DIR}",
  "start": "${START_TS}",
  "end": "${END_TS}",
}, ensure_ascii=False, indent=2))
PY

  RUN_ID="${label}" bash "${SCRIPT_DIR}/manage_eval_vllm.sh" stop || true
  sleep 5
done

"${VENV}/bin/python" "${ROOT}/evaluation/benchmarks/hipho/summarize_hipho_baseline.py" \
  --matrix-dir "${MATRIX_DIR}" \
  --base-label base_8b \
  --output "${MATRIX_DIR}/summary_hipho.json" || true

python3 - <<'PY' "${MATRIX_DIR}" "${MATRIX_DIR}/summary_all.json"
import json, sys
from pathlib import Path
matrix = Path(sys.argv[1])
out_path = Path(sys.argv[2])
entries = []
for label_dir in sorted(matrix.iterdir()):
    if not label_dir.is_dir():
        continue
    row = {"label": label_dir.name}
    for bench in ("hipho", "heldout"):
        score_path = label_dir / f"{bench}_scores.json"
        if score_path.is_file():
            s = json.loads(score_path.read_text(encoding="utf-8"))
            row[f"{bench}_answer_acc"] = s.get("answer_acc")
            row[f"{bench}_n"] = s.get("n_samples")
    if len(row) > 1:
        entries.append(row)
base = next((e for e in entries if e["label"] == "base_8b"), None)
comparisons = []
if base:
    for e in entries:
        if e["label"] == "base_8b":
            continue
        comp = {"label": e["label"]}
        for bench in ("hipho", "heldout"):
            bk = f"{bench}_answer_acc"
            if bk in e and bk in base:
                comp[f"delta_{bk}"] = (e[bk] or 0) - (base[bk] or 0)
        comparisons.append(comp)
payload = {"entries": entries, "comparisons_vs_base": comparisons}
out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

echo "[ok] 8B checkpoint matrix done: ${MATRIX_DIR}"
