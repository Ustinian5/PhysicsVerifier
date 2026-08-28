#!/usr/bin/env bash
# Download official SciYu/HiPhO data and freeze a Text-Only jsonl + provenance manifest.
# Never falls back to held-out / internal expansion data.
set -euo pipefail

BENCH_ROOT="${BENCH_ROOT:-/slow_share/jinjianhan/workspace/benchmarks/hipho}"
REPO_DIR="${REPO_DIR:-${BENCH_ROOT}/HiPhO}"
OUT_JSONL="${OUT_JSONL:-${BENCH_ROOT}/hipho_text_only.jsonl}"
MANIFEST="${MANIFEST:-${BENCH_ROOT}/hipho_official_manifest.json}"
INTERNAL150="${INTERNAL150:-${BENCH_ROOT}/internal150_expansion_eval.jsonl}"
HF_REVISION="${HF_REVISION:-8e196c09a71e4e68b75c422defa512473359e0e5}"
HF_MIRROR_ENDPOINT="${HF_MIRROR_ENDPOINT:-https://hf-mirror.com}"
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
VENV="${VENV:-${ROOT}/.venv}"
PYTHON="${PYTHON:-${VENV}/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

mkdir -p "${BENCH_ROOT}"

quarantine_internal() {
  local path="$1"
  [[ -f "${path}" ]] || return 0
  if "${PYTHON}" - "${path}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
internal = 0
n = 0
for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    n += 1
    row = json.loads(line)
    blob = json.dumps(row, ensure_ascii=False)
    if "evaluation_sample_" in blob or "_expansion.json" in blob:
        internal += 1
ok_official = False
try:
    sample = json.loads(next(x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()))
    exam = (sample.get("exam") or (sample.get("metadata") or {}).get("exam") or "")
    marking = sample.get("marking_schemes") or sample.get("marking") or (sample.get("metadata") or {}).get("marking")
    src = str(sample.get("source") or "")
    ok_official = bool(exam) and src == "SciYu/HiPhO"
except Exception:
    ok_official = False
sys.exit(0 if (internal or not ok_official) else 1)
PY
  then
    echo "[setup] quarantining non-official file ${path} -> ${INTERNAL150}"
    if [[ "${path}" != "${INTERNAL150}" ]]; then
      mv -f "${path}" "${INTERNAL150}"
    fi
  fi
}

quarantine_internal "${OUT_JSONL}"

download_hf() {
  local endpoint="${1:-}"
  echo "[setup] downloading HuggingFace SciYu/HiPhO@${HF_REVISION} endpoint=${endpoint:-https://huggingface.co}"
  mkdir -p "${REPO_DIR}"
  HF_ENDPOINT="${endpoint}" "${PYTHON}" - <<PY
import os
endpoint = os.environ.get("HF_ENDPOINT") or ""
if endpoint:
    os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="SciYu/HiPhO",
    repo_type="dataset",
    revision="${HF_REVISION}",
    local_dir="${REPO_DIR}",
    allow_patterns=["data/*.json", "README.md", ".gitattributes"],
)
print("[ok] huggingface snapshot downloaded endpoint=" + (endpoint or "default"))
PY
}

download_hf_mirror_curl() {
  echo "[setup] curling exam JSON from ${HF_MIRROR_ENDPOINT} revision=${HF_REVISION}"
  mkdir -p "${REPO_DIR}/data"
  local names
  names="$("${PYTHON}" - <<PY
import json, urllib.request
url = "${HF_MIRROR_ENDPOINT}/api/datasets/SciYu/HiPhO/tree/${HF_REVISION}/data"
with urllib.request.urlopen(url, timeout=30) as resp:
    rows = json.load(resp)
for row in rows:
    path = row.get("path") or ""
    if path.endswith(".json"):
        print(path.split("/")[-1])
PY
)"
  local name url
  for name in ${names}; do
    url="${HF_MIRROR_ENDPOINT}/datasets/SciYu/HiPhO/resolve/${HF_REVISION}/data/${name}"
    echo "[setup] get ${name}"
    curl -fsSL --retry 3 --retry-delay 2 --max-time 60 "${url}" -o "${REPO_DIR}/data/${name}"
  done
  curl -fsSL --retry 3 --max-time 30 \
    "${HF_MIRROR_ENDPOINT}/datasets/SciYu/HiPhO/resolve/${HF_REVISION}/README.md" \
    -o "${REPO_DIR}/README.md" || true
  [[ -n "$(find "${REPO_DIR}/data" -maxdepth 1 -name '*.json' -print -quit)" ]]
}

has_exam_json() {
  [[ -d "${REPO_DIR}/data" ]] || return 1
  find "${REPO_DIR}/data" -maxdepth 1 -name '*.json' -print -quit | grep -q .
}

if ! has_exam_json; then
  echo "[setup] fetching official SciYu/HiPhO"
  download_hf "${HF_MIRROR_ENDPOINT}" || true
fi
if ! has_exam_json; then
  download_hf_mirror_curl || true
fi
if ! has_exam_json; then
  download_hf "" || true
fi
if ! has_exam_json; then
  timeout 30 git clone --depth 1 https://github.com/SciYu/HiPhO.git "${REPO_DIR}" || true
fi
if ! has_exam_json; then
  echo "[error] official HiPhO download failed (git and HuggingFace); refusing to fake HiPhO with held-out data" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
"${PYTHON}" "${ROOT}/evaluation/benchmarks/hipho/export_official_hipho.py" \
  --repo-dir "${REPO_DIR}" \
  --out-jsonl "${OUT_JSONL}" \
  --manifest "${MANIFEST}" \
  --hf-revision "${HF_REVISION}"

if [[ ! -s "${OUT_JSONL}" || ! -s "${MANIFEST}" ]]; then
  echo "[error] official HiPhO export did not produce jsonl/manifest" >&2
  exit 2
fi

echo "[ok] official HiPhO-TO frozen at ${OUT_JSONL}"
echo "[ok] manifest ${MANIFEST}"
