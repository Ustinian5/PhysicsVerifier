#!/usr/bin/env bash
# Convert PhysicsVerifier RL jsonl -> OpenRLHF-friendly jsonl (label as string).
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
INPUT="${1:-${ROOT}/data/rl/rl_prompts.jsonl}"
OUTPUT="${2:-${ROOT}/data/rl/openrlhf_prompts.jsonl}"

"${PYTHON}" - <<PY
import json
from pathlib import Path

src = Path("${INPUT}")
dst = Path("${OUTPUT}")
dst.parent.mkdir(parents=True, exist_ok=True)
n = 0
with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        label = row.get("label")
        if isinstance(label, list):
            # Prefer first boxed answer string for OpenRLHF label_key
            label = label[0] if label else ""
        out = {
            "input": row.get("input"),
            "label": label if label is not None else "",
            "metadata": row.get("metadata") or {},
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        n += 1
print(f"[ok] wrote {n} rows -> {dst}")
PY
