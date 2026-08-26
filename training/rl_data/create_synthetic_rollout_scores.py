#!/usr/bin/env python3
"""Create synthetic rollout scores for offline filter smoke testing."""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/rl/_smoke_prompts.jsonl"
OUT = ROOT / "data/rl/baseline_rollout_scores_smoke.jsonl"


def main() -> None:
    rows = []
    with SRC.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    random.seed(42)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows[:50]):
            # Craft mixed pass rates per prompt group
            pass_pattern = [True, False, True, False] if idx % 2 == 0 else [True, True, False, False]
            for i, acc in enumerate(pass_pattern):
                n_errors = 0 if not acc else (1 if i % 2 == 0 else 0)
                score = (1.0 if acc else 0.0) - 0.3 * min(n_errors, 3) / 3
                rec = dict(row)
                rec["rollout_index"] = i
                rec["response"] = f"synthetic response {i}"
                rec["reward"] = {
                    "score": score,
                    "acc": acc,
                    "n_errors": n_errors,
                }
                if acc and n_errors == 0:
                    rec["best_response"] = rec["response"]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "records": 20 * 4}, ensure_ascii=False))


if __name__ == "__main__":
    main()
