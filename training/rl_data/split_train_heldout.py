#!/usr/bin/env python3
"""Split raw RL prompts into train pool and held-out eval set."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--heldout-output", type=Path, required=True)
    parser.add_argument("--heldout-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = _load_jsonl(args.input)
    random.seed(args.seed)
    random.shuffle(rows)
    heldout = rows[: args.heldout_size]
    train = rows[args.heldout_size :]
    _write_jsonl(args.train_output, train)
    _write_jsonl(args.heldout_output, heldout)
    print(
        json.dumps(
            {
                "input": len(rows),
                "train": len(train),
                "heldout": len(heldout),
                "train_output": str(args.train_output),
                "heldout_output": str(args.heldout_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
