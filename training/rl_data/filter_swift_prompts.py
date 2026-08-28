#!/usr/bin/env python3
"""Drop RL prompts whose chat prompt exceeds a token budget (avoids max_model_len crash)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def prompt_text(row: dict) -> str:
    parts = []
    for msg in row.get("messages") or []:
        parts.append(str(msg.get("content") or ""))
    return "\n".join(parts) or str(row.get("question") or "")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("/home/jinjianhan/PhysicsVerifier/data/rl/swift_prompts.jsonl"))
    p.add_argument("--dst", type=Path, default=Path("/home/jinjianhan/PhysicsVerifier/data/rl/swift_prompts_max2048.jsonl"))
    p.add_argument("--tokenizer", default="/slow_share/jinjianhan/models/Qwen3-8B")
    p.add_argument("--max-tokens", type=int, default=2048)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    kept = 0
    dropped = 0
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with args.src.open("r", encoding="utf-8") as fin, args.dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n = len(tok.encode(prompt_text(row), add_special_tokens=True))
            if n > args.max_tokens:
                dropped += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(json.dumps({"kept": kept, "dropped": dropped, "max_tokens": args.max_tokens, "dst": str(args.dst)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
