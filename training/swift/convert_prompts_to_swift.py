#!/usr/bin/env python3
"""Convert OpenRLHF prompt jsonl to ms-swift GRPO format.

OpenRLHF rows: {input: [messages], label, metadata}
Swift GRPO rows: {messages: [...], solution: label, ...extra columns}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _as_messages(raw: Any) -> List[Dict[str, str]]:
    if isinstance(raw, list) and raw:
        out: List[Dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user")
            content = item.get("content")
            if content is None:
                continue
            out.append({"role": role, "content": str(content)})
        if out:
            return out
    text = str(raw or "").strip()
    if not text:
        return []
    return [{"role": "user", "content": text}]


def convert_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    messages = _as_messages(row.get("input"))
    if not messages:
        return None
    solution = row.get("label")
    if solution is None:
        solution = ""
    out: Dict[str, Any] = {
        "messages": messages,
        "solution": solution,
    }
    meta = row.get("metadata")
    if isinstance(meta, dict):
        for key in ("question", "sample_id", "source", "rm_type"):
            if key in meta and meta[key] is not None:
                out[key] = meta[key]
    if "question" not in out:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                out["question"] = msg.get("content") or ""
                break
    return out


def convert_file(src: Path, dst: Path) -> int:
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            converted = convert_row(row)
            if converted is None:
                continue
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        default="/home/jinjianhan/PhysicsVerifier/data/rl/openrlhf_prompts.jsonl",
    )
    p.add_argument(
        "--dst",
        default="/home/jinjianhan/PhysicsVerifier/data/rl/swift_prompts.jsonl",
    )
    p.add_argument("--heldout-src", default="/home/jinjianhan/PhysicsVerifier/data/rl/openrlhf_heldout.jsonl")
    p.add_argument("--heldout-dst", default="/home/jinjianhan/PhysicsVerifier/data/rl/swift_heldout.jsonl")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    n = convert_file(src, dst)
    print(f"[ok] converted {n} rows {src} -> {dst}")

    held_src = Path(args.heldout_src)
    if held_src.is_file() and held_src.stat().st_size > 0:
        hn = convert_file(held_src, Path(args.heldout_dst))
        print(f"[ok] converted {hn} heldout rows {held_src} -> {args.heldout_dst}")
    else:
        print(f"[skip] heldout missing {held_src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
