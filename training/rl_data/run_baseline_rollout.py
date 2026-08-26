#!/usr/bin/env python3
"""Baseline rollout + reward scoring for offline RL prompt filtering."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _sample_response(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: float) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=8192,
    )
    return resp.choices[0].message.content or ""


def _score_response(
    rm_url: str,
    row: Dict[str, Any],
    response: str,
    timeout: float,
) -> Dict[str, Any]:
    question = (row.get("metadata") or {}).get("question", "")
    payload = {
        "prompt": _messages_to_prompt(row.get("input", [])),
        "response": response,
        "label": row.get("label"),
        "question": question,
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(rm_url.rstrip("/") + "/", json=payload)
        r.raise_for_status()
        return r.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=ROOT / "data/rl/rl_prompts_raw.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/rl/baseline_rollout_scores.jsonl")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8766/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "qwen3-30b-a3b-instruct-2507"))
    parser.add_argument("--rm-url", default=os.environ.get("PHYSICS_RM_URL", "http://127.0.0.1:8770"))
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rm-timeout", type=float, default=600.0)
    args = parser.parse_args()

    rows = _load_jsonl(args.prompts)
    if args.max_prompts:
        rows = rows[: args.max_prompts]

    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict[str, Any]] = []

    def _process_one(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        local = []
        messages = row.get("input", [])
        for i in range(args.n_samples):
            response = _sample_response(client, args.model, messages, args.temperature)
            reward = _score_response(args.rm_url, row, response, args.rm_timeout)
            rec = dict(row)
            rec["rollout_index"] = i
            rec["response"] = response
            rec["reward"] = reward
            if reward.get("acc") and reward.get("n_errors", 99) == 0:
                rec["best_response"] = response
            local.append(rec)
        return local

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process_one, row) for row in rows]
        for fut in as_completed(futures):
            out_rows.extend(fut.result())

    with args.output.open("w", encoding="utf-8") as f:
        for rec in out_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "prompts": len(rows),
                "records": len(out_rows),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
