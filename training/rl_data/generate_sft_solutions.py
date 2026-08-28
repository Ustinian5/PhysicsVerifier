#!/usr/bin/env python3
"""Generate SFT solutions via local 30B (or API fallback) with answer rejection sampling.

Local pass: K samples per prompt, keep the shortest boxed answer that grades as correct.
Remaining misses are written to --unsolved for an optional API fill pass.
Held-out sample_ids are never used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from training.compat.math_grading import grade_answer_verl
from training.rl_data.audit_eval_leakage import load_exclusion, row_excluded

SFT_SYSTEM = (
    "You are an expert physics competition solver. "
    "Write a complete, rigorous step-by-step solution. "
    "Put the final answer in \\boxed{}."
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _heldout_ids(path: Optional[Path]) -> set[str]:
    if path is None or not path.is_file():
        return set()
    ids: set[str] = set()
    for row in _load_jsonl(path):
        sid = str(row.get("sample_id") or (row.get("metadata") or {}).get("sample_id") or "")
        if sid:
            ids.add(sid)
    return ids


def _user_question(row: Dict[str, Any]) -> str:
    q = row.get("question")
    if q:
        return str(q)
    for msg in reversed(row.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def _done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(r.get("sample_id") or "") for r in _load_jsonl(path) if r.get("sample_id")}


def _chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return resp.choices[0].message.content or ""


def pick_shortest_correct(candidates: List[str], gold: str) -> Optional[str]:
    correct = [c for c in candidates if c.strip() and grade_answer_verl(c, gold)]
    if not correct:
        return None
    return min(correct, key=lambda x: (len(x), x))


def _make_sft_row(src: Dict[str, Any], solution_text: str) -> Dict[str, Any]:
    messages = list(src.get("messages") or [])
    if not messages:
        messages = [
            {"role": "system", "content": SFT_SYSTEM},
            {"role": "user", "content": _user_question(src)},
        ]
    out_messages = [m for m in messages if m.get("role") != "assistant"]
    out_messages.append({"role": "assistant", "content": solution_text})
    return {
        "messages": out_messages,
        "solution": src.get("solution") or "",
        "question": src.get("question") or _user_question(src),
        "sample_id": src.get("sample_id"),
        "source": src.get("source"),
        "generator": src.get("generator", "local"),
    }


def generate_for_row(
    client: OpenAI,
    model: str,
    row: Dict[str, Any],
    k: int,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> Optional[str]:
    question = _user_question(row)
    gold = str(row.get("solution") or "")
    messages = [
        {"role": "system", "content": SFT_SYSTEM},
        {"role": "user", "content": question},
    ]
    cands: List[str] = []
    sid = row.get("sample_id")
    for i in range(k):
        try:
            print(f"[sft-gen] sample {sid} try {i+1}/{k}", flush=True)
            cands.append(_chat(client, model, messages, temperature, max_tokens, timeout))
        except Exception as exc:  # noqa: BLE001
            print(f"[sft-gen] sample {sid} failed try {i+1}: {exc}", flush=True)
    return pick_shortest_correct(cands, gold)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=ROOT / "data/rl/swift_prompts.jsonl")
    p.add_argument("--heldout", type=Path, default=ROOT / "data/rl/heldout_eval.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/rl/sft_solutions.jsonl")
    p.add_argument("--unsolved", type=Path, default=ROOT / "data/rl/sft_unsolved.jsonl")
    p.add_argument("--report", type=Path, default=ROOT / "data/rl/sft_gen_report.json")
    p.add_argument("--base-url", default=os.environ.get("SFT_GEN_BASE_URL", "http://127.0.0.1:8780/v1"))
    p.add_argument("--model", default=os.environ.get("SFT_GEN_MODEL", "qwen3-30b-a3b"))
    p.add_argument("--api-base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument(
        "--api-model",
        default=os.environ.get("SFT_API_MODEL", "qwen3-30b-a3b-instruct-2507"),
    )
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--api-k", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--max-prompts", type=int, default=0)
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--api-only", action="store_true")
    args = p.parse_args()

    heldout = _heldout_ids(args.heldout)
    excl_ids, excl_hashes = load_exclusion(ROOT / "data/rl/train_manifest.json")
    rows = []
    for r in _load_jsonl(args.prompts):
        sid = str(r.get("sample_id") or "")
        if sid in heldout:
            continue
        if row_excluded(r, excl_ids, excl_hashes):
            continue
        rows.append(r)
    if args.max_prompts:
        rows = rows[: args.max_prompts]
    done = _done_ids(args.output)
    todo = [r for r in rows if str(r.get("sample_id") or "") not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    solved = len(done)
    failed: List[Dict[str, Any]] = []
    local_client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    api_client = None
    if args.api_base_url and args.api_key and args.api_model and not args.local_only:
        api_client = OpenAI(base_url=args.api_base_url, api_key=args.api_key, timeout=args.timeout)
    if args.api_only and api_client is None:
        raise SystemExit("API fill requires OPENAI_BASE_URL, OPENAI_API_KEY, and a model name")

    def work(row: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[str], str]:
        stage = "skip"
        text = None
        if not args.api_only:
            stage = "local"
            text = generate_for_row(
                local_client, args.model, row, args.k, args.temperature, args.max_tokens, args.timeout
            )
        if text is None and api_client is not None:
            stage = "api"
            text = generate_for_row(
                api_client, args.api_model, row, args.api_k, args.temperature, args.max_tokens, args.timeout
            )
        return row, text, stage

    with args.output.open("a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futs = [pool.submit(work, row) for row in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                row, text, stage = fut.result()
                if text is None:
                    failed.append(row)
                else:
                    rec = _make_sft_row(row, text)
                    rec["generator"] = stage
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    solved += 1
                if i % 20 == 0:
                    print(f"[sft-gen] done {i}/{len(todo)} solved_total={solved}", flush=True)

    with args.unsolved.open("w", encoding="utf-8") as uf:
        for row in failed:
            uf.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "n_prompts": len(rows),
        "n_heldout_excluded": len(heldout),
        "n_already_done": len(done),
        "n_solved": solved,
        "n_unsolved": len(failed),
        "coverage": solved / max(len(rows), 1),
        "output": str(args.output),
        "unsolved": str(args.unsolved),
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
