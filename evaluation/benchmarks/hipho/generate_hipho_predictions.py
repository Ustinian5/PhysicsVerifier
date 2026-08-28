#!/usr/bin/env python3
"""Generate model predictions for HiPhO text-only subset."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[3]
SYSTEM = (
    "You are an expert physics olympiad solver. "
    "Provide rigorous step-by-step reasoning and final answers in \\boxed{}."
)


def _question_from_row(row: Dict[str, Any]) -> str:
    question = row.get("question") or row.get("problem")
    if question:
        return str(question)
    inp = row.get("input")
    if isinstance(inp, list):
        parts = []
        for msg in inp:
            if isinstance(msg, dict) and msg.get("content"):
                parts.append(str(msg["content"]))
        if parts:
            return "\n".join(parts)
    if isinstance(inp, str) and inp.strip():
        return inp
    meta = row.get("metadata") or {}
    if meta.get("question"):
        return str(meta["question"])
    return ""


def _messages_from_row(row: Dict[str, Any]) -> List[Dict[str, str]]:
    inp = row.get("input")
    if isinstance(inp, list) and inp:
        messages = []
        for msg in inp:
            if isinstance(msg, dict) and msg.get("content"):
                messages.append({"role": str(msg.get("role", "user")), "content": str(msg["content"])})
        if messages:
            return messages
    question = _question_from_row(row)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _strip_gold(row: Dict[str, Any]) -> Dict[str, Any]:
    drop = {
        "answer",
        "answers",
        "label",
        "labels",
        "marking",
        "marking_scheme",
        "marking_schemes",
        "ground_truth",
    }
    return {k: v for k, v in row.items() if k not in drop}


def _predict_one(
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    row: Dict[str, Any],
    retries: int,
    enable_thinking: bool = False,
) -> str:
    messages = _messages_from_row(row)
    last_err: Exception | None = None
    tokens = max_tokens
    extra_body = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
    for attempt in range(max(1, retries)):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=tokens,
                timeout=timeout,
                extra_body=extra_body,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            msg = str(exc)
            if "maximum context length" in msg or "BadRequestError" in msg:
                tokens = max(256, tokens // 2)
                continue
    raise RuntimeError(f"prediction failed after {retries} tries: {last_err}") from last_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8766/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "qwen3-30b-a3b-instruct-2507"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--start-index", type=int, default=0, help="Skip first N input rows (resume)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output line count")
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("HIPHO_GEN_CONCURRENCY", "16")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("HIPHO_GEN_TIMEOUT", "600")))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Keep Qwen3 <think> tokens. Default off so boxed answers fit in max_tokens.",
    )
    parser.add_argument(
        "--strip-gold",
        action="store_true",
        help="Write question/prediction only; never copy answers or marking schemes.",
    )
    args = parser.parse_args()

    rows = _load_jsonl(args.input)
    if args.max_samples:
        rows = rows[: args.max_samples]

    start_idx = max(args.start_index, 0)
    if args.resume and args.output.is_file():
        with args.output.open("r", encoding="utf-8") as existing:
            start_idx = max(start_idx, sum(1 for line in existing if line.strip()))

    remaining = rows[start_idx:]
    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        timeout=args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if start_idx > 0 and args.output.is_file() else "w"
    concurrency = max(1, args.concurrency)

    def work(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, str]:
        idx, row = item
        pred = _predict_one(
            client,
            args.model,
            args.temperature,
            args.max_tokens,
            args.timeout,
            row,
            args.retries,
            args.enable_thinking,
        )
        return idx, pred

    with args.output.open(mode, encoding="utf-8") as out:
        if concurrency == 1:
            for row in remaining:
                pred = _predict_one(
                    client,
                    args.model,
                    args.temperature,
                    args.max_tokens,
                    args.timeout,
                    row,
                    args.retries,
                    args.enable_thinking,
                )
                rec = dict(row)
                rec["prediction"] = pred
                if args.strip_gold:
                    rec = _strip_gold(rec)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
        else:
            indexed = list(enumerate(remaining))
            results: Dict[int, str] = {}
            next_write = 0
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futs = [pool.submit(work, item) for item in indexed]
                for fut in as_completed(futs):
                    idx, pred = fut.result()
                    results[idx] = pred
                    while next_write in results:
                        rec = dict(remaining[next_write])
                        rec["prediction"] = results.pop(next_write)
                        if args.strip_gold:
                            rec = _strip_gold(rec)
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out.flush()
                        next_write += 1

    print(
        json.dumps(
            {
                "count": len(remaining),
                "start_index": start_idx,
                "output": str(args.output),
                "mode": mode,
                "concurrency": concurrency,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
