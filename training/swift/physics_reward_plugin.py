#!/usr/bin/env python3
"""ms-swift GRPO plugin: score completions via PhysicsVerifier reward server."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

try:
    from swift.rewards import AsyncORM, orms
except ImportError:  # pragma: no cover - tests / missing ms-swift
    try:
        from swift.plugin.orm import AsyncORM, orms  # type: ignore
    except ImportError:
        class AsyncORM:  # type: ignore[no-redef]
            pass

        orms = {}  # type: ignore[assignment]


REWARD_URL = os.environ.get("PHYSICS_REWARD_URL", "http://127.0.0.1:8770/get_reward")
TIMEOUT = float(os.environ.get("PHYSICS_REWARD_TIMEOUT", "3600"))


def _as_list(value: Any, n: int) -> List[Any]:
    if value is None:
        return [""] * n
    if isinstance(value, (list, tuple)):
        out = list(value)
        if len(out) >= n:
            return out[:n]
        if len(out) == 1:
            return list(out) * n
        return out + [out[-1] if out else ""] * (n - len(out))
    return [value] * n


def extract_questions(messages: Sequence[Any], n: int) -> List[str]:
    questions: List[str] = []
    if not messages:
        return [""] * n
    rows = list(messages)
    if len(rows) == 1 and n > 1:
        rows = rows * n
    for item in rows[:n]:
        question = ""
        if isinstance(item, list):
            for msg in reversed(item):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    question = str(msg.get("content") or "")
                    break
        elif isinstance(item, dict) and item.get("role") == "user":
            question = str(item.get("content") or "")
        elif isinstance(item, str):
            question = item
        questions.append(question)
    while len(questions) < n:
        questions.append(questions[-1] if questions else "")
    return questions


def build_reward_payload(
    completions: Sequence[str],
    *,
    messages: Sequence[Any] | None = None,
    solution: Any = None,
    question: Any = None,
) -> Dict[str, List[Any]]:
    n = len(completions)
    queries = [str(c or "") for c in completions]
    if question is not None:
        prompts = [str(x or "") for x in _as_list(question, n)]
    else:
        prompts = extract_questions(messages or [], n)
    labels = _as_list(solution, n)
    return {"query": queries, "prompts": prompts, "labels": labels}


class PhysicsVerifierReward(AsyncORM):
    """Call process-paragraph reward server; one batch POST per GRPO group."""

    async def __call__(
        self,
        completions,
        messages=None,
        solution=None,
        question=None,
        **kwargs,
    ) -> List[float]:
        import aiohttp

        payload = build_reward_payload(
            completions,
            messages=messages,
            solution=solution if solution is not None else kwargs.get("solution"),
            question=question if question is not None else kwargs.get("question"),
        )
        n = len(payload["query"])
        try:
            timeout = aiohttp.ClientTimeout(total=TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(REWARD_URL, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        print(f"[physics_reward] HTTP {resp.status}: {body[:300]}", flush=True)
                        return [0.0] * n
                    data = await resp.json()
            rewards = data.get("rewards")
            if not isinstance(rewards, list) or len(rewards) != n:
                print(f"[physics_reward] bad rewards payload: {data}", flush=True)
                return [0.0] * n
            return [float(x) for x in rewards]
        except Exception as exc:
            print(f"[physics_reward] failed: {exc}", flush=True)
            return [0.0] * n


orms["physics_verifier"] = PhysicsVerifierReward
