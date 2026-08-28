#!/usr/bin/env python3
"""ms-swift GRPO plugin: DeepSeek llm_step_score rewards. Labels/gold never leave this process."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Sequence

try:
    from swift.rewards import AsyncORM, orms
except ImportError:  # pragma: no cover
    try:
        from swift.plugin.orm import AsyncORM, orms  # type: ignore
    except ImportError:
        class AsyncORM:  # type: ignore[no-redef]
            pass

        orms = {}  # type: ignore[assignment]


REWARD_URL = os.environ.get("PHYSICS_REWARD_URL", "http://127.0.0.1:8770/get_reward")
TIMEOUT = float(os.environ.get("PHYSICS_REWARD_TIMEOUT", "3600"))
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


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
    question: Any = None,
    **kwargs: Any,
) -> Dict[str, List[Any]]:
    n = len(completions)
    queries = [str(c or "") for c in completions]
    if question is not None:
        prompts = [str(x or "") for x in _as_list(question, n)]
    else:
        prompts = extract_questions(messages or [], n)
    # Gold labels are intentionally dropped. kwargs may contain solution; ignore it.
    _ = kwargs.get("solution")
    return {"query": queries, "prompts": prompts, "labels": [""] * n}


def group_indices_by_prompt(prompts: Sequence[str]) -> List[List[int]]:
    groups: Dict[str, List[int]] = {}
    order: List[str] = []
    for idx, key in enumerate(prompts):
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(idx)
    return [groups[key] for key in order]


def slice_payload(payload: Dict[str, List[Any]], idxs: Sequence[int]) -> Dict[str, List[Any]]:
    return {key: [payload[key][i] for i in idxs] for key in ("query", "prompts", "labels")}


class LLMStepVerifierReward(AsyncORM):
    """Remote LLM step judge. Transport/5xx are retried; persistent errors abort training."""

    async def _post_payload(self, session: Any, url: str, payload: Dict[str, List[Any]], retries: int) -> List[float]:
        n = len(payload["query"])
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                async with session.post(url, json=payload) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        err = RuntimeError(f"llm_step_verifier HTTP {resp.status}: {body[:500]}")
                        if resp.status in RETRYABLE_STATUS and attempt < retries:
                            last_err = err
                            await asyncio.sleep(min(30.0, 2 ** attempt))
                            continue
                        raise err
                    try:
                        data = json.loads(body)
                    except Exception as exc:  # noqa: BLE001
                        if attempt < retries:
                            last_err = exc
                            await asyncio.sleep(min(30.0, 2 ** attempt))
                            continue
                        raise RuntimeError(f"llm_step_verifier invalid JSON: {body[:500]}") from exc
                rewards = data.get("rewards") if isinstance(data, dict) else None
                if not isinstance(rewards, list) or len(rewards) != n:
                    if attempt < retries:
                        await asyncio.sleep(min(30.0, 2 ** attempt))
                        continue
                    raise RuntimeError(f"llm_step_verifier bad rewards payload: {data!r}"[:500])
                return [float(x) for x in rewards]
            except RuntimeError:
                raise
            except Exception as exc:
                import aiohttp

                if not isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError)):
                    raise
                last_err = exc
                if attempt < retries:
                    await asyncio.sleep(min(30.0, 2 ** attempt))
                    continue
                raise RuntimeError(f"llm_step_verifier transport error: {exc}") from exc
        raise RuntimeError(f"llm_step_verifier failed after retries: {last_err}")

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
            question=question if question is not None else kwargs.get("question"),
            solution=solution,
        )
        n = len(payload["query"])
        timeout = aiohttp.ClientTimeout(total=float(os.environ.get("PHYSICS_REWARD_TIMEOUT", str(TIMEOUT))))
        retries = int(os.environ.get("PHYSICS_REWARD_HTTP_RETRIES", "5"))
        url = os.environ.get("PHYSICS_REWARD_URL", REWARD_URL)
        groups = group_indices_by_prompt(payload["prompts"])
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if len(groups) <= 1:
                return await self._post_payload(session, url, payload, retries)
            scored = await asyncio.gather(
                *[self._post_payload(session, url, slice_payload(payload, idxs), retries) for idxs in groups]
            )
        out = [0.0] * n
        for idxs, rewards in zip(groups, scored):
            for i, reward in zip(idxs, rewards):
                out[i] = reward
        return out


orms["llm_step_verifier"] = LLMStepVerifierReward
