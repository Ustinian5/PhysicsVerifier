from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

from training.swift.llm_step_reward_plugin import build_reward_payload, LLMStepVerifierReward


class LLMStepPluginTests(unittest.TestCase):
    def test_solution_never_enters_payload(self) -> None:
        payload = build_reward_payload(
            ["ans-a", "ans-b"],
            messages=[[{"role": "user", "content": "q1"}]],
            solution=r"\boxed{GOLD}",
        )
        blob = str(payload)
        self.assertNotIn("GOLD", blob)
        self.assertEqual(payload["labels"], ["", ""])
        self.assertEqual(payload["query"], ["ans-a", "ans-b"])
        self.assertEqual(payload["prompts"], ["q1", "q1"])

    def test_http_error_aborts_instead_of_zeros(self) -> None:
        class _Resp:
            status = 502

            async def text(self):
                return "judge down"

            async def json(self, content_type=None):
                return {}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _Session:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def post(self, *a, **k):
                return _Resp()

        plugin = LLMStepVerifierReward()
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            self.skipTest("aiohttp not installed")
        with patch("aiohttp.ClientSession", _Session), patch("aiohttp.ClientTimeout", lambda **k: None), patch(
            "training.swift.llm_step_reward_plugin.asyncio.sleep", new_callable=AsyncMock
        ):
            with self.assertRaises(RuntimeError):
                asyncio.run(plugin(["a", "b"], messages=[[{"role": "user", "content": "q"}]]))

    def test_http_502_then_success(self) -> None:
        class _Resp:
            def __init__(self, status, body):
                self.status = status
                self._body = body

            async def text(self):
                return self._body

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _Session:
            calls = {"n": 0}

            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def post(self, *a, **k):
                _Session.calls["n"] += 1
                if _Session.calls["n"] == 1:
                    return _Resp(502, '{"detail":"judge failed after retries: response is not a JSON object"}')
                return _Resp(200, '{"rewards":[0.4,0.6]}')

        plugin = LLMStepVerifierReward()
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            self.skipTest("aiohttp not installed")
        _Session.calls["n"] = 0
        with patch("aiohttp.ClientSession", _Session), patch("aiohttp.ClientTimeout", lambda **k: None), patch(
            "training.swift.llm_step_reward_plugin.asyncio.sleep", new_callable=AsyncMock
        ):
            out = asyncio.run(plugin(["a", "b"], messages=[[{"role": "user", "content": "q"}]]))
        self.assertEqual(out, [0.4, 0.6])
        self.assertEqual(_Session.calls["n"], 2)

    def test_multi_prompt_posts_overlap(self) -> None:
        inflight = {"n": 0, "max": 0}

        class _Resp:
            status = 200

            def __init__(self, n: int):
                self._n = n

            async def text(self):
                await asyncio.sleep(0.05)
                inflight["n"] -= 1
                return json.dumps({"rewards": [0.1] * self._n})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _Session:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def post(self, url, json=None, **k):
                n = len((json or {}).get("query") or [])
                inflight["n"] += 1
                inflight["max"] = max(inflight["max"], inflight["n"])
                return _Resp(n)

        plugin = LLMStepVerifierReward()
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            self.skipTest("aiohttp not installed")
        messages = [
            [{"role": "user", "content": "q0"}],
            [{"role": "user", "content": "q0"}],
            [{"role": "user", "content": "q1"}],
            [{"role": "user", "content": "q1"}],
        ]
        with patch("aiohttp.ClientSession", _Session), patch("aiohttp.ClientTimeout", lambda **k: None):
            out = asyncio.run(plugin(["a", "b", "c", "d"], messages=messages))
        self.assertEqual(out, [0.1, 0.1, 0.1, 0.1])
        self.assertGreaterEqual(inflight["max"], 2)


if __name__ == "__main__":
    unittest.main()
