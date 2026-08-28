from __future__ import annotations

import asyncio
import json
import unittest

from training.reward_server.llm_step_judge import (
    LLMStepJudge,
    LLMStepJudgeError,
    build_messages,
    extract_json_object,
    thinking_kwargs,
    validate_group_response,
)
from training.swift.monitor_llm_step_reward import evaluate


class _FakeHTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class LLMStepJudgeTests(unittest.TestCase):
    def test_prompt_does_not_include_gold(self) -> None:
        messages = build_messages("What is a?", ["solution A", "solution B"])
        blob = json.dumps(messages)
        self.assertNotIn("gold", blob.casefold())
        self.assertNotIn("reference answer", blob.casefold())
        self.assertNotIn("solution A" if False else "ground_truth", blob)
        self.assertIn("What is a?", blob)
        self.assertIn("solution A", blob)

    def test_json_repair_and_id_reorder(self) -> None:
        text = 'noise {"candidates":[{"id":"c1","score":3},{"id":"c0","score":8}]}'
        data = extract_json_object(text)
        ordered = validate_group_response(data, ["c0", "c1"])
        self.assertEqual([x["id"] for x in ordered], ["c0", "c1"])
        self.assertAlmostEqual(ordered[0]["score"], 0.8)
        self.assertAlmostEqual(ordered[1]["score"], 0.3)

    def test_out_of_range_and_nan_rejected(self) -> None:
        with self.assertRaises(LLMStepJudgeError):
            validate_group_response({"candidates": [{"id": "c0", "score": 11}]}, ["c0"])
        with self.assertRaises(LLMStepJudgeError):
            validate_group_response({"candidates": [{"id": "c0", "score": float("nan")}]}, ["c0"])
        with self.assertRaises(LLMStepJudgeError):
            validate_group_response({"candidates": [{"id": "c0", "score": 1}, {"id": "c0", "score": 2}]}, ["c0"])

    def test_fatal_and_answer_only_caps(self) -> None:
        data = {
            "candidates": [
                {"id": "c0", "score": 9, "fatal_error": True, "answer_only": False},
                {"id": "c1", "score": 8, "fatal_error": False, "answer_only": True},
            ]
        }
        out = validate_group_response(data, ["c0", "c1"])
        self.assertLessEqual(out[0]["score"], 0.4)
        self.assertLessEqual(out[1]["score"], 0.2)

    def test_deepseek_thinking_kwargs(self) -> None:
        kw = thinking_kwargs("deepseek-v4-flash")
        self.assertEqual(kw["chat_template_kwargs"]["thinking"], False)
        qwen = thinking_kwargs("qwen3-8b")
        self.assertEqual(qwen["chat_template_kwargs"]["enable_thinking"], False)

    def test_retry_then_fail_closed(self) -> None:
        calls = {"n": 0}

        def boom(messages, extra_user=None):
            calls["n"] += 1
            raise _FakeHTTPError(429)

        judge = LLMStepJudge(complete_fn=boom, max_retries=3, sleep_fn=lambda _s: None)
        with self.assertRaises(LLMStepJudgeError):
            judge.score_group("q", ["a", "b"])
        self.assertEqual(calls["n"], 4)
        self.assertGreaterEqual(judge.failures, 1)

    def test_parse_retry_uses_correction_prompt(self) -> None:
        seen = []

        def complete(messages, extra_user=None):
            seen.append((messages, extra_user))
            blob = json.dumps(messages, ensure_ascii=False)
            if "上一响应不是符合 schema" in blob:
                return json.dumps({"candidates": [{"id": "c0", "score": 4.0}]})
            return "not json"

        judge = LLMStepJudge(complete_fn=complete, sleep_fn=lambda _s: None)
        out = judge.score_group("q", ["only"])
        self.assertEqual(len(out), 1)
        self.assertGreaterEqual(len(seen), 2)
        retry_messages = seen[1][0]
        roles = [m.get("role") for m in retry_messages]
        self.assertIn("assistant", roles)

    def test_missing_ids_retry_then_succeed(self) -> None:
        calls = {"n": 0}

        def complete(messages, extra_user=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return json.dumps({"candidates": [{"id": "c0", "score": 6.0}]})
            return json.dumps(
                {
                    "candidates": [
                        {"id": "c0", "score": 6.0},
                        {"id": "c1", "score": 2.0},
                    ]
                }
            )

        judge = LLMStepJudge(complete_fn=complete, sleep_fn=lambda _s: None)
        out = judge.score_group("q", ["a", "b"])
        self.assertEqual([x["id"] for x in out], ["c0", "c1"])
        self.assertEqual(calls["n"], 2)
        self.assertEqual(judge.parse_retries, 1)

    def test_concurrent_ascore_group_overlaps(self) -> None:
        inflight = {"n": 0, "max": 0}

        async def complete(messages, extra_user=None):
            inflight["n"] += 1
            inflight["max"] = max(inflight["max"], inflight["n"])
            await asyncio.sleep(0.08)
            inflight["n"] -= 1
            return json.dumps({"candidates": [{"id": "c0", "score": 5.0}]})

        judge = LLMStepJudge(async_complete_fn=complete, sleep_fn=lambda _s: None, concurrency=32)

        async def run():
            return await asyncio.gather(*[judge.ascore_group(f"q{i}", ["a"]) for i in range(8)])

        out = asyncio.run(run())
        self.assertEqual(len(out), 8)
        self.assertGreaterEqual(inflight["max"], 6)

    def test_truncated_json_without_closing_braces(self) -> None:
        text = '{"candidates":[{"id":"c0","score":8.0},{"id":"c1","score":3.5'
        data = extract_json_object(text)
        ordered = validate_group_response(data, ["c0", "c1"])
        self.assertAlmostEqual(ordered[0]["score"], 0.8)
        self.assertAlmostEqual(ordered[1]["score"], 0.35)

    def test_markdown_fence_and_think_tags(self) -> None:
        text = (
            "<think>ignore me</think>\n"
            "```json\n"
            '{"candidates":[{"id":"c0","score":6}]}\n'
            "```\n"
        )
        data = extract_json_object(text)
        ordered = validate_group_response(data, ["c0"])
        self.assertAlmostEqual(ordered[0]["score"], 0.6)

    def test_score_map_object(self) -> None:
        data = extract_json_object('{"c0": 4, "c1": 9}')
        ordered = validate_group_response(data, ["c0", "c1"])
        self.assertAlmostEqual(ordered[0]["score"], 0.4)
        self.assertAlmostEqual(ordered[1]["score"], 0.9)

    def test_empty_then_success(self) -> None:
        calls = {"n": 0}

        def complete(messages, extra_user=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return "   "
            return json.dumps({"candidates": [{"id": "c0", "score": 5.0}]})

        judge = LLMStepJudge(complete_fn=complete, sleep_fn=lambda _s: None)
        out = judge.score_group("q", ["only"])
        self.assertEqual(len(out), 1)
        self.assertEqual(calls["n"], 2)
        self.assertGreaterEqual(judge.retries, 1)


class MonitorSoftFailTests(unittest.TestCase):
    def test_api_fail_rate_warns_but_does_not_stop(self) -> None:
        report = evaluate(
            [{"llm_step_failures": 1.0, "llm_step_api_calls": 8.0, "physics_llm_step_sat01_rate": 0.0}],
            [{"loss": 0.2, "reward": 0.3}],
        )
        self.assertFalse(report["stop"])
        self.assertIn("api_or_schema_fail_rate", report["warnings"])


if __name__ == "__main__":
    unittest.main()
