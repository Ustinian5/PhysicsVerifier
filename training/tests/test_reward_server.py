from __future__ import annotations

import asyncio
import unittest

from fastapi import HTTPException

from training.reward_server import physics_reward_server as server


class _FakeVerifier:
    def __init__(self, result=None) -> None:
        self.calls = 0
        self.result = result

    def verify(self, sample):
        self.calls += 1
        if self.result is not None:
            return self.result
        return {
            "checker_status": "valid_with_diagnostics",
            "checker_failures": [],
            "diagnostics": [
                {"severity": "error", "rule": "test_rule", "message": "test error"}
            ]
        }


class RewardServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_mode = server.REWARD_MODE
        self.original_get_verifier = server._get_verifier
        self.original_append_metrics = server._append_metrics
        self.original_failure_policy = server.VERIFIER_FAILURE_POLICY
        self.original_sample_rate = server.VERIFIER_SAMPLE_RATE
        self.original_semaphore = server._semaphore
        server._append_metrics = lambda record: None
        server._semaphore = None

    def tearDown(self) -> None:
        server.REWARD_MODE = self.original_mode
        server._get_verifier = self.original_get_verifier
        server._append_metrics = self.original_append_metrics
        server.VERIFIER_FAILURE_POLICY = self.original_failure_policy
        server.VERIFIER_SAMPLE_RATE = self.original_sample_rate
        server._semaphore = self.original_semaphore

    def test_wrong_answer_skips_verifier(self) -> None:
        fake = _FakeVerifier()
        server.REWARD_MODE = "answer_low_verifier"
        server._get_verifier = lambda: fake
        result = asyncio.run(
            server.score_one(
                server.ScoreRequest(prompt="question", response=r"\boxed{2}", label="1")
            )
        )
        self.assertFalse(result["acc"])
        self.assertEqual(result["verifier_mode"], "skipped")
        self.assertEqual(fake.calls, 0)

    def test_correct_answer_calls_verifier(self) -> None:
        fake = _FakeVerifier()
        server.REWARD_MODE = "answer_low_verifier"
        server._get_verifier = lambda: fake
        result = asyncio.run(
            server.score_one(
                server.ScoreRequest(prompt="question", response=r"\boxed{1}", label="1")
            )
        )
        self.assertTrue(result["acc"])
        self.assertEqual(result["verifier_mode"], "full")
        self.assertEqual(result["n_errors"], 1)
        self.assertEqual(fake.calls, 1)

    def test_semantic_retrieval_failure_is_not_scored_as_clean(self) -> None:
        fake = _FakeVerifier(
            {
                "selection_strategy": "semantic_error",
                "semantic_selection_error": "judge timeout",
                "checker_status": "complete_no_rules",
                "checker_failures": [],
                "diagnostics": [],
            }
        )
        server.REWARD_MODE = "answer_low_verifier"
        server.VERIFIER_FAILURE_POLICY = "raise"
        server._get_verifier = lambda: fake
        with self.assertRaises(server.VerifierExecutionError):
            asyncio.run(
                server.score_one(
                    server.ScoreRequest(
                        prompt="question", response=r"\boxed{1}", label="1"
                    )
                )
            )

    def test_checker_partial_failure_is_not_scored_as_clean(self) -> None:
        fake = _FakeVerifier(
            {
                "semantic_selection_error": "",
                "checker_status": "partial_failure",
                "checker_failures": [{"rule_id": "r1", "status": "transport_failure"}],
                "diagnostics": [],
            }
        )
        server.REWARD_MODE = "answer_low_verifier"
        server.VERIFIER_FAILURE_POLICY = "raise"
        server._get_verifier = lambda: fake
        with self.assertRaises(server.VerifierExecutionError):
            asyncio.run(
                server.score_one(
                    server.ScoreRequest(
                        prompt="question", response=r"\boxed{1}", label="1"
                    )
                )
            )

    def test_explicit_zero_reward_policy_neutralizes_failed_check(self) -> None:
        fake = _FakeVerifier(
            {
                "semantic_selection_error": "",
                "checker_status": "failed",
                "checker_failures": [{"rule_id": "r1"}],
                "diagnostics": [],
            }
        )
        server.REWARD_MODE = "answer_low_verifier"
        server.VERIFIER_FAILURE_POLICY = "zero_reward"
        server._get_verifier = lambda: fake
        result = asyncio.run(
            server.score_one(
                server.ScoreRequest(prompt="question", response=r"\boxed{1}", label="1")
            )
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["score_noxverify"], 0.0)
        self.assertEqual(result["verifier_mode"], "failed")
        self.assertTrue(result["reward_components"]["verifier_failed"])

    def test_valid_empty_check_remains_successful(self) -> None:
        fake = _FakeVerifier(
            {
                "semantic_selection_error": "",
                "selection_strategy": "semantic_tree_empty",
                "checker_status": "complete_no_rules",
                "checker_failures": [],
                "diagnostics": [],
            }
        )
        server.REWARD_MODE = "answer_low_verifier"
        server._get_verifier = lambda: fake
        result = asyncio.run(
            server.score_one(
                server.ScoreRequest(prompt="question", response=r"\boxed{1}", label="1")
            )
        )
        self.assertEqual(result["verifier_mode"], "full")
        self.assertEqual(result["n_errors"], 0)

    def test_query_response_is_removed_only_when_prompt_is_prefix(self) -> None:
        self.assertEqual(server._response_from_query("promptanswer", "prompt"), "answer")
        with self.assertRaises(ValueError):
            server._response_from_query("answer mentions prompt", "prompt")

    def test_openrlhf_rejects_misaligned_payload(self) -> None:
        request = server.OpenRLHFRewardRequest(
            query=["q1", "q2"], prompts=["p1"], labels=["1", "2"]
        )
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(server.openrlhf_get_reward(request))
        self.assertEqual(ctx.exception.status_code, 422)

    def test_verifier_sampling_is_deterministic_and_not_position_quantized(self) -> None:
        server.VERIFIER_SAMPLE_RATE = 0.37
        first = [
            server._should_run_verifier(True, index % 2, f"generic-sample-{index}")
            for index in range(2000)
        ]
        second = [
            server._should_run_verifier(True, index % 2, f"generic-sample-{index}")
            for index in range(2000)
        ]
        self.assertEqual(first, second)
        observed_rate = sum(first) / len(first)
        self.assertGreater(observed_rate, 0.33)
        self.assertLess(observed_rate, 0.41)


if __name__ == "__main__":
    unittest.main()
