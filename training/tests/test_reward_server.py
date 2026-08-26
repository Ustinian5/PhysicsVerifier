from __future__ import annotations

import asyncio
import unittest

from training.reward_server import physics_reward_server as server


class _FakeVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def verify(self, sample):
        self.calls += 1
        return {
            "diagnostics": [
                {"severity": "error", "rule": "test_rule", "message": "test error"}
            ]
        }


class RewardServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_mode = server.REWARD_MODE
        self.original_get_verifier = server._get_verifier
        self.original_append_metrics = server._append_metrics
        server._append_metrics = lambda record: None

    def tearDown(self) -> None:
        server.REWARD_MODE = self.original_mode
        server._get_verifier = self.original_get_verifier
        server._append_metrics = self.original_append_metrics

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


if __name__ == "__main__":
    unittest.main()
