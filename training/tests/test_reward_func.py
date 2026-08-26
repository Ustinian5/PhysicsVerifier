from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from training.openrlhf import physics_reward_func as reward_module


class _Response:
    def __init__(self, payload) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class RewardFunctionTests(unittest.TestCase):
    def test_input_vectors_must_align_before_request(self) -> None:
        with patch.object(reward_module.requests, "post") as post:
            with self.assertRaisesRegex(ValueError, "identical lengths"):
                reward_module.reward_func(["q1", "q2"], ["p1"], ["1", "2"])
        post.assert_not_called()

    def test_valid_reward_vector(self) -> None:
        with patch.object(
            reward_module.requests,
            "post",
            return_value=_Response({"rewards": [1, 0.25]}),
        ):
            result = reward_module.reward_func(["q1", "q2"], ["p1", "p2"], ["1", "2"])
        self.assertTrue(torch.equal(result, torch.tensor([1.0, 0.25])))

    def test_reward_count_must_match_query_count(self) -> None:
        with patch.object(
            reward_module.requests,
            "post",
            return_value=_Response({"rewards": [1.0]}),
        ):
            with self.assertRaisesRegex(RuntimeError, "count mismatch"):
                reward_module.reward_func(["q1", "q2"], ["p1", "p2"], ["1", "2"])

    def test_non_finite_reward_is_rejected(self) -> None:
        with patch.object(
            reward_module.requests,
            "post",
            return_value=_Response({"rewards": [float("nan")]}),
        ):
            with self.assertRaisesRegex(RuntimeError, "finite"):
                reward_module.reward_func(["q1"], ["p1"], ["1"])


if __name__ == "__main__":
    unittest.main()
