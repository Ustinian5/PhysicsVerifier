from __future__ import annotations

import unittest

from training.swift.convert_prompts_to_swift import convert_row
from training.swift.physics_reward_plugin import build_reward_payload, extract_questions


class SwiftConvertTests(unittest.TestCase):
    def test_keeps_system_and_user_messages(self) -> None:
        row = {
            "input": [
                {"role": "system", "content": "You are a solver."},
                {"role": "user", "content": "What is F=ma?"},
            ],
            "label": "\\boxed{F=ma}",
            "metadata": {"sample_id": "1_2", "question": "What is F=ma?"},
        }
        out = convert_row(row)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["solution"], "\\boxed{F=ma}")
        self.assertEqual(out["messages"][0]["role"], "system")
        self.assertEqual(out["messages"][-1]["role"], "user")
        self.assertEqual(out["sample_id"], "1_2")
        self.assertEqual(out["question"], "What is F=ma?")


class SwiftRewardPayloadTests(unittest.TestCase):
    def test_payload_uses_completion_as_query_and_user_as_prompt(self) -> None:
        messages = [
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q1"},
            ]
        ]
        payload = build_reward_payload(
            ["ans-a", "ans-b"],
            messages=messages,
            solution="\\boxed{1}",
        )
        self.assertEqual(payload["query"], ["ans-a", "ans-b"])
        self.assertEqual(payload["prompts"], ["q1", "q1"])
        self.assertEqual(payload["labels"], ["\\boxed{1}", "\\boxed{1}"])

    def test_extract_questions_from_message_lists(self) -> None:
        qs = extract_questions(
            [[{"role": "user", "content": "hello"}]],
            3,
        )
        self.assertEqual(qs, ["hello", "hello", "hello"])


if __name__ == "__main__":
    unittest.main()
