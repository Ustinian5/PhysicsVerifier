from __future__ import annotations

import unittest

from training.rl_data.build_rl_prompts import _extract_qa, _parse_answer


class RLDataTests(unittest.TestCase):
    def test_numeric_zero_answer_is_preserved(self) -> None:
        self.assertEqual(_parse_answer(0), ["0"])
        self.assertEqual(_parse_answer([0]), ["0"])

    def test_zero_answer_is_not_replaced_by_fallback_label(self) -> None:
        result = _extract_qa(
            {
                "id": 0,
                "question": "What is the net displacement?",
                "answer": 0,
                "label": "incorrect fallback",
            }
        )
        self.assertEqual(
            result,
            ("What is the net displacement?", ["0"], "0"),
        )


if __name__ == "__main__":
    unittest.main()
