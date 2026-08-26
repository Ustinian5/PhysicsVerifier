from __future__ import annotations

import unittest

from training.compat.math_grading import extract_answer, grade_answer_verl


class MathGradingTests(unittest.TestCase):
    def test_extracts_last_boxed_answer(self) -> None:
        self.assertEqual(extract_answer(r"work \boxed{1} then \boxed{2}"), "2")

    def test_accepts_equivalent_fraction(self) -> None:
        self.assertTrue(grade_answer_verl(r"\boxed{\frac{1}{2}}", "0.5"))

    def test_rejects_missing_box(self) -> None:
        self.assertFalse(grade_answer_verl("42", "42"))


if __name__ == "__main__":
    unittest.main()
