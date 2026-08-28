from __future__ import annotations

import unittest

from training.rl_data.generate_sft_solutions import pick_shortest_correct
from training.swift.smoke_self_judge import spearman


class PickShortestCorrectTests(unittest.TestCase):
    def test_picks_shortest_matching_boxed_answer(self) -> None:
        gold = r"\boxed{2}"
        long_ok = "long reasoning\n\\boxed{2}"
        short_ok = "\\boxed{2}"
        wrong = "\\boxed{3}"
        chosen = pick_shortest_correct([wrong, long_ok, short_ok], gold)
        self.assertEqual(chosen, short_ok)

    def test_none_when_all_wrong(self) -> None:
        self.assertIsNone(pick_shortest_correct(["\\boxed{9}", "no box"], r"\boxed{1}"))


class SpearmanTests(unittest.TestCase):
    def test_perfect_rank_correlation(self) -> None:
        self.assertAlmostEqual(spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]), 1.0)

    def test_inverse_rank_correlation(self) -> None:
        self.assertAlmostEqual(spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]), -1.0)


if __name__ == "__main__":
    unittest.main()
