from __future__ import annotations

import unittest

from evaluation.benchmarks.hipho.official_scoring import (
    answer_level_score,
    best_step_score,
    exam_totals,
    extract_all_boxed,
    mean_normalized_score,
    medal_for_points,
    problem_score,
    score_problem_record,
    step_score_from_criteria,
)


class OfficialScoringTests(unittest.TestCase):
    def test_answer_level_full_credit(self) -> None:
        pred = r"Reasoning... \boxed{B}"
        score, details = answer_level_score(pred, [r"\boxed{B}"], [1.0])
        self.assertEqual(score, 1.0)
        self.assertTrue(details[0]["correct"])

    def test_extract_multiple_boxed_in_order(self) -> None:
        text = r"first \boxed{1} then \boxed{2}"
        self.assertEqual(extract_all_boxed(text), ["1", "2"])

    def test_step_level_partial_credit(self) -> None:
        criteria = [
            {"id": "c0", "description": "write F=ma", "weight": 2.0},
            {"id": "c1", "description": "solve for a", "weight": 1.0},
        ]
        grader = [
            {"id": "c0", "s": 1.0, "awarded_points": 2.0, "evidence": "F=ma", "reason": "ok"},
            {"id": "c1", "s": 0.5, "awarded_points": 0.5, "evidence": "a=F/m", "reason": "partial"},
        ]
        score, audited = step_score_from_criteria(grader, criteria)
        self.assertAlmostEqual(score, 2.5)
        self.assertEqual(len(audited), 2)

    def test_max_of_answer_and_step(self) -> None:
        self.assertEqual(problem_score(3.0, 1.2), 3.0)
        self.assertEqual(problem_score(0.0, 1.2), 1.2)

    def test_multiple_marking_schemes_take_max(self) -> None:
        schemes = [
            {"name": "A", "criteria": [{"id": "c0", "weight": 1.0}]},
            {"name": "B", "criteria": [{"id": "c0", "weight": 1.0}]},
        ]

        def score_scheme(scheme):
            s = 0.2 if scheme["name"] == "A" else 0.9
            return s, [{"id": "c0", "s": s, "awarded_points": s, "weight": 1.0}]

        best, payload = best_step_score(schemes, score_scheme)
        self.assertAlmostEqual(best, 0.9)
        self.assertEqual(payload["scheme"], "B")

    def test_criterion_sum_and_exam_mns(self) -> None:
        rec = score_problem_record(
            prediction=r"\boxed{wrong}",
            gold_answers=[r"\boxed{right}"],
            full_marks=[2.0],
            marking_schemes=[{"name": "official", "criteria": [{"id": "c0", "weight": 2.0, "description": "setup"}]}],
            step_grader=lambda _pred, _scheme: [
                {"id": "c0", "s": 0.5, "awarded_points": 1.0, "evidence": "setup", "reason": "partial"}
            ],
        )
        self.assertEqual(rec["answer_score"], 0.0)
        self.assertEqual(rec["step_score"], 1.0)
        self.assertEqual(rec["final_score"], 1.0)
        rows = [
            {"exam": "IPhO_2025", "final_score": 19.7, "full_mark": 30.0},
            {"exam": "F=MA_2024", "final_score": 12.0, "full_mark": 25.0},
        ]
        exams = exam_totals(rows)
        self.assertAlmostEqual(exams["IPhO_2025"]["normalized"], 19.7 / 30.0)
        mns = mean_normalized_score(exams)
        self.assertAlmostEqual(mns, ((19.7 / 30.0) + (12.0 / 25.0)) / 2)
        self.assertEqual(medal_for_points("IPhO_2025", 19.7), "gold")


if __name__ == "__main__":
    unittest.main()
