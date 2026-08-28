from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.benchmarks.hipho.hipho_contract import (
    OfficialHiPhOError,
    is_internal_expansion_row,
    is_official_text_only,
    normalize_official_row,
    validate_official_rows,
)


class HiPhOContractTests(unittest.TestCase):
    def test_rejects_internal_expansion_rows(self) -> None:
        row = {
            "id": "evaluation_sample_12_expansion.json",
            "question": "q",
            "exam": "internal",
            "modality": "text-only",
            "full_mark": 1.0,
            "source": "evaluation_sample_12_expansion.json",
        }
        self.assertTrue(is_internal_expansion_row(row))
        with self.assertRaises(OfficialHiPhOError):
            validate_official_rows([normalize_official_row(row)])

    def test_accepts_official_text_only(self) -> None:
        raw = {
            "id": "F=MA_2024_01",
            "question": "What is h?",
            "answer": [r"\boxed{B}"],
            "points": [1.0],
            "modality": "text-only",
            "field": "Mechanics",
            "source": "F=MA_2024",
            "image_question": [],
        }
        row = normalize_official_row(raw, exam_name="F=MA_2024")
        self.assertTrue(is_official_text_only(row))
        validate_official_rows([row], require_text_only=True)
        self.assertEqual(row["exam"], "F=MA_2024")
        self.assertEqual(row["full_mark"], 1.0)
        self.assertEqual(row["source"], "SciYu/HiPhO")

    def test_parses_official_award_string_marking(self) -> None:
        raw = {
            "id": "EuPhO_2024_1_1",
            "question": "Find v_e",
            "answer": [r"\boxed{v_e}"],
            "points": [6.0],
            "modality": "text-only",
            "source": "EuPhO_2024",
            "marking": [
                [
                    "Award 0.3 pt if the answer realizes that puck is sliding initially. Otherwise, award 0 pt.",
                    "Award 1.0 pt if the answer finds the velocity.",
                ]
            ],
        }
        row = normalize_official_row(raw, exam_name="EuPhO_2024")
        schemes = row["marking_schemes"]
        self.assertEqual(len(schemes), 1)
        self.assertEqual(len(schemes[0]["criteria"]), 2)
        self.assertAlmostEqual(schemes[0]["criteria"][0]["weight"], 0.3)
        self.assertAlmostEqual(schemes[0]["criteria"][1]["weight"], 1.0)

    def test_figure_rows_are_not_text_only(self) -> None:
        raw = {
            "id": "F=MA_2024_03",
            "question": "As shown in the figure...",
            "answer": [r"\boxed{D}"],
            "points": [1.0],
            "modality": "text+variable figure",
            "source": "F=MA_2024",
            "image_question": ["image_question/F=MA_2024_03_1.png"],
        }
        row = normalize_official_row(raw)
        self.assertFalse(is_official_text_only(row))


if __name__ == "__main__":
    unittest.main()
