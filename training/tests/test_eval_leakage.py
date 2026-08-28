from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.rl_data.audit_eval_leakage import audit
from training.swift.analyze_llm_verifier_onset import earliest_onset


class LeakageAuditTests(unittest.TestCase):
    def test_exact_id_and_hash_are_hard_fail(self) -> None:
        train = [
            {"sample_id": "1_2", "question": "A unique physics question about F=ma."},
            {"sample_id": "9_9", "question": "Another train-only question."},
        ]
        eval_rows = [
            {"id": "1_2", "question": "A unique physics question about F=ma."},
        ]
        report = audit(train, {"heldout": eval_rows})
        self.assertTrue(report["hard_fail"])
        self.assertIn("1_2", report["exact_id_overlap"]["heldout"])
        self.assertTrue(report["exact_hash_overlap"]["heldout"])

    def test_disjoint_sets_pass(self) -> None:
        train = [{"sample_id": "t1", "question": "Train question alpha unique tokens xyz."}]
        ev = [{"id": "e1", "question": "Eval question beta completely different wording."}]
        report = audit(train, {"hipho_to": ev})
        self.assertTrue(report["ok"])
        self.assertFalse(report["hard_fail"])

    def test_onset_requires_heldout_and_hipho(self) -> None:
        ckpts = [
            {"step": 0, "heldout_correct": 4, "hipho_mns": 0.10},
            {"step": 20, "heldout_correct": 6, "hipho_mns": 0.09},
            {"step": 40, "heldout_correct": 7, "hipho_mns": 0.12},
            {"step": 60, "heldout_correct": 7, "hipho_mns": 0.13},
        ]
        out = earliest_onset(ckpts, grader_noise=0.005)
        self.assertEqual(out["onset_step"], 40)

    def test_onset_none_when_no_stable_gain(self) -> None:
        ckpts = [
            {"step": 0, "heldout_correct": 4, "hipho_mns": 0.10},
            {"step": 20, "heldout_correct": 5, "hipho_mns": 0.10},
            {"step": 100, "heldout_correct": 5, "hipho_mns": 0.101},
        ]
        out = earliest_onset(ckpts, grader_noise=0.01)
        self.assertIsNone(out["onset_step"])


if __name__ == "__main__":
    unittest.main()
