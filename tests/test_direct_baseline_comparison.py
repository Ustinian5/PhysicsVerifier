"""Offline metric regressions; no model calls or private benchmark fixtures."""
import unittest

from scripts.evaluate_direct_baseline_comparison import (
    all_diagnostics_sensitivity, matched_claim_changes, paired_bootstrap,
)
from scripts.run_gpt_oss_selfcheck import paired_f1_bootstrap


def metric(sid="s", tp=1, pred=1, gt=2):
    return {"details": [{"id": sid, "matched_error_count": tp,
                         "pred_locatable_finding_count": pred, "gt_error_count": gt,
                         "location_matches": [{"gt_error_id": "g"}] if tp else []}]}


class DirectComparisonTest(unittest.TestCase):
    def test_delta_sign_is_base_minus_comparison(self):
        base, other = metric(tp=1), metric(tp=0)
        for result in [paired_bootstrap(base, {"other": other}, repetitions=20),
                       paired_f1_bootstrap({"base": base, "other": other}, "base", repetitions=20)]:
            self.assertEqual(result["delta_f1_percentage_points_ci95"]["other"], [66.67, 66.67])

    def test_paired_scores_reject_sample_or_reference_drift(self):
        for other in [metric(sid="different"), metric(gt=3)]:
            with self.assertRaises(ValueError):
                paired_bootstrap(metric(), {"other": other}, repetitions=20)
            with self.assertRaises(ValueError):
                paired_f1_bootstrap({"base": metric(), "other": other}, "base", repetitions=20)

    def test_sensitivity_counts_unlocatable_predictions_without_changing_gt(self):
        m = metric()
        m["summary"] = {"matched_pred_locatable": 1, "recall": .5,
                        "location_unmatched_pred_findings": 0}
        value = all_diagnostics_sensitivity(m, [{"id": "s", "diagnostics": [{}, {}]},
                                               {"id": "outside", "diagnostics": [{}]}])
        self.assertEqual(value["total_diagnostics"], 2)
        self.assertEqual(value["unlocatable_diagnostics"], 1)
        self.assertEqual((value["precision"], value["recall"], value["f1"]), (.5, .5, .5))

    def test_gain_and_loss_use_question_and_claim_together(self):
        value = matched_claim_changes(metric(sid="a"), metric(sid="b"))
        self.assertEqual((value["shared"], value["gained"], value["lost"]), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
