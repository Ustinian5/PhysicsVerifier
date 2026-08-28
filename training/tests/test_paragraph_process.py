from __future__ import annotations

import unittest

from training.openrlhf.audit_ray_bind import audit_listeners, parse_listeners
from training.reward_server.paragraph_process import (
    ProcessParagraphWeights,
    group_has_variance,
    score_text_with_diagnostics,
)


class ParagraphProcessTests(unittest.TestCase):
    def test_clean_vs_dirty_paragraphs_spread(self) -> None:
        text = (
            "First paragraph states Newton's second law correctly as F = ma. "
            * 4
            + "\n\n"
            + "Second paragraph claims that mass decreases when velocity increases which is wrong. "
            * 4
        )
        clean = score_text_with_diagnostics(
            text,
            [],
            min_len=90,
            target_len=150,
            max_len=180,
        )
        dirty = score_text_with_diagnostics(
            text,
            [{"severity": "error", "start_char": text.find("Second"), "end_char": text.find("Second") + 20}],
            min_len=90,
            target_len=150,
            max_len=180,
        )
        self.assertGreater(clean["score"], dirty["score"])
        self.assertEqual(clean["n_bad_paragraphs"], 0)
        self.assertGreaterEqual(dirty["n_bad_paragraphs"], 1)
        self.assertTrue(group_has_variance([clean["score"], dirty["score"]]))

    def test_later_error_scores_higher_than_early_error(self) -> None:
        text = ("A" * 160) + "\n\n" + ("B" * 160) + "\n\n" + ("C" * 160)
        early = score_text_with_diagnostics(
            text,
            [{"severity": "error", "start_char": 10, "end_char": 20}],
            min_len=90,
            target_len=150,
            max_len=180,
        )
        late = score_text_with_diagnostics(
            text,
            [{"severity": "error", "start_char": len(text) - 20, "end_char": len(text) - 5}],
            min_len=90,
            target_len=150,
            max_len=180,
        )
        self.assertGreater(late["r_first"], early["r_first"])
        self.assertGreater(late["score"], early["score"])

    def test_density_separates_error_counts(self) -> None:
        text = "short solution with one equation F=ma and a boxed answer."
        one = score_text_with_diagnostics(
            text,
            [{"severity": "error", "start_char": 0, "end_char": 5}],
        )
        many = score_text_with_diagnostics(
            text,
            [{"severity": "error", "start_char": 0, "end_char": 5}] * 5,
        )
        self.assertGreater(one["r_dense"], many["r_dense"])

    def test_process_score_ignores_final_answer_and_format(self) -> None:
        text = "First paragraph states Newton's second law correctly as F = ma. " * 6
        weights = ProcessParagraphWeights(clean=0.5, first=0.3, dense=0.2, answer=1.0, format=1.0)
        with_answer = score_text_with_diagnostics(
            text, [], acc=True, boxed=True, weights=weights, process_only=True
        )
        without_answer = score_text_with_diagnostics(
            text, [], acc=False, boxed=False, weights=weights, process_only=True
        )
        self.assertEqual(with_answer["score"], without_answer["score"])

    def test_warning_counts_half_of_error(self) -> None:
        text = "First paragraph states Newton's second law correctly as F = ma. " * 6
        one_error = score_text_with_diagnostics(
            text,
            [{"severity": "error", "start_char": 10, "end_char": 30}],
            min_len=90,
            target_len=150,
            max_len=180,
        )
        one_warning = score_text_with_diagnostics(
            text,
            [{"severity": "warning", "start_char": 10, "end_char": 30}],
            min_len=90,
            target_len=150,
            max_len=180,
        )
        self.assertEqual(one_error["n_errors"], 1.0)
        self.assertEqual(one_warning["n_errors"], 0.5)
        self.assertGreater(one_warning["score"], one_error["score"])

    def test_warning_and_error_mix_produces_variance(self) -> None:
        text = ("A" * 160) + "\n\n" + ("B" * 160)
        clean = score_text_with_diagnostics(text, [], min_len=90, target_len=150, max_len=180)
        warn = score_text_with_diagnostics(
            text,
            [{"severity": "warning", "start_char": 170, "end_char": 190}],
            min_len=90,
            target_len=150,
            max_len=180,
        )
        err = score_text_with_diagnostics(
            text,
            [{"severity": "error", "start_char": 170, "end_char": 190}],
            min_len=90,
            target_len=150,
            max_len=180,
        )
        scores = [clean["score"], warn["score"], err["score"]]
        self.assertTrue(group_has_variance(scores))
        self.assertGreater(clean["score"], warn["score"])
        self.assertGreater(warn["score"], err["score"])

    def test_answer_only_group_has_no_variance(self) -> None:
        self.assertFalse(group_has_variance([0.0] * 8))
        self.assertTrue(group_has_variance([0.0, 0.0, 0.4, 0.0]))


class RayBindAuditTests(unittest.TestCase):
    def test_loopback_gcs_ok(self) -> None:
        ss = "LISTEN 0 4096 127.0.0.1:26379 0.0.0.0:*\nLISTEN 0 4096 127.0.0.1:28265 0.0.0.0:*\n"
        listeners = parse_listeners(ss)
        report = audit_listeners(
            listeners, gcs_port=26379, dashboard_port=28265, client_port=None,
            worker_min=26381, worker_max=27380,
        )
        self.assertTrue(report["ok"])

    def test_ipv4_mapped_loopback_gcs_ok(self) -> None:
        ss = "LISTEN 0 4096 [::ffff:127.0.0.1]:26379 *:*\nLISTEN 0 4096 127.0.0.1:28265 0.0.0.0:*\n"
        report = audit_listeners(
            parse_listeners(ss),
            gcs_port=26379,
            dashboard_port=28265,
            client_port=26380,
            worker_min=26381,
            worker_max=27380,
        )
        self.assertTrue(report["ok"])

    def test_public_gcs_fails(self) -> None:
        ss = "LISTEN 0 4096 0.0.0.0:26379 0.0.0.0:*\nLISTEN 0 4096 127.0.0.1:28265 0.0.0.0:*\n"
        report = audit_listeners(
            parse_listeners(ss),
            gcs_port=26379,
            dashboard_port=28265,
            client_port=None,
            worker_min=26381,
            worker_max=27380,
        )
        self.assertFalse(report["ok"])
        self.assertTrue(any("gcs" in f for f in report["failures"]))

    def test_lan_gcs_fails(self) -> None:
        ss = "LISTEN 0 4096 10.31.112.24:26379 0.0.0.0:*\n"
        report = audit_listeners(
            parse_listeners(ss),
            gcs_port=26379,
            dashboard_port=28265,
            client_port=None,
            worker_min=26381,
            worker_max=27380,
        )
        self.assertFalse(report["ok"])

    def test_worker_wildcard_warns_not_fail(self) -> None:
        ss = (
            "LISTEN 0 4096 127.0.0.1:26379 0.0.0.0:*\n"
            "LISTEN 0 4096 127.0.0.1:28265 0.0.0.0:*\n"
            "LISTEN 0 4096 0.0.0.0:26390 0.0.0.0:*\n"
        )
        report = audit_listeners(
            parse_listeners(ss),
            gcs_port=26379,
            dashboard_port=28265,
            client_port=None,
            worker_min=26381,
            worker_max=27380,
            allow_wildcard_workers=True,
        )
        self.assertTrue(report["ok"])
        self.assertTrue(report["warnings"])


if __name__ == "__main__":
    unittest.main()
