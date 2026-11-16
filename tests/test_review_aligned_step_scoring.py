import json
import tempfile
import unittest
from pathlib import Path

from scripts import review_aligned_step_scoring as review


def span(start, end):
    return dict(start=start, end=end)


def finding(rid, root, start, end):
    return dict(review_id=rid, approved_spans=[span(start, end)], root_links=[
        dict(root_id=root, same_direction=True, acceptable_source_spans=[span(start, end)])] if root else [])


class QuoteRecoveryTests(unittest.TestCase):
    def test_returns_all_repeats_including_format_only(self):
        text = r"first $x+y$ then \boxed{x+y} then x+y"
        found = review.locate_all(text, r"\boxed{x+y}")
        self.assertEqual(len(found), 3)
        self.assertEqual([review.normalize(s['quote'])[0] for s in found], ['x+y'] * 3)
        for s in found:
            review.check_span(text, s)

    def test_distinguishes_case_sign_power_and_groups(self):
        for source, quote in [("M", "m"), ("a-b", "a+b"), ("10^{-18}", "10^{-17}"),
                              ("(x+y)z", "x+yz"), ("x^{ab}", "x^ab"), ("1m", "1M")]:
            with self.subTest(source=source, quote=quote):
                self.assertEqual(review.locate_all(source, quote), [])

    def test_format_changes_and_nested_box_preserve_offsets(self):
        text = r"abc $$\boxed{\frac{x^{2}}{M} - y}$$ def"
        found = review.locate_all(text, r"\frac{x^{2}}{M}-y")
        self.assertEqual(len(found), 1)
        self.assertEqual(text[found[0]['start']:found[0]['end']], found[0]['quote'])
        self.assertNotIn('abc', found[0]['quote'])

    def test_omitted_intermediate_equality_needs_review(self):
        self.assertFalse(review.locate_all('x = a/b = c/d = 5.85e-17', 'x = a/b = 5.85e-17'))

    def test_no_nearest_gold_heuristic(self):
        self.assertEqual([(s['start'], s['end']) for s in review.locate_all('ab ab ab', 'ab')],
                         [(0, 2), (3, 5), (6, 8)])

    def test_altered_review_quote_is_rejected(self):
        with self.assertRaises(ValueError):
            review.check_span('abc', dict(start=0, end=2, quote='ac'))


class ReviewedMatchingTests(unittest.TestCase):
    def setUp(self):
        self.gt = [dict(error_id='q/E1', start_char=100, end_char=110),
                   dict(error_id='q/E2', start_char=200, end_char=210)]
        self.states = {'q/E1': {'state': 'active_error'}, 'q/E2': {'state': 'self_corrected'}}

    def test_duplicate_diagnoses_do_not_multiply_tp(self):
        ds = [finding('D1', 'q/E1', 0, 10), finding('D2', 'q/E1', 0, 10)]
        self.assertEqual(len(review.assign(self.gt, ds, self.states, False)), 1)
        self.assertEqual(sum(bool(d['approved_spans']) for d in ds), 2)

    def test_one_diagnosis_cannot_hit_two_roots(self):
        d = finding('D1', 'q/E1', 0, 10)
        d['root_links'].append(dict(root_id='q/E2', acceptable_source_spans=[span(0, 10)]))
        self.assertEqual(len(review.assign(self.gt, [d], self.states, False)), 1)

    def test_opposite_claim_does_not_hit_by_overlap(self):
        d = finding('D1', None, 100, 110)
        self.assertEqual(review.assign(self.gt, [d], self.states, False), [])

    def test_self_corrected_not_in_active_hits_but_prediction_is_retained(self):
        ds = [finding('D1', 'q/E2', 200, 210)]
        self.assertEqual(len(review.assign(self.gt, ds, self.states, False)), 1)
        self.assertEqual(len(review.assign(self.gt, ds, self.states, True)), 0)
        self.assertTrue(ds[0]['approved_spans'])

    def test_alias_is_explicit_and_thresholds_unchanged(self):
        d = finding('D1', 'q/E1', 101, 103)
        d['root_links'][0]['acceptable_source_spans'] = []
        self.assertEqual(len(review.assign(self.gt, [d], self.states, False)), 1)
        self.assertEqual(review.overlap(span(100, 110), span(108, 120)), -1)

    def test_unknown_overlap_excludes_matched_diagnoses(self):
        ds = [finding('D1', None, 0, 10), finding('D2', None, 10, 20)]
        unknown = [dict(start_char=0, end_char=10, span_valid=True)]
        self.assertEqual(review.unresolved_overlaps(ds, [], unknown), ['D1'])
        self.assertEqual(review.unresolved_overlaps(ds, [{'review_id': 'D1'}], unknown), [])

    def test_empty_predictions_keep_recall_denominator(self):
        self.assertEqual(review.rates(0, 0, 180), dict(precision=0, recall=0, f1=0))

    def test_source_drift_aborts_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'offline_review'
            out.mkdir()
            source = root / 'result.json'
            source.write_text('{}')
            review.write(out / 'protocol.json', dict(policy=review.POLICY,
                source_sha256={'result.json': review.digest(source)}))
            review.verify_sources(root)
            source.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'source drift'):
                review.verify_sources(root)


if __name__ == '__main__':
    unittest.main()
