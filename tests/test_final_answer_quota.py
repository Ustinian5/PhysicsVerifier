import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_final_answer_labels_guarded import quota_status


class QuotaTests(unittest.TestCase):
    def setup_pool(self, root, correct=0, incorrect=0, uncertain=0):
        (root / 'labels').mkdir()
        (root / 'plan.json').write_text(json.dumps({'configuration_sha256': 'test'}))
        (root / 'references_private.json').write_text(json.dumps([
            {'id': str(i), 'reference_complete': i < 679} for i in range(702)]))
        i = 0
        for verdict, count in [('correct', correct), ('incorrect', incorrect), ('uncertain', uncertain)]:
            for _ in range(count):
                (root / 'labels' / f'{i}.json').write_text(json.dumps(dict(
                    id=str(i), state='complete', judgment=dict(verdict=verdict,
                    student_final_answer='answer', reason='fixture'))))
                i += 1

    def test_small_pilot_is_not_population_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.setup_pool(root, 5, 1)
            s = quota_status(root)
            self.assertEqual(s['pending'], 673)
            self.assertEqual(s['maximum_possible'], {'correct': 678, 'incorrect': 674})
            self.assertFalse(s['quota_impossible'])
            self.assertFalse(s['quota_met'])

    def test_exact_minority_bound(self):
        for majority, impossible in [(379, False), (380, True)]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); self.setup_pool(root, majority)
                s = quota_status(root)
                self.assertEqual(s['quota_impossible'], impossible)
                self.assertEqual(s['maximum_possible']['incorrect'], 679-majority)

    def test_uncertain_never_fills_quota(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.setup_pool(root, 300, 299, 80)
            s = quota_status(root)
            self.assertTrue(s['quota_impossible'])
            self.assertFalse(s['quota_met'])
            self.assertEqual(s['pending'], 0)

    def test_duplicate_failure_record_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.setup_pool(root, 1)
            (root / 'generation_failures').mkdir()
            (root / 'generation_failures/0.json').write_text('{"id":"0"}')
            with self.assertRaises((ValueError, RuntimeError)):
                quota_status(root)

    def test_both_quotas_met(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.setup_pool(root, 300, 300)
            s = quota_status(root)
            self.assertTrue(s['quota_met'])
            self.assertFalse(s['quota_impossible'])


if __name__ == '__main__':
    unittest.main()
