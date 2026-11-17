import copy
import unittest

from scripts.review_aligned_step_revision import reusable_decision


class ReviewReuseTests(unittest.TestCase):
    def setUp(self):
        self.d=dict(message='Wrong factor',evidence={'quote':'x=2'},reference_ids=[])
        self.r=dict(id='q1',original_diagnostic=self.d,approved_spans=[{'start':0,'end':3,'quote':'x=2'}],
                    root_links=[{'root_id':'q1/E1'}],disposition='same_root')

    def test_identical_diagnosis_can_reuse_but_not_another_question(self):
        self.assertEqual(reusable_decision([self.r],'q1',self.d),self.r)
        self.assertIsNone(reusable_decision([self.r],'q2',self.d))

    def test_even_small_diagnostic_change_requires_review(self):
        changed=copy.deepcopy(self.d);changed['message']='Correct factor'
        self.assertIsNone(reusable_decision([self.r],'q1',changed))

    def test_conflicting_previous_links_require_review(self):
        other=copy.deepcopy(self.r);other['root_links']=[{'root_id':'q1/E2'}]
        self.assertIsNone(reusable_decision([self.r,other],'q1',self.d))


if __name__=='__main__':unittest.main()
