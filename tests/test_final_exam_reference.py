import json
import unittest
from fractions import Fraction
from itertools import product

from scripts import run_final_exam_reference as trial
from scripts import score_final_exam_reference as score


class FinalReferenceTest(unittest.TestCase):
    def test_neutral_pair_contains_only_current_answer_as_difference(self):
        row = dict(id='pendulum',question='Find its period.',prediction='The period is two seconds.',context='')
        original = trial.final.p.continuation_messages(row,{'queries':['周期 period']},[],arm='pure')
        left,right = trial.messages(original,['2 s','literal <|text|>'])
        self.assertIn(trial.BLIND,original[0]['content'])
        self.assertIn(trial.NEUTRAL,left[0]['content'])
        self.assertEqual(left[:-1],right[:-1])
        self.assertEqual(left[1:],original[1:])
        head,tail = right[-1]['content'].split('\n',1)
        obj = json.loads(head)
        self.assertNotIn('<|',head)
        self.assertEqual(obj.pop(trial.step.FIELD),['2 s','literal <|text|>'])
        self.assertEqual(json.dumps(obj)+'\n'+tail,left[-1]['content'])

    def test_exact_null_matches_brute_force_with_failures(self):
        y = ['incorrect']*4+['correct']*2
        a = ['correct','incorrect',None,'uncertain','correct','incorrect']
        b = ['incorrect','correct','incorrect','correct','incorrect',None]
        one,two = [[score.f1_terms(k,v) for k,v in zip(y,out)] for out in (a,b)]
        def f1(terms):
            n,d = map(sum,zip(*terms))
            return Fraction(n,d) if d else 0
        observed = f1(two)-f1(one)
        extreme = 0
        for mask in product((False,True),repeat=len(y)):
            left = [v if s else u for s,u,v in zip(mask,one,two)]
            right = [u if s else v for s,u,v in zip(mask,one,two)]
            extreme += abs(f1(right)-f1(left)) >= abs(observed)
        actual = score.exact_swap(one,two)
        self.assertEqual(actual['paired_two_sided_p'],extreme/2**len(y))
        self.assertEqual(actual['delta_f1_pp'],100*float(observed))
        self.assertEqual(score.exact_swap(one,one)['paired_two_sided_p'],1)

    def test_frozen_binary_denominator_and_missing_pair_range(self):
        labels = {'a':'incorrect','b':'incorrect','c':'correct','d':'correct'}
        def row(v):
            return dict(status='complete' if v else 'content_failure', value={'verdict':v} if v else None)
        pairs={q:dict(pure=row(a),with_exam_answer=row(b)) for q,a,b in
               [('a','correct','incorrect'),('b',None,'uncertain'),('c','incorrect','correct'),('d','correct',None)]}
        result=score.analyze(labels,pairs,'incorrect')
        self.assertEqual(result['groups']['pure']['valid_binary'],3)
        self.assertEqual(result['groups']['with_exam_answer']['valid_binary'],2)
        self.assertEqual(result['inference']['delta_f1_pp'],100.)
        self.assertEqual(result['common_binary']['corrected'],2)
        bounds=result['registered_sensitivity']
        self.assertEqual(len(bounds['scenarios']),9)
        self.assertEqual(bounds['delta_f1_pp_range'][1],100.)
