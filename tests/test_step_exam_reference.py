import json
import unittest

from rule_framework.flat_skill_rag import continuation_messages
from scripts.run_step_exam_reference import FIELD, add_answer
from scripts.score_step_exam_reference import paired_f1


class ExamReferenceComparisonTest(unittest.TestCase):
    def test_only_current_exam_answer_changes_and_original_messages_stay_frozen(self):
        row = dict(id='pendulum', question='Find the period.', prediction='T = 2 s.', context='')
        original = continuation_messages(row, {'queries': ['周期 period']}, [])
        answers = ['2 s', '', 'literal <|text|>']
        treated = add_answer(original, answers)
        self.assertEqual(original[:-1], treated[:-1])
        head, tail = treated[-1]['content'].split('\n', 1)
        fields = json.loads(head)
        self.assertEqual(fields.pop(FIELD), answers)
        self.assertNotIn('<|', head)
        self.assertEqual(json.dumps(fields, ensure_ascii=False) + '\n' + tail, original[-1]['content'])
        self.assertNotIn(FIELD, original[-1]['content'])

    def test_significance_swaps_whole_questions(self):
        control = [dict(id=str(i), gt=2, hits=0, locatable=2) for i in range(6)]
        treated = [dict(row, hits=2) for row in control]
        result = paired_f1(control, treated)
        self.assertEqual(result['delta_f1_pp'], 100)
        self.assertEqual(result['paired_bootstrap_ci95_pp'], [100, 100])
        self.assertEqual(result['paired_two_sided_p'], 2 / 64)
        self.assertEqual(paired_f1(treated, treated)['paired_two_sided_p'], 1)
        with self.assertRaises(ValueError):
            paired_f1(control, treated[::-1])
