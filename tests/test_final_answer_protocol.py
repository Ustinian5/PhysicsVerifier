import json
import unittest
from rule_framework import final_answer_protocol as p


class FinalAnswerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(question='Find speed.', prediction='v = 2 m/s',
                        answer='PRIVATE_REFERENCE', label='PRIVATE_LABEL',
                        diagnostics=['PRIVATE_DIAG'], library='PRIVATE_LIBRARY')

    def test_verifier_reference_isolation(self):
        text = json.dumps(p.verification_messages(self.row))
        for secret in ('PRIVATE_REFERENCE', 'PRIVATE_LABEL', 'PRIVATE_DIAG', 'PRIVATE_LIBRARY'):
            self.assertNotIn(secret, text)

    def test_labeler_only_sees_explicit_reference_and_student(self):
        messages = p.label_messages(self.row, ['2 m/s', '200 cm/s'])
        payload = json.loads(messages[-1]['content'])
        self.assertEqual(set(payload), {'question', 'student_answer', 'original_reference_answers'})
        self.assertEqual(payload['original_reference_answers'], ['2 m/s', '200 cm/s'])

    def test_missing_reference_or_truncated_generation_not_gold(self):
        for ref in (None, '', [], ['']):
            with self.subTest(reference=ref), self.assertRaises(ValueError):
                p.label_messages(self.row, ref)
        with self.assertRaises(ValueError):
            p.label_messages(dict(self.row, generation_status='incomplete'), '2 m/s')

    def test_step_diagnosis_is_not_final_answer_verdict(self):
        with self.assertRaises(ValueError):
            p.validate_verdict({'diagnostics': [{'message': 'wrong intermediate step'}]})
        self.assertEqual(p.validate_verdict(dict(verdict='uncertain', student_final_answer='',
                                                reason='Reference conflict'))['verdict'], 'uncertain')


if __name__ == '__main__': unittest.main()
