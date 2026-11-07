import copy
import unittest

from scripts import run_corrected_step_generation as m


class ImportSafety(unittest.TestCase):
    def setUp(self):
        self.row = dict(id='test', question='A physics question')
        self.config = dict(generator=m.corpus.MODEL, temperature=.7, top_p=.8,
                           top_k=20, max_tokens=16384, seed=20261107)
        self.record = dict(id='test', state='complete', model=m.corpus.MODEL,
                           prediction='answer', finish_reason='stop', usage=dict(completion_tokens=12))
        self.raw = dict(status='received', request=m.request_body(self.row, self.config),
                        response=dict(model=m.corpus.MODEL, usage=dict(completion_tokens=12),
                                      choices=[dict(finish_reason='stop', message=dict(content='answer'))]))

    def test_exact_import(self):
        m.validate_import(self.row, self.config, self.record, self.raw)

    def test_changed_model_prompt_sampling_or_evidence_rejected(self):
        mutations = [
            lambda x: x['request'].update(temperature=0),
            lambda x: x['request'].update(seed=1),
            lambda x: x['request']['messages'][1].update(content='different question'),
            lambda x: x['request']['messages'][0].update(content='different solver'),
            lambda x: x['response'].update(model='wrong-model'),
            lambda x: x['response']['choices'][0]['message'].update(content='altered answer'),
            lambda x: x['response']['choices'][0].update(finish_reason='length'),
            lambda x: x['response']['usage'].update(completion_tokens=13),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                raw = copy.deepcopy(self.raw); mutate(raw)
                with self.assertRaises(ValueError):
                    m.validate_import(self.row, self.config, self.record, raw)

    def test_solver_sees_question_only(self):
        row = dict(self.row, answer='SECRET', prediction='OLD', context='UNVERIFIED')
        body = m.request_body(row, self.config)
        self.assertEqual(body['messages'][1]['content'], self.row['question'])


if __name__ == '__main__': unittest.main()
