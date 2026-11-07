import json
import unittest
from scripts import run_combined_source_screen as m


class SourceScreenTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(id='a', question='A problem'), dict(id='b', question='Another problem')]
        self.value = {'items': [dict(id=r['id'], domain='physics', self_contained='yes', reason='Complete text') for r in self.rows]}

    def test_review_does_not_see_answers_or_reference(self):
        row = dict(self.rows[0], prediction='PRIVATE_OLD', reference='PRIVATE_GOLD', context='PRIVATE_CONTEXT', reward=1)
        content = m.screen_messages([row])[-1]['content']
        self.assertEqual(json.loads(content), [self.rows[0]])

    def test_ids_must_be_complete_and_unique(self):
        self.value['items'][1]['id'] = 'a'
        with self.assertRaises(ValueError): m.validate(self.value, self.rows)

    def test_unknown_decisions_rejected_and_uncertain_preserved(self):
        self.value['items'][0]['self_contained'] = 'uncertain'
        self.assertEqual(m.validate(self.value, self.rows)[0]['self_contained'], 'uncertain')
        self.value['items'][0]['self_contained'] = True
        with self.assertRaises(ValueError): m.validate(self.value, self.rows)


if __name__ == '__main__': unittest.main()
