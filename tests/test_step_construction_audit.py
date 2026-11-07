import unittest
from scripts import audit_step_construction as a


class SourceAuditTests(unittest.TestCase):
    def test_reordering_is_allowed_but_answer_changes_and_duplicates_are_not(self):
        rows = [{'id':'a','prediction':'30B original'}, {'id':'b','prediction':'other'}]
        a.same_answers(rows, list(reversed(rows)))
        for invalid in ([rows[0], rows[0]], [dict(rows[0], prediction='4B'), rows[1]], rows[:1]):
            with self.assertRaisesRegex(ValueError, 'verified complete'):
                a.same_answers(rows, invalid)

    def test_external_problem_or_reference_in_model_input_is_rejected(self):
        pool = [{'id':'a','question':'original'}]
        a.membership(pool, [dict(pool[0], prediction='30B',answer='',context='')])
        for row in (dict(pool[0],question='textbook'),dict(pool[0],answer='reference'),
                    dict(pool[0],context='extra'),dict(pool[0],id='external')):
            with self.assertRaises(ValueError): a.membership(pool,[row])


if __name__ == '__main__': unittest.main()
