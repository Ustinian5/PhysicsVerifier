import unittest

from scripts.audit_aligned_full_chain import reference_origins


class ReferenceOriginTests(unittest.TestCase):
    quote = 'The incoming momentum flux has been incorrectly omitted.'

    def test_material_only_quote_is_traced_to_complete_source(self):
        row = {'prediction': 'A different solution about electric potential.'}
        refs = [{'id': 'case1', 'title': 'Variable mass', 'text': 'Context. '+self.quote+' More.'}]
        result = reference_origins(row, {'evidence': {'quote': self.quote}}, refs)
        self.assertFalse(result['in_student'])
        self.assertEqual(result['reference_matches'][0]['reference_id'], 'case1')
        self.assertEqual(result['reference_matches'][0]['source_span']['quote'], self.quote)

    def test_shared_quote_does_not_prove_wrong_source(self):
        row = {'prediction': self.quote}
        refs = [{'id': 'case1', 'title': 'Variable mass', 'text': self.quote}]
        result = reference_origins(row, {'evidence': {'quote': self.quote}}, refs)
        self.assertTrue(result['in_student'])
        self.assertEqual(len(result['reference_matches']), 1)

    def test_short_common_formula_not_claimed_as_material_provenance(self):
        result = reference_origins({'prediction': 'x'}, {'evidence': {'quote': 'F=ma'}},
            [{'id': 'case1', 'title': 'Dynamics', 'text': 'F=ma'}])
        self.assertFalse(result['reference_matches'])

    def test_formatting_changes_are_allowed_but_wrong_sign_is_not(self):
        text = r'The field is \boxed{Q/(4\pi\epsilon_0) (1/L - 1/R)}.'
        refs = [{'id': 'case1', 'title': 'Field', 'text': text}]
        q = r'The field is Q/(4\pi\epsilon_0)(1/L-1/R).'
        result = reference_origins({'prediction': 'none'}, {'evidence': {'quote': q}}, refs)
        self.assertEqual(len(result['reference_matches']), 1)
        result = reference_origins({'prediction': 'none'}, {'evidence': {'quote': q.replace('-1/R','+1/R')}}, refs)
        self.assertFalse(result['reference_matches'])


if __name__ == '__main__':
    unittest.main()
