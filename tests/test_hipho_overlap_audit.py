import unittest
from scripts.audit_hipho_overlap import Audit


class HiPhOOverlapTests(unittest.TestCase):
    def test_exact_is_not_misreported_as_numeric(self):
        audit=Audit([dict(id='h',question='Find speed at time 12 s.',context='')])
        hits=audit.matches(dict(id='s',question=' FIND speed at time 12 s. '))
        self.assertEqual(hits[0]['reason'],'normalized_exact')

    def test_numeric_variant(self):
        audit=Audit([dict(id='h',question='Find speed at time 12 s.',context='')])
        self.assertEqual(audit.matches(dict(id='s',question='Find speed at time 24 s.'))[0]['reason'],'numeric_template')

    def test_old_sentinels_are_not_hipho_matches(self):
        audit=Audit([dict(id='h',question='Calculate mass.',context='')])
        self.assertEqual(audit.matches(dict(id='trainphysics_3547',question='Use the previous question.')),[])

    def test_shared_context_detected_without_exact_subquestion(self):
        context=('A massive copper ring rotates around a perpendicular central axis within a uniform external magnetic field. '
                 'The surrounding conducting shell is insulated and moves independently while the ring retains constant temperature. ')
        audit=Audit([dict(id='h',question='Find its inductance.',context=context)])
        hits=audit.matches(dict(id='s',question='Determine its angular momentum.',context=context))
        self.assertTrue(any(x['reference_view']=='context_and_question' and x['reason']=='shared_subproblem_prose' for x in hits))


if __name__=='__main__':unittest.main()
