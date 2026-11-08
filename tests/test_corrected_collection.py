import unittest
from scripts import run_corrected_collection as c


class CollectionTests(unittest.TestCase):
    def split(self):
        return {'build':[dict(id='b'+str(i),question='build '+str(i),answer='secret') for i in range(2000)],
                'reserve':[dict(id='h'+str(i),question='held '+str(i),label='secret') for i in range(702)]}

    def test_populations_and_payload_whitelist(self):
        tasks=c.ordered_tasks(self.split())
        self.assertEqual(len(tasks),2702)
        self.assertEqual([t['role'] for t in tasks[:4]],['build','heldout','build','heldout'])
        self.assertTrue(all(set(t['row'])=={'id','question'} for t in tasks))
        self.assertEqual([t['row']['id'] for t in tasks if t['role']=='heldout'],['h'+str(i) for i in range(702)])

    def test_overlap_and_insufficient_pool_stop(self):
        s=self.split();s['reserve'][0]['id']='b0'
        with self.assertRaises(ValueError):c.ordered_tasks(s)
        s=self.split();s['reserve']=s['reserve'][:599]
        with self.assertRaises(ValueError):c.ordered_tasks(s)

    def test_existing_generator_protocol_is_reused(self):
        row=dict(id='id',question='question',answer='must not leak')
        self.assertEqual(c.corpus.solver_messages(row),c.corpus.solver_messages({'id':'id','question':'question'}))


if __name__=='__main__':unittest.main()
