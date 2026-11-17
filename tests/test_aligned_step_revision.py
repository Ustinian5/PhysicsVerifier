import copy
import unittest

from scripts import run_aligned_step_revision as r


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(id='a',question='A physical question',prediction='Current student solution',context='')
        self.queries={'queries':['力 force']}
        self.refs=[dict(id='case1',title='Historical example',text='Historical text')]

    def test_rag_guard_keeps_prepare_and_reference_text(self):
        before=r.ORIGINAL_CONTINUATION(self.row,self.queries,self.refs)
        after=r.continuation(self.row,self.queries,self.refs)
        self.assertEqual(before[:3],after[:3])
        self.assertEqual(after[-1]['content'],before[-1]['content']+'\n\n'+r.RAG_BINDING)
        self.assertEqual(r.base.p.query_messages(self.row),before[:2])

    def test_numbered_mentions_are_review_candidates_not_rejections(self):
        found=r.numbered_dependencies('Use the result of part C.4 and question 2. Part 1 is fully given below.')
        self.assertEqual([v['text'] for v in found],['part C.4','question 2','Part 1'])
        self.assertTrue(all(x['disposition']=='review_required' for x in found))
        self.assertFalse(r.numbered_dependencies('F = ma. All conditions are supplied.'))

    def test_reuse_contract_includes_visible_text_order_and_budget(self):
        p=dict(messages=r.ORIGINAL_CONTINUATION(self.row,self.queries,self.refs),retrieval=dict(ranked=[1]))
        prefix=dict(evidence=dict(usage=dict(prompt_tokens=10,completion_tokens=7,total_tokens=17)))
        limits=dict(output_tokens_per_sample=100,input_tokens_per_sample=1000)
        count=lambda x:len(str(x))
        before=r.contract(p,prefix,count,limits)
        p['retrieval']['ranked']=[2]
        self.assertEqual(before,r.contract(p,prefix,count,limits))
        changed=copy.deepcopy(p);changed['messages'][-1]['content']+=' different reference'
        self.assertNotEqual(before,r.contract(changed,prefix,count,limits))
        self.assertNotEqual(before,r.contract(p,prefix,count,{**limits,'input_tokens_per_sample':999}))

    def test_provenance_flag_keeps_bad_diagnosis_untouched(self):
        quote='A sufficiently long historical statement about a different object.'
        result=dict(diagnostics=[dict(message='Wrong object claim',evidence={'quote':quote})])
        before=copy.deepcopy(result)
        report=r.origin_report(self.row,result,[dict(id='old',title='Old case',text=quote)])
        self.assertEqual(report['records'][0]['origin'],'historical_only')
        self.assertEqual(result,before)

    def test_pruning_removes_only_approved_source_leaves_and_reindexes(self):
        catalog={'domains':[{'id':'d','topics':[{'id':'t','rules':[
            {'rule_id':'a','title':'bad','trigger':'bad context'},
            {'rule_id':'b','title':'keep','trigger':'good context'}],
            'scenario_clusters':[{'id':'c','name':'mixed learned name','summary':'bad context',
                                  'rule_ids':['a','b']}]}]}]}
        original=copy.deepcopy(catalog)
        clean,info=r.old.a.bounded.previous.prune_catalog(catalog,
                    {'a':[{'sample_id':'bad'}],'b':[{'sample_id':'good'}]},{'bad'})
        topic=clean['domains'][0]['topics'][0]
        self.assertEqual([v['rule_id'] for v in topic['rules']],['b'])
        self.assertNotIn('bad',str(topic['scenario_clusters']))
        self.assertEqual(catalog,original)
        self.assertFalse(info['leaf_content_modified'])


if __name__=='__main__':unittest.main()
