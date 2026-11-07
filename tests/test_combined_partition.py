import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from scripts import partition_combined_source as p


class PartitionTests(unittest.TestCase):
    def test_families_do_not_split_or_depend_on_model_outcomes(self):
        groups=[dict(id=str(i),members=[str(i)+'a',str(i)+'b']) for i in range(303)]
        eligible={x for g in groups for x in g['members']}
        selected,reserve=p.choose(groups,eligible,size=6)
        self.assertEqual(len(selected),6);self.assertEqual(len(reserve),600)
        for g in groups:
            self.assertFalse(set(g['members'])&set(selected) and set(g['members'])&set(reserve))
        self.assertEqual(p.choose(list(reversed(groups)),eligible,size=6),(selected,reserve))

    def test_impossible_exact_size_fails(self):
        groups=[dict(id=str(i),members=[str(i)+'a',str(i)+'b']) for i in range(303)]
        with self.assertRaises(ValueError):p.choose(groups,{x for g in groups for x in g['members']},size=5)

    def test_source_corrections_preserve_raw_decisions_and_exclude_missing_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=[dict(id=x,question=x) for x in ('a','b','c')]
            config={'test':True};plan=dict(configuration=config,configuration_sha256=p.digest(config))
            p.write(root/'plan.json',plan)
            raw=dict(state='complete',signature=p.digest([rows,config]),items=[dict(
                id=x,domain='physics',self_contained='no' if x=='c' else 'yes',reason='model decision')
                for x in ('a','b','c')])
            for name,value in [('questions.json',rows),('batches.json',[rows]),
                    ('reviewed/0000.json',raw),('execution_status.json',{'state':'source_review_complete'}),
                    ('family_components.json',dict(source_plan_sha256=p.sha(root/'plan.json'),code_hashes={},components=[]))]:
                p.write(root/name,value)
            p.write(root/'pilot_acceptance.json',dict(passed=True,configuration_sha256=plan['configuration_sha256'],
                reviewed_sha256=p.sha(root/'reviewed/0000.json'),known_source_issues=[dict(
                id='b',action='exclude_before_split',reason='Missing required prior part')]))
            with patch.object(p.review,'verify',return_value=plan),patch.object(p.review.corpus,'evidence_valid'), \
                    patch.object(p,'choose',return_value=(['a'],[])) as choose:
                p.freeze(root)
                self.assertEqual(choose.call_args.args[1],{'a'})
            result=p.read(root/'split.json')
            self.assertEqual({x['id'] for x in result['excluded']},{'b','c'})
            self.assertEqual(p.read(root/'reviewed/0000.json'),raw)


if __name__=='__main__':unittest.main()
