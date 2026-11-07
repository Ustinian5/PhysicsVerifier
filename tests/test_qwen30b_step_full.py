import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_qwen30b_step_full as f


class FullStepTests(unittest.TestCase):
    def reviewed_fixture(self):
        old=[dict(id='pilot',question='A ball falls freely in vacuum.',context=''),
             dict(id='bad',question='Calculate the missing value from the diagram.',context='')]
        pool=[dict(id='new1',question='What is the latent heat released by condensation of steam?'),
              dict(id='new2',question='An ideal gas expands adiabatically and reversibly.')]
        review={'status':'review_complete','human_gold':False,'selection_uses_diagnostic_scores':False,
                'formal_answers_generated_at_review':0,'target_count':2,'pilot_only_ids':['pilot'],
                'replacement_order':['new1','new2'],
                'entries':[dict(id=r['id'],question_sha256=f.vr.digest(r['question']),
                    decision='keep' if i==0 else 'exclude',reason='input review') for i,r in enumerate(old)],
                'replacement_reviews':[dict(order=i,id=r['id'],question_sha256=f.vr.digest(r['question']),
                    decision='keep',reason='complete physical inputs') for i,r in enumerate(pool)]}
        return old,pool,{'candidate_ids':['new1','new2'],'rejected':{}},review

    def test_review_replenishes_in_order_and_keeps_even_complete_pilot_separate(self):
        old,pool,iso,review=self.reviewed_fixture()
        rows,_=f.reviewed_questions(old,pool,iso,review,['pilot'],2)
        self.assertEqual([r['id'] for r in rows],['new1','new2'])
        for mutate in ('order','text','scores'):
            changed=copy.deepcopy(review)
            if mutate=='order': changed['replacement_reviews'].reverse()
            elif mutate=='text': changed['replacement_reviews'][0]['question_sha256']='different'
            else: changed['selection_uses_diagnostic_scores']=True
            with self.assertRaises(ValueError):f.reviewed_questions(old,pool,iso,changed,['pilot'],2)

    def test_review_does_not_fill_with_same_family_or_incomplete_inputs(self):
        old,pool,iso,review=self.reviewed_fixture()
        pool[1]['question']=pool[0]['question']
        review['replacement_reviews'][1]['question_sha256']=f.vr.digest(pool[1]['question'])
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            f.reviewed_questions(old,pool,iso,review,['pilot'],2)

    def test_cohort_cannot_be_changed_after_formal_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'STOP').touch();(out/'raw').mkdir()
            with self.assertRaisesRegex(ValueError,'executed batch'):
                f.revise_cohort(out)

    def test_full_skills_build_uses_same_cases_and_resumes_without_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rows = [dict(id=str(i), question='Use the previous gas-cycle question.', prediction='30B answer', context='') for i in range(13)]
            cases = [dict(id='case_'+r['id'], source_ids=[r['id']], text='Original Q, 30B answer, 120B audit '+r['id']) for r in rows]
            for name, value in [('sources',rows),('cases',cases),('bench_questions',[{'id':'held','question':'An independent charged insulating sphere oscillates in an externally specified electromagnetic field.'}]),('plan',{})]:
                f.vr.atomic(out/(name+'.json'),value)
            config = {'model': {'model':f.base.r.EXPECTED_MODEL,'api_base':'http://unused',
                'max_tokens':f.knowledge.PASS_OUTPUT_TOKENS,'reasoning_effort':'medium',
                'max_requests':39,'attempts_per_job':3},
                'budget_limits':{'requests':39,'input_tokens':100000,'output_tokens':100000},
                'prior_usage':f.base.old.ZERO_USAGE,'source_count':13,
                'source_plan_sha256':f.vr.sha(out/'plan.json'),'skill_prompt':f.knowledge.SKILL_PROMPT}
            f.vr.atomic(out/'skills_plan.json',{'configuration':config,'configuration_sha256':f.vr.digest(config)})
            seen=[]
            def mock_call(journal, stage, key, payload):
                self.assertEqual(payload['case'], cases[int(key)])
                seen.append(key)
                return {'skills':[{'title':'Check '+key,'text':'Independently derive the gas-cycle work and compare its sign. '+key}]}
            with patch.object(f.followup.BoundedSkillJournal,'__call__',mock_call):
                a=f.skills(out,{},None)
                b=f.skills(out,{},None)
            self.assertEqual(len(seen),13)
            self.assertEqual(len(a.docs),len(b.docs))
            self.assertGreater(len(a.docs),0)
            self.assertFalse(f.vr.read(out/'skills/pilot_review.json')['physical_quality_passed'])

    def test_cohort_screen_retains_real_leakage_but_not_unrelated_parent_sentinel(self):
        held=[{'id':'test','question':'The finite charged sphere inside a concentric insulating enclosure is surrounded by a second grounded conductor with radius twice that of the first sphere.'}]
        screen=f.CohortScreen(held)
        self.assertTrue(screen.matches(dict(held[0],id='source')))
        self.assertEqual(screen.matches({'id':'unrelated','question':'Use the previous question to evaluate a gas cycle.'}),[])
        self.assertTrue(screen.matches({'id':'embedded','question':'Different question',
            'prediction':'Full text: '+held[0]['question']*3}))

    def test_formal_hold_precedes_any_generation_or_source_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); (out/'STOP').write_text('source correction')
            with self.assertRaisesRegex(ValueError, 'Formal execution is held'):
                f.verify(out, acceptance=True)

    def test_semantic_pilot_cannot_substitute_for_construction_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'construction acceptance'):
                f.construction_acceptance(Path(tmp), {'configuration_sha256':'benchmark-only'})

    def test_selection_is_deterministic_and_excludes_build_and_old_faces(self):
        old = [{'id':'old', 'question':'Unique harmonic oscillator with variable spring', 'context':''}]
        pool = [{'id':'build-copy','question':'Falling mass with drag coefficient'},
                {'id':'old-copy','question':old[0]['question']},
                {'id':'new','question':'Quantum interference between separate beams'},
                {'id':'new2','question':'A thermal reservoir has entropy increase'}]
        materials = [{'id':'build','question':pool[0]['question'], 'prediction':'', 'context':''}]
        a = f.select_questions(old, pool, materials, old, count=2)
        b = f.select_questions(old, list(reversed(pool)), materials, old, count=2)
        self.assertEqual(a, b)
        self.assertEqual(a[0][0], old[0])
        self.assertNotIn(a[0][1]['id'], {'old-copy','build-copy'})

    def test_embedded_heldout_in_generated_material_blocks(self):
        q = 'An isolated system contains two identical conducting shells with charged centers and finite separation.'
        old = [{'id':'held','question':q,'context':''}]
        material = [{'id':'index','question':'', 'prediction':'Stored example: '+q}]
        with self.assertRaisesRegex(ValueError, 'invalidates'):
            f.select_questions(old, [{'id':'new','question':'Another isolated example'}], material, [], count=2)

    def test_shortage_does_not_relax_family_or_fill_from_external_sources(self):
        old = [{'id':'a','question':'The original stem', 'context':''}]
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            f.select_questions(old, [], [], old, count=2)

    def test_clone_refuses_escape_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old=root/'old'; new=root/'new'; old.mkdir()
            p=old/'raw.json'; p.write_text('{"model":"fixed"}')
            copied=Path(f.clone_evidence(p,f.vr.sha(p),old,new))
            self.assertEqual(copied.read_bytes(),p.read_bytes())
            with self.assertRaisesRegex(ValueError,'changed or escaped'):
                f.clone_evidence(p,'wrong',old,new)
            with self.assertRaisesRegex(ValueError,'changed or escaped'):
                f.clone_evidence(p,f.vr.sha(p),new,old)

    def test_derived_result_keeps_original_and_rebases_only_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'old'; new=Path(tmp)/'new'; old.mkdir()
            raw=old/'raw.json';raw.write_text('{}')
            entry={}
            for k in ('request','response','raw_sse'):
                entry[k+'_path']=str(raw);entry[k+'_sha256']=f.vr.sha(raw)
            result={'configuration_sha256':'old','trace':{'response_evidence':[entry]},
                    'diagnostics':[{'message':'unchanged'}], 'first_diagnosis_path':str(raw),
                    'first_diagnosis_sha256':f.vr.sha(raw)}
            p=old/'result.json';f.vr.atomic(p,result); before=p.read_bytes()
            with patch.object(f.base.runtime,'audit_result'):
                derived=f.import_result(p,old,new,'old','new')
            self.assertEqual(p.read_bytes(),before)
            self.assertEqual(derived['diagnostics'],result['diagnostics'])
            self.assertEqual(derived['configuration_sha256'],'new')
            self.assertEqual(derived['reuse_origin']['sha256'],f.vr.sha(p))
            self.assertTrue(Path(derived['first_diagnosis_path']).is_relative_to(new))

    def test_pilot_generation_never_reads_new_full_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'full';out.mkdir()
            row={'id':'six','question':'stem','context':''}; config={'generator':'30B'}
            record={'signature':f.corpus.p.digest([f.cohort.solver_row(row),config]),'model':'30B',
                    'state':'complete','prediction':'ORIGINAL'}
            f.vr.atomic(root/'generated/six.json',record)
            f.vr.atomic(out/'generated/six.json',dict(record,prediction='NEW'))
            plan={'configuration':{'pilot_relative':'..','pilot_ids':['six'],'generator_config':config}}
            self.assertEqual(f.generated(out,plan,row)['prediction'],'ORIGINAL')
            changed=copy.deepcopy(row);changed['question']='different'
            with self.assertRaisesRegex(ValueError,'provenance'):
                f.generated(out,plan,changed)


if __name__ == '__main__': unittest.main()
