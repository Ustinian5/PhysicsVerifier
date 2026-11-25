import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_corrected_extraction as e
from tests.test_experience_first_120b import ROWS, TAXONOMY


class ExtractionTests(unittest.TestCase):
    def source_fixture(self, drift=False):
        tasks=[{'role':'build','row':{'id':str(i),'question':'q'+str(i),'answer':'SECRET'}} for i in range(2000)]
        tasks.append({'role':'heldout','row':{'id':'test','question':'DO NOT READ'}})
        config={'generator_config':{'fixed':'sampling'}}
        def read(path):
            if path.name=='tasks.json':return tasks
            if path.name=='execution_status.json':return {'state':'generation_finished'}
            i=int(path.stem);row=tasks[i]['row']
            return {'signature':'drift' if drift else e.digest([row,config['generator_config']]),
                    'state':'incomplete' if i==1999 else 'complete','finish_reason':'length' if i==1999 else 'stop',
                    'prediction':'student','evidence':{'file':'raw/'+str(i),'sha256':'hash'}}
        return config,read

    def test_frozen_build_only_and_no_reference_leak(self):
        config,read=self.source_fixture()
        with patch.object(e.c,'verify',return_value={'configuration':config}),patch.object(e,'read',side_effect=read),\
                patch.object(e,'sha',return_value='hash'),patch.object(e.c.corpus,'evidence_valid') as check:
            rows,failed,proof=e.build_rows(Path('/source'))
        self.assertEqual(len(rows),1999)
        self.assertEqual(failed,[{'id':'1999','state':'incomplete','finish_reason':'length'}])
        self.assertEqual(check.call_count,2000)
        self.assertEqual(len(proof),4000)
        self.assertTrue(all(set(x)=={'id','question','prediction','context','answer'} and x['answer']==x['context']=='' for x in rows))
        self.assertNotIn('SECRET',str(rows));self.assertNotIn('DO NOT READ',str(rows))

    def test_generation_signature_drift_stops(self):
        config,read=self.source_fixture(drift=True)
        with patch.object(e.c,'verify',return_value={'configuration':config}),patch.object(e,'read',side_effect=read):
            with self.assertRaisesRegex(ValueError,'Generation drift'):e.build_rows(Path('/source'))

    def test_prepared_input_tamper_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'construction';source=Path(temp)/'source'
            e.write(source/'plan.json',{'frozen':True})
            with patch.object(e,'build_rows',return_value=([{'id':'a'}],[],{})):
                e.prepare(out,source)
            e.verify(out)
            e.write(out/'inputs.json',[{'id':'tampered'}])
            with self.assertRaisesRegex(ValueError,'Input drift'):e.verify(out)

    def test_new_and_historical_plans_use_their_own_review_and_repair_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source'
            output = Path(temp) / 'construction'
            e.write(source / 'plan.json', {'frozen': True})
            with patch.object(e, 'build_rows', return_value=(ROWS[:1], [], {})):
                e.prepare(output, source)
            config = e.verify(output)['configuration']

        original = dict(sample=ROWS[0], taxonomy=TAXONOMY, max_rules_per_sample=2)
        repair = dict(original=original, draft={'_unparsed_response_text': 'bad JSON'},
                      validation_error='Invalid error ID/type')
        archive = e.old.Journal.__new__(e.old.Journal)
        current_system = e.AUDIT_PROMPT_FILE.read_text(encoding='utf-8').rstrip('\n')
        self.assertEqual(config['audit_system_prompt'], current_system)
        self.assertNotEqual(current_system, archive.messages('extract_experience', original)[0]['content'])
        # Once prepared, even a repair must take its prompt from the plan, not from a live file.
        with patch.object(e, 'AUDIT_PROMPT_FILE', output / 'absent.txt'):
            for stage, payload in [('extract_experience', original), ('repair_experience', repair)]:
                old_messages = archive.messages(stage, payload)
                self.assertEqual(e.request_messages(stage, payload, {}), old_messages)
                self.assertEqual(e.request_messages(stage, payload, config),
                                 [{'role': 'system', 'content': current_system}, *old_messages[1:]])
            self.assertEqual(e.request_messages('taxonomy', TAXONOMY, config),
                             archive.messages('taxonomy', TAXONOMY))

    def test_original_prompt_and_received_prefix_replay(self):
        payload={'domains':[{'id':'d','name':'Physics','topics':[]}]}
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);cfg={'model':'fixed-model','max_tokens':16384,'fixed_date':'2026-11-07',
                              'max_request_input_bytes':400000}
            journal=e.c.corpus.Journal(out,cfg,None)
            messages=e.old.flow.experience_120b_messages('taxonomy',payload)
            body=e.c.corpus.r.request_body(cfg['model'],messages,16384,backend='vllm')
            body.update(stream=False,return_token_ids=True,reasoning_effort='medium');body.pop('stream_options',None)
            raw={'model':cfg['model'],'content':'placeholder'}
            path=out/'raw'/'taxonomy'/e.digest('d')[:24]/'request_001.json'
            e.write(path,{'status':'received','fingerprint':e.digest(['d',body,cfg]),'response':raw})
            with patch.object(e.c.corpus.harmony,'decode_response',return_value={'content':'{"domains":[]}'}),\
                    patch.object(e,'validate_effective_input',return_value={'passed':True}),\
                    patch.object(e.c.corpus.httpx,'Client',side_effect=AssertionError('Must not request again')):
                value,evidence=e.call(out,journal,None,'taxonomy','d',payload)
            self.assertEqual(value,{'domains':[]});self.assertEqual(evidence['sha256'],e.sha(path))


if __name__=='__main__':unittest.main()
