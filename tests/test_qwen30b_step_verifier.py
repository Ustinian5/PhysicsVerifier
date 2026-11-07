import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_qwen30b_step_verifier as s
from scripts.serve_fixed_date_vllm import pinned_system_message
from scripts import run_qwen30b_step_checks as checks
from scripts import run_bench_reannotation as annotation


class StepVerifierTests(unittest.TestCase):
    def test_prepared_checker_uses_continuous_packet_publication_and_bound_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();catalog=root/'catalog.json'
            s.vr.atomic(catalog,{'domains':[]})
            rows=[{'id':str(i),'question':'Q','context':'','prediction':'new'} for i in range(6)]
            s.vr.atomic(source/'inputs.json',rows)
            questions=[{k:x[k] for k in ('id','question','context')} for x in rows]
            s.vr.atomic(source/'questions.json',questions)
            ids=[x['id'] for x in rows]
            config={'source_files':{s.BUILD+'/catalog.json':s.vr.sha(catalog)},'generator':s.corpus.MODEL,
                    'pilot_ids':ids,'sample_ids':ids,'fixed_harmony_date':'2026-11-07'}
            s.vr.atomic(source/'plan.json',{'configuration':config})
            s.vr.atomic(source/'catalog_isolation.json',{'passed':True,'catalog_sha256':s.vr.sha(catalog),
                'questions_sha256':s.vr.sha(source/'questions.json')})
            s.vr.atomic(source/'original_source_mapping.json',{'matched':6,
                'questions_sha256':s.vr.sha(source/'questions.json'),
                'records':[{'id':sid,'original_pool_ids':['source'+sid]} for sid in ids]})
            for row,q in zip(rows,questions):
                raw=source/'raw'/(row['id']+'.json');s.vr.atomic(raw,{'model':s.corpus.MODEL})
                s.vr.atomic(source/'generated'/(row['id']+'.json'),{'model':s.corpus.MODEL,'prediction':'new',
                    'state':'complete','signature':s.corpus.p.digest([s.solver_row(q),config]),
                    'evidence':{'file':str(raw.relative_to(source)),'sha256':s.vr.sha(raw)}})
            with patch.object(s,'verify',return_value={'configuration':config}):
                checks.prepare(root/'checks',source,catalog)
                actual=s.vr.read(root/'checks/plan.json')['configuration']
                self.assertEqual(actual['limits']['review_protocol'],checks.base.old.continuous.PROTOCOL)
                self.assertEqual(actual['arms'],['pure','rule_tree'])
                altered=s.vr.read(source/'generated/0.json');altered['signature']='old4b'
                s.vr.atomic(source/'generated/0.json',altered)
                with self.assertRaisesRegex(ValueError,'binding drift'):
                    checks.prepare(root/'bad_checks',source,catalog)

    def test_new_scoring_counts_quote_at_first_character(self):
        from scripts.evaluate_physics_eval_sets import _match_by_location
        gt=[{'error_id':'E1','span_valid':True,'start_char':0,'end_char':3}]
        pred=[{'span_valid':True,'start_char':0,'end_char':3}]
        self.assertFalse(_match_by_location(gt,pred,.5,.6)[0])
        corrected=_match_by_location(gt,pred,.5,.6,preserve_zero_offsets=True)
        self.assertEqual(corrected[0],{'E1'})
        self.assertEqual(corrected[1][0]['pred_span'],[0,3])

    def test_unlocated_and_uncertain_reference_not_dropped(self):
        rows=[{'id':'p','question':'Q','context':'','prediction':'a=g then a=g'}]
        errors=[{'root_id':'E1','status':'confirmed','claim':'wrong','quote':'a=g'},
                {'root_id':'E2','status':'confirmed','claim':'omitted','quote':''},
                {'root_id':'E3','status':'unresolved','claim':'unknown','quote':'then'}]
        dataset,unknown,counts=checks.reference_rows(rows,{'p':{'errors':errors}})
        self.assertEqual(len(dataset[0]['physics_error_gt']),2)
        self.assertTrue(all(not x['span_valid'] for x in dataset[0]['physics_error_gt']))
        self.assertEqual(len(unknown['p']),1)
        self.assertEqual(counts['confirmed_unlocated'],2)
        with self.assertRaises(ValueError):checks.reference_rows(rows,{})

    def test_actual_server_date_is_checked_from_tokens(self):
        class Tokenizer:
            def decode(self, tokens, **kwargs):
                return '<|start|>system<|message|>Identity\nCurrent date: 2026-11-07\nReasoning: medium\n<|end|>'
        raw={'prompt_token_ids':[1,2,3]}
        self.assertEqual(checks.validate_effective_input(raw,Tokenizer(),'2026-11-07')['token_count'],3)
        with self.assertRaises(ValueError):checks.validate_effective_input(raw,Tokenizer(),'2026-11-08')
        with self.assertRaises(ValueError):checks.validate_effective_input({},Tokenizer(),'2026-11-07')

    def test_six_question_annotation_never_reuses_old_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.json'
            source.write_text(json.dumps([{'id':str(i),'question':'q','prediction':'new30b',
                'answer':'reference','physics_error_gt':['old4b']} for i in range(6)]))
            rows,config=annotation.prepare(source,root/'labels','https://example.test',expected_count=6,pilot_size=6)
            self.assertEqual(config['formal_tasks'],24)
            self.assertEqual(len(config['pilot_ids']),6)
            self.assertTrue(all('physics_error_gt' not in x for x in rows))
            with self.assertRaises(ValueError):
                annotation.prepare(source,root/'labels','https://example.test',expected_count=99)

    def test_server_date_is_fixed_for_default_and_positional_calls(self):
        def original(model=None, start_date=None):return start_date
        pinned=pinned_system_message(original,'2026-11-07')
        self.assertEqual(pinned(),'2026-11-07')
        self.assertEqual(pinned('m',None),'2026-11-07')
        with self.assertRaises(ValueError):pinned(start_date='2026-11-08')
    def test_solver_excludes_labels_and_old_response(self):
        row={'id':'p', 'question':'Q', 'context':'C', 'answer':'SECRET',
             'prediction':'OLD', 'physics_error_gt':[{'claim':'SECRET'}]}
        visible=s.corpus.solver_messages(s.solver_row(row))
        self.assertEqual(visible[-1]['content'], 'C\n\nQ')
        self.assertNotIn('SECRET',str(visible)); self.assertNotIn('OLD',str(visible))

    def test_missing_context_is_not_invented(self):
        self.assertEqual(s.solver_row({'id':'p','question':'Q'}),{'id':'p','question':'Q'})

    def test_export_binds_answer_to_generator_and_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'generated').mkdir()
            row={'id':'p','question':'Q','context':''}; config={'generator':s.corpus.MODEL}
            s.vr.atomic(root/'questions.json',[row])
            s.vr.atomic(root/'reference_answers.json',[{'id':'p','historical_reference_answer':'SECRET'}])
            record={'id':'p','model':s.corpus.MODEL,'state':'complete','prediction':'new answer',
                    'signature':s.corpus.p.digest([s.solver_row(row),config])}
            s.vr.atomic(root/'generated/p.json',record)
            rows=s.export_generated(root,{'configuration':config})
            self.assertNotIn('SECRET',json.dumps(rows))
            self.assertEqual(s.vr.read(root/'annotation_inputs.json')[0]['answer'],'SECRET')
            record['model']='wrong'; s.vr.atomic(root/'generated/p.json',record)
            with self.assertRaises(ValueError):s.export_generated(root,{'configuration':config})

    def test_truncation_is_not_success_or_reference_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'generated').mkdir()
            row={'id':'p','question':'Q','context':''}; config={'generator':s.corpus.MODEL}
            s.vr.atomic(root/'questions.json',[row])
            s.vr.atomic(root/'reference_answers.json',[{'id':'p','historical_reference_answer':'A'}])
            s.vr.atomic(root/'generated/p.json',{'model':s.corpus.MODEL,'state':'incomplete',
                'signature':s.corpus.p.digest([s.solver_row(row),config]),'prediction':'partial'})
            self.assertEqual(s.export_generated(root,{'configuration':config}),[])
            self.assertEqual(len(s.vr.read(root/'generation_summary.json')['failed']),1)


if __name__=='__main__':unittest.main()
