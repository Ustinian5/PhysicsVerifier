from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_llm_checker_baseline import extract_json, structured_output_kwargs, baseline_messages, validate_response_diagnostics, run_audited_responses
from scripts.run_llm_checker_baseline import run_tree_review, tree_review_messages, tree_review_tools
from scripts.run_llm_checker_baseline import run_responses_with_recovery


class RunLlmCheckerBaselineTest(unittest.TestCase):
    def test_auto_recovery_switches_key_only_at_confirmed_quota_floor(self):
        from scripts.run_retrieval_comparison import write_json
        for reason, expected in [('platform_quota_low',['primary','backup']),('HTTP 429',['primary'])]:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                run=Path(tmp);env=run/'env';env.write_text('BACKUP=test-only\n')
                args=SimpleNamespace(out_json=str(run/'results.json'),credential='primary',allow_backup=True,
                    drain_transient=True,env_file=env)
                calls=[]
                def invoke(a):
                    calls.append(a.credential)
                    write_json(run/'progress.json',dict(completed=50 if len(calls)>1 else 0,stop_reason=reason))
                with patch('scripts.run_llm_checker_baseline.run_audited_responses',invoke):
                    run_responses_with_recovery(args)
                self.assertEqual(calls,expected)

    def test_paired_rule_comparison_and_unlocated_sensitivity(self):
        from scripts.evaluate_direct_baseline_comparison import paired_bootstrap, all_diagnostics_sensitivity, matched_claim_changes
        score=dict(details=[dict(id='s',gt_error_count=2,matched_error_count=1,pred_locatable_finding_count=2)],
            summary=dict(matched_pred_locatable=1,location_unmatched_pred_findings=1,recall=.5))
        ci=paired_bootstrap(score,{'pure_terra':score},repetitions=100)
        self.assertEqual(ci['delta_f1_percentage_points_ci95']['pure_terra'],[0.,0.])
        sensitive=all_diagnostics_sensitivity(score,[dict(id='s',diagnostics=[{}, {}, {}])])
        self.assertEqual(sensitive['unlocatable_diagnostics'],1)
        self.assertAlmostEqual(sensitive['precision'],1/3)
        self.assertAlmostEqual(sensitive['f1'],.4)
        with self.assertRaisesRegex(ValueError,'different samples'):
            paired_bootstrap(score,{'pure_terra':dict(details=[])},repetitions=1)
        def group(*ids):return dict(details=[dict(id='s',location_matches=[dict(gt_error_id=i) for i in ids])])
        changes=matched_claim_changes(group('a','b'),group('b','c'))
        self.assertEqual([changes[k] for k in ['shared','gained','lost']],[1,1,1])

    def test_tree_review_checks_in_one_session_with_shared_output_budget(self):
        from scripts.run_retrieval_comparison import Transport, TERRA, write_json
        catalog=json.loads((Path(__file__).resolve().parents[1]/'catalogs/rules_unified_3000.json').read_text())
        row=dict(id='test',question='q',prediction='p',answer='SECRET',physics_error_gt=['SECRET'])
        seen=[]
        def response(transport, endpoint, body, **kwargs):
            import copy
            seen.append(copy.deepcopy(body))
            i=len(seen)
            call=dict(type='function_call',call_id=f'c{i}',name='read_tree' if i==1 else 'submit_diagnostics',
                arguments=json.dumps(dict(node_id='root',note='Review relevant branches') if i==1 else dict(diagnostics=[])))
            calls=[call]
            if i==1:calls.append(dict(call,call_id='c1b'))
            data=dict(id=f'resp{i}',model=TERRA,reasoning=dict(effort='medium'),
                usage=dict(input_tokens=100,output_tokens=100),status='completed',output=calls)
            transport.output_tokens += 100
            write_json(transport.output/f'request_{i:03d}.json',dict(response=data,endpoint=endpoint,
                request=body,http_status=200))
            return data
        with tempfile.TemporaryDirectory() as temp:
            tr=Transport(base='https://example.invalid/v1',key='test',output=temp,model=TERRA,
                limits=dict(output_tokens=16384,per_response=16384))
            with patch('scripts.run_retrieval_comparison.Transport._post_once',response):
                diagnostics, trace=run_tree_review(row,catalog,tr,temp)
            self.assertEqual(diagnostics,[])
            self.assertEqual(len(seen),2)
            self.assertEqual(tr.limits['per_response'],16284)
            self.assertIn('function_call_output',[m.get('type') for m in seen[1]['input']])
            self.assertNotIn('SECRET',json.dumps(seen))
            self.assertNotIn('submit_rules',[t['name'] for t in tree_review_tools()])
            self.assertTrue(trace['tool_returned_chars']>0)
            replay=Transport(base='https://example.invalid/v1',key='test',output=Path(temp)/'replay',model=TERRA,
                limits=dict(output_tokens=16384,per_response=16384),replay_source=temp)
            with patch('scripts.run_retrieval_comparison.Transport._post_once',side_effect=AssertionError('No new calls during recovery')):
                again,_=run_tree_review(row,catalog,replay,Path(temp)/'replay')
            self.assertEqual(again,diagnostics)
            self.assertEqual(len(replay.replayed),2)
            self.assertEqual(replay.requests,0)

    def test_tree_final_malformed_diagnosis_is_failure(self):
        from scripts.run_retrieval_comparison import Transport, TERRA
        catalog=json.loads((Path(__file__).resolve().parents[1]/'catalogs/rules_unified_3000.json').read_text())
        data=dict(status='completed',output=[dict(type='function_call',call_id='c',name='submit_diagnostics',arguments='{}')])
        with tempfile.TemporaryDirectory() as temp:
            tr=Transport(base='https://example.invalid/v1',key='test',output=temp,model=TERRA,
                limits=dict(output_tokens=16384,per_response=16384))
            with patch('scripts.run_retrieval_comparison.Transport.responses',return_value=data):
                with self.assertRaisesRegex(ValueError,'Invalid diagnostics'):
                    run_tree_review(dict(question='q',prediction='p'),catalog,tr,temp)

    def test_audited_run_resumes_empty_successes_without_repeating_calls(self):
        from scripts.run_retrieval_comparison import write_json, sha256_file
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); inp=root/"inputs.json"; selection=root/"selection.json"; env=root/"env"
            rows=[dict(id=f"s{i}",question="q",prediction="p") for i in range(50)]
            write_json(inp,rows);write_json(selection,dict(inputs_sha256=sha256_file(inp),selected_ids=[r["id"] for r in rows]))
            env.write_text("ASTRA_BASE_URL=https://example.invalid/v1\nASTRA_API_KEY=unit-test\n")
            args=SimpleNamespace(model="gpt-5.6-terra",selection_manifest=selection,input=str(inp),env_file=env,
                credential="primary",out_json=str(root/"out"/"results.json"),reasoning_effort="medium",workers=3,limit=50)
            calls=[]
            def response(transport,**kwargs):
                calls.append(kwargs)
                data={"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":'{"diagnostics":[]}'}]}]}
                write_json(transport.output/"request_001.json",dict(http_status=200,request=dict(max_output_tokens=16384),response=data))
                return data
            with patch("scripts.run_retrieval_comparison.Transport.responses",response):
                run_audited_responses(args)
                run_audited_responses(args)
            self.assertEqual(len(calls),50)
            progress=json.loads((root/"out"/"progress.json").read_text())
            self.assertEqual(progress["completed"],50)
            self.assertEqual(progress["failed"],0)

    def test_baseline_never_sends_gold_or_rules(self):
        clean = {"question": "q", "prediction": "s", "context": {"unit": "SI"}}
        contaminated = dict(clean, answer="SECRET_REFERENCE", physics_error_gt=["SECRET_GT"], rules=["SECRET_RULE"])
        self.assertEqual(baseline_messages(clean), baseline_messages(contaminated))

    def test_incomplete_response_is_failure_even_with_parseable_empty_list(self):
        with self.assertRaises(ValueError):
            validate_response_diagnostics({"status": "incomplete", "output": [{"type":"message", "content":[{"type":"output_text", "text":'{"diagnostics":[]}'}]}]})

    def test_bad_diagnostic_is_not_silently_discarded(self):
        for text in ['{}', '{"diagnostics":[{"message":"error","evidence":{}}]}']:
            with self.assertRaises(ValueError):
                validate_response_diagnostics({"status":"completed", "output":[{"type":"message","content":[{"type":"output_text","text":text}]}]})

    def test_genuine_empty_diagnosis_is_success(self):
        self.assertEqual([], validate_response_diagnostics({"status":"completed", "output":[{"type":"message","content":[{"type":"output_text","text":'{"diagnostics":[]}'}]}]}))

    def test_parse_failure_is_not_silently_treated_as_empty_diagnostics(self) -> None:
        parsed = extract_json('{"diagnostics":[{"message":"bad \\q"}]}')

        self.assertIn("_parse_error", parsed)

    def test_forced_tool_call_has_required_diagnostics_schema(self) -> None:
        kwargs = structured_output_kwargs("forced_tool_call")

        self.assertEqual("submit_diagnostics", kwargs["tool_choice"]["function"]["name"])
        parameters = kwargs["tools"][0]["function"]["parameters"]
        self.assertIn("diagnostics", parameters["required"])


if __name__ == "__main__":
    unittest.main()
