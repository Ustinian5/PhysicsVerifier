"""Offline regressions for actual interruption modes and immutable label replay."""
import copy
import functools
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from scripts.bench_reannotation_runtime import Circuit, failure_kind, request_statistics, execution_limits, provider_stream_attempts
from scripts.run_bench_reannotation import MODEL, SYSTEM, evidence, run, prepare, task_body
from scripts.run_retrieval_comparison import canonical
from scripts.experiment_manifest import sha256_file
from scripts.run_retrieval_comparison import Transport, BatchBudget, write_json
from scripts.response_stream import sse_payloads

EMPTY = {"assessment": "no_errors_found", "coverage_notes": "All supplied steps checked.",
         "errors": [], "uncertain_scopes": [], "decisions": []}


def terminal(text=None):
    return {"id": "resp_fixture", "model": MODEL, "reasoning": {"effort": "high"},
            "status": "completed", "usage": {"input_tokens": 20, "output_tokens": 30},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text or json.dumps(EMPTY)}]}]}


def sse_response(events):
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                          text="".join("data: " + json.dumps(e) + "\n\n" for e in events))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def client(self, handler):
        factory = functools.partial(httpx.Client, transport=httpx.MockTransport(handler))
        return patch("scripts.run_retrieval_comparison.httpx.Client", factory)

    def transport(self):
        return Transport(base="https://fixture.invalid/v1", key="test-only", output=self.out,
                         model=MODEL, reasoning_effort="high",
                         limits={"requests": 10, "input_tokens": 100000, "output_tokens": 100000,
                                 "per_response": 32768, "seconds": 900})

    def record(self, **fields):
        data = {"started_unix": time.time()-10, "elapsed_seconds": 1,
                "request": {"stream": True, "max_output_tokens": 32768}, **fields}
        path = self.out / f"request_{len(list(self.out.glob('request_*.json')))+1:03d}.json"
        write_json(path, data)
        return path, data

    def test_multiline_sse_and_comments(self):
        data = list(sse_payloads([": ping", "event: sample", 'data: {"a":', 'data: 1}', "", "data: [DONE]", ""]))
        self.assertEqual(json.loads(data[0]), {"a": 1})
        self.assertEqual(data[1], "[DONE]")

    def test_http_error_body_and_retry_after_survive(self):
        def handler(request):
            return httpx.Response(429, headers={"retry-after": "456"},
                                  json={"error": {"code": "server_is_overloaded", "message": "capacity"}})
        with self.client(handler), self.assertRaises(httpx.HTTPStatusError):
            self.transport().responses(input="json", stream=True)
        record = json.loads((self.out/"request_001.json").read_text())
        self.assertEqual(record["retry_after"], "456")
        self.assertEqual(record["response"]["error"]["code"], "server_is_overloaded")
        self.assertEqual(failure_kind(record, self.out), "upstream_overload")

    def test_html_error_body_is_preserved(self):
        with self.client(lambda r: httpx.Response(503, text="<html>capacity</html>")), self.assertRaises(httpx.HTTPStatusError):
            self.transport().responses(input="json", stream=True)
        self.assertEqual(json.loads((self.out/"request_001.json").read_text())["response_text"], "<html>capacity</html>")

    def test_in_stream_error_is_distinct_from_truncated_json(self):
        events = [{"type": "response.created", "response": {"id": "resp_failed"}},
                  {"type": "error", "error": {"code": "server_is_overloaded", "type": "service_unavailable_error"}}]
        with self.client(lambda r: sse_response(events)), self.assertRaisesRegex(RuntimeError, "in-stream error"):
            self.transport().responses(input="json", stream=True)
        record = json.loads((self.out/"request_001.json").read_text())
        self.assertEqual(record["stream_response_id"], "resp_failed")
        self.assertEqual(record["stream_event_count"], 2)
        self.assertEqual(failure_kind(record, self.out), "upstream_overload")

    def test_completed_stream_retains_usage_and_identity(self):
        response = terminal()
        with self.client(lambda r: sse_response([{"type": "response.completed", "response": response}])):
            t = self.transport()
            self.assertEqual(t.responses(input="json", stream=True), response)
            self.assertEqual(t.output_tokens, 30)

    def test_gateway_buffered_json_response_is_not_lost_or_retried(self):
        response=terminal()
        with self.client(lambda r:httpx.Response(200,json=response)):
            self.assertEqual(self.transport().responses(input='json',stream=True),response)
        self.assertEqual(json.loads((self.out/'request_001.json').read_text())['response'],response)

    def test_malformed_sse_is_saved_and_never_retried_as_transport(self):
        with self.client(lambda r: httpx.Response(200, headers={'content-type':'text/event-stream'}, text='data: {bad\n\n')), self.assertRaises(ValueError):
            self.transport().responses(input="json", stream=True)
        record = json.loads((self.out/"request_001.json").read_text())
        self.assertIsNone(failure_kind(record, self.out))
        self.assertIn("{bad", (self.out/record["stream_events"]).read_text())

    def test_auth_quota_unknown_429_incomplete_and_identity_are_not_retryable(self):
        examples = [
            {"http_status": 429}, {"http_status": 401}, {"http_status": 402},
            {"http_status": 503, "response": {"error": {"type": "invalid_request_error"}}},
            {"http_status": 200, "stream_error": {"type": "rate_limit_error"}},
            {"http_status": 200, "response": {"status": "incomplete"}},
            {"http_status": 200, "response": terminal(), "error": "Provider model identity mismatch"},
        ]
        for item in examples:
            with self.subTest(item=item):
                self.assertIsNone(failure_kind(item, self.out))

    def test_all_technical_errors_activate_persistent_global_cooldown(self):
        circuit = Circuit(self.out)
        cases = [dict(http_status=503), dict(http_status=200, stream_error={"code": "server_is_overloaded"}),
                 dict(error_type="ConnectError", retryable_transport_error=True),
                 dict(http_status=200, stream_error={"code": "request_timeout"})]
        for i, case in enumerate(cases, 1):
            path, rec = self.record(**case)
            circuit.observe(path, rec)
            restored = Circuit(self.out)
            self.assertEqual(restored.state["consecutive_technical_failures"], i)
            self.assertGreater(restored.state["next_request_unix"], time.time()+250)
        same_state = copy.deepcopy(circuit.state)
        circuit.observe(path, rec)
        self.assertEqual(circuit.state, same_state)
        path, rec = self.record(http_status=200, response=terminal())
        circuit.observe(path, rec)
        self.assertEqual(circuit.state["consecutive_technical_failures"], 0)

    def test_retry_after_is_not_truncated_by_backoff(self):
        path, rec = self.record(http_status=503, retry_after="3600")
        circuit = Circuit(self.out)
        circuit.observe(path, rec)
        self.assertGreater(circuit.state["next_request_unix"], time.time()+3500)

    def test_restart_replays_failure_after_last_scheduler_checkpoint(self):
        circuit=Circuit(self.out)
        p, good=self.record(http_status=200,response=terminal())
        circuit.observe(p,good)
        q, bad=self.record(http_status=503)
        restored=Circuit(self.out);restored.initialize_from_history([(p,good),(q,bad)])
        self.assertEqual(restored.state['last_request'],str(q))
        self.assertGreater(restored.state['next_request_unix'],time.time()+250)

    def test_terminal_server_failure_is_retryable_even_without_usage(self):
        rec={'http_status':200,'error':'Missing usage; cannot enforce budget','response':{'status':'failed','error':{'code':'request_timeout'}}}
        self.assertEqual(failure_kind(rec,self.out),'upstream_timeout')

    def test_real_gateway_ticket_timeout_inside_http_200_is_technical(self):
        rec={'http_status':200,'stream_error':{'code':'ticket_wait_timeout','message':'Ticket gateway: ticket_wait_timeout','status_code':503,'type':'server_error'}}
        self.assertEqual(failure_kind(rec,self.out),'upstream_queue_timeout')
        rec['stream_error']['type']='invalid_request_error'
        self.assertIsNone(failure_kind(rec,self.out))

    def test_unknown_usage_and_failure_attempts_are_not_cleared(self):
        folder = self.out/"arm_initial"/"x";folder.mkdir(parents=True)
        write_json(folder/"request_001.json", {"request": {"max_output_tokens": 32768}, "error_type": "ConnectError"})
        budget = BatchBudget(self.out, {"requests":1200,"input_tokens":80000000,"output_tokens":40000000})
        self.assertEqual((budget.requests, budget.unknown_usage, budget.output_tokens), (1, 1, 32768))
        rec = [(folder/"request_001.json", json.loads((folder/"request_001.json").read_text()))]*889
        stats = request_statistics(rec,47,{"max_requests":1200,"formal_tasks":396})
        self.assertEqual(stats["minimum_request_shortfall"],38)

    def test_connection_credit_is_frozen_once_not_expanded_on_restart(self):
        write_json(self.out/'manifest.json', {'frozen':True})
        p, rec = self.record(error_type='ConnectError')
        cfg={'max_requests':1200,'max_input_tokens':80000000,'max_output_tokens':40000000}
        self.assertEqual(execution_limits(self.out,cfg,[(p,rec)],create=True)['requests'],1201)
        second, rec2 = self.record(error_type='ConnectTimeout')
        self.assertEqual(execution_limits(self.out,cfg,[(p,rec),(second,rec2)],create=True)['requests'],1201)
        self.assertEqual(provider_stream_attempts([(p,rec),(second,rec2)]),0)
        rec3={**rec2,'http_status':200,'stream_response_id':'resp_started','retryable_transport_error':True,'error_type':'RemoteProtocolError'}
        self.assertEqual(provider_stream_attempts([(p,rec),(second,rec3)]),1)
        rec['response']={'status':'completed'};write_json(p,rec)
        with self.assertRaisesRegex(ValueError,'credit evidence'):
            execution_limits(self.out,cfg,[(p,rec)])

    def test_dependency_cycle_is_not_silently_accepted(self):
        a = copy.deepcopy(EMPTY);a['assessment']='errors_found'
        error = dict(root_id="E1",claim="a",reason="b",correct_relation="c",quote="x",category="physics",premises="",status="confirmed",depends_on=["E2"])
        a['errors']=[error, {**error, 'root_id':'E2','depends_on':['E1']}]
        with self.assertRaisesRegex(ValueError,'cyclic'):
            evidence(a,'x','initial',{})

    def setup_pilot(self):
        source=self.out/'source.json'
        source.write_text(json.dumps([{'id':str(i),'question':'question','prediction':'solution','answer':'reference','physics_error_gt':['MUST NOT BE SENT']} for i in range(99)]))
        return SimpleNamespace(env=self.out/'unused.env',source=source,output=self.out/'run',mode='pilot',recover_technical=True)

    def test_pilot_empty_results_complete_once_and_resume_without_calls(self):
        args=self.setup_pilot(); calls=[]
        def handler(request):
            body=json.loads(request.content);calls.append(body)
            task=json.loads(body['input'])
            self.assertNotIn('physics_error_gt',task['input'])
            if task['task'].startswith('In this fresh'):
                self.assertEqual(task['previous_annotations'],{})
            return sse_response([{'type':'response.completed','response':terminal()}])
        with patch.dict(os.environ,OPENAI_BASE_URL='https://fixture.invalid/v1',OPENAI_API_KEY='test-only'), self.client(handler), patch('scripts.run_bench_reannotation.platform_quota_probe',return_value={'total_available':100000000,'unlimited_quota':False}):
            self.assertEqual(run(args),0)
            before={str(p):p.read_bytes() for p in args.output.glob('arm*/*/result.json')}
            self.assertEqual(len(before),20)
            self.assertEqual(run(args),0)
            self.assertEqual(len(calls),20)
            self.assertEqual(before,{str(p):p.read_bytes() for p in args.output.glob('arm*/*/result.json')})
            # Any edited result must fail before another model call.
            p=next(args.output.glob('arm_initial/*/result.json'));d=json.loads(p.read_text());d['annotation']['coverage_notes']='tampered';write_json(p,d)
            with self.assertRaisesRegex(ValueError,'differs from raw'):
                run(args)
            self.assertEqual(len(calls),20)

    def test_invalid_completed_content_stops_without_regeneration_on_resume(self):
        args=self.setup_pilot();calls=[]
        def handler(request):
            calls.append(1)
            return sse_response([{'type':'response.completed','response':terminal('{bad')}])
        with patch.dict(os.environ,OPENAI_BASE_URL='https://fixture.invalid/v1',OPENAI_API_KEY='test-only'), self.client(handler), patch('scripts.run_bench_reannotation.platform_quota_probe',return_value={'total_available':100000000,'unlimited_quota':False}):
            for _ in range(2):
                with self.assertRaises(ValueError):run(args)
            self.assertEqual(len(calls),1)

    def test_completed_response_survives_crash_before_result_write(self):
        args=self.setup_pilot();rows,cfg=prepare(args.source,args.output,'https://fixture.invalid/v1')
        row=next(r for r in rows if r['id'] in cfg['pilot_ids'])
        folder=args.output/'arm_initial'/row['id'];folder.mkdir(parents=True)
        raw=folder/'request_001.json'
        write_json(raw,{'endpoint':'/responses','started_unix':time.time()-100,'elapsed_seconds':1,'http_status':200,
                        'request':{'instructions':SYSTEM,'input':canonical(task_body('initial',row,{})),
                                   'text':{'format':{'type':'json_object'}},'stream':True,'model':MODEL,
                                   'reasoning':{'effort':'high'},'max_output_tokens':32768,'store':False},
                        'response':terminal()})
        before=raw.read_bytes();calls=[]
        def handler(request):
            calls.append(1);return sse_response([{'type':'response.completed','response':terminal()}])
        with patch.dict(os.environ,OPENAI_BASE_URL='https://fixture.invalid/v1',OPENAI_API_KEY='test-only'), self.client(handler), patch('scripts.run_bench_reannotation.platform_quota_probe',return_value={'total_available':100000000,'unlimited_quota':False}):
            self.assertEqual(run(args),0)
        self.assertEqual(len(calls),19)
        self.assertEqual(raw.read_bytes(),before)
        self.assertTrue((folder/'result.json').exists())

    def test_overload_enters_batch_cooldown_before_any_different_task(self):
        args=self.setup_pilot();calls=[]; observed=[]
        def handler(request):
            calls.append(1)
            return sse_response([{'type':'error','error':{'code':'server_is_overloaded'}}])
        def stop_during_wait(seconds):
            observed.append(json.loads((args.output/'progress.json').read_text()))
            (args.output/'STOP').write_text('test graceful stop')
        with patch.dict(os.environ,OPENAI_BASE_URL='https://fixture.invalid/v1',OPENAI_API_KEY='test-only'), self.client(handler), patch('scripts.run_bench_reannotation.platform_quota_probe',return_value={'total_available':100000000,'unlimited_quota':False}), patch('scripts.run_bench_reannotation.time.sleep',stop_during_wait):
            self.assertEqual(run(args),0)
        self.assertEqual(len(calls),1)
        self.assertEqual(observed[0]['state'],'cooling_down')
        self.assertGreater(observed[0]['next_request_unix'],time.time()+250)
        self.assertEqual(json.loads((args.output/'progress.json').read_text())['state'],'operator_stopped')

    def test_old_balance_only_429_receipt_cannot_authorize_a_new_generation(self):
        args=self.setup_pilot();rows,cfg=prepare(args.source,args.output,'https://fixture.invalid/v1')
        row=next(r for r in rows if r['id'] in cfg['pilot_ids'])
        folder=args.output/'arm_initial'/row['id'];folder.mkdir(parents=True)
        raw=folder/'request_001.json'
        write_json(raw,{'endpoint':'/responses','started_unix':time.time()-3600,'elapsed_seconds':1,'http_status':429,
                        'request':{'instructions':SYSTEM,'input':canonical(task_body('initial',row,{})),
                                   'text':{'format':{'type':'json_object'}},'stream':True,'model':MODEL,
                                   'reasoning':{'effort':'high'},'max_output_tokens':32768,'store':False},
                        'error':'unknown historical 429'})
        write_json(args.output/'rate_limit_recovery.json',{'request_record_sha256':[sha256_file(raw)],'balance_usd':50})
        calls=[]
        def handler(request):
            calls.append(json.loads(request.content))
            return sse_response([{'type':'response.completed','response':terminal()}])
        with patch.dict(os.environ,OPENAI_BASE_URL='https://fixture.invalid/v1',OPENAI_API_KEY='test-only'), self.client(handler), patch('scripts.run_bench_reannotation.platform_quota_probe',return_value={'total_available':100000000,'unlimited_quota':False}):
            self.assertEqual(run(args),2)
        self.assertEqual(len(calls),17)
        self.assertFalse((folder/'result.json').exists())
        status=json.loads((args.output/'progress.json').read_text())
        self.assertEqual(status['state'],'stopped_needs_review')
        self.assertEqual(status['held_tasks'][0]['id'],row['id'])


if __name__ == '__main__':
    unittest.main()
