import copy
import unittest
import tempfile
from pathlib import Path
import json
from scripts.run_bench_reannotation import evidence, extract, prepare
from scripts.supervise_bench_reannotation import accept
from scripts.experiment_manifest import sha256_file


class AnnotationValidation(unittest.TestCase):
    def empty(self):
        return {"assessment": "no_errors_found", "coverage_notes": "Checked complete input.", "errors": [], "uncertain_scopes": [], "decisions": []}

    def error(self):
        p = self.empty()
        p["assessment"] = "errors_found"
        p["errors"] = [{"root_id": "E1", "claim": "wrong sign", "reason": "negative acceleration",
            "correct_relation": "a=-g", "quote": "a=g", "category": "physics", "premises": "up is positive",
            "status": "confirmed", "depends_on": []}]
        return p

    def test_valid_empty_is_success(self):
        self.assertEqual(evidence(self.empty(), "a=-g", "initial", {})["errors"], [])

    def test_repeated_quote_keeps_all_occurrences(self):
        x = evidence(self.error(), "a=g\n\na=g", "initial", {})["errors"][0]
        self.assertEqual(x["evidence_validation"]["location_status"], "ambiguous")
        self.assertEqual(len(x["evidence_validation"]["occurrences"]), 2)

    def test_invalid_quote_preserves_error_as_unlocated(self):
        p = self.error()
        x = evidence(p, "a=-g", "initial", {})
        self.assertEqual(len(x["errors"]), 1)
        self.assertEqual(x["errors"][0]["evidence_validation"]["location_status"], "unlocated")

    def test_removal_requires_recorded_reason(self):
        with self.assertRaises(ValueError):
            evidence(self.empty(), "a=g", "review", {"initial": self.error()})
        p = self.empty()
        p["decisions"] = [{"source": "initial:E1", "action": "remove", "reason": "axis was downward"}]
        self.assertEqual(evidence(p, "a=g", "review", {"initial": self.error()})["errors"], [])

    def test_unfinished_response_is_not_empty_success(self):
        with self.assertRaises(ValueError):
            extract({"status": "incomplete", "output": []})

    def test_dangling_root_rejected(self):
        p = self.error()
        p["errors"][0]["depends_on"] = ["E2"]
        with self.assertRaises(ValueError):
            evidence(p, "a=g", "initial", {})

    def test_missing_pilot_result_blocks_formal(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "manifest.json").write_text(json.dumps({"pilot_ids": ["p1"]}))
            with self.assertRaises(FileNotFoundError):
                accept(out)

    def test_valid_empty_pilot_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "manifest.json").write_text(json.dumps({"pilot_ids": ["p"+str(i) for i in range(5)]}))
            for stage in ("initial", "blind", "review", "final"):
                for i in range(5):
                    folder = out / ("arm_"+stage) / ("p"+str(i))
                    folder.mkdir(parents=True)
                    raw = folder / "request_001.json"
                    raw.write_text(json.dumps({"response": {"model": "gpt-6.1-sol", "status": "completed", "reasoning": {"effort": "high"}}}))
                    (folder / "result.json").write_text(json.dumps({"response_record": str(raw.relative_to(out)), "response_record_sha256": sha256_file(raw), "annotation": self.empty()}))
            self.assertEqual(accept(out)["pilot_tasks"], 20)
            interim = out / 'arm_initial/p0/result.json'
            item = json.loads(interim.read_text()); item['annotation'] = self.error()
            item['annotation']['errors'][0]['evidence_validation'] = {'exact_match':False}
            interim.write_text(json.dumps(item))
            self.assertEqual(accept(out)['pilot_tasks'],20)
            final = out / 'arm_final/p0/result.json'
            final_item = json.loads(final.read_text()); final_item['annotation'] = item['annotation']
            final.write_text(json.dumps(final_item))
            with self.assertRaisesRegex(ValueError,'no exact original quote'):
                accept(out)
            final_item['annotation']['errors'][0]['quote'] = ''
            final.write_text(json.dumps(final_item))
            self.assertEqual(accept(out)['pilot_tasks'],20)
            final_item['annotation'] = self.empty(); final.write_text(json.dumps(final_item))
            raw.write_text('{}')
            with self.assertRaises(ValueError):
                accept(out)

    def test_recovery_preserves_frozen_manifest_and_rejects_input_drift(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "source.json"
            rows = [{"id": str(i), "question": "q", "prediction": "solution", "answer": "reference", "physics_error_gt": ["must remain hidden"]} for i in range(99)]
            source.write_text(json.dumps(rows))
            out = Path(d) / "output"
            inputs, config = prepare(source, out, "https://example.test/v1")
            self.assertNotIn("physics_error_gt", inputs[0])
            config["runner_sha256"] = "previous-code-hash"
            (out / "manifest.json").write_text(json.dumps(config))
            before = (out / "manifest.json").read_bytes()
            with self.assertRaises(ValueError):
                prepare(source, out, "https://example.test/v1")
            prepare(source, out, "https://example.test/v1", True)
            self.assertEqual((out / "manifest.json").read_bytes(), before)
            rows[0]["prediction"] = "changed input"
            source.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                prepare(source, out, "https://example.test/v1", True)

    def test_stream_records_terminal_usage_and_rejects_missing_terminal(self):
        from unittest.mock import patch
        from scripts.run_retrieval_comparison import Transport
        class Response:
            status_code = 200
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_lines(self):
                return iter(self.lines)
        class Client:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def stream(self, *args, **kwargs): return response
        terminal = {"id":"resp_test", "model":"gpt-6.1-sol", "reasoning":{"effort":"high"},
                    "status":"completed", "usage":{"input_tokens":10,"output_tokens":20}, "output":[]}
        response = Response()
        response.lines = ['data: '+json.dumps({"type":"response.created","response":{"id":"resp_test"}}), '',
                          'data: '+json.dumps({"type":"response.completed","response":terminal}), '']
        with tempfile.TemporaryDirectory() as d, patch('scripts.run_retrieval_comparison.httpx.Client', Client):
            t = Transport(base='https://example.test', key='test', output=d, model='gpt-6.1-sol', reasoning_effort='high',
                          limits={"requests":2,"input_tokens":10000,"output_tokens":1000,"per_response":100,"seconds":100})
            self.assertEqual(t.responses(input='test', stream=True), terminal)
            self.assertEqual(t.output_tokens,20)
            self.assertTrue((Path(d)/'request_001.events.jsonl').exists())
            response.lines = ['data: '+json.dumps({"type":"response.created","response":{"id":"resp_lost"}})]
            with self.assertRaisesRegex(RuntimeError,'without terminal'):
                t.responses(input='test', stream=True)
            self.assertEqual(json.loads((Path(d)/'request_002.json').read_text())["stream_response_id"], 'resp_lost')

    def test_provider_timeout_is_technical_but_auth_and_identity_are_not(self):
        from scripts.run_bench_reannotation import technical_failure
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder/'events.jsonl').write_text(json.dumps({'type':'error','error':{'code':'request_timeout'}})+'\n')
            self.assertTrue(technical_failure({'http_status':200,'stream_events':'events.jsonl'},folder))
            (folder/'events.jsonl').write_text(json.dumps({'type':'error','error':{'code':'server_is_overloaded'}})+'\n')
            self.assertTrue(technical_failure({'http_status':200,'stream_events':'events.jsonl'},folder))
            self.assertFalse(technical_failure({'http_status':429,'stream_events':'events.jsonl'},folder))
            self.assertFalse(technical_failure({'http_status':200,'error':'Provider model identity mismatch'},folder))


if __name__ == "__main__":
    unittest.main()
