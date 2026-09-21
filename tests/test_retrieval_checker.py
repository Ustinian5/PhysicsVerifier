import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from scripts.experiment_manifest import sha256_file
from scripts.run_retrieval_checker import FrozenVerifier, SharedResponses, job_key, matcher_result, read_frozen_jobs, run_job
from scripts.run_retrieval_comparison import ASTRA, TERRA, QWEN, write_json


class RetrievalCheckerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rule = {"rule_id": "test_rule", "title": "Units", "summary": "Compare units", "trigger": "A velocity equation", "check_logic": "Compare both sides", "error_type": "dimension", "symbolic_hint": ""}
        self.catalog = {"metadata": {"catalog_type": "unified_rules_v2"}, "domains": [{"id": "d", "name": "Mechanics", "topics": [{"id": "t", "name": "Units", "rules": [self.rule]}]}]}
        self.catalog_path = self.root / "catalog.json"
        write_json(self.catalog_path, self.catalog)
        self.sample = {"id": "case", "question": "Calculate a velocity.", "prediction": "The velocity is v = 2 m.", "context": "", "answer": "SECRET_REFERENCE", "physics_error_gt": ["SECRET_LABEL"]}
        selected = [{"rule_id": "test_rule", "score": 0.7, "reason": "ASTRA_PREJUDGMENT", "answer_quote": "v = 2 m"}]
        self.autonomous = {"status": "completed", "candidate_source": "autonomous_leaf_tools", "selected_rules": selected, "read_rule_ids": ["test_rule"]}
        self.native = {"status": "completed", "candidate_source": "native_semantic_tree", "selected_rules": selected, "native_result": {"selected_rules": [{"rule_id": "test_rule", "score": 0.7, "reason": "NATIVE_PREJUDGMENT", "domain": "Mechanics", "topic": "Units", "topic_id": "t", "rule_obj": self.rule}]}}
        self.requests = []
        self.payload = [{"rule": "test_rule", "severity": "error", "message": "The velocity unit is inconsistent.", "evidence": {"quote": "v = 2 m"}}]
        original_client = httpx.Client

        def respond(request):
            self.requests.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "test_response", "object": "chat.completion", "created": 1, "model": QWEN, "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(self.payload)}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}})

        client_patch = patch("scripts.run_retrieval_comparison.httpx.Client", side_effect=lambda **kw: original_client(transport=httpx.MockTransport(respond), **kw))
        client_patch.start()
        self.addCleanup(client_patch.stop)

    def transport(self, name):
        return SharedResponses(base="https://unit-test.invalid/v1", key="test_key", output=self.root / name, model=QWEN, response_index=self.root / "response_index.json", limits={"requests": 6, "input_tokens": 600000, "output_tokens": 12288, "seconds": 1800})

    def test_same_checker_input_and_publication_across_retrieval_sources(self):
        a, b = self.transport("native"), self.transport("leaf")
        first = run_job(self.sample, self.native, catalog_path=self.catalog_path, output=a.output, transport=a)
        second = run_job(self.sample, self.autonomous, catalog_path=self.catalog_path, output=b.output, transport=b)
        self.assertEqual((first["status"], second["status"]), ("completed", "completed"))
        self.assertEqual((first["counts"]["released_diagnostics"], second["counts"]["released_diagnostics"]), (1, 1))
        self.assertEqual((len(self.requests), b.requests, len(b.reused)), (1, 0, 1))
        prompt = json.dumps(self.requests[0])
        for secret in ("SECRET_REFERENCE", "SECRET_LABEL", "ASTRA_PREJUDGMENT", "NATIVE_PREJUDGMENT"):
            self.assertNotIn(secret, prompt)
        self.assertEqual(second["trace"]["selection_strategy"], "autonomous_leaf_tools")
        self.assertEqual(second["trace"]["diagnostics"][0]["rule_match"]["retrieval_strategy"], "autonomous_leaf_tools")
        self.assertEqual(second["trace"]["diagnostics"][0]["rule_match"]["score"], 0.7)

    def test_rule_and_location_gates_still_apply(self):
        self.rule["publishable"] = False
        write_json(self.catalog_path, self.catalog)
        t = self.transport("gate")
        result = run_job(self.sample, self.autonomous, catalog_path=self.catalog_path, output=t.output, transport=t)
        self.assertEqual(result["counts"]["rules_passing_gate"], 0)
        self.assertFalse(result["checker_executed"])
        self.assertEqual(t.requests, 0)
        self.assertEqual(result["trace"]["symbolic_check"]["suppressed_diagnostics"][0]["reason"], "rule_publish_gate_precheck")
        del self.rule["publishable"]
        write_json(self.catalog_path, self.catalog)
        self.payload[0]["evidence"]["quote"] = "invented quote not in solution"
        result = run_job(self.sample, self.autonomous, catalog_path=self.catalog_path, output=t.output, transport=t)
        self.assertEqual(result["counts"]["released_diagnostics"], 0)
        self.assertEqual(result["counts"]["checker_diagnostics"], 1)

    def test_reuse_requires_full_input_and_parameters_and_intact_record(self):
        a, b = self.transport("a"), self.transport("b")
        body = {"model": QWEN, "messages": [{"role": "user", "content": "first"}], "max_tokens": 20, "temperature": 0.1}
        a.post("/chat/completions", body)
        b.post("/chat/completions", body)
        b.post("/chat/completions", {**body, "temperature": 0.2})
        b.post("/chat/completions", {**body, "messages": [{"role": "user", "content": "changed"}]})
        self.assertEqual(len(self.requests), 3)
        source = a.output / "request_001.json"
        source.write_text(source.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            b.post("/chat/completions", body)

    def test_failed_retrieval_and_changed_inputs_cannot_run(self):
        with self.assertRaisesRegex(ValueError, "Failed retrieval"):
            matcher_result({**self.autonomous, "status": "failed"}, self.catalog, self.sample)
        v = FrozenVerifier(result=self.autonomous, sample=self.sample, catalog_path=self.catalog_path, output=self.root)
        with self.assertRaisesRegex(ValueError, "differs"):
            v.verify({**self.sample, "prediction": "another answer"})
        self.assertEqual(len(self.requests), 0)

    def test_malformed_checker_response_is_a_failure_not_a_clean_answer(self):
        self.payload = {"rule": "wrong_rule", "severity": "error"}
        t = self.transport("invalid")
        result = run_job(self.sample, self.autonomous, catalog_path=self.catalog_path, output=t.output, transport=t)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["trace"]["checker_failures"])
        self.assertEqual(result["counts"]["released_diagnostics"], 0)

    def test_retry_resolution_preserves_failures_and_waits_for_pending_retry(self):
        source, retry = self.root / "retrieval", self.root / "retry"
        inputs = self.root / "inputs.json"
        write_json(inputs, [self.sample])
        manifest = {"phase": "development", "input_sha256": sha256_file(inputs), "catalog_sha256": sha256_file(self.catalog_path), "configuration_sha256": "config"}
        write_json(source / "manifest.json", manifest)
        path = source / "arm2" / job_key("case") / "result.json"
        result = {"id": "case", "arm": 2, "model": ASTRA, "candidate_source": "native_semantic_tree", "status": "failed", "configuration_sha256": "config"}
        write_json(path, result)
        retry_manifest = {**manifest, "retry_of": {"manifest_sha256": sha256_file(source / "manifest.json"), "jobs": [{"sample_id": "case", "arm": 2, "result_sha256": sha256_file(path)}]}}
        write_json(retry / "manifest.json", retry_manifest)
        jobs = read_frozen_jobs([source, retry], inputs, self.catalog_path, [self.sample], [2])
        self.assertEqual(jobs[("case", 2)]["result"]["status"], "retry_pending")
        later = self.root / "later"
        retry_manifest = json.loads((retry / "manifest.json").read_text())
        write_json(later / "manifest.json", {**retry_manifest, "retry_of": {**retry_manifest["retry_of"], "source": str(retry), "manifest_sha256": sha256_file(retry / "manifest.json")}})
        write_json(later / "arm2" / job_key("case") / "result.json", {**result, "status": "completed"})
        jobs = read_frozen_jobs([source, retry, later], inputs, self.catalog_path, [self.sample], [2])
        self.assertEqual([item["status"] for item in jobs[("case", 2)]["attempts"]], ["failed", "completed"])
        write_json(retry / "arm2" / job_key("case") / "result.json", {**result, "status": "completed"})
        jobs = read_frozen_jobs([source, retry], inputs, self.catalog_path, [self.sample], [2])
        self.assertEqual([item["status"] for item in jobs[("case", 2)]["attempts"]], ["failed", "completed"])
        write_json(path, {**result, "status": "completed"})
        with self.assertRaisesRegex(ValueError, "Only a linked failed"):
            read_frozen_jobs([source, retry], inputs, self.catalog_path, [self.sample], [2])

    def test_checker_accepts_terra_but_rejects_mixed_api_models(self):
        inputs = self.root / "inputs.json"
        write_json(inputs, [self.sample])
        runs = []
        for arm, model, effort, source in [(2, TERRA, "medium", "native_semantic_tree"), (3, ASTRA, "max", "autonomous_tree_tools")]:
            run = self.root / model
            runs.append(run)
            manifest = {"phase": "development", "input_sha256": sha256_file(inputs), "catalog_sha256": sha256_file(self.catalog_path), "configuration_sha256": model, "api_model": model, "reasoning_effort": effort}
            write_json(run / "manifest.json", manifest)
            write_json(run / f"arm{arm}" / job_key("case") / "result.json", {"id": "case", "arm": arm, "status": "completed", "model": model, "reasoning_effort": effort, "candidate_source": source, "configuration_sha256": model})
        jobs = read_frozen_jobs(runs[:1], inputs, self.catalog_path, [self.sample], [2, 3])
        self.assertEqual(jobs[("case", 2)]["result"]["model"], TERRA)
        with self.assertRaisesRegex(ValueError, "Cannot mix API models"):
            read_frozen_jobs(runs, inputs, self.catalog_path, [self.sample], [2, 3])


if __name__ == "__main__":
    unittest.main()
