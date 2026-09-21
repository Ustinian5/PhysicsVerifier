import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_retrieval_comparison import (
    Corpus, Transport, digest, model_input, tool_definitions, validate_selection,
    execute_tool_batch,
    transient_retry_jobs, write_json, BatchBudget, BatchBudgetExceeded,
    validate_batch_inputs, sha256_file,
)


def catalog():
    return json.loads((Path(__file__).resolve().parents[1] / "catalogs/rules_unified_3000.json").read_text())


def check_leaf_only_view_has_same_rules_but_no_tree(case):
    rules = catalog()
    a, b = Corpus(rules, tree=True), Corpus(rules, tree=False)
    assert len(a.ids) == len(b.ids) == 1123
    assert digest(a.leaves) == digest(b.leaves)
    assert a.nodes and not b.nodes
    assert "read_tree" not in {t["name"] for t in tool_definitions(False)}
    with case.assertRaisesRegex(ValueError, "unavailable"):
        b.call("read_tree", {"node_id": "root"}, max_chars=10000)
    for mode in ("bm25", "tfidf", "substring"):
        assert a.search("量纲", mode, 5) == b.search("量纲", mode, 5)


def check_retrieval_input_does_not_include_answers_or_annotations():
    source = {"id": "x", "question": "q", "prediction": "p", "context": "c", "answer": "secret", "physics_error_gt": ["secret"], "target_rule_id": "secret", "meta": {"tree": "secret"}}
    assert model_input(source) == {"id": "x", "question": "q", "prediction": "p", "context": "c"}


def check_tool_budget_does_not_mark_unread_rules_as_read(case):
    corpus = Corpus(catalog(), tree=False)
    rid = corpus.ids[0]
    with case.assertRaisesRegex(RuntimeError, "budget"):
        corpus.call("read_rules", {"rule_ids": [rid]}, max_chars=1)
    assert rid not in corpus.read_ids
    corpus.call("read_rules", {"rule_ids": [rid]}, max_chars=20000)
    assert rid in corpus.read_ids
    selected = [{"rule_id": rid, "score": 0.9, "reason": "applicable", "answer_quote": "v=t"}]
    validate_selection(selected, corpus, {"prediction": "Here v=t."})
    for field, value in [("rule_id", "invented"), ("score", float("nan")), ("score", True), ("answer_quote", "invented quote")]:
        bad = copy.deepcopy(selected)
        bad[0][field] = value
        with case.assertRaises(ValueError):
            validate_selection(bad, corpus, {"prediction": "Here v=t."})
    with case.assertRaises(ValueError):
        validate_selection(selected * 2, corpus, {"prediction": "Here v=t."})


def check_budget_rejects_before_any_network_request(case, tmp_path):
    transport = Transport(base="http://not-used.invalid", key="not-logged", output=tmp_path, model="gpt-6-astra", limits={"requests": 0, "seconds": 5, "input_tokens": 100, "output_tokens": 100})
    with case.assertRaisesRegex(RuntimeError, "budget"):
        transport.post("/responses", {"max_output_tokens": 1})
    assert not list(tmp_path.iterdir())


def check_tree_bridge_preserves_prompt_and_schema_without_sampling_parameters(tmp_path):
    transport = Transport(base="http://not-used.invalid", key="not-logged", output=tmp_path, model="gpt-6-astra", limits={"requests": 1, "seconds": 5, "input_tokens": 10000, "output_tokens": 16384, "per_response": 16384})
    captured = {}
    def post(endpoint, body):
        captured.update(body)
        return {"id": "resp_test", "model": "gpt-6-astra", "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"ok":true}'}]}]}
    transport.post = post
    messages = [{"role": "system", "content": "original instructions"}, {"role": "user", "content": "original input"}]
    schema = {"name": "test", "strict": True, "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}}
    response = transport.chat_create(messages=messages, temperature=0, max_tokens=1024, model="gpt-6-astra", response_format={"type": "json_schema", "json_schema": schema}, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    assert captured["input"] == messages
    assert captured["text"]["format"]["schema"] == schema["schema"]
    assert captured["reasoning"] == {"effort": "max"}
    assert not {"temperature", "max_tokens", "extra_body"} & set(captured)
    assert json.loads(response.choices[0].message.content) == {"ok": True}


class RetrievalComparisonTests(unittest.TestCase):
    def test_quota_probe_is_cached_and_short_outages_do_not_abort_models(self):
        import time
        from unittest.mock import Mock
        probe = Mock(side_effect=[{"total_available": 1000000, "unlimited_quota": False}, RuntimeError("temporary outage"), RuntimeError("still unavailable")])
        with tempfile.TemporaryDirectory() as directory:
            budget = BatchBudget(directory, {"requests": 10, "input_tokens": 10000, "output_tokens": 10000}, quota_probe=probe)
            budget.reserve(1, 1)
            budget.reserve(1, 1)
            self.assertEqual(probe.call_count, 1)
            budget.next_quota_check = 0
            budget.reserve(1, 1)
            self.assertFalse(budget.stopped)
            budget.next_quota_check = 0
            budget.platform_quota["checked_unix"] = time.time() - 301
            with self.assertRaises(BatchBudgetExceeded):
                budget.reserve(1, 1)

    def test_response_prefix_replay_is_exact_and_does_not_call_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = {"model": "gpt-6-astra", "input": "original"}
            response = {"id": "response", "model": "gpt-6-astra", "reasoning": {"effort": "max"}, "usage": {"input_tokens": 12, "output_tokens": 8}}
            write_json(root / "source/request_001.json", {"request": body, "endpoint": "/responses", "http_status": 200, "response": response, "elapsed_seconds": 3})
            options = dict(base="https://not-used.invalid", key="not-logged", limits={}, model="gpt-6-astra", replay_source=root / "source")
            transport = Transport(output=root / "resume", **options)
            self.assertEqual(transport.post("/responses", body), response)
            self.assertEqual((transport.requests, transport.input_tokens, transport.replayed_input_tokens), (0, 0, 12))
            other = Transport(output=root / "mismatch", **options)
            with self.assertRaisesRegex(ValueError, "exact request"):
                other.post("/responses", {**body, "input": "changed"})
            self.assertEqual(other.requests, 0)

    def test_continuation_skips_successes_in_all_ancestors(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, child = root / "original", root / "child"
            config = {"sample_ids": ["paused", "done_in_original", "done_in_child", "not_started"]}
            write_json(original / "manifest.json", config)
            write_json(child / "manifest.json", {**config, "retry_of": {"source": str(original), "manifest_sha256": sha256_file(original / "manifest.json")}})
            for run, sid, state in [(original, "paused", "failed"), (original, "done_in_original", "completed"), (child, "done_in_child", "completed")]:
                job = run / "arm2" / hashlib.sha256(sid.encode()).hexdigest()[:16]
                write_json(job / "result.json", {"status": state, "model": "gpt-6-astra", "error_type": "BatchBudgetExceeded" if state == "failed" else None})
                write_json(job / "request_001.json", {"http_status": 200})
            jobs = transient_retry_jobs(child, config, [2], include_pending=True)
            self.assertEqual(set(jobs), {("paused", 2), ("not_started", 2)})
            self.assertTrue(jobs[("not_started", 2)]["new_pending"])
            self.assertEqual(jobs[("paused", 2)]["source_run"], str(original.resolve()))

    def test_parallel_workers_share_token_reservations(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as directory:
            budget = BatchBudget(directory, {"requests": 5, "input_tokens": 100, "output_tokens": 100})
            def reserve(_):
                try:
                    return budget.reserve(60, 10)
                except BatchBudgetExceeded:
                    return None
            with ThreadPoolExecutor(max_workers=2) as pool:
                tickets = list(pool.map(reserve, range(2)))
            self.assertEqual(sum(t is not None for t in tickets), 1)
            self.assertEqual(budget.requests, 1)
            budget.settle(next(t for t in tickets if t is not None), {"http_status": 504})
            self.assertEqual((budget.input_tokens, budget.output_tokens, budget.unknown_usage), (60, 10, 1))

    def test_resumed_budget_includes_unknown_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = {"model": "gpt-6-astra", "max_output_tokens": 30}
            write_json(root / "arm2/case/request_001.json", {"request": body, "http_status": 504})
            write_json(root / "arm2/case/request_002.json", {"request": body, "response": {"usage": {"input_tokens": 100, "output_tokens": 20}}})
            budget = BatchBudget(root, {"requests": 10, "input_tokens": 1000, "output_tokens": 100})
            self.assertEqual((budget.requests, budget.output_tokens, budget.unknown_usage), (2, 50, 1))
            self.assertGreater(budget.input_tokens, 100)

    def test_low_platform_quota_stops_before_model_request(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = BatchBudget(directory, {"requests": 10, "input_tokens": 10000, "output_tokens": 100}, quota_probe=lambda: {"total_available": 10, "unlimited_quota": False}, quota_floor=20)
            transport = Transport(base="http://not-used.invalid", key="not-logged", output=Path(directory) / "arm2/case", model="gpt-6-astra", limits={"requests": 10, "seconds": 5, "input_tokens": 10000, "output_tokens": 100}, batch_budget=budget)
            with self.assertRaisesRegex(BatchBudgetExceeded, "platform_quota_low"):
                transport.post("/responses", {"max_output_tokens": 10})
            self.assertEqual(transport.requests, 0)
            self.assertFalse(list(Path(directory).glob("arm*/*/request_*.json")))

    def test_batch_requires_unchanged_sanitized_fifty_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"id": str(i), "question": "question", "prediction": "solution"} for i in range(50)]
            write_json(root / "inputs.json", rows)
            manifest = {"inputs_sha256": sha256_file(root / "inputs.json"), "sample_size": 50, "selected_ids": [r["id"] for r in rows]}
            write_json(root / "manifest.json", manifest)
            self.assertEqual(validate_batch_inputs(root / "inputs.json", root / "manifest.json", rows, catalog())["leaf_count"], 1123)
            rows[0]["answer"] = "SECRET_REFERENCE"
            write_json(root / "inputs.json", rows)
            with self.assertRaisesRegex(ValueError, "unchanged"):
                validate_batch_inputs(root / "inputs.json", root / "manifest.json", rows, catalog())
            manifest["inputs_sha256"] = sha256_file(root / "inputs.json")
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "only nonempty model input"):
                validate_batch_inputs(root / "inputs.json", root / "manifest.json", rows, catalog())

    def test_retry_selects_only_failed_gateway_requests(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"sample_ids": ["failed", "successful", "invalid_model_json"]}
            write_json(root / "manifest.json", config)
            for sid, status, http in [("failed", "failed", 504), ("successful", "completed", 200), ("invalid_model_json", "failed", 200)]:
                job = root / "arm2" / hashlib.sha256(sid.encode()).hexdigest()[:16]
                write_json(job / "result.json", {"status": status, "model": "gpt-6-astra"})
                write_json(job / "request_001.json", {"http_status": http})
            self.assertEqual(set(transient_retry_jobs(root, config, [2])), {("failed", 2)})
            with self.assertRaisesRegex(ValueError, "protocol differs"):
                transient_retry_jobs(root, {**config, "catalog_sha256": "changed"}, [2])

    def test_gateway_multiple_tool_calls_are_all_answered(self):
        corpus = Corpus(catalog(), tree=False)
        calls = [{"call_id": str(i), "name": "search_rules", "arguments": json.dumps({"query": q, "mode": "bm25", "limit": 2})} for i, q in enumerate(("量纲", "能量"))]
        outputs, submitted = execute_tool_batch(calls, corpus, {"prediction": "p"}, max_chars=20000)
        self.assertIsNone(submitted)
        self.assertEqual([o["call_id"] for o in outputs], ["0", "1"])
        self.assertEqual(len(corpus.events), 2)
        self.assertTrue(all(json.loads(o["output"]) for o in outputs))

    def test_submission_cannot_use_an_unseen_result_in_the_same_batch(self):
        corpus = Corpus(catalog(), tree=False)
        rid = corpus.ids[0]
        selected = [{"rule_id": rid, "score": 0.9, "reason": "r", "answer_quote": "p"}]
        calls = [{"call_id": "read", "name": "read_rules", "arguments": json.dumps({"rule_ids": [rid]})}, {"call_id": "submit", "name": "submit_rules", "arguments": json.dumps({"rules": selected})}]
        outputs, submitted = execute_tool_batch(calls, corpus, {"prediction": "p"}, max_chars=20000)
        self.assertIsNone(submitted)
        self.assertIn("Submit alone", json.loads(outputs[1]["output"])["error"])
        _, submitted = execute_tool_batch(calls[1:], corpus, {"prediction": "p"}, max_chars=20000)
        self.assertEqual(submitted["rules"], selected)

    def test_qwen_sdk_options_are_mapped_to_http_correctly(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = Transport(base="http://not-used.invalid", key="not-logged", output=directory, model="qwen", limits={})
            captured = {}
            def post(endpoint, body, **kwargs):
                captured.update(body=body, options=kwargs)
                return {"id": "r", "object": "chat.completion", "created": 1, "model": "qwen", "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}]}
            transport.post = post
            transport.chat_create(model="qwen", messages=[], timeout=300, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            self.assertEqual(captured["body"]["chat_template_kwargs"], {"enable_thinking": False})
            self.assertNotIn("extra_body", captured["body"])
            self.assertNotIn("timeout", captured["body"])
            self.assertEqual(captured["options"]["request_timeout"], 300)

    def test_leaf_only_view(self):
        check_leaf_only_view_has_same_rules_but_no_tree(self)

    def test_input_isolation(self):
        check_retrieval_input_does_not_include_answers_or_annotations()

    def test_read_and_selection_validation(self):
        check_tool_budget_does_not_mark_unread_rules_as_read(self)

    def test_request_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            check_budget_rejects_before_any_network_request(self, Path(directory))

    def test_transport_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            check_tree_bridge_preserves_prompt_and_schema_without_sampling_parameters(Path(directory))


if __name__ == "__main__":
    unittest.main()
