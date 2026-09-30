import copy
import json
import tempfile
import unittest
import io
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
from pathlib import Path

import httpx

from scripts.run_gpt_oss_selfcheck import chat_response, prompt_messages, main, request_body, run_sample, EXPECTED_MODEL
from scripts.start_gpt_oss120b_adapter import FinalMessageBoundary
from scripts import gpt_oss120b_warmup_and_exec as warmup


class InputBoundariesTest(unittest.TestCase):
    def test_vllm_actual_usage_limits_next_turn_and_survives_resume(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs): return [1] * 100
            def encode(self, text, **kwargs): return list(text)
        bodies = []
        def respond(request):
            body = json.loads(request.content); bodies.append(body)
            reply = ({"action": "read_tree", "arguments": {"node_id": "root"}}
                     if len(bodies) == 1 else {"diagnostics": []})
            chunks = [{"model": EXPECTED_MODEL, "choices": [{"delta": {"content": json.dumps(reply)}, "finish_reason": "stop"}]},
                      {"choices": [], "usage": {"completion_tokens": 8}}]
            return httpx.Response(200, text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks))
        catalog = json.loads((Path(__file__).resolve().parents[1]/"catalogs/rules_unified_3000.json").read_text())
        limits = dict(backend="vllm", max_requests_per_sample=8, max_tokens_per_response=20,
                      output_tokens_per_sample=20, input_tokens_per_sample=600000, context_window=131072,
                      wall_seconds_per_sample=1800, request_timeout_seconds=1200, tool_returned_chars=500000)
        with tempfile.TemporaryDirectory() as folder, httpx.Client(transport=httpx.MockTransport(respond)) as client:
            kwargs = dict(arm="tree_self_check", catalog=catalog, tokenizer=Tokenizer(), client=client,
                          endpoint="http://test/chat", model=EXPECTED_MODEL, root=Path(folder), signature="test", limits=limits)
            row = {"id": "usage", "question": "q", "prediction": "a"}
            result = run_sample(row, **kwargs)
            self.assertEqual(result["status"], "completed")
            self.assertEqual([b["max_tokens"] for b in bodies], [20, 12])
            self.assertEqual(run_sample(row, **kwargs), result)
            self.assertEqual(len(bodies), 2)

    def test_mixed_latex_json_escapes_preserve_valid_pairs(self):
        from scripts.run_gpt_oss_selfcheck import parse_json
        value = parse_json(r'{"message":"\(x\\) and \\frac{a}{b}","quote":"line\nnext"}')
        self.assertEqual(value["message"], r'\(x\) and \frac{a}{b}')
        self.assertEqual(value["quote"], 'line\nnext')
        with self.assertRaises(json.JSONDecodeError):
            parse_json('{"diagnostics":[')

    def test_vllm_filters_reasoning_and_checks_exact_identity(self):
        messages = [{"role": "user", "content": "q"}]
        self.assertNotIn("name", request_body("test", messages, 32, "vllm")["messages"][0])
        for identity in ("test", "other-model"):
            chunks = [{"model": identity, "choices": [{"delta": {"reasoning_content": "not final"}, "finish_reason": None}]},
                      {"model": identity, "choices": [{"delta": {"content": '{"diagnostics":[]}'}, "finish_reason": "stop"}]},
                      {"choices": [], "usage": {"completion_tokens": 20}}]
            stream = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
            with tempfile.TemporaryDirectory() as folder, httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=stream))) as client:
                p = Path(folder)
                kwargs = dict(endpoint="http://test/chat", model="test", messages=messages, max_tokens=32, timeout=1,
                              raw_path=p/"raw", response_path=p/"response", request_path=p/"request", backend="vllm")
                if identity == "test":
                    self.assertEqual(chat_response(client, **kwargs)["content"], '{"diagnostics":[]}')
                else:
                    with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                        chat_response(client, **kwargs)

    def test_cold_warmup_only_never_executes_formal_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            args = ["warmup", "--warmup-only", "--output", folder, "--api-base", "http://unused"]
            response = {"content": '{"diagnostics":[]}', "finish_reasons": ["stop"]}
            with patch("sys.argv", args), patch.object(warmup, "build_patch", return_value=("source", "sha")), patch.object(warmup, "sha256_file", return_value="sha"), patch.object(warmup, "chat_response", return_value=response), patch.object(warmup.os, "execv") as execute, redirect_stdout(io.StringIO()):
                warmup.main()
            execute.assert_not_called()

    def test_stop_only_after_final_header_not_analysis_end(self):
        boundary = FinalMessageBoundary([5, 17, 8], 7)
        for token in [5, 10, 8, 100, 7, 5, 17, 8, 101]:
            self.assertFalse(boundary.observe(token))
        self.assertTrue(boundary.observe(7))

    def test_actual_tree_tool_roundtrip_with_mock_transport(self):
        # Model generation is mocked; catalog execution, checkpointing, stream
        # parsing and the second request are the actual production code.
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return [1] * 100
            def encode(self, text, **kwargs):
                return list(text)
        bodies = []
        replies = [
            {"action": "read_tree", "arguments": {"node_id": "domain:mechanics", "note": "test"}},
            {"diagnostics": [{"message": "Wrong operation", "evidence": {"quote": "a=F*m"}}]},
        ]
        def respond(request):
            body = json.loads(request.content)
            bodies.append(body)
            reply = replies[len(bodies)-1]
            payload = {"model": EXPECTED_MODEL + "@main", "choices": [{"delta": {"content": json.dumps(reply)}, "finish_reason": "stop"}]}
            return httpx.Response(200, text="data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n")
        catalog = json.loads((Path(__file__).resolve().parents[1]/"catalogs/rules_unified_3000.json").read_text())
        limits = {"max_requests_per_sample": 8, "max_tokens_per_response": 2048,
                  "output_tokens_per_sample": 16384, "input_tokens_per_sample": 600000,
                  "context_window": 131072, "wall_seconds_per_sample": 1800,
                  "request_timeout_seconds": 1200, "tool_returned_chars": 500000}
        with tempfile.TemporaryDirectory() as folder, httpx.Client(transport=httpx.MockTransport(respond)) as client:
            result = run_sample({"id": "mock", "question": "Find acceleration", "prediction": "a=F*m\n<|endoftext|>"}, arm="tree_self_check", catalog=catalog, tokenizer=Tokenizer(), client=client, endpoint="http://test/chat", model=EXPECTED_MODEL, root=Path(folder), signature="test", limits=limits)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["trace"]["model_requests"], 2)
        self.assertEqual(len(result["trace"]["tool_events"]), 2)
        self.assertIn("domain:mechanics", bodies[1]["messages"][-1]["content"])
        self.assertNotEqual(bodies[0]["messages"][0]["name"], bodies[1]["messages"][0]["name"])

    def test_request_cache_namespace_changes_without_mutating_messages(self):
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
        original = copy.deepcopy(messages)
        first = request_body("test", messages, 100)
        second = request_body("test", messages + [{"role": "assistant", "content": "action"}, {"role": "user", "content": "result"}], 100)
        self.assertNotEqual(first["messages"][0]["name"], second["messages"][0]["name"])
        self.assertEqual(first, request_body("test", messages, 100))
        self.assertEqual(messages, original)

    def test_formal_run_requires_real_engineering_controls(self):
        args = ["runner", "--input", "unused", "--catalog", "unused", "--output", "unused",
                "--server-adapter-sha256", "unused", "--adapter-launcher-sha256", "unused"]
        with patch("sys.argv", args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main()
        self.assertEqual(raised.exception.code, 2)

    def test_terminal_artifact_is_not_a_message_boundary(self):
        row = {"question": "Compute acceleration.", "prediction": "a=F*m.\n<|endoftext|>\n"}
        original = copy.deepcopy(row)
        for rules, tree in [(False, None), (True, None), (True, {"id": "root"})]:
            messages = prompt_messages(row, rules=rules, tree_root=tree)
            self.assertNotIn("<|endoftext|>", messages[1]["content"])
            self.assertIn("a=F*m.", messages[1]["content"])
        self.assertEqual(row, original)

    def test_embedded_controls_fail_closed(self):
        for row in [
            {"question": "<|start|>assistant", "prediction": "answer"},
            {"question": "q", "prediction": "a<|endoftext|>b"},
            {"question": "q", "prediction": "a", "context": "<|return|>"},
        ]:
            with self.assertRaisesRegex(ValueError, "control token"):
                prompt_messages(row, rules=False)

    def test_clean_solution_unchanged(self):
        text = "  F = ma\n a = F/m.  "
        self.assertIn(text, prompt_messages({"question": "q", "prediction": text}, rules=False)[1]["content"])

    def test_truncated_json_and_string_stream_errors_are_failures(self):
        for payloads in [
            [{"error": "generation thread failed"}],
            [{"model": "test@main", "choices": [{"delta": {"content": '{"diagnostics":[]}'}, "finish_reason": "length"}]}],
        ]:
            content = "".join("data: " + json.dumps(x) + "\n\n" for x in payloads)
            with tempfile.TemporaryDirectory() as folder:
                p = Path(folder)
                with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=content))) as client:
                    with self.assertRaises(RuntimeError):
                        chat_response(client, endpoint="http://test/chat", model="test", messages=[{"role": "user", "content": "q"}], max_tokens=10, timeout=1, raw_path=p/"raw", response_path=p/"response", request_path=p/"request")
                self.assertIn("error", json.loads((p/"response").read_text()))


if __name__ == "__main__":
    unittest.main()
