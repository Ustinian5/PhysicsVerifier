import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts import resume_continuous_context_v3 as recovery


def response(monkeypatch, channel="final", recipient="<|constrain|>diagnostics",
             content='{"diagnostics":[]}', terminal=200002):
    module = ModuleType("vllm.entrypoints.harmony_utils")
    module.parse_output_into_messages = lambda ids: SimpleNamespace(current_content="",
        messages=[SimpleNamespace(channel=channel, recipient=recipient,
                                 content=[SimpleNamespace(text=content)])])
    entries = ModuleType("vllm.entrypoints")
    entries.harmony_utils = module
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", entries)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.harmony_utils", module)
    model = recovery.execution.runner.r.EXPECTED_MODEL
    raw = {"model": model, "usage": {"prompt_tokens": 681, "completion_tokens": 340},
           "choices": [{"token_ids": [5] * 339 + [terminal], "finish_reason": "stop"}]}
    return recovery.decode_response(raw, model=model, max_tokens=16384, native_tools=None)


def test_complete_schema_label_preserves_text_and_usage(monkeypatch):
    result = response(monkeypatch)
    assert result["content"] == '{"diagnostics":[]}'
    assert result["usage"]["completion_tokens"] == 340
    assert result["serialization_repairs"][0]["field_text_preserved"] is True


def test_quote_patch_format_preserves_original_patch(monkeypatch):
    content = '{"quote_patches":[{"index":0,"quote":"original"}]}'
    result = response(monkeypatch, recipient="<|constrain|>quote_patches", content=content)
    assert result["content"] == content
    assert result["usage"]["completion_tokens"] == 340


def test_stored_quote_response_recovered_without_new_generation(tmp_path, monkeypatch):
    runner = recovery.execution.runner
    parent, output = tmp_path / "parent", tmp_path / "new"
    folder = parent / "formal/samples/C4/case"
    turn = folder / "turn_03"
    write = runner.r.write_json
    write(parent / "plan.json", {})
    write(parent / "manifest.json", {})
    write(parent / "native_io/acceptance.json", {"passed": True})
    write(parent / "execution_status.json", {"state": "stopped_formal", "unix": 100})
    write(turn / "request.json", {"request": {"model": "same", "max_tokens": 1000}})
    write(turn / "response.sse", {"raw": "unchanged"})
    write(turn / "response.json", {"error": "ValueError: Unregistered actual Harmony recipient"})
    original = {"diagnostics": [{"message": "fixed", "evidence": {"quote": "old"}}]}
    write(folder / "first_diagnosis.json", original)
    state = {"turn": 2, "sample_id": "case", "started_unix": 95,
             "response_evidence": [], "reserved_output_tokens": 100,
             "quote_repair": {"status": "technical_failure", "attempts": 1,
                              "error": "ValueError: Unregistered actual Harmony recipient"},
             "pending": {"request_path": str(turn / "request.json"),
                         "raw_path": str(turn / "response.sse"),
                         "response_path": str(turn / "response.json")}}
    write(folder / "state.json", state)
    write(folder / "result.json", {"status": "completed", "quote_repair": state["quote_repair"]})
    write(parent / "formal/samples/C0p/good/result.json", {"status": "completed"})
    before = runner.r.sha256_file(folder / "state.json")
    monkeypatch.setattr(runner, "validate_plan", lambda p: {"configuration_sha256": "sig"})
    monkeypatch.setattr(runner, "audit_result", lambda value: None)
    monkeypatch.setattr(recovery, "decode_response", lambda *a, **kw: {
        "content": '{"quote_patches":[{"index":0,"quote":"original"}]}',
        "usage": {"completion_tokens": 744}})
    recovery.prepare(parent, output)
    restored = json.loads((output / "formal/samples/C4/case/state.json").read_text())
    assert restored["quote_repair"] == {"status": "pending", "attempts": 1}
    assert restored["turn"] == 2 and restored["reserved_output_tokens"] == 100
    assert runner.r.sha256_file(folder / "state.json") == before
    evidence = json.loads((output / "recovery.json").read_text())
    assert evidence["new_model_calls"] == 0
    assert len(evidence["unchanged_successes"]) == 1
    assert json.loads((output / "formal/samples/C4/case/first_diagnosis.json").read_text()) == original


@pytest.mark.parametrize("kwargs", [
    {"channel": "analysis"}, {"recipient": "<|constrain|>other"},
    {"content": '{"diagnostics":[]'}, {"content": '{"diagnostics":[],"extra":1}'},
    {"content": '{"diagnostics":{}}'}, {"terminal": 200012},
])
def test_unknown_routing_partial_schema_and_terminal_rejected(monkeypatch, kwargs):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        response(monkeypatch, **kwargs)
