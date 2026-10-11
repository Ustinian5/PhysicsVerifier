import json
import sys
from types import SimpleNamespace, ModuleType

import pytest

from scripts import resume_continuous_context_v2 as recovery
from scripts import run_continuous_context_v2 as runner


@pytest.mark.parametrize("recipient", ["<|constrain|>json", "<|constrain|>JSON", "<|constrain|>Json"])
def test_complete_json_final_accepts_format_case(monkeypatch, recipient):
    module = ModuleType("vllm.entrypoints.harmony_utils")
    parsed = SimpleNamespace(current_content="", messages=[SimpleNamespace(channel="final",
        recipient=recipient, content=[SimpleNamespace(text='{"diagnostics":[]}')])])
    module.parse_output_into_messages = lambda ids: parsed
    entries = ModuleType("vllm.entrypoints")
    entries.harmony_utils = module
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", entries)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.harmony_utils", module)
    raw = {"model": runner.r.EXPECTED_MODEL,
           "usage": {"prompt_tokens": 10, "completion_tokens": 2},
           "choices": [{"token_ids": [5, 200002], "finish_reason": "stop"}]}
    result = recovery.decode_response(raw, model=runner.r.EXPECTED_MODEL, max_tokens=20, native_tools=None)
    assert json.loads(result["content"]) == {"diagnostics": []}
    assert parsed.messages[0].recipient == recipient
    raw["choices"][0]["token_ids"][-1] = 200012
    with pytest.raises(ValueError):
        recovery.decode_response(raw, model=runner.r.EXPECTED_MODEL, max_tokens=20, native_tools=None)


@pytest.mark.parametrize("channel,recipient", [("analysis", "<|constrain|>JSON"),
    ("final", "functions.search_rules"), ("final", "<|constrain|>xml"),
    ("analysis", "functions.SEARCH_RULES")])
def test_no_normalization_of_routing_or_other_formats(channel, recipient):
    parsed = SimpleNamespace(current_content="", messages=[SimpleNamespace(channel=channel,
        recipient=recipient, content=[])])
    assert recovery.normalize_messages(parsed).messages[0].recipient == recipient


def test_unknown_failure_is_not_retried_or_copied(tmp_path, monkeypatch):
    parent, output = tmp_path / "original", tmp_path / "recovery"
    runner.r.write_json(parent / "execution_status.json", {"state": "stopped_controls"})
    runner.r.write_json(parent / "controls/samples/C0p/id/result.json",
                        {"status": "failed", "failure_reason": "network failure"})
    monkeypatch.setattr(runner, "validate_plan", lambda p: {})
    monkeypatch.setattr(runner, "audit_result", lambda p: None)
    with pytest.raises(ValueError, match="Unreviewed failure"):
        recovery.prepare(parent, output)
    assert not output.exists()
