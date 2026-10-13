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


@pytest.mark.parametrize("kwargs", [
    {"channel": "analysis"}, {"recipient": "<|constrain|>other"},
    {"content": '{"diagnostics":[]'}, {"content": '{"diagnostics":[],"extra":1}'},
    {"content": '{"diagnostics":{}}'}, {"terminal": 200012},
])
def test_unknown_routing_partial_schema_and_terminal_rejected(monkeypatch, kwargs):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        response(monkeypatch, **kwargs)
