import sys
from types import SimpleNamespace as NS

import pytest

from scripts import resume_organization_bench99 as x
from tests.test_resume_organization_comparison import fixture


def setup_failure(tmp_path, monkeypatch):
    path = fixture(tmp_path, monkeypatch)
    value = x.vr.read(path); value['failure_reason'] = x.REASON; x.vr.atomic(path, value)
    record = x.vr.read(tmp_path/'response.json'); record['error'] = x.REASON
    x.vr.atomic(tmp_path/'response.json', record)
    monkeypatch.setattr(x, 'schema_failure_usage', lambda raw, request: raw['usage'])
    return path


def test_failure_is_preserved_and_usage_bound(tmp_path, monkeypatch):
    path = setup_failure(tmp_path, monkeypatch); before = x.vr.sha(path)
    with x.patch.object(x.recovery, 'inspect_failure', x.inspect_failure):
        a = x.recovery.account(path); b = x.recovery.account(path)
        assert a == b and a['usage']['completion_tokens'] == 10
        assert a['new_model_calls'] == 0 and x.vr.sha(path) == before
        (tmp_path/'raw.json').write_text('{}')
        with pytest.raises(ValueError, match='evidence changed'): x.recovery.account(path)


@pytest.mark.parametrize('field,value', [('model', 'other'), ('finish_reason', 'length'), ('http_status', 503)])
def test_transport_identity_and_truncation_stop(tmp_path, monkeypatch, field, value):
    path = setup_failure(tmp_path, monkeypatch)
    raw = x.vr.read(tmp_path/'raw.json'); record = x.vr.read(tmp_path/'response.json')
    if field == 'model': raw['model'] = value
    elif field == 'finish_reason': raw['choices'][0][field] = value
    else: record[field] = value
    x.vr.atomic(tmp_path/'raw.json', raw); record['raw_sse_sha256'] = x.vr.sha(tmp_path/'raw.json')
    x.vr.atomic(tmp_path/'response.json', record)
    with pytest.raises(ValueError, match='not eligible'): x.inspect_failure(path)


@pytest.mark.parametrize('content,accepted', [('[]', True), ('{"diagnostics":[]}', False)])
def test_native_wrong_schema_only(monkeypatch, content, accepted):
    parsed = NS(messages=[NS(channel='final', recipient='<|constrain|>diagnostics',
                             content=[NS(text=content)])], current_content='')
    monkeypatch.setitem(sys.modules, 'vllm', NS())
    monkeypatch.setitem(sys.modules, 'vllm.entrypoints', NS(harmony_utils=NS(parse_output_into_messages=lambda _: parsed)))
    from scripts import resume_continuous_context_v2 as previous
    monkeypatch.setattr(previous, 'normalize_messages', lambda p: p)
    def fail(*a, **kw): raise ValueError(x.REASON.removeprefix('ValueError: '))
    monkeypatch.setattr(x.base, 'decode_response', fail)
    raw = {'choices': [{'token_ids': [200002]}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 1}}
    request = {'model': x.base.r.EXPECTED_MODEL, 'max_tokens': 100}
    if accepted: assert x.schema_failure_usage(raw, request) == raw['usage']
    else:
        with pytest.raises(ValueError, match='Valid schema'): x.schema_failure_usage(raw, request)
