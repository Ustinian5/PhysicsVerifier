import copy
import json

import pytest

from scripts.audit_aligned_organization import Evidence, request_contract, verify_content_failure


def record():
    config = {'model': 'expected', 'max_tokens': 16384}
    messages = [{'role': 'user', 'content': 'Original source, without reference labels'}]
    raw = dict(request=dict(model='expected', messages=messages, temperature=0, top_p=1,
        max_tokens=16384, stream=False, return_token_ids=True, reasoning_effort='medium'),
        endpoint='/chat/completions', http_status=200,
        client_metadata=dict(build_stage='extract_skill', fingerprint='source-bound'),
        response=dict(model='expected', id='one', choices=[dict(finish_reason='stop')],
                      usage=dict(completion_tokens=20), prompt_token_ids=[1, 2]))
    return raw, messages, config


def test_request_exact_source_and_config_required():
    raw, messages, config = record()
    assert request_contract(raw, messages, config, 'source-bound') is raw['response']
    altered = copy.deepcopy(raw)
    altered['request']['messages'][0]['content'] += ' leaked answer'
    with pytest.raises(ValueError, match='Model-visible'):
        request_contract(altered, messages, config, 'source-bound')


@pytest.mark.parametrize('field,value', [('temperature', .2), ('max_tokens', 32768),
                                         ('reasoning_effort', 'high')])
def test_request_setting_drift_rejected(field, value):
    raw, messages, config = record(); raw['request'][field] = value
    with pytest.raises(ValueError): request_contract(raw, messages, config, 'source-bound')


def test_response_identity_is_checked_independently_of_request():
    raw, messages, config = record(); raw['response']['model'] = 'another'
    with pytest.raises(ValueError, match='identity'):
        request_contract(raw, messages, config, 'source-bound')


def test_known_length_failure_is_observable_not_transport_failure():
    raw, messages, config = record(); raw['response']['choices'][0]['finish_reason'] = 'length'
    assert request_contract(raw, messages, config, 'source-bound')['choices'][0]['finish_reason'] == 'length'


def test_evidence_detects_change_even_without_expected_hash(tmp_path):
    path = tmp_path/'original.json'; path.write_text(json.dumps({'v': 1}))
    evidence = Evidence(); evidence.read(path)
    path.write_text(json.dumps({'v': 2}))
    with pytest.raises(ValueError, match='during audit'): evidence.read(path)


def test_malformed_failure_must_reproduce_under_the_same_decoder(monkeypatch):
    raw, _, config = record()
    monkeypatch.setattr('scripts.audit_aligned_organization.navigation.decode_legacy',
                        lambda *args: {'_unparsed_response_text': '{broken'})
    verify_content_failure(raw, 'malformed_json; no resampling', config)
    monkeypatch.setattr('scripts.audit_aligned_organization.navigation.decode_legacy', lambda *args: {'skills': []})
    with pytest.raises(ValueError, match='now parses'):
        verify_content_failure(raw, 'malformed_json; no resampling', config)


def test_missing_final_must_be_the_exact_recorded_failure(monkeypatch):
    raw, _, config = record()
    def failed(*args): raise ValueError('Missing final message')
    monkeypatch.setattr('scripts.audit_aligned_organization.navigation.decode_legacy', failed)
    verify_content_failure(raw, 'missing_final; no resampling', config)
    def wrong(*args): raise ValueError('Wrong model')
    monkeypatch.setattr('scripts.audit_aligned_organization.navigation.decode_legacy', wrong)
    with pytest.raises(ValueError, match='Different'):
        verify_content_failure(raw, 'missing_final; no resampling', config)
