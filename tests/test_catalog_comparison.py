import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import run_catalog_comparison as x
from tests.test_continuous_context_v2 import run as execute_sample, BAD, PATCH


def test_selected_c4_and_limits_are_exact_historical_text():
    selected = x.selected_protocol()
    assert selected['prompts'] == {'base': x.continuous.BASE, 'rules': x.continuous.RULES}
    assert selected['limits']['output_tokens_per_sample'] == 16384
    assert 'make at least ONE actual search_rules call' in selected['prompts']['rules']
    assert x.shared.protocol.KNOWLEDGE != selected['prompts']['rules']


def test_same_c4_context_and_quote_only_repair_in_both_arms(tmp_path):
    prompts = []
    with x.configured():
        for arm in x.ARMS:
            folder = tmp_path / arm
            result, calls = execute_sample(folder, 'C4', [
                {'action': 'search_rules', 'arguments': {'query': '合力', 'mode': 'bm25', 'limit': 6}},
                BAD, PATCH], review_protocol=x.continuous.PROTOCOL)
            assert result['status'] == 'completed'
            assert result['quote_repair']['status'] == 'finished'
            assert result['diagnostics'][0]['message'] == BAD['diagnostics'][0]['message']
            assert len(calls) == 3
            prompts.append(calls[0]['messages'][0]['content'])
            assert any(m['role'] == 'tool' for m in calls[1]['messages'])
            assert calls[2]['native_tools'] is None  # The last call copies quotes; it cannot re-check physics.
            again, more = execute_sample(folder, 'C4', [], review_protocol=x.continuous.PROTOCOL)
            assert again == result and not more
    assert prompts[0] == prompts[1] == x.continuous.BASE + '\n\n' + x.continuous.RULES


def test_true_empty_result_not_repeated(tmp_path):
    with x.configured():
        result, calls = execute_sample(tmp_path, 'C4', [
            {'action': 'search_rules', 'arguments': {'query': '合力', 'mode': 'bm25', 'limit': 6}},
            {'diagnostics': []}], review_protocol=x.continuous.PROTOCOL)
        assert result['status'] == 'completed' and result['diagnostics'] == []
        again, more = execute_sample(tmp_path, 'C4', [], review_protocol=x.continuous.PROTOCOL)
    assert again == result and not more


def test_only_bound_complete_content_failure_can_continue(tmp_path):
    path = tmp_path / 'result.json'
    evidence = {'request_path': 'req', 'response_path': 'res', 'finish_reasons': ['stop']}
    item = {'status': 'failed', 'failure_reason': 'JSONDecodeError: bad json',
            'trace': {'response_evidence': [evidence]}}
    x.vr.atomic(tmp_path / 'state.json', {'pending': {'request_path': 'req', 'response_path': 'res'}})
    with patch.object(x.runtime, 'audit_result') as audit:
        assert x.terminal_content_failure(item, path)
        audit.assert_called_once_with(item)
        for error in ['ReadTimeout: no response', 'ValueError: Wrong model', 'RuntimeError: budget exhausted']:
            assert not x.terminal_content_failure(dict(item, failure_reason=error), path)
        x.vr.atomic(tmp_path / 'state.json', {'pending': {'request_path': 'different', 'response_path': 'res'}})
        assert not x.terminal_content_failure(item, path)


def test_counter_correction_does_not_rewrite_raw_result():
    item = {'id': 'case', 'status': 'failed', 'trace': {'model_requests': 0,
            'response_evidence': [{'usage': {'prompt_tokens': 4, 'completion_tokens': 3}}]}}
    original = copy.deepcopy(item)
    summary = x.compute_summary([item])
    assert summary['model_requests'] == 1 and summary['usage_unknown_responses'] == 0
    assert item == original and len(summary['request_counter_adjustments']) == 1
    with pytest.raises(ValueError, match='successful request counter'):
        x.compute_summary([dict(item, status='completed')])


def test_two_arms_cannot_reuse_each_others_results(tmp_path):
    assert x.result_path(tmp_path, 'old', 'same-id') != x.result_path(tmp_path, 'new', 'same-id')
    assert not x.result_path(tmp_path, 'old', 'same-id').exists()


def test_exact_json_roundtrip_preserves_formula_body():
    payload = [{'rule_id': 'x', 'check_logic': r'Use \frac{a}{b} with a "quoted" assumption.'}]
    rendered = '<|start|>tool read_rules<|message|>' + json.dumps(json.dumps(payload)) + '<|end|><|start|>assistant'
    assert x.decode_tool_payload(rendered) == payload
