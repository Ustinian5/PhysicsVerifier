import copy
import json
from unittest.mock import patch

import pytest

from scripts import resume_aligned_step_evaluation as r


def response_fixture():
    row = dict(id='sample', question='Q', context='', prediction='A')
    req = dict(messages=r.base.p.query_messages(row), model=r.base.r.EXPECTED_MODEL,
               max_tokens=2048, temperature=0, top_p=1, reasoning_effort='medium')
    request = dict(request=req, request_sha256=r.base.r.sha256_value(req))
    usage = dict(prompt_tokens=30, completion_tokens=10, total_tokens=40)
    raw = dict(model=r.base.r.EXPECTED_MODEL, choices=[dict(finish_reason='stop')], usage=usage)
    response = dict(http_status=200, model_ids=[r.base.r.EXPECTED_MODEL], finish_reasons=['stop'],
                    usage=usage, content=json.dumps({'queries': ['a', 'b', 'c', 'd']}))
    return row, request, raw, response


def test_original_three_query_limit_is_preserved():
    row, req, raw, response = response_fixture()
    before = copy.deepcopy(response)
    assert r.validate_response(row, req, raw, response) == 'Invalid query phrases'
    assert response == before
    with pytest.raises(ValueError, match='cannot be discarded'):
        r.schema_failure('{"queries":["a","b","c"]}')


@pytest.mark.parametrize('content', ['{"queries":[]}', '{"queries":[42]}', '{"queries":[" "]}',
    '{"diagnostics":[]}', json.dumps({'queries': ['x' * 301]}), 'not JSON'])
def test_invalid_content_is_never_coerced_into_queries(content):
    assert r.schema_failure(content)


@pytest.mark.parametrize('change', ['identity', 'http', 'error', 'length', 'request', 'usage'])
def test_unknown_or_technical_failure_still_stops(change):
    row, req, raw, response = response_fixture()
    if change == 'identity': response['model_ids'] = ['another model']
    elif change == 'http': response['http_status'] = 503
    elif change == 'error': response['error'] = 'timeout'
    elif change == 'length': response['finish_reasons'] = ['length']
    elif change == 'request': req['request']['messages'][0]['content'] = 'changed'
    elif change == 'usage': response['usage'] = dict(prompt_tokens=30, completion_tokens=-1, total_tokens=29)
    with pytest.raises(ValueError): r.validate_response(row, req, raw, response)


def failed_fixture(tmp_path):
    row, request, raw, response = response_fixture()
    folder = tmp_path / 'queries' / r.vr.digest(row['id'])[:24]
    for name, value in [('request', request), ('raw', raw), ('response', response)]:
        r.vr.atomic(folder / (name + '.json'), value)
    evidence = dict(turn=1, usage=response['usage'], shared_preparation=True,
        model_ids=response['model_ids'], finish_reasons=response['finish_reasons'])
    for field, filename in [('request', 'request'), ('raw_sse', 'raw'), ('response', 'response')]:
        path = folder / (filename + '.json')
        evidence[field + '_path'] = str(path); evidence[field + '_sha256'] = r.vr.sha(path)
    plan = dict(configuration_sha256='frozen')
    receipt = dict(policy=r.POLICY, id=row['id'], configuration_sha256='frozen', input_sha256=r.vr.digest(row),
        reason='Invalid query phrases', failure_stage='prepare', evidence=evidence,
        files={str(p): r.vr.sha(p) for p in folder.glob('*.json')}, new_model_calls=0)
    return row, plan, receipt


def test_failure_blocks_all_four_arms_and_preserves_raw_response(tmp_path):
    row, plan, receipt = failed_fixture(tmp_path)
    protected = dict(receipt['files'])
    results = r.record_failure(tmp_path, row, plan, receipt)
    assert set(results) == set(r.base.p.ARMS)
    for value in results.values():
        assert value['status'] == 'failed' and value['checker_status'] == 'not_run'
        assert value['trace']['model_requests'] == 1 and value['diagnostics'] == []
        assert value['trace']['output_token_limit_reserved'] == 10
        r.base.runtime.audit_result(value)
    assert r.record_failure(tmp_path, row, plan, receipt) == results
    assert all(r.vr.sha(p) == h for p, h in protected.items())
    # This is a failed question with its GT denominator retained, never a valid
    # empty diagnosis or a successful no-candidate retrieval.
    dataset = [dict(row, physics_error_gt=[dict(error_id='error', error_text='error', answer_quote='A',
                    start_char=0, end_char=1, span_valid=True, paragraph_index=-1, paragraph_valid=False)])]
    score = r.run.score_items(dataset, [results['pure']], {'sample': []})
    assert score['summary']['total_gt_errors'] == 1
    assert score['summary']['failed_questions'] == 1
    assert score['summary']['successful_empty_questions'] == 0


def test_completed_output_cannot_be_overwritten_by_failure(tmp_path):
    row, plan, receipt = failed_fixture(tmp_path)
    target = r.base.path_for(tmp_path, 'formal', r.base.p.ARMS[0], row)
    r.vr.atomic(target, dict(status='completed', diagnostics=['existing']))
    original = target.read_bytes()
    with pytest.raises(ValueError, match='must not be overwritten'):
        r.record_failure(tmp_path, row, plan, receipt)
    assert target.read_bytes() == original


def test_english_prepare_failure_only_marks_the_three_knowledge_arms(tmp_path):
    row, plan, receipt = failed_fixture(tmp_path)
    plan['configuration'] = {'prompt_profile': r.base.en.PROFILE}
    target = r.base.path_for(tmp_path, 'formal', 'pure', row)
    with pytest.raises(ValueError, match='affected arms'):
        r.record_failure(tmp_path, row, plan, dict(receipt, affected_arms=list(r.base.p.ARMS)))
    receipt['affected_arms'] = r.base.p.prepared_arms(english=True)
    results = r.record_failure(tmp_path, row, plan, receipt)
    assert set(results) == {'case_rag', 'flat_skills', 'rule_tree'} and not target.parent.exists()
    assert all(item['failure_stage'] == 'prepare' for item in results.values())
    with pytest.raises(ValueError, match='direct baseline'):
        r.failure_result(tmp_path, row, 'pure', plan, receipt)
    r.vr.atomic(tmp_path/'inputs.json', [row])
    with patch.object(r, 'verify', return_value=plan), \
         patch.object(r.base, 'run_condition', side_effect=ValueError('direct verdict failed')), \
         patch.object(r, 'inspect_preparation', side_effect=AssertionError('Read PREPARE for pure baseline')), \
         r.recovered(tmp_path), pytest.raises(ValueError, match='direct verdict failed'):
        r.base.run_condition(tmp_path, row, 'formal', 'pure', None, plan, None, None)


def test_adapter_keeps_success_path_and_classifies_only_formal_prepare(tmp_path):
    row, plan, receipt = failed_fixture(tmp_path); r.vr.atomic(tmp_path / 'inputs.json', [row])
    with patch.object(r, 'verify', return_value=plan), \
         patch.object(r, 'inspect_preparation', return_value=receipt), \
         patch.object(r.base, 'run_condition', side_effect=[{'status': 'completed'}, ValueError('Invalid query phrases')]):
        with r.recovered(tmp_path):
            first = r.base.run_condition(tmp_path, row, 'formal', 'pure', None, plan, None, None)
            assert first == {'status': 'completed'}
            failed = r.base.run_condition(tmp_path, row, 'formal', 'pure', None, plan, None, None)
            assert r.base.complete_content_failure(failed)
            bad = dict(failed, diagnostics=[{'message': 'fabricated'}])
            with pytest.raises(ValueError): r.base.complete_content_failure(bad)
    with patch.object(r, 'verify', return_value=plan), \
         patch.object(r, 'inspect_preparation') as inspect, \
         patch.object(r.base, 'run_condition', side_effect=ValueError('unknown')):
        with r.recovered(tmp_path):
            with pytest.raises(ValueError):
                r.base.run_condition(tmp_path, row, 'controls', 'pure', None, plan, None, None)
            inspect.assert_not_called()
