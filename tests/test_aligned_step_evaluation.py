import copy
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import run_aligned_step_evaluation as e


def review_fixture():
    rows = [dict(id='q', question='Q', context='', prediction='wrong line; final wrong line')]
    finals = {'q': dict(errors=[dict(root_id='E1', quote='wrong line', claim='same root cause', status='confirmed'),
                               dict(root_id='E2', quote='', claim='missing context', status='unresolved')],
                       uncertain_scopes=['missing figure'])}
    review = dict(human_gold=False, additional_experiment_api_calls=0,
        quote_decisions=[dict(id='q', root_id='E1', answer_sha256=hashlib.sha256(rows[0]['prediction'].encode()).hexdigest(),
            original_final_result_sha256='frozen', old_quote='wrong line', new_quote='final wrong line',
            claim='same root cause', old_status='confirmed')],
        physics_decisions=[dict(id='q', root_id='E2', action='retain_unresolved', basis='figure absent')])
    return rows, finals, review, {'q': 'frozen'}


def test_quote_release_changes_only_location_and_keeps_unknown():
    rows, originals, review, hashes = review_fixture(); before = copy.deepcopy(originals)
    revised = e.apply_quote_review(rows, originals, review, hashes)
    assert originals == before
    assert revised['q']['errors'][1] == originals['q']['errors'][1]
    expected = copy.deepcopy(originals); expected['q']['errors'][0]['quote'] = 'final wrong line'
    assert revised == expected
    dataset, unknown, counts = e.reference_rows(rows, revised)
    assert counts == dict(confirmed=1, unresolved=1, confirmed_unlocated=0)
    assert len(dataset[0]['physics_error_gt']) == len(unknown['q']) == 1


@pytest.mark.parametrize('field,value', [('new_quote', 'wrong line'), ('new_quote', 'invented text'),
    ('claim', 'a different error'), ('answer_sha256', 'wrong'), ('original_final_result_sha256', 'wrong')])
def test_rejects_ambiguous_invented_or_changed_review(field, value):
    rows, finals, review, hashes = review_fixture(); review['quote_decisions'][0][field] = value
    with pytest.raises(ValueError): e.apply_quote_review(rows, finals, review, hashes)


def test_unreviewed_unknown_cannot_disappear():
    rows, finals, review, hashes = review_fixture(); review['physics_decisions'] = []
    with pytest.raises(ValueError): e.apply_quote_review(rows, finals, review, hashes)


def test_control_query_reuse_does_not_substitute_formal_query(tmp_path):
    row = dict(id='control', question='Q', context='', prediction='A')
    prefix = dict(input_sha256=e.vr.digest(row), configuration_sha256='old', queries={'queries': ['physics']},
                  evidence={'usage': {'prompt_tokens': 40, 'completion_tokens': 20}})
    e.vr.atomic(tmp_path / 'control_prefixes.json', {'control': prefix})
    plan = {'configuration_sha256': 'new'}
    with patch.object(e.base, 'query', return_value='fresh') as fresh:
        with e.reused_controls(tmp_path, plan):
            got = e.base.query(tmp_path, row, plan, None, None)
            assert got['configuration_sha256'] == 'new'
            assert got['queries'] == prefix['queries'] and got['evidence'] == prefix['evidence']
            assert e.base.query(tmp_path, dict(row, id='new30b'), plan, None, None) == 'fresh'
            assert fresh.call_count == 1
            with pytest.raises(ValueError): e.base.query(tmp_path, dict(row, prediction='changed'), plan, None, None)
    assert e.vr.read(tmp_path / 'control_prefixes.json')['control'] == prefix


def test_scorer_keeps_old_default_and_failure_denominators():
    item = {'status': 'content_failure', 'protocol_audit': {'passed': False}}
    with patch.object(e.base.runtime, 'healthy_result', return_value=False), \
         patch.object(e.base.evaluation, 'score', return_value='scored') as score:
        assert e.score_items(['new reference'], [item], {'new': []}) == 'scored'
        args, kwargs = score.call_args
        assert args[0] == ['new reference'] and args[2] == {'new': []}
        assert args[1][0]['execution_failure'] and args[1][0]['protocol_noncompliance']
        assert kwargs == {'preserve_zero_offsets': False}


def test_released_material_tampering_stops_before_inference(tmp_path):
    e.vr.atomic(tmp_path / 'inputs.json', ['original'])
    c = dict(protocol=e.VERSION, source_hashes={'code': 'sha'}, parent='parent',
             released_files={'inputs': e.vr.sha(tmp_path / 'inputs.json')}, dependencies={})
    e.vr.atomic(tmp_path / 'plan.json', dict(configuration=c, configuration_sha256=e.vr.digest(c)))
    e.vr.atomic(tmp_path / 'inputs.json', ['tampered'])
    with patch.object(e, 'code_hashes', return_value={'code': 'sha'}), patch.object(e.a, 'verify'):
        with pytest.raises(ValueError, match='Released input drift'): e.verify(tmp_path)


def test_old_source_pruner_removes_any_affected_leaf_without_changing_survivor():
    catalog = dict(domains=[dict(id='d', topics=[dict(id='t', rules=[
        dict(rule_id='bad', title='bad title', trigger='bad'),
        dict(rule_id='good', title='good title', trigger='good')],
        scenario_clusters=[dict(id='c', name='mixed', summary='mixed', rule_ids=['good', 'bad'])])])])
    clean, receipt = e.a.bounded.previous.prune_catalog(catalog,
        {'bad': [{'sample_id': 'excluded'}, {'sample_id': 'kept'}], 'good': [{'sample_id': 'kept'}]}, {'excluded'})
    topic = clean['domains'][0]['topics'][0]
    assert topic['rules'] == [catalog['domains'][0]['topics'][0]['rules'][1]]
    assert topic['scenario_clusters'][0]['rule_ids'] == ['good']
    assert receipt['leaf_content_modified'] is False
    assert len(catalog['domains'][0]['topics'][0]['rules']) == 2
