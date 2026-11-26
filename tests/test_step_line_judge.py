import copy
import json

import pytest

from rule_framework import step_match_lines as lines
from scripts import run_step_line_judge as run
from tests.test_final_answer_revision import fake_runtime


def decision(kind='same_root', ids=None, first=2, last=2):
    return dict(disposition=kind, root_ids=['E1'] if ids is None else ids,
                start_line=first, end_line=last, rationale='The published claim refers to this step.')


def test_line_numbers_pin_the_exact_original_occurrence():
    rows, results, _, _ = run.controls()
    row = dict(rows[0], prediction='物理\\,公式\r\n\r\n物理\\,公式\r\n')
    result = results[row['id'],'pure']
    stored = lines.validate(row, decision(first=3, last=3))
    assert stored['span'] == dict(start=len('物理\\,公式\r\n\r\n'), end=len(row['prediction']),
                                  quote='物理\\,公式\r\n')
    scores = run.reuse.location.score_row(row, result, [stored['effective_decision']], [stored['span']])
    assert scores['hits'] == 1 and scores['matches'][0]['prediction_span'] == stored['span']
    msg = lines.messages(row, dict(result['diagnostics'][0], arm='secret'))
    body = json.loads(msg[1]['content'])
    assert body['numbered_student_solution'] == [[1,'物理\\,公式'],[2,''],[3,'物理\\,公式']]
    assert set(body) == {'problem','context','numbered_student_solution','published_diagnosis','active_reference_roots'}
    assert 'secret' not in str(msg) and 'source_quote' not in msg[0]['content']


@pytest.mark.parametrize('choice, expected', [
    (decision(first=None,last=None),'missing_match_evidence'), (decision(first=True),'invalid_range'),
    (decision(first=2,last=1),'invalid_range'), (decision(first=None,last=1),'invalid_range'),
    (decision(first=10,last=11),'invalid_range'), (decision('other_claim',[],None,None),'no_passage'),
])
def test_unusable_range_never_earns_a_match(choice, expected):
    row = run.controls()[0][0]
    val = lines.validate(row, choice)
    assert val['location_status'] == expected and val['effective_decision']['root_ids'] == [] and val['span'] is None
    with pytest.raises(ValueError):
        lines.validate(row, dict(choice,root_ids=['foreign']))


def scripted(messages):
    data = json.loads(messages[1]['content'])
    msg = data['published_diagnosis']['message']
    if msg.startswith('Multiplying'):
        return json.dumps(decision('opposite_claim',first=2,last=3))
    if 'blue light' in msg:
        return json.dumps(decision('other_claim',[],None,None))
    return json.dumps(decision(first=3,last=3))


def runtime(monkeypatch, callback):
    _, _, tok, calls = fake_runtime(monkeypatch, model=run.judge.base.r.EXPECTED_MODEL, content=callback)
    def render(messages, *, tokenize, **_):
        text = ''.join('<|start|>'+m['role']+'<|message|>'+m['content']+'<|end|>' for m in messages)
        return list(text.encode()) if tokenize else text
    monkeypatch.setattr(tok, 'apply_chat_template', render)
    return dict(tokenizer=tok,client=object()), calls


def test_one_request_across_duplicate_publications_and_endpoints(tmp_path, monkeypatch):
    data = run.controls()
    kw, calls = runtime(monkeypatch, scripted)
    pure = (data[0], {k:v for k,v in data[1].items() if k[1]=='pure'}, data[2], dict(arms=['pure']))
    shared = tmp_path/'cache'
    a = run.evaluate(tmp_path/'pure', pure, shared, endpoint=f'http://127.0.0.1:{run.PORT}/v1', **kw)
    with pytest.raises(ValueError, match='Missing 2'):
        run.evaluate(tmp_path/'all', data, shared)
    b = run.evaluate(tmp_path/'all', data, shared, **kw)
    assert len(calls) == 3 and b['first_pass']['pure'] == a['first_pass']['pure']
    assert b['first_pass']['case_rag']['total'] == b['first_pass']['case_rag']['located'] == 2
    assert b['provisional_scores']['case_rag']['hits'] == 1
    assert b['unique']['one_pass_valid'] == 3 and b['unique']['no_passage'] == 1
    assert not b['formal_metrics_released'] and b['acceptance_threshold'] is None
    assert run.evaluate(tmp_path/'all', data, shared) == b
    assert not (tmp_path/'all/metrics.json').exists() and len(calls) == 3
    next((shared/'judgments').glob('*/raw.json')).write_text('{}')
    with pytest.raises(ValueError, match='evidence drift'):
        run.evaluate(tmp_path/'all', data, shared)


@pytest.mark.parametrize('bad,category', [('{}','content_failure'),
                         (json.dumps(decision(first=0)), 'invalid_range'),
                         (json.dumps(decision(first=None,last=None)), 'missing_match_evidence')])
def test_failures_keep_the_published_denominator_and_are_not_retried(tmp_path, monkeypatch, bad, category):
    rows, results, cov, _ = run.controls()
    only = {k:copy.deepcopy(v) for k,v in results.items() if k[1]=='pure'}
    data = rows, only, cov, dict(arms=['pure'])
    kw, calls = runtime(monkeypatch, lambda _:bad)
    for opts in (kw, {}, kw):
        r = run.evaluate(tmp_path/'report',data,tmp_path/'cache',**opts)
        assert r['first_pass']['pure'][category] == r['unique']['technical_failures'] == 1
        assert r['first_pass']['pure']['total'] == r['provisional_scores']['pure']['gt'] == 1
        assert r['provisional_scores']['pure']['hits'] == 0 and r['acceptance_threshold'] is None
    assert len(calls) == 1


def test_handoff_waits_for_released_libraries_and_preserves_an_operator_stop(tmp_path, monkeypatch):
    from scripts import step_judge_handoff as adoption
    monkeypatch.setattr(adoption.workflow, 'WORK', tmp_path)
    monkeypatch.setattr(adoption, 'ROOT', tmp_path/'control')
    current = {'configuration_sha256':'approved'}
    marker = tmp_path/'step/STOP'
    run.vr.atomic(marker, 'operator stop')
    run.vr.atomic(tmp_path/'workflow_exit.json', dict(configuration_sha256=adoption.PARENT,code=1))
    run.vr.atomic(tmp_path/'workflow_status.json', dict(stage='stopped',reason='STOP requested'))
    with pytest.raises(ValueError,match='Unexpected original'):
        adoption.take_boundary(current)
    assert run.vr.read(marker) == 'operator stop'
    run.vr.atomic(marker, adoption.boundary_marker(current))
    def unreleased(*_):
        raise ValueError('unreleased')
    monkeypatch.setattr(adoption.workflow.step,'released_libraries',unreleased)
    with pytest.raises(ValueError, match='unreleased'):
        adoption.take_boundary(current)
    assert marker.exists()
    monkeypatch.setattr(adoption.workflow.step,'released_libraries',lambda *_: {})
    run.vr.atomic(tmp_path/'libraries/construction_acceptance.json', {'passed':True})
    adoption.take_boundary(current)
    assert not marker.exists() and (adoption.ROOT/'handoff.json').is_file()
    run.vr.atomic(marker,'later operator stop')
    with pytest.raises(ValueError,match='separately requested'):
        adoption.take_boundary(current)
