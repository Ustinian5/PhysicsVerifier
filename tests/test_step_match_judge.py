import copy
import json
from pathlib import Path

import pytest

from rule_framework import step_match as matching
from scripts import print_verifier_prompts as prompts
from scripts import run_aligned_step_evaluation as step
from scripts import run_step_match_judge as judge
from tests.test_final_answer_revision import fake_runtime


def decision(disposition='same_root', ids=None, quote='I multiply 12 by 3 to obtain 36 m/s.'):
    return dict(disposition=disposition, root_ids=['E1'] if ids is None else ids,
                source_quote=quote, rationale='The diagnosis addresses the committed relation.')


def response(messages):
    assert [m['role'] for m in messages] == ['system', 'user']
    payload = json.loads(messages[1]['content'])
    assert set(payload) == {'problem', 'context', 'student_solution', 'published_diagnosis', 'active_reference_roots'}
    assert set(payload['published_diagnosis']) == {'message', 'quote'}
    message = payload['published_diagnosis']['message']
    if message.startswith('Multiplying'):
        return json.dumps(decision('opposite_claim'))
    if message.startswith('The unit'):
        return json.dumps(decision('other_claim', [], ''))
    return json.dumps(decision())


def fixture(monkeypatch, contents=response):
    _, _, tokenizer, calls = fake_runtime(monkeypatch, model=judge.base.r.EXPECTED_MODEL, content=contents)
    return tokenizer, calls


def test_blind_dedup_and_raw_replay(tmp_path, monkeypatch):
    data = judge.controls()
    for item in data[1].values():
        for diagnostic in item['diagnostics']:
            diagnostic['reference_ids'] = ['tree-secret']
            diagnostic['private_trace'] = 'checker-secret'
    tokenizer, calls = fixture(monkeypatch)
    live = judge.evaluate(tmp_path, data, client=object(), tokenizer=tokenizer)
    assert len(calls) == 3
    assert all('secret' not in str(call['messages']) for call in calls)
    assert live['judge']['published_diagnoses'] == 5
    assert live['scores']['case_rag']['precision'] == .5
    assert live['scores']['case_rag']['recall'] == 1
    assert live['scores']['flat_skills']['hits'] == 0 and live['scores']['flat_skills']['locatable'] == 2
    assert live['scores']['rule_tree']['gt'] == 1 and live['scores']['rule_tree']['hits'] == 0
    assert judge.evaluate(tmp_path, data, offline=True) == live and len(calls) == 3
    raw = next((tmp_path / 'judgments').glob('*/raw.json'))
    raw.write_text('{}')
    with pytest.raises(ValueError, match='evidence drift'):
        judge.evaluate(tmp_path, data, offline=True)
    assert len(calls) == 3


@pytest.mark.parametrize('value', [
    decision(ids=['E2']), decision(ids=[]), decision(ids=['E1', 'E1']), decision(quote='fabricated'),
    decision(quote=''), decision('other_claim'), decision('unclear'), dict(decision(), unexpected=1),
])
def test_invalid_or_foreign_decisions(value):
    answer = judge.controls()[0][0]['prediction']
    with pytest.raises(ValueError):
        matching.validate(value, answer, {'E1'})


def test_repaired_quote_and_opposite_direction_have_distinct_credit():
    rows, results, _, _ = judge.controls()
    row = rows[0]
    result = copy.deepcopy(results[row['id'], 'pure'])
    result['diagnostics'][0]['evidence']['quote'] = 'The student uses multiplication here'
    same = decision()
    checked = matching.score_row(row, result, [same])
    assert checked['hits'] == checked['locatable'] == 1
    assert checked['matches'][0]['prediction_span']['start'] > row['gt'][0]['end_char']
    opposite = decision('opposite_claim')
    assert matching.validate(opposite, row['prediction'], {'E1'}) == opposite
    assert matching.score_row(row, result, [opposite])['hits'] == 0
    unknown = decision('unclear', [], '')
    checked = matching.score_row(row, result, [unknown])
    assert checked['published'] == checked['unclear'] == 1 and checked['locatable'] == checked['hits'] == 0


def test_empty_or_failed_checkers_keep_recall_and_need_no_requests(tmp_path):
    rows, results, coverage, _ = judge.controls()
    for item in results.values():
        item.update(diagnostics=[], failed=True, status='failed')
    data = rows, results, coverage, {}
    got = judge.evaluate(tmp_path, data, offline=True)
    assert got['judge']['unique_diagnoses'] == 0
    assert all(s['gt'] == s['failed'] == 1 and s['published'] == 0 for s in got['scores'].values())


def test_incomplete_does_not_publish_and_does_not_retry_a_content_failure(tmp_path, monkeypatch):
    data = judge.controls()
    with pytest.raises(ValueError, match='3 missing'):
        judge.evaluate(tmp_path, data, offline=True)
    tokenizer, calls = fixture(monkeypatch, lambda ms: '{}' if 'misspelled' in ms[1]['content'] else response(ms))
    with pytest.raises(ValueError, match='1 content failures'):
        judge.evaluate(tmp_path, data, tokenizer=tokenizer, client=object())
    with pytest.raises(ValueError, match='1 content failures'):
        judge.evaluate(tmp_path, data, offline=True)
    assert len(calls) == 3 and not (tmp_path / 'metrics.json').exists()
    assert judge.vr.read(tmp_path / 'status.json')['state'] == 'incomplete'


def test_resume_uses_successful_raw_results_and_rejects_changed_publication(tmp_path, monkeypatch):
    data = judge.controls()
    tokenizer, calls = fixture(monkeypatch)
    original = judge.transport.request_once
    def interrupt(*args, **kwargs):
        if len(calls) == 1:
            raise ConnectionError('unavailable')
        return original(*args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(judge.transport, 'request_once', interrupt)
        with pytest.raises(ConnectionError):
            judge.evaluate(tmp_path, data, tokenizer=tokenizer, client=object())
    assert len(calls) == 1 and not (tmp_path / 'metrics.json').exists()
    judge.evaluate(tmp_path, data, tokenizer=tokenizer, client=object())
    assert len(calls) == 3
    data[1]['cart_speed', 'pure']['diagnostics'] = []
    with pytest.raises(ValueError, match='overwrite frozen evidence'):
        judge.evaluate(tmp_path, data, offline=True)


def test_hold_precedes_judge_and_prompt_is_exported(tmp_path):
    records = judge.controls()
    records[0][0]['id'] = next(iter(judge.frozen.sources.STEP_HOLD))
    with pytest.raises(ValueError, match='authorized cohort'):
        judge.prepare(tmp_path, *records)
    from rule_framework import step_match_lines
    assert step_match_lines.INSTRUCTION in prompts.render()


def test_normal_step_score_uses_only_the_saved_judge(tmp_path, monkeypatch):
    data = judge.controls()
    tokenizer, _ = fixture(monkeypatch)
    judge.evaluate(tmp_path / 'step_judge', data, tokenizer=tokenizer, client=object())
    bench = tmp_path / 'bench'
    def production(path, published):
        assert (path, published) == (bench, tmp_path)
        return data
    monkeypatch.setattr(judge, 'production', production)
    monkeypatch.setattr(step, 'score_items', lambda *args: pytest.fail('Legacy location scorer used'))
    report = step.score(tmp_path, {'configuration': {'step_match_judge': matching.VERSION, 'step_benchmark': str(bench)}})
    assert report == judge.vr.read(tmp_path / 'metrics.json')


def test_server_pipeline_runs_judge_after_publication_and_skips_held_input(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import sys
    import types

    data = judge.controls()
    row = {k: data[0][0][k] for k in ('id', 'question', 'prediction', 'context')}
    held = {'id': next(iter(judge.frozen.sources.STEP_HOLD))}
    judge.vr.atomic(tmp_path / 'inputs.json', [held, row])
    for name in ('cases', 'skills', 'catalog'):
        judge.vr.atomic(tmp_path / (name + '.json'), {})
    controls = [dict(row, id='control' + str(i)) for i in range(6)]
    cfg = dict(prompt_profile=step.en.PROFILE, step_match_judge=matching.VERSION, step_benchmark=str(tmp_path),
               arms=list(judge.ARMS), api_base='http://127.0.0.1:18083/v1', controls=controls,
               cohort={'held_ids': [held['id']]}, sample_ids=[row['id']])
    plan = dict(configuration=cfg, configuration_sha256=judge.vr.digest(cfg))
    monkeypatch.setattr(step, 'verify', lambda _: plan)
    original_exists = Path.exists
    monkeypatch.setattr(step.Path, 'exists', lambda p: p == Path('/proc') or original_exists(p))
    monkeypatch.setattr(step.base.runtime, 'wait_detached', lambda *args: None)
    monkeypatch.setattr(judge.node, 'service', lambda *args: nullcontext(cfg['api_base']))
    monkeypatch.setattr(step.a, 'configured', lambda *args: nullcontext())
    monkeypatch.setattr(step.base.p, 'FlatMemory', lambda _: None)
    monkeypatch.setattr(step.base.p, 'TreeMemory', lambda _: None)
    tokenizer, calls = fixture(monkeypatch)
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)))
    checked = []
    def check(out, item, phase, arm, *args):
        assert item['id'] != held['id'] and not calls
        checked.append((phase, item['id'], arm))
        return {'status': 'completed', 'protocol_audit': {'passed': True}}
    monkeypatch.setattr(step.a.bench, 'run_condition', check)
    # This orchestration fixture substitutes the transport without creating any raw requests.
    verified = []
    monkeypatch.setattr(step, 'verify_blind_delivery',
                        lambda out, phase, item, arm, *args: verified.append((phase, item['id'], arm)))
    def production(bench, out):
        assert bench == out == tmp_path and len(checked) == 28 and checked == verified
        return data
    monkeypatch.setattr(judge, 'production', production)
    step.run(tmp_path)
    assert len(calls) == 3
    assert judge.vr.read(tmp_path / 'metrics.json')['protocol'] == matching.VERSION
    assert (tmp_path / 'runner.exit_code').read_text() == '0'
