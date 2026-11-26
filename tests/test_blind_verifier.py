import json
from types import SimpleNamespace

import pytest

from scripts import run_aligned_step_evaluation as step
from scripts import run_final_answer_revision as final
from scripts import run_step_match_judge as judge


def sample(sid='example'):
    return dict(id=sid, question='Find the speed.', prediction='The speed is 3 m/s.', context='')


def test_fixed_development_inputs_have_only_public_fields_and_are_disjoint(tmp_path, monkeypatch):
    path = tmp_path / 'pilot_inputs.json'
    rows = [sample(str(i)) for i in range(6)]
    step.vr.atomic(path, rows)
    monkeypatch.setattr(step, 'PILOT_SHA', step.vr.sha(path))
    files = {}
    assert step.frozen_development_inputs(path, {'heldout'}, files) == rows
    assert files == {str(path): step.PILOT_SHA}
    with pytest.raises(ValueError, match='held-out public rows'):
        step.frozen_development_inputs(path, {'3'}, {})
    step.vr.atomic(path, [dict(r, reference=['PRIVATE']) for r in rows])
    with pytest.raises(ValueError, match='Different fixed'):
        step.frozen_development_inputs(path, set(), {})
    monkeypatch.setattr(step, 'PILOT_SHA', step.vr.sha(path))
    with pytest.raises(ValueError, match='public rows'):
        step.frozen_development_inputs(path, set(), {})


@pytest.mark.parametrize('arm', step.base.p.ARMS)
def test_blind_delivery_checks_saved_requests_and_actual_model_input(tmp_path, arm):
    row = dict(sample(), reference='PRIVATE ANSWER', roots=['PRIVATE ROOT'])
    queries = {'queries': ['mean speed']}
    lib = step.base.p.FlatMemory([dict(id='historical', kind='case' if arm == 'case_rag' else 'skill',
                                      title='Kinematics', text='Historical method', source_ids=['train'])])
    memory = SimpleNamespace(memory=lib) if arm == 'rule_tree' else lib
    rendered = []
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return rendered[ids[0]]
    def write_call(folder, raw_name, messages):
        raw = ''.join('<|start|>user<|message|>' + m['content'] + '<|end|>'
                      for m in messages if m['role'] == 'user')
        step.vr.atomic(folder / raw_name, dict(prompt_token_ids=[len(rendered)]))
        rendered.append(raw)
        step.vr.atomic(folder / 'request.json', dict(request=dict(messages=messages)))
    dest = step.base.path_for(tmp_path, 'formal', arm, row).parent
    if arm == 'pure':
        expected = step.base.p.direct_messages(row)
        turn = dest / 'turn_01'
    else:
        shared = tmp_path / 'queries' / step.vr.digest(row['id'])[:24]
        write_call(shared, 'raw.json', step.base.p.query_messages(row, english=True))
        step.vr.atomic(shared / 'result.json', dict(queries=queries))
        step.vr.atomic(dest / 'reference_packet.json', dict(packet=dict(retrieval=dict(provided_complete_ids=['historical']))))
        expected = step.base.p.continuation_messages(row, queries, [lib.view('historical')], english=True, arm=arm)
        turn = dest / 'turn_02'
    write_call(turn, 'response.sse', expected)
    def verify():
        step.verify_blind_delivery(tmp_path, 'formal', row, arm, memory, Tokenizer(), {})
    verify()
    assert 'PRIVATE' not in str(expected)
    rendered[-1] += '<|start|>user<|message|>PRIVATE ANSWER<|end|>'
    with pytest.raises(ValueError, match='model-visible'):
        verify()
    expected[-1]['content'] += 'PRIVATE ANSWER'
    step.vr.atomic(turn/'request.json', dict(request=dict(messages=expected)))
    with pytest.raises(ValueError, match='blind verifier request'):
        verify()


def test_legacy_libraries_cannot_be_used_for_reference_guided_experiment(tmp_path):
    cfg = {'root': str(tmp_path)}
    step.vr.atomic(tmp_path/'plan.json', dict(configuration=cfg, configuration_sha256=step.vr.digest(cfg)))
    with pytest.raises(ValueError, match='official-answer-guided first review'):
        step.released_libraries(tmp_path, {})


@pytest.mark.parametrize('foreign_stage', [None, 'extract_experience', 'repair_experience'])
def test_library_uses_the_official_answer_in_the_actual_first_review_and_repair(tmp_path, foreign_stage):
    row = dict(sample(), official_reference={'label': ['4 m/s', '']})
    value = dict(sample_id=row['id'], semantic_audit={'status': 'error_found'})
    step.vr.atomic(tmp_path/'semantic_experience.json', [value])
    called, evidence = [], {}
    for stage in ('extract_experience', 'repair_experience'):
        path = tmp_path/'raw'/stage/'sample'/'request_001.json'
        payload = dict(sample_id=row['id'], question=row['question'],
                       prediction_lines=[{'line': 1, 'text': row['prediction']}])
        payload[step.en.EXAM_REFERENCE] = ['unrelated'] if stage == foreign_stage else row['official_reference']['label']
        step.vr.atomic(path, dict(status='received', request={'messages': [{}, {'content': json.dumps(payload)}]}))
        evidence[str(path)] = step.vr.sha(path)
        called.append(dict(file=str(path.relative_to(tmp_path)), sha256=evidence[str(path)]))
    record = tmp_path/'experiences'/(row['id']+'.json')
    step.vr.atomic(record, dict(value=value, evidence=called))
    evidence[str(record.resolve())] = step.vr.sha(record)
    if foreign_stage:
        with pytest.raises(ValueError, match='its own official answer'):
            step.verify_library_first_reviews(tmp_path, evidence, [row])
    else:
        step.verify_library_first_reviews(tmp_path, evidence, [row])


def test_single_arm_judge_never_creates_three_empty_comparators(tmp_path):
    rows, results, coverage, _ = judge.controls()
    item = dict(results[(rows[0]['id'], 'pure')], diagnostics=[])
    data = rows, {(rows[0]['id'], 'pure'): item}, coverage, dict(arms=['pure'])
    report = judge.evaluate(tmp_path, data, offline=True)
    assert set(report['scores']) == {'pure'} and report['recorded_tasks'] == {'pure': 1}
    assert report['scores']['pure']['gt'] == 1
    with pytest.raises(ValueError, match='Unexpected published tasks'):
        judge.prepare(tmp_path, rows, results, coverage, dict(arms=['pure']))


def test_final_registered_and_previously_held_ids_are_accounted_for():
    a, b = sample('a'), sample('b')
    cfg = dict(registered_questions=2, eligible_ids=['b'], held_ids=['a'])
    assert final.eligible_inputs([a, b], cfg) == [b]
    with pytest.raises(ValueError, match='cohort mismatch'):
        final.eligible_inputs([a, b], dict(cfg, eligible_ids=['a']))
