import json
from pathlib import Path

import pytest

from rule_framework import verifier_english as en
from scripts import frozen_verifier_exam_answers as exams
from scripts import run_aligned_step_evaluation as step
from scripts import run_final_answer_revision as final
from scripts import run_step_match_judge as judge


def sample(sid='example'):
    return dict(id=sid, question='Find the speed.', prediction='The speed is 3 m/s.', context='')


@pytest.mark.parametrize('task,field,key', [('step', 'answers', 'label'), ('final', 'rows', 'answer')])
def test_formal_source_bound_to_exact_answer_and_full_cohort(tmp_path, monkeypatch, task, field, key):
    row = sample()
    parts = ['4 m/s', '']
    item = {key: parts, 'input_sha256': exams.vr.digest(row)}
    path = tmp_path / (task + '_exam_reference/source_answers.json')
    exams.vr.atomic(path, dict(dataset_sha256=exams.DATA_SHA, **{field: {row['id']: item}}))
    monkeypatch.setitem(exams.RELEASES, task, (exams.vr.sha(path), field, key))
    files = {}
    answers = exams.official(tmp_path, task, [row], files)
    assert set(files) == {str(path)} and answers[row['id']]['label'] == parts
    with pytest.raises(ValueError, match='different 30B input'):
        exams.official(tmp_path, task, [dict(row, prediction='another answer')], {})
    with pytest.raises(ValueError, match='cohort'):
        exams.official(tmp_path, task, [row, sample('missing')], {})
    config = dict(prompt_profile=en.PROFILE, exam_reference_required=True, check_exam_references=answers)
    assert final.check_exam_answer(row, config) == step.base.check_options(row, config)['official_reference'] == parts
    with pytest.raises(ValueError, match='missing'):
        final.check_exam_answer(row, dict(prompt_profile=en.PROFILE, exam_reference_required=True))


def test_development_reads_original_positions_and_reuses_synthetic_controls(tmp_path, monkeypatch):
    synthetic = [exams.source_payload(c) for c in exams.CONTROLS[:2]]
    real = [dict(sample('cl_2_' + str(i)), question=f'Question {i}') for i in range(4)]
    path = tmp_path / 'aligned_old99/final_revision_balanced/pilot_inputs.json'
    exams.vr.atomic(path, real + synthetic)
    data = tmp_path / 'data.json'
    records = [dict(index=i, prompt=r['question'], label=[f'part {i}', '']) for i, r in enumerate(real)]
    exams.vr.atomic(data, [dict(rollout_id=2, samples=records)])
    monkeypatch.setattr(exams, 'DATA_SHA', exams.vr.sha(data))
    monkeypatch.setattr(exams, 'PILOT_SHA', exams.vr.sha(path))
    rows, answers = exams.development(tmp_path, data, {})
    assert rows == real + synthetic and answers[real[2]['id']]['label'] == ['part 2', '']
    assert answers[synthetic[1]['id']]['label'] == ['10 m/s']
    records[0]['prompt'] = 'Foreign question'
    exams.vr.atomic(data, [dict(rollout_id=2, samples=records)])
    monkeypatch.setattr(exams, 'DATA_SHA', exams.vr.sha(data))
    with pytest.raises(ValueError, match='original raw position'):
        exams.development(tmp_path, data, {})


def test_direct_delivery_checks_actual_model_tokens_and_refuses_label_leak(tmp_path):
    row = sample()
    parts = ['4 m/s']
    cfg = dict(prompt_profile=en.PROFILE, exam_reference_required=True,
               check_exam_references={row['id']: exams.binding(row, parts)})
    msg = step.base.p.direct_messages(row, official_reference=parts)
    folder = step.base.path_for(tmp_path, 'formal', 'pure', row).parent / 'turn_01'
    step.vr.atomic(folder / 'request.json', dict(request=dict(messages=msg)))
    step.vr.atomic(folder / 'response.sse', dict(prompt_token_ids=[1]))
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return visible
    visible = msg[-1]['content']
    step.verify_exam_delivery(tmp_path, 'formal', row, 'pure', cfg, Tokenizer(), {})
    visible = 'The reference was truncated'
    with pytest.raises(ValueError, match='model-visible'):
        step.verify_exam_delivery(tmp_path, 'formal', row, 'pure', cfg, Tokenizer(), {})
    payload = json.loads(msg[-1]['content'])
    msg[-1]['content'] = json.dumps(dict(payload, root_labels=['E1']))
    step.vr.atomic(folder / 'request.json', dict(request=dict(messages=msg)))
    with pytest.raises(ValueError, match='Extra or missing'):
        step.verify_exam_delivery(tmp_path, 'formal', row, 'pure', cfg, Tokenizer(), {})


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
