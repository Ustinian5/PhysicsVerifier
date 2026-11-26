import json

import pytest

from scripts import repair_step_judge_locations as repair
from tests.test_final_answer_revision import fake_runtime
from tests.test_step_match_judge import decision


def lines(first, last):
    return dict(start_line=first, end_line=last)


def test_location_uses_original_characters_and_line_endings():
    answer = "速度\r\n\r\nα = 3\n速度\r\n"
    span = repair.selected_span(answer, lines(3, 4))
    assert span == dict(start=6, end=len(answer), quote="α = 3\n速度\r\n")
    assert repair.selected_span(answer, lines(None, None)) is None
    for invalid in [lines(None, 1), lines(True, 2), lines(2, 2), lines(3, 2), lines(0, 1),
                    lines(1, 5), lines("1", 1), dict(start_line=1, end_line=1, quote="速度")]:
        with pytest.raises(ValueError):
            repair.selected_span(answer, invalid)


def test_selection_preserves_root_direction_and_chosen_occurrence():
    rows, results, _, _ = repair.judge.controls()
    row = rows[0]
    diag = results[row['id'], 'pure']['diagnostics'][0]
    quote = diag['evidence']['quote']
    row['prediction'] = quote + '\n' + quote + '\n'
    original = repair.MATCH.messages(row, diag, row['roots'])
    draft = decision(quote='invented substrate quote')
    messages = repair.location_messages(original, draft)
    payload = json.loads(messages[1]['content'])
    assert [m['role'] for m in messages] == ['system', 'user']
    assert set(payload) == {'problem', 'context', 'published_diagnosis', 'active_reference_roots',
                            'numbered_student_solution', 'earlier_judgment'}
    assert payload['numbered_student_solution'] == [[1, quote], [2, quote]]
    assert payload['earlier_judgment'] == dict(disposition='same_root', root_ids=['E1'])
    assert draft['source_quote'] not in messages[1]['content']
    result = results[row['id'], 'pure']
    supported, span = repair.merge_location(row, draft, lines(2, 2))
    score = repair.score_row(row, result, [supported], [span])
    assert score['hits'] == 1 and score['matches'][0]['prediction_span']['start'] == len(quote) + 1
    for name in ('same_root', 'opposite_claim'):
        effective, empty = repair.merge_location(row, decision(name), lines(None, None))
        assert empty is None and repair.score_row(row, result, [effective], [empty])['hits'] == 0
        assert effective['disposition'] == ('unclear' if name == 'same_root' else name)
    opposite, span = repair.merge_location(row, decision('opposite_claim'), lines(2, 2))
    assert repair.score_row(row, result, [opposite], [span])['hits'] == 0


@pytest.mark.parametrize('valid', [True, False])
def test_only_target_contacted_once_and_scores_require_complete_replay(tmp_path, monkeypatch, valid):
    model = repair.judge.base.r.EXPECTED_MODEL
    _, plan, tokenizer, calls = fake_runtime(monkeypatch, model=model, content=json.dumps(lines(1 if valid else 0, 1)))
    config = plan['configuration']
    config.update(model=model, authorized_keys=['pending'], max_tokens=4096)
    plan['configuration_sha256'] = repair.vr.digest(config)
    rows, results, coverage, _ = repair.judge.controls()
    row = rows[0]
    result = results[row['id'], 'pure']
    result['diagnostics'] = [*result['diagnostics'], *result['diagnostics']]
    messages = repair.MATCH.messages(row, result['diagnostics'][0], row['roots'])
    data = rows, {(row['id'], 'pure'): result}, coverage, dict(arms=['pure'])
    work = tmp_path / 'step_judge/location_repair'
    prepared = (work, plan, data, [dict(id=row['id'], arm='pure', keys=['kept', 'pending'])],
                {'kept': (row, messages), 'pending': (row, messages)},
                {'kept': decision('other_claim', [], '')}, {'pending': decision(quote='nonverbatim')})
    with pytest.raises(ValueError, match='1 missing'):
        repair.evaluate(tmp_path, prepared)
    for online in (True, False, True):
        options = dict(client=object(), tokenizer=tokenizer) if online else {}
        if valid:
            report = repair.evaluate(tmp_path, prepared, **options)
            assert report == repair.vr.read(tmp_path / 'metrics.json')
            assert report['scores']['pure']['hits'] == 1 and report['judge']['valid'] == 2
        else:
            with pytest.raises(ValueError, match='1 invalid'):
                repair.evaluate(tmp_path, prepared, **options)
            assert not (tmp_path / 'metrics.json').exists()
    assert len(calls) == 1 and {f.name for f in (work / 'calls').iterdir()} == {'pending'}
