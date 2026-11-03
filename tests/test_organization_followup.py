import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rule_framework import organization_followup as p
from scripts import run_organization_followup as run
from tests.test_organization_comparison import ROW, QUERIES, Tokenizer


def extracted(skills, status='supported_method'):
    return dict(source_review=dict(status=status, evidence_ids=['prediction:0'],
                                  finding='Fixture only: independently reviewed source.'), skills=skills)


def test_partial_shared_stem_excluded_despite_different_subquestions():
    stem = ('An extremely thin current carrying solenoid extends from the origin along the negative axis '
            'and carries a low frequency alternating current with known cross sectional area and turns per length. '
            'Consider the solid angle enclosed by a triangular conductor surrounding the axis at a prescribed distance. ')
    heldout = [dict(id='test', question='Unrelated introductory material. '*80+stem+' Derive the potential.')]
    screen = p.FamilyScreen(heldout)
    row = dict(id='build', question=stem+' Compute triangular loop thermal power. '+'Different detail. '*150)
    found = screen.matches(row)
    assert any(x['heldout_id'] == 'test' for x in found)
    assert not screen.matches(dict(id='clean', question='A uniform rod leans against a smooth vertical wall.'))
    # A shared stem hidden in the case memory must also be excluded.
    found = screen.matches(dict(id='notes', question='A different question.', prediction=row['question']))
    assert any(x['heldout_id'] == 'test' for x in found)


@pytest.mark.skipif(not (run.ROOT/'results/organization_comparison_99_20261102/sources.json').exists(),
                    reason='Private frozen experiment unavailable')
def test_actual_dirac_subproblem_is_rejected_without_using_bench_labels():
    root = run.ROOT/'results/organization_comparison_99_20261102'
    screen = p.FamilyScreen(run.vr.read(root/'bench_questions.json'))
    row = next(x for x in run.vr.read(root/'sources.json') if x['id'] == 'trainphysics_6997')
    assert any(x['reason'] == 'shared_subproblem_prose' and 'diracstring' in x['normalized_shared_span']
               for x in screen.matches(row))


def test_skills_are_complete_methods_with_a_hard_cap_not_posthoc_truncation():
    skill = dict(title='核验代入', text='从题目独立换算单位再代入；将所得数量级与原回答比较。')
    values = [extracted([]), extracted([skill]), extracted([skill, dict(skill, title='另一方法')])]
    for value in values: assert p.validate_skills(value) is value
    too_many = extracted([skill]*3); before = copy.deepcopy(too_many)
    with pytest.raises(ValueError, match='do not truncate'): p.validate_skills(too_many)
    assert too_many == before
    messages = p.skill_messages(dict(ROW, prediction=ROW['prediction']+'<|end|>',
                                    answer='SECRET', rules='SECRET', case_note='SECRET'))
    assert 'SECRET' not in json.dumps(messages) and '<|' not in messages[1]['content']
    assert '同一方法' in messages[0]['content']
    assert messages[1]['content'].endswith(p.SKILL_PROMPT)
    assert '[prediction:0]' in messages[1]['content']
    assert '禁止猜测倍率' in messages[0]['content'] and '跨条引用' in messages[0]['content']


def test_independent_journal_uses_bound_original_fields_not_case_notes(monkeypatch):
    journal = run.IndependentSkillJournal.__new__(run.IndependentSkillJournal)
    journal.sources = {ROW['id']: dict(ROW, answer='SECRET_ANSWER')}
    payload = {'case': {'source_ids': [ROW['id']], 'text': 'SECRET_FALSE_REVIEW'}}
    rendered = json.dumps(journal.messages('extract_skill', payload), ensure_ascii=False)
    assert 'SECRET' not in rendered and ROW['prediction'] in rendered
    good = extracted([dict(title='check', text='Check forces and dimensions.')])
    calls = []
    def generate(*args):
        calls.append(args)
        return good
    monkeypatch.setattr(run.bench.ProcedureJournal, '__call__', generate)
    assert journal('extract_skill', ROW['id'], payload) == good
    assert [x[2] for x in calls] == [ROW['id']+':draft', ROW['id']]
    assert calls[1][3]['draft'] == good
    audit = journal.messages('extract_skill', calls[1][3])
    assert len(audit) == 4 and audit[-1]['content'] == p.AUDIT_PROMPT
    assert audit[0:2] == journal.messages('extract_skill', payload)
    good['source_review']['evidence_ids'] = ['prediction:999999']
    with pytest.raises(run.base.old.tree.ContentFailure, match='Unknown evidence address'):
        journal('extract_skill', ROW['id'], payload)
    with pytest.raises(ValueError, match='Unknown skill source'):
        journal.messages('extract_skill', {'case': {'source_ids': ['unknown']}})


def test_uncertain_source_cannot_publish_skills_or_silently_drop_a_bad_review():
    value = extracted([dict(title='check', text='procedure')], status='uncertain')
    with pytest.raises(ValueError, match='must not generate'):
        p.validate_skills(value)
    value['skills'] = []
    assert p.validate_source_evidence(value, ROW) == value


def test_exact_deduplication_keeps_all_sources_and_failures_without_filling_quota():
    rows = [dict(ROW, id=str(i)) for i in range(4)]
    skill = dict(title='代入', text='独立代入原式并核对单位。')
    outcomes = [dict(id=str(i), status='collected', value=extracted([skill] if i < 2 else [])) for i in range(3)]
    outcomes.append(dict(id='3', status='content_failure', reason='more_than_two'))
    entries, counts = p.materialize_skills(rows, outcomes)
    assert len(entries) == 1 and entries[0]['source_ids'] == ['0', '1']
    assert counts['valid_empty_sources'] == counts['content_failure'] == 1
    assert counts['target_count_is_not_a_quota'] is True
    assert 'source_review' in outcomes[0]['value'] and 'source_review' not in entries[0]


def test_unexpected_library_size_stops_before_benchmark_without_changing_entries(tmp_path, monkeypatch):
    from types import SimpleNamespace
    memory = SimpleNamespace(docs={str(i): i for i in range(999)})
    monkeypatch.setattr(run, 'ORIGINAL_BUILD', lambda *args: memory)
    with pytest.raises(ValueError, match='no filler or resampling'):
        run.build(tmp_path, {'configuration': {'source_count': 1621}}, None)
    assert len(memory.docs) == 999
    memory.docs['999'] = 999
    assert run.build(tmp_path, {'configuration': {'source_count': 1621}}, None) is memory


def test_reused_query_is_not_regenerated_and_completed_results_are_not_repeated(tmp_path, monkeypatch):
    evidence = dict(usage={'prompt_tokens': 20, 'completion_tokens': 10}, turn=1,
                    model_ids=[run.base.r.EXPECTED_MODEL], finish_reasons=['stop'], shared_preparation=True)
    for key in ('request', 'response', 'raw_sse'):
        f = tmp_path/('original_'+key+'.json')
        run.vr.atomic(f, {'model_ids': [run.base.r.EXPECTED_MODEL]})
        evidence[key+'_path'] = str(f); evidence[key+'_sha256'] = run.vr.sha(f)
    prefix = dict(configuration_sha256='old', input_sha256=run.vr.digest(ROW), queries=QUERIES,
                  evidence=evidence, files={})
    run.vr.atomic(tmp_path/'frozen_prefixes.json', {ROW['id']: prefix})
    plan = dict(configuration_sha256='new', configuration=dict(api_base='http://127.0.0.1:18083/v1',
         limits=dict(run.base.runtime.LIMITS, review_protocol=run.base.old.continuous.PROTOCOL)))
    calls = []
    def transport(client, **kw):
        calls.append(kw)
        assert len(kw['messages']) == 4  # Only CHECK; no PREPARE request.
        record = dict(model_ids=[run.base.r.EXPECTED_MODEL], finish_reasons=['stop'],
            content=json.dumps({'diagnostics': []}), usage={'prompt_tokens': 20, 'completion_tokens': 10})
        run.vr.atomic(kw['request_path'], {'request': {'messages': kw['messages']}})
        run.vr.atomic(kw['raw_path'], record); run.vr.atomic(kw['response_path'], record)
        return record
    monkeypatch.setattr(run.base.runtime.transport, 'chat_response', transport)
    original_protocol = run.bench.protocol
    with run.adapted(tmp_path, plan):
        assert run.base.p.PILOT_SIZE == p.PILOT_SIZE == 24
        first = run.bench.run_condition(tmp_path, ROW, 'formal', 'pure', None, plan, Tokenizer(), None)
        second = run.bench.run_condition(tmp_path, ROW, 'formal', 'pure', None, plan, Tokenizer(), None)
        assert first == second and first['trace']['response_evidence'][0] == evidence
        with pytest.raises(ValueError, match='input changed'):
            run.base.query(tmp_path, dict(ROW, prediction='changed'), plan, None, None)
    assert run.bench.protocol is original_protocol and len(calls) == 1
    assert run.vr.read(tmp_path/'frozen_prefixes.json')[ROW['id']]['configuration_sha256'] == 'old'


def test_first_check_full_request_checked_before_any_network_io():
    calls = []
    class Client:
        def post(self, endpoint, **kw): calls.append(kw); return 'ok'
    expected = dict(messages=[{'role': 'user', 'content': 'input'}], temperature=0, max_tokens=100)
    client = run.ExactFirstRequestClient(Client(), expected)
    with pytest.raises(ValueError, match='differs'): client.post('local', json=dict(expected, temperature=.7))
    assert not calls
    assert client.post('local', json=copy.deepcopy(expected)) == 'ok'


def mock_repeat_results(tmp_path, monkeypatch):
    monkeypatch.setattr(run.base.runtime, 'audit_result', lambda x: None)
    monkeypatch.setattr(run.base.runtime, 'healthy_result', lambda x: x['status'] == 'completed')
    row = dict(ROW, id=p.TARGET_ID); run.vr.atomic(tmp_path/'inputs.json', [row])
    run.vr.atomic(tmp_path/'frozen_prefixes.json', {row['id']: {'evidence': {'request_path': 'old-query'}}})
    for repeat in range(1, 6):
        for arm in run.base.p.ARMS:
            run.vr.atomic(run.repeat_path(tmp_path, repeat, arm, row), dict(id=row['id'],
                status='completed', configuration_sha256='bound', diagnostics=[
                    dict(message='alpha is wrong: about 9.2e3 per metre', evidence={'quote': 'original'}, reference_ids=['private-arm'])],
                trace={'response_evidence': [{'request_path': 'old-query'}, {'request_path': f'new-{repeat}-{arm}'}]}))
    return dict(configuration_sha256='bound')


def test_repeat_semantics_not_inferred_from_numbers_locations_or_citations(tmp_path, monkeypatch):
    plan = mock_repeat_results(tmp_path, monkeypatch)
    report = run.score_repeats(tmp_path, plan)
    assert report['target_scores'] is None and report['pending_reviews'] == 20
    assert report['actual_new_check_requests'] == 20 and report['reused_query_requests'] == 1
    review = run.vr.read(tmp_path/'target_review.json')
    assert all('reference_ids' not in json.dumps(x['diagnostics']) and 'arm' not in x for x in review['records'])
    for entry in review['records']:
        entry['review'] = dict(actor='Codex assistant', human_gold=False, status='completed',
            target_verdict='correct_detection', diagnostic_indices=[0], notes='Fixture only: independently checked the target correction.')
    run.vr.atomic(tmp_path/'target_review.json', review)
    digest = run.vr.sha(tmp_path/'target_review.json')
    report = run.score_repeats(tmp_path, plan)
    assert all(v['correct_detection_rate'] == 1 and v['scheduled'] == 5 for v in report['target_scores'].values())
    assert run.vr.sha(tmp_path/'target_review.json') == digest
    # Tampered raw output or edited diagnostic must invalidate a previously filled judgment.
    review['records'][0]['diagnostics'][0]['message'] = 'changed'
    run.vr.atomic(tmp_path/'target_review.json', review)
    with pytest.raises(ValueError, match='output changed'): run.score_repeats(tmp_path, plan)


def test_repeat_partial_batch_cannot_report_rate(tmp_path, monkeypatch):
    plan = mock_repeat_results(tmp_path, monkeypatch)
    row = run.vr.read(tmp_path/'inputs.json')[0]
    run.repeat_path(tmp_path, 5, 'pure', row).unlink()
    report = run.score_repeats(tmp_path, plan)
    assert report['recorded_tasks'] == 19 and report['target_scores'] is None
    assert not (tmp_path/'target_review.json').exists()


def test_failed_repeat_stays_in_denominator_and_is_never_repaired(tmp_path, monkeypatch):
    plan = mock_repeat_results(tmp_path, monkeypatch)
    row = run.vr.read(tmp_path/'inputs.json')[0]
    path = run.repeat_path(tmp_path, 5, 'rule_tree', row)
    value = run.vr.read(path); value.update(status='failed', diagnostics=[])
    run.vr.atomic(path, value)
    monkeypatch.setattr(run.bench.recovery, 'retained_failure', lambda path, **kw: run.vr.read(path))
    monkeypatch.setattr(run.base, 'complete_content_failure', lambda value: True)
    run.score_repeats(tmp_path, plan)
    review = run.vr.read(tmp_path/'target_review.json')
    for entry in review['records']:
        if not entry['execution_failure']:
            entry['review'] = dict(actor='Codex assistant', human_gold=False, status='completed',
                target_verdict='correct_detection', diagnostic_indices=[0], notes='Fixture only, verified target.')
    run.vr.atomic(tmp_path/'target_review.json', review)
    report = run.score_repeats(tmp_path, plan)
    assert report['target_scores']['rule_tree']['correct_detection_rate'] == .8
    assert report['target_scores']['rule_tree']['counts']['technical_failure'] == 1
    assert run.vr.read(path) == value


def test_target_formula_reproduces_review_and_stays_out_of_generation_messages():
    rubric = p.target_rubric()
    assert rubric['correct_value_per_m'] == pytest.approx(9175.813618178812)
    assert rubric['human_gold'] is False and rubric['source_selected_after_observing_results'] is True
    rendered = json.dumps(run.base.p.query_messages(dict(ROW, rubric=rubric)), ensure_ascii=False)
    assert '9175' not in rendered and 'target_verdict' not in rendered


@pytest.mark.skipif(not (run.ROOT/'results/organization_comparison_99_20261102/plan.json').exists(),
                    reason='Private frozen experiment unavailable')
def test_real_case_four_arm_requests_replay_exactly_without_live_model(tmp_path, monkeypatch):
    parent = run.ROOT/'results/organization_comparison_99_20261102'
    plan = run.prepare(tmp_path, parent, 'case-repeat')
    row = run.vr.read(tmp_path/'inputs.json')[0]
    packets = run.vr.read(tmp_path/'frozen_packets.json')
    expected = run.vr.read(tmp_path/'first_check_requests.json')
    posts = []
    class Client:
        def post(self, endpoint, **kw): posts.append(kw['json'])
    def fake_transport(client, **kw):
        body = run.base.r.request_body(kw['model'], kw['messages'], kw['max_tokens'], 'vllm', None)
        body.update(stream=False, return_token_ids=True, reasoning_effort='medium'); body.pop('stream_options', None)
        client.post(kw['endpoint'], json=body)
        record = dict(model_ids=[run.base.r.EXPECTED_MODEL], finish_reasons=['stop'], content='{"diagnostics":[]}',
                      usage=dict(prompt_tokens=200, completion_tokens=20))
        run.vr.atomic(kw['request_path'], {'request': body, 'request_sha256': run.vr.digest(body)})
        run.vr.atomic(kw['raw_path'], record); run.vr.atomic(kw['response_path'], record)
        return record
    monkeypatch.setattr(run.base.runtime.transport, 'chat_response', fake_transport)
    with run.adapted(tmp_path, plan), patch.object(run.base.p, 'reference_packet',
            lambda row, arm, *args: copy.deepcopy(packets[arm])):
        for arm in run.base.p.ARMS:
            path = tmp_path/'repeat_01'
            value = run.bench.run_condition(path, row, 'formal', arm, None, plan, Tokenizer(),
                                             run.ExactFirstRequestClient(Client(), expected[arm]))
            assert value['status'] == 'completed'
            run.bench.run_condition(path, row, 'formal', arm, None, plan, Tokenizer(),
                                    run.ExactFirstRequestClient(Client(), expected[arm]))
    assert len(posts) == 4 and posts == [expected[a] for a in run.base.p.ARMS]
    assert all('correct_value_per_m' not in json.dumps(body) for body in posts)
    assert not (tmp_path/'queries').exists()  # Old prefix referenced in place, never regenerated.


def test_evidence_spans_preserve_all_source_text_without_json_reescaping():
    source = dict(ROW, prediction='a'*799+'\\\\frac{a}{b}\\n'+'b'*900)
    spans = p.source_spans(source)
    assert ''.join(v for k, v in spans.items() if k.startswith('prediction:')) == source['prediction']
    assert p.validate_source_evidence(extracted([]), source)
    value = extracted([]); value['source_review']['evidence_ids'] = ['prediction:0']*2
    with pytest.raises(ValueError, match='Invalid evidence addresses'): p.validate_skills(value)


def test_two_pass_initialization_preserves_binding_and_total_output_cap(tmp_path):
    config = {'model': {'max_tokens': 16384}, 'budget_limits': {'requests': 6000, 'input_tokens': 1800000000, 'output_tokens': 98304000}, 'prior_usage': run.base.old.ZERO_USAGE}
    run.vr.atomic(tmp_path/'sources.json', [ROW])
    journal = run.IndependentSkillJournal(tmp_path/'skills', config, Tokenizer())
    assert journal.binding == run.vr.digest(config)
    assert journal.config['max_tokens'] * p.EXTRACTION_PASSES == config['model']['max_tokens']
    assert config['model']['max_tokens'] == 16384
