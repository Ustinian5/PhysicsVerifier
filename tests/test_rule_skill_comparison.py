import copy
import json

import pytest

from rule_framework import rule_skill_comparison as p, bounded_tree as tree, verified_resume as vr
from scripts import run_rule_skill_comparison as run
from scripts import continuous_context_v3 as continuous
from tests.test_bounded_tree import TAXONOMY, Caller, engine, finish
from tests.test_continuous_context_v2 import run as execute_sample, ROW, BAD, PATCH


SKILL = {'title': '合力与加速度', 'goal': '求惯性系中的加速度',
         'when_to_use': '恒定质量，使用合力', 'procedure': ['先定义系统', '按a=F/m计算'],
         'pitfalls': ['不能以单个力代替合力'], 'verification': ['验证单位m/s²和F→0极限']}
VALUE = {'solution_status': 'complete', 'skills': [SKILL], 'uncertainties': []}
QUESTION = {'id': 'training_q', 'question': '质量2kg受6N合力，求加速度。', 'context': ''}


def test_construction_projection_drops_old_answers_and_labels_and_preserves_question():
    row = dict(QUESTION, prediction='SECRET_30B', answer='SECRET_GT', errors=['SECRET_ERROR'])
    projected = p.questions_only([row])
    assert projected == [QUESTION]
    first = p.skill_messages('solve_own', {'row': row})
    second = p.skill_messages('extract_skill', {'row': row, 'solution': 'a=F/m=3m/s²。'})
    assert 'SECRET' not in json.dumps([first, second])
    assert first == second[:2] and second[2]['role'] == 'assistant'
    assert second[2]['content'] == 'a=F/m=3m/s²。'
    assert second[-1]['content'] == p.DISTILL
    assert 'skills' not in first[0]['content']


def test_valid_empty_skills_are_not_a_failure_or_regenerated():
    calls = []
    def journal(stage, key, payload):
        calls.append((stage, key))
        return {'solution': '无法确定。'} if stage == 'solve_own' else {
            'solution_status': 'insufficient_information', 'skills': [], 'uncertainties': ['缺图']}
    result = run.collect_one(journal, QUESTION)
    assert result['status'] == 'collected' and not result['value']['skills']
    assert len(calls) == 2


@pytest.mark.parametrize('stage', ['solve_own', 'extract_skill'])
def test_invalid_content_is_terminal_without_semantic_reroll(stage):
    calls = []
    def journal(current, key, payload):
        calls.append(current)
        if current == stage:
            raise tree.ContentFailure('length_truncated')
        return {'solution': 'a=3'}
    outcome = run.collect_one(journal, QUESTION)
    assert outcome['status'].startswith('invalid_')
    assert calls == (['solve_own'] if stage == 'solve_own' else ['solve_own', 'extract_skill'])


def test_transport_or_identity_error_cannot_become_empty_skill():
    with pytest.raises(ValueError, match='identity'):
        run.collect_one(lambda *a: (_ for _ in ()).throw(ValueError('identity mismatch')), QUESTION)


@pytest.mark.parametrize('change', [
    {'skills': [SKILL] * 3}, {'solution_status': 'correct'}, {'uncertainties': 'none'},
    {'skills': [dict(SKILL, procedure=[])]}, {'skills': [dict(SKILL, expected_answer='leak')]},
])
def test_bad_skill_schema_rejected(change):
    with pytest.raises(ValueError):
        p.validate_skills(dict(VALUE, **change))


def test_exact_dedup_merges_sources_and_navigation_keeps_all_native_content(tmp_path):
    outcomes = [{'id': 'q1', 'status': 'collected', 'value': VALUE},
                {'id': 'q2', 'status': 'collected', 'value': copy.deepcopy(VALUE)},
                {'id': 'q3', 'status': 'invalid_solution'}]
    materials = p.materialize(outcomes, TAXONOMY)
    assert len(materials['rules']) == 1
    leaf = materials['rules'][0]
    assert leaf['sample_ids'] == ['q1', 'q2'] and leaf['count'] == 2
    assert json.loads(leaf['check_logic']) == SKILL
    products = finish(engine(tmp_path, materials, Caller()))
    assert products['counts']['all_candidates_preserved']
    corpus = run.r.Corpus(products['catalog'], tree=True)
    assert json.loads(corpus.leaves[leaf['rule_id']]['check_logic']) == SKILL


def test_same_checking_prompt_history_budget_and_quote_repair_for_both_arms(tmp_path):
    original = continuous.RULES
    messages = []
    with run.configured_evaluation():
        for arm in p.ARMS:
            result, calls = execute_sample(tmp_path / arm, 'C4', [
                {'action': 'search_rules', 'arguments': {'query': '合力', 'mode': 'bm25', 'limit': 6}},
                BAD, PATCH], review_protocol=continuous.PROTOCOL)
            assert result['status'] == 'completed'
            assert result['trace']['output_token_limit_reserved'] == 150
            assert p.KNOWLEDGE in calls[0]['messages'][0]['content']
            assert any(x['role'] == 'tool' for x in calls[-1]['messages'])
            assert result['diagnostics'][0]['message'] == BAD['diagnostics'][0]['message']
            messages.append(calls[0]['messages'])
            again, repeated = execute_sample(tmp_path / arm, 'C4', [], review_protocol=continuous.PROTOCOL)
            assert again == result and not repeated
    assert messages[0] == messages[1] and continuous.RULES == original


def test_progress_stores_no_historical_solutions_and_outcome_chain_is_checkable(tmp_path):
    binding = 'test'
    row = QUESTION
    record = {'configuration_sha256': binding, 'question_sha256': vr.digest(row),
              'outcome': {'id': row['id'], 'status': 'collected', 'solution': 'a=3', 'value': VALUE}}
    vr.atomic(run.outcome_path(tmp_path, row), record)
    outcomes, chain = run.read_outcomes(tmp_path, [row], binding)
    assert outcomes == [record['outcome']] and chain == vr.digest(['', vr.digest(record)])
    with pytest.raises(ValueError, match='source/plan'):
        run.read_outcomes(tmp_path, [dict(row, question='changed')], binding)


def test_active_navigation_counts_as_lookup_without_forcing_an_extra_search():
    root = {'tool': 'read_tree', 'arguments': {'node_id': 'root'}, 'result': {}}
    child = {'tool': 'read_tree', 'arguments': {'node_id': 'domain:mechanics'}, 'result': {}}
    assert not p.actual_lookup([root])
    assert p.actual_lookup([root, child])
    note = p.progress_note([root, child], tree=True, remaining=5, remaining_output=8000)
    assert 'Before final, consult' not in note
    assert not p.actual_lookup([dict(child, error='missing node')])


def detail(sid, hits, n=1, total=3, failed=False):
    return {'id': sid, 'matched_claim_ids': hits, 'published_diagnostic_count': n,
            'gt_error_count': total, 'execution_failure': failed}


def test_cases_include_each_direction_and_do_not_invent_missing_positive_examples():
    a = {'details': [detail('rules_win', ['a']), detail('skills_win', []), detail('miss', []),
                     detail('failure', ['x'], failed=True)]}
    b = {'details': [detail('rules_win', []), detail('skills_win', ['b']), detail('miss', []),
                     detail('failure', [])]}
    selected = p.case_selection(a, b)
    assert selected['selection'][:2] == [
        {'category': 'rules_only_hits', 'id': 'rules_win'},
        {'category': 'skills_only_hits', 'id': 'skills_win'}]
    assert 'failure' not in {x['id'] for x in selected['selection']}
    assert selected['physics_review_status'] == 'pending_Codex_assistant' and not selected['human_gold']
    tied = p.case_selection(b, b)
    assert tied['eligible_counts']['rules_only_hits'] == 0
    assert tied['eligible_counts']['skills_only_hits'] == 0


def test_journal_reuses_completed_solution_and_rejects_mutated_raw(tmp_path, monkeypatch):
    calls = []
    config = {'model': {'model': run.r.EXPECTED_MODEL, 'api_base': 'http://127.0.0.1:18083/v1',
        'reasoning_effort': 'medium', 'max_tokens': 16384, 'attempts_per_job': 3,
        'max_request_input_bytes': 400000}, 'budget_limits': {'requests': 30, 'input_tokens': 1000000,
        'output_tokens': 1000000}, 'prior_usage': run.ZERO_USAGE}

    class Transport:
        def __init__(self, **kwargs):
            self.output = kwargs['output']
        def post(self, endpoint, body):
            calls.append(body)
            vr.atomic(self.output / 'request_001.json', {'http_status': 200, 'request': body,
                'client_metadata': self.record_metadata, 'response': {'id': 'response1',
                'model': run.r.EXPECTED_MODEL, 'choices': [{'finish_reason': 'stop'}]}})

    monkeypatch.setattr(run.b, 'Transport', Transport)
    monkeypatch.setattr(run.b.harmony, 'decode_response', lambda *a, **k: {'content': 'a=3m/s²'})
    journal = run.SkillJournal(tmp_path, config)
    result = journal('solve_own', QUESTION['id'], {'row': QUESTION})
    assert result == {'solution': 'a=3m/s²'} and len(calls) == 1
    again = run.SkillJournal(tmp_path, config)('solve_own', QUESTION['id'], {'row': QUESTION})
    assert again == result and len(calls) == 1
    raw = next(tmp_path.glob('arm_solve_own/*/request_001.json'))
    raw.write_text('{}')
    with pytest.raises(ValueError, match='Cached response evidence changed|Previously recorded'):
        journal('solve_own', QUESTION['id'], {'row': QUESTION})


def test_raw_tree_runtime_flags_are_not_bypassed_for_an_unrecognized_release(tmp_path):
    vr.atomic(tmp_path / 'release_manifest.json', {'configuration_sha256': run.TREE_CONFIG_SHA,
                                                'usable_for_runtime': False})
    with pytest.raises(ValueError, match='Unexpected rule-tree release'):
        run.rules_asset(tmp_path)


def test_publication_conserves_sources_skills_and_replays_without_model_calls(tmp_path, monkeypatch):
    folder = tmp_path / 'skills'
    folder.mkdir()
    rows = [QUESTION, dict(QUESTION, id='q2', question='另一个训练题。')]
    config = {'benchmarks': {}, 'bench_sha256': run.p.digest([])}
    binding = vr.digest(config)
    plan = {'configuration': config, 'configuration_sha256': binding}
    vr.atomic(tmp_path / 'plan.json', plan)
    vr.atomic(tmp_path / 'questions.json', rows)
    vr.atomic(tmp_path / 'taxonomy.json', TAXONOMY)
    journal = Caller()
    journal.index = {'binding': binding, 'files': {}}
    spaced = copy.deepcopy(VALUE)
    spaced['skills'][0]['procedure'].append('Retain  two spaces and\u3000an ideographic space.')
    for row in rows:
        path = run.outcome_path(folder, row)
        vr.atomic(path, {'configuration_sha256': binding, 'question_sha256': vr.digest(row),
            'outcome': {'id': row['id'], 'status': 'collected', 'solution': 'a=3', 'value': spaced}})
        journal.index['files'][str(path)] = vr.sha(path)
    _, chain = run.read_outcomes(folder, rows, binding)
    progress = vr.Progress(folder / 'collection_checkpoint.json', binding, {'cursor': 2, 'outcome_chain': chain})
    progress.save()
    vr.seal(folder / 'request_index.json', journal.index)
    for name in ('budget_checkpoint', 'runtime', 'pilot_summary', 'pilot_acceptance'):
        vr.atomic(folder / (name + '.json'), {})
    monkeypatch.setattr(run.b, 'benchmark_questions', lambda: ([], {}))
    monkeypatch.setattr(run.p, 'isolate_sources', lambda rows, heldout: (rows, []))
    monkeypatch.setattr(run, 'SOURCE_QUESTIONS', 2)
    run.publish_skills(tmp_path, plan, journal)
    asset = run.skill_asset(tmp_path, full=True)
    assert asset['counts']['source_questions'] == 2
    assert asset['counts']['generated_skills_before_exact_dedup'] == 2
    assert asset['counts']['input_candidates'] == asset['counts']['placed_rules'] == 1
    assert not asset['physical_quality_passed'] and not asset['human_gold']
    corpus = run.r.Corpus(vr.read(folder / 'catalog.json'), tree=True)
    assert json.loads(next(iter(corpus.leaves.values()))['check_logic']) == spaced['skills'][0]
    calls = len(journal.calls)
    run.publish_skills(tmp_path, plan, journal)
    assert len(journal.calls) == calls
    assert run.skill_asset(tmp_path)['release_sha256'] == asset['release_sha256']
    (folder / 'catalog.json').write_text('{}')
    with pytest.raises(ValueError):
        run.skill_asset(tmp_path)


def test_score_does_not_score_an_unfinished_arm_or_overwrite_case_review(tmp_path, monkeypatch):
    root = tmp_path / 'evaluation'
    root.mkdir()
    rows = [dict(ROW, id=str(i)) for i in range(50)]
    inp = tmp_path / 'inputs.json'
    vr.atomic(inp, rows)
    monkeypatch.setattr(run.runtime, 'INPUT', inp)
    plan = {'configuration_sha256': 'test', 'configuration': {'metric': 'proxy'}}
    monkeypatch.setattr(run, 'verify_eval', lambda _: plan)
    dataset = [dict(x, physics_error_gt=[]) for x in rows]
    unknown = {x['id']: [] for x in rows}
    monkeypatch.setattr(run.evaluation, 'reference_data', lambda _: (None, dataset, unknown))
    results = [dict(x, diagnostics=[], status='completed', protocol_audit={'passed': True}) for x in rows]
    monkeypatch.setattr(run, 'collect_evaluation', lambda *a: {'rules': results, 'skills': results[:49]})
    report = run.score(tmp_path)
    assert set(report['scores_all_50']) == {'rules'}
    assert not (root / 'case_review.json').exists()
    monkeypatch.setattr(run, 'collect_evaluation', lambda *a: {'rules': results, 'skills': results})
    run.score(tmp_path)
    review = vr.read(root / 'case_review.json')
    review['operator_note'] = 'A completed review must remain intact.'
    vr.atomic(root / 'case_review.json', review)
    run.score(tmp_path)
    assert vr.read(root / 'case_review.json')['operator_note'] == review['operator_note']
