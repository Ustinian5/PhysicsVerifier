import copy
import hashlib
import json
import os
import sys
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from scripts import reuse_step_baseline as reuse
from scripts import reference_review_pool as pool
from scripts import reference_material_pool as materials_pool
from scripts import run_reference_guided_verifier as workflow
from scripts import run_step_match_judge as judge
from scripts import verifier_build_scope as scope
from tests.test_step_match_judge import decision, fixture, response


def test_skill_pool_partitions_sources_and_resumes_saved_outcomes(tmp_path, monkeypatch):
    vr = materials_pool.vr
    cases = [dict(id='case'+str(i),title='Title',text='History',source_ids=[str(i)]) for i in range(5)]
    vr.atomic(tmp_path/'cases.json',cases)
    cfg = dict(frozen_files={'cases.json':vr.sha(tmp_path/'cases.json')},transport={'fixed_date':'2026-11-07'})
    sig = vr.digest(cfg)
    vr.atomic(tmp_path/'plan.json',dict(configuration=cfg,configuration_sha256=sig))
    opened,requests = [],[]

    @contextmanager
    def service(folder,gpu,port,stop,signature,date):
        opened.append((gpu,port,date))
        yield 'http://127.0.0.1:'+str(port)+'/v1'

    def call(out,journal,tokenizer,stage,key,messages):
        requests.append((key,journal.config['auditor_api']))
        path = out/'raw'/key
        vr.atomic(path,{'id':key})
        return {'skills':[{'title':key,'text':'method'}]},{'file':str(path.relative_to(out)),'sha256':vr.sha(path)}

    monkeypatch.setattr(pool,'replica_service',service)
    monkeypatch.setattr(materials_pool.materials,'model_json',call)
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:object())))
    for _ in range(2):
        for idx,gpu in enumerate((5,6)):
            materials_pool.skill_worker(tmp_path,sig,gpu,idx,2,threading.Event(),os.getppid())
    assert [key for key,_ in requests] == ['case0','case2','case4','case1','case3']
    assert len(opened) == 2
    outcomes = [materials_pool.materials.skill_outcome(tmp_path,cfg,c) for c in cases]
    materials_pool.materials.publish_skills(tmp_path,cases,outcomes)
    assert vr.read(tmp_path/'materials/cases.json') == cases
    assert {tuple(d['source_ids']) for d in vr.read(tmp_path/'materials/skills.json')} == {(str(i),) for i in range(5)}


def test_sources_keep_original_order_failures_and_denominator():
    register = dict(registered=['first', 'held', 'failed', 'last'], excluded={'prior': ['held']})
    rows = [dict(id=s) for s in ['first', 'held', 'last']]
    failures = [dict(id='failed')]
    assert scope.apply_scope(rows, failures, register) == [rows[0], rows[2]]
    for changed_rows, changed_failures in [(rows[::-1], failures), (rows[1:], failures),
                                           (rows, failures * 2), (rows, failures + [rows[0]])]:
        with pytest.raises(ValueError, match='population'):
            scope.apply_scope(changed_rows, changed_failures, register)


def test_retained_passage_keeps_only_the_exact_subquestion_and_full_answer():
    question = 'Unrelated exercise.\n\nCharge on a spherical drop.\nFind its electrostatic energy.'
    answer, reference = 'The electrostatic energy is …', ['Benchmark-independent equation.']
    row = dict(id='drop', question=question, prediction=answer, official_reference={'label':reference})
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
    target = question.split('\n\n')[1]
    view = dict(first_line=3, question_sha256=sha(question), retained_sha256=sha(target),
                prediction_sha256=sha(answer), reference_digest=judge.vr.digest(reference))
    register = dict(retained_passages={'drop': view})
    assert scope.reviewed_input(row, register) == dict(row, question=target)
    assert scope.reviewed_input(dict(row, id='other'), register)['question'] == question
    for field, value in [('question', 'new header\n'+question), ('prediction', answer+' '),
                         ('official_reference', {'label':['changed']})]:
        with pytest.raises(ValueError, match='subquestion'):
            scope.reviewed_input(dict(row, **{field:value}), register)


def test_shared_prefix_flags_review_without_excluding_a_source():
    intro = 'This common table lists physical constants in SI units.'
    build = [dict(id='ice',question=intro+' How thick is the ice?')]
    held = [dict(id='comet',question=intro+' Find the comet orbit.')]
    assert scope.lexical_candidates(build, held)[0]['same_introduction']
    assert scope.apply_scope(build, [], dict(registered=['ice'],excluded={})) == build


def test_all_live_stages_observe_supervisor_stop(tmp_path):
    folders = [tmp_path / name for name in ('construction', 'libraries', 'step', 'final')]
    for folder in folders:
        workflow.connect_stop(folder, tmp_path)
        workflow.connect_stop(folder, tmp_path)
        assert not (folder/'STOP').exists()
    (tmp_path/'STOP').touch()
    assert all((folder/'STOP').is_file() for folder in folders)
    with pytest.raises(ValueError, match='STOP'):
        workflow.connect_stop(folders[0], tmp_path)


def frozen_comparison(out, monkeypatch, policy):
    rows, results, coverage, _ = judge.controls()
    row = rows[0]
    base = results[row['id'], 'pure']
    key = judge.vr.digest([row['id'], judge.match.messages(row, base['diagnostics'][0], row['roots'])])
    old = dict(prompt_profile='english', model=judge.base.r.EXPECTED_MODEL, fixed_harmony_date='2026-11-03',
               pure_prompt='direct', online_reference_visible=False, controls=[], limits={}, budget_limits={},
               step_match_judge=judge.match.VERSION, sample_ids=[row['id']], cohort=coverage)
    cfg = dict(old, reuse_pure=str(out/'pure'), judge_location_policy=policy,
               arms=list(judge.ARMS), prepare_arms=[a for a in judge.ARMS if a != 'pure'])
    checks = dict(configuration=cfg, configuration_sha256=judge.vr.digest(cfg))
    judge.vr.atomic(out/'plan.json', checks)
    previous = (old, rows, {(row['id'], 'pure'): base}, coverage, {key: decision()}, {},
                [reuse.location.score_row(row, base, [decision()], [None])])
    monkeypatch.setattr(reuse, 'baseline', lambda _: copy.deepcopy(previous))
    for arm in cfg['prepare_arms']:
        value = dict(results[row['id'], arm], id=row['id'], organization_arm=arm,
                     configuration_sha256=checks['configuration_sha256'])
        judge.vr.atomic(judge.base.path_for(out, 'formal', arm, row), value)
    # The fixture provides synthetic published documents. The judge transport still saves and verifies its raw requests.
    monkeypatch.setattr(judge.base.runtime, 'audit_result', lambda value: None)
    return checks


@pytest.mark.parametrize('policy,invalid,location_calls,complete', [
    ('locate_once', 'quote', 1, True), ('stop', 'quote', 0, False), ('locate_once', 'root', 0, False),
])
def test_frozen_pure_reuse_and_evidence_only_repair(tmp_path, monkeypatch, policy, invalid, location_calls, complete):
    plan = frozen_comparison(tmp_path, monkeypatch, policy)
    def scripted(messages):
        if messages[0]['content'] == reuse.location.PROMPT:
            return json.dumps({'start_line': None, 'end_line': None})
        diagnostic = json.loads(messages[1]['content'])['published_diagnosis']['message']
        assert not diagnostic.startswith('The numerical'), 'The completed pure judgment must be reused'
        if diagnostic.startswith('Multiplying'):
            draft = decision(quote='not a literal quote') if invalid == 'quote' else decision(ids=['unknown'])
            return json.dumps(draft)
        return response(messages)
    tokenizer, calls = fixture(monkeypatch, scripted)
    live = dict(tokenizer=tokenizer, client=object())
    if complete:
        metrics = reuse.evaluate(tmp_path, plan, **live)
        assert metrics['judge']['baseline_judgments_reused'] == 1
        assert metrics['judge']['new_locations'] == 1
        assert metrics['scores']['pure']['hits'] == metrics['scores']['case_rag']['hits'] == 1
        assert metrics['scores']['flat_skills']['hits'] == 0
        assert reuse.evaluate(tmp_path, plan) == metrics
    else:
        for kwargs in (live, {}):
            expected = reuse.CitationDecisionPending if invalid == 'quote' and policy == 'stop' else ValueError
            with pytest.raises(expected, match='1 invalid') as failure:
                reuse.evaluate(tmp_path, plan, **kwargs)
            assert (type(failure.value) is reuse.CitationDecisionPending) == (expected is reuse.CitationDecisionPending)
        assert not (tmp_path/'metrics.json').exists()
    assert len(calls) == 2 + location_calls
    assert len(list((tmp_path/'step_judge/locations').glob('*/result.json'))) == location_calls


def test_parallel_reviews_release_only_after_ordered_pilot(tmp_path, monkeypatch):
    cfg = {'pilot_size': 4}
    rows = [dict(id=str(i)) for i in range(11)]
    calls = []

    class Worker(threading.Thread):
        @property
        def exitcode(self):
            return 0 if not self.is_alive() else None

    def save(row, owner):
        path = tmp_path/'experiences'/(row['id']+'.json')
        assert not path.exists()
        if int(row['id']) >= cfg['pilot_size']:
            assert (tmp_path/'pilot_acceptance.json').exists()
        calls.append((int(row['id']), owner))
        pool.extract.write(path, row)

    def secondary(out, gpu, positions, gate, aborted, parent, signature):
        for i in positions:
            if i >= cfg['pilot_size']:
                assert gate.wait(5) and not aborted.is_set()
            save(rows[i], gpu)

    def primary(out, row, taxonomy, config, journal, tokenizer, *, saved_only):
        if not saved_only:
            save(row, 4)
        return pool.extract.read(out/'experiences'/(row['id']+'.json'))

    monkeypatch.setattr(pool.multiprocessing, 'get_context', lambda _: SimpleNamespace(Event=threading.Event, Process=Worker))
    monkeypatch.setattr(pool, 'replica', secondary)
    monkeypatch.setattr(pool.extract, 'review_one', primary)
    for i, record in enumerate(pool.ordered_reviews(tmp_path, rows, {}, cfg, None, None, gpus=[4, 5, 6, 7])):
        assert record == rows[i]
        if i+1 == cfg['pilot_size']:
            pool.extract.write(tmp_path/'pilot_acceptance.json', dict(passed=True, configuration_sha256=pool.extract.digest(cfg)))
    assert sorted(calls) == [(i, 4+i%4) for i in range(len(rows))]
