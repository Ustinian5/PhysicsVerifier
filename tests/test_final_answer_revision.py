import json

import pytest

from rule_framework import final_answer_revision as p
from scripts.audit_final_answer_metrics import measure


ROW = dict(id='q1', question='What is 100 m divided by 10 s?',
           prediction='My final answer is 10 m/s.', context='')


def review(verdict='correct', **kwargs):
    value = dict(question_status='complete', reference_status='consistent', verdict=verdict,
                 student_final_answer='10 m/s', condition_check='Distance and time are given.',
                 reference_check='100/10=10 in m/s.', answer_check='The committed answer is 10 m/s.')
    value.update(kwargs)
    return dict(status='complete', value=value)


@pytest.mark.parametrize('label,prediction,cell', [
    ('incorrect','incorrect','TP'), ('incorrect','correct','FN'),
    ('correct','correct','TN'), ('correct','incorrect','FP')])
def test_error_positive_truth_table(label, prediction, cell):
    measured=measure({'x':label},{'x':dict(status='complete',value=dict(verdict=prediction))})
    assert measured['counts'][cell] == 1
    assert sum(measured['counts'].values()) == 1
    assert p.CONFUSION[label,prediction] == cell


def test_precision_and_recall_not_swapped():
    labels={str(i): 'incorrect' if i<4 else 'correct' for i in range(6)}
    values=['incorrect','correct','correct','correct','incorrect','correct']
    result=measure(labels,{str(i):dict(status='complete',value=dict(verdict=x)) for i,x in enumerate(values)})
    assert result['original_binary_view']['precision'] == .5
    assert result['original_binary_view']['recall'] == .25
    assert result['original_binary_view']['f1'] == pytest.approx(1/3)


def test_failed_positive_remains_a_failure():
    result=measure({'x':'incorrect'}, {'x':dict(status='content_failure',value=None)})
    assert result['counts']['failed_positive'] == 1
    assert result['counts']['FN'] == result['counts']['TN'] == 0
    assert result['whole_cohort_view']['recall'] == 0
    assert result['coverage'] == 0


def test_review_never_sees_old_label_or_verifier():
    row=dict(ROW, old_verdict='incorrect', verifier='SECRET_TEST_OUTPUT', reference='SECRET_PRIVATE_VALUE')
    messages=p.review_messages(row,['10 m/s'])
    payload=json.loads(messages[1]['content'])
    assert set(payload) == {'question','student_answer','original_reference_answers'}
    assert payload['original_reference_answers'] == ['10 m/s']
    assert 'SECRET' not in str(messages)


def test_check_never_sees_reference_gold():
    row=dict(ROW, old_verdict='incorrect', reference='SECRET_GOLD', label='SECRET_LABEL')
    messages=p.query_messages(row)
    payload,_=json.JSONDecoder().raw_decode(messages[1]['content'])
    assert set(payload) == {'question','prediction','context'}
    assert 'SECRET' not in str(messages)
    assert p.original.COMMON in messages[0]['content']
    assert p.original.OUTPUT in messages[0]['content']


def test_extra_context_cannot_sneak_in():
    with pytest.raises(ValueError,match='additional context'):
        p.query_messages(dict(ROW,context='private reference'))


@pytest.mark.parametrize('change', [dict(question_status='missing_conditions'),
                                  dict(reference_status='conflicting'),
                                  dict(reference_status='insufficient')])
def test_bad_inputs_cannot_receive_definite_labels(change):
    with pytest.raises(ValueError,match='Unsafe definite'):
        p.validate_review(review(**change)['value'])
    p.validate_review(review(verdict='uncertain',**change)['value'])


def test_machine_disagreement_cannot_automatically_flip_label():
    d=p.decision(dict(ROW,old_verdict='incorrect'),review())
    assert d['state'] == 'pending_adjudication'
    assert d['reason'] == 'independent_review_disagrees'


def test_uncertain_is_not_a_negative_or_exclusion_without_review():
    d=p.decision(dict(ROW,old_verdict='correct'),review(verdict='uncertain'))
    assert d['state'] == 'pending_adjudication'


def test_explicit_adjudication_and_missing_reference():
    candidate=dict(ROW,eligibility_failure='original_reference_missing')
    assert p.decision(candidate,None)['state'] == 'excluded'
    d=dict(id='q1',action='set_label',verdict='incorrect',basis='An independently checkable mismatch.',human_gold=False)
    assert p.decision(ROW,None,d)['verdict'] == 'incorrect'
    with pytest.raises(ValueError,match='incomplete generation/reference'):
        p.decision(candidate,None,d)
    with pytest.raises(ValueError,match='authority|adjudication'):
        p.decision(ROW,None,dict(d,human_gold=True))


def test_quota_uses_frozen_order_not_score_or_confidence():
    rows=[dict(ROW,id=str(i)) for i in range(6)]
    ds={str(i):dict(state='eligible',verdict='incorrect' if i%2 else 'correct') for i in range(6)}
    assert [r['id'] for r in p.select_balanced(rows,ds,quota=2)] == ['0','1','2','3']
    ds['1']=dict(state='excluded')
    assert [r['id'] for r in p.select_balanced(rows,ds,quota=2)] == ['0','2','3','5']
    ds['3']=dict(state='pending_adjudication')
    with pytest.raises(ValueError,match='incomplete'):
        p.select_balanced(rows,ds,quota=2)
    ds['3']=dict(state='excluded')
    with pytest.raises(ValueError,match='quota'):
        p.select_balanced(rows,ds,quota=2)


def test_shared_prefix_and_rag_object_binding():
    queries={'queries':['速度 speed 距离 distance']}
    ref=dict(id='c1',title='Historical example',text='Historical student answered 20 m/s; wrong.')
    rag=p.continuation_messages(ROW,queries,[ref],arm='case_rag')
    pure=p.continuation_messages(ROW,queries,[],arm='pure')
    assert rag[:3] == pure[:3]
    assert p.RAG_BINDING in rag[-1]['content']
    assert p.RAG_BINDING not in pure[-1]['content']
    assert 'diagnostics JSON' not in rag[-1]['content']
    with pytest.raises(ValueError,match='Pure arm'):
        p.continuation_messages(ROW,queries,[ref],arm='pure')


def test_original_reference_selection_and_full_item_admission_are_reused():
    store=p.memory.FlatMemory([dict(id='c1',kind='case',title='speed distance',text='full text speed distance',source_ids=['old1'])])
    before=p.memory.continuation_messages
    packet=p.reference_packet(ROW,'case_rag',{'queries':['speed distance 速度']},store,lambda m:len(str(m)))
    assert p.memory.continuation_messages is before
    assert packet['retrieval']['algorithm']=='flat_bm25'
    assert packet['retrieval']['provided_complete_ids']==['c1']
    assert packet['references'][0]['text']=='full text speed distance'
    assert packet['messages'][-1]['content'].endswith(p.RAG_BINDING)
    assert packet['retrieval']['embedding_calls']==0


def test_review_extra_output_fields_rejected():
    value=review()['value'];value['verifier_output']='should not be here'
    with pytest.raises(ValueError,match='schema'):
        p.validate_review(value)


def fake_runtime(monkeypatch, *, content=None, model=None, error=None, elapsed=lambda messages: 1):
    from scripts import run_final_answer_revision as r
    c=dict(model='expected-model',api_base='http://127.0.0.1:18083/v1',fixed_harmony_date='2026-11-03',
           limits=dict(context_window=131072,request_timeout_seconds=1200,wall_seconds_per_sample=1800,
                       max_requests_per_sample=12,max_tokens_per_response=16384,output_tokens_per_sample=16384,
                       input_tokens_per_sample=600000))
    plan=dict(configuration=c,configuration_sha256=r.vr.digest(c))
    class Tokenizer:
        def apply_chat_template(self,messages,*,tokenize,**kwargs):
            text = '\n'.join(m['content'] for m in messages)
            return self.encode(text) if tokenize else text
        def encode(self,text,**kwargs): return list(text.encode())
        def decode(self,ids,**kwargs): return bytes(ids).decode()
    tokenizer = Tokenizer()
    calls=[]
    def transport(client,**kw):
        calls.append(kw)
        req=dict(messages=kw['messages'],max_tokens=kw['max_tokens'],model=kw['model'],
                 temperature=0,top_p=1,reasoning_effort='medium')
        tokens = tokenizer.apply_chat_template(req['messages'], tokenize=True)
        raw=dict(model=model or c['model'],choices=[dict(finish_reason='stop')],prompt_token_ids=tokens,
                 usage=dict(prompt_tokens=len(tokens),completion_tokens=3,total_tokens=len(tokens)+3))
        r.vr.atomic(kw['request_path'],dict(request=req,request_sha256=r.base.r.sha256_value(req)))
        r.vr.atomic(kw['raw_path'],raw)
        payload = ({'queries':['speed 速度']} if len(req['messages']) < 4 else
                   {'verdict':'correct','student_final_answer':'10 m/s','reason':'Units agree'})
        rec=dict(http_status=200,model_ids=[raw['model']],raw_sse_sha256=r.vr.sha(kw['raw_path']),
                 content=content or json.dumps(payload),elapsed_seconds=elapsed(req['messages']))
        if error:rec['error']=error
        r.vr.atomic(kw['response_path'],rec)
        if error:raise ValueError(error)
        return rec
    monkeypatch.setattr(r.base.runtime.transport,'chat_response',transport)
    monkeypatch.setattr(r,'validate_effective_input',lambda raw,tok,date:
                        dict(sha256=r.vr.digest(raw['prompt_token_ids']),token_count=len(raw['prompt_token_ids'])))
    return r,plan,tokenizer,calls


def request_fixture(tmp_path, monkeypatch, **options):
    r,plan,tokenizer,calls=fake_runtime(monkeypatch,**options)
    messages=[dict(role='user',content='test input')]
    invoke=lambda:r.request_once(tmp_path,tmp_path/'call',messages,20,p.memory.validate_queries,
                                 plan,tokenizer,None)
    return r,invoke,calls


def test_success_is_replayed_without_new_request(tmp_path,monkeypatch):
    r,invoke,calls=request_fixture(tmp_path,monkeypatch)
    first=invoke();assert first['status']=='complete'
    assert invoke()==first and len(calls)==1
    path=tmp_path/'call/result.json';value=r.vr.read(path);value['value']={'queries':['tampered']};r.vr.atomic(path,value)
    with pytest.raises(ValueError,match='differs from original'):
        invoke()
    assert len(calls)==1


def test_complete_invalid_content_is_not_retried(tmp_path,monkeypatch):
    r,invoke,calls=request_fixture(tmp_path,monkeypatch,content='not a JSON object')
    result=invoke();assert result['status']=='content_failure' and result['value'] is None
    assert invoke()==result and len(calls)==1


def test_wrong_model_stops_without_recording_a_terminal_case(tmp_path,monkeypatch):
    r,invoke,calls=request_fixture(tmp_path,monkeypatch,model='different-model')
    for _ in range(2):
        with pytest.raises(ValueError,match='identity'):
            invoke()
    assert len(calls)==1 and not (tmp_path/'call/result.json').exists()


def test_interrupted_request_cannot_be_resampled(tmp_path,monkeypatch):
    r,invoke,calls=request_fixture(tmp_path,monkeypatch)
    r.vr.atomic(tmp_path/'call/request.json',dict(request={}))
    with pytest.raises(ValueError,match='Interrupted request'):
        invoke()
    assert calls==[]


def test_unknown_decoder_error_stops(tmp_path,monkeypatch):
    r,invoke,calls=request_fixture(tmp_path,monkeypatch,error='Unknown envelope structure')
    with pytest.raises(ValueError,match='Unclassified decoder'):
        invoke()
    assert len(calls)==1 and not (tmp_path/'call/result.json').exists()


def test_retained_evidence_cannot_change(tmp_path,monkeypatch):
    r,invoke,calls=request_fixture(tmp_path,monkeypatch)
    invoke();path=tmp_path/'call/response.json';value=r.vr.read(path);value['content']='changed';r.vr.atomic(path,value)
    with pytest.raises(ValueError,match='evidence drift'):
        invoke()
    assert len(calls)==1


def test_failed_shared_prepare_propagates_without_check_calls(tmp_path,monkeypatch):
    from scripts import run_final_answer_revision as r
    prefix=dict(status='content_failure',value=None,reason='Too many queries',wall_seconds=12,
                usage=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15),files={})
    calls=[]
    def failed(out,folder,*args,**kwargs):
        calls.append(folder)
        r.immutable(folder/'result.json',prefix)
        return prefix
    monkeypatch.setattr(r,'request_once',failed)
    c=dict(max_requests_per_sample=12,output_tokens_per_sample=16384,input_tokens_per_sample=600000,wall_seconds_per_sample=1800)
    plan=dict(configuration=dict(limits=c),configuration_sha256='frozen')
    r.run_question(tmp_path,ROW,'formal',plan,None,None,{})
    assert len(calls)==1
    for arm in r.ARMS:
        f=tmp_path/'formal'/arm/r.vr.digest(ROW['id'])[:24]/'result.json'
        got=r.vr.read(f)
        assert got['status']=='content_failure' and got['value'] is None
        assert got['actual_response_count']==1 and got['usage']==prefix['usage']
        assert got['wall_seconds']==12


def test_assistant_decision_must_bind_the_actual_answer(tmp_path):
    from scripts import run_final_answer_revision as r
    candidate=dict(ROW,question_sha256='q',answer_sha256='a',reference_sha256='r')
    decision=dict(id='q1',action='set_label',verdict='incorrect',basis='reason',human_gold=False,
                  question_sha256='q',answer_sha256='changed',reference_sha256='r')
    r.vr.atomic(tmp_path/'assistant_review.json',dict(reviewer='Codex assistant',human_gold=False,decisions=[decision]))
    with pytest.raises(ValueError,match='different question/answer/reference'):
        r.read_overrides(tmp_path,[candidate])


def simulated_question(tmp_path, monkeypatch, check_seconds):
    r,plan,tokenizer,calls = fake_runtime(monkeypatch,elapsed=lambda m: 1000 if len(m)==2 else check_seconds)
    memories = dict(pure=None, case_rag=p.memory.FlatMemory([]), flat_skills=p.memory.FlatMemory([]),
                    rule_tree=p.memory.TreeMemory({'domains':[]}))
    def run():
        r.run_question(tmp_path, ROW, 'formal', plan, tokenizer, None, memories)
    return r,plan,calls,run


def test_shared_prepare_charged_once_per_arm_and_verified_on_resume(tmp_path,monkeypatch):
    r,plan,calls,run = simulated_question(tmp_path,monkeypatch,700)
    run()
    key = r.vr.digest(ROW['id'])[:24]
    assert len(calls) == 5 and calls[0]['timeout'] == 1200
    for call in calls[1:]:
        assert 795 < call['timeout'] <= 800
    for arm in r.ARMS:
        path = tmp_path/'formal'/arm/key/'result.json'
        value = r.vr.read(path)
        assert value['status'] == 'complete' and 1700 <= value['wall_seconds'] < 1705
    run()
    assert len(calls) == 5
    value['wall_seconds'] -= 100
    r.vr.atomic(path,value)
    with pytest.raises(ValueError,match='cumulative wall-time charge'):
        run()
    # Offline scoring revalidates the same raw charges.
    monkeypatch.setattr(r,'verify',lambda out:plan)
    monkeypatch.setattr(r,'released',lambda out,plan:{})
    r.vr.atomic(tmp_path/'inputs.json',[ROW])
    r.vr.atomic(tmp_path/'labels_private.json',{ROW['id']:{'verdict':'correct'}})
    with pytest.raises(ValueError,match='cumulative wall-time charge'):
        r.score(tmp_path)
    assert len(calls) == 5 and not (tmp_path/'metrics.json').exists()


def test_nonempty_material_survives_resume_and_scoring_without_new_calls(tmp_path, monkeypatch):
    r, plan, tokenizer, calls = fake_runtime(monkeypatch)
    monkeypatch.setattr(r, 'ARMS', ('case_rag',))
    material = dict(id='c1', kind='case', title='speed distance', text='full text speed distance', source_ids=['old'])
    store = {'case_rag': p.memory.FlatMemory([material])}
    def run():
        r.run_question(tmp_path, ROW, 'formal', plan, tokenizer, None, store)
    run()
    key = r.vr.digest(ROW['id'])[:24]
    packet_file = tmp_path/'formal/case_rag'/key/'reference_packet.json'
    packet = r.vr.read(packet_file)
    assert list(packet['references'][0]) == ['id', 'text', 'title']
    sent = packet['messages'][-1]['content']
    assert sent.index('"title"') < sent.index('"text"')
    run()
    monkeypatch.setattr(r, 'verify', lambda out: plan)
    monkeypatch.setattr(r, 'released', lambda out, plan: {})
    r.vr.atomic(tmp_path/'inputs.json', [ROW])
    r.vr.atomic(tmp_path/'labels_private.json', {ROW['id']: {'verdict': 'correct'}})
    r.vr.atomic(tmp_path/'bench_release.json', {'synthetic': True})
    r.score(tmp_path)
    assert (tmp_path/'metrics.json').is_file() and len(calls) == 2
    packet['references'][0]['text'] = 'different material'
    r.vr.atomic(packet_file, packet)
    with pytest.raises(ValueError, match='Wrong shared prefix or supplied materials'):
        run()
    assert len(calls) == 2


def test_1000_plus_1000_refused_on_first_call_and_raw_recovery(tmp_path,monkeypatch):
    r,_,calls,run = simulated_question(tmp_path,monkeypatch,1000)
    folder = tmp_path/'formal'/r.ARMS[0]/r.vr.digest(ROW['id'])[:24]
    for _ in range(2):
        with pytest.raises(ValueError,match='cumulative wall-time budget exceeded'):
            run()
        assert len(calls) == 2 and calls[-1]['timeout'] <= 800
        assert (folder/'check/raw.json').exists() and not (folder/'result.json').exists()
        assert not (folder/'check/result.json').exists()


def test_spent_setup_or_invalid_elapsed_is_refused(tmp_path,monkeypatch):
    r,plan,tok,calls = fake_runtime(monkeypatch,elapsed=lambda _:float('nan'))
    args = (tmp_path,tmp_path/'call',p.query_messages(ROW),20,p.memory.validate_queries,plan,tok,None)
    with pytest.raises(ValueError,match='exhausted before request'):
        r.request_once(*args,available=10,started=r.time.monotonic()-11)
    with pytest.raises(ValueError,match='exhausted before request'):
        r.request_once(*args,available=10)
    assert not calls
    with pytest.raises(ValueError,match='Invalid elapsed time'):
        r.request_once(tmp_path,tmp_path/'next',*args[2:],available=10)
    assert len(calls)==1


def test_absolute_deadline_interrupts_a_blocking_call():
    from scripts import run_final_answer_revision as r
    with pytest.raises(TimeoutError,match='wall-time budget'):
        with r.request_deadline(.03):
            r.time.sleep(.5)
    assert r.signal.getitimer(r.signal.ITIMER_REAL) == (0.0,0.0)


def test_amended_cohort_uses_only_pending_reviews_and_largest_balanced_prefix(tmp_path, monkeypatch):
    from scripts import run_final_answer_revision as r

    rows = [dict(ROW, id=str(i), old_verdict='correct', question_sha256='q', answer_sha256='a',
                 reference_sha256='ref') for i in range(6)]
    failed = dict(status='content_failure', value=None, reason='invalid')
    reviews = [review('uncertain'), review(), review(), review(), failed, review()]
    before = {row['id']: p.decision(row, raw) for row, raw in zip(rows, reviews)}
    decision = dict(id='0', question_sha256='q', answer_sha256='a', reference_sha256='ref',
                    action='set_label', verdict='incorrect', human_gold=False, basis='Checked against original.')
    decisions = [decision]
    r.vr.atomic(tmp_path/'assistant_review.json', dict(reviewer='Codex assistant', human_gold=False, decisions=[]))
    r.vr.atomic(tmp_path/'candidates.json', rows)
    r.vr.atomic(tmp_path/'review_status.json', dict(pending=[before['0']]))
    r.vr.atomic(tmp_path/'review/0/result.json', reviews[0])
    assert r.read_overrides(tmp_path, rows) == {}
    assert r.cohort.release_quota(tmp_path, before, 3) == (3, {})
    data = dict(policy=r.cohort.POLICY, reviewer='Codex assistant', human_gold=False, requested_per_class=3,
                effective_per_class=1, eligible_counts=dict(correct=4, incorrect=1),
                decisions=decisions, code_sha256=r.vr.sha(r.cohort.__file__))
    for name in ('candidates', 'assistant_review', 'review_status'):
        data[name+'_sha256'] = r.vr.sha(tmp_path/(name+'.json'))
    path = tmp_path/r.cohort.AMENDMENT
    r.vr.atomic(path, data)
    overrides = r.read_overrides(tmp_path, rows)
    after = {row['id']: p.decision(row, raw, overrides.get(row['id'])) for row, raw in zip(rows, reviews)}
    quota, binding = r.cohort.release_quota(tmp_path, after, 3)
    assert [row['id'] for row in p.select_balanced(rows, after, quota)] == ['0','1']
    assert binding['quota_amendment']['sha256'] == r.vr.sha(path)
    assert after['4']['state'] == 'review_failure'
    parent_hash = r.vr.sha(tmp_path/'review_status.json')
    monkeypatch.setattr(r, 'dispositions', lambda out: after)
    r.run_reviews(tmp_path, dict(configuration=dict(quota_per_class=3)), None, None)
    assert r.vr.sha(tmp_path/'review_status.json') == parent_hash
    r.vr.atomic(path, dict(data, effective_per_class=2))
    with pytest.raises(ValueError, match='largest balanced'):
        r.cohort.release_quota(tmp_path, after, 3)
    r.vr.atomic(path, dict(data, decisions=[dict(decision, id='4')]))
    with pytest.raises(ValueError, match='exactly the pending'):
        r.read_overrides(tmp_path, rows)
    r.vr.atomic(path, data)
    r.vr.atomic(tmp_path/'review/0/result.json', reviews[1])
    with pytest.raises(ValueError, match='original independent review'):
        r.read_overrides(tmp_path, rows)


def test_freeze_and_archive_rescore_bind_the_unselected_candidate(tmp_path,monkeypatch):
    from scripts import run_final_answer_revision as r
    rows = [dict(ROW,id=str(i)) for i in range(3)]
    choices = {str(i):dict(state='eligible',verdict='incorrect' if i==0 else 'correct') for i in range(3)}
    archive = tmp_path/'old'
    plan = dict(configuration_sha256='test',configuration=dict(controls=[],quota_per_class=1,archive=str(archive)))
    monkeypatch.setattr(r,'verify',lambda out:plan)
    monkeypatch.setattr(r,'dispositions',lambda out:choices)
    for name,value in dict(plan=plan,candidates=rows,isolation_candidates={'candidates':[]},
                           review_controls_passed={'passed':True,'configuration_sha256':'test','files':{}}).items():
        r.vr.atomic(tmp_path/(name+'.json'),value)
    r.freeze_bench(tmp_path)
    release = r.vr.read(tmp_path/'bench_release.json')
    assert release['dispositions_sha256'] == r.vr.sha(tmp_path/'dispositions.json')
    assert r.released(tmp_path,plan) == choices
    assert [row['id'] for row in r.vr.read(tmp_path/'inputs.json')] == ['0','1']
    # The third row is outside the new cohort but inside the old archive.
    r.vr.atomic(archive/'final_labels_private.json',{'2':{'verdict':'incorrect'}})
    frozen = {}
    for arm in r.ARMS:
        path = archive/'checks/final/formal'/arm/'2/result.json'
        r.vr.atomic(path,dict(id='2',arm=arm,status='complete',value={'verdict':'correct'}))
        frozen[str(path.relative_to(archive))] = r.vr.sha(path)
    r.vr.atomic(archive/'final_metrics.json',dict(source_results=frozen))
    r.rescore_archive(tmp_path)
    path = tmp_path/'archived_rescore.json'
    assert r.vr.read(path)['changed_labels']==[dict(id='2',before='incorrect',after='correct')]
    before = r.vr.sha(path)
    r.vr.atomic(tmp_path/'dispositions.json',dict(choices,**{'2':{'state':'excluded'}}))
    with pytest.raises(ValueError,match='Released dispositions changed'):
        r.rescore_archive(tmp_path)
    r.vr.atomic(tmp_path/'dispositions.json',choices)
    del release['dispositions_sha256']
    r.vr.atomic(tmp_path/'bench_release.json',release)
    with pytest.raises(ValueError,match='Missing released dispositions'):
        r.rescore_archive(tmp_path)
    assert r.vr.sha(path)==before


def test_repair_can_only_bind_the_prepared_code_and_preserves_its_plan(tmp_path,monkeypatch):
    from scripts import run_final_answer_revision as r
    hashes = {r.RUNNER:r.PREPARED_RUNNER,'other':'frozen'}
    original = dict(source_hashes=hashes,protocol=p.VERSION,contract=p.contract(),inputs='unchanged')
    parent = r.vr.digest(original)
    monkeypatch.setattr(r,'PREPARED_CONFIGURATION',parent)
    live = dict(hashes,**{r.RUNNER:'repaired'})
    monkeypatch.setattr(r,'source_hashes',lambda:dict(live))
    path = tmp_path/'plan.json'
    r.vr.atomic(path,dict(configuration=original,configuration_sha256=parent))
    before = path.read_bytes()
    with pytest.raises(ValueError,match='Missing or changed'):
        r.execution_plan(tmp_path)
    (tmp_path/'pilot').mkdir()
    with pytest.raises(ValueError,match='after execution'):
        r.execution_plan(tmp_path,register=True)
    (tmp_path/'pilot').rmdir()
    live['other']='different'
    with pytest.raises(ValueError,match='outside the authorized'):
        r.execution_plan(tmp_path,register=True)
    live['other']='frozen'
    repaired=r.execution_plan(tmp_path,register=True)
    assert repaired['configuration']==dict(original,source_hashes=live) and repaired['configuration_sha256']!=parent
    assert path.read_bytes()==before and r.execution_plan(tmp_path,register=True)==r.execution_plan(tmp_path)
    live[r.RUNNER]='unregistered change'
    with pytest.raises(ValueError,match='Missing or changed'):
        r.execution_plan(tmp_path)


def test_replay_binding_keeps_online_hash_and_rejects_other_source_or_receipt_changes(tmp_path, monkeypatch):
    from scripts import final_answer_replay_binding as repair, run_final_answer_revision as r
    original = dict(source_hashes={r.RUNNER:r.PREPARED_RUNNER,'retrieval':'fixed'})
    parent = r.vr.digest(original)
    live = {r.RUNNER:repair.REPLAY_RUNNER, 'retrieval':'fixed'}
    online = {**live, r.RUNNER:repair.ONLINE_RUNNER}
    config = r.vr.digest(dict(original, source_hashes=online))
    monkeypatch.setattr(repair, 'CONFIG', config)
    monkeypatch.setattr(r, 'PREPARED_CONFIGURATION', parent)
    plan = dict(configuration=original, configuration_sha256=parent)
    r.vr.atomic(tmp_path/'plan.json', plan)
    r.vr.atomic(tmp_path/'execution_repair.json', dict(runner_after=repair.ONLINE_RUNNER,
                configuration_sha256=config, parent_configuration_sha256=parent, preparation_sha256=r.vr.sha(tmp_path/'plan.json')))
    receipt = repair.binding(tmp_path, plan, live)
    r.vr.atomic(tmp_path/'replay_repair.json', receipt)
    assert repair.bound_generation_sources(tmp_path, plan, live) == online
    with pytest.raises(ValueError, match='Source drift'):
        repair.bound_generation_sources(tmp_path, plan, dict(live, retrieval='changed'))
    with pytest.raises(ValueError, match='Unknown replay'):
        repair.bound_generation_sources(tmp_path, plan, {**live,r.RUNNER:'changed'})
    r.vr.atomic(tmp_path/'replay_repair.json', dict(receipt, binding_code_sha256='changed'))
    with pytest.raises(ValueError, match='Changed replay'):
        repair.bound_generation_sources(tmp_path, plan, live)
