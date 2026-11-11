import copy
import json
from pathlib import Path

import pytest

from rule_framework import corrected_verifier as p, verified_resume as vr
from scripts import run_corrected_pipeline as pipeline, run_corrected_checks as runner
from tests.test_bounded_tree import bundle, engine, finish


def docs(n=17):
    return [dict(id=f'k{i:02}',kind='skill',title='磁场单位',text=f'磁场 高斯 特斯拉 单位换算 {i}',source_ids=[f's{i}']) for i in range(n)]


def test_tree_pages_preserve_top2_route_ranking_no_flat_fallback(tmp_path):
    catalog=finish(engine(tmp_path,bundle(17)))['catalog']
    tree=p.TreeMemory(catalog);k=p.Knowledge('rule_tree',tree)
    first,trace=tree.retrieve('磁场单位')
    page=k.preview('search_memory',dict(query='磁场单位',offset=0))
    assert page['items'][:6]==first and page['retrieval']==trace
    assert len(page['items'])==10
    second=k.preview('search_memory',dict(query='磁场单位',offset=10))
    assert len(second['items'])==7
    assert len({x['id'] for x in page['items']+second['items']})==17
    assert k.preview('search_memory',dict(query='无任何对应概念xyz',offset=0))['items']==[]


def test_reads_single_whole_item_without_six_item_cap():
    k=p.Knowledge('flat_skills',p.FlatMemory(docs()))
    args=dict(query='磁场',offset=0);found=k.preview('search_memory',args);k.commit('search_memory',args,found)
    with pytest.raises(ValueError):k.preview('read_memory',{'id':'invented'})
    with pytest.raises(ValueError):k.preview('read_memory',{'ids':['k00']})
    for hit in found['items']:
        args={'id':hit['id']};value=k.preview('read_memory',args);k.commit('read_memory',args,value)
    assert len(k.read_ids)==10 and not k.audit()['protocol_noncompliance']


@pytest.mark.parametrize('task',['step','final'])
@pytest.mark.parametrize('arm',p.ARMS)
def test_public_whitelist_and_no_shared_prepare(task,arm):
    row=dict(id='a',question='visible question',prediction='visible answer',context='',
             answer='SECRET_REFERENCE',label='SECRET_LABEL',physics_error_gt=['SECRET_GT'])
    text=json.dumps(p.messages(row,arm,task))
    assert 'SECRET_' not in text and 'PREPARE' not in text
    assert ('search_memory' in text)==(arm!='pure')
    assert 'visible question' in text and 'visible answer' in text


def test_first_each_class_selected_in_frozen_order_no_scores():
    rows=[dict(id=str(i)) for i in range(7)]
    outcomes={str(i):dict(state='complete',judgment={'verdict':label}) for i,label in enumerate(
        ['correct','uncertain','incorrect','incorrect','correct','correct','incorrect'])}
    assert [r['id'] for r in pipeline.balanced(rows,outcomes,2)]==['0','2','3','4']
    with pytest.raises(ValueError):pipeline.balanced(rows,outcomes,4)


def test_final_scores_explicit_binary_and_abstention():
    rows=[{'id':str(i)} for i in range(6)]
    labels={str(i):{'verdict':'incorrect' if i%2==0 else 'correct'} for i in range(6)}
    pred={str(i):dict(status='complete',value={'verdict':v}) for i,v in enumerate(
        ['incorrect','incorrect','correct','correct','uncertain','correct'])}
    pred['5']['status']='technical_failure'
    result=p.final_scores(rows,labels,pred)
    assert [result[k] for k in ('TP','FP','FN','TN')]==[1,1,1,1]
    assert result['precision']==result['recall']==result['f1']==.5
    assert result['binary_coverage']==4/6 and not result['complete_binary_evaluation']


class Tokenizer:
    def apply_chat_template(self,messages,**kw):return '\n'.join(m.get('content') or '' for m in messages)
    def encode(self,text,**kw):return list(text.encode())
    def decode(self,ids,**kw):return bytes(ids).decode()


def mock_transport(monkeypatch, responses):
    calls=[]
    def call(self,stage,key,body):
        path=self.output/'raw'/stage/runner.c.digest(key)[:24]/'request_001.json'
        if path.exists():return vr.read(path)['response'],dict(file=str(path.relative_to(self.output)),sha256=vr.sha(path))
        calls.append(copy.deepcopy(body))
        response=responses.pop(0)
        if isinstance(response,Exception):raise response
        text=Tokenizer().apply_chat_template(body['messages'])
        raw=dict(model=body['model'],prompt_token_ids=list(text.encode()),usage=dict(prompt_tokens=len(text.encode()),completion_tokens=10),decoded=response)
        vr.atomic(path,dict(response=raw))
        return raw,dict(file=str(path.relative_to(self.output)),sha256=vr.sha(path))
    monkeypatch.setattr(runner.c.corpus.Journal,'call',call)
    monkeypatch.setattr(runner,'validate_effective_input',lambda *args:{'checked':True})
    monkeypatch.setattr(runner.c.corpus.harmony,'decode_response',lambda raw,**kwargs:raw['decoded'])
    return calls


def tool(name,args,n):
    return dict(reasoning_content='analysis before '+str(n),tool_calls=[dict(id='call'+str(n),type='function',function=dict(name=name,arguments=json.dumps(args)))])


def final(value):return dict(content=json.dumps(value),reasoning_content='final internal analysis')


def configuration():
    return dict(transport=dict(model=runner.c.corpus.r.EXPECTED_MODEL,auditor_api='http://unused',fixed_date='2026-11-07',timeout_seconds=1200))


@pytest.mark.parametrize('task,bad', [
    ('step', None), ('step', []), ('step', {'diagnostics': None}),
    ('step', {'diagnostics': [None]}), ('step', {'diagnostics': ['wrong']}),
    ('step', {'diagnostics': [{'message': []}]}),
    ('step', {'diagnostics': [{'message': 'error', 'evidence': None}]}),
    ('step', {'diagnostics': [{'message': 'error', 'evidence': {'quote': 1}}]}),
    ('step', {'diagnostics': [{'message': 'error', 'evidence': {'quote': 'A'}, 'rule_refs': 'x'}]}),
    ('step', {'diagnostics': [{'message': 'error', 'evidence': {'quote': 'A'}, 'rule_refs': [None]}]}),
    ('final', None), ('final', []), ('final', {'diagnostics': []}),
    ('final', {'verdict': [], 'reason': 'x', 'student_final_answer': 'A'}),
    ('final', {'verdict': 'yes', 'reason': 'x', 'student_final_answer': 'A'}),
    ('final', {'verdict': 'correct', 'reason': '', 'student_final_answer': 'A'}),
    ('final', {'verdict': 'correct', 'reason': 'x', 'student_final_answer': ''}),
    ('final', {'verdict': 'incorrect', 'reason': {}, 'student_final_answer': 'A'}),
])
@pytest.mark.parametrize('arm', ['pure', 'flat_skills'])
def test_schema_fault_matrix_is_terminal_case_failure_not_a_reroll(tmp_path, monkeypatch, task, bad, arm):
    calls = mock_transport(monkeypatch, [final(bad)])
    row = dict(id='q', question='Q', prediction='A')
    memory = None if arm == 'pure' else p.FlatMemory(docs())
    result = runner.run_case(tmp_path, row, arm, task, memory, configuration(), Tokenizer())
    assert result['status'] == 'content_failure' and result['value'] is None
    assert result['usage']['requests'] == 1 and len(result['evidence']) == 1
    before = vr.sha(tmp_path / 'result.json')
    assert runner.run_case(tmp_path, row, arm, task, memory, configuration(), Tokenizer()) == result
    assert len(calls) == 1 and vr.sha(tmp_path / 'result.json') == before


@pytest.mark.parametrize('task', ['step', 'final'])
def test_batch_continues_after_one_format_failure_without_deleting_case(tmp_path, monkeypatch, task):
    good = {'diagnostics': []} if task == 'step' else dict(
        verdict='correct', student_final_answer='A', reason='valid')
    calls = mock_transport(monkeypatch, [final([])] + [final(good)] * 7)
    monkeypatch.setattr(pipeline, 'check_condition', lambda *args: configuration())
    rows = [dict(id=str(i), question='Q', prediction='A') for i in range(2)]
    memories = dict(pure=None, case_rag=p.FlatMemory([dict(d, kind='case') for d in docs()]),
        flat_skills=p.FlatMemory(docs()),
        rule_tree=p.TreeMemory(finish(engine(tmp_path / 'tree', bundle(3)))['catalog']))
    results = pipeline.checks(tmp_path, {}, Tokenizer(), task, 'formal', rows, memories)
    assert len(calls) == 8 and all([r['id'] for r in results[a]] == ['0', '1'] for a in p.ARMS)
    assert results['pure'][0]['status'] == 'content_failure'
    assert sum(r['status'] == 'complete' for rs in results.values() for r in rs) == 7


@pytest.mark.parametrize('message', ['Unregistered actual Harmony recipient', 'Ambiguous tool yield',
    'Contradictory stop metadata', 'Missing or inconsistent token usage', 'Unexpected unrouted Harmony message'])
def test_unreviewed_native_fault_remains_global_stop(tmp_path, monkeypatch, message):
    mock_transport(monkeypatch, [final({'diagnostics': []})])
    def decode(*args, **kwargs):
        raise ValueError(message)
    monkeypatch.setattr(runner.c.corpus.harmony, 'decode_response', decode)
    with pytest.raises(ValueError, match=message):
        runner.run_case(tmp_path, dict(id='q', question='Q', prediction='A'),
                        'pure', 'step', None, configuration(), Tokenizer())
    assert not (tmp_path / 'result.json').exists()
    assert len(list((tmp_path / 'raw').rglob('request_*.json'))) == 1


def test_step_failure_keeps_reference_denominator_and_is_not_successful_empty():
    rows = [dict(id='q', question='Q', prediction='bad claim')]
    refs, unresolved, _ = pipeline.reference_rows(rows, {'q': dict(errors=[dict(
        root_id='E1', status='confirmed', quote='bad claim', claim='error')])})
    score = pipeline.scoring.score(refs, [dict(id='q', diagnostics=[], execution_failure=True,
        source_status='content_failure')], unresolved, preserve_zero_offsets=True)['summary']
    assert score['dataset_size'] == score['total_gt_errors'] == score['failed_questions'] == 1
    assert score['successful_empty_questions'] == 0 and score['recall'] == 0


def test_preflight_uses_request_builder_without_model_call_and_flags_oversize(tmp_path, monkeypatch):
    from scripts import audit_corrected_runtime as audit
    rows = [dict(id=str(i), question='Q', prediction='A', context='') for i in range(100)]
    vr.atomic(tmp_path / 'step_inputs.json', rows)
    def forbidden(*args, **kwargs):
        pytest.fail('Read-only audit must not invoke a model or finalize a case')
    monkeypatch.setattr(runner.c.corpus.Journal, 'call', forbidden)
    monkeypatch.setattr(runner, 'run_case', forbidden)
    monkeypatch.setattr(runner, 'token_count', lambda *args: p.LIMITS['context_tokens'])
    actual, report = audit.audit_cohort(tmp_path, 'step', configuration(), Tokenizer())
    assert actual == rows and report['arm_conditions'] == len(report['oversized']) == 400
    assert list(tmp_path.iterdir()) == [tmp_path / 'step_inputs.json']


def test_preflight_reference_rehearsal_does_not_publish_fake_scores(tmp_path):
    from scripts import audit_corrected_runtime as audit
    rows = [dict(id=str(i)) for i in range(600)]
    vr.atomic(tmp_path / 'final_labels_private.json', {r['id']: dict(
        verdict='correct' if int(r['id']) < 300 else 'incorrect') for r in rows})
    result = audit.audit_scoring(tmp_path, 'final', rows)
    assert result['questions'] == 600 and result['label_balance'] == {'correct': 300, 'incorrect': 300}
    assert not any('metric' in x.name for x in tmp_path.iterdir())


def test_preflight_saved_failure_is_audited_without_rerun_and_tamper_stops(tmp_path, monkeypatch):
    from scripts import audit_corrected_runtime as audit
    row = dict(id='q', question='Q', prediction='A')
    cfg = configuration()
    monkeypatch.setattr(pipeline, 'check_condition', lambda *args: cfg)
    calls = mock_transport(monkeypatch, [final([])])
    folder = tmp_path / 'checks/step/formal/pure/q'
    result = runner.run_case(folder, row, 'pure', 'step', None, cfg, Tokenizer())
    before = vr.sha(folder / 'result.json')
    report = audit.audit_results(tmp_path, cfg, {'step': [row], 'final': []})
    assert report['snapshot_results'] == report['evidence_files'] == len(calls) == 1
    assert report['outcomes']['step']['pure'] == {'content_failure': 1}
    assert vr.sha(folder / 'result.json') == before
    raw = folder / result['evidence'][0]['file']
    raw.write_text(raw.read_text() + ' ')
    with pytest.raises(ValueError, match='Saved raw evidence drift'):
        audit.audit_results(tmp_path, cfg, {'step': [row], 'final': []})


def test_full_conversation_tool_analysis_and_exact_success_replay(tmp_path,monkeypatch):
    answer={'diagnostics':[dict(message='unit error',evidence={'quote':'bad claim'},rule_refs=['k00'])]}
    calls=mock_transport(monkeypatch,[tool('search_memory',{'query':'磁场','offset':0},1),
                                    tool('read_memory',{'id':'k00'},2),final(answer)])
    row=dict(id='x',question='磁场单位?',prediction='bad claim')
    memory=p.FlatMemory(docs());cfg=configuration()
    result=runner.run_case(tmp_path,row,'flat_skills','step',memory,cfg,Tokenizer())
    assert result['status']=='complete' and not result['protocol_audit']['protocol_noncompliance']
    assert len(calls)==3
    assert 'analysis before 1' in json.dumps(calls[-1]['messages'])
    assert 'analysis before 2' in json.dumps(calls[-1]['messages'])
    assert '特斯拉' in json.dumps(calls[-1]['messages'],ensure_ascii=False)
    (tmp_path/'result.json').unlink()  # crash after receiving last response
    resumed=runner.run_case(tmp_path,row,'flat_skills','step',memory,cfg,Tokenizer())
    assert len(calls)==3 and resumed==result


def test_quote_repair_keeps_first_diagnosis_and_no_second_review(tmp_path,monkeypatch):
    original={'diagnostics':[dict(message='unchanged physical allegation',evidence={'quote':'wrong quote'},rule_refs=[])]}
    calls=mock_transport(monkeypatch,[final(original),final({'quote_patches':[{'index':0,'quote':'bad claim'}]})])
    result=runner.run_case(tmp_path,dict(id='a',question='Q',prediction='bad claim'),'pure','step',None,configuration(),Tokenizer())
    assert result['value']['diagnostics'][0]['evidence']['quote']=='bad claim'
    assert result['value']['diagnostics'][0]['message']=='unchanged physical allegation'
    assert vr.read(tmp_path/'first_diagnosis.json')==original
    assert len(calls)==2 and 'tools' not in calls[-1]


def test_invalid_output_is_failure_not_successful_empty(tmp_path,monkeypatch):
    mock_transport(monkeypatch,[final({'not_diagnostics':[]})])
    result=runner.run_case(tmp_path,dict(id='a',question='Q',prediction='A'),'pure','step',None,configuration(),Tokenizer())
    assert result['status']=='content_failure' and result['value'] is None


def test_final_array_is_retained_failure_without_resampling(tmp_path,monkeypatch):
    calls=mock_transport(monkeypatch,[final([dict(message='unchanged allegation',evidence={'quote':'A'})])])
    row=dict(id='a',question='Q',prediction='A');cfg=configuration()
    result=runner.run_case(tmp_path,row,'pure','step',None,cfg,Tokenizer())
    assert result['status']=='content_failure' and result['value'] is None
    assert result['reason']=='Expected diagnostics object'
    originals={str(f):vr.sha(f) for f in (tmp_path/'raw').rglob('request_*.json')}
    (tmp_path/'result.json').unlink()  # Recover after a complete stored final.
    again=runner.run_case(tmp_path,row,'pure','step',None,cfg,Tokenizer())
    assert again==result and len(calls)==1
    assert originals=={str(f):vr.sha(f) for f in (tmp_path/'raw').rglob('request_*.json')}


def test_registered_tool_content_failure_does_not_abort_or_reroll(tmp_path,monkeypatch):
    calls=mock_transport(monkeypatch,[final({})])
    def bad(*args,**kw):raise runner.ModelContentError('Registered tool arguments are not valid JSON')
    monkeypatch.setattr(runner.c.corpus.harmony,'decode_response',bad)
    cfg=configuration();row=dict(id='bad',question='Q',prediction='A')
    result=runner.run_case(tmp_path,row,'pure','step',None,cfg,Tokenizer())
    assert result['status']=='content_failure' and result['value'] is None and result['usage']['requests']==1
    assert runner.run_case(tmp_path,row,'pure','step',None,cfg,Tokenizer())==result and len(calls)==1


@pytest.mark.parametrize('query',[None,13,[],{}])
def test_invalid_search_query_gets_tool_error_with_unchanged_budget(tmp_path,monkeypatch,query):
    calls=mock_transport(monkeypatch,[tool('search_memory',{'query':query,'offset':0},1),final({'diagnostics':[]})])
    result=runner.run_case(tmp_path,dict(id='x',question='Q',prediction='A'),'flat_skills','step',
                           p.FlatMemory(docs()),configuration(),Tokenizer())
    assert result['status']=='complete' and len(calls)==2 and result['protocol_audit']['searches']==0
    returned=json.loads(calls[1]['messages'][-1]['content'])
    assert returned['result']['error']=='Search query must be a string'
    assert returned['budget']['responses_left']==11


def test_format_repair_receipt_cannot_change_protocol_or_retained_results(tmp_path,monkeypatch):
    repo=tmp_path/'repo';out=tmp_path/'run'
    for n in pipeline.SOURCES:
        f=repo/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('original')
    monkeypatch.setattr(pipeline.c,'ROOT',repo)
    vr.atomic(out/'source_evidence.json',{})
    cfg=dict(code={n:vr.sha(repo/n) for n in pipeline.SOURCES},frozen_files={},evidence_digest=vr.digest({}))
    vr.atomic(out/'plan.json',dict(configuration=cfg,configuration_sha256=vr.digest(cfg)))
    old_result=out/'checks/step/formal/pure/a/result.json';vr.atomic(old_result,{'status':'complete'})
    (repo/'scripts/gpt_oss_harmony_transport.py').write_text('decoder repair')
    receipt=dict(configuration_sha256=vr.digest(cfg),previous_code=cfg['code'],
        code={n:vr.sha(repo/n) for n in pipeline.SOURCES},
        retained_files={str(old_result.relative_to(out)):vr.sha(old_result)})
    vr.atomic(out/'final_format_repair.json',receipt);pipeline.verify(out)
    receipt['code']['rule_framework/corrected_verifier.py']='changed prompt'
    vr.atomic(out/'final_format_repair.json',receipt)
    with pytest.raises(ValueError,match='unrelated implementation'):pipeline.verify(out)
    receipt['code']['rule_framework/corrected_verifier.py']=cfg['code']['rule_framework/corrected_verifier.py']
    vr.atomic(out/'final_format_repair.json',receipt);vr.atomic(old_result,{'status':'changed'})
    with pytest.raises(ValueError,match='retained evidence drift'):pipeline.verify(out)


def test_step_reference_ambiguous_quote_not_assigned_first_occurrence():
    row=dict(id='x',question='Q',prediction='same same')
    refs={'x':dict(errors=[dict(root_id='E1',status='confirmed',quote='same',claim='error')])}
    data,unknown,counts=pipeline.reference_rows([row],refs)
    assert data[0]['physics_error_gt'][0]['span_valid'] is False
    assert counts['confirmed_unlocated']==1 and len(data[0]['physics_error_gt'])==1


def test_native_service_preserves_analysis_only_for_tool_assistant():
    from scripts.serve_continuous_context_vllm import preserve_tool_analysis
    parse=preserve_tool_analysis(lambda message:['original'],lambda text:('analysis',text))
    message={'role':'assistant','content':'retained reasoning','tool_calls':[{'id':'x'}]}
    original=copy.deepcopy(message)
    assert parse(message)==[('analysis','retained reasoning'),'original']
    assert message==original
    assert parse({'role':'user','content':'data'})==['original']
    assert parse({'role':'assistant','content':'final'})==['original']
    assert parse({'role':'assistant','content':'','tool_calls':[{'id':'x'}]})==['original']


def test_unknown_usage_consumes_budget_and_is_not_resampled(tmp_path,monkeypatch):
    attempts=[]
    def fail(self,stage,key,body):
        attempts.append(body)
        path=self.output/'raw'/stage/runner.c.digest(key)[:24]/'request_001.json'
        vr.atomic(path,dict(status='failed',retryable=True,usage=None,request=body))
        raise RuntimeError('Technical attempt cap reached; preserve unknown usage and failures')
    monkeypatch.setattr(runner.c.corpus.Journal,'call',fail)
    row=dict(id='a',question='Q',prediction='A');cfg=configuration()
    value=runner.run_case(tmp_path,row,'pure','step',None,cfg,Tokenizer())
    assert value['status']=='technical_failure' and value['value'] is None
    assert value['usage']['output_tokens_or_unknown_reservation']==16384
    assert len(value['evidence'])==1
    assert runner.run_case(tmp_path,row,'pure','step',None,cfg,Tokenizer())==value
    assert len(attempts)==1


def test_identity_failure_stops_instead_of_becoming_empty_result(tmp_path,monkeypatch):
    mock_transport(monkeypatch,[runner.c.corpus.IdentityError('wrong model')])
    with pytest.raises(runner.c.corpus.IdentityError):
        runner.run_case(tmp_path,dict(id='a',question='Q',prediction='A'),'pure','step',None,configuration(),Tokenizer())
    assert not(tmp_path/'result.json').exists()


@pytest.mark.parametrize('task',['step','final'])
@pytest.mark.parametrize('arm',p.ARMS)
def test_approved_revision_is_only_user_suffix(task,arm):
    row=dict(question='Q',prediction='A',context='C',reference='NEVER_VISIBLE')
    old=p.messages(row,arm,task)
    revised=p.messages(row,arm,task,revision=p.CHECK_REVISION)
    expected=copy.deepcopy(old)
    if arm!='pure':expected[-1]['content']+='\n\n'+p.REQUIRED_LOOKUP_SUFFIX
    assert revised==expected
    assert 'NEVER_VISIBLE' not in json.dumps(revised)
    with pytest.raises(ValueError):p.messages(row,arm,task,revision='unapproved')


def test_suffix_revision_keeps_auto_tools_and_total_budget(tmp_path,monkeypatch):
    calls=mock_transport(monkeypatch,[tool('search_memory',{'query':'磁场','offset':0},1),
        tool('read_memory',{'id':'k00'},2),final({'diagnostics':[]})])
    cfg=dict(configuration(),check_prompt_revision=p.CHECK_REVISION)
    result=runner.run_case(tmp_path,dict(id='x',question='Q',prediction='A'),
        'flat_skills','step',p.FlatMemory(docs()),cfg,Tokenizer())
    assert result['status']=='complete' and len(calls)==3
    assert all(b['tool_choice']=='auto' and b['temperature']==0 and b['top_p']==1 for b in calls)
    assert [b['max_tokens'] for b in calls]==[16384,16374,16364]
    assert calls[0]['messages'][-1]['content'].endswith(p.REQUIRED_LOOKUP_SUFFIX)


def test_revision_reuses_original_pure_pilot_condition(tmp_path,monkeypatch):
    vr.atomic(tmp_path/'construction_acceptance.json',{'passed':True})
    old=dict(configuration(),limits=p.LIMITS,protocol=p.VERSION,code={'old':'hash'})
    old_condition=pipeline.check_condition(tmp_path,old)
    cfg=dict(old,code={'new':'hash'},check_prompt_revision=p.CHECK_REVISION,
        pure_pilot_condition=old_condition)
    calls=[]
    def saved(folder,row,arm,task,memory,condition,tokenizer):
        calls.append((arm,condition));folder.mkdir(parents=True,exist_ok=True)
        value=dict(id=row['id'],status='complete',protocol_audit=dict(searches=int(arm!='pure'),
            reads=['k'] if arm!='pure' else []))
        vr.atomic(folder/'result.json',value);return value
    monkeypatch.setattr(pipeline.checking,'run_case',saved)
    pipeline.checks(tmp_path,cfg,None,'step','pilot',[{'id':'x'}],dict.fromkeys(p.ARMS))
    assert dict(calls)['pure']==old_condition
    assert all(v.get('check_prompt_revision')==p.CHECK_REVISION for a,v in calls if a!='pure')
    assert vr.read(tmp_path/'step_pilot_acceptance.json')['passed']


def test_zero_lookup_still_stops_revised_pilot(tmp_path,monkeypatch):
    vr.atomic(tmp_path/'construction_acceptance.json',{'passed':True})
    cfg=dict(configuration(),limits=p.LIMITS,protocol=p.VERSION,code={})
    def saved(folder,row,arm,*args):
        value=dict(id=row['id'],status='complete',protocol_audit=dict(searches=0,reads=[]))
        vr.atomic(folder/'result.json',value);return value
    monkeypatch.setattr(pipeline.checking,'run_case',saved)
    with pytest.raises(ValueError,match='Pilot engineering/budget'):
        pipeline.checks(tmp_path,cfg,None,'step','pilot',[{'id':'x'}],dict.fromkeys(p.ARMS))
    assert not vr.read(tmp_path/'step_pilot_acceptance.json')['passed']


def test_revision_freezes_parent_and_reuses_only_pure(tmp_path,monkeypatch):
    parent=tmp_path/'pipeline';out=parent/'check_prompt_revision';repo=tmp_path/'repo'
    for name in pipeline.SOURCES:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('old code')
    monkeypatch.setattr(pipeline.c,'ROOT',repo)
    exists=Path.exists
    monkeypatch.setattr(Path,'exists',lambda path:True if str(path)=='/proc' else exists(path))
    vr.atomic(parent/'pilot_inputs.json',[{'id':'p0'}])
    vr.atomic(parent/'source_evidence.json',{})
    cfg=dict(configuration(),root=str(tmp_path),limits=p.LIMITS,protocol=p.VERSION,
        code={n:vr.sha(repo/n) for n in pipeline.SOURCES},evidence_digest=vr.digest({}),
        frozen_files={'pilot_inputs.json':vr.sha(parent/'pilot_inputs.json')})
    vr.atomic(parent/'plan.json',dict(configuration=cfg,configuration_sha256=vr.digest(cfg)))
    vr.atomic(parent/'material_isolation.json',{'unresolved':[]})
    materials={}
    for name in ['catalog.json','tree_report.json','skills.json','skills_report.json','cases.json']:
        vr.atomic(parent/'materials'/name,{});materials['materials/'+name]=vr.sha(parent/'materials'/name)
    vr.atomic(parent/'construction_acceptance.json',dict(passed=True,files=materials))
    outputs={}
    for arm in p.ARMS:
        for i in range(6):
            f=parent/'checks/step/pilot'/arm/f'p{i}'/'result.json'
            vr.atomic(f,dict(evidence=[]));outputs[str(f.relative_to(parent))]=vr.sha(f)
    vr.atomic(parent/'step_pilot_acceptance.json',dict(passed=False,blocked=[],
        per_arm={a:dict(completed=6,read=0,searched=0) for a in p.ARMS},output_files=outputs))
    probe=parent/'native_io'/vr.sha(repo/'scripts/serve_continuous_context_vllm.py')[:16]/'acceptance.json'
    vr.atomic(probe,dict(passed=True,configuration_sha256=vr.digest(cfg),files={}))
    (parent/'service').mkdir()
    original_plan=(parent/'plan.json').read_bytes()
    (repo/'rule_framework/corrected_verifier.py').write_text('approved suffix')
    pipeline.prepare_check_revision(out,parent)
    assert (parent/'plan.json').read_bytes()==original_plan
    assert (out/'materials').resolve()==parent/'materials'
    assert (out/'checks/step/pilot/pure').resolve()==parent/'checks/step/pilot/pure'
    assert not (out/'checks/step/pilot/case_rag').exists()
    revised=vr.read(out/'plan.json')['configuration']
    assert revised['pure_pilot_condition']==pipeline.check_condition(parent,cfg)
    assert revised['transport']==cfg['transport'] and revised['limits']==cfg['limits']
    pipeline.verify(out)
    vr.atomic(parent/'checks/step/pilot/case_rag/p0/result.json',{'changed':True})
    with pytest.raises(ValueError,match='Parent experiment evidence drift'):
        pipeline.verify(out)


def test_local_tool_json_round_trip_does_not_hide_missing_context():
    text=json.dumps({'公式':'x = \\frac{a}{b}','title':'"quoted"'},ensure_ascii=False)
    class EscapedTemplate:
        def apply_chat_template(self,messages,**kw):
            return '<|start|>functions.search_memory to=assistant<|channel|>commentary<|message|>'+json.dumps(text)+'<|end|>'
    assert text in runner.rendered_tool_texts(runner.rendered(EscapedTemplate(),[dict(role='tool',content=text)],None))
    with pytest.raises(ValueError,match='Template dropped visible context'):
        runner.rendered(EscapedTemplate(),[dict(role='tool',content=text+'lost')],None)
    with pytest.raises(ValueError,match='Template dropped visible context'):
        runner.rendered(EscapedTemplate(),[dict(role='user',content=text)],None)


@pytest.mark.parametrize('task',['step','final'])
@pytest.mark.parametrize('arm',p.ARMS)
def test_language_revision_changes_only_approved_group_suffixes(task,arm):
    row=dict(question='Q',prediction='A')
    prior=p.messages(row,arm,task,revision=p.CHECK_REVISION)
    expected=copy.deepcopy(prior)
    if arm!='pure':expected[-1]['content']+='\n\n'+p.READ_SUFFIX
    if arm=='rule_tree':expected[-1]['content']+='\n\n'+p.TREE_LANGUAGE_SUFFIX
    assert p.messages(row,arm,task,revision=p.LANGUAGE_REVISION)==expected


@pytest.mark.parametrize('task',['step','final'])
@pytest.mark.parametrize('arm',p.ARMS)
def test_bilingual_amendment_changes_tree_only(task,arm):
    row=dict(question='Q',prediction='A')
    before=p.messages(row,arm,task,revision=p.LANGUAGE_REVISION)
    after=p.messages(row,arm,task,revision=p.BILINGUAL_REVISION)
    if arm=='rule_tree':
        before[-1]['content']=before[-1]['content'].removesuffix(p.TREE_LANGUAGE_SUFFIX)+p.TREE_BILINGUAL_SUFFIX
    assert after==before
