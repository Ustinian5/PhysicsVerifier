import copy
import json

import pytest

from rule_framework import flat_skill_rag as p
from scripts import run_organization_comparison as run


ROW = {'id':'sample', 'question':'恒力加速度 acceleration force', 'prediction':'a=F/m。', 'context':''}
QUERIES = {'queries':['恒力 加速度 constant force acceleration']}


def document(sid, kind='skill', title='恒力 加速度', text='核验力和质量；a=F/m。'):
    return {'id':sid,'kind':kind,'title':title,'text':text,'source_ids':['private_provenance']}


def catalog():
    rules = [{'rule_id':f'r{i}','title':f'恒力 加速度{i}', 'summary':'constant force acceleration',
              'trigger':'质量不变','check_logic':'a=F/m','error_type':'relation','symbolic_hint':'F=ma'}
             for i in range(4)]
    return {'domains':[{'id':'m','name':'恒力','summary':'acceleration','topics':[
        {'id':'t','name':'加速度','summary':'force','rules':rules,'scenario_clusters':[
            {'id':str(i),'name':f'恒力 加速度{i}','summary':'acceleration','rule_ids':[f'r{i}'], 'rule_groups':[]}
            for i in range(4)]}]}]}


def test_skills_count_is_not_capped_at_three():
    value={'answer_status':'has_error','case_note':'局部关系已核验。','uncertainties':[],
           'skills':[{'title':str(i),'text':'可迁移检查方法'} for i in range(7)]}
    assert p.validate_experience(value)==value


def test_stage_has_no_diagnosis_handoff_or_label_access():
    messages=p.query_messages(dict(ROW,answer='SECRET_ANSWER',physics_error_gt=['SECRET_GT']))
    rendered=json.dumps(messages,ensure_ascii=False)
    assert 'SECRET' not in rendered
    assert 'PREPARE stage' in messages[0]['content']
    assert 'do not diagnose' in messages[1]['content']
    assert 'Return only' not in messages[0]['content'] or 'final diagnostics' in messages[0]['content']
    continued=p.continuation_messages(ROW,QUERIES,[])
    assert continued[:2]==messages
    assert continued[2]['role']=='assistant' and json.loads(continued[2]['content'])==QUERIES


def test_tree_scope_actually_excludes_a_branch_and_has_no_flat_fallback():
    memory=p.TreeMemory(catalog())
    hits,trace=memory.retrieve('恒力 加速度 acceleration')
    assert trace['eligible_leaf_count']==2
    assert len(hits)==2 and len(memory.memory.docs)==4
    assert len(trace['routes'])>=5 and not trace['flat_fallback']
    hits,trace=memory.retrieve('unrelatedphotons')
    assert hits==[] and trace['eligible_leaf_count']==0


def test_flat_skill_and_case_retrieval_use_identical_query_and_complete_views():
    for arm,kind in [('flat_skills','skill'),('case_rag','case')]:
        mem=p.FlatMemory([document('x',kind)])
        packet=p.reference_packet(ROW,arm,QUERIES,mem,lambda m:len(json.dumps(m)))
        assert packet['retrieval']['query']==QUERIES['queries'][0]
        assert packet['references']==[mem.view('x')]
        assert 'private_provenance' not in json.dumps(packet)
        assert packet['retrieval']['algorithm']=='flat_bm25'


def test_pure_never_receives_references():
    packet=p.reference_packet(ROW,'pure',QUERIES,None,lambda m:len(json.dumps(m)))
    assert not packet['references'] and not packet['retrieval']['ranked']
    with pytest.raises(ValueError,match='cannot access'):
        p.reference_packet(ROW,'pure',QUERIES,p.FlatMemory([]),lambda m:1)


def test_full_item_budget_never_clips_case_correction(monkeypatch):
    mem=p.FlatMemory([document('large','case',text='恒力'+('x'*20000)+'CORRECTION')])
    packet=p.reference_packet(ROW,'case_rag',QUERIES,mem,lambda m:len(json.dumps(m)))
    assert packet['references']==[] and packet['retrieval']['omitted'][0]['id']=='large'
    assert 'CORRECTION' not in json.dumps(packet['messages'])


@pytest.mark.parametrize('value',[{'queries':[]},{'queries':['a']*4},{'queries':['a'],'diagnostics':[]}])
def test_query_schema_rejects_unbound_content(value):
    with pytest.raises(ValueError):p.validate_queries(value)


def test_citations_are_observed_not_a_gate_for_independent_findings():
    a=run.reference_audit({'diagnostics':[{'reference_ids':[]}]},['r1'])
    assert a['passed']
    b=run.reference_audit({'diagnostics':[{'reference_ids':['invented']}]},['r1'])
    assert not b['passed']


class Tokenizer:
    def apply_chat_template(self,*args,**kwargs):return list(range(20))
    def encode(self,*args,**kwargs):return [1,2]


def test_real_engine_replays_shared_prefix_and_completed_sample(tmp_path,monkeypatch):
    calls=[]
    def transport(client,**kw):
        calls.append(kw)
        is_query=len(kw['messages'])==2
        content=json.dumps(QUERIES if is_query else {'diagnostics':[{
            'message':'独立判断','evidence':{'quote':'a=F/m'},'reference_ids':[]}]})
        record={'model_ids':[run.r.EXPECTED_MODEL], 'finish_reasons':['stop'],
                'content':content,'usage':{'prompt_tokens':20,'completion_tokens':10}}
        body={'messages':kw['messages'],'max_tokens':kw['max_tokens']}
        run.vr.atomic(kw['request_path'],{'request':body})
        run.vr.atomic(kw['raw_path'],{'content':content})
        run.vr.atomic(kw['response_path'],record)
        return record
    monkeypatch.setattr(run.runtime.transport,'chat_response',transport)
    plan={'configuration_sha256':'binding','configuration':{
        'api_base':'http://127.0.0.1:18083/v1','limits':dict(run.runtime.LIMITS,
        review_protocol=run.old.continuous.PROTOCOL)}}
    a=run.run_condition(tmp_path,ROW,'formal','pure',None,plan,Tokenizer(),None)
    assert len(calls)==2 and a['trace']['model_requests']==2
    assert a['trace']['output_token_limit_reserved']==20
    assert a['diagnostics'][0]['reference_ids']==[]
    b=run.run_condition(tmp_path,ROW,'formal','pure',None,plan,Tokenizer(),None)
    assert b==a and len(calls)==2
    mem=p.FlatMemory([document('skill')])
    c=run.run_condition(tmp_path,ROW,'formal','flat_skills',mem,plan,Tokenizer(),None)
    assert len(calls)==3 and c['trace']['response_evidence'][0]==a['trace']['response_evidence'][0]
    assert c['retrieval']['provided_complete_ids']==['skill']
    assert c['first_diagnosis_sha256']==run.vr.sha(c['first_diagnosis_path'])
    queryfile=tmp_path/'queries'/run.vr.digest(ROW['id'])[:24]/'raw.json'
    queryfile.write_text('tampered')
    with pytest.raises(ValueError,match='evidence changed'):
        run.query(tmp_path,ROW,plan,Tokenizer(),None)


@pytest.mark.parametrize('bad_pilot',[False,True])
def test_experience_checkpoint_and_material_release(tmp_path,monkeypatch,bad_pilot):
    rows=[dict(ROW,id='first'),dict(ROW,id='second')]
    run.vr.atomic(tmp_path/'sources.json',rows)
    config={'benchmarks':{},'bench_sha256':'bench'}
    plan={'configuration':config,'configuration_sha256':'binding'}
    calls=[]
    class Journal:
        def __init__(self,output,*args):
            path=output/'request_index.json'
            self.index=run.vr.unseal(path) if path.exists() else {'files':{}}
        def __call__(self,stage,key,payload):
            calls.append(key)
            if bad_pilot:return {'wrong_schema':True}
            return {'answer_status':'has_error','case_note':'具体关系核查','uncertainties':[],
                    'skills':[] if key=='second' else [{'title':'关系','text':'检查完整条件'}]}
    monkeypatch.setattr(run,'ExperienceJournal',Journal)
    monkeypatch.setattr(p,'PILOT_SIZE',1)
    monkeypatch.setattr(run.old.b,'benchmark_questions',lambda:(['heldout'],{}))
    monkeypatch.setattr(run.old.p,'isolate_sources',lambda rows,heldout:(rows,[]))
    if bad_pilot:
        for _ in range(2):
            with pytest.raises(ValueError,match='pilot content protocol failed'):
                run.build(tmp_path,plan,Tokenizer())
        assert calls==['first']  # Resume cannot silently bypass failed pilot.
        assert not (tmp_path/'experience/release_manifest.json').exists()
    else:
        materials=run.build(tmp_path,plan,Tokenizer())
        assert len(materials['case_rag'].docs)==2
        assert len(materials['flat_skills'].docs)==1
        again=run.build(tmp_path,plan,Tokenizer())
        assert calls==['first','second'] and again['flat_skills'].docs==materials['flat_skills'].docs
