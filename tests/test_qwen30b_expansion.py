import copy
import json
from pathlib import Path

import pytest

from scripts import prepare_qwen30b_expansion as prep
from scripts import run_qwen30b_expansion as run
from tests.test_experience_first_120b import ROWS, TAXONOMY, HELDOUT, FakeJournal, response

p,b=prep.p,prep.b


def test_index_matches_existing_bench_screen_exact_numeric_near_embedded():
    questions=[
        'A particle of mass 20 kg follows a circular path of radius 30 metres. Find its acceleration and the tension in the string.',
        '计算电子的德布罗意波长。',
        'An ideal quantum gas occupies a spherical trap and has a specified chemical potential. Determine its heat capacity and density profile.',
    ]
    held=[{'id':str(i),'question':q} for i,q in enumerate(questions)]
    variants=questions+[questions[0].replace('20','9').replace('30','8'),
        'Additional premises. '+questions[2]+' Explain your answer.',
        questions[0].replace('tension in the string','normal force from the track'),
        'A monochromatic beam passes through a triangular glass prism. Determine the angle of minimum deviation.']
    for q in variants:
        _,excluded=p.isolate_sources([{'id':'candidate','question':q,'prediction':'[not generated]'}],held)
        actual={(x['reference_id'],x['reason']) for x in prep.QuestionIndex(held).matches(q)}
        expected={(x['heldout_id'],x['reason']) for row in excluded for x in row['matches']}
        assert actual==expected


def test_selection_is_before_outcomes_and_rejects_teacher_fields():
    held=[{'id':'h','question':'Solve the one dimensional infinite potential well.'}]
    previous=[{'id':'p','question':'Find the angular momentum of a rotating rigid wheel.'}]
    pool=[{'id':'a','question':'Calculate electron diffraction through a double slit.'},
          {'id':'b','question':'Obtain the thermal efficiency of a Carnot refrigerator.'}]
    selected,_=prep.select_new(pool,held,previous,limit=2)
    assert selected==prep.select_new(list(reversed(pool)),held,previous,limit=2)[0]
    with pytest.raises(ValueError,match='Only id/question'):
        prep.select_new([dict(pool[0],reference_answer='secret teacher answer')],held,previous,limit=1)
    with pytest.raises(ValueError,match='Insufficient'):
        prep.select_new(pool,held,previous,limit=3)


def test_bench_and_cross_source_duplicate_never_selected():
    q='A very long charged rod is placed parallel to an infinite conducting plane. Determine the induced electric charge density along the plane.'
    held=[{'id':'h','question':q}]
    old=[{'id':'old','question':'A 20 kg block stretches a spring by 0.3 metres. Calculate the spring constant.'}]
    pool=[{'id':'bench','question':q},{'id':'duplicate','question':old[0]['question'].replace('20','40')},
          {'id':'new','question':'Derive the partition function of a relativistic free electron gas.'}]
    rows,audit=prep.select_new(pool,held,old,limit=1,seed=0)
    assert [x['id'] for x in rows]==['new']


@pytest.mark.parametrize('q',['Use Figure 3.1 to calculate the acceleration.',
    'Find the current shown in the diagram.', 'Continue the previous problem.', '![plot](image.png) Determine the period.'])
def test_missing_context_prefilter(q):
    assert prep.MISSING_CONTEXT.search(q)


def test_domain_selection_has_no_error_or_answer_dependency():
    reserve=[{'id':str(i),'question':f'physics question {i}'} for i in range(5)]
    decisions={str(i):{'domain':d} for i,d in enumerate(['non_physics','physics','invalid','physics','physics'])}
    assert [x['id'] for x in run.freeze_new_questions(reserve,decisions,2)]==['1','3']
    with pytest.raises(ValueError,match='Insufficient'):
        run.freeze_new_questions(reserve,decisions,4)


def test_parent_cached_empty_result_replays_without_network(tmp_path,monkeypatch):
    parent=tmp_path/'parent';target=tmp_path/'child';key='a';stage='extract_experience'
    c={'model':'gpt-oss-test','reasoning_effort':'medium','max_tokens':16384,'max_requests':20,
       'max_request_input_bytes':400000,'attempts_per_job':3,'distill_max_requests':0,'api_base':'http://127.0.0.1:18083/v1'}
    j=run.e.Journal(target,c);payload={'sample':ROWS[0],'taxonomy':TAXONOMY,'max_rules_per_sample':2}
    value={'semantic_audit':{'status':'no_error_found'},'experience_rules':[]}
    body=run.r.request_body(c['model'],j.messages(stage,payload),16384,backend='vllm')
    body.update(stream=False,return_token_ids=True,reasoning_effort='medium');body.pop('stream_options',None)
    folder=parent/('arm_'+stage)/p.digest(key)[:24];raw=folder/'request_001.json'
    run.r.write_json(raw,{'request':body,'response':{'id':'test','model':c['model'],'text':json.dumps(value)},'error':None})
    run.r.write_json(folder/'result.json',{'request_file':raw.name,'request_sha256':run.r.sha256_file(raw),'value':value})
    monkeypatch.setattr(run.b.harmony,'decode_response',lambda raw,**kw:{'content':raw['text']})
    found,actual=run.parent_value(j,parent,stage,key,payload)
    assert found and actual==value
    assert not list(target.rglob('request_*.json'))
    with pytest.raises(ValueError,match='Parent request differs'):
        run.parent_value(j,parent,stage,key,dict(payload,max_rules_per_sample=1))


@pytest.mark.parametrize('text',['{"invalid“: unclosed', '{"x":"'+chr(92)+chr(10)+'"}'])
def test_prefix_receipt_and_bad_json_keep_invalid_content(tmp_path,monkeypatch,text):
    parent=tmp_path/'parent';target=tmp_path/'child';key='x';stage='taxonomy'
    c={'model':'test','reasoning_effort':'medium','max_tokens':16384,'max_requests':20,
       'max_request_input_bytes':400000,'attempts_per_job':3,'distill_max_requests':0,'api_base':'http://127.0.0.1:18083/v1'}
    j=run.e.Journal(target,c);payload={'domains':[]}
    value={'_unparsed_response_text':text}
    body=run.r.request_body(c['model'],j.messages(stage,payload),16384,backend='vllm')
    body.update(stream=False,return_token_ids=True,reasoning_effort='medium');body.pop('stream_options',None)
    raw=tmp_path/'older_request.json'
    run.r.write_json(raw,{'request':body,'response':{'id':'x','model':'test','text':text}})
    folder=parent/('arm_'+stage)/p.digest(key)[:24]
    run.r.write_json(folder/'prefix_replay.json',{'source':str(raw),'source_sha256':run.r.sha256_file(raw),'value_sha256':p.digest(value)})
    monkeypatch.setattr(run.b.harmony,'decode_response',lambda raw,**kw:{'content':raw['text']})
    assert run.parent_value(j,parent,stage,key,payload)==(True,value)


def test_generated_evidence_identity_and_truncation_are_preserved(tmp_path):
    config={'generator':run.g.MODEL,'temperature':.7,'top_p':.8,'top_k':20,'max_tokens':16384,'seed':20261021}
    row={'id':'tbr_1','question':'Question'};body={'model':config['generator'],'messages':run.g.solver_messages(row),
        **{k:config[k] for k in ('temperature','top_p','top_k','max_tokens')},'stream':False,
        'seed':int(p.digest([config['seed'],row['id']])[:8],16)%(2**31)}
    raw=tmp_path/'raw/generate/x/request_001.json'
    record={'status':'received','request':body,'response':{'model':config['generator'],
        'choices':[{'finish_reason':'length','message':{'content':'partial answer'}}]}}
    run.r.write_json(raw,record)
    value={'id':row['id'],'signature':p.digest([row,config]),'model':config['generator'],
        'prediction':'partial answer','finish_reason':'length','state':'incomplete',
        'evidence':{'file':str(raw.relative_to(tmp_path)),'sha256':run.r.sha256_file(raw)}}
    result=tmp_path/'generated/tbr_1.json';run.r.write_json(result,value)
    assert run.validate_generation(tmp_path,row,config)['state']=='incomplete'
    value['state']='complete';run.r.write_json(result,value)
    with pytest.raises(ValueError,match='differs'):run.validate_generation(tmp_path,row,config)


def test_parent_must_finish_before_any_new_inference(tmp_path):
    run.r.write_json(tmp_path/'selection.json',{'parent_build':str(tmp_path/'running_parent')})
    with pytest.raises(ValueError,match='has not completed'):run.parent_ready(tmp_path)


def test_build_reuses_base_and_only_calls_new_extraction_and_new_tree(tmp_path,monkeypatch):
    parent=tmp_path/'parent';base=tmp_path/'base';out=tmp_path/'expansion'
    construction={'model':'test','model_files':{},'reasoning_effort':'medium','max_tokens':16384,
        'max_requests':100,'max_request_input_bytes':400000,'attempts_per_job':3,
        'distill_max_requests':0,'api_base':'http://127.0.0.1:18083/v1','batch_size':12,'max_rules_per_sample':2}
    config={'construction':construction,'generator':{}}
    for path,value in [(base/'answers.json',ROWS[:1]),(parent/'plan.json',{'configuration':{}}),
        (parent/'topic_skeleton.json',TAXONOMY),(parent/'release_manifest.json',{'files':{}}),
        (out/'selection.json',{'base_corpus':str(base)}),(out/'admitted_new_answers.json',ROWS[1:]),
        (out/'new_questions.json',[{k:v for k,v in ROWS[1].items() if k in ('id','question')}]),
        (out/'bench_questions.json',HELDOUT),
        (out/'generation_release.json',{'files':{},'configuration_sha256':p.digest(config),
                                        'admitted_source_sha256':p.digest(ROWS[1:])})]:
        run.r.write_json(path,value)
    calls=FakeJournal();reused=[]
    def replay(journal,parent,stage,key,payload):
        reused.append((stage,key))
        return True,copy.deepcopy(TAXONOMY if stage=='taxonomy' else response())
    monkeypatch.setattr(run,'parent_value',replay)
    monkeypatch.setattr(run.e.Journal,'__call__',lambda self,*args:calls(*args))
    monkeypatch.setattr(run,'auditor_ready',lambda config:None)
    monkeypatch.setattr(run,'validate_generation',lambda *args:None)
    monkeypatch.setattr(run,'verify',lambda *args:None)
    run.build(out,config,parent)
    result=run.b.read(out/'build/release_manifest.json')
    assert result['counts']['rules']==1
    assert result['counts']['support_levels']=={'multi_question':1}
    assert {x[0] for x in reused}=={'taxonomy','extract_experience'}
    assert ('extract_experience','a') in reused
    assert [(stage,key) for stage,key,_ in calls.stages if stage=='extract_experience']==[('extract_experience','b')]
    assert any(stage=='cluster_rules' for stage,_,_ in calls.stages)
    assert not result['physical_quality_passed'] and not result['usable_for_runtime']
    count=len(calls.stages)
    run.build(out,config,parent)
    assert len(calls.stages)==count  # Released result resumes with zero inference.
    with (out/'build/catalog.json').open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='evidence changed'):run.build(out,config,parent)


def test_all_reused_generation_code_is_frozen():
    assert set(run.g.SOURCES).issubset(run.SOURCES)


def test_split_cluster_preserves_members_and_exposes_new_groups_without_editing_input():
    payload={'topic_key':'mechanics/topic','groups':[{'id':'G1','name':'old','summary':'old boundary'}],
             'items':[{'id':f'I{i+1}','rule':{'text':f'leaf {i}'}} for i in range(10)]}
    original=copy.deepcopy(payload);seen=[]
    def call(stage,key,wire):
        seen.append(copy.deepcopy(wire))
        assert stage=='cluster_rules' and len(wire['items'])<=4
        if len(seen)==1:
            return {'new_groups':[{'key':'new1','name':'created','summary':'new boundary'}],
                    'assignments':[{'item_id':x['id'],'group_id':'new1'} for x in wire['items']]}
        assert wire['groups'][-1]=={'id':'G2','name':'created','summary':'new boundary'}
        return {'new_groups':[], 'assignments':[{'item_id':x['id'],'group_id':'G2'} for x in wire['items']]}
    result,keys=run.split_cluster_batch('topic:324',payload,call)
    assert payload==original and [len(x['items']) for x in seen]==[4,4,2]
    assert [a['item_id'] for a in result['assignments']]==[x['id'] for x in payload['items']]
    assert {a['group_id'] for a in result['assignments']}=={result['new_groups'][0]['key']}
    assert keys==['topic:324:batch4:0','topic:324:batch4:4','topic:324:batch4:8']


@pytest.mark.parametrize('kind',['duplicate_member','unknown_group','collision','empty_group'])
def test_split_cluster_rejects_invalid_subbatch_without_next_call(kind):
    payload={'topic_key':'t','groups':[{'id':'G1','name':'old','summary':'s'}],
             'items':[{'id':f'I{i}'} for i in range(8)]};calls=[]
    def call(*args):
        calls.append(args)
        value={'new_groups':[], 'assignments':[{'item_id':f'I{i}','group_id':'G1'} for i in range(4)]}
        if kind=='duplicate_member':value['assignments'][1]['item_id']='I0'
        if kind=='unknown_group':value['assignments'][0]['group_id']='G9'
        if kind=='collision':value['new_groups']=[{'key':'G1','name':'x','summary':'x'}]
        if kind=='empty_group':value['new_groups']=[{'key':'unused','name':'x','summary':'x'}]
        return value
    with pytest.raises(ValueError):run.split_cluster_batch('t:0',payload,call)
    assert len(calls)==1


def test_recovery_binding_does_not_accept_changed_prompt_or_prefix(tmp_path,monkeypatch):
    old={'scripts/run_qwen30b_expansion.py':'old','prompt.py':'same'}
    current={**old,'scripts/run_qwen30b_expansion.py':'new'}
    monkeypatch.setattr(run,'hashes',lambda:current)
    plan={'configuration':{'source_hashes':old},'configuration_sha256':'config'}
    run.r.write_json(tmp_path/'raw.json',{'original':'failure'})
    fix={'policy':'cluster_batch4_after_completed_prefix_v1','configuration_sha256':'config',
         'original_sources':old,'replacement_sources':dict(current),'old_batch_size':12,'new_batch_size':4,
         'prefix_files':{'raw.json':run.r.sha256_file(tmp_path/'raw.json')}}
    run.r.write_json(tmp_path/run.CLUSTER_RECOVERY,fix)
    assert run.verify_cluster_recovery(tmp_path,plan)==fix
    current['prompt.py']='changed'
    with pytest.raises(ValueError,match='binding'):run.verify_cluster_recovery(tmp_path,plan)
    current['prompt.py']='same'
    run.r.write_json(tmp_path/'raw.json',{'original':'overwritten'})
    with pytest.raises(ValueError,match='prefix changed'):run.verify_cluster_recovery(tmp_path,plan)
