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


def test_missing_completion_preserves_accepted_decisions_and_full_context():
    payload={'topic_key':'t','groups':[{'id':'G1','name':'old','summary':'old'}],
             'items':[{'id':f'I{i}','rule':{'text':f'full leaf {i}'}} for i in range(4)]}
    draft={'new_groups':[{'key':'new1','name':'created','summary':'unchanged'}],
           'assignments':[{'item_id':'I0','group_id':'new1'},
                          {'item_id':'I1','group_id':'G1'},{'item_id':'I3','group_id':'G1'}]}
    before=copy.deepcopy((payload,draft));calls=[]
    def complete(stage,key,wire):
        calls.append((stage,key,wire))
        assert wire=={'original':payload,'draft':draft,'missing_ids':['I2']}
        return {'new_groups':[], 'assignments':[{'item_id':'I2','group_id':'new1'}]}
    result=run.complete_cluster_members('t:0:batch4:0',payload,draft,complete)
    assert len(calls)==1 and calls[0][0]==run.MISSING_STAGE
    assert (payload,draft)==before
    assert result['new_groups']==draft['new_groups']
    assert result['assignments'][:3]==draft['assignments']
    assert not run.missing_cluster_ids(result,payload)


@pytest.mark.parametrize('kind',['missing','overwrite','collision','unknown_group','duplicate','unused'])
def test_completion_failure_is_terminal_no_second_call(kind):
    payload={'groups':[{'id':'G1','name':'old','summary':'s'}], 'items':[{'id':'I1'},{'id':'I2'}]}
    draft={'new_groups':[],'assignments':[{'item_id':'I1','group_id':'G1'}]};calls=[]
    def call(*args):
        calls.append(args)
        result={'new_groups':[],'assignments':[{'item_id':'I2','group_id':'G1'}]}
        if kind=='missing':result['assignments']=[]
        if kind=='overwrite':result['assignments'][0]['item_id']='I1'
        if kind=='collision':result['new_groups']=[{'key':'G1','name':'changed','summary':'s'}]
        if kind=='unknown_group':result['assignments'][0]['group_id']='G2'
        if kind=='duplicate':result['assignments']*=2
        if kind=='unused':result['new_groups']=[{'key':'new','name':'unused','summary':'s'}]
        return result
    with pytest.raises(ValueError):run.complete_cluster_members('key',payload,draft,call)
    assert len(calls)==1


def test_invalid_original_does_not_get_completion_and_success_does_not_call_it():
    payload={'groups':[{'id':'G1','name':'old','summary':'s'}], 'items':[{'id':'I1'},{'id':'I2'}]}
    def never(*args):raise AssertionError('unauthorized extra completion')
    bad={'new_groups':[],'assignments':[{'item_id':'I1','group_id':'G1'}]*2}
    with pytest.raises(ValueError):run.split_cluster_batch('k',payload,lambda *a:bad,never)
    good={'new_groups':[],'assignments':[{'item_id':x['id'],'group_id':'G1'} for x in payload['items']]}
    assert run.split_cluster_batch('k',payload,lambda *a:good,never)[0]==good


def test_completion_can_create_group_without_renaming_existing_group():
    payload={'groups':[], 'items':[{'id':'I1'},{'id':'I2'}]}
    draft={'new_groups':[{'key':'a','name':'A','summary':'A'}],
           'assignments':[{'item_id':'I1','group_id':'a'}]}
    add={'new_groups':[{'key':'b','name':'B','summary':'B'}],
         'assignments':[{'item_id':'I2','group_id':'b'}]}
    result=run.complete_cluster_members('k',payload,draft,lambda *a:add)
    assert result['new_groups']==draft['new_groups']+add['new_groups']


def test_completion_messages_keep_original_context_and_only_request_missing():
    journal=object.__new__(run.ExpansionJournal)
    original={'topic_key':'t','groups':[], 'items':[{'id':'I1','text':'<|literal|>'}]}
    draft={'new_groups':[],'assignments':[]}
    messages=journal.messages(run.MISSING_STAGE,{'original':original,'draft':draft,'missing_ids':['I1']})
    assert messages[:2]==journal.messages('cluster_rules',original)
    assert json.loads(messages[2]['content'])==draft
    assert 'I1' in messages[-1]['content'] and '唯一一次' in messages[-1]['content']
    assert run.MISSING_STAGE in journal.supported_stages()


def test_missing_amendment_chains_old_receipt_without_replacing_it(tmp_path,monkeypatch):
    old={'scripts/run_qwen30b_expansion.py':'old','prompt.py':'same'}
    previous={**old,'scripts/run_qwen30b_expansion.py':'batch4'}
    current={**old,'scripts/run_qwen30b_expansion.py':'missing'}
    monkeypatch.setattr(run,'hashes',lambda:current)
    plan={'configuration':{'source_hashes':old},'configuration_sha256':'config'}
    run.r.write_json(tmp_path/'raw.json',{'raw':'unchanged'})
    fix={'policy':'cluster_batch4_after_completed_prefix_v1','configuration_sha256':'config',
         'original_sources':old,'replacement_sources':previous,'old_batch_size':12,'new_batch_size':4,
         'prefix_files':{'raw.json':run.r.sha256_file(tmp_path/'raw.json')}}
    run.r.write_json(tmp_path/run.CLUSTER_RECOVERY,fix)
    oldsha=run.r.sha256_file(tmp_path/run.CLUSTER_RECOVERY)
    follow={'policy':'cluster_missing_only_once_v1','configuration_sha256':'config',
            'previous_receipt_sha256':oldsha,'previous_sources':previous,'replacement_sources':dict(current),
            'instruction_sha256':p.digest(run.MISSING_INSTRUCTION),'max_completion_rounds':1,
            'prefix_files':{run.CLUSTER_RECOVERY:oldsha}}
    assert run.verify_cluster_recovery(tmp_path,plan,proposed=follow)==fix
    assert run.r.sha256_file(tmp_path/run.CLUSTER_RECOVERY)==oldsha
    follow['max_completion_rounds']=2
    with pytest.raises(ValueError,match='binding'):run.verify_cluster_recovery(tmp_path,plan,proposed=follow)
    follow['max_completion_rounds']=1;follow['previous_sources']={}
    with pytest.raises(ValueError,match='binding'):run.verify_cluster_recovery(tmp_path,plan,proposed=follow)


def test_prompt_revision_preserves_old_messages_full_catalog_and_completion_context():
    journal=object.__new__(run.ExpansionJournal)
    payload={'topic_key':'t','groups':[{'id':f'G{i}','name':f'n{i}','summary':f's{i}'} for i in range(370)],
             'items':[{'id':f'I{i}','rule':{'text':f'leaf {i}'}} for i in range(4)]}
    before=copy.deepcopy(payload)
    old=run.e.Journal.messages(journal,'cluster_rules',payload)
    new=journal.messages(run.PROMPT_STAGE,payload)
    assert journal.messages('cluster_rules',payload)==old
    assert new[0]['content']==old[0]['content']+'\n'+run.CLUSTER_INSTRUCTION
    assert new[1:]==old[1:] and payload==before
    draft={'new_groups':[],'assignments':[{'item_id':'I0','group_id':'G0'}]}
    wire={'original':payload,'draft':draft,'missing_ids':['I1','I2','I3']}
    assert journal.messages(run.MISSING_STAGE,wire)[:2]==old
    assert journal.messages(run.PROMPT_MISSING_STAGE,wire)[:2]==new
    assert journal.messages(run.MISSING_STAGE,wire)[2:]==journal.messages(run.PROMPT_MISSING_STAGE,wire)[2:]
    assert {run.PROMPT_STAGE,run.PROMPT_MISSING_STAGE}<=journal.supported_stages()


def test_prompt_routing_replays_all_old_results_and_never_overwrites_failed_raw(tmp_path):
    old_key='t:480:batch4:8';failed_key='t:492:batch4:0';next_key='t:492:batch4:4'
    old_rel='build/arm_cluster_rules/'+p.digest(old_key)[:24]+'/result.json'
    failed_folder=tmp_path/'build/arm_cluster_rules'/p.digest(failed_key)[:24]
    run.r.write_json(tmp_path/old_rel,{'value':'old'})
    run.r.write_json(failed_folder/'request_001.json',{'finish_reason':'length'})
    before=run.r.sha256_file(failed_folder/'request_001.json')
    fix={'completed_cluster_results':[old_rel],'failed_key':failed_key}
    assert run.cluster_stage(tmp_path,old_key,fix)=='cluster_rules'
    assert run.cluster_stage(tmp_path,failed_key,fix)==run.PROMPT_STAGE
    assert run.cluster_stage(tmp_path,next_key,fix)==run.PROMPT_STAGE
    assert run.r.sha256_file(failed_folder/'request_001.json')==before
    unexpected=tmp_path/'build/arm_cluster_rules'/p.digest(next_key)[:24]/'result.json'
    run.r.write_json(unexpected,{'value':'unregistered'})
    with pytest.raises(ValueError,match='Unregistered'):run.cluster_stage(tmp_path,next_key,fix)


def test_new_prompt_completion_retains_once_only_validation():
    payload={'groups':[{'id':'G1','name':'old','summary':'s'}],'items':[{'id':'I1'},{'id':'I2'}]}
    draft={'new_groups':[],'assignments':[{'item_id':'I1','group_id':'G1'}]};calls=[]
    def call(stage,key,wire):
        calls.append((stage,key,wire))
        return {'new_groups':[],'assignments':[]}
    with pytest.raises(ValueError,match='still omits'):
        run.complete_cluster_members('k',payload,draft,call,stage=run.PROMPT_MISSING_STAGE)
    assert len(calls)==1 and calls[0][0]==run.PROMPT_MISSING_STAGE
    assert calls[0][2]['draft']==draft and calls[0][2]['original']==payload


def test_prompt_amendment_appends_third_source_binding(tmp_path,monkeypatch):
    old={'scripts/run_qwen30b_expansion.py':'old','prompt.py':'same'}
    batch={**old,'scripts/run_qwen30b_expansion.py':'batch'}
    missing={**old,'scripts/run_qwen30b_expansion.py':'missing'}
    current={**old,'scripts/run_qwen30b_expansion.py':'prompt'}
    monkeypatch.setattr(run,'hashes',lambda:current)
    plan={'configuration':{'source_hashes':old},'configuration_sha256':'config'}
    run.r.write_json(tmp_path/'failure.json',{'original':'length'})
    prefix={'failure.json':run.r.sha256_file(tmp_path/'failure.json')}
    fix={'policy':'cluster_batch4_after_completed_prefix_v1','configuration_sha256':'config',
         'original_sources':old,'replacement_sources':batch,'old_batch_size':12,'new_batch_size':4,'prefix_files':prefix}
    run.r.write_json(tmp_path/run.CLUSTER_RECOVERY,fix)
    follow={'policy':'cluster_missing_only_once_v1','configuration_sha256':'config',
            'previous_receipt_sha256':run.r.sha256_file(tmp_path/run.CLUSTER_RECOVERY),
            'previous_sources':batch,'replacement_sources':missing,
            'instruction_sha256':p.digest(run.MISSING_INSTRUCTION),'max_completion_rounds':1,'prefix_files':prefix}
    run.r.write_json(tmp_path/run.MISSING_RECOVERY,follow)
    previous_sha=run.r.sha256_file(tmp_path/run.MISSING_RECOVERY)
    concise={'policy':'cluster_item_focused_after_prefix_v1','configuration_sha256':'config',
             'previous_receipt_sha256':previous_sha,'previous_sources':missing,'replacement_sources':dict(current),
             'instruction_sha256':p.digest(run.CLUSTER_INSTRUCTION),'instruction':run.CLUSTER_INSTRUCTION,
             'new_stage':run.PROMPT_STAGE,'prefix_files':prefix,'failed_request':'failure.json',
             'completed_cluster_results':[]}
    assert run.verify_cluster_recovery(tmp_path,plan,proposed=concise)==fix
    assert run.r.sha256_file(tmp_path/run.MISSING_RECOVERY)==previous_sha
    for field,value in [('instruction','changed'),('previous_sources',batch),('instruction_sha256','changed')]:
        bad={**concise,field:value}
        with pytest.raises(ValueError,match='binding'):run.verify_cluster_recovery(tmp_path,plan,proposed=bad)
    run.r.write_json(tmp_path/'failure.json',{'original':'overwritten'})
    with pytest.raises(ValueError,match='prefix changed'):run.verify_cluster_recovery(tmp_path,plan,proposed=concise)
