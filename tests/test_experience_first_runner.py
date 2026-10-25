import json
import pytest
from scripts import run_experience_first_120b as e
from tests.test_experience_first_120b import TAXONOMY, ROWS, response


def config():
    return {'max_requests':30,'max_tokens':16384,'max_request_input_bytes':400000,
        'attempts_per_job':3,'model':e.r.EXPECTED_MODEL,'reasoning_effort':'medium',
        'api_base':'http://127.0.0.1:18083/v1','json_policy':'literal_only','distill_max_requests':0}


def test_completed_raw_response_replays_without_network(tmp_path,monkeypatch):
    c=config();payload={'sample':ROWS[0],'taxonomy':TAXONOMY,'max_rules_per_sample':2}
    stage,key='extract_experience',ROWS[0]['id'];value=response()
    fingerprint=e.p.digest([stage,key,payload,c['model'],c])
    folder=tmp_path/('arm_'+stage)/e.p.digest(key)[:24]
    e.r.write_json(folder/'request_001.json',{'client_metadata':{'fingerprint':fingerprint},
        'request':{'max_tokens':16384},'response':{'id':'test','model':c['model'],'text':json.dumps(value)},'error':None})
    monkeypatch.setattr(e.b.harmony,'decode_response',lambda raw,**kw:{'content':raw['text']})
    def forbidden(*a,**kw):raise AssertionError('Repeated a completed request')
    monkeypatch.setattr(e.b.Transport,'post',forbidden)
    j=e.Journal(tmp_path,c)
    assert j(stage,key,payload)==value
    assert j(stage,key,payload)==value
    assert len(list(folder.glob('request_*.json')))==1
    with pytest.raises(ValueError,match='input mismatch'):
        j(stage,key,dict(payload,max_rules_per_sample=1))


def test_wrong_identity_and_incomplete_response_are_never_accepted(tmp_path):
    path=tmp_path/'response.json'
    e.r.write_json(path,{'response':{'id':'x','model':'another_model'}})
    j=e.Journal(tmp_path,config())
    with pytest.raises(ValueError,match='unknown model'):j.decode_record(path,'extract_experience')


def test_new_runner_dispatches_new_prompt_not_old_pair_support(tmp_path):
    j=e.Journal(tmp_path,config())
    messages=j.messages('extract_experience',{'sample':ROWS[0],'taxonomy':TAXONOMY,'max_rules_per_sample':2})
    assert '一道题足以' in messages[0]['content']
    assert j.supported_stages()=={'taxonomy','extract_experience','review_experience','repair_experience','cluster_rules'}


def test_short_topic_ids_roundtrip_without_guessing():
    from scripts.generate_experience_rules import experience_120b_prompt, normalize_120b_experience
    payload=json.loads(experience_120b_prompt(ROWS[0],TAXONOMY)[1])
    assert payload['topics'][0]['key']=='T1'
    value=response();value['topic_key']='T1'
    result=normalize_120b_experience(value,sample=ROWS[0],taxonomy=TAXONOMY)
    assert result['topic_key']=='electromagnetism/magnetostatics'
    value['topic_key']='T999'
    with pytest.raises(ValueError,match='Unknown experience topic'):
        normalize_120b_experience(value,sample=ROWS[0],taxonomy=TAXONOMY)


def test_parent_handoff_reuses_only_verified_inputs_and_detects_tampering(tmp_path,monkeypatch):
    release=tmp_path/'release_manifest.json';e.r.write_json(release,{'fixture':True})
    e.r.write_json(tmp_path/'answers.json',ROWS)
    binding={'generator':'Qwen3-30B-A3B-Instruct-2507',
        'answers_sha256':e.r.sha256_file(tmp_path/'answers.json'),
        'source_model_evidence':'all_complete_generation_requests_and_responses_verified'}
    monkeypatch.setattr(e.released_corpus,'load',lambda *a,**kw:(binding,ROWS))
    monkeypatch.setattr(e.b,'benchmark_questions',lambda:([{'id':'b','question':'量子束缚态的波函数。'}],{}))
    parent=tmp_path/'parent';child=tmp_path/'child'
    e.prepare(parent,release)
    def forbidden(*a,**kw):raise AssertionError('Repeated the previous generation handoff')
    monkeypatch.setattr(e.released_corpus,'load',forbidden)
    e.prepare(child,release,parent)
    assert e.b.read(child/'inputs.json')==e.b.read(parent/'inputs.json')
    assert not (child/'arm_extract_experience').exists()
    e.verify(child)
    rows=e.b.read(parent/'inputs.json');rows[0]['prediction']+='altered'
    e.r.write_json(parent/'inputs.json',rows)
    with pytest.raises(ValueError,match='Parent frozen artifact changed'):e.verify(child)


def test_v3_example_does_not_teach_composite_categories():
    from scripts.generate_experience_rules import EXPERIENCE_120B_INSTRUCTION as prompt
    assert 'concept|logic' not in prompt
    assert '禁止使用竖线或组合标签' in prompt
    assert '等价表达不是矛盾' in prompt
    assert '不得标included' in prompt


def test_review_keeps_full_context_and_draft():
    j=e.Journal(__import__('pathlib').Path('/tmp/unused'),config())
    original={'sample':ROWS[0],'taxonomy':TAXONOMY,'max_rules_per_sample':2}
    draft=response()
    messages=j.messages('review_experience',{'original':original,'draft':draft})
    assert messages[:2]==j.messages('extract_experience',original)
    assert json.loads(messages[2]['content'])==draft
    assert [m['role'] for m in messages]==['system','user','assistant','user']


def test_prefix_replay_requires_identical_request(tmp_path,monkeypatch):
    c=config();parent=tmp_path/'parent';c['draft_prefix_parent']=str(parent)
    j=e.Journal(tmp_path/'new',c);key=ROWS[0]['id'];stage='extract_experience'
    payload={'sample':ROWS[0],'taxonomy':TAXONOMY,'max_rules_per_sample':2}
    folder=parent/('arm_'+stage)/e.p.digest(key)[:24]
    body=e.r.request_body(c['model'],j.messages(stage,payload),c['max_tokens'],backend='vllm')
    body.update(stream=False,return_token_ids=True,reasoning_effort=c['reasoning_effort']);body.pop('stream_options',None)
    raw=folder/'request_001.json';v=response()
    e.r.write_json(raw,{'request':body,'response':{'id':'x','model':c['model'],'text':json.dumps(v)}})
    e.r.write_json(folder/'result.json',{'request_file':raw.name,'request_sha256':e.r.sha256_file(raw),'value':v})
    monkeypatch.setattr(e.b.harmony,'decode_response',lambda raw,**kw:{'content':raw['text']})
    assert e.replay_draft_prefix(j,stage,key,payload)==v
    with pytest.raises(ValueError,match='Prefix request differs'):
        e.replay_draft_prefix(j,stage,key,dict(payload,max_rules_per_sample=1))


def test_repair_preserves_full_review_conversation():
    j=e.Journal(__import__('pathlib').Path('/tmp/unused'),config())
    payload={'original':{'sample':ROWS[0],'taxonomy':TAXONOMY,'max_rules_per_sample':2},'draft':response()}
    messages=j.messages('repair_experience',{'review_payload':payload,'review':response(),'validation_error':'Invalid error ID/type'})
    assert messages[:4]==j.messages('review_experience',payload)
    assert len(messages)==6 and 'Invalid error ID/type' in messages[-1]['content']
