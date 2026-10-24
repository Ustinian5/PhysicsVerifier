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
    assert j.supported_stages()=={'taxonomy','extract_experience','cluster_rules'}
