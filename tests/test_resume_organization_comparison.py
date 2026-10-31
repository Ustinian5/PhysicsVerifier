import pytest
import sys
from types import SimpleNamespace
from scripts import resume_organization_comparison as x


def fixture(tmp_path,monkeypatch):
    request={'model':x.r.EXPECTED_MODEL,'max_tokens':100}
    usage={'prompt_tokens':20,'completion_tokens':10}
    raw={'model':x.r.EXPECTED_MODEL,'choices':[{'finish_reason':'stop'}],'usage':usage}
    for name,val in [('request.json',{'request':request,'request_sha256':x.r.sha256_value(request)}),('raw.json',raw)]:
        x.vr.atomic(tmp_path/name,val)
    x.vr.atomic(tmp_path/'response.json',{'http_status':200,'error':'JSONDecodeError: Extra data',
        'raw_sse_sha256':x.vr.sha(tmp_path/'raw.json'),'model_ids':[x.r.EXPECTED_MODEL],'usage':usage})
    x.vr.atomic(tmp_path/'state.json',{'pending':{'request_path':str(tmp_path/'request.json'),
        'raw_path':str(tmp_path/'raw.json'),'response_path':str(tmp_path/'response.json'),'max_tokens':100}})
    path=tmp_path/'result.json'
    x.vr.atomic(path,{'id':'x','status':'failed','diagnostics':[],'failure_reason':'JSONDecodeError: Extra data',
                      'trace':{'response_evidence':[]}})
    monkeypatch.setattr(x.base.runtime,'audit_result',lambda value:None)
    monkeypatch.setattr(x,'decode_final',lambda raw,request:{'content':'{"diagnostics":[]}}',
        'model_ids':[x.r.EXPECTED_MODEL],'finish_reasons':['stop'],'usage':usage})
    return path


def test_failed_raw_preserved_and_usage_counted_once(tmp_path,monkeypatch):
    p=fixture(tmp_path,monkeypatch);before=x.vr.sha(p)
    a=x.account(p);b=x.account(p)
    assert a==b and a['usage']['completion_tokens']==10 and x.vr.sha(p)==before
    (tmp_path/'raw.json').write_text('{}')
    with pytest.raises(ValueError,match='evidence changed'):x.account(p)


@pytest.mark.parametrize('field,value',[('model','wrong'),('finish','length'),('error','HTTPError')])
def test_unknown_failure_is_never_skipped(tmp_path,monkeypatch,field,value):
    p=fixture(tmp_path,monkeypatch)
    raw=x.vr.read(tmp_path/'raw.json');rec=x.vr.read(tmp_path/'response.json')
    if field=='model':raw['model']=value
    elif field=='finish':raw['choices'][0]['finish_reason']=value
    else:rec['error']=value
    x.vr.atomic(tmp_path/'raw.json',raw);rec['raw_sse_sha256']=x.vr.sha(tmp_path/'raw.json')
    x.vr.atomic(tmp_path/'response.json',rec)
    with pytest.raises(ValueError,match='not eligible'):x.account(p)


def test_valid_content_cannot_be_discarded(tmp_path,monkeypatch):
    p=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(x,'decode_final',lambda *a:{'content':'{"diagnostics":[]}',
        'model_ids':[x.r.EXPECTED_MODEL],'finish_reasons':['stop']})
    with pytest.raises(ValueError,match='separate review'):x.account(p)


def test_release_check_preserves_catalog_sha_and_rejects_changed_plan(tmp_path):
    library=tmp_path/'library';library.mkdir()
    c={'model':{'model':x.r.EXPECTED_MODEL,'reasoning_effort':'medium'}}
    digest=x.vr.digest(c)
    x.vr.atomic(library/'plan.json',{'configuration':c,'configuration_sha256':digest})
    x.vr.atomic(tmp_path/'plan.json',{'configuration':{'rules':{'path':str(library),'configuration_sha256':digest}}})
    with x.released_artifact_check(tmp_path):
        assert x.base.old.navigation.verify(library)['configuration']==c
        with pytest.raises(ValueError,match='explicit full audit'):x.base.old.navigation.verify(library,full=True)
        c['model']['model']='wrong'
        x.vr.atomic(library/'plan.json',{'configuration':c,'configuration_sha256':digest})
        with pytest.raises(ValueError,match='plan changed'):x.base.old.navigation.verify(library)


def missing_fixture(tmp_path,monkeypatch,channels=('analysis',)):
    path=fixture(tmp_path,monkeypatch)
    raw=x.vr.read(tmp_path/'raw.json')
    raw['choices'][0].update(token_ids=[1]*9+[200002],stop_reason=None)
    x.vr.atomic(tmp_path/'raw.json',raw)
    record=x.vr.read(tmp_path/'response.json')
    record.update(error='ValueError: Missing final message',raw_sse_sha256=x.vr.sha(tmp_path/'raw.json'))
    x.vr.atomic(tmp_path/'response.json',record)
    value=x.vr.read(path);value['failure_reason']=record['error'];x.vr.atomic(path,value)
    parsed=SimpleNamespace(messages=[SimpleNamespace(channel=c,recipient=None,
        content=[SimpleNamespace(text='not a published answer')]) for c in channels],current_content='')
    harmony=SimpleNamespace(parse_output_into_messages=lambda ids:parsed)
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'vllm.entrypoints',SimpleNamespace(harmony_utils=harmony))
    monkeypatch.setitem(sys.modules,'vllm.entrypoints.harmony_utils',harmony)
    return path,parsed


def test_missing_final_preserves_failure_and_accounts_once(tmp_path,monkeypatch):
    path,_=missing_fixture(tmp_path,monkeypatch);before=x.vr.sha(path)
    a=x.account(path)
    assert a==x.account(path) and x.vr.sha(path)==before
    assert a['failure_reason']=='ValueError: Missing final message'
    assert a['usage']=={'prompt_tokens':20,'completion_tokens':10}
    assert a['new_model_calls']==0 and not a['content_modified']


@pytest.mark.parametrize('channels',[('final',),('final','final'),('commentary',)])
def test_non_analysis_output_is_not_classified_missing(tmp_path,monkeypatch,channels):
    path,_=missing_fixture(tmp_path,monkeypatch,channels)
    with pytest.raises(ValueError):x.account(path)


@pytest.mark.parametrize('mutation',['usage','termination','pending','recipient'])
def test_incomplete_or_unknown_native_envelopes_stop(tmp_path,monkeypatch,mutation):
    path,parsed=missing_fixture(tmp_path,monkeypatch)
    raw=x.vr.read(tmp_path/'raw.json')
    if mutation=='usage':raw['choices'][0]['token_ids'].pop(0)
    elif mutation=='termination':raw['choices'][0]['token_ids'][-1]=200007
    elif mutation=='pending':parsed.current_content='unfinished'
    else:parsed.messages[0].recipient='functions.unknown'
    x.vr.atomic(tmp_path/'raw.json',raw)
    rec=x.vr.read(tmp_path/'response.json');rec['raw_sse_sha256']=x.vr.sha(tmp_path/'raw.json')
    x.vr.atomic(tmp_path/'response.json',rec)
    with pytest.raises(ValueError):x.account(path)


def test_missing_final_requires_amendment_and_stays_failed(tmp_path,monkeypatch):
    path,_=missing_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='explicit amendment'):
        x.retained_failure(path,output_amended=False)
    value=x.retained_failure(path,output_amended=True)
    assert value['status']=='failed' and value['diagnostics']==[]


def test_successful_protocol_violation_cannot_be_skipped(tmp_path,monkeypatch):
    path=fixture(tmp_path,monkeypatch);value=x.vr.read(path)
    value.update(status='completed',protocol_audit={'passed':False})
    x.vr.atomic(path,value)
    with pytest.raises(ValueError,match='protocol violation'):
        x.retained_failure(path,output_amended=True)


def test_control_summary_does_not_claim_failed_control_passed(tmp_path,monkeypatch):
    folder=tmp_path/'controls/case_rag/samples/C0p/test';folder.mkdir(parents=True)
    path,_=missing_fixture(folder,monkeypatch)
    before=x.vr.sha(path);summary=x.control_summary(tmp_path)
    assert summary['case_rag']['failed']==1 and summary['case_rag']['passed']==0
    assert not summary['case_rag']['all_passed']
    assert summary['case_rag']['distinct_actual_requests']==1
    assert x.vr.sha(path)==before
