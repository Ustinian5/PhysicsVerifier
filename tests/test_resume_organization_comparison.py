import pytest
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
