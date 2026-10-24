import json
from pathlib import Path
import threading

import pytest

from scripts import rebuild_qwen30b_corpus as m


def test_solver_cannot_see_old_labels_or_answers():
    row = {'id':'q','question':'A body has mass 2 kg.', 'prediction':'OLD ANSWER',
           'solution':'SECRET SOLUTION','criteria_json':'SECRET RUBRIC','messages':'SECRET HINT'}
    wire = json.dumps(m.solver_messages(row))
    assert 'A body has mass 2 kg.' in wire
    assert all(x not in wire for x in ('OLD ANSWER','SECRET'))


def test_isolation_before_sampling_and_number_variant_dedup():
    bench=[{'id':'b','question':'A body of mass 2 kg experiences a force of 6 N. What is its acceleration?'}]
    pool=[{'id':'a','question':bench[0]['question']},
          {'id':'b','question':bench[0]['question'].replace('2 kg','7 kg')},
          {'id':'c','question':'Explain the uncertainty principle in quantum mechanics.'},
          {'id':'d','question':'Explain the uncertainty principle in quantum mechanics.'}]
    selected, report=m.select_pool(pool,bench,3000,42)
    assert len(selected)==1
    assert 'quantum' in selected[0]['question']
    assert report['prefilter']['duplicate_numeric_template']==2
    assert len(report['excluded'])==1


def test_audit_control_markers_are_lossless_data():
    text='x <|im_end|> y'
    messages=m.audit_messages({'question':'Q'},text)
    assert '<|' not in messages[1]['content']
    assert json.loads(messages[1]['content'])['student_answer']==text


def test_invalid_evidence_is_not_a_wrong_answer_or_correct_answer():
    value={'errors':[{'quote':'invented','message':'wrong','correction':'right','error_type':'logic'}], 'uncertain':[]}
    result=m.classify_audit(value,'actual student text')
    assert result['state']=='invalid_audit' and not result['errors']
    assert not result['human_gold']


def test_empty_judgment_differs_from_unknown():
    assert m.classify_audit({'errors':[],'uncertain':[]},'answer')['state']=='no_error_found'
    assert m.classify_audit({'errors':[],'uncertain':[{'scope':'figure','reason':'missing'}]},'answer')['state']=='uncertain'


def test_domain_screen_requires_exactly_one_decision_per_frozen_question():
    rows=[{'id':'a','question':'physics'},{'id':'b','question':'math'}]
    value={'items':[{'id':'a','domain':'physics','reason':'mass and force'},
                    {'id':'b','domain':'non_physics','reason':'number theory'}]}
    assert m.validate_domains(value,rows)==value['items']
    for items in [value['items'][:1], [value['items'][0]]*2,
                  [dict(value['items'][0],id='invented'),value['items'][1]]]:
        with pytest.raises(ValueError): m.validate_domains({'items':items},rows)


def test_received_response_is_replayed_without_network(tmp_path,monkeypatch):
    c={'technical_attempts':3}
    body={'model':m.MODEL,'messages':[]}
    folder=tmp_path/'raw'/'generate'/m.p.digest('q')[:24]
    folder.mkdir(parents=True)
    response={'model':m.MODEL,'choices':[]}
    m.r.write_json(folder/'request_001.json',{'fingerprint':m.p.digest(['q',body,c]),
                 'status':'received','response':response})
    monkeypatch.setattr(m.httpx,'Client',lambda **_:pytest.fail('cached completion must not call network'))
    got,evidence=m.Journal(tmp_path,c,threading.Event()).call('generate','q',body)
    assert got==response and evidence['sha256']
    with pytest.raises(ValueError,match='input changed'):
        m.Journal(tmp_path,c,threading.Event()).call('generate','q',dict(body,temperature=1))


def test_model_mismatch_stops_even_when_response_is_cached(tmp_path):
    c={'technical_attempts':3}; body={'model':m.MODEL}
    folder=tmp_path/'raw'/'generate'/m.p.digest('q')[:24]; folder.mkdir(parents=True)
    m.r.write_json(folder/'request_001.json',{'fingerprint':m.p.digest(['q',body,c]),
                'status':'received','response':{'model':'different-model'}})
    with pytest.raises(m.IdentityError):
        m.Journal(tmp_path,c,threading.Event()).call('generate','q',body)


def test_auth_failure_cannot_be_retried(tmp_path,monkeypatch):
    c={'technical_attempts':3}; body={'model':m.MODEL}
    folder=tmp_path/'raw'/'generate'/m.p.digest('q')[:24]; folder.mkdir(parents=True)
    m.r.write_json(folder/'request_001.json',{'fingerprint':m.p.digest(['q',body,c]),
                'status':'failed','retryable':False,'http_status':401})
    monkeypatch.setattr(m.httpx,'Client',lambda **_:pytest.fail('auth failure must stop'))
    with pytest.raises(ValueError,match='non-transient'):
        m.Journal(tmp_path,c,threading.Event()).call('generate','q',body)


def test_truncation_saved_but_not_rerolled(tmp_path,monkeypatch):
    c={'generator':m.MODEL,'temperature':0.7,'top_p':0.8,'top_k':20,'max_tokens':16,'seed':42}
    journal=m.Journal(tmp_path,c,threading.Event())
    raw={'model':m.MODEL,'choices':[{'finish_reason':'length','message':{'content':'unfinished'}}],
         'usage':{'completion_tokens':16}}
    path=tmp_path/'evidence.json'; m.r.write_json(path,raw)
    monkeypatch.setattr(journal,'call',lambda *a:(raw,{'file':path.name,'sha256':m.r.sha256_file(path)}))
    row={'id':'q','question':'question'}
    assert m.generate(journal,row)['state']=='incomplete'
    monkeypatch.setattr(journal,'call',lambda *a:pytest.fail('truncation is not a transport retry'))
    assert m.generate(journal,row)['state']=='incomplete'
