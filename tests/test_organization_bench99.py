import copy
import json
from pathlib import Path

import pytest

from rule_framework import organization_bench99 as p
from scripts import run_organization_bench99 as run
from tests.test_organization_comparison import catalog, ROW, QUERIES, Tokenizer


@pytest.mark.skipif(not (run.ROOT/p.INPUT).exists(),reason='Private released benchmark unavailable')
def test_99_release_and_legacy50_scoring_inputs_are_identical():
    rows, data, unknown = p.load_bench(run.ROOT)
    _, old50, old_unknown = run.base.evaluation.reference_data(run.ROOT)
    byid = {r['id']: r for r in data}
    assert len(rows) == len(data) == 99
    assert sum(len(x['physics_error_gt']) for x in data) == 402
    assert sum(map(len, unknown.values())) == 7
    assert all(p.flat.source_payload(byid[x['id']])==p.flat.source_payload(x) and
               byid[x['id']]['physics_error_gt']==x['physics_error_gt'] for x in old50)
    assert all(unknown[k] == v for k, v in old_unknown.items())
    assert all(set(x) == {'id','question','prediction','context'} for x in rows)


def test_skill_input_has_only_original_case_not_taxonomy_or_bench_labels():
    case = dict(text='原题\n历史错答\n120B机器核查', source_ids=['secret'], rule='SECRET_RULE', gold='SECRET_GT')
    messages = p.skill_messages(case)
    assert messages[1]['content'] == case['text']
    assert 'SECRET' not in json.dumps(messages)
    assert '如何利用验证结果' in messages[0]['content']
    assert '不要求套用统一分步格式' in messages[0]['content']
    assert '允许使用物理知识和公式' in messages[0]['content']


@pytest.mark.parametrize('value',[
    {'skills':[{'title':'x','text':'y','trigger':'z'}]},
    {'skills':[{'title':'x','text':''}]},
    {'skills':{'domain':[]}}, {'skills':[], 'taxonomy':[]}])
def test_structured_rule_projection_is_rejected(value):
    with pytest.raises(ValueError):p.validate_skills(value)


def test_flat_skill_collection_retains_failures_and_empty_without_count_quota():
    rows=[dict(ROW,id=str(i)) for i in range(3)]
    skills=[{'title':str(i),'text':'独立推导关系并检查极限'} for i in range(8)]
    outcomes=[dict(id='0',status='collected',value={'skills':skills}),
              dict(id='1',status='collected',value={'skills':[]}),
              dict(id='2',status='content_failure',reason='malformed')]
    result,counts=p.materialize_skills(rows,outcomes)
    assert len(result)==8 and counts['valid_empty_sources']==1 and counts['content_failure']==1
    assert counts['procedural_quality_verified'] is False
    assert all(x['source_ids']==['0'] for x in result)


def test_sibling_family_exclusion_does_not_depend_on_text_similarity():
    screen=p.FamilyScreen([dict(id='unrelated',question='A pendulum moves on a circular arc.')])
    kept,excluded=screen.partition([dict(ROW,id='trainphysics_3547')])
    assert not kept
    assert any(x['reason']=='confirmed_same_parent_problem' for x in excluded[0]['matches'])


def test_family_screen_keeps_numeric_and_embedded_guards_and_dependent_questions():
    question='A unique long physical apparatus uses a perfect rectangular enclosure to measure pressure at temperature 350 K.'
    screen=p.FamilyScreen([dict(id='heldout',question=question)])
    assert screen.matches(dict(ROW,question=question.replace('350','700')))
    assert any(x['reason']=='embedded_benchmark_question' for x in
               screen.matches(dict(ROW,prediction=question)))
    assert screen.matches(dict(ROW,question='Use the previous problem to determine the final speed.'))
    assert not screen.matches(dict(ROW,question='An electron accelerates in an electric field.'))


def test_catalog_exclusion_removes_cluster_text_not_just_leaf():
    c=catalog();t=c['domains'][0]['topics'][0]
    t['scenario_clusters'][0]['rule_ids']=['r0','r1']
    t['scenario_clusters'][0]['summary']='CONTAMINATED_SCENARIO'
    t['scenario_clusters']=t['scenario_clusters'][:1]+t['scenario_clusters'][2:]
    before=copy.deepcopy(c)
    provenance={f'r{i}':[{'sample_id':str(i)}] for i in range(4)}
    filtered,report=p.prune_catalog(c,provenance,{'0'})
    assert c==before
    assert report['kept_leaves']==3 and report['omitted_cluster_members']==[]
    assert len(report['reindexed_clusters'])==1
    assert 'CONTAMINATED_SCENARIO' not in json.dumps(filtered)
    assert {x['rule_id'] for x in filtered['domains'][0]['topics'][0]['rules']}=={'r1','r2','r3'}
    assert run.base.p.TreeMemory(filtered).memory.docs


def test_catalog_requires_every_leaf_source():
    with pytest.raises(ValueError,match='provenance'):p.prune_catalog(catalog(),{},set())


def test_new_checker_reuses_exact_shared_prefix_and_never_reads_frozen50(tmp_path,monkeypatch):
    calls=[]
    def transport(client,**kw):
        calls.append(kw)
        content=json.dumps(QUERIES if len(kw['messages'])==2 else {'diagnostics':[]})
        record={'model_ids':[run.base.r.EXPECTED_MODEL], 'finish_reasons':['stop'],
                'content':content,'usage':{'prompt_tokens':20,'completion_tokens':10}}
        run.vr.atomic(kw['request_path'],{'request':{'messages':kw['messages'],'max_tokens':kw['max_tokens']}})
        run.vr.atomic(kw['raw_path'],{'content':content});run.vr.atomic(kw['response_path'],record)
        return record
    monkeypatch.setattr(run.base.runtime.transport,'chat_response',transport)
    monkeypatch.setattr(run.base.runtime,'frozen_rows',lambda:pytest.fail('Must not load frozen50'))
    plan={'configuration_sha256':'new-99','configuration':{'api_base':'http://127.0.0.1:18083/v1',
          'limits':dict(run.base.runtime.LIMITS,review_protocol=run.base.old.continuous.PROTOCOL)}}
    first=run.run_condition(tmp_path,ROW,'formal','pure',None,plan,Tokenizer(),None)
    again=run.run_condition(tmp_path,ROW,'formal','pure',None,plan,Tokenizer(),None)
    assert first==again and len(calls)==2 and first['diagnostics']==[]
    empty=run.base.p.FlatMemory([])
    other=run.run_condition(tmp_path,ROW,'formal','flat_skills',empty,plan,Tokenizer(),None)
    assert len(calls)==3 and first['trace']['response_evidence'][0]==other['trace']['response_evidence'][0]


def test_skill_resume_preserves_empty_and_failed_items(tmp_path,monkeypatch):
    rows=[dict(ROW,id='a'),dict(ROW,id='b'),dict(ROW,id='c')]
    cases=[dict(id='case'+r['id'],kind='case',title='q',text='original '+r['id'],source_ids=[r['id']]) for r in rows]
    for name,value in [('sources',rows),('cases',cases),('bench_questions',[dict(id='h',question='unrelated')])]:
        run.vr.atomic(tmp_path/(name+'.json'),value)
    calls=[]
    class Journal:
        def __init__(self,folder,*args):
            path=folder/'request_index.json';self.index=run.vr.unseal(path) if path.exists() else {'files':{}}
        def __call__(self,stage,key,payload):
            calls.append(key)
            if key=='b':raise run.base.old.tree.ContentFailure('length')
            return {'skills':[] if key=='a' else [{'title':'检查映射','text':'提取符号，核对其定义，再独立代入原式。'}]}
    monkeypatch.setattr(run,'ProcedureJournal',Journal)
    plan={'configuration':{},'configuration_sha256':'new'}
    with pytest.raises(run.PilotReviewRequired):run.build(tmp_path,plan,Tokenizer())
    with pytest.raises(run.PilotReviewRequired):run.build(tmp_path,plan,Tokenizer())
    assert calls==['a','b','c']
    entries=[]
    for row,verdict in zip(rows,['valid_empty','unassessable_output_failure','acceptable']):
        entries.append(dict(id=row['id'],outcome_sha256=run.vr.sha(run.base.old.outcome_path(tmp_path/'skills',row)),
                            method_distinction=verdict,notes='Offline test review fixture, not a real experiment.'))
    run.vr.atomic(tmp_path/'skills/pilot_review.json',dict(configuration_sha256='new',actor='Codex assistant',
                 human_gold=False,decision='continue',samples=entries))
    first=run.build(tmp_path,plan,Tokenizer());again=run.build(tmp_path,plan,Tokenizer())
    assert calls==['a','b','c'] and first.docs==again.docs and len(first.docs)==1
    counts=run.vr.read(tmp_path/'skills/summary.json')
    assert counts['content_failure']==1 and counts['valid_empty_sources']==1
    review_path=tmp_path/'skills/pilot_review.json'
    review=run.vr.read(review_path);review['samples'][0]['notes']='Changed after publication'
    run.vr.atomic(review_path,review)
    with pytest.raises(ValueError):run.build(tmp_path,plan,Tokenizer())


def test_scorer_requires_all99_before_main_score_and_counts_failed_delivery(tmp_path,monkeypatch):
    rows=[dict(ROW,id=str(i)) for i in range(99)]
    dataset=[dict(r,physics_error_gt=[]) for r in rows]
    monkeypatch.setattr(p,'load_bench',lambda root:(rows,dataset,{r['id']:[] for r in rows}))
    monkeypatch.setattr(run.base.runtime,'audit_result',lambda value:None)
    monkeypatch.setattr(run.recovery,'retained_failure',lambda path,**kw:run.vr.read(path))
    monkeypatch.setattr(run.base,'complete_content_failure',lambda x:True)
    monkeypatch.setattr(run.base.runtime,'healthy_result',lambda x:x['status']=='completed')
    run.vr.atomic(tmp_path/'inputs.json',rows)
    for row in rows[:98]:
        path=run.base.path_for(tmp_path,'formal','pure',row)
        run.vr.atomic(path,dict(id=row['id'],configuration_sha256='99',status='completed',diagnostics=[],
                                protocol_audit={'passed':True},trace={'response_evidence':[]}))
        run.vr.atomic(path.parent/'reference_packet.json',{'packet':{'retrieval':{'provided_complete_ids':[], 'reference_tokens':0}}})
    plan={'configuration_sha256':'99'}
    partial=run.score(tmp_path,plan)
    assert 'pure' not in partial['scores_all_99'] and 'scores_all_50' not in partial
    row=rows[-1];path=run.base.path_for(tmp_path,'formal','pure',row)
    run.vr.atomic(path,dict(id=row['id'],configuration_sha256='99',status='failed',diagnostics=[],trace={'response_evidence':[]}))
    run.vr.atomic(path.parent/'reference_packet.json',{'packet':{'retrieval':{'provided_complete_ids':['material'], 'reference_tokens':40}}})
    complete=run.score(tmp_path,plan)
    assert complete['scores_all_99']['pure']['summary']['dataset_size']==99
    assert complete['scores_all_99']['pure']['summary']['failed_questions']==1
    assert complete['usage']['pure']['questions_with_full_references']==1


def test_bound_artifact_rejects_changed_or_ambiguous_reuse(tmp_path):
    f=tmp_path/'experience/cases.json';run.vr.atomic(f,[])
    release={'files':{'/server/experience/cases.json':run.vr.sha(f)}}
    run.bound_artifact(release,f)
    release['files']['/other/experience/cases.json']=run.vr.sha(f)
    with pytest.raises(ValueError):run.bound_artifact(release,f)
