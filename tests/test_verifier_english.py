import json
from pathlib import Path

import pytest

from rule_framework import final_answer_revision as final, flat_skill_rag as step
from rule_framework import organization_followup as skills, semantic_index_retrieval as semantic
from rule_framework import verifier_english as en
from scripts import confirm_skill_retrieval as rewrite, pilot_retrieval_fusion as fusion
from scripts import print_verifier_prompts as review, run_corrected_extraction as build
from scripts import run_final_answer_revision as execution, run_organization_comparison as online
from tests.test_organization_comparison import catalog


ROW = dict(id='test', question='小车 in motion <|question|>', prediction='原答 <|prediction|>', context='',
           reference=['PRIVATE REFERENCE'], verdict='PRIVATE VERDICT')
QUERIES = {'queries':['constant force and acceleration Δv']}
EXAM = ['官方原文 <|label|>', '']
AUDIT = {'status':'error_found', 'summary':'Keep this review unchanged 原文'}
PAST = dict(id='case_train', kind='case', title='Different historical problem', source_ids=['train'],
            text=en.OLD_QUESTION + '旧题' + en.OLD_ANSWER + '旧答' + en.OLD_REVIEW + json.dumps(AUDIT, ensure_ascii=False))
PROFILE = {'prompt_profile':en.PROFILE}


@pytest.mark.parametrize('module', [step, final])
@pytest.mark.parametrize('arm', step.ARMS)
def test_current_english_prepare_and_check_have_only_their_permitted_inputs(module, arm):
    opts = dict(arm=arm, english=True)
    refs = [{k:PAST[k] for k in ('id','title','text')}] if arm == 'case_rag' else []
    prefix = module.query_messages(ROW, english=True)
    assert 'English only' in prefix[-1]['content'] and 'bilingual' not in str(prefix)
    fields = json.loads(prefix[1]['content'].split('\n',1)[0])
    assert fields == {k:ROW[k] for k in ('question','prediction','context')}
    check = module.continuation_messages(ROW, QUERIES, refs, **opts)
    assert check[:2] == prefix and json.loads(check[2]['content']) == QUERIES
    data = json.loads(check[-1]['content'].split('\n',1)[0])
    assert en.EXAM_REFERENCE not in data and 'PRIVATE' not in str(check) and '<|' not in str(check)
    if arm == 'case_rag':
        assert '旧题' in data['Reference material'][0]['text'] and en.REVIEW in data['Reference material'][0]['text']
    if module is final:
        opened = module.continuation_messages(ROW, QUERIES, refs, official_reference=EXAM, **opts)
        assert opened[:-1] == check[:-1] and '<|' not in str(opened)
        payload, suffix = opened[-1]['content'].split('\n',1)
        opened_data = json.loads(payload)
        assert opened_data.pop(en.EXAM_REFERENCE) == EXAM and opened_data == data
        assert suffix == check[-1]['content'].split('\n',1)[1]


@pytest.mark.parametrize('phrase', ['力学', 'force 力学', 'force сила', 'Ω'])
def test_bilingual_and_non_english_queries_cannot_reach_retrieval(phrase):
    value = {'queries':[phrase]}
    with pytest.raises(ValueError):
        final.reference_packet(ROW,'pure',value,None,lambda m: 1, english=True)
    with pytest.raises(ValueError):
        step.continuation_messages(ROW,value,[], english=True)
    with pytest.raises(ValueError):
        rewrite.query({'query':phrase}, english=True)
    with pytest.raises(ValueError):
        semantic.Index.__new__(semantic.Index).fused_selection([phrase], None, english=True)
    assert step.validate_queries(value) == value


def test_frozen_evidence_with_translated_labels_and_scientific_notation():
    native = en.case_view(PAST)
    assert native['text'] == en.case_text('旧题','旧答',AUDIT) and en.case_view(native) == native
    assert json.loads(en.review_text(native['text'])) == json.loads(en.review_text(PAST['text'])) == AUDIT
    assert semantic.item_card(PAST, 'case_rag') == semantic.item_card(native, 'case_rag')
    assert json.loads(en.review_text('old\n120B原始核查说明'+json.dumps(AUDIT))) == AUDIT
    assert skills.skill_messages(PAST, english=True)[1]['content'] == native['text']
    assert 'English' in skills.skill_messages(PAST, english=True)[0]['content']
    assert step.validate_queries({'queries':['Lorentz force Δv and 5 µm', 'Joule–Thomson coefficient']}, english=True)
    prefix = final.query_messages(ROW, english=True) + [dict(role='assistant',content=json.dumps(QUERIES))]
    assert fusion.rewrite_messages(ROW,prefix,english=True)[-1]['content'] == en.REWRITE
    with pytest.raises(ValueError):
        fusion.rewrite_messages(ROW,final.query_messages(ROW) + prefix[-1:], english=True)


def test_real_bm25_packets_count_the_sent_english_text_and_share_one_prepare():
    topic_store = step.TreeMemory(catalog())
    stores = dict(pure=None,case_rag=step.FlatMemory([dict(PAST,title='Constant force')]),
                  flat_skills=step.FlatMemory([dict(PAST,id='S1',title='Force',kind='skill', text='Check force.')]),
                  rule_tree=topic_store)
    count = lambda msgs: sum(len(m['content']) for m in msgs)
    for module in (final,step):
        for arm in step.ARMS:
            result = module.reference_packet(ROW,arm,QUERIES,stores[arm],count,english=True)
            refs = result['references']
            assert result['messages'] == module.continuation_messages(ROW,QUERIES,refs,arm=arm,english=True)
            empty = module.continuation_messages(ROW,QUERIES,[],arm=arm,english=True)
            assert result['retrieval']['reference_tokens'] == count(result['messages'])-count(empty)
            if arm == 'case_rag':
                assert len(refs)==1 and en.REVIEW in refs[0]['text']


def test_new_step_plan_rejects_archived_bilingual_receipt_before_check(tmp_path):
    plan = dict(configuration=PROFILE,configuration_sha256='fresh')
    folder = tmp_path/'queries'/online.vr.digest(ROW['id'])[:24]
    online.vr.atomic(folder/'result.json',dict(configuration_sha256='fresh',input_sha256=online.vr.digest(ROW),
                                              files={}, queries=QUERIES))
    online.vr.atomic(folder/'request.json',dict(request={'messages':step.query_messages(ROW)}))
    with pytest.raises(ValueError, match='prompt changed'):
        online.query(tmp_path,ROW,plan,None,None)
    online.vr.atomic(folder/'request.json',dict(request={'messages':step.query_messages(ROW,english=True)}))
    online.vr.atomic(folder/'response.json',dict(content=json.dumps({'queries':['force 力']})))
    with pytest.raises(ValueError, match='English retrieval phrase'):
        online.query(tmp_path,ROW,plan,None,None)


def test_official_reference_is_bound_to_the_current_answer():
    config = dict(PROFILE,check_exam_references={'test':dict(label=EXAM,input_sha256=execution.vr.digest(final.public(ROW)))})
    assert execution.check_exam_answer(ROW,PROFILE) is None and execution.check_exam_answer(ROW,config) == EXAM
    with pytest.raises(ValueError, match='different input'):
        execution.check_exam_answer(dict(ROW,prediction='A different response'),config)


def test_frozen_source_location_beats_another_variant_and_rejects_swaps(tmp_path):
    raw = tmp_path/'dataset.json'
    inv = tmp_path/'inventory.json'
    screen = tmp_path/'screen'
    loc = dict(rollout_id=1,sample_index=7,sample_position=1)
    other = dict(rollout_id=1,sample_index=4,sample_position=0)
    build.write(raw,[dict(rollout_id=1,samples=[dict(index=4,prompt='Q',label=['OTHER']),
                                               dict(index=7,prompt='Q',label=EXAM)])])
    build.write(inv,dict(questions=[dict(id='A', question='Q', first_source=loc)]))
    rows = [dict(id='A',question='Q')]
    variants = [dict(first_source=other,value=['OTHER']),dict(first_source=loc,value=EXAM)]
    def snapshot(values):
        index=dict(source_sha256=build.sha(raw),references=[dict(id='A',variants=values)])
        cfg=dict(source_inventory_sha256=build.sha(inv),source_refs_sha256=build.digest(index),
                 source_sha256=build.sha(raw))
        build.write(screen/'plan.json',dict(configuration=cfg,configuration_sha256=build.digest(cfg)))
        build.write(screen/'references_private.json',index)
        build.write(screen/'split.json',dict(source_plan_sha256=build.sha(screen/'plan.json'),build=rows))
    snapshot(variants)
    refs,_=build.build_references(rows,screen,inv,raw)
    assert refs == {'A':dict(label=EXAM,original_location=loc)}
    snapshot([dict(first_source=loc,value=['OTHER'])])
    with pytest.raises(ValueError,match='frozen source index'):
        build.build_references(rows,screen,inv,raw)
    with pytest.raises(ValueError,match='outside build split'):
        build.build_references([dict(id='B',question='Q')],screen,inv,raw)


def test_review_document_matches_actual_english_messages():
    rendered = review.render()
    assert (Path(__file__).resolve().parents[1]/'docs/Verifier英文提示词.md').read_text() == rendered
    assert all(ord(char)<128 for char in rendered) and 'bilingual' not in rendered.lower()
    assert en.STEP_CHECK in rendered and en.FINAL_CHECK in rendered and en.FINAL_QUERY in rendered
    assert en.REWRITE in rendered and en.EXAM_REFERENCE in rendered and en.TREE_REPAIR.split('{missing}')[0] in rendered


def test_annotation_profile_follows_saved_manifest(tmp_path):
    annotator = review.annotator
    inputs = tmp_path/'source.json'
    inputs.write_text(json.dumps([dict(id='safe',question='Distance?',prediction='3 m.',answer='["4 m"]')]))
    out = tmp_path/'annotations'
    _, first = annotator.prepare(inputs,out,'offline',expected_count=1,pilot_size=1,prompt_profile=en.PROFILE)
    _, resumed = annotator.prepare(inputs,out,'offline',expected_count=1,pilot_size=1)
    assert first == resumed and annotator.saved_system(out) == en.STEP_ANNOTATION
    assert first['prompts_sha256'] == annotator.digest([en.STEP_ANNOTATION,annotator.INSTRUCTIONS])
    assert annotator.annotation_system({}) == annotator.SYSTEM


@pytest.mark.parametrize('with_exam',[False,True])
def test_one_actual_final_executor_prepare_is_shared_by_four_checks(tmp_path,monkeypatch,with_exam):
    config = dict(PROFILE,limits=dict(wall_seconds_per_sample=1800,input_tokens_per_sample=600000,
        output_tokens_per_sample=16384,max_requests_per_sample=12,max_tokens_per_response=16384))
    if with_exam:
        config['check_exam_references']={'test':dict(label=EXAM,input_sha256=execution.vr.digest(final.public(ROW)))}
    plan=dict(configuration=config,configuration_sha256=execution.vr.digest(config))
    sent=[]
    def request(out,folder,messages,cap,validator,plan,tok,client,**kwargs):
        sent.append(messages)
        value = QUERIES if len(sent)==1 else dict(verdict='uncertain',student_final_answer='',reason='No numerical input.')
        rec=dict(status='complete',value=validator(value),usage=dict(prompt_tokens=100,completion_tokens=10,total_tokens=110),
                 reason=None,files={},wall_seconds=.1)
        execution.vr.atomic(folder/'result.json',rec)
        return rec
    class Tok:
        def apply_chat_template(self,messages,**kwargs):
            return [0]*sum(len(m['content']) for m in messages)
    arm_stores = dict(pure=None,case_rag=step.FlatMemory([dict(PAST,title='Constant force')]),
                      flat_skills=step.FlatMemory([]),rule_tree=step.TreeMemory(catalog()))
    monkeypatch.setattr(execution,'request_once',request)
    execution.run_question(tmp_path,ROW,'pilot',plan,Tok(),None,arm_stores)
    assert len(sent)==5 and sent[0]==final.query_messages(ROW,english=True) and 'PRIVATE' not in str(sent)
    for check in sent[1:]:
        assert check[:2] == sent[0] and check[2]['content'] == json.dumps(QUERIES,ensure_ascii=False)
        fields=json.loads(check[-1]['content'].split('\n',1)[0])
        assert fields.get(en.EXAM_REFERENCE) == (EXAM if with_exam else None)
