import json
from pathlib import Path

import pytest

from rule_framework import corrected_verifier, final_answer_revision as final, flat_skill_rag as step
from rule_framework import organization_followup as skills, semantic_index_retrieval as semantic
from rule_framework import verifier_english as en
from scripts import audit_aligned_step_evaluation as step_audit, confirm_skill_retrieval as rewrite, pilot_retrieval_fusion as fusion
from scripts import print_verifier_prompts as review, run_corrected_extraction as build
from scripts import run_final_answer_revision as execution, run_organization_comparison as online
from tests.test_final_answer_revision import fake_runtime
from tests.test_organization_comparison import Tokenizer, catalog


ROW = dict(id='test', question='小车 in motion <|question|>', prediction='原答 <|prediction|>', context='',
           reference=['PRIVATE REFERENCE'], verdict='PRIVATE VERDICT')
QUERIES = {'queries':['constant force and acceleration Δv']}
EXAM = ['官方原文 <|label|>', '']
AUDIT = {'status':'error_found', 'summary':'Keep this review unchanged 原文'}
PAST = dict(id='case_train', kind='case', title='Different historical problem', source_ids=['train'],
            text=en.OLD_QUESTION + '旧题' + en.OLD_ANSWER + '旧答' + en.OLD_REVIEW + json.dumps(AUDIT, ensure_ascii=False))
PROFILE = {'prompt_profile':en.PROFILE}
KNOWLEDGE_ARMS = ('case_rag', 'flat_skills', 'rule_tree')


@pytest.mark.parametrize('module', [step, final])
@pytest.mark.parametrize('arm', KNOWLEDGE_ARMS)
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
    opened = module.continuation_messages(ROW, QUERIES, refs, official_reference=EXAM, **opts)
    assert opened[:-1] == check[:-1] and '<|' not in str(opened)
    opened_data = json.loads(opened[-1]['content'].split('\n',1)[0])
    assert opened_data.pop(en.EXAM_REFERENCE) == EXAM and opened_data == data
    assert opened[-1]['content'].split('\n',1)[1] == check[-1]['content'].split('\n',1)[1]


@pytest.mark.parametrize('module', [step, final])
def test_pure_receives_only_the_direct_check(module):
    messages = module.direct_messages(ROW)
    assert corrected_verifier.messages(ROW, 'pure', 'step' if module is step else 'final', english=True) == messages
    assert [m['role'] for m in messages] == ['system', 'user']
    assert json.loads(messages[1]['content']) == {k: ROW[k] for k in ('question','prediction','context')}
    assert all(word not in str(messages) for word in ('PREPARE','queries','Reference material','PRIVATE','<|'))
    opened = module.direct_messages(ROW, official_reference=EXAM)
    assert opened[0] == messages[0]
    assert json.loads(opened[1]['content']) == dict(json.loads(messages[1]['content']), **{en.EXAM_REFERENCE: EXAM})
    assert '<|' not in str(opened)
    with pytest.raises(ValueError, match='direct_messages'):
        module.continuation_messages(ROW, QUERIES, [], arm='pure', english=True)
    with pytest.raises(ValueError, match='direct_messages'):
        module.reference_packet(ROW, 'pure', QUERIES, None, lambda m: 1, english=True)


@pytest.mark.parametrize('phrase', ['力学', 'force 力学', 'force сила', 'Ω'])
def test_bilingual_and_non_english_queries_cannot_reach_retrieval(phrase):
    value = {'queries':[phrase]}
    with pytest.raises(ValueError):
        final.reference_packet(ROW,'flat_skills',value,None,lambda m: 1, english=True)
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
        for arm in KNOWLEDGE_ARMS:
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
    assert all(prompt in rendered for prompt in (en.STEP_DIRECT, en.FINAL_DIRECT, en.STEP_CHECK, en.FINAL_CHECK, en.FINAL_QUERY))
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
@pytest.mark.parametrize('valid_query', [True, False])
def test_final_direct_baseline_and_three_shared_checks(tmp_path, monkeypatch, with_exam, valid_query):
    prepare = final.query_messages(ROW, english=True)
    verdict = dict(verdict='incorrect', student_final_answer='3 m/s', reason='The units and number do not agree.')
    def respond(messages):
        if messages == prepare:
            return json.dumps(QUERIES if valid_query else {'queries':[]})
        return json.dumps(verdict)
    r, plan, tok, calls = fake_runtime(monkeypatch, content=respond)
    config = plan['configuration']
    config.update(PROFILE)
    if with_exam:
        config['check_exam_references']={'test':dict(label=EXAM,input_sha256=execution.vr.digest(final.public(ROW)))}
    plan['configuration_sha256'] = r.vr.digest(config)
    stores = dict(case_rag=step.FlatMemory([dict(PAST,title='Constant force')]),
                  flat_skills=step.FlatMemory([]),rule_tree=step.TreeMemory(catalog()))
    def run():
        r.run_question(tmp_path, ROW, 'formal', plan, tok, None, stores)
    run()
    direct = final.direct_messages(ROW, official_reference=EXAM if with_exam else None)
    sent = [c['messages'] for c in calls]
    assert sent[:2] == [direct, prepare] and len(sent) == (5 if valid_query else 2)
    assert 'PRIVATE' not in str(sent)
    key = r.vr.digest(ROW['id'])[:24]
    pure = tmp_path/'formal/pure'/key
    result = r.vr.read(pure/'result.json')
    check = r.vr.read(pure/'check/result.json')
    assert result['actual_response_count'] == 1 and result['value'] == verdict
    assert result['usage'] == check['usage'] and result['wall_seconds'] == check['wall_seconds']
    assert result['retrieval'] == step.no_retrieval() and not (pure/'reference_packet.json').exists()
    assert all(Path(f).is_relative_to(pure/'check') for f in result['files'])
    assert calls[0]['max_tokens'] == 16384 and calls[1]['max_tokens'] == step.QUERY_MAX_TOKENS
    assert r.vr.read(pure/'check/request_budget.json')['available_seconds'] == config['limits']['wall_seconds_per_sample']
    for call in calls[2:]:
        check = call['messages']
        assert check[:2] == prepare and json.loads(check[2]['content']) == QUERIES
        fields = json.loads(check[-1]['content'].split('\n',1)[0])
        assert fields.get(en.EXAM_REFERENCE) == (EXAM if with_exam else None)
        assert call['max_tokens'] == 16384 - 3
    for arm in KNOWLEDGE_ARMS:
        result = r.vr.read(tmp_path/'formal'/arm/key/'result.json')
        assert result['status'] == ('complete' if valid_query else 'content_failure')
        assert result['actual_response_count'] == (2 if valid_query else 1)
    run()
    monkeypatch.setattr(r, 'verify', lambda out: plan)
    monkeypatch.setattr(r, 'released', lambda out, plan: {})
    for name, data in [('inputs',[ROW]), ('labels_private',{ROW['id']:{'verdict':'incorrect'}}), ('bench_release',{})]:
        r.vr.atomic(tmp_path/(name+'.json'), data)
    r.score(tmp_path)
    assert r.vr.read(tmp_path/'metrics.json')['groups']['pure']['original_binary_view']['f1'] == 1
    assert len(calls) == len(sent)
    r.vr.atomic(pure/'reference_packet.json', {})
    with pytest.raises(ValueError, match='material evidence'):
        run()


@pytest.mark.parametrize('needs_repair', [False, True])
def test_step_direct_with_independent_usage_quote_repair_and_resume(tmp_path, monkeypatch, needs_repair):
    row = dict(ROW, question='What is the speed for 12 m in 3 s?', prediction='The speed is 3 m/s.')
    config = dict(PROFILE, api_base='http://localhost', limits=dict(online.runtime.LIMITS,
                  review_protocol=online.old.continuous.PROTOCOL))
    plan = dict(configuration=config, configuration_sha256=online.vr.digest(config))
    direct = step.direct_messages(row)
    def transport(client, **kw):
        messages = kw['messages']
        if messages == direct:
            value = {'diagnostics':[{'message':'Distance divided by elapsed time gives 4 m/s.',
                     'evidence':{'quote':'three' if needs_repair else '3 m/s'}, 'reference_ids':[]}]}
        elif messages[:2] == direct and len(messages) == 4:
            assert 'quote_patches' in messages[-1]['content']
            value = {'quote_patches':[{'index':0, 'quote':'3 m/s'}]}
        elif messages == step.query_messages(row, english=True):
            value = QUERIES
        else:
            assert messages[:2] == step.query_messages(row, english=True) and len(messages) == 4
            value = {'diagnostics':[]}
        body = {'messages':messages, 'max_tokens':kw['max_tokens']}
        online.vr.atomic(kw['request_path'], {'request':body})
        online.vr.atomic(kw['raw_path'], value)
        response = dict(content=json.dumps(value), model_ids=[online.r.EXPECTED_MODEL], finish_reasons=['stop'],
                        usage=dict(prompt_tokens=20,completion_tokens=10))
        online.vr.atomic(kw['response_path'], response)
        return response
    monkeypatch.setattr(online.runtime.transport, 'chat_response', transport)
    def run(arm, memory):
        return online.run_condition(tmp_path, row, 'formal', arm, memory, plan, Tokenizer(), None)
    result = run('pure', None)
    folder = online.path_for(tmp_path, 'formal', 'pure', row).parent
    assert not (tmp_path/'queries').exists() and not (folder/'reference_packet.json').exists()
    assert result['diagnostics'][0]['evidence']['quote'] == '3 m/s'
    trace = result['trace']
    turns = 1 + int(needs_repair)
    assert trace['model_requests'] == turns and trace['input_tokens'] == 20*turns
    assert trace['output_token_limit_reserved'] == 10*turns and trace['tool_events'] == []
    assert trace['read_rule_ids'] == [] and len(trace['response_evidence']) == turns
    assert online.vr.read(folder/'turn_01/request.json')['request']['max_tokens'] == 16384
    stores = dict(case_rag=step.FlatMemory([]), flat_skills=step.FlatMemory([]), rule_tree=step.TreeMemory(catalog()))
    prefixes = [run(arm, stores[arm])['trace']['response_evidence'][0] for arm in KNOWLEDGE_ARMS]
    assert prefixes[0] == prefixes[1] == prefixes[2] and prefixes[0]['shared_preparation']
    assert len(list(tmp_path.glob('**/request.json'))) == turns + 4
    audit = step_audit.Audit(tmp_path, Tokenizer())
    def read_response(path, messages, cap):
        assert online.vr.read(path)['request'] == dict(messages=messages, max_tokens=cap)
        record = online.vr.read(path.with_name('response.json'))
        return record['usage'], record
    monkeypatch.setattr(audit, 'response', read_response)
    assert audit.sample(row, 'pure', None, plan, None) == result
    monkeypatch.setattr(online.runtime.transport, 'chat_response', lambda *a, **k: pytest.fail('repeat request'))
    assert run('pure', None) == result
    online.vr.atomic(folder/'reference_packet.json', {})
    with pytest.raises(ValueError, match='Staged evidence'):
        run('pure', None)
