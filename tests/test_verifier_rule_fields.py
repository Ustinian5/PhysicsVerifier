import copy
import json

import pytest

from rule_framework import final_answer_revision as final, flat_skill_rag as step, review_fields, verifier_english as en
from rule_framework import bounded_tree as tree
from scripts import project_verifier_first_reviews as replay
from scripts import run_corrected_extraction as extract
from tests.test_bounded_tree import Caller, bundle, engine, finish
from tests.test_experience_first_120b import ROWS, TAXONOMY, response


def normalize(data, *, legacy=False):
    return extract.old.extraction.normalize_120b_experience(
        data, sample=ROWS[0], taxonomy=TAXONOMY, legacy_symbolic=legacy)


def test_only_retired_key_can_vary_and_is_absent_from_both_review_turns():
    data = response()
    expected = review_fields.without_symbolic_hint(normalize(data, legacy=True))
    for hint in [None, {}, {'other': 'unrequested value'}]:
        data['experience_rules'][0]['symbolic_hint'] = hint
        if hint is None:
            del data['experience_rules'][0]['symbolic_hint']
        assert normalize(data) == expected
        with pytest.raises(ValueError, match='Unexpected symbolic program'):
            normalize(data, legacy=True)

    cfg = dict(rule_schema=review_fields.SCHEMA, prompt_profile=en.PROFILE,
               audit_system_prompt=extract.AUDIT_PROMPT_FILE.read_text())
    row = dict(ROWS[0], official_reference={'label': ['Official competition answer.']})
    payload = dict(sample=row, taxonomy=TAXONOMY, max_rules_per_sample=2)
    repair = dict(original=payload, draft=data, validation_error='Missing rule evidence')
    data['experience_rules'][0].update(auxiliary={'node_summary':'unused'}, rule_status='unrequested')
    for stage, request in [('extract_experience',payload), ('repair_experience',repair)]:
        messages = extract.request_messages(stage,request,cfg)
        assert 'symbolic_hint' not in json.dumps(messages)
        assert 'auxiliary' not in json.dumps(messages)
        if stage == 'repair_experience':
            draft = json.loads(messages[2]['content'])
            assert 'rule_status' not in draft['experience_rules'][0]
            assert draft['semantic_audit']['key_errors'][0]['rule_status'] == 'included'
        assert json.loads(messages[1]['content'])[en.EXAM_REFERENCE] == row['official_reference']['label']
    data['semantic_audit']['key_errors'][0]['line_start'] = 0
    with pytest.raises(ValueError, match='lines'):
        normalize(data)


@pytest.mark.parametrize('use_repair', [False, True])
def test_replay_prefers_first_valid_draft_and_binds_both_saved_attempts(tmp_path, use_repair):
    row = ROWS[0]
    cfg = dict(max_rules_per_sample=2)
    payload = dict(sample=row, taxonomy=TAXONOMY, max_rules_per_sample=2)
    first, second = response(), response()
    for draft in (first, second):
        draft['experience_rules'][0].update(auxiliary={'scene_cues':['rare']},rule_status='redundant')
    first['experience_rules'][0]['symbolic_hint'] = 'historical noise'
    second['experience_rules'][0]['title'] = 'Different valid rule'
    if use_repair:
        first['experience_rules'][0]['source_error_ids'] = []
    record = dict(id=row['id'], signature=replay.digest([payload,cfg]), evidence=[])
    selected = 'existing_repair' if use_repair else 'first'
    canonical = copy.deepcopy(first)
    canonical['experience_rules'][0]['symbolic_hint'] = response()['experience_rules'][0]['symbolic_hint']
    expected = normalize(second if use_repair else canonical, legacy=True)
    audit = dict(id=row['id'], canonical_branch=selected, canonical_status='error_found',
                 canonical_value_sha256=replay.digest(expected), attempts=[])
    for i, (stage,data) in enumerate(zip(('extract_experience','repair_experience'),(first,second))):
        raw = tmp_path/'raw'/stage/'example'/'response.json'
        decoded = tmp_path/'decoded'/stage/(replay.digest(row['id'])+'.json')
        replay.write(raw, data)
        call = dict(file=str(raw.relative_to(tmp_path)), sha256=replay.sha(raw))
        record['evidence'].append(call)
        replay.write(decoded, dict(evidence=call, value=data))
        audit['attempts'].append(dict(stage=stage, raw_sha256=call['sha256'], decoded_sha256=replay.sha(decoded),
                                      canonical_error='Missing rule evidence' if use_repair and i==0 else None))
    replay.write(tmp_path/'experiences'/(row['id']+'.json'), record)
    branch, value, error, evidence = replay.select(row,cfg,TAXONOMY,tmp_path,audit)
    assert branch == selected and value == review_fields.without_rule_metadata(review_fields.without_symbolic_hint(expected))
    distilled = extract.old.extraction._build_distilled_library([value],1,legacy_symbolic=False,include_auxiliary=False)
    assert 'auxiliary' not in distilled['rules'][0] and 'rule_status' not in value['experience_rules'][0]
    assert value['semantic_audit']['key_errors'][0]['rule_status'] == 'included'
    assert error is None and len(evidence) == (2 if use_repair else 1)
    replay.write(raw, {})
    with pytest.raises(ValueError, match='Archived response'):
        replay.select(row,cfg,TAXONOMY,tmp_path,audit)


def test_both_online_tasks_read_only_current_rule_fields(tmp_path):
    material = bundle(n=1)
    material['rule_schema'] = review_fields.SCHEMA
    material['taxonomy']['domains'][0]['summary'] = 'magnetic field'
    material['taxonomy']['domains'][0]['topics'][0]['summary'] = 'magnetic field'
    material['rules'][0].update(title='magnetic field', symbolic_hint={'old': 'field'})
    caller = Caller()

    def decisions(stage, key, payload):
        result = caller(stage,key,payload)
        for group in result.get('new_groups', []):
            group['summary'] = 'magnetic field'
        return result

    tree = finish(engine(tmp_path, material, decisions))['catalog']
    memory = step.TreeMemory(tree)
    assert set(next(iter(memory.corpus.leaves.values()))) == set(review_fields.LEAF_FIELDS)
    queries = dict(queries=['magnetic field'])
    for make_packet in (step.reference_packet, final.reference_packet):
        packet = make_packet(ROWS[0], 'rule_tree', queries, memory, lambda msgs:len(str(msgs))//4, english=True)
        assert len(packet['references']) == 1 and 'check_logic' in str(packet['references'])
        assert 'symbolic_hint' not in json.dumps(packet)
    archive = copy.deepcopy(tree)
    del archive['metadata']['rule_schema']
    with pytest.raises(ValueError, match='Leaf schema'):
        step.TreeMemory(archive)
    tree['domains'][0]['topics'][0]['rules'][0]['symbolic_hint'] = {}
    with pytest.raises(ValueError, match='Leaf schema'):
        step.TreeMemory(tree)


def test_topic_metadata_and_resolved_reasons_are_optional_only_at_their_own_stage():
    payload = dict(items=[{'id':'I1'}], topics=[{'key':k} for k in ('main','sec','other')])
    route = dict(item_id='I1',topic_key='main',reason='main relation')
    for extra in ({}, {'secondary_topics':'invalid'}, {'secondary_topics':['sec','main',{},'sec','unknown','other']}):
        record = dict(route, **extra)
        assert tree.validate('route_topics',payload,{'decisions':[record]},optional_fields=True) == []
        saved = tree.recorded_topic(record,{'main','sec','other'})
        if isinstance(extra.get('secondary_topics'),list):
            assert saved['secondary_topics'] == ['sec','other']
    with pytest.raises(tree.ContentFailure,match='Unknown topic ID'):
        tree.validate('route_topics',payload,{'decisions':[dict(route,topic_key='bad')]},optional_fields=True)
    groups = dict(items=[{'id':'I1'}],groups=[{'id':'G1'}])
    for extra in ({},{'reason':''},{'reason':'matches'}):
        val = dict(new_groups=[],assignments=[dict(item_id='I1',group_id='G1',**extra)])
        assert tree.validate('place_groups',groups,val,optional_fields=True) == []
    for gid, message in ((None,'Missing decision reason'),('G2','Unknown group ID')):
        with pytest.raises(tree.ContentFailure,match=message):
            tree.validate('place_groups',groups,dict(new_groups=[],assignments=[dict(item_id='I1',group_id=gid)]),
                          optional_fields=True)
    unresolved=dict(new_groups=[],assignments=[dict(item_id='I1',group_id=None,reason='unclear conditions')])
    assert not tree.validate('place_groups',groups,unresolved,optional_fields=True)
