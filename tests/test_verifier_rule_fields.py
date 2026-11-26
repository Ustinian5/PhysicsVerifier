import copy
import json

import pytest

from rule_framework import final_answer_revision as final, flat_skill_rag as step, review_fields, verifier_english as en
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
    for stage, request in [('extract_experience',payload), ('repair_experience',repair)]:
        messages = extract.request_messages(stage,request,cfg)
        assert 'symbolic_hint' not in json.dumps(messages)
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
    first['experience_rules'][0]['symbolic_hint'] = 'historical noise'
    second['experience_rules'][0]['title'] = 'Different valid rule'
    if use_repair:
        first['experience_rules'][0]['source_error_ids'] = []
    record = dict(id=row['id'], signature=replay.digest([payload,cfg]), evidence=[])
    selected = 'existing_repair' if use_repair else 'first'
    expected = normalize(second if use_repair else response(), legacy=True)
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
    assert branch == selected and value == review_fields.without_symbolic_hint(expected)
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
