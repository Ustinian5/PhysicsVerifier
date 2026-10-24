import copy
import json

import pytest

from rule_framework import oss120b_construction as p, released_corpus
from scripts import build_120b_rules as b, rebuild_qwen30b_corpus as c
from tests.test_oss120b_construction import ROWS, TAXONOMY, TOPIC, mock_model


def test_json_escape_repair_preserves_literal_latex_and_legal_json_escapes():
    text = r'{"text":"中文 \(H^2\) and \(1/H\)","legal":"line\nquote\"slash\\tab\t","u":"\u4e2d"}'
    with pytest.raises(json.JSONDecodeError): b.parse_json_escapes(text)
    value, normalized, positions = b.parse_json_escapes(text, repair=True)
    assert value['text'] == r'中文 \(H^2\) and \(1/H\)'
    assert value['legal'] == 'line\nquote"slash\\tab\t' and value['u'] == '中'
    assert len(positions) == 4 and json.loads(normalized) == value
    assert b.parse_json_escapes(normalized, repair=True) == (value, normalized, [])


@pytest.mark.parametrize('text', ['{"x":', '{"x":"\\uZZZZ"}', '{"x":"\\(bad",}', '{"x":"unterminated}'])
def test_json_escape_repair_never_repairs_other_json_structure(text):
    with pytest.raises(json.JSONDecodeError): b.parse_json_escapes(text, repair=True)


def test_short_group_wire_and_restore_preserve_content():
    payload = {'topic_key': TOPIC, 'items': [{'id': 'err_long', 'message': 'physical content'}],
               'groups': [{'id': 'g_long', 'name': 'law', 'summary': 'conditions'}]}
    before = copy.deepcopy(payload)
    wire, mapping = b.short_group_payload(payload)
    assert wire['items'] == [{'id': 'I1', 'message': 'physical content'}]
    assert wire['groups'] == [{'id': 'G1', 'name': 'law', 'summary': 'conditions'}]
    value = {'new_groups': [], 'assignments': [{'item_id': 'I1', 'group_id': 'G1'}]}
    restored, saved_mapping, changes = b.restore_short_groups(value, payload)
    assert restored['assignments'] == [{'item_id': 'err_long', 'group_id': 'g_long'}]
    assert saved_mapping == mapping and not changes and payload == before


@pytest.mark.parametrize('fault', ['unknown_item', 'unknown_group', 'missing', 'conflict'])
def test_short_groups_reject_invented_or_conflicting_references(fault):
    payload = {'items': [{'id': 'err_long'}], 'groups': [{'id': 'g_long'}, {'id': 'g_other'}]}
    v = {'new_groups': [], 'assignments': [{'item_id': 'I1', 'group_id': 'G1'}]}
    if fault == 'unknown_item': v['assignments'][0]['item_id'] = 'I2'
    elif fault == 'unknown_group': v['assignments'][0]['group_id'] = 'G99'
    elif fault == 'missing': v['assignments'] = []
    else: v['assignments'].append({'item_id': 'I1', 'group_id': 'G2'})
    with pytest.raises(ValueError): b.restore_short_groups(v, payload)


def test_short_groups_preserve_new_definitions_and_exact_duplicate_normalization():
    payload = {'items': [{'id': 'rule_long'}], 'groups': []}
    a = {'item_id': 'I1', 'group_id': 'new1'}
    v = {'new_groups': [{'key': 'new1', 'name': 'mechanism', 'summary': 'conditions'}],
         'assignments': [a, dict(a)]}
    result, _, changes = b.restore_short_groups(v, payload)
    assert result['new_groups'] == v['new_groups']
    assert result['assignments'] == [{'item_id': 'rule_long', 'group_id': 'new1'}]
    assert changes[0]['removed_indices'] == [1]


def test_short_protocol_still_replays_successful_old_receipt(tmp_path, monkeypatch):
    config = {'model': 'model', 'max_request_input_bytes': 400000}
    payload = {'items': [{'id': 'err'}], 'groups': []}
    stage, key = 'group_errors', 'topic:0'
    value = {'new_groups': [{'key': 'new1', 'name': 'm', 'summary': 's'}],
             'assignments': [{'item_id': 'err', 'group_id': 'new1'}]}
    folder = tmp_path / ('arm_' + stage) / p.digest(key)[:24]; folder.mkdir(parents=True)
    b.r.write_json(folder / 'request_001.json', {'value': value})
    b.r.write_json(folder / 'result.json', {'fingerprint': p.digest([stage,key,payload,'model',config]),
        'value': value, 'request_file': 'request_001.json', 'request_sha256': b.r.sha256_file(folder/'request_001.json')})
    b.r.write_json(tmp_path/b.RECOVERIES['short_ids'][0], {'affected_jobs': []})
    monkeypatch.setattr(b, 'decode', lambda record,*args: record['value'])
    monkeypatch.setattr(b, 'Transport', lambda **kwargs: pytest.fail('Successful prefix must not call model'))
    j = b.Journal.__new__(b.Journal); j.output=tmp_path; j.config=config
    assert j(stage,key,payload) == value
    assert not (tmp_path/'arm_group_errors_shortids').exists()


def test_group_prefix_resolution_preserves_content_and_raw():
    payload = {'groups': [{'id': 'g_ec78155a845bc1f0d245'}], 'items': [{'id': 'err1'}]}
    value = {'new_groups': [], 'assignments': [{'item_id': 'err1', 'group_id': 'g_ec78155'}]}
    original = copy.deepcopy(value)
    normalized, changes = b.normalized_groups(value, payload)
    assert normalized['assignments'][0]['group_id'] == payload['groups'][0]['id']
    assert changes == [{'index': 0, 'original': 'g_ec78155', 'resolved': 'g_ec78155a845bc1f0d245'}]
    assert value == original
    assert b.normalized_groups(normalized, payload) == (normalized, [])


def test_group_duplicate_normalization_preserves_raw_and_prefix_receipts():
    payload = {'groups': [{'id': 'g_ec78155a845bc1f0d245'}], 'items': [{'id': 'err1'}]}
    a = {'item_id': 'err1', 'group_id': 'g_ec78155'}
    value = {'new_groups': [], 'assignments': [a, dict(a)]}
    raw = copy.deepcopy(value)
    result, changes = b.normalized_group_duplicates(value, payload)
    assert value == raw and len(result['assignments']) == 1
    assert changes[0]['removed_indices'] == [1]
    assert changes[0]['id_expansions'][0]['resolved'] == payload['groups'][0]['id']
    prefix_only = dict(value, assignments=[a])
    assert b.normalized_group_duplicates(prefix_only, payload)[1] == []


@pytest.mark.parametrize('fault', ['conflict', 'missing', 'invented'])
def test_group_duplicate_normalization_cannot_repair_membership(fault):
    payload = {'groups': [{'id': 'g_ec78155a845bc1f0d245'}, {'id': 'g_12345678901234567890'}],
               'items': [{'id': 'err1'}]}
    a = {'item_id': 'err1', 'group_id': payload['groups'][0]['id']}
    value = {'new_groups': [], 'assignments': [a, dict(a)]}
    if fault == 'conflict': value['assignments'][1]['group_id'] = payload['groups'][1]['id']
    elif fault == 'missing': payload['items'].append({'id': 'err2'})
    else: value['assignments'].append(dict(a, item_id='invented'))
    with pytest.raises(ValueError): b.normalized_group_duplicates(value, payload)


@pytest.mark.parametrize('fault', ['ambiguous', 'unknown', 'short', 'missing', 'duplicate', 'empty_group'])
def test_group_prefix_resolution_rejects_unsafe_changes(fault):
    payload = {'groups': [{'id': 'g_ec78155a845bc1f0d245'}], 'items': [{'id': 'err1'}]}
    value = {'new_groups': [], 'assignments': [{'item_id': 'err1', 'group_id': 'g_ec78155'}]}
    if fault == 'ambiguous': payload['groups'].append({'id': 'g_ec78155fffffffffffff'})
    elif fault == 'unknown': value['assignments'][0]['group_id'] = 'g_fffffff'
    elif fault == 'short': value['assignments'][0]['group_id'] = 'g_ec'
    elif fault == 'missing': value['assignments'] = []
    elif fault == 'duplicate': value['assignments'] *= 2
    else: value['new_groups'] = [{'key': 'new1', 'name': 'unused', 'summary': 'unused'}]
    with pytest.raises(ValueError): b.normalized_groups(value, payload)


def audits():
    return {row['id']: {'state': 'error_found', 'human_gold': False, 'uncertain': [],
        'errors': [{k: v for k, v in mock_model('audit', row['id'], {'sample': row})['errors'][0].items()
                    if k != 'topic_key'}]} for row in ROWS}


def imported_model(stage, key, payload):
    assert stage != 'audit', 'Released answers must not be re-audited'
    if stage == 'assign_topics':
        return {'assignments': [{'item_id': x['id'], 'topic_key': TOPIC} for x in payload['items']]}
    return mock_model(stage, key, payload)


def test_released_claims_are_reused_verbatim_without_another_audit():
    a = audits()
    built = p.build_catalog(ROWS, imported_model, source_audits=a)
    assert built['counts']['rules'] == 1 and built['counts']['errors'] == 2
    assert built['catalog']['metadata']['builder'] == p.CORPUS_VERSION
    for error in built['provenance']['errors']:
        original = a[error['sample_id']]['errors'][error['source_error_index']]
        assert all(error[k] == v for k, v in original.items())


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'unknown_topic', 'invented'])
def test_topic_assignment_cannot_change_source_membership(fault):
    def call(stage, key, payload):
        value = imported_model(stage, key, payload)
        if stage == 'assign_topics':
            if fault == 'missing': value['assignments'].pop()
            elif fault == 'duplicate': value['assignments'].append(value['assignments'][0])
            elif fault == 'unknown_topic': value['assignments'][0]['topic_key'] = 'other/topic'
            else: value['assignments'][0]['item_id'] = 'invented'
        return value
    with pytest.raises(ValueError): p.build_catalog(ROWS, call, source_audits=audits())


def test_import_rejects_invalid_quotes_and_pretended_human_gold():
    for field, bad in [('quote', 'not in answer'), ('human_gold', True)]:
        a = audits()
        if field == 'quote': a['a']['errors'][0][field] = bad
        else: a['a'][field] = bad
        with pytest.raises(ValueError): p.build_catalog(ROWS, imported_model, source_audits=a)


def test_isolation_also_screens_visible_audit_text():
    a = audits()
    q = 'This is a held-out benchmark problem about a spinning charged shell in a magnetic field and its boundary conditions.'
    a['a']['errors'][0]['correction'] += q
    with pytest.raises(ValueError, match='exceeds'):
        b.isolation_artifacts(ROWS, [{'id': 'heldout', 'question': q}], {}, 2, 17, a)


@pytest.fixture
def released(tmp_path, monkeypatch):
    config = {'generator': c.MODEL, 'auditor': b.r.EXPECTED_MODEL, 'reasoning_effort': 'medium',
        'samples_per_question': 1, 'human_gold': False, 'old_answers_used': False,
        'reference_answers_used': False, 'prompts_sha256': p.digest([c.SOLVER, c.AUDITOR, c.DOMAIN_SCREEN]),
        'temperature': .7, 'top_p': .8, 'top_k': 20, 'max_tokens': 16384, 'seed': 20261021}
    row = {'id': 'a', 'question': ROWS[0]['question']}
    prediction = ROWS[0]['prediction']
    raw_judgment = {'errors': audits()['a']['errors'], 'uncertain': []}
    gen_body = {'model': c.MODEL, 'messages': c.solver_messages(row), 'temperature': .7,
        'top_p': .8, 'top_k': 20, 'max_tokens': 16384, 'stream': False,
        'seed': int(p.digest([config['seed'], 'a'])[:8], 16) % (2**31)}
    audit_body = b.r.request_body(config['auditor'], c.audit_messages(row, prediction), 16384, backend='vllm')
    audit_body.update(stream=False, return_token_ids=True, reasoning_effort='medium')
    audit_body.pop('stream_options', None)
    artifacts = {
        'plan.json': {'configuration': config, 'configuration_sha256': p.digest(config)},
        'physics_inputs.json': [row],
        'wrong_answers.json': [dict(row, prediction=prediction, answer='', context='')],
        'raw/generate/a.json': {'status': 'received', 'request': gen_body,
            'response': {'model': c.MODEL, 'choices': [{'finish_reason': 'stop', 'message': {'content': prediction}}]}},
        'raw/audit/a.json': {'status': 'received', 'request': audit_body,
            'response': {'model': config['auditor'], 'content': json.dumps(raw_judgment)}}}
    for name, value in artifacts.items(): b.r.write_json(tmp_path/name, value)
    b.r.write_json(tmp_path/'generated/a.json', {'id': 'a', 'state': 'complete', 'model': c.MODEL,
        'prediction': prediction, 'signature': p.digest([row, config]),
        'evidence': {'file': 'raw/generate/a.json', 'sha256': b.r.sha256_file(tmp_path/'raw/generate/a.json')}})
    b.r.write_json(tmp_path/'audited/a.json', dict(c.classify_audit(raw_judgment, prediction),
        id='a', model=config['auditor'], signature=p.digest([row, prediction, config]), raw_judgment=raw_judgment,
        evidence={'file': 'raw/audit/a.json', 'sha256': b.r.sha256_file(tmp_path/'raw/audit/a.json')}))
    manifest = {'protocol': c.VERSION, 'human_gold': False, 'counts': {'state': 'completed', 'wrong_answers': 1},
        'configuration_sha256': p.digest(config), 'generator': c.MODEL, 'auditor': config['auditor'],
        'files': {str(f.relative_to(tmp_path)): b.r.sha256_file(f) for f in tmp_path.rglob('*.json')}}
    b.r.write_json(tmp_path/'release_manifest.json', manifest)
    monkeypatch.setattr(c.harmony, 'decode_response', lambda raw, **kwargs: {'content': raw['content']})
    return tmp_path


def test_corpus_handoff_checks_generated_answer_and_raw_audit(released):
    binding, accepted = released_corpus.load(released/'release_manifest.json', released/'wrong_answers.json', full=True)
    assert binding['accepted_claims'] == 1 and binding['candidate_questions'] == 1
    assert set(accepted['a']) == {'state', 'errors', 'uncertain', 'human_gold', 'evidence'}
    assert 'raw_judgment' not in accepted['a']
    (released/'raw/audit/a.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        released_corpus.load(released/'release_manifest.json', released/'wrong_answers.json')


def test_incomplete_release_and_substituted_source_are_refused(released):
    with pytest.raises(ValueError, match='substitution'):
        released_corpus.load(released/'release_manifest.json', released/'other.json')
    path = released/'release_manifest.json'; manifest = b.read(path)
    manifest['counts']['state'] = 'running'; b.r.write_json(path, manifest)
    with pytest.raises(ValueError, match='completed'):
        released_corpus.load(path, released/'wrong_answers.json')


@pytest.mark.parametrize('kind', ['file', 'parent', 'traversal'])
def test_release_paths_cannot_follow_symlinks_or_escape(released, kind):
    target = released/'raw/audit/a.json'
    if kind == 'file':
        content = target.read_bytes(); target.unlink()
        other = released/'outside.json'; other.write_bytes(content)
        target.symlink_to(other)
    elif kind == 'parent':
        parent = target.parent; moved = released/'moved'; parent.rename(moved); parent.symlink_to(moved, target_is_directory=True)
    else:
        path = released/'release_manifest.json'; manifest = b.read(path)
        manifest['files']['../outside.json'] = '0'*64; b.r.write_json(path, manifest)
    with pytest.raises(ValueError):
        released_corpus.load(released/'release_manifest.json', released/'wrong_answers.json', full=True)


def test_fresh_mode_does_not_accept_a_new_audit_call():
    config = {'corpus_release': 'release_manifest.json'}
    assert 'audit' not in b.prompts(config) and 'assign_topics' in b.prompts(config)
    with pytest.raises(ValueError): b.construction_messages('audit', {}, config)
    assert set(b.model_roles(config).values()) == {b.r.EXPECTED_MODEL}


def test_control_only_never_starts_a_historical_formal_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'verify', lambda output: {'configuration': {}})
    with pytest.raises(ValueError, match='requires the released-corpus'):
        b.run(tmp_path, control_only=True)


def test_exact_topic_duplicates_preserve_complete_build_and_raw_value():
    raw_values = []
    def call(stage, key, payload):
        value = imported_model(stage, key, payload)
        if stage == 'assign_topics':
            value['assignments'].append(copy.deepcopy(value['assignments'][0]))
            original = copy.deepcopy(value)
            normalized, removed = b.normalized_topics(value, payload)
            assert removed == [len(payload['items'])] and value == original
            raw_values.append(original)
            return normalized
        return value
    expected = p.build_catalog(ROWS, imported_model, source_audits=audits())
    assert p.build_catalog(ROWS, call, source_audits=audits()) == expected
    assert len(raw_values) == 1


@pytest.mark.parametrize('fault', ['missing', 'conflict', 'unknown_topic', 'invented', 'extra_field'])
def test_normalization_never_resolves_content_or_membership_failures(fault):
    payload = {'taxonomy': TAXONOMY, 'items': [{'id': 'a'}, {'id': 'b'}]}
    value = {'assignments': [{'item_id': 'a', 'topic_key': TOPIC}, {'item_id': 'b', 'topic_key': TOPIC}]}
    if fault == 'missing': value['assignments'].pop()
    elif fault == 'invented': value['assignments'][0]['item_id'] = 'c'
    elif fault == 'unknown_topic': value['assignments'][0]['topic_key'] = 'other/topic'
    elif fault == 'extra_field': value['assignments'][0]['correction'] = 'must not be dropped'
    else:
        alternate = copy.deepcopy(TAXONOMY['domains'][0]['topics'][0]); alternate['id'] = 'other'
        payload['taxonomy'] = copy.deepcopy(TAXONOMY)
        payload['taxonomy']['domains'][0]['topics'].append(alternate)
        value['assignments'].append({'item_id': 'a', 'topic_key': TAXONOMY['domains'][0]['id'] + '/other'})
    with pytest.raises(ValueError): b.normalized_topics(value, payload)


def test_cached_normalization_does_not_call_model_or_modify_raw_receipt(tmp_path, monkeypatch):
    config = {'corpus_release': 'release', 'model': b.r.EXPECTED_MODEL, 'max_requests': 20,
              'max_request_input_bytes': 400000, 'max_tokens': 16384}
    payload = {'taxonomy': TAXONOMY, 'items': [{'id': 'a'}]}
    value = {'assignments': [{'item_id': 'a', 'topic_key': TOPIC}] * 2}
    folder = tmp_path / 'arm_assign_topics' / p.digest('0')[:24]
    raw = folder / 'request_001.json'; receipt = folder / 'result.json'
    b.r.write_json(raw, {'test_value': value, 'request': {'max_tokens': 16384},
                        'response': {'usage': {'prompt_tokens': 100, 'completion_tokens': 30}}})
    b.r.write_json(receipt, {'fingerprint': p.digest(['assign_topics', '0', payload, config['model'], config]),
                            'value': value, 'request_file': raw.name, 'request_sha256': b.r.sha256_file(raw)})
    before = [raw.read_bytes(), receipt.read_bytes()]
    monkeypatch.setattr(b, 'decode', lambda record, stage, config: record['test_value'])
    def fail(*args, **kwargs): raise AssertionError('Cached recovery made a model request')
    monkeypatch.setattr(b, 'Transport', fail)
    journal = b.Journal(tmp_path, config)
    assert journal('assign_topics', '0', payload) == value  # strict prior behavior
    b.r.write_json(tmp_path / b.TOPIC_RECOVERY, {'policy': b.TOPIC_POLICY})
    assert journal('assign_topics', '0', payload)['assignments'] == value['assignments'][:1]
    assert journal('assign_topics', '0', payload)['assignments'] == value['assignments'][:1]
    assert [raw.read_bytes(), receipt.read_bytes()] == before
    assert b.read(folder / 'normalization.json')['removed_indices'] == [1]


def test_recovery_binds_only_adapter_change_and_original_prefix(tmp_path, monkeypatch):
    original = {'scripts/build_120b_rules.py': 'old', 'prompts.py': 'same'}
    current = dict(original, **{'scripts/build_120b_rules.py': 'new'})
    plan = {'configuration': {'source_hashes': original, 'corpus_release': 'release'}, 'configuration_sha256': 'sig'}
    b.r.write_json(tmp_path / 'plan.json', plan); b.r.write_json(tmp_path / 'raw.json', {'saved': True})
    monkeypatch.setattr(b, 'source_hashes', lambda: current)
    with pytest.raises(ValueError, match='source changed'): b.verify_sources(tmp_path, plan)
    update = {'policy': b.TOPIC_POLICY, 'configuration_sha256': 'sig',
              'plan_sha256': b.r.sha256_file(tmp_path / 'plan.json'), 'original_sources': original,
              'replacement_sources': current.copy(), 'prefix_files': {'raw.json': b.r.sha256_file(tmp_path / 'raw.json')},
              'stopped_status': {'error': 'Missing/duplicate imported topic member'}}
    b.verify_sources(tmp_path, plan, update)
    b.r.write_json(tmp_path / b.TOPIC_RECOVERY, update)
    current['prompts.py'] = 'changed'
    with pytest.raises(ValueError, match='binding changed'): b.verify_sources(tmp_path, plan)
    current['prompts.py'] = 'same'; (tmp_path / 'raw.json').write_text('{}')
    with pytest.raises(ValueError, match='prefix changed'): b.verify_sources(tmp_path, plan)


def test_duplicate_rejections_keep_all_reasons_and_leave_rules_unchanged():
    errors = [{'id': eid, 'question_key': eid, 'sample_id': eid, 'topic_key': TOPIC} for eid in 'abcd']
    value = mock_model('distill', '', {'errors': errors[:2]})
    value['rejected'] = [{'error_id': 'c', 'reason': 'not supported'},
                         {'error_id': 'd', 'reason': 'missing premise'},
                         {'error_id': 'd', 'reason': 'Duplicate entry removed; already listed.'}]
    before = copy.deepcopy(value)
    view, removed = b.normalized_rejections(value, {'errors': errors})
    assert value == before and view['rules'] == before['rules']
    assert removed == [2] and len(view['rejected']) == 2
    assert view['rejected'][1]['reason'] == 'missing premise\nDuplicate entry removed; already listed.'
    assert len(p.distilled_rules(view, errors, TOPIC)) == 1


def test_short_distillation_roundtrip_preserves_evidence_and_rules():
    errors = [{'id': eid, 'question_key': eid, 'sample_id': eid, 'topic_key': TOPIC,
               'quote': 'unchanged evidence'} for eid in ['long_a', 'long_b', 'long_c']]
    payload = {'errors': errors, 'samples': [{'id': 'sample', 'question': 'original question'}],
               'mechanism': {'name': 'same mechanism'}}
    wire, mapping = b.short_distill_payload(payload)
    assert [x['id'] for x in wire['errors']] == ['E1', 'E2', 'E3']
    assert wire['samples'] == payload['samples'] and wire['mechanism'] == payload['mechanism']
    value = mock_model('distill', '', {'errors': wire['errors'][:2]})
    value['rejected'] = [{'error_id': 'E3', 'reason': 'insufficient support'}]
    before = copy.deepcopy(value)
    result, got_mapping, removed = b.restore_short_distill(value, payload)
    assert value == before and mapping == got_mapping and removed == []
    assert [x['error_id'] for x in result['rules'][0]['support']] == ['long_a', 'long_b']
    assert result['rejected'][0]['error_id'] == 'long_c'
    assert all(result['rules'][0][k] == value['rules'][0][k] for k in value['rules'][0] if k != 'support')


@pytest.mark.parametrize('fault', ['invented', 'overlap', 'omitted', 'one_question'])
def test_short_distillation_cannot_relax_support_gates(fault):
    errors = [{'id': eid, 'question_key': eid, 'sample_id': eid, 'topic_key': TOPIC} for eid in 'ab']
    payload = {'errors': errors}
    wire, _ = b.short_distill_payload(payload)
    value = mock_model('distill', '', wire)
    if fault == 'invented': value['rules'][0]['support'][0]['error_id'] = 'E99'
    elif fault == 'overlap': value['rules'].append(copy.deepcopy(value['rules'][0]))
    elif fault == 'omitted': value['rules'] = []
    else: errors[1]['question_key'] = errors[0]['question_key']
    with pytest.raises(ValueError): b.restore_short_distill(value, payload)


def test_unknown_rejected_source_is_quarantined_without_changing_rules():
    errors = [{'id': eid, 'question_key': eid, 'sample_id': eid, 'topic_key': TOPIC} for eid in 'ab']
    v = mock_model('distill', '', {'errors': errors})
    v['rejected'] = [{'error_id': 'not_in_input', 'reason': 'model invented this rejected source'}]
    before = copy.deepcopy(v)
    result, changes = b.normalized_unknown_rejections(v, {'errors': errors})
    assert v == before and result['rules'] == v['rules'] and result['rejected'] == []
    assert changes['excluded_unknown_rejections'] == [{'index': 0, 'record': v['rejected'][0]}]


def test_explicit_valid_type_list_projects_metadata_only_and_keeps_raw():
    v = {'rules': [{'error_type': 'logic|calculation', 'title': 'same', 'support': [{'error_id': 'E1'}]}]}
    raw = copy.deepcopy(v)
    projected, changes = b.project_error_types(v)
    assert v == raw
    assert projected == {'rules': [dict(raw['rules'][0], error_type='logic')]}
    assert changes == [{'rule_index': 0, 'original': 'logic|calculation',
                        'labels': ['logic', 'calculation'], 'projected': 'logic'}]
    assert b.project_error_types(projected) == (projected, [])


@pytest.mark.parametrize('label', ['physics', 'logic|physics', 'logic|', '', None])
def test_type_projection_cannot_guess_unknown_labels(label):
    with pytest.raises(ValueError): b.project_error_types({'rules': [{'error_type': label}]})


@pytest.mark.parametrize('fault', ['missing', 'invented_support', 'overlap', 'one_question'])
def test_unknown_rejection_exclusion_never_fills_or_changes_real_support(fault):
    errors = [{'id': eid, 'question_key': eid, 'sample_id': eid, 'topic_key': TOPIC} for eid in 'ab']
    v = mock_model('distill', '', {'errors': errors})
    v['rejected'] = [{'error_id': 'fake', 'reason': 'not a source'}]
    if fault == 'missing': errors.append({'id': 'c', 'question_key': 'c', 'sample_id': 'c', 'topic_key': TOPIC})
    elif fault == 'invented_support': v['rules'][0]['support'][0]['error_id'] = 'fake'
    elif fault == 'overlap': v['rules'].append(copy.deepcopy(v['rules'][0]))
    else: errors[1]['question_key'] = errors[0]['question_key']
    with pytest.raises(ValueError): b.normalized_unknown_rejections(v, {'errors': errors})


@pytest.mark.parametrize('fault', ['missing', 'invented', 'support_reject_conflict', 'reused_support', 'one_question'])
def test_rejection_recovery_keeps_membership_and_support_gates(fault):
    errors = [{'id': eid, 'question_key': eid, 'sample_id': eid, 'topic_key': TOPIC} for eid in 'abc']
    value = mock_model('distill', '', {'errors': errors[:2]})
    value['rejected'] = [{'error_id': 'c', 'reason': 'unsupported'}] * 2
    if fault == 'missing': value['rejected'] = []
    elif fault == 'invented': value['rejected'] = [{'error_id': 'd', 'reason': 'unsupported'}] * 2
    elif fault == 'support_reject_conflict': value['rejected'].append({'error_id': 'a', 'reason': 'also reject'})
    elif fault == 'reused_support': value['rules'].append(copy.deepcopy(value['rules'][0]))
    else: errors[1]['question_key'] = errors[0]['question_key']
    with pytest.raises(ValueError): b.normalized_rejections(value, {'errors': errors})


def test_recovery_chain_preserves_parent_binding(tmp_path, monkeypatch):
    original = {'scripts/build_120b_rules.py': 'old', 'other.py': 'same'}
    first = dict(original, **{'scripts/build_120b_rules.py': 'first'})
    current = dict(original, **{'scripts/build_120b_rules.py': 'second'})
    plan = {'configuration': {'source_hashes': original, 'corpus_release': 'release'}, 'configuration_sha256': 'sig'}
    b.r.write_json(tmp_path / 'plan.json', plan)
    parent = {'policy': b.TOPIC_POLICY, 'configuration_sha256': 'sig',
        'plan_sha256': b.r.sha256_file(tmp_path/'plan.json'), 'original_sources': original,
        'replacement_sources': first, 'prefix_files': {},
        'stopped_status': {'error': b.RECOVERIES['assign_topics'][2]}}
    b.r.write_json(tmp_path / b.TOPIC_RECOVERY, parent)
    child = dict(parent, policy=b.REJECTION_POLICY, original_sources=first, replacement_sources=current,
                 stopped_status={'error': b.RECOVERIES['distill'][2]},
                 previous_recovery_sha256=b.r.sha256_file(tmp_path / b.TOPIC_RECOVERY))
    monkeypatch.setattr(b, 'source_hashes', lambda: current)
    b.verify_sources(tmp_path, plan, child)
    b.r.write_json(tmp_path / b.REJECTION_RECOVERY, child)
    b.verify_sources(tmp_path, plan)
    parent['actor'] = 'changed'
    b.r.write_json(tmp_path / b.TOPIC_RECOVERY, parent)
    with pytest.raises(ValueError, match='binding changed'): b.verify_sources(tmp_path, plan)


def complete_fixture(root):
    release=b.read(root/'release_manifest.json')
    rows=b.read(root/'wrong_answers.json')
    b.r.write_json(root/'answers.json',rows)
    release['counts']['generation']={'complete':len(rows),'incomplete':0}
    release['files']['answers.json']=b.r.sha256_file(root/'answers.json')
    b.r.write_json(root/'release_manifest.json',release)
    return release


def test_complete_answer_handoff_ignores_old_audit_labels(released):
    manifest=complete_fixture(released)
    # A new protocol must not need or leak old semantic judgments.
    (released/'audited/a.json').unlink()
    binding,rows=released_corpus.load(released/'release_manifest.json',released/'answers.json',complete_answers=True)
    assert binding['complete_answers']==1 and len(rows)==1
    assert binding['claim_policy']=='fresh_continuous_review_no_old_judgment_imported'
    assert set(rows[0])=={'id','question','prediction','answer','context'}


def test_complete_answer_handoff_rejects_shortened_cohort(released):
    manifest=complete_fixture(released)
    b.r.write_json(released/'answers.json',[])
    manifest['files']['answers.json']=b.r.sha256_file(released/'answers.json')
    manifest['counts']['generation']['complete']=0
    b.r.write_json(released/'release_manifest.json',manifest)
    with pytest.raises(ValueError,match='shortened'):
        released_corpus.load(released/'release_manifest.json',released/'answers.json',complete_answers=True)
