import copy
import json
import os
from pathlib import Path

import pytest

from rule_framework import bounded_tree as tree, verified_resume as vr
from scripts import reorganize_qwen30b_expansion as run
from tests.test_experience_first_120b import TAXONOMY, TOPIC


def bundle(n=8):
    return {'taxonomy': copy.deepcopy(TAXONOMY), 'rules': [
        {'rule_id': f'r{i:04d}', 'title': '磁场换算', 'trigger': '用毫高斯表示磁场',
         'check_logic': '1mG=10^-7T', 'error_type': 'units', 'domain': 'OLD_WRONG_DOMAIN',
         'topic': 'OLD_WRONG_TOPIC', 'sample_ids': [f's{i}'], 'count': 1}
        for i in range(n)]}


class Caller:
    def __init__(self):
        self.calls = []

    def __call__(self, stage, key, payload):
        self.calls.append((stage, key, copy.deepcopy(payload)))
        assert tree.input_size(stage, payload) <= tree.POLICY['max_input_bytes']
        if stage == 'route_topics':
            assert 'OLD_WRONG' not in json.dumps(payload)
            return {'decisions': [{'item_id': x['id'], 'topic_key': TOPIC, 'secondary_topics': [],
                                  'reason': '磁场关系'} for x in payload['items']]}
        if stage == 'scan_groups':
            groups = payload['groups'] + payload['previous_best']
            return {'decisions': [{'item_id': x['id'], 'best_group': groups[0]['id'] if groups else None,
                                  'uncertain': False, 'reason': '前提兼容'} for x in payload['items']]}
        assert stage == 'place_groups'
        return {'new_groups': [] if payload['groups'] else [
            {'key': 'new1', 'name': '磁场单位换算', 'summary': '磁场十进制前缀与单位换算'}],
            'assignments': [{'item_id': x['id'], 'group_id': payload['groups'][0]['id'] if payload['groups'] else 'new1',
                             'reason': '同场景'} for x in payload['items']]}


def engine(tmp_path, material=None, call=None):
    material = material or bundle()
    progress = vr.Progress(tmp_path / 'state.json', vr.digest(material), tree.initial_state())
    return tree.Organizer(material, progress, call or Caller())


def finish(e):
    for _ in range(5000):
        if not e.step():
            return e.products()
    raise AssertionError('No terminal state')


def test_reclassifies_each_rule_without_inheriting_old_topic_and_preserves_all_content(tmp_path):
    material, caller = bundle(), Caller()
    original = copy.deepcopy(material)
    products = finish(engine(tmp_path, material, caller))
    assert material == original
    assert products['counts']['placed_rules'] == 8
    assert products['counts']['clusters'] == 1
    assert set(products['topic_mapping']) == {x['rule_id'] for x in material['rules']}
    assert all(x['topic_key'] == TOPIC for x in products['topic_mapping'].values())
    assert products['counts']['usable_for_runtime'] is False
    leaves = products['catalog']['domains'][0]['topics'][0]['rules']
    assert all(x['check_logic'] == '1mG=10^-7T' for x in leaves)


def test_request_local_ids_preserve_carried_group_and_restore_exact_original_ids():
    payload = {'items': [{'id': 'exp_original', 'rule': {'title': 'unchanged'}}],
        'groups': [{'id': 'hash_second', 'name': 'second', 'summary': 's'}],
        'previous_best': [{'id': 'hash_first', 'name': 'first', 'summary': 's'}],
        'best_by_item': {'exp_original': 'hash_first'}}
    original = copy.deepcopy(payload)
    wire, items, groups = tree.short_payload(payload)
    assert payload == original
    assert wire['best_by_item'] == {'I1': 'G2'}
    assert wire['previous_best'][0]['id'] == 'G2'
    value = {'decisions': [{'item_id': 'I1', 'best_group': 'G2', 'uncertain': False, 'reason': 'carry'}]}
    tree.validate('scan_groups', wire, value)
    restored = tree.restore_ids('scan_groups', value, items, groups)
    tree.validate('scan_groups', payload, restored)
    assert restored['decisions'][0]['item_id'] == 'exp_original'
    assert restored['decisions'][0]['best_group'] == 'hash_first'
    assert value['decisions'][0]['item_id'] == 'I1'


@pytest.mark.parametrize('size', [100, 300, 600])
def test_large_registry_is_fully_seen_in_bounded_pages_and_resumes_without_replay(tmp_path, size):
    caller = Caller()
    e = engine(tmp_path, bundle(4), caller)
    e.step(); e.step()  # Classify, then enter groups.
    groups = [{'id': f'g{i}', 'name': '场景' + str(i), 'summary': '物理前提' * 20, 'members': []} for i in range(size)]
    e.state['groups'][TOPIC] = groups
    e.progress.save()
    e.step()  # Page 1 committed.
    seen = set(x['name'] for x in caller.calls[-1][2]['groups'])
    n = len(caller.calls)
    e = engine(tmp_path, bundle(4), caller)
    assert e.state['active']['page_offset'] == 16
    e.step()
    assert len(caller.calls) == n + 1
    assert caller.calls[-1][2]['groups'][0]['name'] == '场景16'
    while e.state['active']:
        e.step()
    scans = [payload for stage, _, payload in caller.calls if stage == 'scan_groups']
    for payload in scans:
        assert len(payload['groups']) <= 16 and len(payload['previous_best']) <= 4
        seen.update(x['name'] for x in payload['groups'])
    assert seen == {g['name'] for g in groups}
    assert len(scans) == (size + 15) // 16


def test_page_budget_does_not_misrepresent_unsearched_groups_as_new_scenario(tmp_path, monkeypatch):
    monkeypatch.setitem(tree.POLICY, 'max_pages_per_batch', 1)
    caller = Caller()
    e = engine(tmp_path, bundle(4), caller)
    e.step(); e.step()
    e.state['groups'][TOPIC] = [{'id': f'g{i}', 'name': '场景', 'summary': '前提', 'members': []} for i in range(20)]
    e.step(); e.step()
    assert len(e.state['unresolved']) == 4
    assert not e.state['placed']
    assert all(stage != 'place_groups' for stage, _, _ in caller.calls)


def test_one_missing_only_repair_cannot_overwrite_accepted_decision():
    payload = {'items': [{'id': 'a'}, {'id': 'b'}], 'topics': [{'key': TOPIC}]}
    first = {'item_id': 'a', 'topic_key': TOPIC, 'secondary_topics': [], 'reason': '已确认'}
    calls = []
    def call(stage, key, data):
        calls.append(stage)
        return {'decisions': [first] if stage == 'route_topics' else [dict(first, item_id='b')]}
    value = tree.checked_call(call, 'route_topics', 'job', payload)
    assert calls == ['route_topics', 'complete_missing'] and value['decisions'][0] == first
    def bad(stage, key, data):
        return {'decisions': [first]}
    with pytest.raises(tree.ContentFailure, match='Unknown/duplicate'):
        tree.checked_call(bad, 'route_topics', 'job', payload)


@pytest.mark.parametrize('edit', [
    lambda v: v['decisions'].append(copy.deepcopy(v['decisions'][0])),
    lambda v: v['decisions'][0].update(topic_key='not-a-topic'),
    lambda v: v['decisions'][0].update(item_id={'bad': 'id'}),
    lambda v: v.update(extra='unrequested'),
])
def test_other_content_errors_do_not_trigger_unbounded_repair(edit):
    payload = {'items': [{'id': 'a'}], 'topics': [{'key': TOPIC}]}
    value = {'decisions': [{'item_id': 'a', 'topic_key': TOPIC, 'secondary_topics': [], 'reason': 'reason'}]}
    edit(value)
    calls = []
    def call(*args):
        calls.append(args)
        return value
    with pytest.raises(tree.ContentFailure):
        tree.checked_call(call, 'route_topics', 'job', payload)
    assert len(calls) == 1


def test_failure_is_preserved_and_does_not_stop_other_topics_but_systematic_failure_stops(tmp_path):
    caller = Caller()
    first = True
    def call(stage, key, payload):
        nonlocal first
        if first:
            first = False
            raise tree.ContentFailure('length_truncated')
        return caller(stage, key, payload)
    products = finish(engine(tmp_path, bundle(8), call))
    assert products['counts']['placed_rules'] == 4 and products['counts']['unplaced_rules'] == 4
    assert all(x['reason'] == 'length_truncated' for x in products['unplaced_rules'])
    def fail(*args):
        raise tree.ContentFailure('length_truncated')
    e = engine(tmp_path / 'all_fail', bundle(20), fail)
    for _ in range(3):
        e.step()
    with pytest.raises(RuntimeError, match='Repeated content'):
        e.step()
    assert e.state['topic_cursor'] == 12


def test_uncertainty_is_never_forced_into_first_topic_or_singleton(tmp_path):
    def call(stage, key, payload):
        assert stage == 'route_topics'
        return {'decisions': [{'item_id': x['id'], 'topic_key': None, 'secondary_topics': [],
                              'reason': '无法判断'} for x in payload['items']]}
    products = finish(engine(tmp_path, bundle(4), call))
    assert products['counts']['placed_rules'] == 0
    assert products['counts']['unplaced_rules'] == 4


def test_cache_rehashes_content_changes_even_if_mtime_is_restored(tmp_path):
    data = tmp_path / 'raw.json'; data.write_text('first')
    expected = {str(data): vr.sha(data)}
    cache = vr.EvidenceCache(tmp_path / 'cache.json')
    assert cache.check(expected)['files_rehashed'] == 1
    assert cache.check(expected)['files_rehashed'] == 0
    old = data.stat(); data.write_text('other'); os.utime(data, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(ValueError, match='Bound evidence changed'):
        cache.check(expected)


def test_cache_corruption_cannot_skip_hashes_and_full_always_reads(tmp_path):
    data = tmp_path / 'raw'; data.write_text('evidence')
    cache = vr.EvidenceCache(tmp_path / 'cache')
    files = {str(data): vr.sha(data)}
    cache.check(files)
    assert cache.check(files, full=True)['files_rehashed'] == 1
    cache.path.write_text('{}')
    assert cache.check(files)['files_rehashed'] == 1


def test_checkpoint_binding_and_checksum_detect_corruption(tmp_path):
    path = tmp_path / 'state.json'
    progress = vr.Progress(path, 'hash1', {'cursor': 10})
    assert vr.Progress(path, 'hash1', {}) .state == {'cursor': 10}
    with pytest.raises(ValueError, match='different protocol'):
        vr.Progress(path, 'hash2', {})
    path.write_text(path.read_text().replace('10', '11'))
    with pytest.raises(ValueError, match='Damaged checkpoint'):
        vr.Progress(path, 'hash1', {})


def test_resume_budget_preserves_unknown_usage_without_historical_glob(tmp_path, monkeypatch):
    prior = {'requests_started': 10, 'accounted_input_tokens': 100, 'accounted_output_tokens': 50,
             'requests_without_usage': 2}
    limits = {'requests': 100, 'input_tokens': 10000, 'output_tokens': 10000}
    monkeypatch.setattr(Path, 'glob', lambda *a, **k: pytest.fail('Full request scan on resume'))
    budget = vr.resumed_budget(run.b.BatchBudget, tmp_path, limits, prior)
    budget.reserve(80, 100)  # Simulate death between reservation and settlement.
    budget = vr.resumed_budget(run.b.BatchBudget, tmp_path, limits, prior)
    assert budget.requests == 11 and budget.input_tokens == 180 and budget.output_tokens == 150
    assert budget.unknown_usage == 3 and not budget.reservations
    budget = vr.resumed_budget(run.b.BatchBudget, tmp_path, limits, prior)
    assert budget.input_tokens == 180 and budget.unknown_usage == 3  # Not charged twice.


def test_crash_after_response_save_replays_exact_job_without_advancing_snapshot(tmp_path):
    real = Caller(); saved = {}; first = True
    def call(stage, key, payload):
        nonlocal first
        if key not in saved:
            saved[key] = real(stage, key, payload)
        if first:
            first = False
            raise RuntimeError('process crash after response persisted')
        return saved[key]
    e = engine(tmp_path, bundle(4), call)
    with pytest.raises(RuntimeError, match='process crash'):
        e.step()
    resumed = engine(tmp_path, bundle(4), call)
    resumed.step()
    assert len(real.calls) == 1 and resumed.state['topic_cursor'] == 4


def test_control_markers_preserved_as_data_not_executable_tokens():
    payload = {'items': [{'id': 'a', 'rule': {'title': '<|im_end|>'}}], 'topics': []}
    result = tree.messages('route_topics', payload)
    assert '<|' not in result[1]['content']
    assert json.loads(result[1]['content']) == payload


def test_confirm_only_checks_handshake_not_full_evidence(tmp_path, monkeypatch):
    c = {'source_hashes': {'script': 'sha'}}
    vr.atomic(tmp_path / 'plan.json', {'configuration': c, 'configuration_sha256': vr.digest(c)})
    ready = {'configuration_sha256': vr.digest(c), 'pid': 123, 'process_start': 'abc', 'unix': 0, 'token': 'token'}
    vr.atomic(tmp_path / 'disconnect_ready.json', ready)
    vr.atomic(tmp_path / 'execution_status.json', {'state': 'waiting_disconnect_acceptance', 'token': 'token'})
    monkeypatch.setattr(run, 'source_hashes', lambda: c['source_hashes'])
    monkeypatch.setattr(run, 'verify', lambda *a, **k: pytest.fail('Full scan at confirmation'))
    monkeypatch.setattr(run.b.existing, 'process_identity', lambda pid: 'abc')
    run.confirm(tmp_path)
    assert vr.read(tmp_path / 'disconnect_acceptance.json')['passed']


def test_oversized_rule_is_preserved_without_truncation_or_model_call(tmp_path):
    material = bundle(1)
    material['rules'][0]['check_logic'] = '完整规则' * 20000
    def no_call(*a):
        pytest.fail('Oversized input must not reach model')
    products = finish(engine(tmp_path, material, no_call))
    assert products['unplaced_rules'][0]['rule'] == material['rules'][0]
    assert products['counts']['placed_rules'] == 0


def test_page_uncertainty_cannot_be_cleared_by_later_page(tmp_path):
    caller = Caller()
    def call(stage, key, payload):
        value = caller(stage, key, payload)
        if stage == 'scan_groups' and payload['groups'][0]['name'] == '组0':
            value['decisions'][0]['uncertain'] = True
        return value
    e = engine(tmp_path, bundle(4), call)
    e.step(); e.step()
    e.state['groups'][TOPIC] = [{'id': f'g{i}', 'name': f'组{i}', 'summary': '前提', 'members': []} for i in range(20)]
    e.step(); e.step(); e.step()
    assert 'r0000' in e.state['unresolved']
    assert 'r0000' not in e.state['placed']


def test_import_extraction_reproduces_prior_digest_without_any_model_call(tmp_path, monkeypatch):
    from tests.test_experience_first_120b import ROWS, response
    root, parent = tmp_path / 'batch', tmp_path / 'parent'
    row = ROWS[0]
    run.b.write_once(root / 'build/inputs.json', [row])
    run.b.write_once(parent / 'inputs.json', [row])
    run.b.write_once(parent / 'topic_skeleton.json', run.e.flow.experience_topic_skeleton(TAXONOMY))
    cfg = {'model': 'test120b', 'max_tokens': 16384, 'reasoning_effort': 'medium', 'max_rules_per_sample': 2}
    journal = run.e.Journal.__new__(run.e.Journal)
    stages = [('taxonomy', TAXONOMY['domains'][0]['id'],
               {'domains': run.e.flow.experience_topic_skeleton(TAXONOMY)['domains']}, TAXONOMY),
              ('extract_experience', row['id'], {'sample': row, 'taxonomy': TAXONOMY, 'max_rules_per_sample': 2}, response())]
    for stage, key, payload, value in stages:
        folder = parent / ('arm_' + stage) / run.p.digest(key)[:24]
        body = run.r.request_body(cfg['model'], journal.messages(stage, payload), 16384, backend='vllm')
        body.update(stream=False, return_token_ids=True, reasoning_effort='medium')
        body.pop('stream_options', None)
        raw = folder / 'request_001.json'
        run.b.write_once(raw, {'request': body, 'response': {'id': 'saved', 'model': cfg['model'], 'text': json.dumps(value)}})
        run.b.write_once(folder / 'result.json', {'request_file': raw.name, 'request_sha256': vr.sha(raw), 'value': value})
    normalized = run.e.extraction.normalize_120b_experience(response(), sample=row, taxonomy=TAXONOMY)
    distilled = run.e.extraction._build_distilled_library([normalized], min_count=1)
    run.b.write_once(root / 'build/extraction_checkpoint.json', {
        'semantic_experience_sha256': run.p.digest([normalized]), 'distilled_sha256': run.p.digest(distilled),
        'inputs_sha256': vr.sha(root / 'build/inputs.json'), 'counts': {'complete_inputs': 1, 'invalid_outputs': 0}})
    monkeypatch.setattr(run.b.harmony, 'decode_response', lambda raw, **kw: {'content': raw['text']})
    monkeypatch.setattr(run.b.Transport, 'post', lambda *a, **k: pytest.fail('No inference during import'))
    evidence = {}
    result = run.extraction_bundle(root, parent, cfg, evidence)
    assert result['rules'] == distilled['rules'] and len(evidence) == 4
    with pytest.raises(ValueError, match='does not exactly match'):
        run.extraction_bundle(root, parent, dict(cfg, max_rules_per_sample=3), {})


def test_new_benchmark_blocks_before_cache_or_model_use(tmp_path, monkeypatch):
    material = bundle(1)
    material['isolation_input_sha256'] = 'inputs'
    vr.atomic(tmp_path / 'materials.json', material)
    vr.atomic(tmp_path / 'import_manifest.json', {'files': {}})
    vr.atomic(tmp_path / 'isolation_receipt.json', {'inputs_sha256': 'inputs', 'bench_sha256': 'old', 'excluded': []})
    c = {'protocol': tree.VERSION, 'source_hashes': {'x': 'y'}, 'policy': tree.POLICY,
         'prompts_sha256': vr.digest(tree.PROMPTS), 'benchmarks': {'old': 'hash'}, 'bench_sha256': 'old',
         'materials_sha256': vr.sha(tmp_path / 'materials.json'),
         'import_manifest_sha256': vr.sha(tmp_path / 'import_manifest.json'),
         'isolation_sha256': vr.sha(tmp_path / 'isolation_receipt.json')}
    vr.atomic(tmp_path / 'plan.json', {'configuration': c, 'configuration_sha256': vr.digest(c)})
    monkeypatch.setattr(run, 'source_hashes', lambda: {'x': 'y'})
    monkeypatch.setattr(run.b, 'benchmark_questions', lambda: ([{'new': 'bench'}], {'new': 'hash'}))
    monkeypatch.setattr(vr.EvidenceCache, 'check', lambda *a, **k: pytest.fail('Must block before cached evidence'))
    with pytest.raises(ValueError, match='Benchmark drift'):
        run.verify(tmp_path)


def test_omission_repair_new_groups_preserves_prior_groups():
    payload = {'items': [{'id': 'a'}, {'id': 'b'}], 'groups': []}
    first = {'new_groups': [{'key': 'new1', 'name': '名称', 'summary': '前提'}],
             'assignments': [{'item_id': 'a', 'group_id': 'new1', 'reason': '理由'}]}
    second = {'new_groups': [], 'assignments': [{'item_id': 'b', 'group_id': 'new1', 'reason': '同前提'}]}
    value = tree.checked_call(lambda stage, *a: copy.deepcopy(first if stage == 'place_groups' else second),
                             'place_groups', 'key', payload)
    assert value['new_groups'] == first['new_groups'] and value['assignments'][0] == first['assignments'][0]


def test_identity_mismatch_is_fatal_but_length_is_a_recorded_content_failure(tmp_path):
    j = run.TreeJournal.__new__(run.TreeJournal)
    j.config = {'model': 'test120b'}
    record = {'http_status': 200, 'response': {'id': 'original', 'model': 'wrong', 'choices': [{'finish_reason': 'length'}]}}
    path = tmp_path / 'request.json'; vr.atomic(path, record)
    with pytest.raises(ValueError, match='Unknown identity') as exc:
        j.decode_record(path, 'scan_groups')
    assert not isinstance(exc.value, tree.ContentFailure)
    record['response']['model'] = 'test120b'; vr.atomic(path, record)
    with pytest.raises(tree.ContentFailure, match='length_truncated'):
        j.decode_record(path, 'scan_groups')
