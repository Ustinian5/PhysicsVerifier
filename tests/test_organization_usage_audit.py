import json

import pytest

from rule_framework.verified_resume import atomic, digest, sha
from scripts.analyze_organization_comparison import evidence_path, sample_record
from scripts import analyze_organization_comparison as audit


def sample(tmp_path, status='failed', citations=()):
    root = tmp_path / 'run'
    folder = root / 'formal' / 'skills' / 'one'
    turn = folder / 'turn_02'
    turn.mkdir(parents=True)
    messages = [{'role': 'user', 'content': 'Original answer and complete reference'}]
    request = {'model': 'pinned-model', 'messages': messages}
    atomic(turn / 'request.json', {'request': request, 'request_sha256': digest(request)})
    (turn / 'response.sse').write_text('raw response')
    atomic(turn / 'response.json', {'model_ids': ['pinned-model'],
                                  'raw_sse_sha256': sha(turn / 'response.sse')})
    diagnostics = [{'message': 'A specific finding', 'evidence': {'quote': 'Original answer'},
                    'reference_ids': list(citations)}] if status == 'completed' else []
    result = {'id': 'one', 'status': status, 'configuration_sha256': 'binding',
              'diagnostics': diagnostics, 'trace': {'response_evidence': []}}
    atomic(folder / 'result.json', result)
    atomic(folder / 'reference_packet.json', {'configuration_sha256': 'binding',
        'input_sha256': 'input', 'query_evidence_sha256': 'query', 'packet': {
            'messages': messages, 'references': [{'id': 'skill-1', 'text': 'A useful fact'}],
            'retrieval': {'provided_complete_ids': ['skill-1'], 'reference_tokens': 20, 'omitted': []}}})
    return root, folder


def test_failed_generation_does_not_erase_delivered_material(tmp_path):
    root, folder = sample(tmp_path)
    item = sample_record(folder / 'result.json', root, 'binding', 'pinned-model', {})
    assert item['status'] == 'failed'
    assert item['provided_ids'] == ['skill-1']
    assert item['reference_tokens'] == 20
    assert item['published_diagnostic_count'] == 0


def test_packet_claim_alone_cannot_prove_delivery(tmp_path):
    root, folder = sample(tmp_path)
    path = folder / 'turn_02/request.json'
    value = json.loads(path.read_text())
    value['request']['messages'] = [{'role': 'user', 'content': 'No reference was sent'}]
    value['request_sha256'] = digest(value['request'])
    atomic(path, value)
    with pytest.raises(ValueError, match='not delivered verbatim'):
        sample_record(folder / 'result.json', root, 'binding', 'pinned-model', {})


def test_unsupplied_citation_is_rejected(tmp_path):
    root, folder = sample(tmp_path, status='completed', citations=['invented-reference'])
    with pytest.raises(ValueError, match='unsupplied reference'):
        sample_record(folder / 'result.json', root, 'binding', 'pinned-model', {})


def test_declared_bad_reference_is_preserved_as_violation_not_material_use(tmp_path):
    root, folder = sample(tmp_path, status='completed', citations=['invented-reference'])
    path = folder/'result.json'; value = json.loads(path.read_text())
    value['protocol_audit'] = {'passed': False, 'violations': ['unsupplied_reference']}
    atomic(path, value)
    result = sample_record(path, root, 'binding', 'pinned-model', {})
    assert result['citations'][0]['unsupplied_reference_ids'] == ['invented-reference']
    assert result['provided_ids'] == ['skill-1']


def test_99_audit_uses_frozen_complete_cohort_and_binds_reference(tmp_path, monkeypatch):
    from rule_framework import organization_bench99 as bench
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    rows = [dict(id=str(i), question='q', prediction='a', context='') for i in range(99)]
    run = tmp_path/'run'
    atomic(run/'inputs.json', rows)
    dependencies = {}
    for name in (bench.INPUT, audit.scoring.RELEASE, audit.scoring.REFERENCE):
        atomic(tmp_path/name, {})
        dependencies['/original/repo/'+name] = sha(tmp_path/name)
    monkeypatch.setattr(bench, 'load_bench', lambda root: (rows, ['dataset'], {'unknown': []}))
    config = dict(protocol=bench.VERSION, sample_count=99,
                  input_sha256=sha(run/'inputs.json'), files=dependencies)
    inputs, data, unknown, key = audit.audit_inputs(run, config)
    assert len(inputs) == 99 and data == ['dataset'] and key == 'scores_all_99'
    atomic(tmp_path/audit.scoring.REFERENCE, {'changed': True})
    with pytest.raises(ValueError, match='dependency drift'):
        audit.audit_inputs(run, config)


def test_corrupted_raw_response_is_rejected(tmp_path):
    root, folder = sample(tmp_path)
    (folder / 'turn_02/response.sse').write_text('changed response')
    with pytest.raises(ValueError, match='Raw response hash drift'):
        sample_record(folder / 'result.json', root, 'binding', 'pinned-model', {})


def test_evidence_path_cannot_escape_run(tmp_path):
    with pytest.raises(ValueError, match='escapes run'):
        evidence_path(tmp_path / 'run', '/server/run/../other/response.json')


def test_external_prefix_requires_both_manifest_and_exact_file_hash(tmp_path):
    root, parent = tmp_path/'new', tmp_path/'old'
    actual = parent/'queries/q/request.json'; atomic(actual, {'request': 'old prefix'})
    recorded = '/server/old/queries/q/request.json'
    prefixes = {'row': {'files': {recorded: sha(actual)}}}
    atomic(root/'frozen_prefixes.json', prefixes)
    c = dict(reused_queries=True, parent='/server/old', files={recorded: sha(actual),
        '/server/new/frozen_prefixes.json': sha(root/'frozen_prefixes.json')})
    atomic(root/'plan.json', dict(configuration=c, configuration_sha256=digest(c)))
    assert evidence_path(root, recorded) == actual.resolve()
    with pytest.raises(ValueError, match='Unbound external query'):
        evidence_path(root, '/server/old/unregistered.json')
    actual.write_text('changed')
    with pytest.raises(ValueError, match='evidence changed'): evidence_path(root, recorded)
