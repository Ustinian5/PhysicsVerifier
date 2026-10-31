import json

import pytest

from rule_framework.verified_resume import atomic, digest, sha
from scripts.analyze_organization_comparison import evidence_path, sample_record


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


def test_corrupted_raw_response_is_rejected(tmp_path):
    root, folder = sample(tmp_path)
    (folder / 'turn_02/response.sse').write_text('changed response')
    with pytest.raises(ValueError, match='Raw response hash drift'):
        sample_record(folder / 'result.json', root, 'binding', 'pinned-model', {})


def test_evidence_path_cannot_escape_run(tmp_path):
    with pytest.raises(ValueError, match='escapes run'):
        evidence_path(tmp_path / 'run', '/server/run/../other/response.json')
