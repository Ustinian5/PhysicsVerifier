from types import SimpleNamespace

import pytest

from scripts.step_check_format import LABEL, LABELS, normalize, publish_replay, vr


def frame(body, recipient=LABEL, channel='final'):
    return SimpleNamespace(channel=channel, recipient=recipient, content=[SimpleNamespace(text=body)])


def parsed(*messages):
    return SimpleNamespace(messages=list(messages), current_content=None)


@pytest.mark.parametrize('label', LABELS)
def test_complete_tag_leaves_body_and_analysis_intact(label):
    reason = frame('reasoning', None, 'analysis')
    reply = frame('{"diagnostics":[]}', label)
    actual = normalize(parsed(reason, reply), [200002], [])
    assert actual.messages[0] is reason
    assert actual.messages[1].recipient == '<|constrain|>json'
    assert actual.messages[1].content is reply.content
    assert reply.recipient == label


@pytest.mark.parametrize('body', ['[]', '{"diagnostics":[{}]}', '{"diagnostics":[]}{}',
                                  '{"diagnostics":[],"queries":[]}',
                                  '{"diagnostics":[{"message":"m","evidence":{"quote":"q"},"reference_ids":["x"]}]}'])
@pytest.mark.parametrize('label', LABELS)
def test_invalid_bodies_cannot_become_successes(body, label):
    with pytest.raises(ValueError):
        normalize(parsed(frame(body, label)), [200002], [])


def test_tools_other_labels_and_wrong_termination_stay_distinct():
    for message in (frame('{}','functions.CHECK'),frame('{}','<|constrain|>Check'),frame('{}',LABEL,'analysis')):
        record = parsed(message)
        assert normalize(record, [200012], []) is record
    for label in LABELS:
        with pytest.raises(ValueError):
            normalize(parsed(frame('{"diagnostics":[]}',label)), [200012], [])
    with pytest.raises(ValueError):
        normalize(parsed(frame('{}','functions.x','analysis'),frame('{"diagnostics":[]}')), [200002], [])


def test_replay_publishes_once_and_preserves_both_originals(tmp_path):
    path, new = tmp_path/'result.json', tmp_path/'replayed.json'
    path.write_bytes(b'original failure')
    new.write_bytes(b'validated result')
    before = vr.sha(path)
    for _ in range(2):
        archive = publish_replay(path, new, before)
        assert vr.sha(archive) == before
        assert path.read_bytes() == new.read_bytes() == b'validated result'
    path.unlink()
    path.write_bytes(b'other publication')
    with pytest.raises(ValueError):
        publish_replay(path, new, before)
    assert path.read_bytes() == b'other publication'
