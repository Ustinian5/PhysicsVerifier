from types import SimpleNamespace

import pytest

from scripts.step_check_format import LABEL, normalize


def frame(body, recipient=LABEL, channel='final'):
    return SimpleNamespace(channel=channel, recipient=recipient, content=[SimpleNamespace(text=body)])


def parsed(*messages):
    return SimpleNamespace(messages=list(messages), current_content=None)


def test_complete_tag_leaves_body_and_analysis_intact():
    reason = frame('reasoning', None, 'analysis')
    reply = frame('{"diagnostics":[]}')
    actual = normalize(parsed(reason, reply), [200002], [])
    assert actual.messages[0] is reason
    assert actual.messages[1].recipient == '<|constrain|>json'
    assert actual.messages[1].content is reply.content
    assert reply.recipient == LABEL


@pytest.mark.parametrize('body', ['[]', '{"diagnostics":[{}]}', '{"diagnostics":[]}{}',
                                  '{"diagnostics":[],"queries":[]}',
                                  '{"diagnostics":[{"message":"m","evidence":{"quote":"q"},"reference_ids":["x"]}]}'])
def test_invalid_bodies_cannot_become_successes(body):
    with pytest.raises(ValueError):
        normalize(parsed(frame(body)), [200002], [])


def test_tools_other_labels_and_wrong_termination_stay_distinct():
    for message in (frame('{}','functions.CHECK'),frame('{}','<|constrain|>check'),frame('{}',LABEL,'analysis')):
        record = parsed(message)
        assert normalize(record, [200012], []) is record
    with pytest.raises(ValueError):
        normalize(parsed(frame('{"diagnostics":[]}')), [200012], [])
    with pytest.raises(ValueError):
        normalize(parsed(frame('{}','functions.x','analysis'),frame('{"diagnostics":[]}')), [200002], [])
