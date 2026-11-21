import pytest

from scripts import run_node14_library_scale as node
from scripts import run_verifier_library_scale as library


def test_dedup_and_rule_support_use_only_this_draw():
    def experience(sid):
        return dict(sample_id=sid, topic_guess={'domain':'Physics','topic':'Units'},
                    semantic_audit={'key_errors':[dict(id='e', claim='units')]},
                    experience_rules=[dict(title='Length units',trigger='length',check_logic='compare units',
                                           source_error_ids=['e'])])
    first, second = experience('a'), experience('b')
    rules, support = library.local_rules([first,second], {'a','b'})
    small, small_support = library.local_rules([first], {'a'})
    rid = small[0]['rule_id']
    assert len(rules)==len(small)==1
    assert {r['sample_id'] for r in support[rid]}=={'a','b'}
    assert small[0]['sample_ids']==['a'] and [r['sample_id'] for r in small_support[rid]]==['a']
    with pytest.raises(ValueError, match='outside this draw'):
        library.local_rules([first,second], {'a'})


def test_source_hold_retains_the_original_draw_and_never_substitutes(monkeypatch):
    held, = library.SOURCE_HOLD
    chosen = {str(i) for i in range(249)} | {held}
    split = dict(build=[dict(id=i) for i in sorted(chosen | {str(i) for i in range(249,1999)})],
                 reserve=[dict(id='reserve'+str(i)) for i in range(702)])
    monkeypatch.setattr(library, 'family_levels', lambda *args: iter([(250, chosen)]))
    with pytest.raises(ValueError, match='disposition'):
        library.source_draw(split, [], library.SEEDS[0], False)
    rows, removed = library.source_draw(split, [], library.SEEDS[0], True)
    assert {r['id'] for r in rows}==chosen and removed=={held}


def test_idle_card_is_restricted_to_the_requested_gpu():
    idle = lambda i: [str(i), 'uuid'+str(i), 'A800', '81920','3','0']
    snapshot = dict(gpus=[idle(1),idle(4),idle(5)],compute_processes=[('uuid4','987')])
    assert node.available_card(snapshot,4) is None
    assert node.available_card(snapshot,5)==('5','uuid5')
    with pytest.raises(ValueError, match='outside'):
        node.available_card(snapshot,1)
