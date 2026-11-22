import pytest

from scripts import audit_verifier_library_scale as replay
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
    held = library.SOURCE_HOLD
    count = 250-len(held)
    chosen = {str(i) for i in range(count)} | held
    split = dict(build=[dict(id=i) for i in sorted(chosen | {str(i) for i in range(count,2000-len(held))})],
                 reserve=[dict(id='reserve'+str(i)) for i in range(702)])
    monkeypatch.setattr(library, 'family_levels', lambda *args: iter([(250, chosen)]))
    with pytest.raises(ValueError, match='disposition'):
        library.source_draw(split, [], library.SEEDS[0], False)
    rows, removed = library.source_draw(split, [], library.SEEDS[0], True)
    assert {r['id'] for r in rows}==chosen and removed==held


def test_larger_sizes_extend_the_smaller_draw_without_splitting_a_family():
    import hashlib
    seed = library.SEEDS[1]
    ordered = sorted(map(str, range(6)), key=lambda k: hashlib.sha256(f'{seed}:{k}'.encode()).digest())
    ids = [f'q{i}' for i in range(2000)]
    sizes = (249, 251, 1, 250, 249, 1000)
    families = [dict(id=key, members=ids[sum(sizes[:i]):sum(sizes[:i+1])])
                for i, key in enumerate(ordered)]
    split = dict(build=[dict(id=s) for s in ids], reserve=[dict(id=f'r{i}') for i in range(702)])
    draws = []
    for n in library.BUILD_SIZES:
        rows, held = library.source_draw(split, families, seed, False, n)
        assert len(rows) == n and not held
        draws.append({r['id'] for r in rows})
    assert draws[0] < draws[1] < draws[2]
    assert set(families[1]['members']).isdisjoint(draws[1])
    assert set(families[1]['members']) <= draws[2]


def test_ports_are_distinct_across_concurrent_sizes_and_seeds():
    ports = {node.service_port(n,s) for n in library.BUILD_SIZES for s in library.SEEDS}
    assert len(ports) == len(library.BUILD_SIZES) * len(library.SEEDS)
    assert node.service_port(250, 20261122) == 18122


def test_random_seed_registration_was_frozen_before_membership():
    import hashlib
    import json
    reg = dict(library.RANDOM_REGISTRATION)
    rows = reg.pop('draws')
    seed_only = reg.pop('seed_registration_sha256')
    encoded = (json.dumps(reg,ensure_ascii=False,indent=2)+'\n').encode()
    assert hashlib.sha256(encoded).hexdigest() == seed_only
    assert len(set(reg['seeds'])) == len(reg['seeds']) == 3
    assert set(reg['seeds']).isdisjoint(library.ORIGINAL_SEEDS)
    assert [(r['seed'],r['source_questions']) for r in rows] == [
        (seed,n) for seed in reg['seeds'] for n in library.BUILD_SIZES]


def test_idle_card_is_restricted_to_the_requested_gpu():
    idle = lambda i: [str(i), 'uuid'+str(i), 'A800', '81920','3','0']
    snapshot = dict(gpus=[idle(1),idle(4),idle(5)],compute_processes=[('uuid4','987')])
    assert node.available_card(snapshot,4) is None
    assert node.available_card(snapshot,5)==('5','uuid5')
    with pytest.raises(ValueError, match='outside'):
        node.available_card(snapshot,1)


def test_replay_refuses_a_response_to_a_different_material_prompt(tmp_path):
    key, stage, config = 'job', 'tree_route_topics', {'model':'120b'}
    body = dict(model='120b', messages=[{'role':'user','content':'one candidate'}])
    digest = library.construction.c.corpus.p.digest
    relative = f'raw/{stage}/{digest(key)[:24]}/request_001.json'
    path = tmp_path/relative
    response = dict(model='120b')
    library.vr.atomic(path, dict(status='received',http_status=200,request=body,response=response,
                                 fingerprint=digest([key,body,config])))
    evidence = dict(file=relative,sha256=library.vr.sha(path))
    library.vr.atomic(tmp_path/'decoded'/stage/(library.vr.digest(key)+'.json'),dict(evidence=evidence))
    reader = replay.RecordedResponses(tmp_path,{relative:evidence['sha256']},config)
    assert reader.call(stage,key,body)==(response,evidence)
    with pytest.raises(ValueError,match='request differs'):
        reader.call(stage,key,dict(body,messages=[]))
