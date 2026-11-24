"""Regression checks for independent index coverage, safe IDs, and paired timing outputs."""
from types import SimpleNamespace

import pytest

from rule_framework import semantic_index_retrieval as semantic
from scripts import score_timed_semantic_retrieval as scoring
from scripts import run_timed_semantic_retrieval as timing


def test_flat_semantic_route_visits_every_id_independently_of_bm25():
    index = semantic.Index.__new__(semantic.Index)
    index.safe_tree = None
    index.ids = [str(i) for i in range(1452)]
    index.cards = {i: (i, 'summary') for i in index.ids}
    index.lexical = lambda query: (['1451'], None)
    seen = []

    def select(key, prompt, payload, validate):
        candidates = payload['candidates']
        if key.startswith('semantic_page'):
            seen.extend(card[1] for card in candidates)
        return validate({'ids': [1]})

    chosen, trace = index.fused_selection(['same query'], select)
    assert seen == index.ids
    assert trace['semantic'] == ['0']
    assert trace['lexical'] == ['1451']
    assert set(trace['fused']) == {'0', '1451'} and chosen


def test_semantic_tree_only_visits_selected_branches():
    index = semantic.Index.__new__(semantic.Index)
    nodes = {'root': {'name':'root', 'children':['a','b']},
             'a': {'name':'one', 'summary':'', 'children':[], 'rule_ids':['1']},
             'b': {'name':'two', 'summary':'', 'children':[], 'rule_ids':['2']}}
    index.safe_tree = SimpleNamespace(corpus=SimpleNamespace(nodes=nodes))
    index.cards = {i: (i, 'safe') for i in ['1','2']}
    index.ids = ['2','1']
    calls = []

    def select(key, prompt, payload, validate):
        calls.append(key)
        if key == 'semantic_leaves':
            assert [v[1] for v in payload['candidates']] == ['2']
            return validate({'ids':[1]})
        return validate({'ids':[2]})

    assert index.semantic(['query'], select) == ['2']
    assert calls == ['semantic_route_1','semantic_leaves']
    with pytest.raises(ValueError):
        semantic.ranking_validator(3, 4, ['parent']*3)({'ids':[1,2,3]})


def test_paired_counts_use_only_valid_binary_and_timing_uses_preparation_once():
    def item(verdict):
        return dict(status='complete', value={'verdict': verdict})
    truth = {'a':'incorrect','b':'correct','c':'correct'}
    baseline = {'a':item('correct'),'b':item('correct'),'c':item('uncertain')}
    new = {'a':item('incorrect'),'b':item('incorrect'),'c':item('correct')}
    assert scoring.paired(truth, baseline, new) == dict(improved=['a'],regressed=['b'],same=[],
                                                        outside_common_binary=['c'])
    row = dict(id='a',lookup_seconds=8.,prepare_seconds=1.,check_seconds=2.,
               pipeline_seconds=11.,after_prepare_seconds=10.)
    assert timing.timing_prefix([row])['projected_468_seconds'] == 5148.


def test_scan_profile_reuses_a_failed_page_and_only_submits_missing_pages(tmp_path, monkeypatch):
    from scripts import time_remaining_index_pages as profile

    q, stamp = 'frozen_id', 'primary'
    arm = tmp_path/'rag'
    original = arm/'rows'/q
    history_record = dict(status='complete',value={'queries':['query']},wall_seconds=1.,
                          usage=dict(completion_tokens=5))
    monkeypatch.setattr(profile.b, 'prefix_record', lambda *args:(None,history_record))
    receipts = {}
    for number, status in [(1,'complete'),(2,'content_failure')]:
        dest = original/f'semantic_page_{number}'/'result.json'
        profile.b.final.immutable(dest,dict(status=status,reason=None,wall_seconds=3.,
                                 usage=dict(completion_tokens=9)))
        receipts[f'semantic_page_{number}'] = profile.vr.sha(dest)
    profile.b.final.immutable(original/'result.json',dict(id=q,signature=stamp,status='content_failure',files=receipts))
    submitted = []
    class Budget:
        limits=dict(output_tokens_per_sample=16384,max_requests_per_sample=12,wall_seconds_per_sample=1800)
        def call(self, folder, messages, cap, validator, history, started):
            path = folder/'result.json'
            if not path.exists():
                submitted.append(folder.name)
                profile.b.final.immutable(path,dict(status='complete',reason=None,wall_seconds=2.,
                                          usage=dict(completion_tokens=14)))
            return profile.vr.read(path)
    idx=semantic.Index.__new__(semantic.Index)
    idx.ids=[str(i) for i in range(729)]
    idx.cards={i:('title','summary') for i in idx.ids}
    result=profile.profile({'id':q},arm,idx,None,Budget(),stamp)
    assert submitted==['semantic_page_3','semantic_page_4']
    assert result['cards_exposed']==729 and result['cards_validly_ranked']==729-256
    assert result['first_pass_seconds']==10. and result['additional_calls']==2
    assert profile.vr.read(original/'result.json')['status']=='content_failure'
