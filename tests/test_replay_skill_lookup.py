import json
import pytest

from scripts import replay_skill_lookup as probe


def test_search_returns_counts_for_broad_queries_and_preserves_symbol_case():
    docs = {f'S{i}': {'title': 'optical', 'text': 'mW fiber'} for i in range(25)}
    result = probe.search(docs, [{'text': [['Fiber', '光纤'], ['optical']]}])
    assert result == {'counts': [25], 'total': 25, 'cards': []}
    assert probe.search(docs, [{'exact': [['MW']]}])['total'] == 0


def test_replay_binds_selected_full_text_to_a_displayed_id_and_accounts_for_titles():
    row = dict(id='dev', question='An optical fiber', prediction='answer', context='')
    queries = dict(queries=['fiber 光纤'])
    skills = [dict(id='a', title='Snell', text='Refraction law'), dict(id='b', title='other', text='other')]
    count = lambda messages: len(json.dumps(messages, ensure_ascii=False))
    cfg = dict(limits=dict(context_window=100000, output_tokens_per_sample=1000),
               contract=dict(reference_token_cap=8192, max_items=6))
    def run(selected):
        return probe.replay(row, queries, skills, [{'search': [{'text': [['Snell']]}]}, {'select': selected}],
                            count, cfg)
    out = run(['S1'])
    assert out['provided_ids'] == ['a'] and out['preview_tokens'] > 0 and out['full_material_tokens'] > 0
    assert run([])['provided_ids'] == []
    with pytest.raises(ValueError, match='undisplayed'):
        run(['S2'])
    cfg['contract']['reference_token_cap'] = out['preview_tokens']
    assert run(['S1'])['budget_skipped'] == ['S1']
