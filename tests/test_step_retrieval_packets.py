from unittest.mock import patch

from scripts import step_retrieval_packets as trial


def test_first_check_caps_match_and_selection_is_charged_once():
    limits = dict(output_tokens_per_sample=16384, input_tokens_per_sample=600000,
                  max_requests_per_sample=12, wall_seconds_per_sample=1800)
    prepared = dict(wall_seconds=4, usage=dict(completion_tokens=300))
    charges = dict(seconds=37, requests=8, prompt_tokens=12000, completion_tokens=3980)
    left = trial.check_limits(limits, prepared, charges, 'bm25')
    right = trial.check_limits(limits, prepared, charges, 'fusion')
    assert left['output_tokens_per_sample'] - 300 == right['output_tokens_per_sample'] - 300 == 12104
    assert right['max_requests_per_sample'] == 4
    assert right['wall_seconds_per_sample'] == 1759
    assert right['input_tokens_per_sample'] == 588000
    assert left['max_requests_per_sample'] == 12


def test_oversize_candidate_does_not_displace_later_complete_skill():
    class Store:
        def view(self, key):
            return dict(id=key, title=key, text='x' * (8300 if key=='long' else 20))
    class Index:
        cards = {'long','short'}
        store = Store()
    with patch.object(trial.frozen.step,'ORIGINAL_CONTINUATION',lambda row,q,refs:refs):
        result = trial.packet({},'flat_skills',{'queries':{}},['long','short'],Index(),
                              lambda refs:sum(len(r['text']) for r in refs),
                              dict(context_window=131072,output_tokens_per_sample=16384))
    assert result['retrieval']['provided_complete_ids']==['short']
    assert result['references']==result['messages']
