"""Regression checks for restoring the actual recorded old99 protocol."""
import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import run_aligned_organization as run
from tests.test_organization_comparison import ROW, QUERIES, Tokenizer


LEGACY = run.ROOT / 'results/organization_bounded99_20261103_v15'


@pytest.mark.skipif(not (LEGACY / 'inputs.json').exists(), reason='Private historical run unavailable')
def test_all_396_first_check_requests_reconstructed_exactly():
    run.assert_legacy(LEGACY)
    value = run.replay_request_contract(LEGACY)
    assert value['checked'] == 396 and value['all_messages_exact']


def test_pure_still_has_prepare_and_charges_its_usage():
    row = copy.deepcopy(ROW)
    packet = run.base.p.reference_packet(row, 'pure', QUERIES, None, lambda m: len(str(m)))
    evidence = dict(usage=dict(prompt_tokens=123, completion_tokens=45), turn=1)
    with run.base.checking(row, 'pure', packet, dict(evidence=evidence)):
        state = run.base.r.initial_state()
    assert len(state['messages']) == 4
    assert state['messages'][:2] == run.base.p.query_messages(row)
    assert state['messages'][2]['role'] == 'assistant'
    assert state['turn'] == 1 and state['input_tokens'] == 123
    assert state['reserved_output_tokens'] == 45
    assert state['read_rule_ids'] == [] and packet['references'] == []


def test_formal_checks_cannot_bypass_unresolved_source_and_bench_release(tmp_path):
    with pytest.raises(ValueError, match='not released'):
        run.run(tmp_path, 'formal')


def test_case_build_exact_original_prompt_ignores_old_verdict_and_reference():
    row = dict(ROW, answer='SECRET_REFERENCE', semantic_audit={'status': 'SECRET_OLD_LABEL'})
    journal = run.base.ExperienceJournal.__new__(run.base.ExperienceJournal)
    actual = journal.messages('extract_skill', {'row': row})
    assert actual == run.base.p.build_messages(ROW)
    assert 'SECRET_' not in str(actual)


def test_resume_reuses_success_and_rejects_tampered_original(tmp_path, monkeypatch):
    row = run.base.p.source_payload(ROW)
    run.vr.atomic(tmp_path / 'sources.json', [row])
    calls = []
    class Journal:
        def __init__(self, folder, config, tokenizer):
            self.folder = folder
            p = folder / 'request_index.json'
            self.index = run.vr.unseal(p) if p.exists() else dict(binding='frozen', files={})
        def __call__(self, stage, key, payload):
            calls.append(payload)
            original = self.folder / 'raw.json'
            run.vr.atomic(original, dict(id=key, original=True))
            self.index['files'][str(original)] = run.vr.sha(original)
            return dict(answer_status='has_error', case_note='The force-to-mass division is wrong.', skills=[], uncertainties=[])
    monkeypatch.setattr(run.base, 'ExperienceJournal', Journal)
    plan = dict(configuration={}, configuration_sha256='frozen')
    first = run.build_cases(tmp_path, plan, Tokenizer(), pilot=True)
    assert len(calls) == 1 and first[0]['source_ids'] == [row['id']]
    assert run.build_cases(tmp_path, plan, Tokenizer(), pilot=True) == first
    assert len(calls) == 1
    run.vr.atomic(tmp_path / 'experience/raw.json', dict(original=False))
    with pytest.raises(ValueError):
        run.build_cases(tmp_path, plan, Tokenizer(), pilot=True)
    assert len(calls) == 1


def test_bounded_skills_consume_case_note_and_preserve_original_source_identity(tmp_path, monkeypatch):
    row = run.base.p.source_payload(ROW)
    run.vr.atomic(tmp_path / 'sources.json', [row])
    case = dict(id='case_one', kind='case', title='source', text='Original answer plus NEW_120B_NOTE', source_ids=[row['id']])
    calls = []
    def call(self, stage, key, payload):
        calls.append((key, self.messages(stage, payload)))
        return dict(skills=[dict(title='Force check', text='Independently divide the net force by mass.')])
    monkeypatch.setattr(run.bench.ProcedureJournal, '__call__', call)
    plan = dict(configuration=dict(model={'max_tokens': 16384}, budget_limits=dict(requests=10, input_tokens=10000, output_tokens=10000),
                                   prior_usage=run.base.old.ZERO_USAGE), configuration_sha256='frozen')
    docs = run.build_skills(tmp_path, plan, Tokenizer(), [case], pilot=True)
    assert calls == [(row['id'], run.bounded.skill_messages(case))]
    assert docs[0]['source_ids'] == [row['id']]
    assert run.build_skills(tmp_path, plan, Tokenizer(), [case], pilot=True) == docs
    assert len(calls) == 1
    with pytest.raises(ValueError, match='Skill source drift'):
        run.build_skills(tmp_path, plan, Tokenizer(), [dict(case, text='CHANGED_NOTE')], pilot=True)
