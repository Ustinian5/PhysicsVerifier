"""Bind explicit adjudications and a balanced quota amendment to a single review."""
from collections import Counter
from pathlib import Path

from rule_framework import final_answer_revision as protocol
from rule_framework import verified_resume as vr
from scripts import final_answer_replay_binding as replay

AMENDMENT = 'cohort_amendment.json'
RUNNER_WITH_COHORT_RELEASE = '41f6bd2836a55f1e28a6f8e9008507466c4fc292b7e0e0d21836ed147c7624a9'
POLICY = 'largest_balanced_subset_in_original_order_capped_at_original_quota'


def replay_sources(live):
    # The requests and accepted pilot predate the cohort-only release change.
    # The previous reader still checks its own receipt and every other source.
    if live[replay.RUNNER] == RUNNER_WITH_COHORT_RELEASE:
        return {**live, replay.RUNNER: replay.REPLAY_RUNNER}
    return live


def read_amendment(out):
    data = vr.read(out / AMENDMENT)
    protocol.require(data['policy'] == POLICY and data['human_gold'] is False
                     and data['reviewer'] == 'Codex assistant', 'Invalid cohort amendment')
    for name in ('candidates', 'assistant_review', 'review_status'):
        protocol.require(data[name + '_sha256'] == vr.sha(out / (name + '.json')), 'Cohort amendment parent changed: ' + name)
    protocol.require(data['code_sha256'] == vr.sha(Path(__file__)), 'Cohort amendment code changed')
    return data


def additional_decisions(out, candidates):
    if not (out / AMENDMENT).is_file():
        return []
    data = read_amendment(out)
    pending = {row['id']: row for row in vr.read(out / 'review_status.json')['pending']}
    decisions = data['decisions']
    ids = [d['id'] for d in decisions]
    protocol.require(len(ids) == len(pending) and set(ids) == set(pending), 'Adjudications do not cover exactly the pending questions')
    for sid in ids:
        row = candidates[sid]
        reviewed = vr.read(out / 'review' / sid / 'result.json')
        original = protocol.decision(row, reviewed)
        protocol.require(original['state'] == 'pending_adjudication' and original == pending[sid],
                         'Adjudication does not match the original independent review: ' + sid)
    return decisions


def release_quota(out, dispositions, requested):
    path = out / AMENDMENT
    if not path.is_file():
        return requested, {}
    data = read_amendment(out)
    counts = Counter(d['verdict'] for d in dispositions.values() if d['state'] == 'eligible')
    protocol.require(not any(d['state'].startswith('pending') for d in dispositions.values()),
                     'Adjudications incomplete')
    quota = min(requested, counts['correct'], counts['incorrect'])
    protocol.require(0 < quota < requested and data['requested_per_class'] == requested
                     and data['eligible_counts'] == dict(counts) and data['effective_per_class'] == quota,
                     'Quota differs from the largest balanced eligible cohort')
    return quota, dict(quota_amendment=dict(sha256=vr.sha(path), requested_per_class=requested,
                                           effective_per_class=quota, eligible_counts=dict(counts), policy=POLICY))
