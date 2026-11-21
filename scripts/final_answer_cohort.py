"""Bind explicit adjudications and a balanced quota amendment to a single review."""
from collections import Counter
from pathlib import Path

from rule_framework import final_answer_revision as protocol
from rule_framework import verified_resume as vr
from scripts import final_answer_replay_binding as replay

AMENDMENT = 'cohort_amendment.json'
RUNNER_WITH_COHORT_RELEASE = '41f6bd2836a55f1e28a6f8e9008507466c4fc292b7e0e0d21836ed147c7624a9'
RUNNER_WITH_LINKED_READS = '0f03d6ead5c5ea4995136fdc40d2ba66e18352c84ce1aecd04d6b6ada6fa96d7'
WITHDRAWN_AMENDMENT = '5ea5ed330bb74379e3f7d392e509cc0299ef8bf5f783e3a354a119842f5afb17'
WITHDRAWN_RELEASE = '39987c442ebdef6adcfa9d6e80645bc69d142b02a05292b2c92763bbbea3b4ed'
POLICY = 'largest_balanced_subset_in_original_order_capped_at_original_quota'


def replay_sources(live):
    # The requests and accepted pilot predate the cohort-only release change.
    # The previous reader still checks its own receipt and every other source.
    if live[replay.RUNNER] in (RUNNER_WITH_COHORT_RELEASE, RUNNER_WITH_LINKED_READS):
        return {**live, replay.RUNNER: replay.REPLAY_RUNNER}
    return live


def read_amendment(out):
    data = vr.read(out / AMENDMENT)
    protocol.require(data['policy'] == POLICY and data['human_gold'] is False
                     and data['reviewer'] == 'Codex assistant', 'Invalid cohort amendment')
    for name in ('candidates', 'assistant_review', 'review_status'):
        protocol.require(data[name + '_sha256'] == vr.sha(out / (name + '.json')), 'Cohort amendment parent changed: ' + name)
    archived = vr.sha(out / AMENDMENT) == WITHDRAWN_AMENDMENT
    expected_code = 'f04d9f8a0a6a46faac6a5fec00cbe61b091b4193443f9c070f9a095678cb2916' if archived else vr.sha(Path(__file__))
    protocol.require(data['code_sha256'] == expected_code, 'Cohort amendment code changed')
    if 'supersedes' in data:
        parent = out.parent / 'final_revision'
        protocol.require(out.name == 'final_revision_balanced' and vr.sha(parent / AMENDMENT) == WITHDRAWN_AMENDMENT
                         and vr.sha(parent / 'bench_release.json') == WITHDRAWN_RELEASE
                         and data['supersedes'] == WITHDRAWN_RELEASE, 'Superseded evidence changed')
        before = {d['id']: d for d in vr.read(parent / AMENDMENT)['decisions']}
        after = {d['id']: d for d in data['decisions']}
        changes = sorted(sid for sid in before if before[sid] != after.get(sid))
        protocol.require(set(before) == set(after) and changes == data['changed_ids'], 'Undeclared adjudication changes')
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


def reuse_prior_requests(out):
    """Reuse exact completed requests from the stopped release after the new IDs are frozen."""
    parent = out.parent / 'final_revision'
    data = read_amendment(out)
    protocol.require(data.get('supersedes') == WITHDRAWN_RELEASE and (parent / 'STOP').is_file(),
                     'No stopped parent release for reuse')
    attempt = parent / 'node14_formal/attempts/2'
    receipt = vr.read(attempt / 'release.json')
    protocol.require((attempt / 'runner.exit_code').is_file() and not receipt['owned_gpu_pids']
                     and not any(receipt['gauges'].values()), 'Parent release is still active')
    rows = vr.read(out / 'inputs.json')
    groups = ('queries', *protocol.memory.ARMS)
    linked = {group: 0 for group in groups}
    for row in rows:
        key = vr.digest(row['id'])[:24]
        for group in groups:
            source = parent / 'formal' / group / key
            target = out / 'formal' / group / key
            if not (source / 'result.json').is_file():
                continue
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source, target_is_directory=True)
            protocol.require(target.is_symlink() and target.resolve() == source.resolve(),
                             'Refusing to replace an existing group result')
            linked[group] += 1
    return dict(parent_benchmark_sha256=WITHDRAWN_RELEASE, benchmark_sha256=vr.sha(out/'bench_release.json'),
                reused_completed_results=linked)
