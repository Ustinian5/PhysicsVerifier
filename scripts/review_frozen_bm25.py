"""Use the published step-label and location scoring protocol for the frozen-tree comparison."""
import argparse
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from scripts import review_aligned_step_revision as review
from scripts import run_frozen_step_bm25 as step

vr, require = step.vr, step.require
ROOT = step.ROOT
OUT = ROOT / 'offline_review'


@contextmanager
def verified_run():
    original = review.run.verify
    def verify(path):
        return step.verify(path) if Path(path).resolve() == ROOT.resolve() else original(path)
    with patch.object(review.run, 'verify', verify):
        yield


def prepare():
    if (OUT / 'review_frozen.json').exists():
        review.verify(ROOT, complete=True)
        return
    review.prepare(ROOT)
    path = step.PREVIOUS / 'offline_review/evidence.json'
    expected = vr.read(step.PREVIOUS / 'offline_review/review_frozen.json')['sha256']['evidence.json']
    require(vr.sha(path) == expected, 'Previously reviewed diagnostics changed')
    old = vr.read(path)['records']
    evidence = vr.read(OUT / 'evidence.json')
    for record in evidence['records']:
        if record['disposition'] != 'pending':
            continue
        match = review.reusable_decision(old, record['id'], record['original_diagnostic'])
        if match is not None:
            record.update({k: match[k] for k in (*review.FIELDS, 'rationale')}, reused_from=dict(
                path=str(path), sha256=expected, review_id=match['review_id'], condition='Identical complete diagnosis'))
    vr.atomic(OUT / 'evidence.json', evidence)
    _, _, labels, records, _ = review.verify(ROOT)
    pending = []
    for r in records:
        if r['disposition'] != 'pending':
            continue
        pending.append(dict(review_id=r['review_id'], id=r['id'], diagnostic=r['original_diagnostic'],
                            spans=r['automatic_spans'], roots=[v for v in labels.values() if v['id'] == r['id']]))
    vr.atomic(OUT / 'pending_blind.json', pending)
    return len(pending)


def score():
    review.freeze(ROOT)
    review.score(ROOT)
    raw = vr.read(OUT / 'metrics.json')
    groups = {}
    for arm, g in raw['groups'].items():
        included = [row for row in g['details'] if row['id'] not in step.sources.STEP_HOLD]
        require(len(included) == 99 and sum(row['gt'] for row in included) == 163, 'Step denominator changed')
        totals = {k: sum(row[k] for row in included) for k in ('hits', 'published', 'locatable', 'gt', 'failed')}
        groups[arm] = dict(**totals, **review.old.rates(totals['hits'], totals['locatable'], totals['gt']),
                           unlocatable=totals['published'] - totals['locatable'],
                           status_counts=dict(Counter(row['source_status'] for row in included)),
                           full_100_reference_roots=164)
    step.write(ROOT / 'metrics_99.json', dict(registered_questions=100, compared_questions=99,
               registered_roots=164, compared_roots=163, groups=groups, scope_hold=sorted(step.sources.STEP_HOLD),
               full_coverage_sha256=vr.sha(OUT / 'metrics.json'), human_gold=False, scoring_model_calls=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'apply', 'score'))
    parser.add_argument('--decisions', type=Path)
    args = parser.parse_args()
    with step.base.r.run_lock(ROOT / 'review.lock'), verified_run():
        if args.action == 'prepare':
            print('Pending blind reviews:', prepare())
        elif args.action == 'apply':
            require(args.decisions is not None, 'Supply decisions keyed by blind review ID')
            review.apply(ROOT, vr.read(args.decisions))
        else:
            score()


if __name__ == '__main__':
    main()
