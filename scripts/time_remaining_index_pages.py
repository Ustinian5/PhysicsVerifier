"""Measure the full flat scan on ten frozen questions, including pages after a ranking failure.

The scored decision and all prior response attempts remain immutable. Additional calls only
visit previously unseen pages. No decision is made from a partially validated ranking.
"""
import argparse
import time
from pathlib import Path
from statistics import mean

import httpx

from scripts import run_timed_semantic_retrieval as timed
from scripts import timed_retrieval_runtime as runtime

b, vr, require, cohort = timed.b, timed.vr, timed.require, timed.cohort


def pages(ids):
    first, later = timed.semantic.FIRST_PAGE_SIZE, timed.semantic.PAGE_SIZE
    return [ids[:first], *(ids[n:n+later] for n in range(first, len(ids), later))]


def signature(arm_dir):
    return vr.digest([vr.sha(__file__), vr.sha(arm_dir.parent/'design.json'),
                      vr.sha(arm_dir/'timing_10.json'), vr.sha(arm_dir/'attempts/2/exit.json')])


def profile(row, arm_dir, index, old, budget, stamp):
    q = row['id']
    original = arm_dir/'rows'/q
    parent = vr.read(original/'result.json')
    require(parent['id'] == q and parent['signature'] == stamp, 'Wrong primary outcome')
    directory = arm_dir/'index_profile/rows'/q
    target = directory/'profile.json'
    if target.exists():
        return vr.read(target)
    _, prep = b.prefix_record(b.FORMAL, old, 'formal', row)
    history = [prep]
    query = (prep['value'] or {}).get('queries', [])
    covered, ranked, duration, new_calls = 0, 0, 0., 0
    records = {}
    limits = budget.limits
    scope = pages(index.ids)
    reason = 'all_index_pages_presented'
    if prep['status'] != 'complete':
        reason = 'shared_prepare_failure'
        scope = []
    for n, ids in enumerate(scope, 1):
        key = 'semantic_page_'+str(n)
        old_stage = original/key
        previous = old_stage.joinpath('result.json').exists()
        folder = old_stage if previous else directory/key
        remaining = limits['output_tokens_per_sample'] - sum(x['usage']['completion_tokens'] for x in history)
        cap = min(2048, remaining-timed.RESERVED_CHECK_OUTPUT)
        if cap <= 0 or len(history)+1 >= limits['max_requests_per_sample']:
            reason = 'original_cumulative_budget_exhausted'
            break
        if not previous:
            require(parent['status'] != 'complete' and 'check' not in parent['files']
                    and any(x['status'] != 'complete' for x in history) and not old_stage.exists(),
                    'Refusing a duplicate or unaccounted stage')
        started = time.monotonic()

        def observe(key_unused, prompt, payload, validator):
            nonlocal covered, ranked, duration, new_calls
            messages = timed.model_text(prompt, payload)
            record = budget.call(folder, messages, cap, validator, history, started)
            digest = vr.sha(folder/'result.json')
            if previous:
                require(parent['files'][key] == digest, 'Primary page response changed')
            history.append(record)
            covered += len(ids)
            ranked += len(ids) if record['status'] == 'complete' else 0
            duration += record['wall_seconds']
            new_calls += int(not previous)
            records[key] = dict(source='original' if previous else 'timing_only',
                    sha256=digest, material_count=len(ids), status=record['status'], reason=record['reason'],
                    wall_seconds=record['wall_seconds'], usage=record['usage'])
            return {'ids': []}

        index.select(ids, query, observe, key, timed.semantic.TOP_K)
    require(covered <= len(index.ids) and sum(x['wall_seconds'] for x in history) <= limits['wall_seconds_per_sample'],
            'Cumulative scan budget exceeded')
    result = dict(id=q, groupsize=len(index.ids), cards_exposed=covered, cards_validly_ranked=ranked,
                  first_pass_seconds=duration, additional_calls=new_calls, stages=records, reason=reason,
                  primary_sha256=vr.sha(original/'result.json'))
    b.final.immutable(target, result)
    return result


def run(folder, arm, arm_dir, receipt):
    from transformers import AutoTokenizer
    primary_stamp = vr.digest(vr.read(arm_dir.parent/'design.json'))
    ids = vr.read(arm_dir/'timing_10.json')['question_ids']
    rows = vr.read(arm_dir.parent.parent/'inputs.json')[:10]
    require([row['id'] for row in rows] == ids and vr.read(arm_dir/'attempts/2/exit.json')['code'] == 0
            and (arm_dir/'attempts/2/release.json').is_file(), 'Primary measurement still active')
    old = b.final.execution_plan(b.FORMAL)
    index = timed.base.load_index(b.RESULTS, arm)
    tok = AutoTokenizer.from_pretrained(b.host.HOT/'models/gpt-oss-120b', local_files_only=True)
    directory = arm_dir/'index_profile'
    plan = runtime.checking_plan(arm, primary_stamp)
    totals = []
    with runtime.large_service(folder, arm, receipt, directory/'STOP'), httpx.Client(trust_env=False) as client:
        budget = timed.base.previous.Budget(directory, plan, tok, client)
        for n, row in enumerate(rows, 1):
            require(not (directory/'STOP').exists(), 'STOP requested')
            totals.append(profile(row, arm_dir, index, old, budget, primary_stamp))
            vr.atomic(directory/'progress.json', dict(n=n, target=10, unix=time.time()))
            if n in (5, 10):
                costs = [r['first_pass_seconds'] for r in totals if r['cards_exposed'] == len(index.ids)]
                b.final.immutable(directory/f'scan_{n}.json', dict(n=n, exposed_entire_index=len(costs),
                    mean_full_scan_seconds=mean(costs) if costs else None,
                    mean_all_attempt_seconds=mean(r['first_pass_seconds'] for r in totals),
                    all_pages_ranked=sum(r['cards_validly_ranked']==len(index.ids) for r in totals),
                    extra_calls=sum(r['additional_calls'] for r in totals), by_id={r['id']:r for r in totals},
                    primary_design_sha256=vr.sha(arm_dir.parent/'design.json')))
    b.final.immutable(folder/'complete.json', dict(receipt=receipt, finished_unix=time.time()))


def report():
    directory = b.RESULTS/cohort.RESULT/timed.NAME
    scores = vr.read(directory/'metrics_10.json')
    ids = scores['ids']
    require(len(ids)==10 and scores['design_sha256']==vr.sha(directory/'design.json'), 'Different ten-question run')
    summary, evidence = {}, {'metrics':vr.sha(directory/'metrics_10.json')}
    comparable = set(ids)
    for arm in cohort.ARMS:
        group = directory/arm
        records = {q:vr.read(group/'rows'/q/'result.json') for q in ids}
        if arm == 'rule_tree':
            scan = {q:sum(v['seconds'] for k,v in r['stage_times'].items() if k.startswith('semantic_'))
                    for q,r in records.items()}
            valid_paths = sum('selection' in r['files'] for r in records.values())
            comparable.intersection_update(q for q,r in records.items() if 'selection' in r['files'])
            exposed = None
        else:
            path = group/'index_profile/scan_10.json'
            data = vr.read(path)
            evidence[arm] = vr.sha(path)
            require(list(sorted(data['by_id']))==sorted(ids), 'Different profile cohort')
            for q, r in data['by_id'].items():
                require(r['primary_sha256'] == vr.sha(group/'rows'/q/'result.json'), 'Changed primary result')
                for key, v in r['stages'].items():
                    root = group if v['source']=='original' else group/'index_profile'
                    stage = root/'rows'/q/key/'result.json'
                    require(vr.sha(stage)==v['sha256'], 'Unbound page')
                    require(all(vr.sha(Path(n))==digest for n,digest in vr.read(stage)['files'].items()),
                            'Changed raw page evidence')
            scan = {q:v['first_pass_seconds'] for q,v in data['by_id'].items()}
            valid_paths = data['all_pages_ranked']
            exposed = data['exposed_entire_index']
            comparable.intersection_update(q for q,v in data['by_id'].items() if v['cards_exposed']==v['groupsize'])
        failed = [r for r in records.values() if 'check' not in r['files']]
        summary[arm] = dict(first_pass_seconds_by_id=scan, mean_first_pass_seconds=mean(scan.values()),
                            first_pass_valid_paths=valid_paths, fully_exposed_flat_indexes=exposed,
                            reached_check=len(ids)-len(failed), failure_before_check={r['id']:r['reason'] for r in failed},
                            scores=scores['groups'][arm]['scores'],
                            means=scored_means(records), lower_bound_468_scan_only_hours=468*mean(scan.values())/3600)
    b.final.immutable(directory/'timing_comparison_10.json', dict(
        ids=ids, evidence=evidence, groups=summary, common_reached_check=scores['common_reached_check'],
        complete_flat_exposure_and_tree_route=dict(ids=sorted(comparable), means={
            a:mean(s['first_pass_seconds_by_id'][q] for q in sorted(comparable)) for a,s in summary.items()
        } if comparable else {}),
        scope='Full-index exposure includes every original or timing-only page once. Tree includes traversed semantic nodes and candidate leaf ranking. Both include lexical lookup charged in the first stage. All-arm first-pass means include failed outputs. Scored-path means include early termination. A whole successful workflow requires a completed CHECK.',
        bm25_same_material=vr.read(directory.parent/'bm25_tree_top24.json')['summary']))


def scored_means(records):
    keys = ('lookup_seconds','check_seconds','pipeline_seconds')
    checked = [r for r in records.values() if 'check' in r['files']]
    return dict(all_terminal={key:mean(r[key] for r in records.values()) for key in keys},
                reached_check={key:mean(r[key] for r in checked) for key in keys} if checked else None)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('run','confirm','report'))
    p.add_argument('arm', choices=('case_rag','flat_skills'), nargs='?')
    p.add_argument('--attempt', type=int, default=1)
    args = p.parse_args()
    if args.action == 'report':
        b.node.node()
        report()
        raise SystemExit(0)
    require(args.arm is not None,'Flat arm required')
    arm_dir = b.RESULTS/cohort.RESULT/timed.NAME/args.arm
    receipt = signature(arm_dir)
    runtime.detached(arm_dir/'index_profile', args.attempt, receipt,
                     lambda folder:run(folder,args.arm,arm_dir,receipt), args.action=='confirm')
