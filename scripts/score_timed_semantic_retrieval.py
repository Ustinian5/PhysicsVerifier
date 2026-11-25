"""Score a complete paired prefix of the 50-question retrieval timing trial on node14."""
import argparse
from pathlib import Path
from statistics import mean
import time

from scripts import run_timed_semantic_retrieval as timing

b, vr, require, cohort = timing.b, timing.vr, timing.require, timing.cohort


def paired(truth, original, new):
    difference = dict(improved=[], regressed=[], same=[], outside_common_binary=[])
    for q, label in truth.items():
        def binary(record):
            return (record.get('value') or {}).get('verdict') if record['status'] == 'complete' else None
        left, right = binary(original[q]), binary(new[q])
        if left not in ('correct', 'incorrect') or right not in ('correct', 'incorrect'):
            key = 'outside_common_binary'
        elif left == right:
            key = 'same'
        else:
            key = 'improved' if right == label else 'regressed'
        difference[key].append(q)
    return difference


def baseline_scores(out, ids, truth):
    published = vr.read(out/'bm25_baseline.json')
    aligned = vr.read(out/'bm25_aligned/metrics.json')
    require(aligned['cohort'] == published['cohort'] == cohort.IDS
            and vr.sha(out/'bm25_aligned/metrics.json')=='379519ebeddb39fbca9d9dcbc9b9cd7bbcbfd842d0106c7853fef4892490b52e'
            and vr.sha(out/'bm25_baseline.json') == aligned['parent_sha256'], 'Unbound baseline')
    rows = published['rows']
    replacement_id, = aligned['changed']
    replacement = aligned['replacement']
    require(vr.sha(out/'bm25_aligned/rows'/replacement_id/'check/result.json') == replacement['file_sha256'],
            'Unbound tree control')
    rows['rule_tree'][replacement_id] = replacement
    selected = {arm: {q: group[q] for q in ids} for arm, group in rows.items()}
    scores = {arm: b.final.measure(truth, group) for arm, group in selected.items()}
    if len(ids) == 50:
        require(scores == aligned['groups'], 'Corrected control score mismatch')
    return selected, scores


def completed_index(rows, arm):
    if arm == 'rule_tree':
        return dict(routed_to_a_final_selection=sum('selection' in r['files'] for r in rows.values()))
    size = {'case_rag':729, 'flat_skills':1452}[arm]
    sizes = [timing.semantic.FIRST_PAGE_SIZE]
    remaining = size-sizes[0]
    while remaining:
        page = min(remaining, timing.semantic.PAGE_SIZE)
        sizes.append(page)
        remaining -= page
    exposure = {}
    for q, row in rows.items():
        stages = row['stage_times']
        seen = sum(n for i,n in enumerate(sizes,1) if f'semantic_page_{i}' in stages)
        valid = sum(n for i,n in enumerate(sizes,1)
                    if stages.get(f'semantic_page_{i}',{}).get('status') == 'complete')
        exposure[q] = dict(seen=seen, successfully_ranked=valid)
    return dict(size=size, fully_seen=sum(v['seen']==size for v in exposure.values()),
                fully_ranked=sum(v['successfully_ranked']==size for v in exposure.values()), by_question=exposure)


def matched_times(outputs):
    common = set.intersection(*(set(q for q,r in group.items() if 'check' in r['files']) for group in outputs.values()))
    return dict(n=len(common), ids=sorted(common), means={arm:{
        key:mean(group[q][key] for q in sorted(common)) for key in ('lookup_seconds','pipeline_seconds')
        } for arm,group in outputs.items()} if common else {})


def retrieval_stats(rows, folder):
    first_pass = {}
    clipped, ranking_calls, selected, new_calls = 0, 0, 0, 0
    for q,r in rows.items():
        stats = r['stage_times']
        first_pass[q] = sum(d['seconds'] for key,d in stats.items()
                            if key.startswith('semantic_') and key != 'semantic_merge')
        clipped += sum(v['encoding'].get('reached_analysis_limit',False) for v in stats.values())
        ranking_calls += len(stats)
        new_calls += r['model_calls']-1  # PREPARE was replayed from the frozen benchmark.
        if 'selection' in r['files']:
            trace = vr.read(folder/'rows'/q/'selection.json')['trace']
            selected += bool(trace['semantic'])
    return dict(first_pass_seconds=first_pass, mean_first_pass_seconds=mean(first_pass.values()),
                selector_calls=ranking_calls, selectors_at_analysis_boundary=clipped,
                nonempty_semantic_rankings=selected, fresh_model_calls=new_calls)


def score(prefix):
    from transformers import AutoTokenizer
    root = b.RESULTS
    out = root/cohort.RESULT
    dest = out/timing.NAME
    stamp = timing.design(root)
    registry = vr.read(out/'cohort.json')
    require(vr.sha(b.FORMAL/'labels_private.json') == registry['labels_sha256'], 'Frozen labels changed')
    if prefix == 50:
        timing.expansion_gate(dest,stamp)
    ids = registry['ids'][:prefix]
    masked = vr.read(out/'inputs.json')[:prefix]
    require([r['id'] for r in masked] == ids, 'Input order changed')
    labels = vr.read(b.FORMAL/'labels_private.json')
    truth = {q: labels[q]['verdict'] for q in ids}
    controls, baseline = baseline_scores(out, ids, truth)
    old = b.final.execution_plan(b.FORMAL)
    tok = AutoTokenizer.from_pretrained(b.host.HOT/'models/gpt-oss-120b', local_files_only=True)
    groups, checkpoints, outputs = {}, {}, {}
    for arm in cohort.ARMS:
        folder = dest/arm
        index = timing.base.load_index(root,arm)
        plan = timing.runtime.checking_plan(arm,stamp)
        rows = {row['id']:timing.evaluate(row,arm,folder,old,plan,index,tok,None,stamp,replay=True) for row in masked}
        hashes = {q:vr.sha(folder/'rows'/q/'result.json') for q in rows}
        checkpoint = vr.read(folder/f'timing_{prefix}.json')
        require(checkpoint == timing.timing_prefix(list(rows.values())), 'Timing checkpoint mismatch')
        checkpoints[arm] = checkpoint
        outputs[arm] = rows
        scores = b.final.measure(truth, rows)
        groups[arm] = dict(scores=scores, same_question_changes=paired(truth, controls[arm], rows),
                           delta_f1=scores['original_binary_view']['f1']-baseline[arm]['original_binary_view']['f1'],
                           whole_material_coverage=sum(bool(r['provided']) for r in rows.values()),
                           mean_preview_tokens=mean(r['preview_tokens'] for r in rows.values()),
                           mean_delivered_tokens=mean(r['full_tokens'] for r in rows.values()),
                           reached_check=sum('check' in r['files'] for r in rows.values()),
                           index_completion=completed_index(rows,arm), results_sha256=hashes,
                           retrieval=retrieval_stats(rows,folder))
    denominator = 468
    report = dict(n=prefix, ids=ids, positive_class='incorrect', human_gold=False, design_sha256=vr.sha(dest/'design.json'),
                  cohort_sha256=vr.sha(out/'cohort.json'), aligned_control_sha256=vr.sha(out/'bm25_aligned/metrics.json'),
                  groups=groups, bm25=baseline, timing=checkpoints, common_reached_check=matched_times(outputs),
                  estimate_questions=denominator,
                  three_dedicated_gpus_after_frozen_prepare_seconds=denominator*max(
                      c['means']['after_prepare_seconds'] for c in checkpoints.values()),
                  three_dedicated_gpus_including_prepare_seconds=denominator*max(
                      c['means']['pipeline_seconds'] for c in checkpoints.values()))
    target = dest/f'metrics_{prefix}.json'
    b.final.immutable(target, report)
    if prefix==10:
        for arm,g in groups.items():
            require(g['reached_check']==10 and g['index_completion'].get(
                'fully_ranked',g['index_completion'].get('routed_to_a_final_selection'))==10,
                'Ranking repair did not cover the frozen ten: '+arm)
        b.final.immutable(dest/'acceptance_10.json',dict(passed=True,signature=stamp,cohort=cohort.IDS,
            ids=ids,accepted_rows={a:g['results_sha256'] for a,g in groups.items()},metrics_sha256=vr.sha(target),
            tests='Same ten in all arms; source and raw-request replay, page coverage, legal unique IDs, tree quotas, original final CHECK, cumulative budgets; no score criterion.',
            scoring_model_calls=0,scorer_sha256=vr.sha(__file__)))
    return report


def after_attempt(prefix, attempt):
    dest = b.RESULTS/cohort.RESULT/timing.NAME
    folders = [dest/a/'attempts'/str(attempt) for a in cohort.ARMS]
    status = dest/f'scoring_{prefix}_attempt_{attempt}.json'
    with b.final.base.r.run_lock(dest/f'scoring_{prefix}.lock'):
        try:
            while True:
                exits = [vr.read(f/'exit.json') if (f/'exit.json').is_file() else None for f in folders]
                require(all(e is None or e['code']==0 for e in exits), 'A generating worker exited with a failure')
                if all((f/f'wall_{prefix}.json').is_file() for f in folders) and (prefix==5 or all(exits)):
                    break
                time.sleep(8)
            if prefix!=5:
                for f in folders:
                    require(vr.read(f/'complete.json')['n']==prefix and
                            vr.read(f/'release.json')['gauges']=={'running':0,'waiting':0},
                            'Missing completion or release evidence')
            score(prefix)
            vr.atomic(status,dict(code=0,metrics_sha256=vr.sha(dest/f'metrics_{prefix}.json')))
        except Exception as exc:
            vr.atomic(status,dict(code=1,error=str(exc)))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=int, choices=(5, 10, 50))
    parser.add_argument('--after-attempt', type=int)
    b.node.node()
    args = parser.parse_args()
    if args.after_attempt is None:
        score(args.prefix)
    else:
        after_attempt(args.prefix,args.after_attempt)
