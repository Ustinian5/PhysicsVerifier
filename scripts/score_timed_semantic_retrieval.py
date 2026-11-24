"""Score a complete paired prefix of the 50-question retrieval timing trial on node14."""
import argparse
from pathlib import Path
from statistics import mean

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


def checked_result(folder, q, stamp):
    path = folder/'result.json'
    r = vr.read(path)
    require(r['signature'] == stamp and r['id'] == q, 'Unbound sample outcome')
    require(len(r['provided']) <= 6 and r['full_tokens'] <= 8192, 'Full-text budget exceeded')
    require(abs(r['prepare_seconds']+r['lookup_seconds']+r['check_seconds']-r['pipeline_seconds']) < 1e-7,
            'Timing does not sum')
    for key, digest in r['files'].items():
        entry = folder/'selection.json' if key == 'selection' else folder/key/'result.json'
        require(vr.sha(entry) == digest, 'Stage evidence changed')
        if key != 'selection':
            record = vr.read(entry)
            require(all(vr.sha(Path(name)) == checksum for name, checksum in record['files'].items()),
                    'Raw stage evidence changed')
            if key == 'check':
                require((record['status'], record['value']) == (r['status'], r['value']), 'Wrong final prediction')
            else:
                stat = r['stage_times'][key]
                require((stat['prompt_tokens'], stat['output_tokens']) ==
                        (record['usage']['prompt_tokens'], record['usage']['completion_tokens']),
                        'Token accounting changed')
    require('check' in r['files'] or r['status'] != 'complete', 'Missing terminal CHECK')
    return r, vr.sha(path)


def score(prefix):
    root = b.RESULTS
    out = root/cohort.RESULT
    design_path = out/timing.NAME/'design.json'
    stamp = vr.digest(vr.read(design_path))
    registry = vr.read(out/'cohort.json')
    require(vr.sha(out/'inputs.json') == registry['inputs_sha256'] and vr.digest(registry['ids']) == cohort.IDS
            and vr.sha(b.FORMAL/'labels_private.json') == registry['labels_sha256'], 'Cohort or labels changed')
    ids = registry['ids'][:prefix]
    labels = vr.read(b.FORMAL/'labels_private.json')
    truth = {q: labels[q]['verdict'] for q in ids}
    controls, baseline = baseline_scores(out, ids, truth)
    groups, checkpoints = {}, {}
    for arm in cohort.ARMS:
        folder = out/timing.NAME/arm
        rows, hashes = {}, {}
        for q in ids:
            rows[q], hashes[q] = checked_result(folder/'rows'/q, q, stamp)
        checkpoint = vr.read(folder/f'timing_{prefix}.json')
        require(checkpoint == timing.timing_prefix(list(rows.values())), 'Timing checkpoint mismatch')
        checkpoints[arm] = checkpoint
        scores = b.final.measure(truth, rows)
        groups[arm] = dict(scores=scores, same_question_changes=paired(truth, controls[arm], rows),
                           delta_f1=scores['original_binary_view']['f1']-baseline[arm]['original_binary_view']['f1'],
                           whole_material_coverage=sum(bool(r['provided']) for r in rows.values()),
                           mean_preview_tokens=mean(r['preview_tokens'] for r in rows.values()),
                           mean_delivered_tokens=mean(r['full_tokens'] for r in rows.values()), results_sha256=hashes)
    denominator = 468
    report = dict(n=prefix, ids=ids, positive_class='incorrect', human_gold=False, design_sha256=vr.sha(design_path),
                  cohort_sha256=vr.sha(out/'cohort.json'), aligned_control_sha256=vr.sha(out/'bm25_aligned/metrics.json'),
                  groups=groups, bm25=baseline, timing=checkpoints,
                  estimate_questions=denominator,
                  three_dedicated_gpus_after_frozen_prepare_seconds=denominator*max(
                      c['means']['after_prepare_seconds'] for c in checkpoints.values()),
                  three_dedicated_gpus_including_prepare_seconds=denominator*max(
                      c['means']['pipeline_seconds'] for c in checkpoints.values()))
    b.final.immutable(out/timing.NAME/f'metrics_{prefix}.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=int, choices=(5, 10, 50))
    b.node.node()
    score(parser.parse_args().prefix)
