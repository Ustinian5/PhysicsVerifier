"""Compare the published 50-question fusion trial with its same-question BM25 controls."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import analyze_final_answer_sample_size as stability
from scripts import audit_final_answer_metrics as binary
from scripts import score_timed_semantic_retrieval as scored


def paired_f1_interval(labels, control, fusion):
    """Resample questions within each frozen class, keeping control and fusion together."""
    strata = stability.cohort_order(labels)
    draws = stability.BOOTSTRAP_DRAWS
    rng = np.random.default_rng([stability.SEED, len(labels)])
    totals = np.zeros((draws, 2, 2), dtype=np.int64)
    for label in stability.CLASSES:
        ids = strata[label]
        terms = np.empty((len(ids), 2, 2), dtype=np.int64)
        for i, q in enumerate(ids):
            for j, rows in enumerate((control, fusion)):
                counts = binary.measure({q: label}, {q: rows[q]})['counts']
                numerator = 2 * counts['TP']
                terms[i, j] = numerator, numerator + counts['FP'] + counts['FN']
        weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=draws)
        totals += (weights @ terms.reshape(len(ids), 4)).reshape(draws, 2, 2)
    num, den = totals[:, :, 0], totals[:, :, 1]
    f1 = np.divide(100.0 * num, den, out=np.zeros_like(num, dtype=float), where=den != 0)
    return np.quantile(f1[:, 1] - f1[:, 0], [0.025, 0.975]).tolist()


def score_subset(ids, labels, rows):
    return binary.measure({q: labels[q] for q in ids}, {q: rows[q] for q in ids})


def leave_one_out(labels, control, fusion):
    differences = {}
    for omitted in labels:
        keep = [q for q in labels if q != omitted]
        old = score_subset(keep, labels, control)['original_binary_view']['f1']
        new = score_subset(keep, labels, fusion)['original_binary_view']['f1']
        differences[omitted] = 100 * (new - old)
    return differences


def summarize(root):
    vr, require = scored.vr, scored.require
    trial = root / scored.cohort.RESULT
    dest = trial / scored.timing.NAME
    path = dest / 'metrics_50.json'
    result = vr.read(path)
    receipt = vr.read(dest / 'scoring_50_attempt_2.json')
    registry = vr.read(trial / 'cohort.json')
    label_path = root / scored.cohort.BENCH / 'labels_private.json'
    ids = result['ids']
    require(receipt['code'] == 0 and receipt['metrics_sha256'] == vr.sha(path)
            and result['cohort_sha256'] == vr.sha(trial / 'cohort.json')
            and registry['labels_sha256'] == vr.sha(label_path)
            and ids == registry['ids'] and result['n'] == 50 and result['positive_class'] == 'incorrect',
            'Expected the scored, frozen 50-question trial')
    frozen_labels = vr.read(label_path)
    labels = {q: frozen_labels[q]['verdict'] for q in ids}
    controls, baseline = scored.baseline_scores(trial, ids, labels)
    require(baseline == result['bm25'], 'Aligned controls differ from the published scores')

    groups = {}
    for arm in scored.cohort.ARMS:
        group = result['groups'][arm]
        rows, failures = {}, {}
        only_semantic = 0
        for q in ids:
            folder = dest / arm / 'rows' / q
            require(vr.sha(folder / 'result.json') == group['results_sha256'][q], 'Row changed: ' + q)
            row = rows[q] = vr.read(folder / 'result.json')
            if row['status'] != 'complete':
                failures[q] = dict(stage='CHECK' if 'check' in row['files'] else 'selection', reason=row['reason'])
            if 'selection' in row['files']:
                require(vr.sha(folder / 'selection.json') == row['files']['selection'], 'Selection changed: ' + q)
                trace = vr.read(folder / 'selection.json')['trace']
                semantic_only = set(trace['semantic']).difference(trace['lexical'])
                only_semantic += bool(semantic_only.intersection(row['provided']))
        old = controls[arm]
        changes = scored.paired(labels, old, rows)
        require(changes == group['same_question_changes'] and binary.measure(labels, rows) == group['scores'],
                'Paired rows do not reproduce the published report')
        common = [q for q in ids if q not in changes['outside_common_binary']]
        groups[arm] = dict(
            delta_f1_pp=100 * group['delta_f1'],
            paired_bootstrap_ci95_pp=paired_f1_interval(labels, old, rows),
            leave_one_out_delta_f1_pp=leave_one_out(labels, old, rows),
            changed_by_gold={k:dict(Counter(labels[q] for q in changes[k]))
                             for k in ('improved', 'regressed', 'outside_common_binary')},
            both_valid=dict(n=len(common), bm25=score_subset(common, labels, old),
                            fusion=score_subset(common, labels, rows)),
            expansion_40=dict(bm25=score_subset(ids[10:], labels, old), fusion=score_subset(ids[10:], labels, rows)),
            semantic_only_material_delivered_questions=only_semantic, content_failures=failures)
    return dict(metrics_sha256=vr.sha(path), analysis_sha256=vr.sha(__file__), positive_class='incorrect',
                bootstrap_draws=stability.BOOTSTRAP_DRAWS, bootstrap_seed=[stability.SEED, len(ids)],
                unit='one frozen question per registered family; paired within each answer class',
                model_calls=0, groups=groups)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='qwen30b_step_verifier results directory')
    root = parser.parse_args().root.resolve()
    output = root / scored.cohort.RESULT / scored.timing.NAME / 'paired_comparison_50.json'
    scored.b.final.immutable(output, summarize(root))
