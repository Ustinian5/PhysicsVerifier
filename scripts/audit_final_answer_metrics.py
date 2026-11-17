"""Recompute frozen final-answer metrics without changing labels or predictions.

The original binary-output-only view is retained. Whole-cohort detection,
two-class metrics and constant-prediction controls are explicitly additional
views, never replacement ground truth or a way to tune a desired score.
"""
import argparse
from collections import Counter
import fcntl
import math
import os
from pathlib import Path
import time

from rule_framework import verified_resume as vr

ARMS = ('pure', 'case_rag', 'flat_skills', 'rule_tree')


def divide(a, b):
    return a / b if b else 0.0


def measure(labels, predictions):
    if set(labels) != set(predictions):
        raise ValueError('Missing or extra predictions')
    c = Counter(TP=0, FP=0, FN=0, TN=0, failed_positive=0, failed_negative=0,
                uncertain_positive=0, uncertain_negative=0)
    for sid, gold in labels.items():
        if gold not in {'correct', 'incorrect'}:
            raise ValueError('Nonbinary frozen label')
        pred = predictions[sid]
        positive = gold == 'incorrect'
        if pred['status'] != 'complete':
            c['failed_positive' if positive else 'failed_negative'] += 1
            continue
        value = pred['value']['verdict']
        if value == 'uncertain':
            c['uncertain_positive' if positive else 'uncertain_negative'] += 1
        elif value in {'correct', 'incorrect'}:
            c[{('incorrect', 'incorrect'): 'TP', ('correct', 'incorrect'): 'FP',
               ('incorrect', 'correct'): 'FN', ('correct', 'correct'): 'TN'}[gold, value]] += 1
        else:
            raise ValueError('Invalid final-answer verdict')
    tp, fp, fn, tn = (c[k] for k in ('TP', 'FP', 'FN', 'TN'))
    n = len(labels); valid = tp + fp + fn + tn
    positives = sum(v == 'incorrect' for v in labels.values()); negatives = n - positives
    p, r = divide(tp, tp + fp), divide(tp, tp + fn)
    f1 = divide(2 * tp, 2 * tp + fp + fn)
    nf1 = divide(2 * tn, 2 * tn + fp + fn)
    specificity = divide(tn, tn + fp)
    return dict(counts=dict(c), n=n, labeled_incorrect=positives, labeled_correct=negatives,
        valid_binary=valid, coverage=divide(valid, n),
        original_binary_view=dict(precision=p, recall=r, f1=f1),
        binary_supplement=dict(accuracy=divide(tp + tn, valid), specificity=specificity,
            false_positive_rate=divide(fp, tn + fp), balanced_accuracy=(r + specificity) / 2,
            correct_class_f1=nf1, macro_f1=(f1 + nf1) / 2,
            mcc=divide(tp * tn - fp * fn, math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))),
        whole_cohort_view=dict(precision=p, recall=divide(tp, positives),
            f1=divide(2 * tp, tp + fp + positives), successful_decision_rate=divide(tp + tn, n),
            specificity_success=divide(tn, negatives),
            note='Failures/abstentions remain separate, never recoded as correct. '
                 'Positive-label failures count as undetected in recall; no new positive prediction is fabricated.'))


def audit(root, output):
    original = vr.read(root / 'final_metrics.json')
    rows = vr.read(root / 'final_inputs.json')
    gold = vr.read(root / 'final_labels_private.json')
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or len(ids) != 600:
        raise ValueError('Not the frozen 600 questions')
    labels = {sid: gold[sid]['verdict'] for sid in ids}
    if Counter(labels.values()) != {'correct': 300, 'incorrect': 300}:
        raise ValueError('Frozen class inventory changed')
    files = {str(root / name): vr.sha(root / name) for name in
             ('final_metrics.json', 'final_inputs.json', 'final_labels_private.json')}
    groups = {}; predictions = {}
    for arm in ARMS:
        pred = {}
        for sid in ids:
            relative = f'checks/final/formal/{arm}/{sid}/result.json'
            path = root / relative; expected = original['source_results'][relative]
            if vr.sha(path) != expected:
                raise ValueError('Frozen result changed: ' + relative)
            value = vr.read(path)
            if value['id'] != sid:
                raise ValueError('Wrong answer/result association')
            files[str(path)] = expected; pred[sid] = value
        result = measure(labels, pred); prior = original['groups'][arm]
        if any(result['counts'][k] != prior[k] for k in ('TP', 'FP', 'FN', 'TN')):
            raise ValueError('Confusion-matrix mismatch')
        if any(abs(result['original_binary_view'][k] - prior[k]) > 1e-12 for k in ('precision','recall','f1')):
            raise ValueError('Original formula mismatch')
        groups[arm] = result; predictions[arm] = pred
    common = [sid for sid in ids if all(predictions[a][sid]['status'] == 'complete' and
              predictions[a][sid]['value']['verdict'] in {'correct', 'incorrect'} for a in ARMS)]
    paired = {}
    for arm in ARMS[1:]:
        counts = Counter()
        for sid in common:
            b = predictions['pure'][sid]['value']['verdict'] == labels[sid]
            a = predictions[arm][sid]['value']['verdict'] == labels[sid]
            counts['both_correct' if a and b else 'both_wrong' if not a and not b else
                   'fixed_vs_pure' if a else 'regressed_vs_pure'] += 1
        paired[arm] = dict(counts)
    controls = {v: measure(labels, {sid: dict(status='complete', value=dict(verdict=v)) for sid in ids})
                for v in ('incorrect', 'correct')}
    report = dict(task='existing600_final_answer_correctness_not_step_error_presence',
        original_formula_parity=True, positive_class='incorrect', human_gold=False,
        labels_changed=False, model_calls=0, files=files, groups=groups,
        constant_prediction_controls=controls, common_binary_questions=len(common), paired_vs_pure=paired,
        limitations=['120B reference-assisted machine labels already contain documented reference/label errors.',
                     'This is the old tool-driven 600-question run, not the new aligned-v15 step revision.',
                     'No label correction, sample removal, or new model evaluation is performed.'],
        code_sha256=vr.sha(__file__))
    if output.exists() and vr.read(output) != report:
        raise ValueError('Refusing to overwrite a different audit')
    for path, digest in files.items():
        if vr.sha(path) != digest:
            raise ValueError('Evidence changed during audit')
    vr.atomic(output, report)
    print({a: x['original_binary_view'] for a, x in groups.items()}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--await-release', action='store_true')
    args = p.parse_args(); out = args.output
    if not Path('/proc').exists():
        raise ValueError('Formal recomputation must run on the server')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.await_release:
            ready = out.with_suffix('.ready.json')
            vr.atomic(ready, dict(pid=os.getpid(), state='awaiting_reconnect', unix=time.time()))
            deadline = time.monotonic() + 120
            while not out.with_suffix('.ALLOW').exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError('Detached verification missing')
                time.sleep(1)
        audit(args.root.resolve(), out.resolve())


if __name__ == '__main__':
    main()
