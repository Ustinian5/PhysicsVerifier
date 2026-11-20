"""Measure sample-size stability using the released 600-question final-answer outputs."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from scripts import audit_final_answer_metrics as binary
from scripts import run_final_answer_revision as final
from scripts.analyze_verifier_knowledge_groups import checked_json


SIZES = (100, 200, 400, 600)
SEED = 20261121
SUBSET_DRAWS = 1000
BOOTSTRAP_DRAWS = 10000
CLASSES = ('correct', 'incorrect')


def cohort_order(labels):
    if Counter(labels.values()) != {'correct': 300, 'incorrect': 300}:
        raise ValueError('Expected the released 300 correct and 300 incorrect answers')

    def order_key(sid):
        return hashlib.sha256(f'{SEED}:{sid}'.encode('utf-8')).digest(), sid

    return {label: sorted((sid for sid in labels if labels[sid] == label), key=order_key) for label in CLASSES}


def f1_differences(sums):
    numerator, denominator = sums[..., 0], sums[..., 1]
    f1_points = np.divide(100.0 * numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                          where=denominator != 0)
    return f1_points - f1_points[:, [binary.ARMS.index('pure')]]


def analyze(labels, predictions):
    order = cohort_order(labels)
    if set(predictions) != set(binary.ARMS) or any(set(arm) != set(labels) for arm in predictions.values()):
        raise ValueError('Every arm must contain the same 600 questions, including failures')
    # Store the additive numerator and denominator of the published binary F1 per question.
    terms = np.empty((2, 300, len(binary.ARMS), 2), dtype=np.int64)
    for row, label in enumerate(CLASSES):
        for i, sid in enumerate(order[label]):
            for col, arm in enumerate(binary.ARMS):
                counts = binary.measure({sid: label}, {sid: predictions[arm][sid]})['counts']
                numerator = 2 * counts['TP']
                terms[row, i, col] = numerator, numerator + counts['FP'] + counts['FN']

    rng = np.random.default_rng(SEED)
    permutations = np.array([[rng.permutation(300) for _ in range(SUBSET_DRAWS)] for _ in CLASSES])
    levels = []
    for size in SIZES:
        half = size // 2
        ids = order['correct'][:half] + order['incorrect'][:half]
        gold = {sid: labels[sid] for sid in ids}
        groups = {arm: binary.measure(gold, {sid: predictions[arm][sid] for sid in ids}) for arm in binary.ARMS}
        # Prefixes of the same permutations give nested, class-balanced subsets in each draw.
        subset_counts = sum(terms[c][permutations[c, :, :half]].sum(axis=1) for c in range(len(CLASSES)))
        split_delta = f1_differences(subset_counts)
        # Resample the fixed subset with replacement; each question carries all four outputs.
        rng_boot = np.random.default_rng([SEED, size])
        bootstrap_counts = np.zeros((BOOTSTRAP_DRAWS, len(binary.ARMS) * 2), dtype=np.int64)
        for c in range(len(CLASSES)):
            multiplicity = rng_boot.multinomial(half, np.full(half, 1.0 / half), size=BOOTSTRAP_DRAWS)
            bootstrap_counts += multiplicity @ terms[c, :half].reshape(half, -1)
        bootstrap_delta = f1_differences(bootstrap_counts.reshape(BOOTSTRAP_DRAWS, len(binary.ARMS), 2))
        versus_pure = {}
        for col, arm in enumerate(binary.ARMS):
            if arm == 'pure':
                continue
            diff = split_delta[:, col]
            versus_pure[arm] = dict(
                fixed_delta_f1_pp=100 * (groups[arm]['original_binary_view']['f1'] -
                                         groups['pure']['original_binary_view']['f1']),
                paired_bootstrap_ci95_pp=np.quantile(bootstrap_delta[:, col], [.025, .975]).tolist(),
                balanced_subsets=dict(higher=int((diff > 0).sum()), tied=int((diff == 0).sum()),
                                      lower=int((diff < 0).sum())))
        levels.append(dict(n=size, groups=groups, versus_pure=versus_pure))
    return dict(seed=SEED, draws_without_replacement=SUBSET_DRAWS, paired_bootstrap_draws=BOOTSTRAP_DRAWS,
                subset_order=order, sizes=levels, model_calls=0)


def run(root):
    metrics_path = root / 'metrics.json'
    if not metrics_path.is_file():
        raise ValueError('The formal four-arm metrics have not been published; no curve was computed')
    official = final.vr.read(metrics_path)
    plan = final.execution_plan(root)
    final.released(root, plan)
    release_hash = final.vr.sha(root / 'bench_release.json')
    final.require(official['benchmark_sha256'] == release_hash and official['positive_class'] == 'incorrect',
                  'Metrics belong to another release or positive class')
    labels = {sid: row['verdict'] for sid, row in final.vr.read(root / 'labels_private.json').items()}
    input_hashes = {row['id']: final.vr.digest(final.p.public(row)) for row in final.vr.read(root / 'inputs.json')}
    predictions = {}
    for arm in binary.ARMS:
        predictions[arm] = {}
        for sid in labels:
            path = root / 'formal' / arm / final.vr.digest(sid)[:24] / 'result.json'
            value = checked_json(path, official['files'])
            final.require(value['id'] == sid and value['arm'] == arm and value['input_sha256'] == input_hashes[sid]
                          and value['configuration_sha256'] == plan['configuration_sha256'], 'Wrong result binding')
            predictions[arm][sid] = value
    report = analyze(labels, predictions)
    final.require(report['sizes'][-1]['groups'] == official['groups'], '600-question endpoint differs from release')
    report.update(bench_release_sha256=release_hash, official_metrics_sha256=final.vr.sha(metrics_path),
                  analysis_sha256=final.vr.sha(__file__))
    final.immutable(root / 'sample_size_stability.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='Current aligned_old99/final_revision result root')
    run(parser.parse_args().root.resolve())
