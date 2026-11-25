"""Paired inference for the frozen final-answer metric, with incorrect as positive."""
from collections import Counter
from fractions import Fraction
from itertools import product
from math import comb

import numpy as np

from scripts import audit_final_answer_metrics as binary


SEED = 20261126
DRAWS = 10000


def f1_terms(label, verdict):
    if verdict not in ('correct', 'incorrect'):
        return (0, 0)
    if label == verdict == 'incorrect':
        return (2, 2)
    return (0, int(label != verdict))


def difference(left, right):
    def fraction(pair):
        n, d = pair
        return Fraction(n, d) if d else Fraction()
    return fraction(right) - fraction(left)


def exact_swap(left, right):
    """Exact two-sided test of delta F1 under independent within-question swaps."""
    a = tuple(map(sum, zip(*left)))
    b = tuple(map(sum, zip(*right)))
    observed = difference(a, b)
    changes = Counter((y[0]-x[0], y[1]-x[1]) for x,y in zip(left,right) if x != y)
    ways = Counter({(0, 0): 1})
    for (dn, dd), n in changes.items():
        terms = [(i*dn, i*dd, comb(n,i)) for i in range(n+1)]
        updated = Counter()
        for (sn, sd), count in ways.items():
            for xn, xd, mult in terms:
                updated[sn+xn, sd+xd] += count * mult
        ways = updated
    extreme = 0
    for (sn, sd), count in ways.items():
        delta = difference((a[0]+sn, a[1]+sd), (b[0]-sn, b[1]-sd))
        if abs(delta) >= abs(observed):
            extreme += count
    pairs = sum(changes.values())
    return dict(delta_f1_pp=100*float(observed), paired_two_sided_p=extreme/(2**pairs),
                informative_pairs=pairs, exact_permutation=True)


def bootstrap(labels, left, right):
    """Resample whole pairs within each frozen target class, at its observed size."""
    rng = np.random.default_rng(SEED)
    totals = np.zeros((DRAWS, 4), dtype=np.int64)
    for label in ('correct', 'incorrect'):
        terms = np.asarray([(*a,*b) for y,a,b in zip(labels,left,right) if y==label], dtype=np.int64)
        n = len(terms)
        if n:
            weights = rng.multinomial(n, np.full(n, 1/n), size=DRAWS)
            totals += weights @ terms
    num, den = totals[:, ::2], totals[:, 1::2]
    scores = np.divide(100.*num, den, out=np.zeros_like(num, dtype=float), where=den!=0)
    return np.quantile(scores[:,1]-scores[:,0], [.025,.975]).tolist()


def analyze(labels, pairs, held_label):
    if set(labels) != set(pairs) or held_label not in ('correct','incorrect'):
        raise ValueError('Unpaired outcomes or unknown frozen label')
    arms = ('pure','with_exam_answer')
    conditions, conditions_terms = {}, []
    for arm in arms:
        rows = {q:pairs[q][arm] for q in labels}
        conditions[arm] = binary.measure(labels, rows)
        conditions_terms.append([f1_terms(y, rows[q]['value']['verdict'] if rows[q]['value'] else None)
                                 for q,y in labels.items()])
    left,right = conditions_terms
    main = exact_swap(left,right)
    main.update(stratified_paired_ci95_pp=bootstrap(list(labels.values()),left,right),
                bootstrap_draws=DRAWS, bootstrap_seed=SEED)
    reported = 100*(conditions[arms[1]]['original_binary_view']['f1'] -
                    conditions[arms[0]]['original_binary_view']['f1'])
    if abs(reported-main['delta_f1_pp']) > 1e-10:
        raise ValueError('Paired contributions disagree with frozen F1')
    # An abstention and a content failure contribute identically to binary F1.
    bounds = []
    for a,b in product(('correct','incorrect','nonbinary'),repeat=2):
        result = exact_swap([*left, f1_terms(held_label,a)], [*right, f1_terms(held_label,b)])
        bounds.append(dict(pure=a,with_exam_answer=b,**result))
    common = [q for q in labels if all(pairs[q][a]['value'] and
              pairs[q][a]['value']['verdict'] in ('correct','incorrect') for a in arms)]
    flips = Counter()
    for q in common:
        a,b = [pairs[q][arm]['value']['verdict'] for arm in arms]
        if a != b:
            flips['corrected' if b == labels[q] else 'regressed'] += 1
    both = {arm:binary.measure({q:labels[q] for q in common},{q:pairs[q][arm] for q in common})
            for arm in arms}
    return dict(groups=conditions, inference=main, common_binary=dict(n=len(common),groups=both,
                corrected=flips['corrected'],regressed=flips['regressed']),
                registered_sensitivity=dict(held_label=held_label,scenarios=bounds,
                    delta_f1_pp_range=[min(r['delta_f1_pp'] for r in bounds), max(r['delta_f1_pp'] for r in bounds)],
                    max_p=max(r['paired_two_sided_p'] for r in bounds)))
