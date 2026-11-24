"""Score the fixed six-question retrieval pilot from method-blind binary relevance labels."""
import argparse
from collections import Counter
from pathlib import Path

from rule_framework import verified_resume as vr


def score(folder):
    rows = vr.read(folder / 'summary.json')
    blind = vr.read(folder / 'blind.json')
    reviews = vr.read(folder / 'relevance.json')
    if len(rows) != 6 or set(blind) != set(reviews) or set(blind) != {r['id'] for r in rows}:
        raise ValueError('Incomplete pilot or blind review')
    methods = ('bm25', 'relation_only', 'alternating', 'rrf', 'rrf_rerank')
    if all('pool_runs' in r for r in rows):
        methods += ('original_pool_rerank', 'relation_pool_rerank')
    report = {}
    for control in (False, True):
        group = {}
        for method in methods:
            counts = Counter()
            hits_by_question = {}
            reciprocal = 0.0
            elapsed = 0.0
            for row in rows:
                sid = row['id']
                if sid.startswith('control_') != control:
                    continue
                grades = reviews[sid]
                pool = {d['id'] for d in blind[sid]}
                union = {item for docs in row['methods'].values() for item in docs}
                if set(grades) != pool or pool != union or not all(type(x) is int and x in (0, 1) for x in grades.values()):
                    raise ValueError('Blind pool or labels differ for ' + sid)
                ids = row['methods'].get(method, [])
                if len(ids) != len(set(ids)) or len(ids) > 6:
                    raise ValueError('Invalid delivery')
                positions = [p for p, key in enumerate(ids, 1) if grades[key]]
                hits_by_question[sid] = len(positions)
                counts.update(questions=1, valid=int(method in row['methods']), delivered=len(ids),
                              relevant=len(positions), questions_with_hit=int(bool(positions)))
                if positions:
                    reciprocal += 1 / positions[0]
                if method != 'bm25':
                    elapsed += row['rewrite_seconds']
                if method in row.get('pool_runs', {}):
                    elapsed += row['pool_runs'][method]['seconds']
                if method == 'rrf_rerank':
                    elapsed += row.get('rank_seconds', 0)
            n = counts['questions']
            group[method] = dict(counts, relevant_by_question=hits_by_question,
                                 relevant_per_question=counts['relevant']/n, mrr=reciprocal/n,
                                 mean_extra_model_seconds=elapsed/n)
        report['controls' if control else 'real'] = group
    return dict(groups=report, hashes={f:vr.sha(folder/f) for f in
                ('design.json','summary.json','blind.json','relevance_rubric.json','relevance.json')}, human_gold=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    args = p.parse_args()
    vr.atomic(args.folder/'metrics.json',score(args.folder))
