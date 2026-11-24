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


def score_confirmation(folder):
    from math import comb
    from random import Random
    from statistics import mean
    from scripts.confirm_skill_retrieval import METHODS

    rows = vr.read(folder/'summary.json')
    pools = vr.read(folder/'blind.json')
    labels = vr.read(folder/'relevance.json')
    design = vr.read(folder/'design.json')
    inputs = vr.read(folder/'inputs.json')
    assert vr.sha(folder/'inputs.json')==design['input_sha256']
    assert set(rows)==set(pools)==set(labels)=={r['id'] for r in inputs}
    counts = {m:Counter() for m in METHODS}
    paired = {m:{} for m in METHODS}
    for k,arms in rows.items():
        un = set().union(*(a['ids'] for a in arms.values()))
        assert set(arms)==set(METHODS) and set(labels[k])==un=={d['id'] for d in pools[k]}
        assert all(type(x) is int and x in (0,1) for x in labels[k].values())
        for m,a in arms.items():
            ids = a['ids']
            assert len(ids)<=6 and len(ids)==len(set(ids))
            hits = [pos for pos,i in enumerate(ids,1) if labels[k][i]]
            verdict = a.get('verdict') if a['status']=='complete' else 'failure'
            paired[m][k] = (len(hits),verdict)
            counts[m].update(questions=1,delivered=len(ids),useful=len(hits),hits=int(bool(hits)),
                             reciprocal_rank=1/hits[0] if hits else 0,
                             extra_model_seconds=a.get('extra_model_seconds',0))
            counts[m][verdict]+=1
    comparisons = {}
    for m,base in [('relation','bm25'),('relation','rrf'),('rrf','bm25')]:
        delta,changed = [],Counter()
        for k in sorted(rows):
            a,va = paired[m][k]
            b,vb = paired[base][k]
            delta.append(a-b)
            changed[f'{vb}_to_{va}']+=1
        wins,losses = sum(v>0 for v in delta), sum(v<0 for v in delta)
        n=wins+losses
        p=min(1,2*sum(comb(n,j) for j in range(min(wins,losses)+1))/2**n)
        rng=Random(20261124)
        bootstrap=sorted(mean(rng.choices(delta,k=len(delta))) for _ in range(10000))
        comparisons[m+'_vs_'+base]=dict(wins=wins,losses=losses,ties=len(rows)-n,
            mean_gain=mean(delta),paired_bootstrap_95=[bootstrap[249],bootstrap[9749]],exact_two_sided_sign_p=p,
            downstream_transitions=dict(changed))
    return dict(groups=counts,comparisons=comparisons,human_gold=False,
                hashes={n:vr.sha(folder/n) for n in ('inputs.json','design.json','rubric.json','relevance.json','blind.json','summary.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    args = p.parse_args()
    scorer = score_confirmation if (args.folder/'rubric.json').is_file() and (args.folder/'inputs.json').is_file() else score
    vr.atomic(args.folder/'metrics.json',scorer(args.folder))
