"""Freeze and score the first 50 step questions with paired checks on all arms."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scripts import review_step_retrieval_comparison as review

trial, vr, require = review.trial, review.vr, review.require
old = review.review.old
OUT = trial.ROOT / 'paired50'


def prepare():
    signature = trial.design()
    tasks = vr.read(trial.f.ROOT / 'tasks.json')
    excluded = {t['id'] for t in tasks if t['arm'] in trial.ARMS and
                t['action'] in ('retain_prepare_failure', 'scope_hold')}
    rows = vr.read(trial.f.PARENT / 'inputs.json')
    selected = [r for r in rows if r['id'] not in excluded][:50]
    require(len(selected) == 50, 'Insufficient eligible questions')
    parent = trial.f.PARENT / 'offline_review'
    require(vr.sha(parent / 'labels.json') == vr.read(parent / 'labels_frozen.json')['labels_sha256'], 'Label drift')
    labels = vr.read(parent / 'labels.json')['records']
    blind, operator, pairs = {}, [], {}
    for row in selected:
        sid = row['id']
        for arm in trial.ARMS:
            folder = trial.ROOT / 'retrieval' / arm / 'formal' / sid
            pair = vr.read(folder / 'pair.json')
            require(pair['signature'] == signature and pair['id'] == sid and
                    vr.sha(folder / 'selection.json') == pair['selection_sha256'], 'Pair changed')
            pairs[str(folder / 'pair.json')] = vr.sha(folder / 'pair.json')
            for mode in trial.MODES:
                path = trial.f.base.path_for(trial.ROOT / mode, 'formal', arm, row)
                require(pair['results'][str(path)] == vr.sha(path), 'Result changed')
                result = vr.read(path)
                require(result['id'] == sid and result['configuration_sha256'] == signature, 'Incorrect result')
                keys = []
                for diagnostic in result['diagnostics']:
                    key = vr.digest([sid, diagnostic])
                    spans = old.locate_all(row['prediction'], diagnostic['evidence']['quote'])
                    blind[key] = dict(id=sid, message=diagnostic['message'], quote=diagnostic['evidence']['quote'],
                                      automatic_spans=spans)
                    keys.append(key)
                operator.append(dict(id=sid, arm=arm, mode=mode, keys=keys, result=str(path),
                                     result_sha256=vr.sha(path), status=result['status']))
    ids = [r['id'] for r in selected]
    require(len(operator) == 300 and len(set(ids)) == 50, 'Incomplete prefix')
    sources = [parent/'labels.json', trial.f.PARENT/'inputs.json', trial.f.PARENT/'reference.json',
               trial.f.ROOT/'offline_review/metrics.json', trial.f.ROOT/'metrics_99.json', trial.ROOT/'design.json']
    trial.write(OUT/'cohort.json', dict(ids=ids, rule='First 50 frozen inputs with usable shared PREPARE on all three arms; retain CHECK failures',
                signature=signature, pairs=pairs, sources={str(p):vr.sha(p) for p in sources}))
    trial.write(OUT/'operator.json', operator)
    trial.write(OUT/'blind.json', dict(diagnostics=blind, roots=[r for r in labels if r['id'] in ids]))


def score():
    prepare()
    cohort = vr.read(OUT/'cohort.json')
    blind = vr.read(OUT/'blind.json')
    decisions = vr.read(OUT/'decisions.json')
    require(set(decisions) == set(blind['diagnostics']), 'Incomplete or unknown review decisions')
    labels = {r['root_id']:r for r in blind['roots']}
    inputs = {r['id']:r['prediction'] for r in vr.read(trial.f.PARENT/'inputs.json')}
    refs = {r['id']:r['physics_error_gt'] for r in vr.read(trial.f.PARENT/'reference.json')}
    records = {}
    for key, d in decisions.items():
        raw = blind['diagnostics'][key]
        sid = raw['id']
        spans = raw['automatic_spans']
        if 'source_quote' in d:
            spans = [old.source_span(inputs[sid], s, e) for s, e in old.occurrences(inputs[sid], d['source_quote'])]
            require(bool(spans), 'Nonverbatim source repair')
        for span in spans:
            old.check_span(inputs[sid], span)
        links = []
        for rid in d['roots']:
            root = sid + '/' + rid
            require(root in labels and bool(spans), 'Root lacks supporting source')
            links.append(dict(root_id=root, same_direction=True, acceptable_source_spans=spans, rationale=d['rationale']))
        require(bool(d['rationale']) and (d['disposition']=='same_root')==bool(links), 'Unreviewed diagnosis')
        records[key] = dict(review_id=key, approved_spans=spans, root_links=links)
    rows = defaultdict(list)
    for item in vr.read(OUT/'operator.json'):
        sid, mode, arm = item['id'], item['mode'], item['arm']
        diagnoses = [records[k] for k in item['keys']]
        matches = old.assign(refs[sid], diagnoses, labels, True)
        folder = trial.ROOT/'retrieval'/arm/'formal'/sid
        pair = vr.read(folder/'pair.json')
        elapsed = pair['check_seconds'].get(mode, 0)
        lookup = vr.read(folder/'selection.json')['charges']['seconds'] if mode=='fusion' else 0
        rows[mode, arm].append(dict(id=sid, gt=sum(labels[g['error_id']]['state']=='active_error' for g in refs[sid]),
                  hits=len(matches), locatable=sum(bool(d['approved_spans']) for d in diagnoses), published=len(diagnoses),
                  failed=item['status']!='completed', matches=matches, check_seconds=elapsed, lookup_seconds=lookup))
    frozen = vr.read(trial.f.ROOT/'offline_review/metrics.json')
    pure = [row for row in frozen['groups']['pure']['details'] if row['id'] in cohort['ids']]
    results = {}
    for (mode,arm), group in [*rows.items(), (('archived','pure'),pure)]:
        require({r['id'] for r in group} == set(cohort['ids']) and len(group)==50, 'Incomplete group')
        totals = {k:sum(r[k] for r in group) for k in ('gt','hits','locatable','published','failed')}
        stats = dict(**totals, **old.rates(totals['hits'],totals['locatable'],totals['gt']))
        if mode!='archived':
            stats['mean_check_seconds'] = mean(r['check_seconds'] for r in group)
        if mode=='fusion':
            stats['mean_after_prepare_seconds'] = mean(r['lookup_seconds']+r['check_seconds'] for r in group)
        results.setdefault(mode,{})[arm] = stats
    require(len({s['gt'] for m in results.values() for s in m.values()})==1, 'Unequal recall denominators')
    trial.write(OUT/'details.json', {f'{m}/{a}':g for (m,a),g in rows.items()})
    trial.write(OUT/'metrics.json', dict(n=50, human_gold=False, scoring_model_calls=0, groups=results,
                sha256={p:vr.sha(OUT/p) for p in ('cohort.json','blind.json','decisions.json','operator.json','details.json')},
                code_sha256=vr.sha(__file__)))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','score'))
    action = parser.parse_args().action
    trial.t.b.node.node()
    OUT.mkdir(parents=True, exist_ok=True)
    with trial.f.base.r.run_lock(OUT/'report.lock'):
        (prepare if action=='prepare' else score)()
