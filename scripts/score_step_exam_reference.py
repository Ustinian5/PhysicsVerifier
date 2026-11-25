"""Review masked diagnoses and compare step-error F1 using the frozen scoring protocol."""
import argparse
from pathlib import Path

import numpy as np

from scripts import review_aligned_step_revision as review
from scripts import run_step_exam_reference as trial

vr, require, write, old = trial.vr, trial.require, trial.write, review.old
OUT = trial.ROOT/trial.TRIAL
SEED = 20261126


def paired_f1(control, with_answer):
    def terms(rows):
        return np.array([[2*r['hits'], r['locatable'] + r['gt']] for r in rows], dtype=np.int64)

    def points(t):
        return np.divide(100.*t[...,0], t[...,1], out=np.zeros(t.shape[:-1]), where=t[...,1]!=0)

    require(len(control)==len(with_answer)>0 and
            all((a['id'],a['gt'])==(b['id'],b['gt']) for a,b in zip(control,with_answer)), 'Unpaired roots')
    left, right = terms(control), terms(with_answer)
    a, b = left.sum(axis=0), right.sum(axis=0)
    observed = float(points(b)-points(a))
    rng = np.random.default_rng(SEED)
    n = len(left)
    weights = rng.multinomial(n, np.full(n,1/n), size=10000)
    interval = np.quantile(points(weights@right)-points(weights@left), [.025,.975]).tolist()
    changes = (right-left)[np.any(left!=right,axis=1)]
    k = len(changes)
    exact = k<=18
    draws = 2**k if exact else 100000
    extreme = 0
    for start in range(0,draws,2048):
        size = min(2048,draws-start)
        if exact:
            swap = (np.arange(start,start+size)[:,None] >> np.arange(k)) & 1
        else:
            swap = rng.integers(0,2,size=(size,k),dtype=np.int8)
        shift = swap@changes
        permuted = points(b-shift)-points(a+shift)
        extreme += int(np.count_nonzero(np.abs(permuted)>=abs(observed)-1e-10))
    return dict(delta_f1_pp=observed,paired_bootstrap_ci95_pp=interval,bootstrap_draws=10000,
                paired_two_sided_p=extreme/draws if exact else (extreme+1)/(draws+1),
                permutation_draws=draws,exact_permutation=exact,informative_pairs=k,seed=SEED)


def gather(complete):
    plan=trial.design(trial.ROOT)
    frozen=trial.frozen
    ids=plan['configuration']['ids']
    parent = frozen.PARENT/'offline_review'
    require(vr.sha(parent/'labels.json') == frozen.LABELS, 'Step labels changed')
    labels = {r['root_id']:r for r in vr.read(parent/'labels.json')['records'] if r['id'] in ids}
    inputs = {r['id']:r for r in trial.inputs(trial.ROOT)}
    old_review = frozen.ROOT/'offline_review'
    hashes=vr.read(old_review/'review_frozen.json')['sha256']
    require(vr.sha(old_review/'evidence.json') == hashes['evidence.json'], 'Archived decisions changed')
    prior=vr.read(old_review/'evidence.json')['records']
    rows, pending, reused, evidence = {}, {}, {}, {}
    answers=vr.read(OUT/'source_answers.json')['answers']
    for gpu in trial.GPUS:
        shard=OUT/f'gpu{gpu}'
        receipt=shard/('complete.json' if complete else 'progress.json')
        if not receipt.exists() and not complete:
            continue
        data=vr.read(receipt)
        if complete:
            require(data['signature']==plan['configuration_sha256'], 'Shard changed')
        for sid, bound in data['results'].items():
            path=Path(bound['path'])
            require(sid in ids and sid not in rows and vr.sha(path)==bound['sha256'], 'Result changed')
            result=vr.read(path)
            require(result['id']==sid and (result.get('failure_stage')=='prepare' if bound['shared_failure']
                    else result['configuration_sha256']==plan['configuration_sha256']), 'Unbound outcome')
            keys=[]
            for d in result['diagnostics']:
                key=vr.digest([sid,d])
                keys.append(key)
                matched=review.reusable_decision(prior,sid,d)
                if matched is not None:
                    reused[key]=dict(review_id=key,**{k:matched[k] for k in (*review.FIELDS,'rationale')})
                    continue
                pending[key]=dict(id=sid, message=d['message'], quote=d['evidence']['quote'],
                                  automatic_spans=old.locate_all(inputs[sid]['prediction'],d['evidence']['quote']),
                                  roots=[l for l in labels.values() if l['id']==sid])
            rows[sid]=dict(keys=keys, failed=result['status']!='completed',status=result['status'],
                           all_answer_parts_present=answers[sid]['all_parts_present'],
                           shared_failure=bound['shared_failure'],new_requests=bound['new_requests'])
            evidence[sid]=bound['sha256']
    if complete:
        require(set(rows)==set(ids), 'Incomplete frozen cohort')
    return inputs, labels, rows, pending, reused, evidence


def masked():
    _,_,rows,pending,_,_=gather(False)
    vr.atomic(OUT/'review'/'pending.json',pending)
    print(f'{len(rows)} terminal questions, {len(pending)} unique diagnoses requiring review',flush=True)


def decide(key, item, value, inputs, labels):
    text=inputs[item['id']]['prediction']
    spans=item['automatic_spans']
    if 'source_quote' in value:
        spans=[old.source_span(text,s,e) for s,e in old.occurrences(text,value['source_quote'])]
        require(spans and value['source_quote'].strip(), 'Nonverbatim source repair')
    if value.get('unlocatable'):
        require(not spans,'Discarded locatable quote')
    roots=[item['id']+'/'+r for r in value['roots']]
    disp=value['disposition']
    require(disp in {'same_root','other_claim','opposite_claim','unclear'} and
            bool(roots)==(disp=='same_root') and (not roots or spans) and value['rationale'].strip(),
            'Unresolved or inconsistent diagnosis')
    require(len(set(roots))==len(roots) and all(r in labels for r in roots), 'Foreign step annotation')
    for s in spans:
        old.check_span(text,s)
    return dict(review_id=key,disposition=disp,approved_spans=spans,rationale=value['rationale'],
                root_links=[dict(root_id=r,same_direction=True,acceptable_source_spans=spans,
                                 rationale=value['rationale']) for r in roots])


def summary(rows):
    t={k:sum(r[k] for r in rows) for k in ('hits','gt','locatable','published','failed')}
    return dict(**t,**old.rates(t['hits'],t['locatable'],t['gt']),unlocatable=t['published']-t['locatable'])


def score():
    inputs,labels,rows,pending,reviewed,evidence=gather(True)
    decisions=vr.read(OUT/'review'/'decisions.json')
    require(set(decisions)==set(pending), 'Missing or extra masked decisions')
    for key,item in pending.items():
        reviewed[key]=decide(key,item,decisions[key],inputs,labels)
    parent=trial.frozen.PARENT
    ref=parent/'reference.json'
    require(vr.sha(ref)=='d33e4f2355def946e63077a02d7107534c2fca28e7ea37a2d35d5bf7c0ef566d','Root spans changed')
    refs={r['id']:r['physics_error_gt'] for r in vr.read(ref)}
    frozen=trial.frozen.ROOT
    control_summary=vr.read(frozen/'metrics_99.json')
    control_file=frozen/'offline_review/metrics.json'
    require(vr.sha(control_file)==control_summary['full_coverage_sha256'], 'Control score changed')
    by_id={r['id']:r for r in vr.read(control_file)['groups']['pure']['details']}
    control,treated=[],[]
    for sid in inputs:
        row=rows[sid]
        diagnoses=[reviewed[k] for k in row['keys']]
        matches=old.assign(refs[sid],diagnoses,labels,True)
        total=sum(labels[g['error_id']]['state']=='active_error' for g in refs[sid])
        control.append(by_id[sid])
        treated.append(dict(id=sid,hits=len(matches),gt=total,locatable=sum(bool(d['approved_spans']) for d in diagnoses),
                            published=len(diagnoses),failed=row['failed'],matches=matches))
    groups=dict(pure=summary(control),pure_with_exam_answer=summary(treated))
    require(groups['pure']['gt']==groups['pure_with_exam_answer']['gt']==163 and
            all(v==control_summary['groups']['pure'][k] for k,v in groups['pure'].items()),
            'Frozen control or root denominator differs')
    complete=[i for i,r in enumerate(control) if rows[r['id']]['all_answer_parts_present']]
    def root_ids(group):
        return {match['root_id'] for row in group for match in row['matches']}
    gains=root_ids(treated)-root_ids(control)
    losses=root_ids(control)-root_ids(treated)
    write(OUT/'review'/'frozen.json',dict(decisions_sha256=vr.sha(OUT/'review'/'decisions.json'),
                                         root_labels_sha256=trial.frozen.LABELS,evidence=evidence,
                                         distinct_rereviewed=len(pending),reused=len(reviewed)-len(pending)))
    write(OUT/'details.json',dict(pure=control,pure_with_exam_answer=treated))
    write(OUT/'metrics.json',dict(n=len(control),groups=groups,inference=paired_f1(control,treated),
          completely_specified_answers=dict(n=len(complete),pure=summary([control[i] for i in complete]),
                                             pure_with_exam_answer=summary([treated[i] for i in complete])),
          newly_matched_roots=sorted(gains),previously_matched_roots_lost=sorted(losses),
          shared_prepare_failures=sum(r['shared_failure'] for r in rows.values()),
          new_model_requests=sum(r['new_requests'] for r in rows.values()),scoring_model_calls=0,human_gold=False,
          design_sha256=vr.sha(OUT/'design.json'),details_sha256=vr.sha(OUT/'details.json'),
          review_sha256=vr.sha(OUT/'review'/'frozen.json'),scorer_sha256=vr.sha(__file__)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('masked','score'))
    action=p.parse_args().action
    trial.frozen.node.node()
    with trial.frozen.base.r.run_lock(OUT/'score.lock'):
        masked() if action=='masked' else score()
