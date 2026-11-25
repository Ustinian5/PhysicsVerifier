"""Collect paired step outputs, hide retrieval condition for review, and score on node14."""
import argparse
import time
from pathlib import Path
from unittest.mock import patch

from scripts import review_frozen_bm25 as published
from scripts import run_step_retrieval_comparison as trial

f, vr, require, ROOT = trial.f, trial.vr, trial.require, trial.ROOT
review = published.review


def bind_tasks(mode):
    work = ROOT/mode
    signature = trial.design()
    tasks = []
    receipts = {arm:vr.read(ROOT/'receipts'/arm/'formal.json') for arm in trial.ARMS}
    require(all(r['signature']==signature and len(r['rows'])==99 for r in receipts.values()),'Missing formal arms')
    for task in vr.read(f.ROOT/'tasks.json'):
        sid,arm = task['id'],task['arm']
        if arm=='pure' or task['action'] in ('retain_prepare_failure','scope_hold'):
            path = trial.prior_task(task,'formal')
            if arm!='pure' and sid not in f.sources.STEP_HOLD:
                require(receipts[arm]['rows'][sid]['results']=={str(path):vr.sha(path)},'Changed PREPARE failure')
        else:
            path = f.base.path_for(work,'formal',arm,dict(id=sid))
            require(receipts[arm]['rows'][sid]['results'][str(path)]==vr.sha(path),'Changed paired result')
        tasks.append(dict(id=sid,arm=arm,action='reuse',parent_result=str(path),parent_result_sha256=vr.sha(path)))
    require(len(tasks)==400,'Wrong review coverage')
    f.write(work/'tasks.json',tasks)


def prepare():
    frozen = f.ROOT/'offline_review'
    path = frozen/'evidence.json'
    require(vr.sha(path)==vr.read(frozen/'review_frozen.json')['sha256']['evidence.json'],'Unbound prior review')
    prior = vr.read(path)['records']
    pending,operator = {},{}
    for mode in trial.MODES:
        work = ROOT/mode
        bind_tasks(mode)
        if (work/'offline_review/review_frozen.json').exists():
            review.verify(work,complete=True)
            continue
        review.prepare(work)
        evidence_path=work/'offline_review/evidence.json'
        evidence=vr.read(evidence_path)
        for row in evidence['records']:
            if row['disposition']=='pending':
                known=review.reusable_decision(prior,row['id'],row['original_diagnostic'])
                if known is not None:
                    row.update({k:known[k] for k in (*review.FIELDS,'rationale')},
                               reused_from=dict(path=str(path),sha256=vr.sha(path),review_id=known['review_id']))
        vr.atomic(evidence_path,evidence)
        _,_,labels,records,_=review.verify(work)
        for row in records:
            if row['disposition']!='pending':
                continue
            key=vr.digest([row['id'],row['original_diagnostic']])
            pending[key]=dict(id=row['id'],diagnostic=row['original_diagnostic'],spans=row['automatic_spans'],
                              roots=[v for v in labels.values() if v['id']==row['id']])
            operator.setdefault(key,[]).append([mode,row['review_id']])
    vr.atomic(ROOT/'blind_review/pending.json',pending)
    vr.atomic(ROOT/'blind_review/operator_map.json',operator)
    vr.atomic(ROOT/'review_status.json',dict(stage='pending_blind_review' if pending else 'ready_to_score',
                                            pending=len(pending)))


def apply(decisions):
    operator=vr.read(ROOT/'blind_review/operator_map.json')
    require(set(decisions)<=set(operator),'Unknown blind review ID')
    by_mode={mode:{} for mode in trial.MODES}
    for key,decision in decisions.items():
        for mode,rid in operator[key]:
            by_mode[mode][rid]=decision
    for mode,items in by_mode.items():
        if items:
            review.apply(ROOT/mode,items)


def score():
    summaries={}
    for mode in trial.MODES:
        work=ROOT/mode
        with patch.object(published,'ROOT',work),patch.object(published,'OUT',work/'offline_review'):
            published.score()
        summaries[mode]=vr.read(work/'metrics_99.json')['groups']
    f.write(ROOT/'metrics.json',dict(compared_questions=99,compared_roots=163,groups=summaries,
                                    source={m:vr.sha(ROOT/m/'metrics_99.json') for m in trial.MODES},
                                    positive_unit='confirmed step error',scoring_model_calls=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('after-run','prepare','apply','score'))
    parser.add_argument('--decisions',type=Path)
    args=parser.parse_args()
    trial.t.b.node.node()
    if args.action=='after-run':
        while not all((ROOT/'receipts'/a/'formal.json').exists() for a in trial.ARMS):
            require(not (ROOT/'STOP').exists(),'STOP requested')
            time.sleep(20)
    plan=vr.read(ROOT/'design.json')
    require(trial.design()==plan['configuration_sha256'],'Changed design')
    def verify_work(path):
        require(Path(path) in {ROOT/m for m in trial.MODES},'Unexpected review path')
        return plan
    with f.base.r.run_lock(ROOT/'blind_review.lock'),patch.object(review.run,'verify',verify_work):
        if args.action in ('after-run','prepare'):
            prepare()
            if vr.read(ROOT/'review_status.json')['pending']==0:
                score()
        elif args.action=='apply':
            require(args.decisions is not None,'Supply blind review decisions')
            apply(vr.read(args.decisions))
        else:
            score()
