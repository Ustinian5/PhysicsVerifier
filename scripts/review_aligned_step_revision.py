"""Reuse exact prior adjudications and explicitly review new revision diagnoses.

No API calls, automatic semantic credit, label edits, or experiment reruns.
An incomplete collection/review cannot produce final metrics.
"""
import argparse
from collections import Counter
import copy
import fcntl
import os
from pathlib import Path
import time

from scripts import run_aligned_step_revision as run
from scripts import review_aligned_step_scoring as old

vr, require = run.vr, run.require
FIELDS=('approved_spans','root_links','disposition')


def reusable_decision(records, sid, diagnostic):
    matches=[r for r in records if r['id']==sid and r['original_diagnostic']==diagnostic]
    if not matches:return None
    if len({vr.digest({k:r[k] for k in FIELDS}) for r in matches})!=1:return None
    return copy.deepcopy(matches[0])


def prepare(root):
    plan=run.verify(root); parent=Path(plan['configuration']['parent']); out=root/'offline_review'
    require(not (out/'review_frozen.json').exists(),'Review is frozen')
    prior=vr.read(parent/'offline_review/evidence.json')['records']
    rows={r['id']:r for r in vr.read(parent/'inputs.json')}
    saved=vr.read(out/'evidence.json') if (out/'evidence.json').exists() else dict(records=[])
    previous={r['review_id']:r for r in saved['records']}
    operator={};records=[];files={};available=0
    for task in vr.read(root/'tasks.json'):
        path=Path(task['parent_result']) if task['action']!='run_check' else \
             run.base.path_for(root,'formal',task['arm'],dict(id=task['id']))
        if not path.exists():continue
        result,path=run.result_for(root,task);files[str(path)]=vr.sha(path);available+=1
        require(result['id']==task['id'],'Result id mismatch')
        if task['action']=='run_check':
            require(result['configuration_sha256']==plan['configuration_sha256'],'Wrong revision result')
        for i,d in enumerate(result['diagnostics']):
            rid='D'+vr.digest([task['id'],task['arm'],i])[:16]
            operator[rid]=dict(id=task['id'],arm=task['arm'],index=i,result_path=str(path))
            if rid in previous:
                record=previous[rid]
                require(record['id']==task['id'] and record['original_diagnostic']==d,'Reviewed output changed')
            else:
                match=reusable_decision(prior,task['id'],d)
                if match is not None:
                    source_id=match['review_id'];record=match
                    record.update(review_id=rid,reused_from=dict(path=str(parent/'offline_review/evidence.json'),
                        review_id=source_id,condition='Identical question, answer, and full original diagnostic'))
                else:
                    record=dict(review_id=rid,id=task['id'],original_diagnostic=d,
                        automatic_spans=old.locate_all(rows[task['id']]['prediction'],d.get('evidence',{}).get('quote','')),
                        approved_spans=[],root_links=[],disposition='pending',rationale='')
            records.append(record)
    require(set(previous)<={r['review_id'] for r in records},'Previously available result disappeared')
    for name in ('inputs','reference','unresolved'):
        files[str(parent/(name+'.json'))]=vr.sha(parent/(name+'.json'))
    for name in ('labels','labels_frozen','evidence','review_frozen'):
        files[str(parent/'offline_review'/(name+'.json'))]=vr.sha(parent/'offline_review'/(name+'.json'))
    policy=dict(old.POLICY)
    policy['scope']='All 400 revision tasks; prior labels frozen, identical diagnostic reviews reused; new outputs explicitly reviewed'
    protocol=dict(configuration_sha256=plan['configuration_sha256'],policy=policy,
        parent=str(parent),available_tasks=available,expected_tasks=400,source_sha256=files,
        reviewer_code_sha256=vr.sha(__file__))
    vr.atomic(out/'protocol.json',protocol)
    vr.atomic(out/'operator_map.json',operator)
    vr.atomic(out/'evidence.json',dict(human_gold=False,group_names_hidden_in_review=True,
                                    records=sorted(records,key=lambda r:r['review_id'])))
    print(dict(tasks=available,diagnoses=len(records),states=dict(Counter(r['disposition'] for r in records)),
               reused=sum('reused_from' in r for r in records)),flush=True)


def verify(root, *, complete=False):
    out=root/'offline_review';protocol=vr.read(out/'protocol.json')
    require(protocol['configuration_sha256']==run.verify(root)['configuration_sha256'],'Review plan mismatch')
    require(protocol['reviewer_code_sha256']==vr.sha(__file__),'Review code changed')
    if complete:require(protocol['available_tasks']==400,'Incomplete collection cannot be scored')
    vr.EvidenceCache(out/'evidence_cache.json').check(protocol['source_sha256'])
    parent=Path(protocol['parent']);rows={r['id']:r for r in vr.read(parent/'inputs.json')}
    label_data=vr.read(parent/'offline_review/labels.json')
    require(vr.sha(parent/'offline_review/labels.json')==vr.read(parent/'offline_review/labels_frozen.json')['labels_sha256'],
            'Prior labels changed')
    labels={r['root_id']:r for r in label_data['records']}
    require(Counter(r['state'] for r in labels.values())==dict(active_error=164,self_corrected=16,unresolved=2),
            'Label states changed')
    records=vr.read(out/'evidence.json')['records'];operator=vr.read(out/'operator_map.json')
    require(len(records)==len(operator) and {r['review_id'] for r in records}==set(operator),'Coverage mismatch')
    result_cache={};seen=set()
    for r in records:
        b=operator[r['review_id']];identity=(b['id'],b['arm'],b['index'])
        require(identity not in seen,'Duplicate diagnosis');seen.add(identity)
        path=b['result_path']
        if path not in result_cache:result_cache[path]=vr.read(path)
        require(r['original_diagnostic']==result_cache[path]['diagnostics'][b['index']] and r['id']==b['id'],
                'Review changed the original diagnosis')
        if complete:require(r['disposition']!='pending','Unreviewed diagnosis')
        if r['disposition']=='pending':continue
        require(r['disposition'] in {'same_root','other_claim','opposite_claim','unclear'} and r['rationale'],
                'Invalid review disposition')
        require(bool(r['root_links'])==(r['disposition']=='same_root'),'Root/disposition mismatch')
        for s in r['approved_spans']:old.check_span(rows[r['id']]['prediction'],s)
        for link in r['root_links']:
            require(labels[link['root_id']]['id']==r['id'] and link['same_direction'] and link['rationale'],
                    'Invalid root link')
            require(r['approved_spans'],'Root match requires located evidence')
            for s in link['acceptable_source_spans']:old.check_span(rows[r['id']]['prediction'],s)
    expected=sum(len(vr.read(path)['diagnostics']) for path in protocol['source_sha256'] if path.endswith('/result.json'))
    require(expected==len(records),'Missing published diagnosis from review')
    return protocol,rows,labels,records,operator


def apply(root,decisions):
    out=root/'offline_review';require(not (out/'review_frozen.json').exists(),'Frozen review')
    _,rows,_,records,_=verify(root)
    byid={r['review_id']:r for r in records}
    for rid,d in decisions.items():
        r=byid[rid];require(r['disposition']=='pending','Existing adjudication cannot be overwritten')
        text=rows[r['id']]['prediction']
        if 'source_quote' in d:
            spans=[dict(**old.source_span(text,s,e),method='assistant_source_repair')
                   for s,e in old.occurrences(text,d['source_quote'])]
            require(spans,'Repair is not a verbatim continuous original span')
        elif d.get('unlocatable'):spans=[]
        else:spans=r['automatic_spans']
        roots=d.get('roots',[])
        require(d.get('rationale') and (not roots or spans),'Explicit rationale and source required')
        r.update(approved_spans=spans,disposition=d.get('disposition','same_root' if roots else 'other_claim'),
                 rationale=d['rationale'],root_links=[dict(root_id=r['id']+'/'+e,same_direction=True,
                     acceptable_source_spans=spans,rationale=d['rationale']) for e in roots])
    vr.atomic(out/'evidence.json',dict(human_gold=False,group_names_hidden_in_review=True,records=records))
    verify(root)


def freeze(root):
    verify(root,complete=True);out=root/'offline_review'
    hashes={name:vr.sha(out/name) for name in ('protocol.json','evidence.json','operator_map.json')}
    path=out/'review_frozen.json'
    if path.exists():require(vr.read(path)['sha256']==hashes,'Frozen review changed')
    else:vr.atomic(path,dict(sha256=hashes,unix=time.time(),human_gold=False))


def score(root):
    require(Path('/proc').exists(),'Official score must execute on the server')
    protocol,rows,labels,records,operator=verify(root,complete=True);out=root/'offline_review'
    for name,digest in vr.read(out/'review_frozen.json')['sha256'].items():
        require(vr.sha(out/name)==digest,'Frozen evidence changed')
    parent=Path(protocol['parent']);ref={r['id']:r for r in vr.read(parent/'reference.json')}
    unknown=vr.read(parent/'unresolved.json');tasks=vr.read(root/'tasks.json')
    indexed={(operator[r['review_id']]['id'],operator[r['review_id']]['arm'],operator[r['review_id']]['index']):r
             for r in records};groups={}
    for arm in run.base.p.ARMS:
        details=[]
        for t in [t for t in tasks if t['arm']==arm]:
            result,_=run.result_for(root,t);sid=t['id']
            ds=[indexed[sid,arm,i] for i in range(len(result['diagnostics']))]
            gt=ref[sid]['physics_error_gt'];matches=old.assign(gt,ds,labels,True)
            details.append(dict(id=sid,published=len(ds),locatable=sum(bool(d['approved_spans']) for d in ds),
                gt=sum(labels[g['error_id']]['state']=='active_error' for g in gt),hits=len(matches),matches=matches,
                unresolved_overlap=old.unresolved_overlaps(ds,matches,unknown[sid]),
                failed=result['status']!='completed',source_status=result['status'],action=t['action']))
        sums={k:sum(d[k] for d in details) for k in ('published','locatable','gt','hits','failed')}
        require(len(details)==100 and sums['gt']==164,'Denominator mismatch')
        groups[arm]=dict(summary=dict(**sums,**old.rates(sums['hits'],sums['locatable'],sums['gt']),
            unlocatable=sums['published']-sums['locatable'],
            all_published_sensitivity=old.rates(sums['hits'],sums['published'],sums['gt'])),details=details)
    report=dict(policy=protocol['policy'],groups=groups,configuration_sha256=protocol['configuration_sha256'],
        frozen_review_sha256=vr.sha(out/'review_frozen.json'),human_gold=False,independent_blind_review=False,
        reference_root_count=164,scoring_model_calls=0,scorer_sha256=vr.sha(__file__))
    path=out/'metrics.json'
    if path.exists():require(vr.read(path)==report,'Cannot overwrite a different reviewed score')
    else:vr.atomic(path,report)
    print({a:v['summary'] for a,v in groups.items()},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','verify','apply','freeze','score'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--decisions',type=Path)
    p.add_argument('--await-release',action='store_true')
    args=p.parse_args();root=args.root.resolve();out=root/'offline_review';out.mkdir(exist_ok=True)
    with (out/'review.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.await_release:
            require(args.command=='score','Handshake is for scoring')
            vr.atomic(out/'runner_ready.json',dict(pid=os.getpid(),unix=time.time(),state='awaiting_reconnect'))
            deadline=time.monotonic()+120
            while not (out/'ALLOW_SCORE').exists():
                require(time.monotonic()<deadline,'Reconnect confirmation timed out');time.sleep(1)
        if args.command=='prepare':prepare(root)
        elif args.command=='verify':verify(root)
        elif args.command=='apply':apply(root,vr.read(args.decisions))
        elif args.command=='freeze':freeze(root)
        else:score(root)


if __name__=='__main__':main()
