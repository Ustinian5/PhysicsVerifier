"""Compare two pure final-answer CHECKs on frozen 30B responses on node14."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
import multiprocessing
from pathlib import Path
import time
from unittest.mock import patch

import httpx

from rule_framework.oss120b_construction import question_key
from scripts import run_step_exam_reference as step
from scripts import score_final_exam_reference as scoring
from scripts import timed_retrieval_cohort as cohort
from scripts import timed_retrieval_runtime as runtime
from scripts.combined_language_io import iter_rollout_batches
from scripts.combined_language_samples import extract_question


base, vr, require = cohort.original, cohort.vr, cohort.require
final = base.final
OUT = base.RESULTS / 'final_exam_reference'
REGISTERED = '7f8c945a840e6714aed1a9a81700c51b71cec78cc995a320524ffe8e83ba29aa'
CANDIDATES = '116367f1f6fa191dc038b7e5ca1882840b04a8bdd59d1d1ea48c79a2b45279ef'
HOLD = '17eeba3a44d555e6ca4dc53464776d5b43b4418d62513b065136b92acc42a3fc'
PORTS = {4:18249, 5:18250, 6:18251, 7:18252}
STREAMS = 4
ARMS = ('pure','with_exam_answer')
BLIND = ('You do NOT have the reference answer or benchmark label.\n'
         'Determine the final-answer correctness from the original question and student response.')
NEUTRAL = ('The benchmark label is hidden. Determine final-answer correctness using the original '
           "question, the student response, and the current problem's official reference answer if supplied in CHECK.")


def registered(root):
    parent = root/cohort.BENCH
    release = vr.read(parent/'bench_release.json')
    require(vr.sha(parent/'bench_release.json') == base.BENCH and
            vr.sha(parent/'inputs.json') == release['inputs_sha256'] and
            vr.sha(parent/'labels_private.json') == release['labels_sha256'] and
            vr.sha(parent/'dispositions.json') == release['dispositions_sha256'] and
            vr.sha(root/cohort.BM25/'final/metrics.json') == cohort.CURRENT, 'Changed frozen release')
    all_rows = vr.read(parent/'inputs.json')
    rows = [r for r in all_rows if r['id'] not in base.sources.FINAL_HOLD]
    require(len(all_rows) == 474 and len(rows) == 468 and vr.digest([r['id'] for r in rows]) == REGISTERED
            and all(set(r)=={'id','question','context','prediction'} and not r['context'] for r in rows),
            'Different eligible questions or unexpected public fields')
    return rows


def bind_source(root, data):
    parent = root/cohort.BENCH
    rows = [r for r in registered(root) if r['id'] != HOLD]
    require(vr.sha(data)==step.DATA_SHA and vr.sha(parent/'candidates.json')==CANDIDATES, 'Changed source data')
    snapshot = vr.read(root/'corrected/source_screen/references_private.json')
    require(snapshot['source_sha256']==step.DATA_SHA, 'Wrong raw reference index')
    variants = {r['id']:r['variants'] for r in snapshot['references']}
    candidates = {r['id']:r for r in vr.read(parent/'candidates.json')}
    pending = defaultdict(list)
    for row in rows:
        q = row['id']
        c = candidates[q]
        answer = c['reference']
        require(final.p.public(c)==row and c['reference_sha256']==vr.digest(answer) and
                isinstance(answer,list) and answer and all(isinstance(t,str) and t.strip() for t in answer),
                'Incomplete source answer or different 30B response')
        choice, = [v for v in variants[q] if v['value']==answer]
        loc = choice['first_source']
        pending[loc['rollout_id']].append((row,answer,loc))
    refs = {}
    for batch in iter_rollout_batches(data):
        for row,answer,loc in pending.pop(batch['rollout_id'], []):
            sample = batch['samples'][loc['sample_position']]
            require(sample['index']==loc['sample_index'] and sample['label']==answer and
                    question_key(extract_question(sample))==question_key(row['question']), 'Raw source mismatch')
            refs[row['id']] = dict(answer=answer,original_location=loc,input_sha256=vr.digest(row))
        if not pending:
            break
    require(not pending and len(refs)==467 and HOLD not in refs, 'Incomplete reference binding')
    final.immutable(root/OUT.name/'source_answers.json',dict(dataset_sha256=step.DATA_SHA,rows=refs))


def design(root):
    rows = registered(root)
    dest = root/OUT.name
    source = vr.read(dest/'source_answers.json')
    ids = [r['id'] for r in rows if r['id']!=HOLD]
    require(source['dataset_sha256']==step.DATA_SHA and set(source['rows'])==set(ids) and
            all(vr.digest(r)==source['rows'][r['id']]['input_sha256'] for r in rows if r['id']!=HOLD),
            'Reference not bound to current question and answer')
    truth = vr.read(root/cohort.BENCH/'labels_private.json')
    require(Counter(truth[r['id']]['verdict'] for r in rows)=={'incorrect':235,'correct':233} and
            HOLD in {r['id'] for r in rows}, 'Changed classes or hold')
    cfg = dict(registered=468,ids=ids,registered_ids_sha256=REGISTERED, additional_hold=HOLD,
               release_sha256=base.BENCH,source_sha256=vr.sha(dest/'source_answers.json'),
               labels_sha256=vr.sha(root/cohort.BENCH/'labels_private.json'),old_pure_metrics=cohort.CURRENT,
               parent_configuration=base.host.CONFIG,model=final.base.r.EXPECTED_MODEL,arms=list(ARMS),
               neutral_text=NEUTRAL,replaces=BLIND,reference_field=step.FIELD,shared_prepare='archived',
               field_scope='CHECK only. Raw current-problem label. No benchmark verdict, marking points or library.',
               scheduling='Both arms on the same GPU, alternating order by original row; matched native budgets.',
               gpu_ports={str(g):p for g,p in PORTS.items()},concurrent_sequences_per_gpu=STREAMS,
               positive='incorrect',primary='Frozen binary F1 on all measured pairs; abstentions and failures tracked by arm.',
               inference='Exact two-sided within-question swap test of delta F1 at alpha=0.05, with independent '
                         'class-stratified paired bootstrap 95% CI. Bound full-cohort delta and p over every possible '
                         'binary or nonbinary outcome of the held pair.',
               bootstrap_draws=scoring.DRAWS,seed=scoring.SEED,human_gold=False,
               runner_sha256=vr.sha(__file__),scorer_sha256=vr.sha(scoring.__file__))
    obj = dict(configuration=cfg,configuration_sha256=vr.digest(cfg))
    final.immutable(dest/'design.json',obj)
    return obj


def messages(original, answer):
    old = original[0]
    require(old['role']=='system' and old['content'].count(BLIND)==1, 'Expected frozen blind verifier instruction')
    neutral = [dict(old,content=old['content'].replace(BLIND,NEUTRAL)),*original[1:]]
    return neutral, step.add_answer(neutral,answer)


@lru_cache(maxsize=1)
def worker_environment(gpu, stamp):
    base.node.node()
    from transformers import AutoTokenizer
    old = final.execution_plan(base.FORMAL)
    require(old['configuration_sha256']==base.host.CONFIG, 'Changed native final-answer protocol')
    cfg = dict(old['configuration'],api_base=f'http://127.0.0.1:{PORTS[gpu]}/v1')
    plan = dict(configuration=cfg,configuration_sha256=stamp)
    tok = AutoTokenizer.from_pretrained(base.host.HOT/'models/gpt-oss-120b',local_files_only=True)
    files = vr.read(base.FORMAL/'metrics.json')['files']
    return old,plan,tok,httpx.Client(trust_env=False),files


def pair(gpu, stamp, index, row, reference):
    q = row['id']
    require(q!=HOLD and q not in base.sources.FINAL_HOLD and vr.digest(row)==reference['input_sha256'], 'Outside scope')
    old,plan,tok,client,archive = worker_environment(gpu,stamp)
    limits = plan['configuration']['limits']
    folder,prep = base.prefix_record(base.FORMAL,old,'formal',row)
    parent = base.FORMAL/'formal/pure'/vr.digest(q)[:24]
    require(vr.sha(parent/'result.json')==archive[str(parent/'result.json')], 'Archived pure row changed')
    prior = vr.read(parent/'result.json')
    require(vr.sha(folder/'result.json')==prior['files'][str(folder/'result.json')], 'Unbound shared query')
    target = OUT/'rows'/q
    arms = {}
    if prep['status']!='complete':
        require(prior['status']=='content_failure' and not prior['value'] and not (parent/'check').exists(),
                'Unrecognized shared PREPARE failure')
        arms = {arm:dict(status='content_failure',value=None,stage='PREPARE') for arm in ARMS}
    else:
        original = final.p.continuation_messages(row,prep['value'],[],arm='pure')
        body_file = parent/'check/request.json'
        require(vr.sha(body_file)==prior['files'][str(body_file)], 'Archived model input changed')
        original_body = vr.read(body_file)['request']
        maximum = min(limits['max_tokens_per_response'],limits['output_tokens_per_sample']-prep['usage']['completion_tokens'])
        require(base.eligible_for_reuse(dict(messages=original),original_body,plan['configuration']['model'],maximum),
                'Archived CHECK or per-question cap differs')
        available = limits['wall_seconds_per_sample']-prep['wall_seconds']
        paired = dict(zip(ARMS,messages(original,reference['answer'])))
        for arm in (ARMS if index%2==0 else reversed(ARMS)):
            msg = paired[arm]
            start = time.monotonic()
            length = len(tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,reasoning_effort='medium'))
            require(length+prep['usage']['prompt_tokens']<=limits['input_tokens_per_sample'] and
                    length+maximum<=limits['context_window'] and maximum>0 and available>0,
                    'Held output or cumulative input/time budget does not fit')
            turn = target/arm
            check = final.request_once(OUT,turn,msg,maximum,final.p.original.validate_verdict,plan,tok,client,
                                       available=available,started=start)
            require(vr.read(turn/'request.json')['request']==dict(original_body,messages=msg) and
                    check['usage']['prompt_tokens']+prep['usage']['prompt_tokens']<=limits['input_tokens_per_sample'],
                    'Paired request differs from the frozen decode or exceeds cumulative budget')
            arms[arm] = dict(status=check['status'],value={'verdict':check['value']['verdict']} if check['value'] else None,
                             stage='CHECK',check_sha256=vr.sha(turn/'result.json'))
    record = dict(id=q,configuration_sha256=stamp,arms=arms,input_sha256=vr.digest(row),
                  reference_sha256=vr.digest(reference['answer']),prefix_sha256=vr.sha(folder/'result.json'))
    final.immutable(target/'pair.json',record)
    return q,vr.sha(target/'pair.json')


def run_card(gpu, folder, stamp, jobs):
    service = folder/f'gpu{gpu}'
    service.mkdir()
    progress_file = OUT/f'gpu{gpu}.progress.json'
    done = {}
    todo = []
    for job in jobs:
        q = job[1]['id']
        path = OUT/'rows'/q/'pair.json'
        if path.exists():
            record = vr.read(path)
            require(record['id']==q and record['configuration_sha256']==stamp, 'Wrong saved pair')
            done[q]=vr.sha(path)
        else:
            todo.append(job)
    def progress():
        vr.atomic(progress_file,dict(completed=len(done),total=len(jobs),unix=time.time()))
    progress()
    if todo:
        original = base.host.service_script
        def launcher(config):
            script, paths = original(config)
            require(script.count('18083')==2 and script.count('--max-num-seqs 1 ')==1, 'Changed service launcher')
            return script.replace('18083',str(PORTS[gpu])).replace('--max-num-seqs 1 ',f'--max-num-seqs {STREAMS} '),paths
        with patch.object(base.host,'PORT',PORTS[gpu]), patch.object(base.host,'service_script',launcher), \
             base.node.service(service,gpu,OUT/'STOP',stamp), \
             ProcessPoolExecutor(max_workers=STREAMS,mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs_iter = iter(todo)
            q,h = pool.submit(pair,gpu,stamp,*next(jobs_iter)).result()
            done[q]=h
            progress()
            active = set()
            while True:
                while len(active)<STREAMS:
                    job = next(jobs_iter,None)
                    if job is None:
                        break
                    active.add(pool.submit(pair,gpu,stamp,*job))
                if not active:
                    break
                completed,active = wait(active,return_when=FIRST_COMPLETED)
                for fut in completed:
                    q,h = fut.result()
                    done[q]=h
                progress()
    final.immutable(OUT/f'gpu{gpu}.complete.json',dict(configuration_sha256=stamp,rows=done))


def score(stamp):
    raw_labels = vr.read(base.FORMAL/'labels_private.json')
    design_doc = design(base.RESULTS)
    require(stamp==design_doc['configuration_sha256'], 'Scoring a different run')
    ids = design_doc['configuration']['ids']
    pairs, hashes = {}, {}
    for gpu in PORTS:
        receipt = vr.read(OUT/f'gpu{gpu}.complete.json')
        require(receipt['configuration_sha256']==stamp, 'Changed shard plan')
        for q,h in receipt['rows'].items():
            target=OUT/'rows'/q
            require(q not in pairs and vr.sha(target/'pair.json')==h, 'Duplicate or changed pair')
            record=vr.read(target/'pair.json')
            require(record['id']==q and record['configuration_sha256']==stamp, 'Foreign pair')
            for arm,r in record['arms'].items():
                if r['stage']=='CHECK':
                    check=target/arm/'result.json'
                    require(vr.sha(check)==r['check_sha256'] and
                            all(vr.sha(f)==s for f,s in vr.read(check)['files'].items()), 'Changed CHECK evidence')
            pairs[q]=record['arms']
            hashes[q]=h
    require(set(pairs)==set(ids), 'Missing paired outcomes')
    labels={q:raw_labels[q]['verdict'] for q in ids}
    result=scoring.analyze(labels,pairs,raw_labels[HOLD]['verdict'])
    final.immutable(OUT/'details.json',dict(rows=[dict(id=q,label=labels[q],arms=pairs[q],pair_sha256=hashes[q]) for q in ids]))
    final.immutable(OUT/'metrics.json',dict(registered=468,measured=len(ids),held=1,positive_class='incorrect',
                    human_gold=False,shared_prepare_failures=sum(x['pure']['stage']=='PREPARE' for x in pairs.values()),
                    check_calls=sum(r['stage']=='CHECK' for a in pairs.values() for r in a.values()),
                    design_sha256=vr.sha(OUT/'design.json'),details_sha256=vr.sha(OUT/'details.json'),**result))


def run(folder, stamp):
    refs=vr.read(OUT/'source_answers.json')['rows']
    rows=registered(base.RESULTS)
    jobs=[(i,r,refs[r['id']]) for i,r in enumerate(rows) if r['id']!=HOLD]
    with ProcessPoolExecutor(max_workers=len(PORTS),mp_context=multiprocessing.get_context('spawn')) as cards:
        tasks=[cards.submit(run_card,g,folder,stamp,jobs[i::len(PORTS)]) for i,g in enumerate(PORTS)]
        for task in tasks:
            task.result()
    score(stamp)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','confirm','score'))
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args()
    base.node.node()
    if args.action=='prepare':
        bind_source(base.RESULTS,base.host.ROOT/'data/combined_language_only.json')
    stamp=design(base.RESULTS)['configuration_sha256']
    if args.action=='score':
        score(stamp)
    elif args.action in ('run','confirm'):
        runtime.detached(OUT,args.attempt,stamp,lambda folder:run(folder,stamp),args.action=='confirm')
