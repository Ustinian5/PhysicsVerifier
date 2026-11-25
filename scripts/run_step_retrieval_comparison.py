"""Run paired BM25 and semantic retrieval with the original step CHECK and quote repair."""
import argparse
import time
from pathlib import Path
from unittest.mock import patch

import httpx

from scripts import step_retrieval_packets as packets

f, t, vr, require, ROOT = packets.frozen, packets.timed, packets.vr, packets.require, packets.ROOT
ARMS = t.cohort.ARMS
MODES = ('bm25', 'fusion')
write = f.write


def design():
    parent = f.verify()
    rows = vr.read(f.PARENT / 'inputs.json')
    sources = t.b.RESULTS / t.cohort.RESULT / t.NAME / 'design.json'
    require(vr.sha(f.ROOT / 'metrics_99.json') == 'f96adeaf66db3ed0f000fbfb5f5832820227fb27761b9ac7051d66b68cd4b0b5'
            and vr.sha(f.ROOT / 'tasks.json') == '98e966e35defc4912e7b98a2eccb9ec255bb7adb69ed9918cafeff480353736c'
            and vr.sha(sources) == vr.read(sources.with_name('metrics_50.json'))['design_sha256'],
            'Changed published step or retrieval release')
    ids = [r['id'] for r in rows if r['id'] not in f.sources.STEP_HOLD]
    require(len(set(ids)) == len(ids) == 99, 'Step population changed')
    cfg = dict(parent=str(f.PARENT), parent_plan=vr.sha(f.ROOT/'plan.json'), step_tasks=vr.sha(f.ROOT/'tasks.json'),
               formal=ids, pilot=[r['id'] for r in parent['configuration']['pilot_inputs']], selector=vr.sha(sources),
               sources=vr.sha(f.sources.OUTPUT/'sources.json'), labels=f.LABELS, inputs=f.INPUTS,
               limits=parent['configuration']['limits'], arms=ARMS, modes=MODES,
               protocol='Frozen step PREPARE, original full-text CHECK and quote repair. 883 eligible leaves in both arms. BM25 first CHECK uses exactly the fusion output cap on the same question. Missing PREPAREs retain the archived failure. Selection content failures stay in the formal denominator.',
               scoring='Reuse frozen step labels, blind evidence review and original location scoring. Pure control is archived.',
               code={Path(mod).name:vr.sha(mod) for mod in (__file__, packets.__file__, t.__file__,
                     f.__file__,f.step.__file__, t.semantic.__file__, t.runtime.__file__, t.PROCESSOR)})
    # JSON arrays are also used when validating an existing design after reconnect.
    cfg.update(arms=list(ARMS), modes=list(MODES))
    repair = ROOT/'check_replay_repair.json'
    if repair.is_file():
        receipt = vr.read(repair)
        name = Path(__file__).name
        require(receipt['original_design_sha256']==vr.sha(ROOT/'design.json')
                == '430c7c4b88dcbcf8a5d163b3231548e14b947e95b254b9198aa59a6ad1d3cfe9'
                and receipt['old_code_sha256']=='9be03c4e854442f488edd4d4eeefe5617767b625ee324d65bde62db149063577'
                and receipt['new_code_sha256']==cfg['code'][name], 'Unbound CHECK verification repair')
        cfg['code'][name] = receipt['old_code_sha256']
    signature = vr.digest(cfg)
    write(ROOT/'design.json', dict(configuration=cfg,configuration_sha256=signature))
    return signature


def prior_task(task, phase):
    if task['action']=='run_check':
        return f.base.path_for(f.ROOT, phase, task['arm'], dict(id=task['id']))
    path = Path(task['parent_result'])
    if task.get('parent_result_sha256'):
        require(vr.sha(path)==task['parent_result_sha256'], 'Frozen baseline changed')
    return path


def row_pair(row, arm, phase, tasks, index, tok, client, plan):
    sid = row['id']
    parent = f.PARENT if phase == 'formal' else f.PREVIOUS
    dest = ROOT/'retrieval'/arm/phase/sid
    task = tasks[sid,arm]
    stamp = plan['configuration_sha256']
    record = dest/'pair.json'
    if record.exists():
        value = vr.read(record)
        require(value['signature']==stamp and all(vr.sha(p)==h for p,h in value['results'].items()), 'Pair changed')
        return value
    options = {'english': True} if f.base.en.active(plan['configuration']) else {}
    prefix, prefix_hash, prepared = packets.prefix_for(parent, row, **options)
    limits = plan['configuration']['limits']
    budget = t.base.previous.Budget(ROOT,plan,tok,client)
    selection = packets.select(dest, prefix, prepared, arm, index, budget, **options)
    write(dest/'selection.json',selection)
    charges = selection['charges']
    results, caps, seconds, statuses = {}, {}, {}, {}
    for mode in MODES:
        require(not (ROOT/'STOP').exists(), 'STOP requested')
        work = ROOT/mode
        path = f.base.path_for(work, phase, arm, row)
        if mode=='fusion' and selection['failure']:
            require(phase=='formal', 'Pilot selection failure')
            write(path,dict(id=sid,organization_arm=arm,status='selection_content_failure',diagnostics=[],
                            configuration_sha256=stamp,reason=selection['failure']))
            results[str(path)] = vr.sha(path)
            statuses[mode] = 'selection_content_failure'
            continue
        packet = packets.packet(row,arm,prefix,selection[mode],index,budget.count,limits, **options)
        if mode=='bm25':
            old = vr.read(prior_task(task,phase))
            ids = (old.get('retrieval') or {}).get('provided_complete_ids',old.get('trace',{}).get('read_rule_ids',[]))
            supplied = packet['retrieval']['provided_complete_ids']
            original_request = vr.read(prior_task(task,phase).parent/'turn_02/request.json')['request']
            if arm!='rule_tree' or set(ids)<=set(index.cards):
                require(packet['messages']==original_request['messages'], 'Control packet differs: '+sid)
            else:
                require(not set(supplied)-set(index.cards), 'Excluded leaf sent')
        remaining = packets.check_limits(limits, prepared, charges, mode)
        cfg = dict(plan['configuration'], limits=remaining)
        local_plan = dict(plan,configuration=cfg)
        # The native step engine appends its response to the input message list in place.
        expected_messages = vr.digest(packet['messages'])
        with f.step.old.a.configured(work,tok), patch.object(f.base,'query',return_value=prefix), \
                patch.object(f.base.p,'reference_packet',return_value=packet):
            result = f.step.old.a.bench.run_condition(work,row,phase,arm,None,local_plan,tok,client)
        require(result['id']==sid and result['configuration_sha256']==stamp,'CHECK identity drift')
        turn = path.parent/'turn_02'
        body = vr.read(turn/'request.json')['request']
        require(vr.digest(body['messages'])==expected_messages and body['max_tokens']==remaining['output_tokens_per_sample']-
                prepared['usage']['completion_tokens'] and 'vllm_xargs' not in body, 'Step CHECK protocol changed')
        if phase=='pilot':
            require(f.base.runtime.healthy_result(result) and result['protocol_audit']['passed'], 'Pilot CHECK failure')
            if sid.startswith('control_'):
                require(bool(result['diagnostics'])==(sid=='control_wrong_final'), 'Pilot control failed')
        results[str(path)] = vr.sha(path)
        statuses[mode] = result['status']
        caps[mode] = body['max_tokens']
        seconds[mode] = sum(vr.read(p)['elapsed_seconds'] for p in path.parent.glob('turn_*/response.json'))
    require(len(set(caps.values()))<=1,'Unmatched first CHECK output caps')
    saved = dict(id=sid,signature=stamp,results=results,status=statuses,check_output_caps=caps,
                 check_seconds=seconds,prefix_sha256=prefix_hash,selection_sha256=vr.sha(dest/'selection.json'))
    write(record,saved)
    return saved


def run_phase(folder, phase, arm, stamp):
    from transformers import AutoTokenizer
    receipt = ROOT/'receipts'/arm/(phase+'.json')
    if receipt.exists():
        require(vr.read(receipt)['signature']==stamp, 'Different completed phase')
        return
    cfg = f.verify()['configuration']
    rows = cfg['pilot_inputs'] if phase=='pilot' else vr.read(f.PARENT/'inputs.json')
    tasks = {(z['id'],z['arm']):z for z in vr.read(f.ROOT/('pilot_tasks.json' if phase=='pilot' else 'tasks.json'))}
    targets = [r for r in rows if r['id'] not in f.sources.STEP_HOLD]
    plan = t.runtime.checking_plan(arm,stamp)
    require(plan['configuration']['limits']==cfg['limits'],'Step budget drift')
    plan['configuration_sha256'] = stamp
    tok = AutoTokenizer.from_pretrained(t.b.host.HOT/'models/gpt-oss-120b',local_files_only=True)
    t.rank_output.number_tokens(tok)
    index = t.base.load_index(t.b.RESULTS,arm)
    folder.mkdir(parents=True,exist_ok=True)
    done = {}
    with t.runtime.large_service(folder,arm,stamp,ROOT/'STOP',rank_output=True), httpx.Client(trust_env=False) as client:
        for n,row in enumerate(targets,1):
            require(not (ROOT/'STOP').exists(),'STOP requested')
            sid = row['id']
            if tasks[sid,arm]['action']=='retain_prepare_failure':
                path = prior_task(tasks[sid,arm],phase)
                require(vr.read(path)['failure_stage']=='prepare','Missing frozen PREPARE failure')
                done[sid]=dict(prepare_failure=True,results={str(path):vr.sha(path)})
            else:
                pair=row_pair(row,arm,phase,tasks,index,tok,client,plan)
                done[sid]=dict(pair_sha256=vr.sha(ROOT/'retrieval'/arm/phase/sid/'pair.json'),results=pair['results'])
            vr.atomic(folder/'progress.json',dict(phase=phase,visited=n,target=len(targets),id=sid,unix=time.time()))
    write(receipt,dict(signature=stamp,phase=phase,rows=done,release_sha256=vr.sha(folder/'release.json')))


def run(folder, arm, stamp):
    run_phase(folder/'pilot','pilot',arm,stamp)
    while not all((ROOT/'receipts'/a/'pilot.json').is_file() for a in ARMS):
        require(not (ROOT/'STOP').exists(),'STOP requested')
        for a in ARMS:
            p = ROOT/a/'attempts'/folder.name/'exit.json'
            if p.is_file():
                if vr.read(p)['code'] and not (ROOT/'receipts'/a/'pilot.json').exists():
                    raise RuntimeError('Another pilot requires recovery')
        time.sleep(10)
    require(all(vr.read(ROOT/'receipts'/a/'pilot.json')['signature']==stamp for a in ARMS),'Pilot design mismatch')
    run_phase(folder/'formal','formal',arm,stamp)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','confirm'))
    parser.add_argument('arm',choices=ARMS,nargs='?')
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args()
    t.b.node.node()
    stamp=design()
    if args.action!='prepare':
        require(args.arm is not None,'Missing retrieval arm')
        t.runtime.detached(ROOT/args.arm,args.attempt,stamp,lambda folder:run(folder,args.arm,stamp),args.action=='confirm')
