"""Paired step verification with the current problem's Combined Language exam answer."""
import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import httpx

from rule_framework.oss120b_construction import question_key
from scripts import run_frozen_step_bm25 as frozen
from scripts import step_retrieval_packets as packets
from scripts import timed_retrieval_runtime as runtime
from scripts.combined_language_io import iter_rollout_batches
from scripts.combined_language_samples import extract_question

vr, require, write = frozen.vr, frozen.require, frozen.write
ROOT = frozen.sources.RESULTS
TRIAL = Path('step_exam_reference')
FIELD = 'Official exam reference answer for the current problem'
DATA_SHA = 'c1c2401fb8dd8a730dd3777f5d7103bcd4c1a2750f42b5b58911ec2b49da7b24'
TASKS_SHA = '98e966e35defc4912e7b98a2eccb9ec255bb7adb69ed9918cafeff480353736c'
OLD_METRICS = 'f96adeaf66db3ed0f000fbfb5f5832820227fb27761b9ac7051d66b68cd4b0b5'
GPUS = (5, 6, 7)


def inputs(root):
    path = root/'aligned_old99/evaluation/inputs.json'
    require(vr.sha(path) == frozen.INPUTS, 'Different frozen 30B answers')
    rows = [r for r in vr.read(path) if r['id'] not in frozen.sources.STEP_HOLD]
    require(len(rows) == len({r['id'] for r in rows}) == 99, 'Different step cohort')
    return rows


def source_answers(root, source):
    """Read the raw label, including partial answers, at its recorded rollout position."""
    rows = inputs(root)
    source_index = vr.read(root/'corrected/source_screen/references_private.json')
    require(vr.sha(source) == source_index['source_sha256'] == DATA_SHA, 'Different Combined Language file')
    inventory = vr.read(root/'source_question_inventory.json')['questions']
    ids = {question_key(r['question']): r['id'] for r in inventory}
    variants = {r['id']: r['variants'] for r in source_index['references']}
    wanted = defaultdict(list)
    for row in rows:
        key = question_key(row['question'])
        source_id = ids[key]
        choices = variants[source_id]
        if len(choices) == 1:
            loc = choices[0]['first_source']
            wanted[loc['rollout_id']].append((row, key, source_id, loc['sample_index'], loc['sample_position']))
        else:
            # A historical cl_<rollout>_<index> identifies the exact raw record even
            # when a different occurrence of the same question has a different label.
            tag, batch, index = row['id'].split('_')
            require(tag == 'cl', 'Cannot identify the original source variant')
            wanted[int(batch)].append((row, key, source_id, int(index), None))
    found = {}
    for batch in iter_rollout_batches(source):
        for row, key, sid, index, pos in wanted.pop(batch['rollout_id'], []):
            if pos is None:
                pos, = [i for i,s in enumerate(batch['samples']) if s.get('index') == index]
            sample = batch['samples'][pos]
            value = sample.get('label')
            require(sample.get('index') == index and question_key(extract_question(sample)) == key
                    and value in [v['value'] for v in variants[sid]], 'Question/answer source mismatch')
            require(isinstance(value,list) and value and all(isinstance(x,str) for x in value)
                    and any(x.strip() for x in value), 'No usable source reference')
            found[row['id']] = dict(source_id=sid, rollout=batch['rollout_id'], index=index, position=pos,
                                    label=value, all_parts_present=all(x.strip() for x in value),
                                    input_sha256=vr.digest(row))
        if not wanted:
            break
    require(set(found) == {r['id'] for r in rows}, 'Incomplete source lookup')
    write(root/TRIAL/'source_answers.json', dict(dataset_sha256=DATA_SHA, answers=found))


def design(root):
    out = root/TRIAL
    arch = root/'library_source_rerun_20261124/bm25_n2000/step'
    rows = inputs(root)
    labels = root/'aligned_old99/evaluation/offline_review/labels.json'
    require(vr.sha(arch/'tasks.json') == TASKS_SHA and vr.sha(arch/'metrics_99.json') == OLD_METRICS
            and vr.sha(labels) == frozen.LABELS, 'Different controls or step annotation')
    source = vr.read(out/'source_answers.json')
    require(source['dataset_sha256'] == DATA_SHA and set(source['answers']) == {r['id'] for r in rows}
            and all(source['answers'][r['id']]['input_sha256'] == vr.digest(r) for r in rows), 'Unbound answer')
    cfg = dict(ids=[r['id'] for r in rows], inputs=frozen.INPUTS, labels=frozen.LABELS,
               source=vr.sha(out/'source_answers.json'), controls=TASKS_SHA, control_metrics=OLD_METRICS,
               field=FIELD, model=frozen.base.r.EXPECTED_MODEL, date=frozen.node.CHECK_DATE,
               code=vr.sha(__file__), gpus=list(GPUS), human_gold=False,
               comparison='Frozen pure PREPARE and pure CHECK control. Add only the raw current-problem label '
                          'to the final user CHECK JSON. All prior messages, output cap, decode and quote repair match.',
               primary='99 paired questions, frozen active step roots. F1=2 hits/(locatable diagnoses+roots). '
                       'Keep shared PREPARE failures and partial source answers. Blind root/direction and quote review.',
               inference='Two-sided paired question-level swap test of delta F1 at 0.05; exact up to 18 informative '
                         'pairs, otherwise 100000 draws. Paired 10000-draw question bootstrap CI. Seed 20261126.')
    result = dict(configuration=cfg, configuration_sha256=vr.digest(cfg))
    write(out/'design.json', result)
    return result


def add_answer(messages, answer):
    head, separator, instructions = messages[-1]['content'].partition('\n')
    fields = json.loads(head)
    require(separator and fields == {'stage':'CHECK', 'Reference material':[]}, 'Expected an empty pure CHECK')
    fields[FIELD] = answer
    content = json.dumps(fields, ensure_ascii=False).replace('<|', '\\u003c|') + separator + instructions
    return [*messages[:-1], dict(role='user', content=content)]


def check_one(work, row, answer, task, plan, tokenizer, client):
    old_path = Path(task['parent_result'])
    require(vr.sha(old_path) == task['parent_result_sha256'], 'Pure control changed')
    base = frozen.base
    sid = row['id']
    if task['action'] == 'retain_prepare_failure':
        old = vr.read(old_path)
        require(old['failure_stage'] == 'prepare' and not old['diagnostics'], 'Unexpected shared failure')
        return dict(path=str(old_path), sha256=vr.sha(old_path), shared_failure=True, new_requests=0)
    require(task['action'] == 'reuse', 'Unexpected pure control')
    prefix, prefix_hash, prepared = packets.prefix_for(frozen.PARENT,row)
    old_req = vr.read(old_path.parent/'turn_02/request.json')['request']
    count = lambda m: len(tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True))
    packet = base.p.reference_packet(row, 'pure', prefix['queries'], None, count)
    require(packet['messages'] == old_req['messages'], 'Cannot reproduce the frozen pure request')
    messages = add_answer(packet['messages'], answer)
    expected = vr.digest(dict(old_req, messages=messages))
    # The native engine appends subsequent assistant and quote-repair turns in place.
    packet['messages'] = list(messages)
    limits = dict(plan['configuration']['limits'])
    limits['wall_seconds_per_sample'] -= prepared['wall_seconds']
    cfg = dict(plan, configuration=dict(plan['configuration'], limits=limits))
    with frozen.step.old.a.configured(work,tokenizer), patch.object(base,'query',return_value=prefix), \
            patch.object(base.p,'reference_packet',return_value=packet):
        result = frozen.step.old.a.bench.run_condition(work,row,'formal','pure',None,cfg,tokenizer,client)
    path = base.path_for(work,'formal','pure',row)
    turn = path.parent/'turn_02'
    require(vr.digest(vr.read(turn/'request.json')['request']) == expected
            and result['id'] == sid and result['configuration_sha256'] == plan['configuration_sha256'],
            'New CHECK differs in more than the current problem answer')
    raw = vr.read(turn/'response.sse')
    actual = tokenizer.decode(raw['prompt_token_ids'],skip_special_tokens=False)
    require(all(m['content'] in actual for m in messages), 'Model did not receive the complete exam answer or prompt')
    turns = list(path.parent.glob('turn_*/request.json'))
    return dict(path=str(path),sha256=vr.sha(path),control_sha256=task['parent_result_sha256'],
                prefix_sha256=prefix_hash, check_body_sha256=expected,
                actual_input_sha256=vr.digest(raw['prompt_token_ids']), new_requests=len(turns),shared_failure=False)


def run_shard(gpu, folder, stamp):
    out = ROOT/TRIAL
    slot = next(k for k,v in runtime.GPUS.items() if v == gpu)
    rows = inputs(ROOT)[GPUS.index(gpu)::len(GPUS)]
    work = out/f'gpu{gpu}'
    work.mkdir(exist_ok=True)
    receipt = work/'complete.json'
    if receipt.is_file():
        done = vr.read(receipt)
        require(done['signature'] == stamp and all(vr.sha(r['path']) == r['sha256'] for r in done['results'].values()),
                'Changed completed shard')
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(frozen.node.host.HOT/'models/gpt-oss-120b',local_files_only=True)
    tasks = {r['id']:r for r in vr.read(frozen.ROOT/'tasks.json') if r['arm']=='pure'}
    answers = vr.read(out/'source_answers.json')['answers']
    plan = runtime.checking_plan(slot,stamp)
    plan['configuration_sha256'] = stamp
    require(plan['configuration']['limits'] == frozen.verify()['configuration']['limits'], 'Different native step budget')
    service_dir = folder/f'gpu{gpu}'
    service_dir.mkdir()
    results = {}
    with runtime.large_service(service_dir,slot,stamp,out/'STOP') as endpoint, httpx.Client(trust_env=False) as client:
        require(endpoint == plan['configuration']['api_base'], 'Wrong private service')
        for row in rows:
            require(not (out/'STOP').exists(), 'Stop requested')
            sid = row['id']
            results[sid] = check_one(work,row,answers[sid]['label'],tasks[sid],plan,tok,client)
            vr.atomic(work/'progress.json', dict(completed=len(results),total=len(rows),unix=time.time(),results=results))
    write(receipt,dict(signature=stamp,results=results))


def run(folder, stamp):
    with ProcessPoolExecutor(max_workers=len(GPUS)) as workers:
        futures = [workers.submit(run_shard,gpu,folder,stamp) for gpu in GPUS]
        for future in futures:
            future.result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','confirm'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--source',type=Path,default=frozen.node.host.ROOT/'data/combined_language_only.json')
    parser.add_argument('--attempt',type=int,default=1)
    args = parser.parse_args()
    if args.action == 'prepare':
        source_answers(args.root,args.source)
    stamp = design(args.root)['configuration_sha256']
    if args.action != 'prepare':
        require(args.root == ROOT, 'Run on node14')
        runtime.detached(ROOT/TRIAL,args.attempt,stamp,lambda folder:run(folder,stamp),args.action=='confirm')
