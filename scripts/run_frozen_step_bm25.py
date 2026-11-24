"""Run only the two first-review flat arms against the archived step BM25 release."""
import argparse
from collections import Counter
from pathlib import Path
import re
import time
from unittest.mock import patch

import httpx

from scripts import frozen_bm25_sources as sources
from scripts import run_aligned_step_revision as step
from scripts import serve_frozen_bm25 as node

base, vr, require = step.base, step.vr, step.require
write = sources.final.immutable
ROOT = sources.OUTPUT / 'step'
PREVIOUS = (sources.RESULTS / 'aligned_old99/evaluation_revision').resolve()
PARENT = (sources.RESULTS / 'aligned_old99/evaluation').resolve()
ARMS = ('case_rag', 'flat_skills')
ACCEPTANCE = '53bc6feaf2593d1c8ecda717e351b5418a2b6fec49a51a8a5d4256e716889005'
INPUTS = 'ebbbe347e2afe84fa9a8ae148fb1796979461e08363daf8eb5841f7e2926a327'
LABELS = 'd1b1aff4353c7b9728b06372262b9dbf95b62e9500a35de6f56bdf73ef899f3f'
PLAN = '94c615022ce87d64722b80011ce91f9d858792b3ea80757739d04e183b606ea4'
RECIPIENT_ERROR = 'ValueError: Unregistered actual Harmony recipient'


def unoffered_tool_output(parsed):
    """A completed tool yield without a final diagnosis in a tool-free CHECK."""
    messages = list(parsed.messages)
    return bool(messages and not parsed.current_content
                and all(m.channel == 'analysis' and not m.recipient for m in messages[:-1])
                and messages[-1].channel in ('analysis', 'commentary')
                and re.fullmatch(r'[a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)*', messages[-1].recipient or ''))


def inspect_check_failure(path):
    value = vr.read(path)
    if value.get('failure_reason') != RECIPIENT_ERROR:
        return step.old.a.native.inspect_failure(path)
    base.runtime.audit_result(value)
    require(value['status'] == 'failed' and value['diagnostics'] == []
            and value['configuration_sha256'] == verify()['configuration_sha256'], 'Unexpected failed outcome')
    state = vr.read(path.with_name('state.json'))
    turn = path.parent / 'turn_02'
    paths = [turn / n for n in ('request.json', 'response.sse', 'response.json')]
    require(state['turn'] == 1 and state['sample_id'] == value['id']
            and [state['pending'][k] for k in ('request_path', 'raw_path', 'response_path')] == list(map(str, paths)),
            'Expected the first CHECK immediately after the frozen PREPARE')
    envelope, raw, response = map(vr.read, paths)
    request = envelope['request']
    require(envelope['request_sha256'] == base.r.sha256_value(request) and 'tools' not in request
            and 'tool_choice' not in request and request['model'] == base.r.EXPECTED_MODEL
            and request['max_tokens'] == state['pending']['max_tokens']
            and response.get('http_status') == 200 and response.get('error') == RECIPIENT_ERROR
            and response.get('model_ids') == [request['model']] and raw['model'] == request['model']
            and response['raw_sse_sha256'] == vr.sha(paths[1]) and response['usage'] == raw['usage'],
            'Unverified request, transport, model or usage')
    try:
        base.decode_response(raw, model=request['model'], max_tokens=request['max_tokens'], native_tools=None)
    except ValueError as exc:
        require('ValueError: ' + str(exc) == RECIPIENT_ERROR, 'Different frozen-decoder error')
    else:
        raise ValueError('A valid diagnosis must not be discarded')
    from vllm.entrypoints.harmony_utils import parse_output_into_messages
    tokens = raw['choices'][0]['token_ids']
    require(tokens[-1] == 200012 and unoffered_tool_output(parse_output_into_messages(tokens))
            and len(raw['prompt_token_ids']) == raw['usage']['prompt_tokens']
            and raw['usage']['total_tokens'] == raw['usage']['prompt_tokens'] + len(tokens)
            and str(paths[0]) not in {e['request_path'] for e in value['trace']['response_evidence']},
            'Not an isolated, uncounted model tool yield')
    return dict(id=value['id'], result_sha256=vr.sha(path), request_path=str(paths[0]), usage=raw['usage'],
                files={str(p): vr.sha(p) for p in (path, path.with_name('state.json'), *paths)},
                status='failed_content_preserved', failure_reason=RECIPIENT_ERROR, new_model_calls=0,
                content_modified=False)


def verify(out=ROOT):
    p = vr.read(out / 'plan.json')
    require(vr.digest(p['configuration']) == p['configuration_sha256']
            and p['configuration']['input_sha256'] == vr.sha(PARENT / 'inputs.json') == INPUTS
            and p['configuration']['labels_sha256'] == vr.sha(PARENT / 'offline_review/labels.json') == LABELS
            and p['configuration']['sources_sha256'] == vr.sha(sources.OUTPUT / 'sources.json')
            and vr.sha(PREVIOUS / 'catalog.json') == sources.TREE, 'Step protocol or evidence changed')
    return p


def plan():
    release = vr.read(sources.OUTPUT / 'sources.json')
    require(vr.sha(PREVIOUS / 'acceptance.json') == ACCEPTANCE and vr.sha(PREVIOUS / 'plan.json') == PLAN
            and vr.read(PREVIOUS / 'acceptance.json')['passed'], 'Published step release changed')
    old = vr.read(PREVIOUS / 'plan.json')['configuration']
    cfg = {k: old[k] for k in ('api_base', 'limits', 'fixed_harmony_date', 'pilot_inputs', 'arms')}
    cfg.update(model=base.r.EXPECTED_MODEL, parent=str(PARENT), baseline=str(PREVIOUS), input_sha256=INPUTS,
               labels_sha256=LABELS, sources_sha256=vr.sha(sources.OUTPUT / 'sources.json'),
               material_files=release['groups']['n2000']['files'], previous_plan_sha256=PLAN,
               acceptance_sha256=ACCEPTANCE, frozen_tree_sha256=sources.TREE, step_hold=sorted(sources.STEP_HOLD),
               changed_arms=list(ARMS), retriever='original BM25', rag_binding=step.RAG_BINDING)
    write(ROOT / 'plan.json', dict(configuration=cfg, configuration_sha256=vr.digest(cfg)))
    return verify()


def materials():
    source = vr.read(sources.OUTPUT / 'sources.json')['groups']['n2000']
    folder = Path(source['material_path'])
    require(all(vr.sha(folder / n) == h for n, h in source['files'].items()), 'Changed material')
    return {a: base.p.FlatMemory(vr.read(folder / n)) for a, n in zip(ARMS, ('cases.json', 'skills.json'))}


def make_tasks(phase, rows, plan, tokenizer, memories):
    previous = {(t['id'], t['arm']): t for t in vr.read(PREVIOUS / 'tasks.json')}
    bound = vr.read(PREVIOUS / ('acceptance.json' if phase == 'formal' else 'pilot_acceptance.json'))['files']
    count = lambda m: len(tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True))
    tasks = []
    for row in rows:
        sid = row['id']
        prefix_parent = PARENT if phase == 'formal' else PREVIOUS
        query_path = prefix_parent / 'queries' / vr.digest(sid)[:24] / 'result.json'
        prefix = None
        if sid not in sources.STEP_HOLD and query_path.exists():
            prefix, _ = step.prefix_at(prefix_parent, row)
            require(all(vr.sha(k) == v for k, v in prefix['files'].items()), 'Original PREPARE drift')
        for arm in base.p.ARMS:
            if sid in sources.STEP_HOLD:
                local = base.path_for(ROOT, phase, arm, row)
                write(local, dict(id=sid, organization_arm=arm, status='scope_hold', diagnostics=[]))
                tasks.append(dict(id=sid, arm=arm, action='scope_hold', parent_result=str(local)))
                continue
            if phase == 'pilot':
                path = base.path_for(PREVIOUS, phase, arm, row)
                saved = vr.read(path)
            else:
                saved, path = step.result_for(PREVIOUS, previous[sid, arm])
            digest = vr.sha(path)
            known = bound.get(str(path))
            if known is None and phase == 'formal':
                known = previous[sid, arm]['parent_result_sha256'] if previous[sid, arm]['action'] != 'run_check' else None
            require(digest == known and saved['id'] == sid, 'Original result is not in the release')
            task = dict(id=sid, arm=arm, parent_result=str(path), parent_result_sha256=digest, action='reuse')
            if not prefix:
                require(saved.get('failure_stage') == 'prepare' and not saved['diagnostics'], 'Unknown prefix failure')
                task['action'] = 'retain_prepare_failure'
            elif arm in ARMS:
                new = step.packet(row, arm, prefix, memories[arm], count)
                body = vr.read(path.parent / 'turn_02/request.json')['request']
                maximum = min(plan['configuration']['limits']['max_tokens_per_response'],
                              plan['configuration']['limits']['output_tokens_per_sample'] - prefix['evidence']['usage']['completion_tokens'])
                expected = dict(model=base.r.EXPECTED_MODEL, messages=new['messages'], temperature=0, top_p=1,
                                max_tokens=maximum, stream=False, return_token_ids=True, reasoning_effort='medium')
                require(set(body) == set(expected) and all(body[k] == expected[k] for k in expected if k != 'messages'),
                        'Frozen CHECK settings differ')
                task['action'] = 'reuse' if body == expected else 'run_check'
                task['request_sha256'] = vr.digest(expected)
                write(ROOT / 'packets' / arm / (vr.digest(sid)[:24] + '.json'), new)
            tasks.append(task)
        vr.atomic(ROOT / (phase + '_preparation.json'), dict(prepared=len(tasks) // 4, total=len(rows)))
    write(ROOT / ('tasks.json' if phase == 'formal' else 'pilot_tasks.json'), tasks)
    return tasks


def collect(phase, tasks, rows, plan, tokenizer, memories, client):
    byid = {r['id']: r for r in rows}
    parent = PARENT if phase == 'formal' else PREVIOUS
    def frozen_query(_out, row, current, _tokenizer, _client):
        value, _ = step.prefix_at(parent, row)
        return dict(value, configuration_sha256=current['configuration_sha256'])
    controls, counts = {}, Counter()
    with step.revised(ROOT, plan, tokenizer), patch.object(base, 'query', frozen_query), \
            patch.object(step.old.a.bench.recovery, 'inspect_failure', inspect_check_failure):
        for i, t in enumerate(tasks, 1):
            require(not (sources.OUTPUT / 'STOP').exists(), 'STOP requested')
            sid, arm = t['id'], t['arm']
            if t['action'] == 'scope_hold':
                counts[t['action']] += 1
                continue
            if t['action'] == 'run_check':
                require(arm in ARMS and sid not in sources.STEP_HOLD, 'Unrequested CHECK')
                result = step.old.a.bench.run_condition(ROOT, byid[sid], phase, arm, memories[arm], plan, tokenizer, client)
                path = base.path_for(ROOT, phase, arm, byid[sid])
                require(vr.digest(vr.read(path.parent / 'turn_02/request.json')['request']) == t['request_sha256'],
                        'Actual CHECK differs from the frozen request')
            else:
                path = Path(t['parent_result'])
                require(vr.sha(path) == t['parent_result_sha256'], 'Archived baseline changed')
                result = vr.read(path)
            if phase == 'pilot':
                require(base.runtime.healthy_result(result) and result['protocol_audit']['passed'], 'Pilot engineering failure')
                if sid.startswith('control_'):
                    controls[f'{arm}/{sid}'] = bool(result['diagnostics']) == (sid == 'control_wrong_final')
            counts[t['action']] += 1
            vr.atomic(ROOT / (phase + '_progress.json'), dict(processed=i, total=len(tasks), counts=counts, unix=time.time()))
    if phase == 'pilot':
        require(len(controls) == 8 and all(controls.values()), 'Pilot control failure')
    write(ROOT / (phase + '_complete.json'), dict(passed=True, counts=counts, controls=controls,
          tasks=len(tasks), configuration_sha256=plan['configuration_sha256']))


def run(attempt, gpu):
    node.node()
    with base.r.run_lock(ROOT / 'runner.lock'):
        config = plan()
        base.runtime.wait_detached(attempt, config['configuration_sha256'])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(node.host.HOT / 'models/gpt-oss-120b', local_files_only=True)
        memories = materials()
        pilots = config['configuration']['pilot_inputs']
        inputs = vr.read(PARENT / 'inputs.json')
        require(len(pilots) == 6 and len(inputs) == 100 and sources.STEP_HOLD <= {r['id'] for r in inputs},
                'Benchmark drift')
        pilots_tasks = make_tasks('pilot', pilots, config, tokenizer, memories)
        formal_tasks = make_tasks('formal', inputs, config, tokenizer, memories)
        with node.service(attempt, gpu, sources.OUTPUT / 'STOP', config['configuration_sha256']) as endpoint:
            require(endpoint == config['configuration']['api_base'], 'Endpoint mismatch')
            with httpx.Client(trust_env=False) as client:
                collect('pilot', pilots_tasks, pilots, config, tokenizer, memories, client)
                collect('formal', formal_tasks, inputs, config, tokenizer, memories, client)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'confirm'])
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=4)
    args = parser.parse_args()
    attempt = ROOT / 'attempts' / str(args.attempt)
    if args.action == 'confirm':
        node.confirm(attempt, verify()['configuration_sha256'])
    else:
        attempt.mkdir(parents=True, exist_ok=True)
        try:
            run(attempt, args.gpu)
            vr.atomic(attempt / 'exit.json', dict(code=0, unix=time.time()))
        except Exception as error:
            vr.atomic(attempt / 'exit.json', dict(code=1, error=str(error), unix=time.time()))
            raise
