"""Recognize the two approved CHECK format spellings for blind step diagnoses."""
from contextlib import contextmanager
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_aligned_step_evaluation as step

base, vr, require = step.base, step.vr, step.require
AUDIT_SHA = '018d9fd5beb770b1928c3236d1a07910737ecee923939fe6495537f11b8861a3'
LABEL = '<|constrain|>CHECK'
LABELS = (LABEL, '<|constrain|>check')
PILOT_RECEIPT_SHA = '6b7905a83ee33d81d8fd74a0ad2e9f2940e0cc22ae35fca8f0696eacd850336c'
FORMAL_FAILURE_SHA = '1b82649237a586b9f53df5eeed10607fca6fb8065a11a57397266b050e51ad00'
FORMAL_RESPONSE_SHA = '195f70b005fdc177f029401f6b09f8f6d7e969e8e64724367eb55f65f0ffee7c'
CASE_RELEASE = 'check_format_case_release.json'


def normalize(parsed, tokens, provided):
    frames = list(parsed.messages)
    tagged = [m for m in frames if m.channel == 'final' and m.recipient in LABELS]
    if not tagged:
        return parsed
    require(not parsed.current_content and tokens[-1:] == [200002] and len(tagged) == 1
            and all(m is tagged[0] or (m.channel == 'analysis' and not m.recipient)
                    for m in frames), 'Incomplete or routed CHECK response')
    message = tagged[0]
    value = json.loads(''.join(part.text for part in message.content))
    require(isinstance(value, dict) and set(value) == {'diagnostics'}, 'Invalid tagged diagnosis object')
    base.r.validate_diagnostics(value)
    require(base.reference_audit(value, provided)['passed'], 'Tagged diagnosis cites unsupplied material')
    messages = [SimpleNamespace(channel=m.channel, recipient='<|constrain|>json', content=m.content)
                if m is message else m for m in frames]
    return SimpleNamespace(messages=messages, current_content=parsed.current_content)


@contextmanager
def label_decoder(provided):
    from vllm.entrypoints import harmony_utils
    parse = harmony_utils.parse_output_into_messages
    with patch.object(harmony_utils, 'parse_output_into_messages',
                      lambda tokens: normalize(parse(tokens), tokens, provided)):
        yield


def affected(out):
    audit = out/'check_format_audit.json'
    require(vr.sha(audit) == AUDIT_SHA, 'Different original step failure audit')
    item, = vr.read(audit)['affected']
    for name, sha in item['files'].items():
        require(vr.sha(out.parent/name) == sha, 'Original failed control changed')
    return item


def replay_check(out, plan, item, tokenizer):
    """Consume the saved pending turn with the ordinary publisher, with inference disabled."""
    arm, phase = item['group'], item['phase']
    rows = plan['configuration']['controls'] if phase == 'controls' else vr.read(out/'inputs.json')
    row, = [r for r in rows if r['id'] == item['id']]
    src = base.path_for(out, phase, arm, row).parent
    overlay = out/'check_format_replay'
    dest = base.path_for(overlay, phase, arm, row).parent
    if (dest/'result.json').is_file():
        result = vr.read(dest/'result.json')
    else:
        packet = vr.read(src/'reference_packet.json')['packet']
        prefix = vr.read(out/'queries'/vr.digest(row['id'])[:24]/'result.json')
        state = vr.read(src/'state.json')
        failed = vr.read(src/'result.json')
        require(failed['failure_reason'] == 'ValueError: Unregistered actual Harmony recipient'
                and state['turn'] == 1 and state['sample_id'] == row['id'], 'Wrong pending step CHECK')
        pending = state['pending']
        turn = src/'turn_02'
        require([Path(pending[k]) for k in ('request_path','raw_path','response_path')]
                == [turn/n for n in ('request.json','response.sse','response.json')], 'Different pending turn')
        request, raw, old = [vr.read(turn/n) for n in ('request.json','response.sse','response.json')]
        body = request['request']
        require(request['request_sha256'] == base.r.sha256_value(body) and body['messages'] == packet['messages']
                and not body.get('tools') and raw['model'] == body['model'] == base.r.EXPECTED_MODEL
                and old['http_status'] == 200 and old['raw_sse_sha256'] == vr.sha(turn/'response.sse'),
                'Different CHECK request or response')
        with step.a.configured(overlay, tokenizer), label_decoder(packet['retrieval']['provided_complete_ids']):
            decoded = base.runtime.transport.decode_response(raw, model=body['model'],
                                                             max_tokens=body['max_tokens'], native_tools=None)
        require(json.loads(decoded['content']) == item['diagnostic'] and not decoded.get('tool_calls')
                and decoded['usage'] == raw['usage'], 'Changed offline diagnosis or token accounting')
        response = dict(old, **decoded)
        response.pop('error')
        pending['response_path'] = str(dest/'turn_02/response.json')
        vr.atomic(Path(pending['response_path']), response)
        state.pop('last_error', None)
        vr.atomic(dest/'state.json', state)
        at_failure = (src/'result.json').stat().st_mtime
        with base.checking(row, arm, packet, prefix, english=True), \
                patch.object(base.r, 'time', SimpleNamespace(time=lambda: at_failure)), \
                patch.object(base.runtime.transport, 'chat_response', side_effect=AssertionError('Offline replay')):
            result = base.r.run_sample(row, arm='C0p', catalog={'domains': []}, tokenizer=tokenizer, client=None,
                                       endpoint='', model=body['model'], root=overlay/phase/arm,
                                       signature=plan['configuration_sha256'], limits=plan['configuration']['limits'])
    base.runtime.audit_result(result)
    require(base.runtime.healthy_result(result) and result['protocol_audit']['passed']
            and result['diagnostics'] == item['diagnostic']['diagnostics'] and result['trace']['model_requests'] == 2,
            'Step CHECK did not replay from its original two responses')
    return dest/'result.json'


@contextmanager
def checks(out, item, replay_path):
    original = step.check_published

    def check(root, row, phase, arm, memory, plan, tokenizer, client):
        if (phase, arm, row['id']) == (item['phase'], item['group'], item['id']):
            result = vr.read(replay_path)
            base.runtime.audit_result(result)
            return result
        first = base.path_for(root, phase, arm, row).parent/'turn_02/request.json'
        send = base.runtime.transport.chat_response

        def transport(*args, **kwargs):
            if Path(kwargs['request_path']) == first and not kwargs.get('native_tools'):
                packet = vr.read(first.parent.parent/'reference_packet.json')['packet']
                require(kwargs['messages'] == packet['messages'], 'Different CHECK messages')
                with label_decoder(packet['retrieval']['provided_complete_ids']):
                    return send(*args, **kwargs)
            return send(*args, **kwargs)

        with patch.object(base.runtime.transport, 'chat_response', transport):
            return original(root, row, phase, arm, memory, plan, tokenizer, client)

    with patch.object(step, 'check_published', check):
        yield


def original_protocol(out):
    from scripts import step_judge_release as release
    receipt = vr.read(out/'check_format_protocol.json')
    item = affected(out)
    require((vr.sha(out/'check_format_protocol.json') == PILOT_RECEIPT_SHA
             or receipt['code_sha256'] == vr.sha(Path(__file__))) and receipt['audit_sha256'] == AUDIT_SHA
            and receipt['plan_sha256'] == vr.sha(out/'plan.json') == vr.read(out/'check_format_audit.json')['step_plan_sha256']
            and vr.sha(out.parent/'step_judge_lines/pure/first_pass.json') == release.PURE_FIRST_PASS,
            'Different step protocol or pure baseline')
    for name, sha in receipt['retained_files'].items():
        require(vr.sha(out/name) == sha, 'Changed retained or replayed control')
    return receipt, item


def verified(out):
    receipt, item = original_protocol(out)
    parent_hash = vr.sha(out/'check_format_protocol.json')
    if parent_hash != PILOT_RECEIPT_SHA:
        return receipt, item
    change = vr.read(out/CASE_RELEASE)
    require(change['parent_sha256'] == parent_hash and change['code_sha256'] == vr.sha(Path(__file__))
            and change['labels'] == list(LABELS), 'Different CHECK spelling release')
    for name, expected in change['files'].items():
        require(vr.sha(out/name) == expected, 'Changed step recovery evidence: ' + name)
    return dict(receipt, case_release_sha256=vr.sha(out/CASE_RELEASE)), item


def publish_replay(path, replayed, failed_sha):
    """Preserve the initial failure and atomically publish the replay at the normal scoring path."""
    initial = path.with_name('initial_decode_failure.json')
    if not initial.exists():
        require(vr.sha(path) == failed_sha, 'Expected the original failed result')
        initial.hardlink_to(path)
    require(vr.sha(initial) == failed_sha, 'Changed initial decode failure')
    current, new = vr.sha(path), vr.sha(replayed)
    if current != new:
        require(current == failed_sha, 'Refusing to replace a different step result')
        temp = path.with_name('.check_replay.tmp')
        if not temp.exists():
            temp.hardlink_to(replayed)
        require(vr.sha(temp) == new, 'Changed staged replay')
        temp.replace(path)
    return initial


def recover_formal(out, tokenizer):
    from scripts.run_final_answer_revision import immutable
    plan = step.verify(out)
    paths = sorted((out/'formal').glob('*/samples/C0p/*/result.json'))
    retained, affected_paths = {}, []
    for path in paths:
        base.runtime.audit_result(vr.read(path))
        initial = path.with_name('initial_decode_failure.json')
        if vr.sha(initial if initial.exists() else path) == FORMAL_FAILURE_SHA:
            affected_paths.append(path)
        else:
            retained[str(path.relative_to(out))] = vr.sha(path)
    require(len(retained) == 127 and len(affected_paths) == 1, 'Different stopped formal prefix')
    path, = affected_paths
    src = path.parent
    raw = src/'turn_02/response.sse'
    require(vr.sha(raw) == FORMAL_RESPONSE_SHA, 'Different stopped formal CHECK response')
    arm = path.relative_to(out).parts[1]
    sid = vr.read(path)['id']
    row, = [r for r in vr.read(out/'inputs.json') if r['id'] == sid]
    require(sid in plan['configuration']['sample_ids'] and arm == 'case_rag'
            and path == base.path_for(out, 'formal', arm, row), 'Wrong stopped formal input')
    item = dict(id=sid, group=arm, phase='formal', diagnostic={'diagnostics': []})
    replayed = replay_check(out, plan, item, tokenizer)
    result = vr.read(replayed)
    require(result['organization_arm'] == arm and result['configuration_sha256'] == plan['configuration_sha256'],
            'Recovered CHECK has a different group or plan')
    step.verify_blind_delivery(out,'formal',row,arm,base.p.FlatMemory(vr.read(out/'cases.json')),tokenizer,result)
    accepted = out/'engineering_acceptance.json'
    acceptance = vr.read(accepted)
    require(acceptance['passed'] and acceptance['checks'] == 18
            and acceptance['configuration_sha256'] == plan['configuration_sha256'], 'Different accepted fixed pilot')
    initial = publish_replay(path, replayed, FORMAL_FAILURE_SHA)
    files = dict(retained)
    for evidence in (initial, path, replayed, accepted, src/'state.json', src/'reference_packet.json',
                     raw, raw.with_name('request.json'), raw.with_name('response.json')):
        files[str(evidence.relative_to(out))] = vr.sha(evidence)
    immutable(out/CASE_RELEASE, dict(parent_sha256=vr.sha(out/'check_format_protocol.json'),
        code_sha256=vr.sha(Path(__file__)), labels=list(LABELS), id=sid, group=arm,
        replay_result=str(path.relative_to(out)), retained_other_results=len(retained),
        new_model_calls_for_replay=0, files=files))


def prepare(out):
    from transformers import AutoTokenizer
    from scripts import serve_frozen_bm25 as node
    from scripts.run_final_answer_revision import immutable
    with base.r.run_lock(out/'active.lock'):
        if (out/'check_format_protocol.json').is_file():
            original_protocol(out)
            if vr.sha(out/'check_format_protocol.json') == PILOT_RECEIPT_SHA and not (out/CASE_RELEASE).exists():
                tokenizer = AutoTokenizer.from_pretrained(node.host.HOT/'models/gpt-oss-120b',local_files_only=True)
                recover_formal(out, tokenizer)
            return verified(out)
        plan, item = step.verify(out), affected(out)
        tokenizer = AutoTokenizer.from_pretrained(node.host.HOT/'models/gpt-oss-120b',local_files_only=True)
        row, = [r for r in plan['configuration']['controls'] if r['id'] == item['id']]
        saved = {}
        for path in (out/'controls').rglob('result.json'):
            result = vr.read(path)
            base.runtime.audit_result(result)
            if base.runtime.healthy_result(result):
                saved[str(path.relative_to(out))] = vr.sha(path)
            else:
                require(path == base.path_for(out,item['phase'],item['group'],row), 'Unreviewed step control')
        require(len(saved) == 7 and not (out/'formal').exists(), 'Different stopped step pilot')
        path = replay_check(out, plan, item, tokenizer)
        step.verify_blind_delivery(out, item['phase'], row, item['group'],
                                   base.p.FlatMemory(vr.read(out/'cases.json')), tokenizer, vr.read(path))
        saved[str(path.relative_to(out))] = vr.sha(path)
        immutable(out/'check_format_protocol.json', dict(
            code_sha256=vr.sha(Path(__file__)), plan_sha256=vr.sha(out/'plan.json'), audit_sha256=AUDIT_SHA,
            policy='final CHECK or check label with an intact schema-valid diagnosis and supplied citations',
            replay_result=str(path.relative_to(out)), reused_successes=7, new_replay_requests=0, retained_files=saved))
        return verified(out)


def main():
    from scripts import serve_frozen_bm25 as node
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare','run','confirm','score'))
    args = p.parse_args()
    node.node()
    out = node.host.HOT/'results/qwen30b_step_verifier/english_blind_verifier/step_formal'
    if args.command == 'prepare':
        prepare(out)
        return
    receipt, item = verified(out)
    signature = vr.digest(receipt)
    if args.command == 'confirm':
        node.confirm(out, signature)
    elif args.command == 'score':
        step.score(out)
    else:
        with base.r.run_lock(out/'check_format_runner.lock'):
            code = 1
            try:
                def resume(folder, _):
                    base.runtime.wait_detached(folder, signature)
                    with (folder/'service.log').open('a') as log:
                        print(time.strftime('%Y-%m-%d %H:%M:%S %z'), 'Starting the step verifier service', file=log)
                with checks(out, item, out/receipt['replay_result']):
                    step.run(out, gpu=4, wait_detached=resume)
                report = step.score(out)
                require(len(report['first_pass']) == 4 and all(n==99 for n in report['recorded_tasks'].values()),
                        'Missing step cohort or judgment')
                code = 2 if report['groups_for_review'] else 0
            finally:
                vr.atomic(out/'check_format_exit.json',dict(code=code, configuration_sha256=signature))
            return code


if __name__ == '__main__':
    raise SystemExit(main())
