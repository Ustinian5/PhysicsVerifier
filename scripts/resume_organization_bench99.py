"""Continue the frozen 99-case run after audited format-label schema failures.

No response is repaired, retried, or promoted to success. Original runner,
prompts, scoring and outputs remain immutable; this adds failure accounting.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from scripts import run_organization_bench99 as run

base, recovery, vr = run.base, run.recovery, run.vr
require = base.require
REASON = 'ValueError: Format label without corresponding complete JSON'
RECEIPT = 'format_schema_recovery.json'
ORIGINAL_INSPECT = recovery.inspect_failure


def schema_failure_usage(raw, request):
    from vllm.entrypoints import harmony_utils
    from scripts.resume_continuous_context_v2 import normalize_messages
    try:
        base.decode_response(raw, model=request['model'], max_tokens=request['max_tokens'], native_tools=None)
    except ValueError as exc:
        require('ValueError: ' + str(exc) == REASON, 'Different decoder failure')
    else:
        raise ValueError('Valid response must not be discarded')
    tokens = raw['choices'][0]['token_ids']
    require(tokens and tokens[-1] == 200002, 'Unterminated native response')
    parsed = normalize_messages(harmony_utils.parse_output_into_messages(tokens))
    finals = [m for m in parsed.messages if m.channel == 'final']
    require(not parsed.current_content and len(finals) == 1 and all(
        m.channel == 'final' or (m.channel == 'analysis' and not m.recipient)
        for m in parsed.messages), 'Unknown native message layout')
    final = finals[0]
    require(final.recipient in {'<|constrain|>diagnostics', '<|constrain|>quote_patches'},
            'Unknown format label')
    value = json.loads(''.join(part.text for part in final.content))
    key = final.recipient.split('>')[-1]
    require(not (isinstance(value, dict) and set(value) == {key} and isinstance(value[key], list)),
            'Valid schema must not be discarded')
    return raw['usage']


def inspect_failure(path):
    value = vr.read(path)
    if value.get('failure_reason') != REASON:
        return ORIGINAL_INSPECT(path)
    base.runtime.audit_result(value)
    require(value['status'] == 'failed' and value.get('diagnostics') == [], 'Not a preserved failure')
    pending = vr.read(path.with_name('state.json'))['pending']
    paths = [Path(pending[k]) for k in ('request_path', 'raw_path', 'response_path')]
    saved, raw, record = [vr.read(p) for p in paths]
    request = saved['request']
    require(saved['request_sha256'] == base.r.sha256_value(request) and
            request['model'] == base.r.EXPECTED_MODEL and not request.get('tools') and
            request['max_tokens'] == pending['max_tokens'] and 0 < request['max_tokens'] <= 16384,
            'Unexpected failed request')
    require(record.get('http_status') == 200 and record.get('error') == REASON and
            record['raw_sse_sha256'] == vr.sha(paths[1]) and raw.get('model') == base.r.EXPECTED_MODEL and
            record.get('model_ids') == [base.r.EXPECTED_MODEL] and len(raw.get('choices', [])) == 1 and
            raw['choices'][0]['finish_reason'] == 'stop' and raw.get('usage') == record.get('usage'),
            'Transport, identity or completion failure is not eligible')
    usage = schema_failure_usage(raw, request)
    require(str(paths[0]) not in {e['request_path'] for e in value['trace']['response_evidence']},
            'Response already accounted')
    require(usage['prompt_tokens'] > 0 and 0 <= usage['completion_tokens'] <= request['max_tokens'],
            'Invalid usage')
    return dict(id=value['id'], result_sha256=vr.sha(path), request_path=str(paths[0]), usage=usage,
                files={str(p): vr.sha(p) for p in [path, path.with_name('state.json'), *paths]},
                status='failed_content_preserved', failure_reason=REASON,
                new_model_calls=0, content_modified=False)


def prepare(output):
    with base.r.run_lock(output/'active.lock'), base.r.run_lock(output.parent/'organization_service_18083.lock'):
        require(not (output/'runner.pid').exists(), 'Existing runner')
        if (output/RECEIPT).exists():
            return verify(output)
        plan = run.verify(output)
        affected = []
        with patch.object(recovery, 'inspect_failure', inspect_failure):
            for phase in ('controls', 'formal'):
                for path in sorted((output/phase).glob('*/samples/C0p/*/result.json')):
                    if vr.read(path).get('failure_reason') == REASON:
                        affected.append(recovery.account(path))
        require(affected, 'No audited schema failure')
        files = {}
        for phase in ('controls', 'formal', 'queries'):
            for path in (output/phase).rglob('*'):
                if path.is_file() and path.name in {'result.json', 'state.json', 'request.json',
                    'response.json', 'response.sse', 'reference_packet.json', 'native_format_accounting.json'}:
                    files[str(path)] = vr.sha(path)
        vr.atomic(output/RECEIPT, dict(configuration_sha256=plan['configuration_sha256'],
            plan_sha256=vr.sha(output/'plan.json'), recovery_code_sha256=vr.sha(Path(__file__)),
            prefix_files=files, affected=affected, new_model_calls=0, content_modified=False,
            policy='All arms retain complete wrong-schema labeled finals as failures; no repair or retry.'))
        return plan


def verify(output):
    plan = run.verify(output)
    receipt = vr.read(output/RECEIPT)
    require(receipt['configuration_sha256'] == plan['configuration_sha256'] and
            receipt['plan_sha256'] == vr.sha(output/'plan.json') and
            receipt['recovery_code_sha256'] == vr.sha(Path(__file__)), 'Recovery binding drift')
    vr.EvidenceCache(output/'format_schema_recovery_cache.json').check(receipt['prefix_files'])
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify', 'run', 'score', 'confirm-detached'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    if args.mode == 'prepare':
        prepare(output); return
    plan = verify(output)
    with patch.object(recovery, 'inspect_failure', inspect_failure):
        if args.mode == 'run': run.run(output)
        elif args.mode == 'score': run.score(output, plan)
        elif args.mode == 'confirm-detached':
            with patch.object(base, 'verify', lambda path: plan): base.confirm(output)
    print(plan['configuration_sha256'])


if __name__ == '__main__': main()
