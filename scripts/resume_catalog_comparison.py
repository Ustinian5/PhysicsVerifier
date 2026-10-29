"""Resume C4 after one audited malformed tool JSON; never retry that sample."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch
from scripts import run_catalog_comparison as x

RAW_SHA = '24c7d606881f263808d6302e7a2853fbbbf6e22d98cd99c088c999719fb865df'
REL = 'formal/old/samples/C4/b825a35bcbe1325f/result.json'


def audit(output):
    x.verify(output)
    path = output / REL
    item = x.vr.read(path)
    state = x.vr.read(path.parent / 'state.json')
    pending = state['pending']
    raw_path, response_path = Path(pending['raw_path']), Path(pending['response_path'])
    raw, response = x.vr.read(raw_path), x.vr.read(response_path)
    request_record = x.vr.read(Path(pending['request_path']))
    request = request_record['request']
    x.require(x.vr.sha(raw_path) == RAW_SHA == response['raw_sse_sha256'], 'Raw evidence changed')
    x.require(request_record['request_sha256'] == x.r.sha256_value(request), 'Request changed')
    x.require(item['id'] == 'cl_172_95214' and item['status'] == 'failed' and
              item['failure_reason'].startswith('JSONDecodeError: Invalid') and
              state['turn'] == 0 and not item['trace']['response_evidence'], 'Unexpected failed result')
    x.require(response['http_status'] == 200 and response['model_ids'] == [x.r.EXPECTED_MODEL]
              and raw['model'] == request['model'] == x.r.EXPECTED_MODEL, 'Identity/HTTP failure')
    c = raw['choices'][0]
    x.require(len(raw['choices']) == 1 and c['finish_reason'] == 'stop' and
              c['stop_reason'] == c['token_ids'][-1] == 200012 and
              len(c['token_ids']) == raw['usage']['completion_tokens'] <= request['max_tokens']
              and response['usage'] == raw['usage'], 'Incomplete or unmetered response')
    # Replay the frozen decoder offline: the same fully saved tool output must
    # fail at JSON decoding, not identity, routing, framing, or truncation.
    with x.configured():
        try:
            x.runtime.transport.decode_response(raw, model=request['model'],
                max_tokens=request['max_tokens'], native_tools=request['tools'])
        except json.JSONDecodeError as exc:
            x.require('JSONDecodeError: ' + str(exc) == item['failure_reason'], 'Different decode failure')
        else:
            raise ValueError('Expected malformed tool JSON')
    return {'result_path': str(path), 'result_sha256': x.vr.sha(path),
            'raw_sha256': RAW_SHA, 'response_sha256': x.vr.sha(response_path),
            'usage': raw['usage'], 'decision': 'retain_terminal_format_failure_no_retry',
            'configuration_sha256': item['configuration_sha256'], 'new_calls_for_failed_sample': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['audit', 'run', 'score'])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    receipt = audit(output)
    receipt['recovery_script_sha256'] = x.vr.sha(Path(__file__))
    target = output / 'format_failure_recovery.json'
    if target.exists():
        x.require(x.vr.read(target) == receipt, 'Recovery receipt changed')
    else:
        x.vr.atomic(target, receipt)
    original_terminal, original_compute = x.terminal_content_failure, x.compute_summary

    def terminal(item, path):
        if path.resolve() == (output / REL).resolve():
            x.require(x.vr.sha(path) == receipt['result_sha256'], 'Failed result changed')
            return True
        return original_terminal(item, path)

    def compute(items):
        copied = copy.deepcopy(items)
        for item in copied:
            if item.get('original_result_path') == str(output / REL):
                x.require(item['original_result_sha256'] == receipt['result_sha256'], 'Accounting source changed')
                item['trace']['model_requests'] = 1
                item['trace']['response_evidence'] = [{'usage': receipt['usage']}]
        result = original_compute(copied)
        result['audited_predecode_failure_receipt'] = str(target)
        return result

    with patch.object(x, 'terminal_content_failure', terminal), patch.object(x, 'compute_summary', compute):
        if args.mode == 'run': x.run(output)
        elif args.mode == 'score': x.score(output)
        else: print(json.dumps(receipt))


if __name__ == '__main__':
    main()
