"""Continue the frozen v15 run after a schema-invalid shared PREPARE.

No prompts, queries, retrieval, budgets, decoder, or metric are changed. A clean
but invalid preparation blocks CHECK in all four arms, is counted as failure,
and is never truncated, repaired, retried, or substituted with an empty query.
"""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch

from scripts import run_aligned_step_evaluation as run

base, vr, require = run.base, run.vr, run.require
RECEIPT = 'preparation_failure_recovery.json'
FAILURE = 'preparation_failure.json'
POLICY = 'retain_invalid_prepare_as_shared_content_failure_no_retry_v1'


def schema_failure(content):
    try:
        value = base.r.parse_json(content)
    except json.JSONDecodeError as exc:
        return 'JSONDecodeError: ' + str(exc)
    try:
        base.p.validate_queries(value)
    except ValueError as exc:
        require(str(exc) in {'Preparation is queries only', 'Invalid query phrases'},
                'Unknown preparation validation error')
        return str(exc)
    raise ValueError('Valid preparation cannot be discarded')


def validate_response(row, request_record, raw, response):
    request = request_record['request']
    require(request_record['request_sha256'] == base.r.sha256_value(request) and
            request['messages'] == base.p.query_messages(row) and
            request['model'] == base.r.EXPECTED_MODEL and request['max_tokens'] == base.p.QUERY_MAX_TOKENS and
            request['temperature'] == 0 and request['top_p'] == 1 and request['reasoning_effort'] == 'medium' and
            not request.get('tools'), 'Preparation request/input/protocol mismatch')
    require(response.get('http_status') == 200 and not response.get('error') and
            response.get('model_ids') == [base.r.EXPECTED_MODEL] and response.get('finish_reasons') == ['stop'] and
            raw.get('model') == base.r.EXPECTED_MODEL and len(raw.get('choices', [])) == 1 and
            raw['choices'][0]['finish_reason'] == 'stop' and raw.get('usage') == response.get('usage'),
            'Transport, identity, or unfinished response is not a schema failure')
    usage = response['usage']
    require(type(usage['prompt_tokens']) is int and usage['prompt_tokens'] > 0 and
            type(usage['completion_tokens']) is int and 0 <= usage['completion_tokens'] <= base.p.QUERY_MAX_TOKENS and
            usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'Invalid response usage')
    return schema_failure(response['content'])


def inspect_preparation(out, row, plan):
    folder = out / 'queries' / vr.digest(row['id'])[:24]
    require(not (folder / 'result.json').exists(), 'Successful PREPARE must be reused unchanged')
    request, raw_path, response_path = [folder / n for n in ('request.json', 'raw.json', 'response.json')]
    require(all(p.is_file() for p in (request, raw_path, response_path)),
            'Incomplete request evidence; do not resample or classify as content failure')
    req, raw, response = [vr.read(p) for p in (request, raw_path, response_path)]
    require(response.get('raw_sse_sha256') == vr.sha(raw_path), 'Raw response hash mismatch')
    reason = validate_response(row, req, raw, response)
    # Re-decode with exactly the frozen decoder stack. This is offline and does
    # not repair the invalid query JSON or accept a new native message shape.
    with run.a.base.old.comparison.execution.configured(), \
         patch.object(run.a.recovery, 'ORIGINAL_DECODE', run.a.historical_transport().decode_response):
        decoded = base.decode_response(raw, model=base.r.EXPECTED_MODEL,
                                       max_tokens=base.p.QUERY_MAX_TOKENS, native_tools=None)
    for key in ('content', 'reasoning_content', 'model_ids', 'finish_reasons', 'usage'):
        require(decoded.get(key) == response.get(key), 'Decoded response differs from original: ' + key)
    tokens = raw['prompt_token_ids']; effective = out / 'effective_inputs' / (vr.digest(tokens) + '.json')
    actual = vr.read(effective)
    require(actual['sha256'] == vr.digest(tokens) and actual['token_count'] == len(tokens) and
            '\nCurrent date: ' + run.a.DATE + '\n' in actual['system_header'] and
            '\nReasoning: medium\n' in actual['system_header'], 'Actual input date/reasoning drift')
    evidence = dict(turn=1, request_path=str(request), request_sha256=vr.sha(request),
        raw_sse_path=str(raw_path), raw_sse_sha256=vr.sha(raw_path), response_path=str(response_path),
        response_sha256=vr.sha(response_path), model_ids=response['model_ids'],
        finish_reasons=response['finish_reasons'], usage=response['usage'], shared_preparation=True)
    return dict(policy=POLICY, id=row['id'], configuration_sha256=plan['configuration_sha256'],
        input_sha256=vr.digest(row), failure_stage='prepare', reason=reason, evidence=evidence,
        files={str(p): vr.sha(p) for p in (request, raw_path, response_path, effective)},
        new_model_calls=0, response_edited=False, retry=False, affected_arms=list(base.p.ARMS))


def failure_result(out, row, arm, plan, receipt):
    evidence = receipt['evidence']; usage = evidence['usage']
    path = out / 'queries' / vr.digest(row['id'])[:24] / FAILURE
    return dict(id=row['id'], model=base.r.EXPECTED_MODEL, verifier='C0p', organization_arm=arm,
        status='failed', checker_status='not_run', failure_stage='prepare',
        failure_reason='PREPAREContentFailure: ' + receipt['reason'], diagnostics=[],
        configuration_sha256=plan['configuration_sha256'],
        preparation_failure=dict(path=str(path), sha256=vr.sha(path)),
        protocol_audit=dict(passed=False, violations=['invalid_shared_preparation'], adherence_only=True),
        stage_failure_evidence=[dict(stage='prepare', reason=receipt['reason'], shared_request_path=evidence['request_path'])],
        trace=dict(model_requests=1, input_tokens=usage['prompt_tokens'],
            output_token_limit_reserved=usage['completion_tokens'], visible_final_channel_tokens=0,
            read_rule_ids=[], tool_events=[], tree_access=False, response_evidence=[copy.deepcopy(evidence)]))


def record_failure(out, row, plan, receipt):
    folder = out / 'queries' / vr.digest(row['id'])[:24]; path = folder / FAILURE
    if path.exists(): require(vr.read(path) == receipt, 'Preparation failure receipt changed')
    else: vr.atomic(path, receipt)
    results = {}
    for arm in base.p.ARMS:
        target = base.path_for(out, 'formal', arm, row)
        result = failure_result(out, row, arm, plan, receipt)
        packet = dict(configuration_sha256=plan['configuration_sha256'], input_sha256=vr.digest(row),
            preparation_failure_sha256=vr.sha(path),
            packet=dict(references=[], retrieval=dict(status='not_executed_prepare_failure',
                        provided_complete_ids=[], reference_tokens=0)))
        for dest, value in [(target.with_name('reference_packet.json'), packet), (target, result)]:
            if dest.exists(): require(vr.read(dest) == value, 'Existing result must not be overwritten')
            else: vr.atomic(dest, value)
        base.runtime.audit_result(result); results[arm] = result
    return results


def verify(out):
    plan = run.verify(out); receipt = vr.read(out / RECEIPT)
    require(receipt['configuration_sha256'] == plan['configuration_sha256'] and
            receipt['recovery_code_sha256'] == vr.sha(Path(__file__)) and
            receipt['plan_sha256'] == vr.sha(out / 'plan.json'), 'Recovery binding drift')
    vr.EvidenceCache(out / 'recovery_evidence_cache.json').check(receipt['prefix_files'])
    return plan


def register(out):
    plan = run.verify(out)
    with base.r.run_lock(out / 'active.lock'), base.r.run_lock(Path(plan['configuration']['service_lock'])):
        require(not (out / 'runner.pid').exists(), 'Inspect existing runner before recovery')
        if (out / RECEIPT).exists(): return verify(out)
        affected = []
        for row in vr.read(out / 'inputs.json'):
            folder = out / 'queries' / vr.digest(row['id'])[:24]
            if (folder / 'request.json').exists() and not (folder / 'result.json').exists():
                affected.append((row, inspect_preparation(out, row, plan)))
        require(affected, 'No observed invalid shared PREPARE')
        files = {str(p): vr.sha(p) for part in ('queries', 'controls', 'formal')
                 for p in (out / part).rglob('*') if p.is_file() and p.suffix in ('.json', '.sse')}
        lifecycle = {n: dict(sha256=vr.sha(out / n), value=vr.read(out / n)) for n in (
            'execution_status.json', 'launch.json', 'disconnect_ready.json', 'disconnect_acceptance.json',
            'service/launch.json', 'service/startup.json', 'service/released_after_evaluation.json') if (out / n).exists()}
        receipt = dict(policy=POLICY, configuration_sha256=plan['configuration_sha256'],
            plan_sha256=vr.sha(out / 'plan.json'), recovery_code_sha256=vr.sha(Path(__file__)),
            prefix_files=files, previous_lifecycle=lifecycle, previous_exit_code=(out / 'runner.exit_code').read_text(),
            affected=[value for _, value in affected], new_model_calls=0,
            prompts_retrieval_budgets_decoder_and_scoring_changed=False, unix=time.time())
        vr.atomic(out / RECEIPT, receipt)
        for row, value in affected: record_failure(out, row, plan, value)
        return verify(out)


@contextmanager
def recovered(out):
    plan = verify(out); rows = {r['id']: r for r in vr.read(out / 'inputs.json')}
    original_run, original_failure, original_score = base.run_condition, base.complete_content_failure, run.score

    def complete_failure(value):
        if 'preparation_failure' not in value: return original_failure(value)
        row = rows[value['id']]; path = out / 'queries' / vr.digest(row['id'])[:24] / FAILURE
        require(value['preparation_failure'] == dict(path=str(path), sha256=vr.sha(path)), 'Failed PREPARE binding drift')
        receipt = vr.read(path)
        for name, sha in receipt['files'].items(): require(vr.sha(Path(name)) == sha, 'Failed PREPARE evidence drift')
        require(value == failure_result(out, row, value['organization_arm'], plan, receipt), 'Failed result changed')
        base.runtime.audit_result(value)
        return True

    def condition(root, row, phase, arm, memory, current, tokenizer, client):
        try: return original_run(root, row, phase, arm, memory, current, tokenizer, client)
        except ValueError:
            folder = out / 'queries' / vr.digest(row['id'])[:24]
            if phase != 'formal' or row['id'] not in rows or (folder / 'result.json').exists(): raise
            require(root == out and current['configuration_sha256'] == plan['configuration_sha256'] and
                    row == rows[row['id']], 'Wrong recovery cohort')
            receipt = inspect_preparation(out, row, plan)
            results = record_failure(out, row, plan, receipt)
            base.old.emit(out, 'formal', sample_id=row['id'], arm=arm,
                          result_status='shared_prepare_content_failure_preserved', reason=receipt['reason'])
            return results[arm]

    def score(root, current=None):
        report = original_score(root, current)
        failures = [vr.read(p) for p in (out / 'queries').glob('*/' + FAILURE)]
        report['shared_preparation_failures'] = dict(questions=len(failures), affected_arm_tasks=4 * len(failures),
            actual_requests=len(failures), ids=[x['id'] for x in failures], retained_in_full_cohort=True,
            classified_as_successful_empty=False, retried=False)
        vr.atomic(out / 'metrics.json', report)
        return report

    with patch.object(base, 'run_condition', condition), patch.object(base, 'complete_content_failure', complete_failure), \
         patch.object(run, 'score', score):
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'verify', 'run', 'score', 'confirm-detached'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    if args.command == 'register': register(out)
    elif args.command == 'verify': verify(out)
    elif args.command == 'confirm-detached': verify(out); run.confirm(out)
    else:
        with recovered(out):
            if args.command == 'run': run.run(out)
            else: run.score(out)


if __name__ == '__main__': main()
