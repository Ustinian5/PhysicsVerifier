"""Accept a completed old-v15/30B run offline, without changing its results.

Rebuild retrieval packets and CHECK continuations, decode every saved response,
reproduce all four complete score objects, and bind the original evidence.
Only acceptance.json and acceptance_progress.json are written in the run.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch

import httpx

from scripts import resume_aligned_step_evaluation as recovery
from scripts.audit_aligned_organization import Evidence

run, base, vr, require = recovery.run, recovery.base, recovery.vr, recovery.require
a = run.a


def check_request(saved, messages, limit):
    expected = dict(model=base.r.EXPECTED_MODEL, messages=messages, temperature=0, top_p=1,
                    max_tokens=limit, stream=False, return_token_ids=True, reasoning_effort='medium')
    require(saved['request'] == expected and saved['request_sha256'] == vr.digest(expected),
            'Request differs from frozen messages/parameters/budget')


def recount(details):
    """Independent arithmetic over the published per-question match records."""
    tp = sum(d['matched_error_count'] for d in details)
    pred = sum(d['pred_locatable_finding_count'] for d in details)
    gt = sum(d['gt_error_count'] for d in details)
    require(tp == sum(d['matched_pred_locatable'] for d in details), 'Non-bijective matches')
    return dict(precision=tp / pred if pred else 0., recall=tp / gt if gt else 0.,
                f1=2 * tp / (pred + gt) if pred + gt else 0.)


class Audit:
    def __init__(self, root, tokenizer):
        self.root, self.tokenizer = root, tokenizer
        self.evidence, self.requests, self.decoded = Evidence(), {}, {}

    def response(self, path, messages, limit):
        e = self.evidence
        saved = e.read(path)
        check_request(saved, messages, limit)
        if str(path) in self.requests:
            return self.requests[str(path)], self.decoded[str(path)]
        response = e.read(path.with_name('response.json'))
        raw_path = path.with_name('raw.json' if path.parent.parent.name == 'queries' else 'response.sse')
        raw = e.read(raw_path, response['raw_sse_sha256'])
        require(response.get('http_status') == 200 and response.get('model_ids') == [base.r.EXPECTED_MODEL]
                and raw['model'] == base.r.EXPECTED_MODEL and len(raw['choices']) == 1,
                'Transport/model identity mismatch')
        usage = raw['usage']
        require(usage == response['usage'] and len(raw['prompt_token_ids']) == usage['prompt_tokens']
                and 0 < usage['completion_tokens'] <= limit
                and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'],
                'Usage or actual input count mismatch')
        require(usage['prompt_tokens'] + limit <= 131072 and raw['choices'][0]['finish_reason'] == 'stop',
                'Context/termination mismatch')
        effective = a.validate_effective_input(raw, self.tokenizer, a.DATE)
        require(e.read(self.root / 'effective_inputs' / (effective['sha256'] + '.json')) == effective,
                'Actual model-input receipt differs')
        try:
            decoded = base.decode_response(raw, model=base.r.EXPECTED_MODEL, max_tokens=limit, native_tools=None)
        except (ValueError, TypeError, KeyError) as exc:
            require(response.get('error') == type(exc).__name__ + ': ' + str(exc),
                    'Original decoder failure cannot be reproduced')
            decoded = None
        else:
            require(not response.get('error'), 'Stored decoder failure now succeeds')
            for key in ('content', 'reasoning_content', 'model_ids', 'finish_reasons', 'usage'):
                require(decoded.get(key) == response.get(key), 'Saved decode drift: ' + key)
        self.requests[str(path)], self.decoded[str(path)] = usage, decoded
        return usage, decoded

    def sample(self, row, arm, memory, plan, prefix, phase='formal'):
        e, out = self.evidence, self.root
        path = base.path_for(out, phase, arm, row)
        item = e.read(path); base.runtime.audit_result(item)
        require(item['id'] == row['id'] and item['configuration_sha256'] == plan['configuration_sha256'],
                'Wrong sample/configuration')
        if item.get('failure_stage') == 'prepare':
            require(base.complete_content_failure(item), 'Unverified failed PREPARE')
            require(not list(path.parent.glob('turn_*/request.json')), 'CHECK executed after failed PREPARE')
            require(item['diagnostics'] == [] and item['status'] == 'failed', 'Failed PREPARE is not success')
            packet = e.read(path.with_name('reference_packet.json'))
            require(packet['packet']['retrieval']['status'] == 'not_executed_prepare_failure', 'False empty retrieval')
            return item
        count = lambda messages: len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        packet = base.p.reference_packet(row, arm, prefix['queries'], memory, count)
        actual = e.read(path.with_name('reference_packet.json'))
        require(actual == dict(configuration_sha256=plan['configuration_sha256'], packet=packet,
            query_evidence_sha256=vr.digest(prefix), input_sha256=vr.digest(row)), 'Retrieval or payload drift')
        paths = sorted(path.parent.glob('turn_*/request.json'))
        require(paths and [p.parent.name for p in paths] == ['turn_%02d' % n for n in range(2, 2 + len(paths))],
                'Missing/duplicate CHECK turns')
        require(len(paths) <= 2, 'Unexpected extra CHECK or quote-repair attempt')
        used_in = prefix['evidence']['usage']['prompt_tokens']
        used_out = prefix['evidence']['usage']['completion_tokens']
        messages, decoded, first = packet['messages'], None, None
        for index, request_path in enumerate(paths):
            if index:
                require(first is not None and decoded is not None, 'Repair lacks initial diagnosis')
                messages = messages + [dict(role='assistant', content=decoded['content']), dict(role='user',
                    content=base.quotes.repair_message(first, base.quotes.invalid_quotes(first, row)))]
            usage, decoded = self.response(request_path, messages, 16384 - used_out)
            used_in += usage['prompt_tokens']; used_out += usage['completion_tokens']
            if index == 0 and path.with_name('first_diagnosis.json').exists():
                first = e.read(path.with_name('first_diagnosis.json'))
                require(decoded is not None and base.r.parse_json(decoded['content']) == first,
                        'First diagnosis differs from original response')
        require(used_in <= 600000 and used_out <= 16384, 'Per-question budget exceeded')
        if item['status'] == 'completed':
            require(first is not None and decoded is not None, 'Completed output lacks a complete final')
            final = first
            if len(paths) == 2:
                try: final, _ = base.quotes.apply_quote_patches(first, json.loads(decoded['content']), row)
                except (ValueError, TypeError, KeyError): pass  # original policy retains the first diagnosis
            require(final['diagnostics'] == item['diagnostics'], 'Publication changed diagnoses or quote policy')
            require(item['protocol_audit'] == base.reference_audit(final, packet['retrieval']['provided_complete_ids']),
                    'Citation protocol audit differs')
        else:
            require(item['diagnostics'] == [], 'Unexpected diagnostics on failed CHECK')
            if base.complete_content_failure(item): pass
            else:
                inspected = a.native.inspect_failure(path)
                stored = e.read(path.with_name('native_format_accounting.json'))
                require(inspected == stored, 'Failed CHECK cannot be reproduced from raw response')
        return item


def audit(out):
    require(Path('/proc').exists(), 'Run the formal acceptance audit on the server')
    plan = recovery.verify(out); config = plan['configuration']
    with ExitStack() as stack:
        for p in (out / 'active.lock', Path(config['service_lock']), out / 'acceptance.lock'):
            stack.enter_context(base.r.run_lock(p))
        def deny(*args, **kwargs): raise AssertionError('Offline audit must never issue a model request')
        stack.enter_context(patch.object(httpx.Client, 'send', deny))
        stack.enter_context(a.base.old.comparison.execution.configured())
        stack.enter_context(patch.object(a.recovery, 'ORIGINAL_DECODE', a.historical_transport().decode_response))
        stack.enter_context(patch.object(a.bench.recovery, 'inspect_failure', a.native.inspect_failure))
        stack.enter_context(recovery.recovered(out))
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base.r.EXPECTED_MODEL, local_files_only=True, trust_remote_code=False)
        x = Audit(out, tokenizer); e = x.evidence
        for name, sha in config['dependencies'].items(): e.read_raw(name, sha)
        receipt = e.read(out / recovery.RECEIPT)
        for name, sha in receipt['prefix_files'].items(): e.read_raw(name, sha)
        for name in ('plan', 'runtime', 'engineering_acceptance', 'engineering_protocol_audit',
                     'disconnect_acceptance', 'label_release', 'historical_scoring_parity', 'isolation'):
            e.read(out / (name + '.json'))
        require(e.read(out / 'execution_status.json')['state'] == 'completed' and
                e.read_raw(out / 'runner.exit_code').strip() == b'0' and not (out / 'runner.pid').exists(),
                'Run has not exited successfully')
        release = e.read(out / 'service/released_after_recovery.json')
        require(release['released'] and release['ownership_verified'] and not release['remaining_processes'],
                'Owned model service not released')
        live = Path('/proc') / str(release['pid']) / 'stat'
        require(not live.exists() or base.runtime.process_identity(release['pid']) != release['process_start'],
                'Owned service process still exists')
        rows, dataset, unknown, metrics = [e.read(out / (n + '.json')) for n in ('inputs','reference','unresolved','metrics')]
        require(len(rows) == 100 and [base.p.source_payload(r) for r in dataset] == rows,
                'Wrong cohort or reference answers')
        memories = dict(pure=None, case_rag=base.p.FlatMemory(e.read(out / 'cases.json')),
            flat_skills=base.p.FlatMemory(e.read(out / 'skills.json')), rule_tree=base.p.TreeMemory(e.read(out / 'catalog.json')))
        groups = {a: [] for a in base.p.ARMS}; query_failed = []; controls = e.read(out / 'control_prefixes.json')
        for phase, source in [('controls', config['controls']), ('formal', rows)]:
            for i, row in enumerate(source):
                prefix = controls.get(row['id']) if phase == 'controls' else None
                if phase == 'formal':
                    folder = out / 'queries' / vr.digest(row['id'])[:24]
                    _, decoded = x.response(folder / 'request.json', base.p.query_messages(row), 2048)
                    require(decoded is not None, 'Unknown PREPARE decoder failure')
                    if (folder / recovery.FAILURE).exists():
                        failed = e.read(folder / recovery.FAILURE)
                        require(failed == recovery.inspect_preparation(out, row, plan), 'PREPARE failure receipt differs')
                        query_failed.append(row['id'])
                    else:
                        prefix = e.read(folder / 'result.json')
                        require(prefix['input_sha256'] == vr.digest(row) and
                                prefix['queries'] == base.p.validate_queries(base.r.parse_json(decoded['content'])),
                                'PREPARE output was replaced')
                else:
                    prefix = copy.deepcopy(prefix); prefix['configuration_sha256'] = plan['configuration_sha256']
                for arm in base.p.ARMS:
                    result = x.sample(row, arm, memories[arm], plan, prefix, phase)
                    if phase == 'formal': groups[arm].append(result)
                    else: require(base.runtime.healthy_result(result) and result['protocol_audit']['passed'], 'Control failed')
                if (i+1) % 10 == 0:
                    vr.atomic(out / 'acceptance_progress.json', dict(phase=phase, checked=i+1, unix=time.time()))
        expected_files = {base.path_for(out,'formal',arm,row) for row in rows for arm in base.p.ARMS}
        require(expected_files == set((out/'formal').glob('**/result.json')), 'Missing/extra formal results')
        actual_requests = {str(p) for p in (out/'formal').glob('**/request.json')} | {str(p) for p in (out/'queries').glob('*/request.json')}
        checked = {p for p in x.requests if '/formal/' in p or '/queries/' in p}
        require(actual_requests == checked and len(checked) == metrics['formal_actual_requests_shared_prefix_counted_once'],
                'Unaccounted or repeated requests')
        require(sorted(query_failed) == sorted(metrics['shared_preparation_failures']['ids']), 'Failure counts differ')
        summaries = {}
        for arm, items in groups.items():
            score = run.score_items(dataset, items, unknown)
            require(score == metrics['scores'][arm], 'Full independent score recomputation differs: '+arm)
            for k,v in recount(score['details']).items():
                require(abs(v-score['summary'][k]) < 1e-12, 'Score arithmetic differs')
            summaries[arm] = score['summary']
        historical = run.score_parity(Path(config['legacy']))
        require(historical == e.read(out/'historical_scoring_parity.json'), 'Historical scoring parity changed')
        failures=[dict(id=d['id'],arm=arm,reason=d.get('failure_reason')) for arm,items in groups.items() for d in items if d['status']!='completed']
        report = dict(passed=True, audited_unix=time.time(), configuration_sha256=plan['configuration_sha256'],
            audit_source_sha256=vr.sha(Path(__file__)), new_model_calls=0, formal_tasks=400,
            completed_tasks=400-len(failures), content_failures=failures, unknown_technical_failures=0,
            actual_formal_requests=len(checked), shared_prepare_failures=query_failed, controls=20,
            scores=summaries, full_score_objects_reproduced=True, old_396_scores_reproduced=True,
            packets_and_requests_reconstructed=True, actual_model_date_reasoning_and_tokens_verified=True,
            original_outputs_unchanged=True, owned_service_released=release, files=e.files,
            human_gold=False, physical_quality_certified=False, independent_test=False,
            limitation='Frozen offline location proxy; two unresolved labels remain. No semantic judgment or retrial.')
        vr.atomic(out / 'acceptance.json',report)
        vr.atomic(out / 'acceptance_progress.json',dict(state='completed',passed=True,unix=time.time()))
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit(args.output.resolve())


if __name__ == '__main__': main()
