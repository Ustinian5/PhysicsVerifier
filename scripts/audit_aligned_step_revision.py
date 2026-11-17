"""Accept the approved revision using the existing request/decoder auditor.

Runs on the server after inference and explicit diagnosis review finish.
No model requests, changed labels, or replacement predictions are permitted.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import os
from pathlib import Path
import time
from unittest.mock import patch

import httpx

from scripts import audit_aligned_step_evaluation as original
from scripts import review_aligned_step_revision as review

run, vr, require = review.run, review.vr, review.require
base = run.base


def audit(root):
    require(Path('/proc').exists(), 'Official acceptance must run on the server')
    plan = run.verify(root)
    config = plan['configuration']
    parent = Path(config['parent'])
    with ExitStack() as stack:
        for path in (root/'active.lock', Path(config['service_lock']),
                     root/'offline_review/review.lock'):
            stack.enter_context(base.r.run_lock(path))
        stack.enter_context(patch.object(httpx.Client, 'send',
            side_effect=AssertionError('Offline acceptance cannot call a model')))
        require(vr.read(root/'execution_status.json')['state'] == 'collection_complete'
                and (root/'runner.exit_code').read_text().strip() == '0'
                and not (root/'runner.pid').exists(), 'Collection has not completed')
        release = vr.read(root/'service/released_after_revision.json')
        require(release['released'] and release['ownership_verified']
                and not release['remaining_processes'], 'Owned service not released')
        live = Path('/proc')/str(release['pid'])/'stat'
        require(not live.exists() or base.runtime.process_identity(release['pid']) != release['process_start'],
                'Owned service is still running')
        review.verify(root, complete=True)
        review.score(root)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base.r.EXPECTED_MODEL,
            local_files_only=True, trust_remote_code=False)
        verifier = original.Audit(root, tokenizer)
        evidence = verifier.evidence
        for path, digest in config['dependencies'].items():
            evidence.read_raw(path, digest)
        pilot = evidence.read(root/'pilot_acceptance.json')
        require(pilot['passed'] and pilot['configuration_sha256'] == plan['configuration_sha256'],
                'Pilot acceptance differs')
        for path, digest in pilot['files'].items():
            evidence.read_raw(path, digest)
        rows = {r['id']: r for r in evidence.read(parent/'inputs.json')}
        tasks = evidence.read(root/'tasks.json')
        require(len(tasks) == 400 and len({(t['id'], t['arm']) for t in tasks}) == 400,
                'Missing or duplicate revision tasks')
        memories = run.memories(root)
        failures, origins = [], {arm: Counter() for arm in base.p.ARMS}
        expected_results = set()
        with run.revised(root, plan, tokenizer):
            for index, task in enumerate(tasks):
                row, arm = rows[task['id']], task['arm']
                require(vr.digest(row) == task['input_sha256'], 'Wrong answer binding')
                result, path = run.result_for(root, task)
                evidence.read(path)
                if task['action'] == 'run_check':
                    prefix, _ = run.prefix_at(parent, row)
                    prefix = copy.deepcopy(prefix)
                    prefix['configuration_sha256'] = plan['configuration_sha256']
                    verifier.sample(row, arm, memories[arm], plan, prefix)
                    expected_results.add(path)
                else:
                    receipt = root/'reused'/arm/(vr.digest(row['id'])[:24]+'.json')
                    require(evidence.read(receipt) == task, 'Reuse receipt changed')
                    if task['action'] == 'reuse':
                        require(task['before_contract_sha256'] == task['after_contract_sha256'],
                                'Reuse with a changed model-visible request')
                    else:
                        require(task['action'] == 'retain_prepare_failure'
                                and result.get('failure_stage') == 'prepare'
                                and result['status'] == 'failed', 'Failure was replaced')
                packet = evidence.read(path.with_name('reference_packet.json'))['packet']
                observed = run.origin_report(row, result, packet.get('references', []))
                saved = root/'origins'/arm/(vr.digest(row['id'])[:24]+'.json')
                require(evidence.read(saved) == observed, 'Quote-origin evidence changed')
                origins[arm].update(r['origin'] for r in observed['records'])
                if result['status'] != 'completed':
                    failures.append(dict(id=row['id'], arm=arm, action=task['action'],
                        status=result['status'], reason=result.get('failure_reason')))
                if (index+1) % 20 == 0:
                    vr.atomic(root/'acceptance_progress.json',
                        dict(state='auditing', checked=index+1, total=400, unix=time.time()))
        require(expected_results == set((root/'formal').glob('**/result.json')),
                'Extra or missing new results')
        require(set(verifier.requests) == {str(p) for p in (root/'formal').glob('**/request.json')},
                'Extra or missing new requests')
        metrics = evidence.read(root/'offline_review/metrics.json')
        for arm, group in metrics['groups'].items():
            s = group['summary']
            require(s['failed'] == sum(f['arm'] == arm for f in failures), 'Failure count differs')
        report = dict(passed=True, audited_unix=time.time(),
            configuration_sha256=plan['configuration_sha256'], audit_code_sha256=vr.sha(__file__),
            actions=dict(Counter(t['action'] for t in tasks)), formal_tasks=400,
            completed_tasks=400-len(failures), failures=failures,
            new_formal_requests=len(verifier.requests), origins={k: dict(v) for k,v in origins.items()},
            actual_requests_and_responses_verified=True, reused_results_and_labels_unchanged=True,
            scoring_model_calls=0, human_gold=False, independent_test=False,
            physical_quality_certified=False, owned_service_released=release, files=evidence.files)
        vr.atomic(root/'acceptance.json', report)
        vr.atomic(root/'acceptance_progress.json', dict(state='completed', passed=True, unix=time.time()))
        print({k: v for k,v in report.items() if k not in {'files', 'failures'}}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--await-release', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    with base.r.run_lock(root/'acceptance.lock'):
        if args.await_release:
            vr.atomic(root/'acceptance_ready.json', dict(pid=os.getpid(), unix=time.time()))
            deadline = time.monotonic()+120
            while not (root/'ALLOW_ACCEPT').exists():
                require(time.monotonic() < deadline, 'Acceptance reconnect timed out')
                time.sleep(1)
        audit(root)


if __name__ == '__main__':
    main()
