"""Resume frozen Sol annotations after an explicitly identified 60-second limit.

Keep the frozen runner, requests, model and budgets unchanged. Unknown 429,
authentication and quota failures retain their original stop behavior.
"""
import argparse
from pathlib import Path
import time

from scripts import run_corrected_annotations as runner
from scripts import bench_reannotation_runtime as runtime
from scripts import run_retrieval_comparison as transport


def known_window_limit(record):
    error = (record.get('response') or {}).get('error') or {}
    return (record.get('http_status') == 429 and
            error.get('code') == 'account_rate_limit_exceeded' and
            error.get('type') == 'rate_limit_error' and
            error.get('message') == 'The selected upstream account has reached its rolling 60-second request limit.')


def run(out, env):
    runner.verify(out)
    # No original source file or frozen protocol binding is edited.
    old_kind = runtime.failure_kind
    old_overload = transport.upstream_overloaded
    runtime.failure_kind = lambda rec, folder: (
        'upstream_account_window_limit' if known_window_limit(rec) else old_kind(rec, folder))
    runner.teacher.failure_kind = runtime.failure_kind
    transport.upstream_overloaded = lambda rec: known_window_limit(rec) or old_overload(rec)
    runtime.upstream_overloaded = transport.upstream_overloaded
    original_call = transport.Transport._post_once
    history = [runner.read(p).get('started_unix', 0)
               for p in out.glob('arm*/*/request_*.json')]
    last_start = max(history, default=0)

    def spaced_call(client, *args, **kwargs):
        nonlocal last_start
        while time.time() < last_start + 61:
            runner.require(not (out / 'STOP').exists(), 'STOP requested')
            time.sleep(min(1, last_start + 61 - time.time()))
        last_start = time.time()
        return original_call(client, *args, **kwargs)

    transport.Transport._post_once = spaced_call
    receipt = dict(script_sha256=runner.sha(Path(__file__)),
                   binding_sha256=runner.sha(out / 'binding.json'),
                   minimum_request_interval_seconds=61,
                   accepted_failure='exact account_rate_limit_exceeded rolling 60-second message only',
                   retry_policy='original 600s per-task cooldown, global backoff, 6 attempts and all budgets unchanged',
                   model_protocol_unchanged=True, started_unix=time.time())
    with runner.generation.corpus.r.run_lock(out / 'window_limit_recovery.lock'):
        runner.write(out / 'window_limit_recovery.json', receipt)
        try:
            runner.run(out, env)
        finally:
            runtime.failure_kind = old_kind
            runner.teacher.failure_kind = old_kind
            transport.upstream_overloaded = old_overload
            runtime.upstream_overloaded = old_overload
            transport.Transport._post_once = original_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env', type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.env.resolve())


if __name__ == '__main__':
    main()
