"""Continue the frozen paired batch past one audited malformed final.

The failed result stays failed and remains in the full-cohort score. No response
repair, retry, model substitution, or change to prompts/budgets is performed.
Other failures still stop the runner for inspection.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

from scripts import run_rule_skill_comparison as run

FAILED = 'formal/rules/samples/C4/ab080d237e7a999f/result.json'
FAILED_SHA = 'c4fdce5e51188dabb6a0e420514107057a24701307a02f5145fc4c007f499aa7'


def resume(output):
    folder = output / 'evaluation'
    failed_path = folder / FAILED
    run.require(run.vr.sha(failed_path) == FAILED_SHA, 'Reviewed failure changed')
    failed = run.vr.read(failed_path)
    run.runtime.audit_result(failed)
    run.require(failed['status'] == 'failed' and failed['failure_reason'].startswith('JSONDecodeError:'),
                'Unexpected failure classification')
    response = run.vr.read(Path(failed['trace']['response_evidence'][-1]['response_path']))
    run.require(response['finish_reasons'] == ['stop'] and not response.get('tool_calls')
                and not response.get('error'), 'Not a complete malformed final')
    receipt = folder / 'content_failure_continuation.json'
    with run.r.run_lock(output / 'active.lock'), run.r.run_lock(folder / 'active.lock'):
        if not receipt.exists():
            run.vr.atomic(receipt, {
                'configuration_sha256': failed['configuration_sha256'],
                'failed_path': str(failed_path), 'failed_sha256': FAILED_SHA,
                'original_stop': run.vr.read(folder / 'execution_status.json'),
                'recovery_source_sha256': run.vr.sha(Path(__file__)),
                'policy': 'retain this audited failure in full-cohort scoring; continue remaining tasks',
                'failed_item_retries': 0, 'failure_remains_failure': True})
        else:
            saved = run.vr.read(receipt)
            run.require(saved['failed_sha256'] == FAILED_SHA and
                        saved['recovery_source_sha256'] == run.vr.sha(Path(__file__)), 'Recovery binding changed')
    original_sample, original_require = run.r.run_sample, run.require
    permit = False

    def sample(*args, **kwargs):
        nonlocal permit
        result = original_sample(*args, **kwargs)
        permit = (result == failed and Path(kwargs['root']) == folder / 'formal/rules'
                  and run.vr.sha(failed_path) == FAILED_SHA)
        return result

    def require(ok, message):
        nonlocal permit
        if message == 'Technical failure retained; inspect exact request before recovery':
            allowed, permit = permit, False
            if not ok and allowed:
                print('Preserved audited malformed final as failed; continuing remaining samples.', flush=True)
                return
        original_require(ok, message)

    with patch.object(run.r, 'run_sample', sample), patch.object(run, 'require', require):
        run.run_eval(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    resume(parser.parse_args().output.resolve())
