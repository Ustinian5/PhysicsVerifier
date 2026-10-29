"""Offline scoring with evidence-based request count for the audited JSON failure."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

from scripts import run_rule_skill_comparison as run
from scripts.resume_rule_skill_evaluation import FAILED, FAILED_SHA


def score(output):
    folder = output / 'evaluation'
    with run.r.run_lock(output / 'active.lock'), run.r.run_lock(folder / 'active.lock'):
        path = folder / FAILED
        run.require(run.vr.sha(path) == FAILED_SHA, 'Audited failure changed')
        failed = run.vr.read(path)
        run.runtime.audit_result(failed)
        evidence = failed['trace']['response_evidence']
        run.require(len(evidence) == failed['trace']['model_requests'] + 1 and
                    [x['turn'] for x in evidence] == list(range(1, len(evidence) + 1)),
                    'Not the expected uncommitted malformed-final turn')
        original = run.comparison.trace_summary

        def summary(items):
            adjusted = copy.deepcopy(items)
            for item in adjusted:
                if item.get('original_result_sha256') == FAILED_SHA:
                    run.require(item['trace'] == failed['trace'] and item['status'] == 'failed', 'Failure drift')
                    item['trace']['model_requests'] = len(evidence)
            return original(adjusted)

        with patch.object(run.comparison, 'trace_summary', summary):
            report = run.score(output)
        run.vr.atomic(folder / 'scoring_recovery.json', {
            'configuration_sha256': report['configuration_sha256'],
            'original_stop': run.vr.read(folder / 'execution_status.json'),
            'failed_result_sha256': FAILED_SHA, 'source_sha256': run.vr.sha(Path(__file__)),
            'original_counter': failed['trace']['model_requests'], 'evidence_requests': len(evidence),
            'adjustment': 'compute-summary copy only; raw result, failed status and physics scoring unchanged',
            'new_model_calls': 0, 'metrics_sha256': run.vr.sha(folder / 'metrics.json')})
        run.emit(folder, 'completed_evaluation', recorded_tasks=100, successful_tasks=99,
                 failed_tasks=1, case_review='pending', scoring_recovery='scoring_recovery.json')
        print(json.dumps({a: v['summary'] for a, v in report['scores_all_50'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    score(parser.parse_args().output.resolve())
