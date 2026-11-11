"""Zero-generation preflight of frozen cohorts and saved checking results.

Safe alongside the coordinator: never calls run_case, rewrites a result, takes
the experiment lock, or advances its stages. verify() may refresh its existing
source hash cache. Only this audit's report/progress files are written.
"""
import argparse
from collections import Counter
from pathlib import Path
import time

from rule_framework import corrected_verifier as p, verified_resume as vr
from scripts import run_corrected_pipeline as pipeline
from scripts import run_corrected_checks as checking

require = pipeline.require


def audit_cohort(out, task, cfg, tokenizer):
    rows = vr.read(out / (task + '_inputs.json'))
    ids = [r['id'] for r in rows]
    expected = {'step': 100, 'final': 600}[task]
    require(len(ids) == len(set(ids)) == expected, 'Cohort size/duplicate ID: ' + task)
    measurements = []
    for row in rows:
        require(set(row) == {'id', 'question', 'context', 'prediction'}, 'Public input fields drift')
        require(all(isinstance(row[k], str) for k in row), 'Non-string public input')
        require(row['question'].strip() and row['prediction'].strip(), 'Empty question/answer')
        for arm in p.ARMS:
            messages = p.messages(row, arm, task, revision=cfg.get('check_prompt_revision'))
            native = p.tools() if arm != 'pure' else None
            # Exercise the same request builder and control-token guard as live
            # execution, without handing the body to any transport/client.
            body = checking.c.corpus.r.request_body(cfg['transport']['model'], messages,
                p.LIMITS['output_tokens'], backend='vllm', native_tools=native)
            require(body['messages'] == messages, 'Request builder changed visible messages')
            count = checking.token_count(tokenizer, messages, native)
            measurements.append(dict(id=row['id'], arm=arm, initial_input_tokens=count))
    oversized = [x for x in measurements if
        x['initial_input_tokens'] + p.LIMITS['output_tokens'] > p.LIMITS['context_tokens'] or
        x['initial_input_tokens'] > p.LIMITS['input_tokens']]
    return rows, dict(questions=len(rows), arm_conditions=len(measurements),
        largest_initial_input=max(measurements, key=lambda x: x['initial_input_tokens']),
        oversized=oversized, tokenizer_estimates_only=True)


def audit_scoring(out, task, rows):
    """Rehearse all references using synthetic failures, never publish metrics."""
    if task == 'step':
        refs = vr.read(out / 'step_reference.json')
        unresolved = vr.read(out / 'step_unresolved.json')
        require([r['id'] for r in refs] == [r['id'] for r in rows], 'Reference cohort/order drift')
        require(set(unresolved) == {r['id'] for r in rows}, 'Unresolved-reference cohort drift')
        predictions = [dict(id=r['id'], diagnostics=[], execution_failure=True,
                            source_status='content_failure') for r in rows]
        summary = pipeline.scoring.score(refs, predictions, unresolved,
                                        preserve_zero_offsets=True)['summary']
        require(summary['dataset_size'] == len(rows) and summary['failed_questions'] == len(rows)
                and summary['successful_empty_questions'] == 0, 'Step failure denominator drift')
        return dict(synthetic_rehearsal=True, questions=len(rows),
                    confirmed_reference_errors=summary['total_gt_errors'],
                    locatable_reference_errors=summary['total_gt_locatable_errors'],
                    all_failed_cases_retained=True)
    labels = vr.read(out / 'final_labels_private.json')
    require(set(labels) == {r['id'] for r in rows}, 'Final label cohort drift')
    balance = Counter(x['verdict'] for x in labels.values())
    require(balance == {'correct': 300, 'incorrect': 300}, 'Final label balance drift')
    predictions = {r['id']: dict(status='content_failure', value=None) for r in rows}
    result = p.final_scores(rows, labels, predictions)
    require(result['total'] == result['execution_failure'] == len(rows) and
            result['binary_coverage'] == 0 and not result['complete_binary_evaluation'],
            'Final failure coverage drift')
    return dict(synthetic_rehearsal=True, questions=len(rows), label_balance=dict(balance),
                failures_not_converted_to_binary_verdicts=True)


def audit_results(out, cfg, rows_by_task):
    # Snapshot the list first: later results from the healthy runner are not
    # falsely flagged as an inconsistent snapshot or silently counted.
    paths = sorted(path for task in rows_by_task
        for path in (out / 'checks' / task / 'formal').glob('*/*/result.json'))
    by_task = {task: {r['id']: r for r in rows} for task, rows in rows_by_task.items()}
    condition = pipeline.check_condition(out, cfg)
    counts = {t: {a: Counter() for a in p.ARMS} for t in rows_by_task}
    evidence_hashes, result_hashes, failures = {}, {}, []
    for path in paths:
        task, _, arm, sid, _ = path.relative_to(out / 'checks').parts
        require(task in by_task and arm in p.ARMS and sid in by_task[task], 'Unregistered result')
        row = by_task[task][sid]
        result = vr.read(path)
        require((result['id'], result['arm'], result['task']) == (sid, arm, task), 'Result identity drift')
        require(result['signature'] == vr.digest([row, arm, task, condition]), 'Saved condition drift')
        require(result['status'] in {'complete', 'content_failure', 'technical_failure', 'budget_exhausted'},
                'Unrecognized result status')
        require(result['usage']['requests'] <= p.LIMITS['responses'] and
                result['usage']['output_tokens_or_unknown_reservation'] <= p.LIMITS['output_tokens'] and
                result['usage']['input_tokens'] <= p.LIMITS['input_tokens'], 'Saved budget exceeded')
        if result['status'] == 'complete':
            if task == 'final':
                p.final.validate_verdict(result['value'])
            else:
                require(isinstance(result['value'], dict) and set(result['value']) == {'diagnostics'},
                        'Successful step result has no diagnostics object')
                checking.c.corpus.r.validate_diagnostics(result['value'])
        else:
            require(result['value'] is None and result['reason'], 'Failure masquerades as success')
            failures.append(dict(task=task, arm=arm, id=sid, status=result['status'], reason=result['reason']))
        for ev in result['evidence']:
            source = path.parent / ev['file']
            require(source.resolve().is_relative_to(path.parent.resolve()), 'Evidence path escaped case')
            require(vr.sha(source) == ev['sha256'], 'Saved raw evidence drift')
            evidence_hashes[str(source.relative_to(out))] = ev['sha256']
        result_hashes[str(path.relative_to(out))] = vr.sha(path)
        counts[task][arm][result['status']] += 1
    return dict(snapshot_results=len(paths), outcomes=counts, failures=failures,
                result_hashes=result_hashes, evidence_files=len(evidence_hashes),
                evidence_manifest_sha256=vr.digest(evidence_hashes))


def run(out, tokenizer):
    started = time.time()
    plan = pipeline.verify(out)
    cfg = plan['configuration']
    require((out / 'construction_acceptance.json').is_file(), 'Material acceptance not yet available')
    pipeline.accept_materials(out, cfg)  # Existing acceptance: hashes only.
    require(cfg['limits'] == p.LIMITS and cfg['arms'] == list(p.ARMS), 'Limits/arms drift')
    rows, cohorts, rehearsal = {}, {}, {}
    for task in ('step', 'final'):
        rows[task], cohorts[task] = audit_cohort(out, task, cfg, tokenizer)
        rehearsal[task] = audit_scoring(out, task, rows[task])
    require(not ({r['id'] for r in rows['step']} & {r['id'] for r in rows['final']}), 'Bench ID overlap')
    results = audit_results(out, cfg, rows)
    return dict(passed=not any(x['oversized'] for x in cohorts.values()),
        started_unix=started, finished_unix=time.time(), generation_calls=0,
        configuration_sha256=plan['configuration_sha256'], script_sha256=vr.sha(Path(__file__)),
        cohorts=cohorts, offline_scoring_rehearsal=rehearsal, saved_result_snapshot=results,
        scope='Initial inputs, frozen code/data/repair chain, material hashes, saved results and offline scoring. '
              'No claim about future model output syntax, future context growth or physical accuracy.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    with checking.c.corpus.r.run_lock(out / 'runtime_preflight.lock'):
        try:
            from transformers import AutoTokenizer
            cfg = vr.read(out / 'plan.json')['configuration']
            tokenizer = AutoTokenizer.from_pretrained(cfg['transport']['model'], local_files_only=True)
            report = run(out, tokenizer)
        except Exception as exc:
            vr.atomic(out / 'runtime_preflight.json', dict(passed=False, finished_unix=time.time(),
                generation_calls=0, error_type=type(exc).__name__, error=str(exc)))
            raise
        vr.atomic(out / 'runtime_preflight.json', report)
        print({k: report[k] for k in ('passed', 'generation_calls', 'cohorts')}, flush=True)
        require(report['passed'], 'Frozen initial inputs do not fit; do not replace or truncate them')


if __name__ == '__main__':
    main()
