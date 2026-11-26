"""Apply the accepted 5% first-pass criterion to published diagnoses, with no model calls."""
from scripts import run_step_line_judge as lines

POLICY = 'one_pass_technical_failures_at_most_5_percent_of_published'
PURE_FIRST_PASS = '33ef7ac3eafb719088a21ab5e4b6172f1a5fdbcf3198723a88cff3e767e515d0'
FAILURES = ('invalid_range', 'missing_match_evidence', 'content_failure')
vr, require = lines.vr, lines.require


def assess(report):
    decisions, scores = {}, {}
    for arm, counts in report['first_pass'].items():
        score = report['provisional_scores'][arm]
        total = counts['total']
        failed = sum(counts[k] for k in FAILURES)
        require(total == score['published'] == failed + counts['located'] + counts['no_passage']
                and failed == counts['technical_failures'], 'Inconsistent published-diagnosis denominator')
        passed = total > 0 and 100 * failed <= 5 * total
        decisions[arm] = dict(counts, passed=passed, reason=(
            'passed' if passed else 'no_published_diagnoses' if total == 0 else 'technical_failure_rate_over_limit'))
        if passed:
            scores[arm] = score
    return dict(protocol=report['protocol'], release_policy=POLICY, acceptance_threshold=0.05,
                first_pass=decisions, coverage=report['coverage'], recorded_tasks=report['recorded_tasks'],
                formal_metrics_released=len(scores) == len(decisions), scores=scores,
                groups_for_review=sorted(set(decisions)-set(scores)), human_gold=False)


def frozen_pure(shared, receipt=None):
    path = shared/'pure/first_pass.json'
    require(vr.sha(path) == PURE_FIRST_PASS, 'Frozen pure first-pass measurement changed')
    # With no client, an absent raw response is a failure and cannot be regenerated.
    report = lines.measure_pure(shared, endpoint='http://127.0.0.1:18253/v1')
    result = assess(report)
    require(vr.sha(path) == PURE_FIRST_PASS and result['first_pass']['pure']['passed'], 'Different pure release')
    if receipt is not None:
        lines.judge.write(receipt, dict(result, first_pass_sha256=PURE_FIRST_PASS))
    return report


def publish(out, report):
    path = out/'step_judge_first_pass/first_pass.json'
    require(report == vr.read(path), 'First-pass report does not match frozen evidence')
    pure = vr.read(out.parent/'step_judge_lines/pure/first_pass.json')
    require(vr.sha(out.parent/'step_judge_lines/pure/first_pass.json') == PURE_FIRST_PASS and
            set(report['first_pass']) == set(lines.judge.ARMS) and
            report['first_pass']['pure'] == pure['first_pass']['pure'] and
            report['provisional_scores']['pure'] == pure['provisional_scores']['pure'],
            'Different arms or frozen pure scores')
    result = dict(assess(report), first_pass_sha256=vr.sha(path), details_sha256=report['details_sha256'],
                  publication_sha256=report['publication_sha256'])
    lines.judge.write(out/'judge_acceptance.json', result)
    if result['formal_metrics_released']:
        lines.judge.write(out/'metrics.json', result)
    return result
