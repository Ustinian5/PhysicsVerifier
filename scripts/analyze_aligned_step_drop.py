"""Describe the frozen old-v15/new-30B score gap without generation or rescoring.

The case dispositions below are an assistant's targeted comparison of saved
claims, not replacement gold, a semantic metric, or a causal experiment.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

from scripts import run_aligned_step_evaluation as run
from scripts import evaluate_physics_eval_sets as location


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def counts(details):
    fields = ('gt_error_count', 'published_diagnostic_count',
              'pred_locatable_finding_count', 'matched_error_count')
    return dict(questions=len(details), questions_with_gt=sum(d['gt_error_count'] > 0 for d in details),
                successful_empty=sum(d['successful_empty'] for d in details),
                failures=sum(d['execution_failure'] for d in details),
                **{f: sum(d[f] for d in details) for f in fields})


def aggregate(root, score_key):
    inputs = read(root / 'inputs.json')
    metrics = read(root / 'metrics.json')
    score = metrics[score_key]['pure']
    stages = {1: [], 2: []}
    finishes = []
    for row in inputs:
        result = read(run.base.path_for(root, 'formal', 'pure', row))
        for event in result['trace']['response_evidence']:
            turn = event.get('turn')
            if turn in stages and event.get('usage'):
                stages[turn].append(event['usage'])
                if turn == 2:
                    finishes.extend(event.get('finish_reasons', []))
    return dict(
        inputs=binding(root / 'inputs.json'), metrics=binding(root / 'metrics.json'),
        pure_summary=score['summary'],
        pure_breakdown={name: counts([d for d in score['details'] if predicate(d)])
                        for name, predicate in (
                            ('empty', lambda d: d['successful_empty']),
                            ('published', lambda d: d['published_diagnostic_count'] > 0),
                            ('failure', lambda d: d['execution_failure']))},
        answer_characters=dict(mean=statistics.mean(len(r['prediction']) for r in inputs),
                               median=statistics.median(len(r['prediction']) for r in inputs)),
        check_usage=dict(recorded=len(stages[2]),
                         mean_input=statistics.mean(x['prompt_tokens'] for x in stages[2]),
                         mean_output=statistics.mean(x['completion_tokens'] for x in stages[2]),
                         max_output=max(x['completion_tokens'] for x in stages[2]),
                         finish_reasons=dict(Counter(finishes))),
        prepare_mean_output=statistics.mean(x['completion_tokens'] for x in stages[1]))


def additional_case_review(output):
    """Inspect saved text/positions, without choosing new positions or rescoring."""
    refs = {r['id']: r for r in read(output / 'reference.json')}
    annotations = read(output / 'reviewed_annotations.json')
    scores = {r['id']: r for r in read(output / 'metrics.json')['scores']['pure']['details']}
    results = {sid: read(run.base.path_for(output, 'formal', 'pure', row))
               for sid, row in refs.items()}
    repeated = []
    for sid, row in refs.items():
        result = results[sid]
        findings = location._fill_missing_pred_locations(
            location._collect_pred_findings(result, {}), row['prediction'])
        for i, diagnosis in enumerate(result['diagnostics']):
            quote = diagnosis['evidence']['quote'].strip()
            occurrences = [list(m.span()) for m in re.finditer(re.escape(quote), row['prediction'])] if quote else []
            if len(occurrences) < 2:
                continue
            selected = [findings[i].get('start_char'), findings[i].get('end_char')]
            later_inside = []
            for gt in row['physics_error_gt']:
                if not gt.get('span_valid'):
                    continue
                start, end = gt['start_char'], gt['end_char']
                contained = [span for span in occurrences if start <= span[0] and span[1] <= end]
                if contained and not (start <= selected[0] and selected[1] <= end):
                    later_inside.append(dict(error_id=gt['error_id'], occurrences=contained))
            repeated.append(dict(id=sid, diagnostic_index=i, quote=quote,
                occurrences=occurrences, actual_selected_span=selected,
                later_occurrences_inside_reference_span=later_inside,
                result=binding(run.base.path_for(output, 'formal', 'pure', row))))

    cases = []
    specifications = (
        ('trainphysics_3860', [0, 1], ['E1'], 'same_arithmetic_error_unlocatable_quotes',
         'Both diagnoses identify the reference factor-of-ten error and its consequence. The first quote drops an intermediate equality and adds So,; neither quote is found. These are two diagnoses of one reference root, not two independent true positives.'),
        ('cl_230_153846', [3], ['E6'], 'same_formula_first_occurrence_mismatch',
         'The final-speed claim agrees with E6. The exact quoted formula appears eight times; the scorer selects the first, while the last lies inside the reference span.'),
        ('cl_260_185285', [1], ['E5'], 'same_formula_first_occurrence_mismatch',
         'Both reject Qr/R as the transferred charge. The quoted formula appears five times; the selected first occurrence is outside E5, while the fifth is inside.'),
        ('trainphysics_11662', [1], ['E4'], 'verifier_repeats_the_student_error',
         'The verifier rejects the correct minimum-speed formula, repeating the student rejection that E4 flags. This is a genuine false claim, not an uncredited semantic match.'))
    for sid, indices, root_ids, disposition, explanation in specifications:
        row, result = refs[sid], results[sid]
        findings = location._fill_missing_pred_locations(
            location._collect_pred_findings(result, {}), row['prediction'])
        gt = [e for e in row['physics_error_gt'] if e['error_id'].split('/')[-1] in root_ids]
        assert len(gt) == len(root_ids) and scores[sid]['matched_error_count'] == 0
        cases.append(dict(id=sid, disposition=disposition, explanation=explanation,
            question=row['question'], result=binding(run.base.path_for(output, 'formal', 'pure', row)),
            diagnostic_indices=indices, diagnostics=[result['diagnostics'][i] for i in indices],
            actual_locations=[{k: findings[i].get(k) for k in
                              ('start_char', 'end_char', 'locatable_valid', 'locate_method')} for i in indices],
            references=gt, quote_repair=result.get('quote_repair'),
            reference_reason=[e['reason'] for e in annotations[sid]['errors'] if e['root_id'] in root_ids],
            official_matched_error_count=scores[sid]['matched_error_count']))
    assert len(repeated) == 15
    assert {e['id'] for e in repeated if e['later_occurrences_inside_reference_span']} == {'cl_230_153846', 'cl_260_185285'}
    return dict(source=binding(Path(__file__)), human_gold=False, new_model_calls=0,
        modified_predictions_or_labels=False, rescoring_calls=0,
        scope='All 60 published pure diagnoses were read in this targeted assistant review; only the explicit cases below are adjudicated. This is not an all-diagnosis semantic score or physical certification of the gold.',
        repeated_exact_quote_count=len(repeated), later_occurrence_inside_reference_count=2,
        repeated_quote_inventory=repeated, cases=cases,
        projectile_formula_check=dict(
            assumptions='Uniform g, no drag, point target (x,y); minimize launch speed over flight time t>0.',
            derivation='v0^2 = x^2/t^2 + (y/t + g*t/2)^2 = (x^2+y^2)/t^2 + g*y + g^2*t^2/4 >= g*(sqrt(x^2+y^2)+y). Equality at t^2=2*sqrt(x^2+y^2)/g.',
            conclusion='The formula the verifier rejects is correct under the stated problem assumptions.'),
        interpretation='Repeated-quote counts describe the existing locator, not extra awarded hits. Claims and corrections in other diagnoses may still be wrong; no semantic precision/recall is inferred.')


def extend_case_audit(output):
    """Update the single post-hoc analysis; all frozen experiment files stay read-only."""
    target = output / 'score_drop_analysis.json'
    report = read(target)
    for evidence in [*report['dependencies'].values(), report['new']['metrics']]:
        assert binding(evidence['path']) == evidence, evidence['path']
    extra = additional_case_review(output)
    if 'additional_case_review' in report:
        assert report['additional_case_review'] == extra, 'Existing case audit differs; inspect instead of overwriting.'
        print(json.dumps(dict(output=str(target), unchanged=True)))
        return
    report['additional_case_review'] = extra
    tmp = target.with_suffix('.json.tmp')
    with tmp.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    tmp.replace(target)
    print(json.dumps(dict(output=str(target), additional_cases=len(extra['cases']),
                          repeated_quotes=extra['repeated_exact_quote_count'],
                          later_reference_occurrences=extra['later_occurrence_inside_reference_count'],
                          new_model_calls=0, rescoring_calls=0)))


def analyze(output):
    plan = read(output / 'plan.json')['configuration']
    old = Path(plan['legacy'])
    old_plan = read(old / 'plan.json')['configuration']
    equal_keys = ('model', 'limits', 'check_prompt', 'skill_prompt', 'query_max_tokens',
                  'reference_token_cap', 'max_items', 'tree_branches')
    assert all(plan[k] == old_plan[k] for k in equal_keys)
    assert read(old / 'runtime.json') == read(output / 'runtime.json')
    new_scores = read(output / 'metrics.json')['scores']
    old_scores = read(old / 'metrics.json')['scores_all_99']
    new_rows = read(output / 'inputs.json'); old_rows = read(old / 'inputs.json')
    refs = {r['id']: r for r in read(output / 'reference.json')}
    annotations = read(output / 'reviewed_annotations.json')
    old_by_question = {r['question']: r for r in old_rows}
    new_by_question = {r['question']: r for r in new_rows}
    common = set(old_by_question) & set(new_by_question)
    assert len(common) == 41
    common_scores = {}
    for label, by_question, scores in [('old', old_by_question, old_scores),
                                       ('new', new_by_question, new_scores)]:
        ids = {by_question[q]['id'] for q in common}
        details = [d for d in scores['pure']['details'] if d['id'] in ids]
        common_scores[label] = dict(counts(details),
            mean_answer_characters=statistics.mean(len(by_question[q]['prediction']) for q in common))

    # Exhaustive inspection of ALL tree hits, rather than selected winning cases.
    dispositions = {
        'cl_132_54083': ('same_claim_quote_not_locatable', [1],
            'Both diagnose nonrelativistic kinetic energy; the pure quote omits an intervening equality.'),
        'cl_265_190251': ('same_claim_different_occurrence', [0],
            'Both diagnose the missing r sin(theta) line element; the pure quote is the downstream formula. Tree received no full reference.'),
        'trainphysics_2284': ('same_claim_quote_not_locatable', [0, 1],
            'Both diagnose the monopole force sign/normalization; the pure quote differs from the boxed source formula.'),
        'trainphysics_4317': ('pure_content_failure', [],
            'Pure CHECK failed JSON parsing. This is not a clean comparison of two published physical assessments.'),
        'trainphysics_7959': ('shared_location_hit', [0],
            'Both receive the same pressure-gradient sign location hit.'),
    }
    hit_details = [d for d in new_scores['rule_tree']['details'] if d['matched_error_count']]
    assert {d['id'] for d in hit_details} == set(dispositions)
    cases = []
    for detail in hit_details:
        sid = detail['id']; row = refs[sid]
        pp = run.base.path_for(output, 'formal', 'pure', row)
        tp = run.base.path_for(output, 'formal', 'rule_tree', row)
        pure, tree = read(pp), read(tp)
        disposition, pure_indices, explanation = dispositions[sid]
        findings = location._fill_missing_pred_locations(location._collect_pred_findings(pure, {}), row['prediction'])
        cases.append(dict(id=sid, disposition=disposition, explanation=explanation,
            pure_result=binding(pp), tree_result=binding(tp), pure_status=pure['status'],
            pure_diagnostics=[pure['diagnostics'][i] for i in pure_indices],
            pure_locations=[{k: findings[i].get(k) for k in
                ('start_char', 'end_char', 'locatable_valid', 'locate_method')} for i in pure_indices],
            tree_diagnostics=[tree['diagnostics'][i] for i in detail['matched_finding_indices']],
            tree_location_matches=detail['location_matches'],
            tree_provided_ids=tree['retrieval']['provided_complete_ids']))

    boundaries = []
    for sid, ids in [('cl_116_37137', ['E1']), ('cl_270_195010', ['E1']),
                     ('trainphysics_6208', ['E1', 'E2'])]:
        annotation = annotations[sid]
        errors = [e for e in annotation['errors'] if e['root_id'] in ids]
        assert len(errors) == len(ids) and all(e['status'] == 'confirmed' for e in errors)
        result_path = run.base.path_for(output, 'formal', 'pure', refs[sid])
        result = read(result_path)
        assert result['status'] == 'completed' and result['diagnostics'] == []
        boundaries.append(dict(id=sid, annotation_coverage=annotation['coverage_notes'],
            retained_claims=[{k: e[k] for k in ('root_id', 'claim', 'quote', 'reason')} for e in errors],
            pure_result=binding(result_path),
            interpretation='The annotation explicitly retains corrected intermediate errors; the verifier prompt distinguishes corrected tentative steps. No label edited.'))

    # A short arithmetic counterexample prevents treating every empty result as
    # either an evaluation artefact or a genuinely error-free answer.
    sid = 'cl_123_44459'
    empty_path = run.base.path_for(output, 'formal', 'pure', refs[sid])
    assert read(empty_path)['diagnostics'] == []
    actual = 2 * __import__('math').pi / 86400
    error = next(e for e in annotations[sid]['errors'] if e['root_id'] == 'E1')
    report = dict(
        schema='aligned_step_score_drop_analysis_v1', human_gold=False,
        new_model_calls=0, rescoring_calls=0, modified_predictions_or_labels=False,
        scope='Post-hoc explanation of stored metrics and targeted assistant claim comparison; no semantic precision/recall estimate.',
        dependencies={n: binding(output / n) for n in
                      ('plan.json', 'acceptance.json', 'reference.json', 'reviewed_annotations.json')},
        source=binding(Path(__file__)), old=aggregate(old, 'scores_all_99'), new=aggregate(output, 'scores'),
        protocol=dict(equal_configuration_keys=list(equal_keys), equal_model_files_and_packages=True,
                      full_request_and_scoring_verification='See unchanged acceptance.json; equality of results is not required by method alignment.'),
        common_questions=dict(n=41, identical_answers=0, groups=common_scores,
                              scope='Exploratory same-question subset; student answers and labels differ.'),
        tree_hit_review=dict(cases=cases, counts=dict(Counter(c['disposition'] for c in cases)),
            conclusion='The four extra scored tree hits consist of three same-claim quote/location differences and one pure content failure; they do not establish extra physical errors discovered because of the rule library.'),
        annotation_boundary_examples=boundaries,
        annotation_boundary_scope='At least four confirmed claims across these three inspected questions; not an estimate of the full prevalence and not authorization to remove labels.',
        genuine_missed_error=dict(id=sid, pure_result=binding(empty_path),
            reference_claim=error['claim'], reference_quote=error['quote'],
            independently_computed_earth_omega=actual,
            correct_speed_difference_m_per_s=actual * 100,
            scope='Elementary numerical check; not certification of every annotation in this question.'),
        unmeasured=['Semantic precision/recall on all diagnoses',
                    'Full prevalence of corrected-intermediate reference claims',
                    'Causal contribution of longer answers versus different error difficulty',
                    'Repeated-run variance or rule-library causal benefit'])
    target = output / 'score_drop_analysis.json'
    with target.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(target), tree_review=report['tree_hit_review']['counts'],
                          confirmed_boundary_examples=4, new_model_calls=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extend-case-audit', action='store_true',
                        help='Append verified raw cases to the existing post-hoc analysis; no generation or rescoring.')
    args = parser.parse_args()
    if args.extend_case_audit:
        extend_case_audit(args.output)
    else:
        analyze(args.output)
