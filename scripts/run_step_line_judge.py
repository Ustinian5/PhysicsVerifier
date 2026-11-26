"""Measure one-pass line-based step judgments from frozen checker publications on node14."""
from collections import Counter
from functools import partial
from pathlib import Path
import time

from rule_framework import step_match_lines as lines, verifier_english as english
from scripts import reuse_step_baseline as reuse

judge, vr, require = reuse.judge, reuse.vr, reuse.require
WORK = judge.node.host.HOT/'results/qwen30b_step_verifier/english_blind_verifier'
ROOT = WORK/'step_judge_lines'
PORT = 18253
CODE = (*judge.CODE, 'rule_framework/step_match_lines.py', 'scripts/run_step_line_judge.py',
        'rule_framework/verifier_english.py', 'scripts/repair_step_judge_locations.py', 'scripts/reuse_step_baseline.py')


def contract():
    return dict(protocol=lines.VERSION, instruction=lines.INSTRUCTION, model=judge.base.r.EXPECTED_MODEL,
                prompt_profile=english.PROFILE, fixed_harmony_date=judge.node.CHECK_DATE, reasoning_effort='medium',
                temperature=0, top_p=1, max_tokens=judge.MAX_OUTPUT, calls_per_unique_diagnosis=1,
                limits=dict(context_window=131072, request_timeout_seconds=1200),
                code={n: vr.sha(judge.transport.a.ROOT/n) for n in CODE})


def publication(pure, out=None, checker=None):
    previous, rows, results, coverage, *_ = reuse.baseline(pure)
    arms = ['pure']
    source = dict(pure_plan_sha256=reuse.PURE_CONFIG, pure_publication_sha256=vr.sha(pure/'step_judge/publication.json'),
                  benchmark_files=judge.BENCH_FILES)
    if checker is not None:
        cfg = checker['configuration']
        require(Path(cfg['reuse_pure']) == pure and cfg['step_match_judge'] == lines.VERSION, 'Different line judgment task')
        reuse.compatible(cfg, previous, same_judge=False)
        reuse.add_knowledge_checks(out, checker, rows, results)
        source['checker_plan_sha256'] = vr.sha(out/'plan.json')
        arms = cfg['arms']
    return rows, results, coverage, dict(source, arms=arms)


def controls():
    rows, results, coverage, source = judge.controls()
    row = rows[0]
    row['prediction'] = 'I record the distance and time.\r\n' + row['roots'][0]['quote'] + '\r\n' + results[
        row['id'], 'pure']['diagnostics'][0]['evidence']['quote']
    start = row['prediction'].index(row['roots'][0]['quote'])
    row['gt'][0].update(start_char=start, end_char=start + len(row['roots'][0]['quote']))
    # A real null-evidence case, alongside supported, opposite and duplicate claims.
    results[row['id'], 'flat_skills']['diagnostics'][1] = dict(
        message='The solution claims the cart carries a blue light; that claim is unsupported.',
        evidence=dict(quote='The cart carries a blue light.'))
    for item in results.values():
        item['sha256'] = vr.digest(item['diagnostics'])
    return rows, results, coverage, dict(source, arms=list(judge.ARMS))


def bind(out, data, shared):
    rows, results, coverage, source = data
    jobs, bindings = {}, []
    cfg = contract()
    judge.write(shared/'protocol.json', cfg)
    for row in rows:
        sid = row['id']
        require(sid not in judge.frozen.sources.STEP_HOLD, 'Source outside the approved step cohort')
        for arm in source['arms']:
            item = results[sid, arm]
            keys = []
            for diagnosis in item['diagnostics']:
                messages = lines.messages(row, diagnosis)
                key = vr.digest([sid, messages])
                jobs[key] = (row, messages)
                keys.append(key)
            bindings.append(dict(id=sid, arm=arm, result_sha256=item['sha256'], keys=keys))
    require(len(bindings) == len(results) == len(rows)*len(source['arms']), 'Incomplete published checks')
    release = dict(protocol_sha256=vr.digest(cfg), coverage=coverage, source=source, bindings=bindings)
    judge.write(out/'publication.json', release)
    return cfg, release, jobs


def request(shared, key, row, messages, cfg, endpoint, client, tokenizer):
    folder = shared/'judgments'/key
    route = folder/'transport.json'
    allowed = {f'http://127.0.0.1:{port}/v1' for port in (18083, PORT)}
    if route.exists():
        endpoint = vr.read(route)['api_base']
    require(endpoint in allowed, 'Unregistered judge service')
    run_cfg = dict(cfg, api_base=endpoint)
    plan = dict(configuration=run_cfg, configuration_sha256=vr.digest(run_cfg))
    args = (folder, messages, judge.MAX_OUTPUT, partial(lines.validate, row), plan)
    timeout = cfg['limits']['request_timeout_seconds']
    if (folder/'result.json').exists():
        require(route.is_file(), 'Missing original judge endpoint')
        saved = judge.transport.saved_request(*args, available=timeout)
        if saved['status'] == 'content_failure':
            rec = vr.read(folder/'response.json')
            if not rec.get('error'):
                try:
                    lines.validate(row, judge.base.r.parse_json(rec['content']))
                except (ValueError, TypeError) as exc:
                    require(saved['reason'] == type(exc).__name__ + ': ' + str(exc), 'Failure replay differs')
                else:
                    raise ValueError('Archived first-pass failure now parses as valid')
        return saved
    if client is None:
        return None
    require(not (shared/'STOP').exists(), 'STOP requested')
    judge.write(route, dict(api_base=endpoint))
    return judge.transport.request_once(shared, *args, tokenizer, client, available=timeout)


def counts(values):
    categories = Counter(v['location_status'] for v in values)
    total = len(values)
    located, absent = categories['located'], categories['no_passage']
    failures = total - located - absent
    return dict(total=total, located=located, no_passage=absent, invalid_range=categories['invalid_range'],
                missing_match_evidence=categories['missing_match_evidence'], content_failure=categories['content_failure'],
                technical_failures=failures, one_pass_valid=located+absent,
                technical_failure_rate=failures/total if total else None,
                located_rate=located/total if total else None)


def evaluate(out, data, shared, *, endpoint='http://127.0.0.1:18083/v1', tokenizer=None, client=None):
    rows, results, coverage, source = data
    cfg, release, jobs = bind(out, data, shared)
    values, missing = {}, []
    for key, (row, messages) in jobs.items():
        saved = request(shared, key, row, messages, cfg, endpoint, client, tokenizer)
        if saved is None:
            missing.append(key)
            continue
        if saved['status'] == 'complete':
            values[key] = saved['value']
        else:
            require(saved['status'] == 'content_failure', 'Unclassified judge transport failure')
            values[key] = dict(location_status='content_failure', reason=saved['reason'], span=None,
                               effective_decision=dict(disposition='unclear', root_ids=[], source_quote='',
                                                       rationale='No usable first-round judgment.'))
        if client is not None:
            vr.atomic(out/'progress.json', dict(terminal=len(values), total=len(jobs), unix=time.time()))
    require(not missing, f'Missing {len(missing)} one-pass judge responses')
    by_id = {row['id']: row for row in rows}
    details, grouped = {a: [] for a in source['arms']}, {a: [] for a in source['arms']}
    for binding in release['bindings']:
        sid, arm = binding['id'], binding['arm']
        items = [values[k] for k in binding['keys']]
        grouped[arm].extend(items)
        details[arm].append(reuse.location.score_row(by_id[sid], results[sid, arm],
            [i['effective_decision'] for i in items], [i['span'] for i in items]))
    report = dict(protocol=lines.VERSION, publication_sha256=vr.sha(out/'publication.json'), coverage=coverage,
                  formal_metrics_released=False, acceptance_threshold=None, human_gold=False,
                  first_pass={arm: counts(vals) for arm, vals in grouped.items()}, unique=counts(values.values()),
                  recorded_tasks={a:len(v) for a,v in details.items()}, details_sha256=vr.digest(details),
                  provisional_scores={a:judge.match.summarize(v) for a,v in details.items()})
    if source.get('synthetic_controls'):
        expected = [v['effective_decision']['disposition'] for v in grouped['flat_skills']]
        require(expected[0] == 'opposite_claim' and expected[1] in ('other_claim','unclear') and grouped['flat_skills'][1]['span'] is None and
                {a:s['hits'] for a,s in report['provisional_scores'].items()} ==
                    {'pure':1,'case_rag':1,'flat_skills':0,'rule_tree':0} and
                report['unique']['total'] == 3 and report['unique']['technical_failures'] == 0,
                'One-pass synthetic evidence or root-direction control failed')
    judge.write(out/'details.json', details)
    judge.write(out/'decisions.json', values)
    judge.write(out/'first_pass.json', report)
    return report


def measure_pure(root, *, endpoint, tokenizer=None, client=None):
    for name, data in [('controls', controls()), ('pure', publication(root.parent/'step_pure'))]:
        report = evaluate(root/name, data, root, endpoint=endpoint, tokenizer=tokenizer, client=client)
        if name == 'pure':
            require(report['recorded_tasks'] == {'pure':99} and report['coverage']['evaluated_active_roots'] == 163
                    and report['unique']['total'] == report['first_pass']['pure']['total'] == 63, 'Different frozen pure release')
    return report


def audit_pure(root, *, measured):
    pure = root.parent/'step_pure'
    _, rows, results, _, older, fixed, _ = reuse.baseline(pure)
    old_locations = vr.read(pure/'step_judge/location_repair/locations.json')
    current = vr.read(root/'pure/decisions.json') if measured else {}
    counts_by_case = Counter()
    checks = []
    for row in rows:
        sid, answer = row['id'], row['prediction']
        for index, diagnosis in enumerate(results[sid,'pure']['diagnostics']):
            old = vr.digest([sid, judge.match.messages(row, diagnosis, row['roots'])])
            key = vr.digest([sid, lines.messages(row, diagnosis)])
            old_decision = older[old]
            if old in old_locations:
                span = fixed[old]
                draft = {k:old_decision[k] for k in ('disposition','root_ids','rationale')}
                selection = lines.validate(row, dict(draft, **old_locations[old]['lines']))
                require(selection['span'] == span and selection['effective_decision'] == old_decision,
                        'Existing line selection no longer recovers the same evidence or decision')
                counts_by_case['historical_line_ranges_verified'] += 1
            if measured:
                new = current[key]
                span = new['span']
                earlier = [fixed[old]] if old in fixed and fixed[old] else [
                    judge.match.location.source_span(answer,a,b) for a,b in
                    judge.match.location.occurrences(answer,old_decision['source_quote'])]
                overlaps = bool(span and any(judge.match.location.overlap(span,p) >= 0 for p in earlier))
                agrees = (new['location_status'] != 'content_failure' and
                          new['effective_decision']['disposition'] == old_decision['disposition'] and
                          set(new['effective_decision']['root_ids']) == set(old_decision['root_ids']))
                counts_by_case['prior_judge_cited_evidence'] += bool(earlier)
                counts_by_case['new_range_overlaps_prior_evidence'] += overlaps
                counts_by_case['same_disposition_and_root_ids'] += agrees
                checks.append(dict(id=sid,index=index,key=key,previously_cited=bool(earlier),overlaps=overlaps,
                                   same_decision=agrees,previously_needed_location=old in old_locations))
    require(counts_by_case['historical_line_ranges_verified'] == 31 and (not measured or len(checks) == 63),
            'Different published pure cohort')
    file = root/'pure'/('comparison_with_previous.json' if measured else 'historical_line_replay.json')
    judge.write(file, dict(model_calls_for_audit=0, historical_metrics_sha256=reuse.PURE_METRICS,
                           fresh_first_round_measured=measured, counts=dict(counts_by_case), cases=checks))


def comparison(out, checker, *, tokenizer=None, client=None):
    shared = out.parent/'step_judge_lines'
    args = dict(endpoint=checker['configuration']['api_base'], tokenizer=tokenizer, client=client)
    with judge.base.r.run_lock(shared/'judgments.lock'):
        pure = measure_pure(shared, **args)
        data = publication(Path(checker['configuration']['reuse_pure']), out, checker)
        report = evaluate(out/'step_judge_first_pass', data, shared, **args)
        require(report['first_pass']['pure'] == pure['first_pass']['pure'] and
                report['provisional_scores']['pure'] == pure['provisional_scores']['pure'],
                'The four-arm comparison differs from the new one-pass pure decisions')
        return report
