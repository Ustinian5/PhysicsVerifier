"""Replay the published English pure baseline in a four-arm step comparison."""
from functools import partial
from pathlib import Path

from scripts import repair_step_judge_locations as location
from scripts import run_step_match_judge as judge

vr, require = judge.vr, judge.require
PURE_CONFIG = 'eac43473b16c0da5e63d4797ecde6d570f38c560dd3f41d55edc9f72ce5b6c45'
PURE_METRICS = 'eae83906ecc3ec4c84842a15b41a716d1f9182eab5531a35335dc3aabbf92126'
PURE_PILOT = '2d1457ec5d1169f1f006ca52317eb46767b12fedb8c2c705e3ec1c4e14cb2c84'
POLICIES = ('stop', 'locate_once')


def baseline(folder):
    plan = vr.read(folder / 'plan.json')
    cfg = plan['configuration']
    require(vr.digest(cfg) == plan['configuration_sha256'] == PURE_CONFIG and cfg['arms'] == ['pure']
            and cfg['online_reference_visible'] is False and vr.sha(folder / 'metrics.json') == PURE_METRICS
            and vr.sha(folder / 'engineering_acceptance.json') == PURE_PILOT, 'Different baseline or acceptance')
    for row in cfg['controls']:
        v = vr.read(judge.base.path_for(folder, 'controls', 'pure', row))
        judge.base.runtime.audit_result(v)
        require(v['configuration_sha256'] == PURE_CONFIG and v['id'] == row['id'] and
                v['protocol_audit']['passed'] and judge.base.runtime.healthy_result(v), 'Different pure control')
    out, repair_plan, data, bindings, jobs, decisions, drafts = location.prepare(folder)
    spans = {}
    for key, draft in drafts.items():
        row, messages = jobs[key]
        args = (out/'calls'/key, location.location_messages(messages, draft), judge.MAX_OUTPUT,
                partial(location.validate_location, row['prediction']), repair_plan)
        saved = judge.transport.saved_request(*args, available=repair_plan['configuration']['limits']['request_timeout_seconds'])
        require(saved['status'] == 'complete', 'Missing accepted baseline location')
        decisions[key], spans[key] = location.merge_location(row, draft, saved['value'])
    rows, results, coverage, _ = data
    by_id = {r['id']: r for r in rows}
    details = [location.score_row(by_id[b['id']], results[b['id'], 'pure'], [decisions[k] for k in b['keys']],
                                  [spans.get(k) for k in b['keys']]) for b in bindings]
    report = vr.read(folder / 'metrics.json')
    require(vr.digest({'pure': details}) == report['details_sha256'] and judge.match.summarize(details) ==
            report['scores']['pure'] and coverage == report['coverage'], 'Frozen baseline replay failed')
    return cfg, rows, results, coverage, decisions, spans, details


def compatible(current, previous):
    keys = ('prompt_profile', 'model', 'fixed_harmony_date', 'pure_prompt', 'online_reference_visible',
            'controls', 'limits', 'budget_limits', 'step_match_judge', 'sample_ids', 'cohort')
    require(all(current[k] == previous[k] for k in keys), 'Pure task, population or inference parameters changed')


def request(folder, root, messages, validator, plan, tokenizer, client):
    timeout = plan['configuration']['limits']['request_timeout_seconds']
    args = (folder, messages, judge.MAX_OUTPUT, validator, plan)
    if (folder / 'result.json').is_file():
        return judge.transport.saved_request(*args, available=timeout)
    if client is None:
        return None
    require(not (root / 'STOP').exists(), 'STOP requested')
    return judge.transport.request_once(root, *args, tokenizer, client, available=timeout)


class CitationDecisionPending(ValueError):
    """All first judgments are recorded; only exact-source citations remain unresolved."""


def evaluate(out, checker, *, tokenizer=None, client=None):
    cfg = checker['configuration']
    base_path = Path(cfg['reuse_pure'])
    previous, rows, results, coverage, decisions, spans, expected = baseline(base_path)
    compatible(cfg, previous)
    reused = set(decisions)
    for row in rows:
        for arm in cfg['prepare_arms']:
            path = judge.base.path_for(out, 'formal', arm, row)
            value = vr.read(path)
            failed = not judge.base.runtime.healthy_result(value)
            require(value['id'] == row['id'] and value['configuration_sha256'] == checker['configuration_sha256']
                    and (value.get('organization_arm') == arm or
                         ('organization_arm' not in value and failed and value['diagnostics'] == [])),
                    'Different published knowledge-arm check')
            judge.base.runtime.audit_result(value)
            results[row['id'], arm] = dict(sha256=vr.sha(path), source=str(path), failed=failed,
                                           status=value['status'], diagnostics=value['diagnostics'])
    directory = out / 'step_judge'
    policy = cfg['judge_location_policy']
    require(policy in POLICIES, 'Unknown location policy')
    source = dict(checker_plan_sha256=vr.sha(out/'plan.json'), frozen_pure_metrics_sha256=PURE_METRICS,
                  benchmark_files=judge.BENCH_FILES, location_policy=policy, code_sha256=vr.sha(Path(__file__)),
                  location_code_sha256=vr.sha(Path(location.__file__)))
    plan, bindings, jobs = judge.prepare(directory, rows, results, coverage, source)
    require(reused <= jobs.keys(), 'The released checks do not contain the original baseline diagnoses')
    missing, invalid, repairs = [], {}, {}
    for key, (row, messages) in jobs.items():
        if key in reused:
            continue
        folder = directory/'judgments'/key
        validator = partial(judge.match.validate, answer=row['prediction'], roots={r['root_id'] for r in row['roots']})
        saved = request(folder, out, messages, validator, plan, tokenizer, client)
        if saved is None:
            missing.append(key)
        elif saved['status'] == 'complete':
            decisions[key] = saved['value']
        elif policy == 'locate_once' and saved['reason'] in location.QUOTE_ERRORS:
            draft = judge.base.r.parse_json(vr.read(folder/'response.json')['content'])
            # All schema, disposition and root checks precede citation validation.
            try:
                validator(draft)
            except ValueError as exc:
                require('ValueError: ' + str(exc) == saved['reason'], 'Wrong initial citation failure')
            else:
                raise ValueError('Valid original decision scheduled for repair')
            second = request(directory/'locations'/key, out, location.location_messages(messages, draft),
                             partial(location.validate_location, row['prediction']), plan, tokenizer, client)
            if second is None:
                missing.append(key)
            elif second['status'] == 'complete':
                decisions[key], spans[key] = location.merge_location(row, draft, second['value'])
                repairs[key] = dict(lines=second['value'], source_span=spans[key], disposition=decisions[key]['disposition'])
            else:
                invalid[key] = second['reason']
        else:
            invalid[key] = saved['reason']
        if client is not None:
            vr.atomic(directory/'progress.json', dict(valid=len(decisions), total=len(jobs), invalid=len(invalid),
                                                      reused=len(reused), locations=len(repairs)))
    status = dict(unique_diagnoses=len(jobs), published_diagnoses=sum(len(b['keys']) for b in bindings),
                  valid=len(decisions), baseline_judgments_reused=len(reused), new_locations=len(repairs),
                  missing=missing, content_failures=invalid, configuration_sha256=plan['configuration_sha256'])
    if invalid or missing:
        vr.atomic(directory/'status.json', dict(status, state='incomplete'))
        if policy == 'stop' and not missing and set(invalid.values()) <= location.QUOTE_ERRORS:
            raise CitationDecisionPending(f'Judge incomplete: {len(invalid)} invalid citations, 0 missing')
        raise ValueError(f'Judge incomplete: {len(invalid)} invalid, {len(missing)} missing')
    by_id = {r['id']: r for r in rows}
    details = {arm: [] for arm in cfg['arms']}
    for b in bindings:
        sid, arm, keys = b['id'], b['arm'], b['keys']
        details[arm].append(location.score_row(by_id[sid], results[sid, arm], [decisions[k] for k in keys],
                                               [spans.get(k) for k in keys]))
    require(details['pure'] == expected, 'Four-arm baseline differs from the frozen pure results')
    report = dict(protocol=judge.match.VERSION, human_gold=False, coverage=coverage, judge=status,
                  configuration_sha256=plan['configuration_sha256'], details_sha256=vr.digest(details),
                  recorded_tasks={arm: len(ds) for arm, ds in details.items()},
                  scores={arm: judge.match.summarize(ds) for arm, ds in details.items()})
    for name, item in [('details', details), ('located_evidence', repairs), ('metrics', report)]:
        judge.write(directory/(name+'.json'), item)
    judge.write(out/'metrics.json', report)
    vr.atomic(directory/'status.json', dict(status, state='complete'))
    return report
