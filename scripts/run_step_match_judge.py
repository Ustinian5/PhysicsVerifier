"""Judge frozen step diagnoses with independent server-side GPT-OSS-120B conversations."""
import argparse
from collections import Counter
from functools import partial
from pathlib import Path
import time

import httpx

from rule_framework import step_match as match
from scripts import run_frozen_step_bm25 as frozen
from scripts import run_final_answer_revision as transport
from scripts import serve_frozen_bm25 as node

base, vr, require = frozen.base, frozen.vr, frozen.require
write = transport.immutable
ARMS = base.p.ARMS
MAX_OUTPUT = 4096
BENCH_FILES = {
    'inputs.json': frozen.INPUTS,
    'reference.json': 'd33e4f2355def946e63077a02d7107534c2fca28e7ea37a2d35d5bf7c0ef566d',
    'reviewed_annotations.json': '0f1920457a42a0b8a021d4d15f4a54d2db7486c99269f23893a72a4de5844fad',
    'unresolved.json': 'ab0d9bb84c902f8d5358f5e96e3b0b5377b8ad5692c6a41d1097bcbf08326063',
    'offline_review/labels.json': frozen.LABELS,
}
CODE = ('rule_framework/step_match.py', 'scripts/run_step_match_judge.py', 'scripts/review_aligned_step_scoring.py',
        'scripts/run_final_answer_revision.py', 'scripts/run_qwen30b_step_checks.py', 'scripts/gpt_oss_harmony_transport.py')


def benchmark(directory):
    for name, sha in BENCH_FILES.items():
        require(vr.sha(directory / name) == sha, 'Frozen benchmark changed: ' + name)
    labels = vr.read(directory / 'offline_review/labels.json')['records']
    states = {r['root_id']: r['state'] for r in labels}
    reference = {r['id']: r for r in vr.read(directory / 'reference.json')}
    annotations = vr.read(directory / 'reviewed_annotations.json')
    unknown = vr.read(directory / 'unresolved.json')
    inputs = vr.read(directory / 'inputs.json')
    rows = []
    for row in inputs:
        sid = row['id']
        if sid in frozen.sources.STEP_HOLD:
            continue
        ref = reference[sid]
        require(all(ref[k] == v for k, v in row.items()), 'Wrong 30B answer for root labels')
        gt = [g for g in ref['physics_error_gt'] if states[g['error_id']] == 'active_error']
        by_id = {e['root_id']: e for e in annotations[sid]['errors']}
        roots = []
        for g in gt:
            rid = g['error_id'].removeprefix(sid + '/')
            e = by_id[rid]
            require(g['span_valid'] and row['prediction'][g['start_char']:g['end_char']] == e['quote'] == g['answer_quote'],
                    'Reference source evidence changed')
            roots.append({k: e[k] for k in ('root_id', 'claim', 'reason', 'correct_relation', 'premises', 'quote')})
        rows.append(dict(row, roots=roots, gt=gt, unresolved=unknown[sid]))
    coverage = dict(registered_questions=len(inputs), registered_active_roots=sum(v == 'active_error' for v in states.values()),
                    evaluated_questions=len(rows), evaluated_active_roots=sum(len(r['gt']) for r in rows),
                    held_ids=sorted(frozen.sources.STEP_HOLD))
    require(len(inputs) == 100 and len(rows) == 99 and coverage['registered_active_roots'] == 164
            and coverage['evaluated_active_roots'] == 163, 'Different step population')
    return rows, coverage


def production(bench, published):
    rows, coverage = benchmark(bench)
    plan = vr.read(published / 'plan.json')
    signature = plan['configuration_sha256']
    require(vr.digest(plan['configuration']) == signature, 'Published plan changed')
    tasks_path = published / 'tasks.json'
    files = {'plan.json': vr.sha(published / 'plan.json')}
    tasks = {}
    if tasks_path.exists():
        require(vr.sha(tasks_path) == '98e966e35defc4912e7b98a2eccb9ec255bb7adb69ed9918cafeff480353736c'
                and vr.sha(published / 'formal_complete.json') == 'f7a9263690b2db8a01e0b8de9e748700b7770bdf1660c824edaa441984cae1e5'
                and files['plan.json'] == '130b8597dc28d1c7f9529e695d7a5529b6eefae29d00ffe18da5475aea19cd4f',
                'Archived BM25 publication changed')
        receipt = vr.read(published / 'formal_complete.json')
        tasks_list = vr.read(tasks_path)
        tasks = {(t['id'], t['arm']): t for t in tasks_list}
        expected = {(sid, a) for sid in [*(r['id'] for r in rows), *coverage['held_ids']] for a in ARMS}
        require(receipt['passed'] and receipt['configuration_sha256'] == signature and
                len(tasks_list) == len(tasks) == receipt['tasks'] and set(tasks) == expected, 'Unreleased step tasks')
        files.update({n: vr.sha(published / n) for n in ('tasks.json', 'formal_complete.json')})
    else:
        require(vr.sha(published / 'inputs.json') == frozen.INPUTS, 'Verifier cohort or answers changed')
        files['inputs.json'] = frozen.INPUTS
    arms = list(ARMS) if tasks else plan['configuration']['arms']
    require(arms in (['pure'], list(ARMS)), 'Unrecognized published comparison arms')
    results = {}
    for row in rows:
        for arm in arms:
            key = row['id'], arm
            path = base.path_for(published, 'formal', arm, row)
            task = tasks.get(key)
            reused = task is not None and task['action'] in ('reuse', 'retain_prepare_failure')
            if reused:
                path = Path(task['parent_result'])
            sha = vr.sha(path)
            value = vr.read(path)
            # The historical CHECK engine omitted organization_arm on empty terminal failures.
            # Bind those failures to the original arm-specific result path and release hash.
            terminal_failure = ('organization_arm' not in value and value['status'] == 'failed' and
                                value['diagnostics'] == [] and path.parts[-6:] == base.path_for(
                                    published, 'formal', arm, row).parts[-6:])
            require(value['id'] == row['id'] and (value.get('organization_arm') == arm or terminal_failure) and
                    (sha == task['parent_result_sha256'] if reused else value['configuration_sha256'] == signature)
                    and (task is None or reused or task['action'] == 'run_check'), 'Wrong published result')
            base.runtime.audit_result(value)
            results[key] = dict(status=value['status'], failed=not base.runtime.healthy_result(value),
                                diagnostics=value['diagnostics'], source=str(path), sha256=sha)
    return rows, results, coverage, dict(benchmark=str(bench), benchmark_files=BENCH_FILES,
                                        published=str(published), files=files, **({'arms': arms} if arms == ['pure'] else {}))


def controls():
    quote = 'I multiply 12 by 3 to obtain 36 m/s.'
    answer = 'I multiply distance by time. ' + quote
    root = dict(root_id='E1', claim='The student multiplies distance by elapsed time.',
                reason='Mean speed is distance divided by elapsed time.', correct_relation='v = d/t',
                premises='Constant speed over the reported interval.', quote='I multiply distance by time.')
    sid = 'cart_speed'
    row = dict(id=sid, question='A cart travels 12 m in 3 s. Find its mean speed.', context='', prediction=answer,
               roots=[root], gt=[dict(error_id=sid + '/E1', start_char=0, end_char=len(root['quote']))], unresolved=[])
    def diagnosis(message):
        return dict(message=message, evidence=dict(quote=quote))
    hit = diagnosis('The numerical step uses a product. Divide the path length by the elapsed time.')
    reverse = diagnosis('Multiplying distance by time is the correct speed law. Dividing by time would be the mistake.')
    unrelated = diagnosis('The unit symbol for metres is misspelled; the calculation itself is outside this diagnosis.')
    candidates = dict(pure=[hit], case_rag=[hit, hit], flat_skills=[reverse, unrelated], rule_tree=[])
    results = {(sid, arm): dict(status='completed', failed=False, diagnostics=ds, sha256=vr.digest(ds))
               for arm, ds in candidates.items()}
    coverage = dict(registered_questions=1, registered_active_roots=1, evaluated_questions=1,
                    evaluated_active_roots=1, held_ids=[])
    return [row], results, coverage, dict(synthetic_controls=True)


def prepare(out, rows, results, coverage, source):
    jobs, bindings = {}, []
    arms = source.get('arms', ARMS)
    require(list(arms) in (['pure'], list(ARMS)), 'Invalid judge population')
    for row in rows:
        sid = row['id']
        require(sid not in frozen.sources.STEP_HOLD, 'Question outside the authorized cohort')
        for arm in arms:
            item = results[sid, arm]
            keys = []
            for diagnosis in item['diagnostics']:
                messages = match.messages(row, diagnosis, row['roots'])
                key = vr.digest([sid, messages])
                jobs[key] = (row, messages)
                keys.append(key)
            bindings.append(dict(id=sid, arm=arm, status=item['status'], failed=item['failed'], keys=keys,
                                 published_sha256=item['sha256'], path=item.get('source')))
    require(len(results) == len(bindings) == len(rows) * len(arms), 'Unexpected published tasks')
    publication = dict(source=source, coverage=coverage, bindings=bindings)
    cfg = dict(protocol=match.VERSION, publication_sha256=vr.digest(publication), instruction=match.INSTRUCTION,
               code={name: vr.sha(transport.a.ROOT / name) for name in CODE},
               model=base.r.EXPECTED_MODEL, reasoning_effort='medium', temperature=0, top_p=1,
               fixed_harmony_date=node.CHECK_DATE, api_base=f'http://127.0.0.1:{node.host.PORT}/v1',
               max_tokens=MAX_OUTPUT, limits=dict(context_window=131072, request_timeout_seconds=1200))
    plan = dict(configuration=cfg, configuration_sha256=vr.digest(cfg))
    write(out / 'publication.json', publication)
    plan_path = out / 'judge_plan.json'
    if plan_path.exists():
        saved = vr.read(plan_path)
        if saved['configuration_sha256'] in {
                '9c13cfcaa35b4d521f774137a8a0cf8aebdf38497b12eabdf39f1bf11cebfa6c',
                '685441da5fe7a014c541df938c8f8ceadfc8f4f2b7cb65d6dc61148dcda514e8'}:
            # The earlier four-arm receipts predate arm selection and final-bench binding.
            # Their exact inputs, checker outputs, judge messages, validator and reducer must still agree.
            earlier_code = {**cfg['code'],
                'scripts/run_step_match_judge.py': '325b07bab930c8f7557dec7396df2ba56061bcb91eb7b8395406b2cc23507110',
                'scripts/run_final_answer_revision.py': 'fb8bbcdfd15ad0ef4941c22f8b4b03753b7ecc243dd2bffc048f0f41bb352cfc'}
            require(saved['configuration'] == dict(cfg, code=earlier_code) and
                    vr.digest(saved['configuration']) == saved['configuration_sha256'], 'Frozen judge contract drift')
            plan = saved
    write(plan_path, plan)
    return plan, bindings, jobs


def evaluate(out, data, *, tokenizer=None, client=None, offline=False, stop=None):
    rows, results, coverage, source = data
    plan, bindings, jobs = prepare(out, *data)
    decisions, failed, missing = {}, {}, []
    timeout = plan['configuration']['limits']['request_timeout_seconds']
    for key, (row, messages) in jobs.items():
        folder = out / 'judgments' / key
        validator = partial(match.validate, answer=row['prediction'], roots={r['root_id'] for r in row['roots']})
        if (folder / 'result.json').exists():
            saved = transport.saved_request(folder, messages, MAX_OUTPUT, validator, plan, available=timeout)
        elif offline:
            missing.append(key)
            continue
        else:
            require(stop is None or not stop.exists(), 'STOP requested')
            saved = transport.request_once(out, folder, messages, MAX_OUTPUT, validator, plan, tokenizer, client,
                                           available=timeout)
        if saved['status'] == 'complete':
            decisions[key] = saved['value']
        else:
            failed[key] = saved['reason']
        if not offline:
            vr.atomic(out / 'progress.json', dict(terminal=len(decisions) + len(failed), total=len(jobs),
                                                  content_failures=len(failed)))
    status = dict(configuration_sha256=plan['configuration_sha256'], unique_diagnoses=len(jobs),
                  published_diagnoses=sum(len(b['keys']) for b in bindings), valid=len(decisions),
                  missing=missing, content_failures=failed)
    if failed or missing:
        vr.atomic(out / 'status.json', dict(status, state='incomplete'))
        raise ValueError(f"Judge coverage incomplete: {len(missing)} missing, {len(failed)} content failures")
    by_id = {row['id']: row for row in rows}
    groups = {a: [] for a in source.get('arms', ARMS)}
    for b in bindings:
        sid, arm = b['id'], b['arm']
        groups[arm].append(match.score_row(by_id[sid], results[sid, arm], [decisions[k] for k in b['keys']]))
    summaries = {arm: match.summarize(details) for arm, details in groups.items()}
    report = dict(protocol=match.VERSION, configuration_sha256=plan['configuration_sha256'], human_gold=False,
                  coverage=coverage, judge=status, recorded_tasks={a: len(ds) for a, ds in groups.items()},
                  scores=summaries, details_sha256=vr.digest(groups))
    if source.get('synthetic_controls'):
        dispositions = [decisions[k]['disposition'] for b in bindings if b['arm'] == 'flat_skills' for k in b['keys']]
        require(dispositions == ['opposite_claim', 'other_claim'] and [summaries[a]['hits'] for a in ('pure', 'case_rag', 'flat_skills', 'rule_tree')] == [1, 1, 0, 0]
                and summaries['case_rag']['locatable'] == summaries['flat_skills']['locatable'] == 2
                and status['valid'] == 3, 'Synthetic root matching or duplicate accounting failed')
    write(out / 'details.json', groups)
    write(out / 'metrics.json', report)
    vr.atomic(out / 'status.json', dict(status, state='complete'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'confirm', 'score'])
    parser.add_argument('--output', type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--controls', action='store_true')
    choice.add_argument('--published', type=Path, help='Complete four-arm step publication')
    parser.add_argument('--benchmark', type=Path, default=frozen.PARENT)
    parser.add_argument('--gpu', type=int, default=4, choices=(4, 5, 6, 7))
    args = parser.parse_args()
    out = args.output.resolve()
    if args.published:
        require(out not in {args.benchmark.resolve(), args.published.resolve()}, 'Use a separate judge results directory')
    if args.command == 'confirm':
        node.node()
        return node.confirm(out, vr.read(out / 'judge_plan.json')['configuration_sha256'])
    if args.command == 'run':
        node.node()
    with base.r.run_lock(out / 'runner.lock'):
        data = controls() if args.controls else production(args.benchmark.resolve(), args.published.resolve())
        plan, _, jobs = prepare(out, *data)
        if args.command == 'prepare':
            return
        if args.command == 'score':
            return evaluate(out, data, offline=True)
        signature = plan['configuration_sha256']
        try:
            if all((out / 'judgments' / key / 'result.json').exists() for key in jobs):
                report = evaluate(out, data, offline=True)
            else:
                base.runtime.wait_detached(out, signature)
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(node.host.HOT / 'models/gpt-oss-120b', local_files_only=True)
                with node.service(out, args.gpu, out / 'STOP', signature) as endpoint:
                    require(endpoint == plan['configuration']['api_base'], 'Wrong judge endpoint')
                    with httpx.Client(trust_env=False) as client:
                        report = evaluate(out, data, tokenizer=tokenizer, client=client)
            vr.atomic(out / 'exit.json', dict(code=0, configuration_sha256=signature,
                                              valid=report['judge']['valid'], unix=time.time()))
        except Exception as exc:
            vr.atomic(out / 'exit.json', dict(code=1, configuration_sha256=signature, error=str(exc), unix=time.time()))
            raise


if __name__ == '__main__':
    main()
