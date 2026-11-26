"""Evaluate frozen 30B solutions with the step-level verifier and the scorer bound to its release.

The English profile uses a direct 120B baseline and one shared preparation
for RAG, skills and the rule tree. The saved plan selects the checking protocol.
"""
import argparse
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from rule_framework import verifier_english as en

import httpx

from scripts import run_aligned_organization as a
from scripts.run_qwen30b_step_checks import reference_rows

base, vr, require = a.base, a.vr, a.require
VERSION = 'corrected100_oldv15_released_v1'
ENGLISH_VERSION = 'english_step_blind_with_120b_judge_v1'
PILOT_SHA = '618991c554aebc6d2875aab871eaa6a8713e6089008534b7718e586254385855'
KNOWN_SOURCE = '62a07c4b8297c063fcb852b972e21db38f50ed9137326ff13b104f085544bdf0'


def code_hashes(*, english=False):
    names = set(a.code_hashes()) | {'scripts/run_aligned_step_evaluation.py',
        'scripts/run_bench_reannotation.py', 'scripts/bench_reannotation_runtime.py'}
    if english:
        from scripts import run_step_match_judge as judge
        names.update((*judge.CODE, 'rule_framework/verifier_english.py', 'scripts/resume_aligned_step_evaluation.py',
                      'scripts/serve_frozen_bm25.py', 'scripts/run_node14_final_pilot.py'))
    return {n: vr.sha(a.ROOT / n) for n in sorted(names)}


def apply_quote_review(rows, finals, review, final_hashes):
    """Only exact quote edits; physical claims, statuses and order are preserved."""
    require(review['human_gold'] is False and review['additional_experiment_api_calls'] == 0,
            'Review must be identified as assistant review, not human gold')
    byid = {r['id']: r for r in rows}; revised = copy.deepcopy(finals); seen = set()
    bad = {(sid, e['root_id']) for sid, value in finals.items() for e in value['errors']
           if e['status'] == 'confirmed' and (not e['quote'] or byid[sid]['prediction'].count(e['quote']) != 1)}
    for decision in review['quote_decisions']:
        sid, rid = decision['id'], decision['root_id']; key = (sid, rid)
        require(key in bad and key not in seen, 'Unexpected or repeated quote edit')
        seen.add(key); row = byid[sid]
        require(decision['answer_sha256'] == hashlib.sha256(row['prediction'].encode()).hexdigest() and
                decision['original_final_result_sha256'] == final_hashes[sid], 'Review input drift')
        error = next(e for e in revised[sid]['errors'] if e['root_id'] == rid)
        require(error['quote'] == decision['old_quote'] and error['claim'] == decision['claim'] and
                error['status'] == decision['old_status'], 'Reviewed claim changed')
        require(decision['new_quote'] and row['prediction'].count(decision['new_quote']) == 1,
                'Replacement is not a unique original-answer quote')
        error['quote'] = decision['new_quote']
    require(seen == bad, 'Not all invalid confirmed quotes reviewed')
    unresolved = {(sid, e['root_id']) for sid, value in finals.items() for e in value['errors']
                  if e['status'] == 'unresolved'}
    require({(d['id'], d['root_id']) for d in review['physics_decisions']} == unresolved and
            all(d['action'] == 'retain_unresolved' and d['basis'] for d in review['physics_decisions']),
            'Every uncertain claim requires an explicit disposition')
    return revised


def collect(output, plan, rows):
    groups, usage, all_requests = {}, {}, set()
    for arm in base.p.ARMS:
        items = []; provided = reference_tokens = 0; requests = set(); supplemental = []
        for row in rows:
            path = base.path_for(output, 'formal', arm, row)
            if not path.exists(): continue
            item = vr.read(path); base.runtime.audit_result(item)
            require(item['configuration_sha256'] == plan['configuration_sha256'], 'Mixed experiment outputs')
            if not base.runtime.healthy_result(item):
                item = a.bench.recovery.retained_failure(path, output_amended=True)
                if not base.complete_content_failure(item):
                    extra = a.bench.recovery.account(path); supplemental.append(extra); requests.add(extra['request_path'])
            delivery = base.recorded_delivery(path, row, arm, plan, item)
            provided += bool(delivery['provided_complete_ids']); reference_tokens += delivery['reference_tokens']
            requests.update(e['request_path'] for e in item['trace']['response_evidence']); items.append(item)
        groups[arm] = items; all_requests.update(requests)
        usage[arm] = dict(completed_records=len(items), questions_with_full_references=provided,
            reference_tokens=reference_tokens, distinct_actual_requests=len(requests), supplemental_failure_usage=supplemental)
    return groups, usage, all_requests


def score_items(dataset, items, unknown):
    return base.evaluation.score(dataset, [dict(x, execution_failure=not base.runtime.healthy_result(x),
        source_status=x['status'], protocol_noncompliance=not x.get('protocol_audit', {}).get('passed', False))
        for x in items], unknown, preserve_zero_offsets=False)


def score_parity(legacy):
    rows, dataset, unknown = a.bounded.load_bench(a.ROOT)
    require(rows == vr.read(legacy / 'inputs.json'), 'Historical benchmark changed')
    plan = vr.read(legacy / 'plan.json'); old = vr.read(legacy / 'metrics.json')
    groups, _, _ = collect(legacy, plan, rows); values = {}
    for arm in base.p.ARMS:
        require(len(groups[arm]) == 99, 'Incomplete old predictions')
        value = score_items(dataset, groups[arm], unknown)
        expected = old['scores_all_99'][arm]
        require(value == expected, 'Old scoring parity failed: ' + arm)
        values[arm] = value['summary']
    return dict(passed=True, predictions=396, scoring_preserve_zero_offsets=False,
                exact_full_score_objects=True, scores=values, old_metrics_sha256=vr.sha(legacy / 'metrics.json'))


def frozen_development_inputs(path, formal_ids, files):
    require(vr.sha(path) == PILOT_SHA, 'Different fixed development inputs')
    rows = vr.read(path)
    ids = {r['id'] for r in rows}
    require(len(rows) == len(ids) == 6 and ids.isdisjoint(formal_ids)
            and all(r == base.p.source_payload(r) for r in rows), 'Development inputs are not six held-out public rows')
    files[str(path)] = PILOT_SHA
    return rows


def verify_library_first_reviews(build, evidence, inputs):
    rows = {r['id']: r for r in inputs}
    for experience in vr.read(build / 'semantic_experience.json'):
        if experience['semantic_audit']['status'] != 'error_found':
            continue
        sid = experience['sample_id']
        row = rows[sid]
        record_path = build / 'experiences' / (sid + '.json')
        require(vr.sha(record_path) == evidence[str(record_path.resolve())], 'Source experience changed')
        record = vr.read(record_path)
        calls = record['evidence']
        require(record['value'] == experience and 1 <= len(calls) <= 2, 'Different first-review experience')
        lines = [{'line': i, 'text': text} for i, text in enumerate(row['prediction'].splitlines(keepends=True), 1)]
        for stage, call in zip(('extract_experience', 'repair_experience'), calls):
            raw_path = build / call['file']
            require(raw_path.parent.parent.name == stage and vr.sha(raw_path) == call['sha256'] == evidence[str(raw_path)],
                    'Unbound construction request')
            raw = vr.read(raw_path)
            payload = json.loads(raw['request']['messages'][1]['content'])
            require(raw['status'] == 'received' and payload['sample_id'] == sid and payload['question'] == row['question']
                    and payload['prediction_lines'] == lines and payload.get(en.EXAM_REFERENCE) == row['official_reference']['label'],
                    'First-review request did not receive its own official answer')


def released_libraries(parent, files):
    require(parent is not None, 'Four-arm evaluation needs a released library parent')
    pc = vr.read(parent / 'plan.json')
    cfg = pc['configuration']
    require(vr.digest(cfg) == pc['configuration_sha256'], 'Parent input plan changed')
    require(en.active(cfg) and cfg.get('scope') == 'materials_only',
            'The knowledge arms require libraries derived from the same official-answer-guided first review')
    evidence = vr.read(parent / 'source_evidence.json')
    require(vr.digest(evidence) == cfg['evidence_digest'], 'Construction provenance changed')
    build = Path(cfg['root']) / 'construction'
    for name in ('plan.json', 'inputs.json', 'semantic_experience.json'):
        path = (build / name).resolve(strict=True)
        require(vr.sha(path) == evidence[str(path)], 'First-review source changed: ' + name)
        files[str(path)] = evidence[str(path)]
    first = vr.read(build / 'plan.json')
    inputs = vr.read(build / 'inputs.json')
    require(en.active(first['configuration']) and vr.digest(first['configuration']) == first['configuration_sha256']
            and vr.digest(inputs) == first['configuration']['inputs_sha256'] and inputs
            and all(en.exam_answer(r['official_reference']['label']) for r in inputs),
            'Missing official answers in the frozen building inputs')
    verify_library_first_reviews(build, evidence, inputs)
    receipt_path = parent / 'construction_acceptance.json'
    receipt = vr.read(receipt_path)
    paths = {k: parent / 'materials' / (k + '.json') for k in ('cases', 'skills', 'catalog')}
    require(receipt['passed'] and all(receipt['files'][str(p.relative_to(parent))] == vr.sha(p) for p in paths.values())
            and receipt['isolation_sha256'] == vr.sha(parent / 'material_isolation.json'), 'Library receipt drift')
    for path in (parent / 'plan.json', parent / 'source_evidence.json', parent / 'material_isolation.json', receipt_path):
        files[str(path)] = vr.sha(path)
    return paths, 'Common official-answer-guided first review of the construction questions'


def prepare(out, parent, *, prompt_profile=None, benchmark=None, pure_only=False):
    """Read a released material snapshot and the frozen benchmark as data."""
    from scripts import run_step_match_judge as judge
    require(not (out / 'plan.json').exists(), 'Prepared already; never overwrite a release')
    require(prompt_profile in (None, en.PROFILE), 'Unknown prompt profile')
    bench = benchmark or (parent / 'evaluation' if parent and (parent / 'evaluation/inputs.json').exists() else judge.frozen.sources.RESULTS / 'aligned_old99/evaluation')
    rows, coverage = judge.benchmark(bench)
    files = {str(bench / name): sha for name, sha in judge.BENCH_FILES.items()}
    runtime_plan = bench / 'plan.json'
    require(vr.sha(runtime_plan) == 'bd1dfa374044e22c1349853b0709bea8b4a8cfdc587c55dac3d37c06194df23b',
            'Frozen checker configuration changed')
    prior = vr.read(runtime_plan)['configuration']
    files[str(runtime_plan)] = vr.sha(runtime_plan)
    pilot = judge.frozen.sources.RESULTS / 'aligned_old99/final_revision_balanced/pilot_inputs.json'
    controls = frozen_development_inputs(pilot, {r['id'] for r in rows} | set(coverage['held_ids']), files)
    arms = ['pure'] if pure_only else list(base.p.ARMS)
    if pure_only:
        paths, material_source = {}, None
    else:
        paths, material_source = released_libraries(parent, files)
    paths['inputs'] = bench / 'inputs.json'
    released = {}
    for name, path in paths.items():
        files[str(path)] = released[name] = vr.sha(path)
        link = out / (name + '.json')
        if link.is_symlink():
            require(link.resolve(strict=True) == path.resolve(), 'Different release link')
        else:
            link.symlink_to(path.resolve())
    c = {k: prior[k] for k in ('api_base', 'model', 'model_files', 'limits', 'controls', 'arms', 'fixed_harmony_date',
                               'budget_limits', 'query_max_tokens', 'reference_token_cap', 'max_items', 'tree_branches')}
    c.update(protocol=ENGLISH_VERSION, prompt_profile=en.PROFILE, pure_prompt=en.STEP_DIRECT, check_prompt=en.STEP_CHECK,
             prepare_instruction=en.QUERY, prepare_arms=[arm for arm in arms if arm != 'pure'], arms=arms,
             controls=controls, online_reference_visible=False,
             parent=str(parent) if parent else None,
             source_hashes=code_hashes(english=True), dependencies=files, released_files=released,
             step_match_judge=judge.match.VERSION, step_benchmark=str(bench), cohort=coverage,
             sample_ids=[r['id'] for r in rows], planned_checks=len(rows) * len(arms), material_source=material_source,
             formal_released=True, human_gold=False)
    vr.atomic(out / 'plan.json', dict(configuration=c, configuration_sha256=vr.digest(c)))
    return verify(out)


def verify(out):
    plan = vr.read(out / 'plan.json'); c = plan['configuration']
    version = ENGLISH_VERSION if 'step_match_judge' in c else VERSION
    require(c['protocol'] == version and vr.digest(c) == plan['configuration_sha256'], 'Plan drift')
    require(c['source_hashes'] == code_hashes(english=en.active(c)), 'Code drift')
    if 'step_match_judge' in c:
        from rule_framework import step_match
        require(c.get('online_reference_visible') is False and en.online_english(c)
                and c['step_match_judge'] == step_match.VERSION, 'Wrong online inputs or step adjudicator')
    else:
        a.verify(Path(c['parent']))
    for name, sha in c['released_files'].items():
        require(vr.sha(out / (name + '.json')) == sha, 'Released input drift: ' + name)
    vr.EvidenceCache(out / 'evidence_cache.json').check(c['dependencies'])
    return plan


@contextmanager
def reused_controls(out, plan):
    if en.active(plan.get('configuration', {})):
        yield
        return
    original = base.query; prefixes = vr.read(out / 'control_prefixes.json')
    def query(root, row, current, tokenizer, client):
        if row['id'] not in prefixes: return original(root, row, current, tokenizer, client)
        value = copy.deepcopy(prefixes[row['id']])
        require(value['input_sha256'] == vr.digest(row), 'Control prefix input drift')
        value['configuration_sha256'] = plan['configuration_sha256']
        return value
    with patch.object(base, 'query', query): yield


def confirm(out):
    plan = verify(out); ready = vr.read(out / 'disconnect_ready.json')
    require(base.runtime.process_identity(ready['pid']) == ready['process_start'] and
            time.time() - ready['unix'] >= 5 and ready['configuration_sha256'] == plan['configuration_sha256'],
            'Fresh reconnect verification required')
    vr.atomic(out / 'disconnect_acceptance.json', dict(ready, passed=True, confirmed_unix=time.time()))


def score(out, plan=None):
    plan = plan or verify(out)
    if 'step_match_judge' in plan['configuration']:
        from scripts import run_step_match_judge as judge
        data = judge.production(Path(plan['configuration']['step_benchmark']), out)
        with base.r.run_lock(out / 'step_judge/runner.lock'):
            report = judge.evaluate(out / 'step_judge', data, offline=True)
        judge.write(out / 'metrics.json', report)
        return report
    rows = vr.read(out / 'inputs.json')
    dataset, unknown = vr.read(out / 'reference.json'), vr.read(out / 'unresolved.json')
    require([base.p.source_payload(r) for r in dataset] == rows and set(unknown) == {r['id'] for r in rows},
            'Score input/reference mismatch')
    groups, usage, requests = collect(out, plan, rows)
    scores = {arm: score_items(dataset, items, unknown) for arm, items in groups.items() if len(items) == len(rows)}
    report = dict(configuration_sha256=plan['configuration_sha256'], sample_count=len(rows), scores=scores,
        recorded_tasks={arm: len(items) for arm, items in groups.items()}, usage=usage,
        formal_actual_requests_shared_prefix_counted_once=len(requests),
        metric='historical_location_proxy_not_physics_accuracy', scoring_preserve_zero_offsets=False,
        human_gold=False, independent_test=False, label_release_sha256=vr.sha(out / 'label_release.json'),
        comparisons={arm: base.evaluation.bootstrap(s, scores['pure']) for arm, s in scores.items()
                     if arm != 'pure' and 'pure' in scores})
    vr.atomic(out / 'metrics.json', report)
    if len(scores) == 4 and not (out / 'case_review.json').exists():
        selections = []
        for other in ('pure', 'case_rag', 'flat_skills'):
            picked = base.old.protocol.case_selection(scores['rule_tree'], scores[other])
            selections.extend(dict(x, comparison=['rule_tree', other]) for x in picked['selection'])
        byid = {r['id']: r for r in dataset}; results = {arm: {r['id']: r for r in items} for arm, items in groups.items()}
        vr.atomic(out / 'case_review.json', dict(human_gold=False, cases=[dict(x, reference_and_input=byid[x['id']],
            outputs={arm: results[arm][x['id']] for arm in base.p.ARMS}, review_status='pending') for x in selections]))
    return report


def verify_blind_delivery(out, phase, row, arm, memory, tokenizer, result):
    def check(folder, raw_name, expected):
        sent = vr.read(folder / 'request.json')['request']['messages']
        require(sent == expected, 'Unexpected input in the blind verifier request')
        raw = vr.read(folder / raw_name)
        rendered = tokenizer.decode(raw['prompt_token_ids'], skip_special_tokens=False)
        en.verify_visible_users(rendered, sent)

    folder = base.path_for(out, phase, arm, row).parent
    if arm == 'pure':
        expected = base.p.direct_messages(row)
        turn = 1
    else:
        shared = out / 'queries' / vr.digest(row['id'])[:24]
        check(shared, 'raw.json', base.p.query_messages(row, english=True))
        if result.get('failure_stage') == 'prepare':
            return
        queries = vr.read(shared / 'result.json')['queries']
        packet = vr.read(folder / 'reference_packet.json')['packet']
        store = memory.memory if arm == 'rule_tree' else memory
        items = [store.view(sid) for sid in packet['retrieval']['provided_complete_ids']]
        expected = base.p.continuation_messages(row, queries, items, arm=arm, english=True)
        turn = 2
    first_turn = folder / f'turn_{turn:02d}'
    check(first_turn, 'response.sse', expected)
    repair = folder / f'turn_{turn + 1:02d}'
    if (repair / 'request.json').exists():
        diagnosis = vr.read(folder / 'first_diagnosis.json')
        quote_prompt = base.quotes.repair_message(diagnosis, base.quotes.invalid_quotes(diagnosis, row), english=True)
        check(repair, 'response.sse', expected + [
            dict(role='assistant', content=vr.read(first_turn / 'response.json')['content']),
            dict(role='user', content=quote_prompt)])


def check_published(out, row, phase, arm, memory, plan, tokenizer, client):
    from scripts import resume_aligned_step_evaluation as recovery
    shared = (phase == 'formal' and 'step_match_judge' in plan['configuration'] and arm != 'pure')
    folder = out / 'queries' / vr.digest(row['id'])[:24]
    if not shared or not (folder / recovery.FAILURE).exists():
        try:
            return a.bench.run_condition(out, row, phase, arm, memory, plan, tokenizer, client)
        except ValueError:
            if not shared or (folder / 'result.json').exists() or not (folder / 'request.json').exists():
                raise
    receipt = recovery.inspect_preparation(out, row, plan)
    return recovery.record_failure(out, row, plan, receipt)[arm]


def run(out, *, gpu=4):
    require(Path('/proc').exists(), 'Formal execution must run on the server')
    with base.r.run_lock(out / 'active.lock'), ExitStack() as services:
        plan = verify(out); c = plan['configuration']; exit_code = 1
        judged = 'step_match_judge' in c
        (out / 'runner.pid').write_text(str(os.getpid()))
        try:
            base.runtime.wait_detached(out, plan['configuration_sha256'])
            from transformers import AutoTokenizer
            model_path = base.r.EXPECTED_MODEL
            if judged:
                from scripts import run_step_match_judge as judge
                if (out / 'step_judge/metrics.json').exists():
                    report = score(out, plan)
                    base.old.emit(out, 'completed', recorded_tasks=report['recorded_tasks'])
                    exit_code = 0
                    return
                endpoint = services.enter_context(judge.node.service(out, gpu, out / 'STOP', plan['configuration_sha256']))
                require(endpoint == c['api_base'], 'Wrong verifier or judge endpoint')
                model_path = judge.node.host.HOT / 'models/gpt-oss-120b'
            else:
                services.enter_context(base.r.run_lock(Path(c['service_lock'])))
                require(vr.read(out / 'service/startup.json')['fixed_harmony_date'] == a.DATE, 'Wrong service date')
                actual = base.old.server_preflight(out, dict(c, model=base.r.EXPECTED_MODEL))
                require(actual == vr.read(Path(c['legacy']) / 'runtime.json'), 'Historical runtime package/model mismatch')
                base.old.bind_runtime(out / 'runtime.json', actual)
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
            memories = {'pure': None}
            if any(arm != 'pure' for arm in c['arms']):
                memories.update(case_rag=base.p.FlatMemory(vr.read(out / 'cases.json')),
                                flat_skills=base.p.FlatMemory(vr.read(out / 'skills.json')),
                                rule_tree=base.p.TreeMemory(vr.read(out / 'catalog.json')))
            with a.configured(out, tokenizer), reused_controls(out, plan), httpx.Client(trust_env=False) as client:
                for phase, rows in [('controls', c['controls']), ('formal', vr.read(out / 'inputs.json'))]:
                    if phase == 'formal':
                        if judged:
                            rows = [r for r in rows if r['id'] not in c['cohort']['held_ids']]
                            require([r['id'] for r in rows] == c['sample_ids'], 'Eligible cohort changed')
                        accepted = vr.read(out / 'engineering_acceptance.json')
                        require(accepted['passed'] and accepted['configuration_sha256'] == plan['configuration_sha256'],
                                'Released-material controls not accepted')
                    results = []
                    for i, row in enumerate(rows):
                        offset = i % len(c['arms'])
                        arms = c['arms'][offset:] + c['arms'][:offset]
                        if en.active(c):
                            arms = ['pure', *(arm for arm in arms if arm != 'pure')]
                        for arm in arms:
                            require(not (out / 'STOP').exists(), 'STOP requested')
                            result = check_published(out, row, phase, arm, memories[arm], plan, tokenizer, client)
                            if judged:
                                verify_blind_delivery(out, phase, row, arm, memories[arm], tokenizer, result)
                            results.append(result)
                    if phase == 'controls':
                        require(len(results) == len(rows) * len(c['arms']) and all(base.runtime.healthy_result(r) and r['protocol_audit']['passed']
                                for r in results), 'Control content/protocol failure; no bulk release')
                        vr.atomic(out / 'engineering_acceptance.json', dict(configuration_sha256=plan['configuration_sha256'],
                            passed=True, checks=len(results), physical_quality_certified=False, score_threshold_used=False))
            if judged:
                # Close the checker hooks before the independent conversations start.
                data = judge.production(Path(c['step_benchmark']), out)
                with base.r.run_lock(out / 'step_judge/runner.lock'), httpx.Client(trust_env=False) as client:
                    report = judge.evaluate(out / 'step_judge', data, tokenizer=tokenizer, client=client, stop=out / 'STOP')
                judge.write(out / 'metrics.json', report)
            else:
                report = score(out, plan)
            expected = len(c['sample_ids'])
            require(all(n == expected for n in report['recorded_tasks'].values()), 'Incomplete formal cohort')
            services.close()
            base.old.emit(out, 'completed', recorded_tasks=report['recorded_tasks'])
            exit_code = 0
        except BaseException as exc:
            base.old.emit(out, 'stopped', reason=str(exc)); raise
        finally:
            (out / 'runner.exit_code').write_text(str(exit_code))
            (out / 'runner.pid').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify', 'run', 'score', 'confirm-detached'])
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--parent', type=Path)
    parser.add_argument('--prompt-profile', choices=[en.PROFILE], help='Bind new online prompts when preparing from an older input release')
    parser.add_argument('--benchmark', type=Path, help='Frozen step benchmark for a new English release')
    parser.add_argument('--gpu', type=int, choices=(4, 5, 6, 7), default=4)
    parser.add_argument('--pure-only', action='store_true', help='Freeze only the direct 120B baseline')
    args = parser.parse_args(); out = args.output.resolve()
    require(not args.pure_only or args.command == 'prepare', 'Arms are frozen at prepare')
    require(args.prompt_profile is None or args.command == 'prepare', 'Prompt profile is frozen at prepare')
    if args.command == 'prepare':
        if args.parent is None and not args.pure_only:
            parser.error('four-arm prepare requires --parent')
        parent = args.parent.resolve() if args.parent else None
        with base.r.run_lock(out / 'active.lock'):
            prepare(out, parent, prompt_profile=args.prompt_profile, benchmark=args.benchmark.resolve() if args.benchmark else None,
                    pure_only=args.pure_only)
    elif args.command == 'verify': verify(out)
    elif args.command == 'score': score(out)
    elif args.command == 'run': run(out, gpu=args.gpu)
    else: confirm(out)


if __name__ == '__main__': main()
