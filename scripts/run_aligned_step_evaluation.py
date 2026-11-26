"""Evaluate frozen 30B solutions with the step-level verifier and location scorer.

The English profile uses a direct 120B baseline and one shared preparation
for RAG, skills and the rule tree. The saved plan selects the checking protocol.
"""
import argparse
from contextlib import contextmanager
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
from scripts import run_bench_reannotation as teacher
from scripts.run_qwen30b_step_checks import reference_rows

base, vr, require = a.base, a.vr, a.require
VERSION = 'corrected100_oldv15_released_v1'
KNOWN_SOURCE = '62a07c4b8297c063fcb852b972e21db38f50ed9137326ff13b104f085544bdf0'


def code_hashes(*, english=False):
    names = set(a.code_hashes()) | {'scripts/run_aligned_step_evaluation.py',
        'scripts/run_bench_reannotation.py', 'scripts/bench_reannotation_runtime.py'}
    if english:
        names.add('rule_framework/verifier_english.py')
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


def release_labels(out, annotations, rows, files):
    labeled = vr.read(annotations / 'annotation_inputs.json')
    require([base.p.source_payload(r) for r in labeled] == rows, 'Labels concern different answers')
    review = vr.read(out / 'label_review.json')
    require(review['input_sha256'] == vr.sha(annotations / 'annotation_inputs.json'), 'Review cohort drift')
    for name in ('annotation_inputs.json', 'manifest.json'):
        files[str(annotations / name)] = vr.sha(annotations / name)
    manifest = vr.read(annotations / 'manifest.json')
    require(manifest['model'] == teacher.MODEL and manifest['reasoning_effort'] == 'high', 'Wrong teacher')
    finals, final_hashes = {}, {}
    for stage in teacher.STAGES:
        for row in labeled:
            saved = teacher.validate_saved(annotations, stage, row,
                       teacher.previous_annotations(annotations, stage, row['id']))
            path = annotations / ('arm_' + stage) / row['id'] / 'result.json'
            files[str(path)] = vr.sha(path)
            raw = annotations / saved['response_record']
            files[str(raw)] = saved['response_record_sha256']
            if stage == 'final':
                finals[row['id']] = saved['annotation']; final_hashes[row['id']] = vr.sha(path)
    revised = apply_quote_review(rows, finals, review, final_hashes)
    dataset, unknown, counts = reference_rows(rows, revised)
    require(counts == dict(confirmed=180, unresolved=2, confirmed_unlocated=0), 'Unexpected reference inventory')
    for name, value in [('reference', dataset), ('unresolved', unknown), ('reviewed_annotations', revised)]:
        vr.atomic(out / (name + '.json'), value)
    vr.atomic(out / 'label_release.json', dict(version=VERSION, human_gold=False,
        reviewer='Codex assistant', model=teacher.MODEL, reasoning_effort='high',
        counts=counts, original_reference_counts=dict(confirmed=180, uniquely_locatable=163, unresolved=2),
        quote_repairs=len(review['quote_decisions']), claims_or_statuses_changed=0,
        new_annotation_calls=0, original_stage_results=4 * len(rows),
        uncertain_scopes=review['uncertain_scope_disposition'], source_links=review['source_links'],
        all_physics_or_statements_certified=False, full_case_physics_reannotation=False,
        unresolved_policy='Retain separately, outside confirmed recall denominator, as in old v15.',
        scoring_preserve_zero_offsets=False))


def release_materials(out, parent, current, audit, files):
    """Apply the old conservative filter and source propagation without generation."""
    heldout = []
    for kind, name in [('step', 'step_inputs'), ('pilot', 'pilot_inputs'), ('final', 'final_inputs')]:
        path = current / (name + '.json'); files[str(path)] = vr.sha(path)
        heldout.extend(dict(id=kind + ':' + r['id'], question=r['question']) for r in vr.read(path))
    extpath = parent.parent / 'hipho_overlap_audit.json'; ext = vr.read(extpath)
    files[str(extpath)] = vr.sha(extpath)
    original = Path(ext['benchmark_path'])
    require(vr.sha(original) == ext['benchmark_sha256'], 'External heldout changed')
    files[str(original)] = vr.sha(original)
    for row in ext['question_references']:
        heldout.append(dict(id='hipho:' + row['id'], question=row['question']))
        if row.get('context'):
            heldout.append(dict(id='hipho_context:' + row['id'], question=row['context'] + '\n' + row['question']))
    screen = a.bounded.FamilyScreen(heldout)
    sources = vr.read(parent / 'sources.json'); cases = vr.read(parent / 'materials/cases.json')
    byid = {d['source_ids'][0]: d for d in cases}; excluded = []
    for row in sources:
        matches = screen.matches(dict(row, prediction=byid[row['id']]['text']))
        if matches: excluded.append(dict(source_id=row['id'], matches=matches))
    require(excluded == audit['isolation']['lexical_source_candidates'], 'Screen differs from inspected evidence')
    base.old.emit(out, 'preparing_source_release', screened_sources=len(sources), screen_candidates=len(excluded))
    known = audit['isolation']['known_family_exposure']
    require(known['build_source'] == KNOWN_SOURCE and known['source_retained'], 'Known family evidence changed')
    excluded_ids = {x['source_id'] for x in excluded} | {KNOWN_SOURCE}
    kept = [r for r in sources if r['id'] not in excluded_ids]
    kept_cases = [d for d in cases if not excluded_ids.intersection(d['source_ids'])]
    outcomes = []
    parent_binding = vr.read(parent / 'plan.json')['configuration_sha256']
    for row in kept:
        path = base.old.outcome_path(parent / 'skills', row); saved = vr.read(path)
        require(saved['source_sha256'] == vr.digest(row) and saved['case_sha256'] == vr.digest(byid[row['id']])
                and saved['configuration_sha256'] == parent_binding, 'Saved skill binding drift')
        files[str(path)] = vr.sha(path); outcomes.append(saved['outcome'])
    skills, summary = a.bounded.materialize_skills(kept, outcomes)
    origin = Path(vr.read(current / 'plan.json')['configuration']['root'])
    bundlepath = origin / 'pipeline/tree_bundle.json'; files[str(bundlepath)] = vr.sha(bundlepath)
    catalog, changes = a.bounded.previous.prune_catalog(vr.read(current / 'materials/catalog.json'),
                            vr.read(bundlepath)['provenance'], excluded_ids)
    memory = base.p.TreeMemory(catalog)
    # Identical to old bench99.build: append all methods to the source answer,
    # then screen once per source. Case-note screening happened above.
    require(1000 <= len(skills) <= 2 * len(kept), 'Old bounded-skill release count gate')
    by_source = {r['id']: dict(r) for r in kept}
    for doc in skills:
        require(not set(doc['source_ids']) & excluded_ids, 'Excluded skill provenance retained')
        for sid in doc['source_ids']:
            by_source[sid]['prediction'] += '\n' + doc['title'] + '\n' + doc['text']
    _, derived_excluded = screen.partition(list(by_source.values()))
    require(not derived_excluded, 'Generated skill has a new family candidate; inspect before release')
    for name, value in [('sources', kept), ('cases', kept_cases), ('skills', skills), ('catalog', catalog),
                        ('skill_summary', summary), ('bench_questions', heldout)]:
        vr.atomic(out / (name + '.json'), value)
    vr.atomic(out / 'isolation.json', dict(policy=a.bounded.ISOLATION_POLICY,
        original_sources=len(sources), kept_sources=len(kept), excluded=excluded,
        known_family_exclusion=known, excluded_source_ids=sorted(excluded_ids), tree_changes=changes,
        material_counts=dict(rag=len(kept_cases), skills=len(skills), tree=len(memory.corpus.leaves)),
        new_model_calls=0, human_gold=False, semantic_family_independence_proven=False,
        content_failures_preserved=True, scope='Original source filter plus confirmed same-family exclusion; no source replacement.'))


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


def prepare(out, parent, *, prompt_profile=None):
    require(not (out / 'plan.json').exists(), 'Prepared already; never overwrite a release')
    plan = a.verify(parent); pc = plan['configuration']; current = Path(pc['current'])
    audit = vr.read(parent / 'readiness_audit.json')
    require(audit['configuration_sha256'] == plan['configuration_sha256'] and
            audit['materials_and_recorded_execution_verified'], 'Parent materials not audited')
    # Recheck the frozen audit inventory once at release, not once per request.
    files = dict(audit['files'])
    vr.EvidenceCache(out / 'evidence_cache.json').check(files, full=True)
    base.old.emit(out, 'preparing_release', audited_files_verified=len(files), new_model_calls=0)
    files[str(parent / 'plan.json')] = vr.sha(parent / 'plan.json')
    files[str(parent / 'readiness_audit.json')] = vr.sha(parent / 'readiness_audit.json')
    rows = vr.read(current / 'step_inputs.json')
    require(len(rows) == 100 and len({r['id'] for r in rows}) == 100, 'Not the frozen 100 questions')
    vr.atomic(out / 'inputs.json', rows)
    release_materials(out, parent, current, audit, files)
    base.old.emit(out, 'preparing_label_release', new_model_calls=0)
    origin = Path(vr.read(current / 'plan.json')['configuration']['root'])
    release_labels(out, origin / 'annotations', rows, files)
    base.old.emit(out, 'checking_historical_score_parity', new_model_calls=0)
    vr.atomic(out / 'historical_scoring_parity.json', score_parity(Path(pc['legacy'])))
    english = en.active(pc if prompt_profile is None else {'prompt_profile': prompt_profile})
    prefixes = {}
    if not english:
        for row in pc['controls']:
            value, _ = a.followup.frozen_prefix(parent, row, plan['configuration_sha256'], files)
            prefixes[row['id']] = value
    vr.atomic(out / 'control_prefixes.json', prefixes)
    names = ('inputs', 'sources', 'cases', 'skills', 'catalog', 'skill_summary', 'bench_questions',
             'isolation', 'reference', 'unresolved', 'reviewed_annotations', 'label_review',
             'label_release', 'historical_scoring_parity', 'control_prefixes')
    c = copy.deepcopy(pc)
    c.update(protocol=VERSION, parent=str(parent), parent_configuration_sha256=plan['configuration_sha256'],
        source_hashes=code_hashes(english=english), dependencies=files,
        released_files={name: vr.sha(out / (name + '.json')) for name in names},
        formal_blockers=[], formal_released=True, sample_ids=[r['id'] for r in rows],
        source_count=len(vr.read(out / 'sources.json')), planned_checks=400,
        service_lock=str(parent.parent / 'organization_service_18083.lock'),
        controls_policy=('Pure checks directly. RAG, skills and rule tree share one English PREPARE per control.' if english else
                         'Reuse exact successful PREPARE; 20 fresh CHECKs against released libraries, then formal.'),
        current_materials_policy='Reuse parent generation; apply original v15 source filter and provenance pruning.',
        label_policy='Four Sol high passes plus assistant quote/ambiguity release; two unresolved remain separate.',
        final_answer_task='Not part of this release; no 600-case generation or rescoring.')
    if english:
        c.pop('shared_query', None)
        c.update(prompt_profile=en.PROFILE, pure_prompt=en.STEP_DIRECT, check_prompt=en.STEP_CHECK, prepare_instruction=en.QUERY,
                 prepare_arms=base.p.prepared_arms(english=True),
                 query_policy='Share and charge one PREPARE per input across RAG, skills and rule tree only.')
    vr.atomic(out / 'plan.json', dict(configuration=c, configuration_sha256=vr.digest(c)))
    base.old.emit(out, 'released_pending_engineering_acceptance', formal_checks_completed=0,
                  new_model_calls=0, materials=vr.read(out / 'isolation.json')['material_counts'])
    return verify(out)


def verify(out):
    plan = vr.read(out / 'plan.json'); c = plan['configuration']
    require(c['protocol'] == VERSION and vr.digest(c) == plan['configuration_sha256'], 'Plan drift')
    require(c['source_hashes'] == code_hashes(english=en.active(c)), 'Code drift')
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
    plan = plan or verify(out); rows = vr.read(out / 'inputs.json')
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


def run(out):
    require(Path('/proc').exists(), 'Formal execution must run on the server')
    c = vr.read(out / 'plan.json')['configuration']
    with base.r.run_lock(out / 'active.lock'), base.r.run_lock(Path(c['service_lock'])):
        plan = verify(out); c = plan['configuration']; exit_code = 1
        require(vr.read(out / 'service/startup.json')['fixed_harmony_date'] == a.DATE, 'Wrong service date')
        actual = base.old.server_preflight(out, dict(c, model=base.r.EXPECTED_MODEL))
        require(actual == vr.read(Path(c['legacy']) / 'runtime.json'), 'Historical runtime package/model mismatch')
        base.old.bind_runtime(out / 'runtime.json', actual)
        (out / 'runner.pid').write_text(str(os.getpid()))
        try:
            base.runtime.wait_detached(out, plan['configuration_sha256'])
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(base.r.EXPECTED_MODEL, local_files_only=True, trust_remote_code=False)
            memories = dict(pure=None, case_rag=base.p.FlatMemory(vr.read(out / 'cases.json')),
                flat_skills=base.p.FlatMemory(vr.read(out / 'skills.json')),
                rule_tree=base.p.TreeMemory(vr.read(out / 'catalog.json')))
            with a.configured(out, tokenizer), reused_controls(out, plan), httpx.Client(trust_env=False) as client:
                for phase, rows in [('controls', c['controls']), ('formal', vr.read(out / 'inputs.json'))]:
                    if phase == 'formal':
                        accepted = vr.read(out / 'engineering_acceptance.json')
                        require(accepted['passed'] and accepted['configuration_sha256'] == plan['configuration_sha256'],
                                'Released-material controls not accepted')
                    results = []
                    for i, row in enumerate(rows):
                        arms = c['arms'][i % 4:] + c['arms'][:i % 4]
                        if en.active(c):
                            arms = ['pure', *(arm for arm in arms if arm != 'pure')]
                        for arm in arms:
                            require(not (out / 'STOP').exists(), 'STOP requested')
                            results.append(a.bench.run_condition(out, row, phase, arm, memories[arm], plan, tokenizer, client))
                    if phase == 'controls':
                        require(len(results) == 20 and all(base.runtime.healthy_result(r) and r['protocol_audit']['passed']
                                for r in results), 'Control content/protocol failure; no bulk release')
                        vr.atomic(out / 'engineering_acceptance.json', dict(configuration_sha256=plan['configuration_sha256'],
                            passed=True, checks=20, physical_quality_certified=False, score_threshold_used=False))
            report = score(out, plan)
            require(all(n == 100 for n in report['recorded_tasks'].values()), 'Incomplete formal cohort')
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
    args = parser.parse_args(); out = args.output.resolve()
    require(args.prompt_profile is None or args.command == 'prepare', 'Prompt profile is frozen at prepare')
    if args.command == 'prepare':
        parent = args.parent.resolve()
        with base.r.run_lock(out / 'active.lock'), base.r.run_lock(parent.parent / 'organization_service_18083.lock'):
            prepare(out, parent, prompt_profile=args.prompt_profile)
    elif args.command == 'verify': verify(out)
    elif args.command == 'score': score(out)
    elif args.command == 'run': run(out)
    else: confirm(out)


if __name__ == '__main__': main()
