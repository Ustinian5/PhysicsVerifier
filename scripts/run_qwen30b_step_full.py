"""Frozen 100-question extension of the accepted six-question 30B pilot.

No old answer labels are transferred. Original pilot evidence is reused through
hash-bound derived views; the original files and generation settings stay intact.
All GPU work is server-local and requires the pilot's semantic acceptance.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from unittest.mock import patch

import httpx

from scripts import run_qwen30b_step_verifier as cohort
from scripts import run_qwen30b_step_checks as checks
from scripts import run_organization_followup as followup

base, vr, require = checks.base, checks.vr, checks.require
knowledge, corpus, ROOT = cohort.knowledge, cohort.corpus, cohort.ROOT
VERSION = 'original_source_30b_100_four_arms_v1'
ARMS = ['pure', 'case_rag', 'flat_skills', 'rule_tree']
POOL = 'results/qwen30b_corpus'


class CohortScreen(knowledge.FamilyScreen):
    """Only actual held-out IDs are leakage hits; other sentinels are not IDs."""
    def __init__(self, heldout):
        super().__init__(heldout)
        self.heldout_ids = {r['id'] for r in heldout}

    def matches(self, row):
        return [h for h in super().matches(row) if h['heldout_id'] in self.heldout_ids]


def code_hashes():
    return {**checks.code_hashes(), **followup.source_hashes(),
            'scripts/run_qwen30b_step_full.py': vr.sha(Path(__file__)),
            'scripts/run_qwen30b_step_verifier.py': vr.sha(ROOT/'scripts/run_qwen30b_step_verifier.py'),
            'scripts/rebuild_qwen30b_corpus.py': vr.sha(ROOT/'scripts/rebuild_qwen30b_corpus.py')}


def visible_materials(build):
    """One preparation-time scan, including generated audit and index prose."""
    audits = {x['sample_id']: x for x in vr.read(build/'semantic_experience.json')}
    for row in vr.read(build/'inputs.json'):
        yield dict(row, answer='', prediction=row['prediction']+'\n'+
                   json.dumps(audits.get(row['id'], {}), ensure_ascii=False))
    for domain in vr.read(build/'catalog.json')['domains']:
        yield {'id': 'catalog/'+domain['id'], 'question': '', 'context': '',
               'prediction': json.dumps(domain, ensure_ascii=False), 'answer': ''}


def select_questions(existing, original_pool, materials, old_questions, count=100):
    require(len(existing) < count, 'Extension must increase the frozen cohort')
    key = knowledge.previous.isolation.question_key
    occupied = {key(x['question']) for x in old_questions}
    occupied.update(key(x['question']) for x in materials if x['question'])
    candidates = [dict(id=r['id'], question=r['question'], context='')
                  for r in sorted(original_pool, key=lambda r: vr.digest([20261107, 'extension', r['id']]))
                  if key(r['question']) not in occupied and not re.search(
                      knowledge.previous.FAMILY_POLICY['dependent_question'], r['question'], re.I)]
    # Fixed candidate pool, selected without answers, error labels or model outputs.
    candidates = candidates[:256]
    screen = knowledge.FamilyScreen(existing+candidates)
    rejected = {}
    held_ids = {r['id'] for r in existing+candidates}
    for i, row in enumerate(materials):
        for match in screen.matches(row):
            if match['heldout_id'] in held_ids:
                rejected.setdefault(match['heldout_id'], []).append(
                    dict(source_id=row['id'], **match))
        if i % 300 == 0:
            print('full isolation', i, len(materials), flush=True)
    require(not ({x['id'] for x in existing} & set(rejected)),
            'New visible-material scan invalidates a retained pilot question; stop before generation')
    chosen = list(existing)
    for row in candidates:
        if row['id'] in rejected:
            continue
        hits = knowledge.FamilyScreen(chosen).matches(row)
        reverse = knowledge.FamilyScreen([row])
        hits += [h for other in chosen for h in reverse.matches(other)
                 if h['heldout_id'] == row['id']]
        if hits:
            rejected[row['id']] = hits
            continue
        chosen.append(row)
        if len(chosen) == count:
            break
    require(len(chosen) == count, 'Insufficient isolated source questions; do not relax screening')
    return chosen, {'candidate_ids': [r['id'] for r in candidates], 'rejected': rejected,
                    'retained_old': len(existing), 'added': count-len(existing),
                    'selection_uses_new_answers_or_labels': False,
                    'semantic_family_independence_proven': False}


def reviewed_questions(old_rows, original_pool, old_isolation, review, pilot_ids, count=100):
    """Select from input-only reviews in the already frozen order, never by outputs."""
    require(review['status'] == 'review_complete' and review['human_gold'] is False and
            review['selection_uses_diagnostic_scores'] is False and
            review['formal_answers_generated_at_review'] == 0 and
            review['target_count'] == count and set(review['pilot_only_ids']) == set(pilot_ids),
            'Input review is incomplete or used outcome-based selection')
    entries = review['entries']; pool = {r['id']: r for r in original_pool}
    require([e['id'] for e in entries] == [r['id'] for r in old_rows] and
            len(pool) == len(original_pool), 'Review/source ID mismatch')
    old_ids = {r['id'] for r in old_rows}
    order = [i for i in old_isolation['candidate_ids']
             if i not in old_ids and i not in old_isolation['rejected']]
    extra = review['replacement_reviews']
    require(review['replacement_order'] == order and
            [e['id'] for e in extra] == order[:len(extra)] and
            [e['order'] for e in extra] == list(range(len(extra))), 'Replacement order changed')
    candidates = list(old_rows)+[dict(id=e['id'], question=pool[e['id']]['question'], context='') for e in extra]
    chosen, rejected = [], {}
    for row, entry in zip(candidates, entries+extra):
        require(entry['question_sha256'] == vr.digest(row['question']) and
                entry['decision'] in {'keep','exclude'} and entry.get('reason'), 'Unreviewed/changed question')
        if row['id'] in pilot_ids or entry['decision'] != 'keep':
            continue
        hits = CohortScreen(chosen).matches(row) if chosen else []
        reverse = CohortScreen([row])
        hits += [h for other in chosen for h in reverse.matches(other)]
        if hits:
            rejected[row['id']] = hits
        elif len(chosen) < count:
            chosen.append(row)
    require(len(chosen) == count, 'Insufficient complete isolated questions; continue input review in fixed order')
    return chosen, rejected


def revise_cohort(out):
    """User-authorized pre-call correction; original six-case evidence stays untouched."""
    with base.r.run_lock(out/'active.lock'):
        require((out/'STOP').exists(), 'Hold execution before cohort correction')
        require(not any((out/n).exists() for n in ('generated','raw','checks','annotations','skills')),
                'An executed batch cannot replace questions')
        require(not (out/'cohort_revision.json').exists(), 'Cohort already revised; do not overwrite its receipt')
        previous = vr.read(out/'plan.json'); c = previous['configuration']
        require(vr.digest(c) == previous['configuration_sha256'], 'Previous plan drift')
        for n,h in c['frozen_files'].items(): require(vr.sha(out/n)==h, 'Previous input drift: '+n)
        for n,h in c['source_files'].items(): require(vr.sha(ROOT/n)==h, 'Source drift: '+n)
        review = vr.read(out/'question_review.json')
        require(review['original_questions_sha256'] == vr.sha(out/'questions.json'), 'Review belongs to other cohort')
        rows, pair_rejections = reviewed_questions(vr.read(out/'questions.json'),
            vr.read(ROOT/POOL/'question_pool.json'), vr.read(out/'isolation.json'), review, c['pilot_ids'])
        materials = list(visible_materials(ROOT/cohort.BUILD)); screen = CohortScreen(rows)
        overlaps = []
        for i, row in enumerate(materials):
            hits = screen.matches(row)
            if hits: overlaps.append(dict(source_id=row['id'], matches=hits))
            if i % 300 == 0: print('revised cohort isolation', i, len(materials), flush=True)
        require(not overlaps, 'New cohort overlaps construction material; do not relax isolation')
        refs = {r['id']: r for r in vr.read(ROOT/POOL/'reference_pool.json')}
        refs.update({r['id']:r for r in vr.read(out/'reference_answers.json')})
        isolation = {'heldout_questions':len(rows), 'overlaps':overlaps,
            'heldout_isolation_passed':True, 'visible_material_count':len(materials),
            'library_source_hashes':c['source_files'], 'pair_rejections':pair_rejections,
            'retained_old':sum(r['id'] in set(c['sample_ids']) for r in rows),
            'added':sum(r['id'] not in set(c['sample_ids']) for r in rows),
            'pilot_reused':0, 'selection_uses_new_answers_or_labels':False,
            'semantic_family_independence_proven':False,
            'question_review_sha256':vr.sha(out/'question_review.json')}
        # Preserve the previous uncalled freeze in one provenance receipt, not parallel runs.
        receipt = {'reason':'User approved input-completeness screening and deterministic original-pool replacement; six cases remain pilot only.',
            'model_calls_before_revision':0, 'previous_plan':previous,
            'previous_files':{n:(out/n).read_text() for n in
                ('questions.json','reference_answers.json','isolation.json','bench_questions.json',
                 'skills_plan.json','construction_acceptance.json')},
            'question_review_sha256':vr.sha(out/'question_review.json'), 'created_unix':time.time()}
        vr.atomic(out/'cohort_revision.json',receipt)
        for n,v in [('questions',rows),('bench_questions',rows),
                    ('reference_answers',[refs[r['id']] for r in rows]),('isolation',isolation)]:
            vr.atomic(out/(n+'.json'),v)
        updated = copy.deepcopy(c)
        updated.update(source_hashes=code_hashes(), sample_ids=[r['id'] for r in rows], pilot_ids=[],
            selection='Input-completeness-reviewed original pool; six pilot cases excluded; frozen hash order replenishment')
        updated['frozen_files'] = {n:vr.sha(out/n) for n in c['frozen_files']}
        updated['frozen_files']['question_review.json'] = vr.sha(out/'question_review.json')
        updated['frozen_files']['cohort_revision.json'] = vr.sha(out/'cohort_revision.json')
        vr.atomic(out/'plan.json',{'configuration':updated,'configuration_sha256':vr.digest(updated)})
        sp = vr.read(out/'skills_plan.json'); sp['configuration']['source_plan_sha256']=vr.sha(out/'plan.json')
        sp['configuration_sha256']=vr.digest(sp['configuration']);vr.atomic(out/'skills_plan.json',sp)
        # The former acceptance was bound to different held-out questions, and cannot authorize calls.
        (out/'construction_acceptance.json').unlink()
        return verify(out)


def accept_revised_construction(out, audit_path):
    """Retain the completed construction audit and bind a fresh held-out scan."""
    require(Path('/proc').exists(), 'Construction acceptance is server-only')
    with base.r.run_lock(out/'active.lock'):
        require((out/'STOP').exists(), 'Hold execution before acceptance')
        plan = verify(out); c = plan['configuration']
        receipt = vr.read(out/'cohort_revision.json'); audit = vr.read(audit_path)
        prior = json.loads(receipt['previous_files']['construction_acceptance.json'])
        require(prior['passed'] and prior['configuration_sha256']==receipt['previous_plan']['configuration_sha256']
                and prior['evidence'].get(str(audit_path.resolve()))==vr.sha(audit_path)
                and audit['provenance_passed'] and audit['original_source_only'], 'Prior source audit is not accepted')
        evidence = {str(audit_path.resolve()):vr.sha(audit_path)}
        for n,h in prior['evidence'].items():
            # The old question bytes are retained in the revision, not at the newly frozen path.
            if Path(n).resolve() == (out/'questions.json').resolve(): continue
            require(vr.sha(Path(n))==h, 'Prior construction source changed: '+n)
            evidence[n]=h
        isolation = vr.read(out/'isolation.json')
        require(isolation['heldout_isolation_passed'] and not isolation['overlaps'] and
                isolation['library_source_hashes']==c['source_files'] and
                isolation['question_review_sha256']==vr.sha(out/'question_review.json'), 'New heldout scan differs')
        for n in ('questions.json','isolation.json','question_review.json','cohort_revision.json','catalog.json'):
            evidence[str((out/n).resolve())]=vr.sha(out/n)
        accepted = copy.deepcopy(prior)
        accepted.update(configuration_sha256=plan['configuration_sha256'],
            heldout_questions_sha256=vr.sha(out/'questions.json'), evidence=evidence,
            scope_note='Original-only v5 construction audit retained; revised 100 complete question faces re-screened against all visible construction inputs/audits/catalog. Six cases pilot only.')
        vr.atomic(out/'construction_acceptance.json',accepted)
        return construction_acceptance(out, plan)


def prepare(out, pilot):
    out.mkdir(parents=True, exist_ok=True)
    with base.r.run_lock(out/'active.lock'):
        require(not (out/'plan.json').exists(), 'Already frozen; verify existing plan')
        parent = cohort.verify(pilot)
        pooldir, build = ROOT/POOL, ROOT/cohort.BUILD
        original = vr.read(pooldir/'question_pool.json')
        provenance = vr.read(pooldir/'reference_pool_manifest.json')
        require(provenance['question_pool_sha256'] == vr.sha(pooldir/'question_pool.json') and
                provenance['reference_pool_sha256'] == vr.sha(pooldir/'reference_pool.json') and
                provenance['original_source_sha256'] == 'ff31782395fffce1e367e29cdbd4ac1a68abf7b7ea4d4363541218e40c7c9b8b',
                'Original source/reference provenance differs')
        rows, isolation = select_questions(vr.read(pilot/'questions.json'), original,
            list(visible_materials(build)), vr.read(ROOT/cohort.QUESTIONS))
        refs = {x['id']: x for x in vr.read(pooldir/'reference_pool.json')}
        refs.update({x['id']: x for x in vr.read(pilot/'reference_answers.json')})
        reference = [refs[x['id']] for x in rows]
        source_by_id = {x['id']: x for x in vr.read(build/'inputs.json')}
        outcomes = vr.read(build/'semantic_experience.json')
        cases, sources = [], []
        for outcome in outcomes:
            if outcome['status'] != 'ok' or outcome['semantic_audit']['status'] != 'error_found':
                continue
            source = source_by_id[outcome['sample_id']]
            sources.append({k: source.get(k, '') for k in ('id','question','context','prediction')})
            payload = {'question': source['question'], 'context': source.get('context', ''),
                       'student_solution': source['prediction'], 'review': outcome['semantic_audit']}
            cases.append({'id': 'case_'+source['id'], 'kind': 'case', 'title': source['question'][:160],
                          'text': json.dumps(payload, ensure_ascii=False), 'source_ids': [source['id']]})
        require(cases, 'No error experiences')
        leaf_sources = vr.read(build/'provenance.json')
        catalog_ids = {r['rule_id'] for d in vr.read(build/'catalog.json')['domains']
                       for t in d['topics'] for r in t['rules']}
        require(catalog_ids == {x['rule_id'] for x in leaf_sources} and
                all(set(x['sample_ids']) <= {r['id'] for r in sources} for x in leaf_sources),
                'The three libraries must share the same original error-experience source pool')
        # Provenance is checked separately from indexing: RAG never uses distilled rule leaves.
        for name, value in [('questions', rows), ('reference_answers', reference), ('isolation', isolation),
                            ('sources', sources), ('cases', cases), ('bench_questions', rows),
                            ('catalog', vr.read(build/'catalog.json'))]:
            vr.atomic(out/(name+'.json'), value)
        c = {'protocol': VERSION, 'pilot_relative': os.path.relpath(pilot, out),
             'pilot_plan_sha256': vr.sha(pilot/'plan.json'), 'source_hashes': code_hashes(),
             'generator_config': parent['configuration'],
             'pilot_ids': parent['configuration']['pilot_ids'], 'sample_ids': [r['id'] for r in rows],
             'arms': ARMS, 'source_count': len(sources), 'model': base.r.EXPECTED_MODEL,
             'api_base': 'http://127.0.0.1:18083/v1', 'model_files': base.runtime.MODEL_FILES,
             'limits': {**base.runtime.LIMITS, 'review_protocol': base.old.continuous.PROTOCOL},
             'fixed_harmony_date': '2026-11-07', 'reasoning_effort': 'medium',
             'max_tokens': knowledge.PASS_OUTPUT_TOKENS,
             'check_prompt': base.p.STAGED_BASE, 'skill_prompt': knowledge.SKILL_PROMPT,
             'source_files': {str(p.relative_to(ROOT)): vr.sha(p) for p in [
                 pooldir/'question_pool.json', pooldir/'reference_pool.json', pooldir/'reference_pool_manifest.json',
                 build/'inputs.json', build/'semantic_experience.json', build/'catalog.json', build/'provenance.json']},
             'frozen_files': {n+'.json': vr.sha(out/(n+'.json')) for n in
                 ('questions','reference_answers','isolation','sources','cases','bench_questions','catalog')},
             'human_gold': False, 'independent_test': False,
             'selection': '81 previously frozen development faces plus 19 deterministic original-pool faces',
             'reuse': 'same input, prompt, model, settings and exact raw evidence; no successful output rerolls'}
        vr.atomic(out/'plan.json', {'configuration': c, 'configuration_sha256': vr.digest(c)})
        sc = {'model': {'model': c['model'], 'api_base': c['api_base'], 'model_files': c['model_files'],
                       'protocol': VERSION, 'reasoning_effort': 'medium', 'max_tokens': knowledge.PASS_OUTPUT_TOKENS,
                       'max_request_input_bytes': 400000, 'max_requests': len(sources)*3,
                       'attempts_per_job': 3, 'json_policy': 'literal_invalid_backslashes_only_with_evidence_no_content_repair'},
              'budget_limits': {'requests': len(sources)*3, 'input_tokens': len(sources)*3*300000,
                                'output_tokens': len(sources)*3*knowledge.PASS_OUTPUT_TOKENS},
              'prior_usage': base.old.ZERO_USAGE, 'source_count': len(sources),
              'source_plan_sha256': vr.sha(out/'plan.json'), 'skill_prompt': knowledge.SKILL_PROMPT}
        vr.atomic(out/'skills_plan.json', {'configuration': sc, 'configuration_sha256': vr.digest(sc)})
        base.old.emit(out, 'prepared_no_model_calls', questions=len(rows), planned_checks=len(rows)*len(ARMS))


def verify(out, acceptance=False):
    if acceptance:
        require(not (out/'STOP').exists(), 'Formal execution is held; construction/workflow correction required')
    plan = vr.read(out/'plan.json'); c = plan['configuration']
    require(c['protocol'] == VERSION and vr.digest(c) == plan['configuration_sha256'], 'Full plan drift')
    require(c['source_hashes'] == code_hashes(), 'Full code drift')
    for n, h in c['frozen_files'].items(): require(vr.sha(out/n) == h, 'Full input drift: '+n)
    for n, h in c['source_files'].items(): require(vr.sha(ROOT/n) == h, 'Source drift: '+n)
    pilot = (out/c['pilot_relative']).resolve()
    require(vr.sha(pilot/'plan.json') == c['pilot_plan_sha256'], 'Pilot plan drift')
    if acceptance:
        construction_acceptance(out, plan)
        a = vr.read(pilot/'pilot_acceptance.json')
        require(a.get('passed') is True and a['configuration_sha256'] == vr.read(pilot/'plan.json')['configuration_sha256']
                and a.get('semantic_review_passed') is True, 'Six-question semantic acceptance required')
        for n, h in a['evidence'].items(): require(vr.sha(pilot/n) == h, 'Acceptance evidence drift')
        if a.get('pilot_primary_metrics_ready') is False:
            require(a.get('scope') == 'engineering_and_semantic_workflow_only' and not c['pilot_ids'] and
                    not (set(c['sample_ids']) & set(a['pilot_only_ids'])) and
                    a.get('approved_full_configuration_sha256') == plan['configuration_sha256'] and
                    'question_review.json' in c['frozen_files'],
                    'Uncertain pilot cannot release metrics or be reused as formal cases')
    return plan


def construction_acceptance(out, plan):
    """A completed benchmark pilot cannot approve a different library's source."""
    path = out/'construction_acceptance.json'
    require(path.exists(), 'Corrected original-source construction acceptance is required')
    a = vr.read(path)
    require(a.get('passed') is True and a.get('configuration_sha256') == plan['configuration_sha256'] and
            a.get('original_source_only') is True and a.get('answer_model') == corpus.MODEL and
            a.get('construction_model') == base.r.EXPECTED_MODEL and a.get('embedding_calls') == 0 and
            a.get('heldout_questions_sha256') == vr.sha(out/'questions.json') and
            a.get('catalog_sha256') == vr.sha(out/'catalog.json') and
            a.get('source_count') == plan['configuration']['source_count'] and
            a.get('scope') == 'generation_and_library_provenance_and_heldout_isolation' and
            a.get('human_gold') is False and bool(a.get('evidence')),
            'Construction acceptance missing or belongs to different inputs/library')
    for n, h in a['evidence'].items():
        evidence = Path(n)
        require(evidence.is_absolute() and vr.sha(evidence) == h, 'Construction evidence drift')
    return a


def accept_construction(out, audit_path):
    """Bind the completed source audit before any formal calls, not physical certification."""
    require(Path('/proc').exists(), 'Construction acceptance is server-only')
    with base.r.run_lock(out/'active.lock'):
        require((out/'STOP').exists(), 'Freeze formal execution before preflight revision')
        old = vr.read(out/'plan.json'); c = old['configuration']
        require(vr.digest(c) == old['configuration_sha256'], 'Previous plan changed')
        require(not any((out/n).exists() for n in ('generated','raw','checks','annotations','skills'))
                and not list(out.glob('arm*/*/request_*.json')), 'Cannot revise an executed batch')
        require(not (out/'preflight_revision.json').exists(), 'Preflight already revised')
        for n,h in c['frozen_files'].items(): require(vr.sha(out/n)==h, 'Frozen input changed')
        for n,h in c['source_files'].items(): require(vr.sha(ROOT/n)==h, 'Source changed')
        audit = vr.read(audit_path)
        require(audit['state']=='audit_complete' and audit['provenance_passed'] is True
                and audit['heldout_isolation_passed'] is True and audit['original_source_only'] is True
                and audit['model_calls']==0 and not audit['overlaps']
                and audit['error_experiences']==c['source_count'], 'Construction provenance audit failed')
        for n,h in audit['source_files'].items(): require(vr.sha(Path(n))==h, 'Audit source changed')
        require(audit['source_files'].get(str((out/'questions.json').resolve()))==vr.sha(out/'questions.json'),
                'Construction audit used a different held-out cohort')
        build = ROOT/cohort.BUILD
        require(audit['source_files'].get(str((build/'catalog.json').resolve())) == vr.sha(build/'catalog.json')
                and vr.read(build/'catalog.json') == vr.read(out/'catalog.json'),
                'Construction audit used a different catalog')
        require(audit['generation_binding']==vr.read(build/'source_release.json'), 'Generation binding changed')
        # Only engineering code bindings change. No scientific inputs, prompts, budgets, or sources change.
        new = copy.deepcopy(old); new['configuration']['source_hashes'] = code_hashes()
        new['configuration_sha256'] = vr.digest(new['configuration'])
        vr.atomic(out/'preflight_revision.json', {'previous_plan':old,
            'previous_skills_plan':vr.read(out/'skills_plan.json'),
            'new_configuration_sha256':new['configuration_sha256'], 'model_calls':0,
            'reason':'Bind source acceptance and corrected heldout-ID screening before formal execution.'})
        vr.atomic(out/'plan.json', new)
        sp = vr.read(out/'skills_plan.json'); sp['configuration']['source_plan_sha256']=vr.sha(out/'plan.json')
        sp['configuration_sha256']=vr.digest(sp['configuration']); vr.atomic(out/'skills_plan.json', sp)
        vr.atomic(out/'construction_acceptance.json', {
            'passed':True,'configuration_sha256':new['configuration_sha256'], 'original_source_only':True,
            'answer_model':corpus.MODEL,'construction_model':base.r.EXPECTED_MODEL,'embedding_calls':0,
            'heldout_questions_sha256':vr.sha(out/'questions.json'),'catalog_sha256':vr.sha(out/'catalog.json'),
            'source_count':c['source_count'],'scope':'generation_and_library_provenance_and_heldout_isolation',
            'human_gold':False,'physical_quality_certified':False,
            'verified_answer_count':audit['verified_complete_30b_answers'],'rule_leaves':audit['rule_leaves'],
            'evidence':{str(audit_path.resolve()):vr.sha(audit_path), **audit['source_files']},
            'scope_note':'Reuse original-only v5 evidence; exclude external-textbook 5000 expansion. Automatic content errors remain measured outcomes.'})
        return verify(out)


def pilot_root(out, plan):
    return (out/plan['configuration']['pilot_relative']).resolve()


def generated(out, plan, row):
    c = plan['configuration']; pilot = pilot_root(out, plan)
    source = pilot if row['id'] in c['pilot_ids'] else out
    record = vr.read(source/'generated'/(row['id']+'.json'))
    corpus.evidence_valid(source, record)
    require(record['signature'] == corpus.p.digest([cohort.solver_row(row), c['generator_config']]) and
            record['model'] == c['generator_config']['generator'], '30B generation provenance differs')
    return record


def export(out, plan):
    refs = {x['id']: x for x in vr.read(out/'reference_answers.json')}
    inputs, labeled, failures = [], [], []
    for row in vr.read(out/'questions.json'):
        record = generated(out, plan, row)
        if record['state'] != 'complete':
            failures.append({'id': row['id'], 'state': record['state']}); continue
        item = dict(row, prediction=record['prediction']); inputs.append(item)
        labeled.append(dict(item, answer=refs[row['id']]['historical_reference_answer']))
    vr.atomic(out/'generation_summary.json', {'complete': len(inputs), 'failures': failures,
        'reused_pilot': len(plan['configuration']['pilot_ids']), 'no_replacement_sampling': True})
    require(not failures and len(inputs) == 100, 'Incomplete answers retained; no silent cohort reduction')
    vr.atomic(out/'inputs.json', inputs); vr.atomic(out/'annotation_inputs.json', labeled)
    return inputs


def confirm(out):
    plan = verify(out, acceptance=True); ready = vr.read(out/'disconnect_ready.json')
    require(base.runtime.process_identity(ready['pid']) == ready['process_start'] and
            time.time()-ready['unix'] >= 5 and ready['configuration_sha256'] == plan['configuration_sha256'],
            'Fresh SSH confirmation required')
    vr.atomic(out/'disconnect_acceptance.json', dict(ready, passed=True, confirmed_unix=time.time()))


def generate(out):
    require(Path('/proc').exists(), '30B generation is server-only')
    with base.r.run_lock(out/'active.lock'), base.r.run_lock(out.parent/'qwen30b_step_generation.lock'):
        plan = verify(out, acceptance=True); c = plan['configuration']; code = 1
        (out/'runner.pid').write_text(str(os.getpid()))
        try:
            base.runtime.wait_detached(out, plan['configuration_sha256'])
            with httpx.Client(trust_env=False, timeout=10) as client:
                response = client.get(c['generator_config']['generator_api']+'/models'); response.raise_for_status()
                require(any(x['id'] == c['generator_config']['generator'] and x.get('max_model_len',0) >= 32768 and
                            x.get('root') == '/slow_share2/huhengyuan/models/Qwen3-30B-A3B-Instruct-2507'
                            for x in response.json()['data']), '30B service differs')
            journal = corpus.Journal(out, c['generator_config'], threading.Event())
            for row in vr.read(out/'questions.json'):
                if row['id'] in c['pilot_ids']:
                    generated(out, plan, row)
                else:
                    corpus.generate(journal, cohort.solver_row(row))
                base.old.emit(out, 'generating', sample_id=row['id'])
            export(out, plan); base.old.emit(out, '100_answers_ready'); code = 0
        finally:
            (out/'runner.exit_code').write_text(str(code)); (out/'runner.pid').unlink(missing_ok=True)


def clone_evidence(path, expected, old, new):
    """Same bytes, hash-bound link inside the new run, never alter originals."""
    path = Path(path)
    require(path.is_relative_to(old) and vr.sha(path) == expected, 'Pilot evidence changed or escaped')
    target = new/'pilot_reuse'/path.relative_to(old)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try: os.link(path, target)
        except OSError: shutil.copyfile(path, target)
    require(vr.sha(target) == expected, 'Reused evidence differs')
    return str(target)


def import_result(source_path, old, new, old_binding, new_binding):
    result = vr.read(source_path); base.runtime.audit_result(result)
    require(result['configuration_sha256'] == old_binding, 'Wrong pilot diagnosis binding')
    derived = copy.deepcopy(result)
    for e in derived['trace']['response_evidence']:
        for k in ('request', 'response', 'raw_sse'):
            e[k+'_path'] = clone_evidence(e[k+'_path'], e[k+'_sha256'], old, new)
    if derived.get('first_diagnosis_path'):
        derived['first_diagnosis_path'] = clone_evidence(derived['first_diagnosis_path'],
            derived['first_diagnosis_sha256'], old, new)
    derived['configuration_sha256'] = new_binding
    derived['reuse_origin'] = {'path': str(source_path), 'sha256': vr.sha(source_path),
                              'configuration_sha256': old_binding, 'new_model_calls': 0}
    base.runtime.audit_result(derived)
    return derived


def prepare_checks(out, plan):
    target = out/'checks'; c = plan['configuration']; pilot = pilot_root(out, plan)/'checks_pilot'
    rows = vr.read(out/'inputs.json'); prior = checks.verify(pilot); pc = prior['configuration']
    require(c['check_prompt'] == pc['check_prompt'] and c['limits'] == pc['limits'] and
            c['model'] == pc['model'] and c['fixed_harmony_date'] == pc['fixed_harmony_date'] and
            vr.sha(out/'catalog.json') == vr.sha(pilot/'catalog.json'), 'Pilot/new checking settings differ')
    pilot_rows = {r['id']: r for r in vr.read(pilot/'inputs.json')}
    require(all(row == pilot_rows[row['id']] for row in rows if row['id'] in pilot_rows), 'Pilot input changed')
    cc = {k: copy.deepcopy(c[k]) for k in ('arms','model','model_files','api_base','limits',
          'fixed_harmony_date','check_prompt','source_hashes','human_gold','independent_test')}
    cc.update(protocol=VERSION, inputs_sha256=vr.sha(out/'inputs.json'),
              full_plan_sha256=vr.sha(out/'plan.json'), catalog_sha256=vr.sha(out/'catalog.json'),
              sample_ids=[r['id'] for r in rows], reasoning_effort='medium',
              query_max_tokens=base.p.QUERY_MAX_TOKENS, reference_token_cap=base.p.REFERENCE_TOKEN_CAP)
    checkplan = {'configuration': cc, 'configuration_sha256': vr.digest(cc)}
    target.mkdir(exist_ok=True)
    if (target/'plan.json').exists():
        require(vr.read(target/'plan.json') == checkplan, 'Checker plan drift')
    else:
        vr.atomic(target/'plan.json', checkplan); vr.atomic(target/'inputs.json', rows)
    for row in rows:
        if row['id'] not in pilot_rows: continue
        for arm in pc['arms']:
            source = base.path_for(pilot, 'formal', arm, row)
            dest = base.path_for(target, 'formal', arm, row)
            derived = import_result(source, pilot, target, prior['configuration_sha256'], checkplan['configuration_sha256'])
            if dest.exists(): require(vr.read(dest) == derived, 'Imported diagnosis changed')
            else: vr.atomic(dest, derived)
            packet = source.parent/'reference_packet.json'
            clone_evidence(packet, vr.sha(packet), pilot, target)
        receipt = pilot/'queries'/vr.digest(row['id'])[:24]/'result.json'
        prefix = copy.deepcopy(vr.read(receipt))
        require(prefix['input_sha256'] == vr.digest(row) and
                prefix['configuration_sha256'] == prior['configuration_sha256'], 'Pilot query drift')
        prefix['files'] = {clone_evidence(p, h, pilot, target): h for p, h in prefix['files'].items()}
        for k in ('request','response','raw_sse'):
            e = prefix['evidence']; e[k+'_path'] = clone_evidence(e[k+'_path'], e[k+'_sha256'], pilot, target)
        request = vr.read(Path(prefix['evidence']['request_path']))['request']
        require(request['messages'] == base.p.query_messages(row) and request['model'] == c['model'] and
                request['max_tokens'] == base.p.QUERY_MAX_TOKENS, 'Reused query does not match exact prompt')
        prefix['configuration_sha256'] = checkplan['configuration_sha256']
        dest = target/'queries'/vr.digest(row['id'])[:24]/'result.json'
        if dest.exists(): require(vr.read(dest) == prefix, 'Imported query changed')
        else: vr.atomic(dest, prefix)
    return checkplan


def skills(out, plan, tokenizer):
    sp = vr.read(out/'skills_plan.json'); sc = sp['configuration']
    require(vr.digest(sc) == sp['configuration_sha256'] and
            sc['source_plan_sha256'] == vr.sha(out/'plan.json') and sc['skill_prompt'] == knowledge.SKILL_PROMPT,
            'Skills plan drift')
    # Reuse the incremental journal; no historical 1,000-skill quota or semantic zero-error gate.
    with patch.object(followup.bench, 'protocol', knowledge), \
         patch.object(followup.bench, 'ProcedureJournal', followup.BoundedSkillJournal), \
         patch.object(followup.bench, 'check_pilot', followup.check_pilot), \
         patch.object(knowledge, 'FamilyScreen', CohortScreen), \
         patch.object(base.p, 'PILOT_SIZE', knowledge.PILOT_SIZE):
        return followup.ORIGINAL_BUILD(out, sp, tokenizer)


def audit_checks(out, checkplan, tokenizer):
    target = out/'checks'; rows = vr.read(target/'inputs.json'); c = checkplan['configuration']
    result_files, raw_files, counts = {}, {}, {}
    for arm in c['arms']:
        count = {'recorded': 0, 'completed': 0, 'diagnoses': 0, 'reused': 0}
        for row in rows:
            path = base.path_for(target, 'formal', arm, row); result = vr.read(path)
            base.runtime.audit_result(result)
            require(result['configuration_sha256'] == checkplan['configuration_sha256'], 'Wrong result batch')
            result_files[str(path)] = vr.sha(path)
            count['recorded'] += 1; count['completed'] += base.runtime.healthy_result(result)
            count['diagnoses'] += len(result['diagnostics']); count['reused'] += 'reuse_origin' in result
            for e in result.get('trace', {}).get('response_evidence', []):
                for k in ('request','response','raw_sse'):
                    raw_files[e[k+'_path']] = e[k+'_sha256']
                checks.validate_effective_input(vr.read(Path(e['raw_sse_path'])), tokenizer, c['fixed_harmony_date'])
        counts[arm] = count
    vr.atomic(target/'engineering_audit.json', {'configuration_sha256': checkplan['configuration_sha256'],
        'execution_integrity_passed': True, 'result_files': result_files, 'raw_files': raw_files,
        'counts': counts, 'physical_quality_passed': False, 'human_gold': False})


def run_checks(out):
    require(Path('/proc').exists(), 'Checks are server-only')
    with base.r.run_lock(out/'active.lock'), base.r.run_lock(out.parent/'organization_service_18083.lock'):
        plan = verify(out, acceptance=True); c = plan['configuration']; code = 1
        require(len(vr.read(out/'inputs.json')) == 100, 'Need complete fixed 100-answer cohort')
        checkplan = prepare_checks(out, plan); target = out/'checks'
        require(vr.read(out.parent/'oss_service/startup.json')['fixed_harmony_date'] == c['fixed_harmony_date'],
                'Fixed-date service missing')
        base.old.bind_runtime(out/'runtime.json', base.old.server_preflight(out, c))
        (out/'runner.pid').write_text(str(os.getpid()))
        try:
            base.runtime.wait_detached(out, plan['configuration_sha256'])
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(c['model'], local_files_only=True, trust_remote_code=False)
            original_decoder = base.old.b.harmony.decode_response
            def decode(raw, **kwargs):
                checks.validate_effective_input(raw, tokenizer, c['fixed_harmony_date'])
                return base.decode_response(raw, **kwargs)
            def skill_decode(raw, **kwargs):
                checks.validate_effective_input(raw, tokenizer, c['fixed_harmony_date'])
                return original_decoder(raw, **kwargs)
            with base.old.comparison.execution.configured(), \
                 patch.object(base.runtime.transport, 'decode_response', decode), \
                 patch.object(base.old.b.harmony, 'decode_response', skill_decode), \
                 patch.object(followup.bench.recovery, 'inspect_failure', followup.native.inspect_failure), \
                 httpx.Client(trust_env=False) as client:
                memories = {'pure': None, 'rule_tree': base.p.TreeMemory(vr.read(out/'catalog.json')),
                            'case_rag': base.p.FlatMemory(vr.read(out/'cases.json')),
                            'flat_skills': skills(out, plan, tokenizer)}
                for phase, rows in [('controls', base.runtime.CONTROLS[:2]), ('formal', vr.read(out/'inputs.json'))]:
                    for i, row in enumerate(rows):
                        for arm in ARMS[i % 4:]+ARMS[:i % 4]:
                            require(not (out/'STOP').exists(), 'STOP requested')
                            followup.bench.run_condition(target, row, phase, arm, memories[arm], checkplan, tokenizer, client)
                audit_checks(out, checkplan, tokenizer)
            base.old.emit(out, '400_checks_complete_pending_independent_scoring'); code = 0
        finally:
            (out/'runner.exit_code').write_text(str(code)); (out/'runner.pid').unlink(missing_ok=True)


def prepare_annotations(out, env):
    """Reuse six exact four-pass jobs; all new tasks keep the original Sol protocol."""
    from dotenv import load_dotenv
    from scripts import run_bench_reannotation as teacher
    load_dotenv(env, override=False)
    plan = verify(out, acceptance=True); target = out/'annotations'
    prior = pilot_root(out, plan)/'annotation_pilot'
    rows, manifest = teacher.prepare(out/'annotation_inputs.json', target, os.environ['OPENAI_BASE_URL'],
                                    expected_count=100, pilot_size=100)
    oldrows = {r['id']: r for r in vr.read(prior/'inputs.json')}
    for row in rows:
        if row['id'] not in oldrows: continue
        require(row == oldrows[row['id']], 'Reused teacher task has different answer/reference')
        for stage in teacher.STAGES:
            folder = target/('arm_'+stage)/row['id']; source = prior/('arm_'+stage)/row['id']
            teacher.validate_saved(prior, stage, row, teacher.previous_annotations(prior, stage, row['id']))
            folder.parent.mkdir(parents=True, exist_ok=True)
            if folder.exists(): require(folder.resolve() == source.resolve(), 'Teacher reuse path differs')
            else: folder.symlink_to(os.path.relpath(source, folder.parent), target_is_directory=True)
    audit = teacher.audit(target, rows, manifest)
    vr.atomic(target/'reuse_receipt.json', {'pilot_manifest_sha256': vr.sha(prior/'manifest.json'),
        'pilot_ids': sorted({r['id'] for r in rows} & set(oldrows)),
        'reused_complete_tasks': sum(audit['completed'].values()),
        'new_model_calls': 0, 'prior_failed_attempts_and_unknown_usage_retained': True,
        'mode_note': 'legacy runner pilot_size=100 executes all 100 questions; separate six-question pilot evidence is retained'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare','verify','generate','export','confirm-detached','checks','annotations',
                                     'accept-construction','revise-cohort','accept-revised-construction'])
    p.add_argument('--output', type=Path, required=True); p.add_argument('--pilot', type=Path)
    p.add_argument('--env', type=Path, default=ROOT/'.env')
    p.add_argument('--audit', type=Path)
    a = p.parse_args(); out = a.output.resolve()
    if a.command == 'prepare': prepare(out, a.pilot.resolve())
    elif a.command == 'verify': verify(out)
    elif a.command == 'generate': generate(out)
    elif a.command == 'export': export(out, verify(out))
    elif a.command == 'checks': run_checks(out)
    elif a.command == 'annotations': prepare_annotations(out, a.env)
    elif a.command == 'accept-construction': accept_construction(out, a.audit.resolve())
    elif a.command == 'revise-cohort': revise_cohort(out)
    elif a.command == 'accept-revised-construction': accept_revised_construction(out, a.audit.resolve())
    else: confirm(out)


if __name__ == '__main__': main()
