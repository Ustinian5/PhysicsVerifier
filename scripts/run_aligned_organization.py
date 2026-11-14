"""Restore the recorded old99 protocol while keeping corrected 30B data.

This adapter imports the old PREPARE/retrieval/CHECK engine and its prompts.
It never resumes the obsolete full coordinator or edits completed experiments.
prepare/verify are offline; pilot/build require detached server execution.
Full evaluation remains gated on the benchmark and source release, separately
from successful engineering checks.
"""
import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import types
from unittest.mock import patch

import httpx

from scripts import run_organization_comparison as base
from scripts import run_organization_bench99 as bench
from scripts import run_organization_followup as followup
from scripts import resume_organization_bench99 as native
from scripts import resume_continuous_context_v2 as recovery
from scripts.run_qwen30b_step_checks import validate_effective_input
from rule_framework import organization_followup as bounded

vr, require, ROOT = base.vr, base.require, base.ROOT
VERSION = 'corrected_data_old99_protocol_v1'
DATE = '2026-11-03'
TRANSPORT_COMMIT = '1285c5c'
TRANSPORT_SHA = '84c14b5d6dfc9f6e8c265e64a9ab476d705897aecf3321acde2002d6c4b85566'
PILOT_IDS = ('cl_202_125689', 'cl_212_135843', 'cl_196_119740', 'cl_174_97192')
CORE = ('rule_framework/flat_skill_rag.py', 'rule_framework/organization_followup.py',
        'rule_framework/bounded_tree.py', 'scripts/run_organization_comparison.py',
        'scripts/continuous_context_v2.py', 'scripts/continuous_context_v3.py',
        'scripts/run_gpt_oss_selfcheck.py', 'scripts/run_continuous_context_v2.py',
        'scripts/resume_continuous_context_v2.py', 'scripts/resume_continuous_context_v3.py',
        'scripts/run_experience_first_120b.py')


def historical_transport():
    """Load the hash-verified Git version in memory, without a source backup."""
    name = 'scripts/gpt_oss_harmony_transport.py'
    source = subprocess.check_output(['git', 'show', TRANSPORT_COMMIT + ':' + name], cwd=ROOT)
    require(hashlib.sha256(source).hexdigest() == TRANSPORT_SHA, 'Historical decoder drift')
    module = types.ModuleType('physicsverifier_historical_transport')
    exec(compile(source, TRANSPORT_COMMIT + ':' + name, 'exec'), module.__dict__)
    return module


def code_hashes():
    names = set(followup.SOURCE_FILES) | set(CORE) | {
        'scripts/run_aligned_organization.py', 'scripts/run_organization_bench99.py',
        'scripts/resume_organization_bench99.py', 'scripts/serve_fixed_date_vllm.py',
        'scripts/run_qwen30b_step_checks.py', 'scripts/start_oss120b_step_service.sh'}
    return {n: vr.sha(ROOT / n) for n in sorted(names)}


def assert_legacy(legacy):
    plan = vr.read(legacy / 'plan.json'); c = plan['configuration']
    require(vr.digest(c) == plan['configuration_sha256'], 'Legacy plan drift')
    require(c['protocol'] == bounded.VERSION and c['check_prompt'] == base.p.STAGED_BASE,
            'Use the actual completed old99 v15 protocol')
    # Only reporting/scoring extensions and the decoder have changed. The
    # decoder is pinned below; scoring keeps the historical default explicitly.
    extensions = {'scripts/analyze_organization_comparison.py', 'scripts/evaluate_physics_eval_sets.py',
                  'scripts/rescore_frozen_bench.py', 'scripts/gpt_oss_harmony_transport.py'}
    for name, sha in c['source_hashes'].items():
        if name not in extensions:
            require(sha == vr.sha(ROOT / name), 'Legacy core changed: ' + name)
    require(c['skill_prompt'] == bounded.SKILL_PROMPT and c['max_skills_per_source'] == 2,
            'Bounded skill prompt changed')
    for key, actual in [('query_max_tokens', base.p.QUERY_MAX_TOKENS),
                        ('reference_token_cap', base.p.REFERENCE_TOKEN_CAP),
                        ('max_items', base.p.MAX_ITEMS), ('tree_branches', base.p.TREE_BRANCHES)]:
        require(c[key] == actual, 'Legacy parameter changed: ' + key)
    historical = historical_transport()
    source = subprocess.check_output(['git', 'show', TRANSPORT_COMMIT + ':scripts/gpt_oss_harmony_transport.py'], cwd=ROOT).decode()
    def chat_ast(text):
        return ast.dump(next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'chat_response'))
    require(chat_ast(source) == chat_ast((ROOT / 'scripts/gpt_oss_harmony_transport.py').read_text()),
            'Actual request transport changed')
    return plan


def replay_request_contract(legacy):
    """Reconstruct all recorded first CHECK messages; no model calls."""
    rows = {x['id']: x for x in vr.read(legacy / 'inputs.json')}
    checked, files = 0, {}
    for arm in base.p.ARMS:
        for row in rows.values():
            path = base.path_for(legacy, 'formal', arm, row).parent
            request = vr.read(path / 'turn_02/request.json')['request']
            packet = vr.read(path / 'reference_packet.json')['packet']
            queries = json.loads(request['messages'][2]['content'])
            # Evidence JSON sorts keys on disk. FlatMemory.view constructs this
            # exact insertion order in the model-visible nested JSON string.
            refs = [{k: d[k] for k in ('id', 'title', 'text')} for d in packet['references']]
            expected = base.p.continuation_messages(row, queries, refs)
            require(request['messages'] == expected, 'Historical first CHECK message mismatch')
            require(request['model'] == base.r.EXPECTED_MODEL and request['temperature'] == 0 and
                    request['top_p'] == 1 and request['reasoning_effort'] == 'medium' and
                    not request.get('tools'), 'Historical sampling/tool contract mismatch')
            for p in [path / 'turn_02/request.json', path / 'reference_packet.json']:
                files[str(p)] = vr.sha(p)
            checked += 1
    require(checked == 396, 'Incomplete historical four-arm request inventory')
    return dict(checked=checked, all_messages_exact=True, files_sha256=vr.digest(files))


def prepare(out, legacy, current):
    require(not (out / 'plan.json').exists(), 'Prepared already; never overwrite a frozen plan')
    prior = assert_legacy(legacy); pc = prior['configuration']
    parent = vr.read(current / 'plan.json'); cp = parent['configuration']
    require(vr.digest(cp) == parent['configuration_sha256'], 'Corrected parent drift')
    root = Path(cp['root']); construction = root / 'construction'
    cases = vr.read(current / 'materials/cases.json')
    selected = {sid for case in cases for sid in case['source_ids']}
    sources = [base.p.source_payload(x) for x in vr.read(construction / 'inputs.json') if x['id'] in selected]
    require(len(sources) == len(cases) == len(selected) == 756, 'Corrected error-source pool drift')
    sources.sort(key=lambda x: vr.digest([bounded.previous.SEED, x['id']]))
    rows = vr.read(current / 'step_inputs.json')
    require(len(rows) == 100 and all(set(r) == {'id', 'question', 'context', 'prediction'} for r in rows),
            'Need the unchanged 100 new 30B inputs with no labels')
    pilots = [r for r in vr.read(current / 'pilot_inputs.json') if r['id'] in PILOT_IDS]
    require({r['id'] for r in pilots} == set(PILOT_IDS), 'Complete-condition pilots missing')
    paths = [legacy / 'plan.json', legacy / 'runtime.json', current / 'plan.json',
             current / 'step_inputs.json', current / 'step_reference.json',
             current / 'step_unresolved.json', current / 'step_label_release.json',
             current / 'materials/catalog.json', current / 'materials/cases.json',
             construction / 'plan.json', construction / 'inputs.json']
    cfg = dict(protocol=VERSION, legacy=str(legacy), current=str(current), model=copy.deepcopy(pc['model']),
        model_files=pc['model_files'], api_base=pc['api_base'], limits=pc['limits'], arms=list(base.p.ARMS),
        fixed_harmony_date=DATE, check_prompt=pc['check_prompt'], skill_prompt=pc['skill_prompt'],
        case_prompt=base.p.BUILD_PROMPT, source_count=len(sources), sample_count=100,
        query_max_tokens=pc['query_max_tokens'], reference_token_cap=pc['reference_token_cap'],
        max_items=pc['max_items'], tree_branches=pc['tree_branches'],
        controls=pc['controls'], budget_limits=pc['budget_limits'], prior_usage=base.old.ZERO_USAGE,
        source_hashes=code_hashes(), dependencies={str(p): vr.sha(p) for p in paths},
        scoring_preserve_zero_offsets=False, human_gold=False,
        query_policy='Generate once for each new 30B input; replay the exact prefix in all four arms; charge its usage in each arm.',
        current_materials_policy='Reuse the old-method tree; regenerate original-case notes and bounded skills with the original prompts.',
        formal_blockers=['New reference has not received the old quote/ambiguity release review.',
                         'Confirmed source-family conflict and incomplete statements remain unresolved; no clean-main-result claim.'],
        final_answer_task='No historical final-answer task in old99; do not silently adapt or launch the 600-case task.')
    for name, value in [('sources', sources), ('pilot_inputs', pilots)]:
        vr.atomic(out / (name + '.json'), value)
        cfg['dependencies'][str(out / (name + '.json'))] = vr.sha(out / (name + '.json'))
    vr.atomic(out / 'historical_request_parity.json', replay_request_contract(legacy))
    cfg['dependencies'][str(out / 'historical_request_parity.json')] = vr.sha(out / 'historical_request_parity.json')
    vr.atomic(out / 'plan.json', dict(configuration=cfg, configuration_sha256=vr.digest(cfg)))
    base.old.emit(out, 'prepared', formal_released=False, new_model_calls=0)
    return verify(out)


def verify(out):
    plan = vr.read(out / 'plan.json'); c = plan['configuration']
    require(c['protocol'] == VERSION and vr.digest(c) == plan['configuration_sha256'], 'Plan drift')
    require(c['source_hashes'] == code_hashes(), 'Source drift')
    for name, sha in c['dependencies'].items():
        require(vr.sha(Path(name)) == sha, 'Frozen dependency drift: ' + name)
    assert_legacy(Path(c['legacy']))
    return plan


@contextmanager
def configured(out, tokenizer):
    original = historical_transport()
    def pinned_decode(raw, **kwargs):
        ev = validate_effective_input(raw, tokenizer, DATE)
        vr.atomic(out / 'effective_inputs' / (ev['sha256'] + '.json'), ev)
        return original.decode_response(raw, **kwargs)
    with base.old.comparison.execution.configured(), \
         patch.object(recovery, 'ORIGINAL_DECODE', pinned_decode), \
         patch.object(base.runtime.transport, 'decode_response', base.decode_response), \
         patch.object(bench.recovery, 'inspect_failure', native.inspect_failure):
        yield


def confirm(out):
    plan = verify(out); ready = vr.read(out / 'disconnect_ready.json')
    require(base.runtime.process_identity(ready['pid']) == ready['process_start'] and
            time.time() - ready['unix'] >= 5 and ready['configuration_sha256'] == plan['configuration_sha256'],
            'Fresh reconnect verification required')
    vr.atomic(out / 'disconnect_acceptance.json', dict(ready, passed=True, confirmed_unix=time.time()))


def build_cases(out, plan, tokenizer, *, pilot):
    """Same original case-note call and renderer; preserve the new source pool."""
    c = plan['configuration']; rows = vr.read(out / 'sources.json')
    if pilot: rows = rows[:bounded.PILOT_SIZE]
    folder = out / 'experience'; folder.mkdir(exist_ok=True)
    journal = base.ExperienceJournal(folder, c, tokenizer); outcomes = []
    vr.EvidenceCache(folder / 'evidence_cache.json').check(journal.index['files'])
    for i, row in enumerate(rows):
        require(not (out / 'STOP').exists(), 'STOP requested')
        path = base.old.outcome_path(folder, row)
        if path.exists():
            saved = vr.read(path)
            require(saved['source_sha256'] == vr.digest(row) and saved['configuration_sha256'] == plan['configuration_sha256'], 'Case drift')
            outcome = saved['outcome']
        else:
            try:
                value = journal('extract_skill', row['id'], {'row': row})
                try: base.p.validate_experience(value)
                except (ValueError, TypeError, KeyError) as exc: raise base.old.tree.ContentFailure(str(exc)) from exc
                outcome = dict(id=row['id'], status='collected', value=value)
            except base.old.tree.ContentFailure as exc:
                outcome = dict(id=row['id'], status='content_failure', reason=str(exc))
            saved = dict(configuration_sha256=plan['configuration_sha256'], source_sha256=vr.digest(row), outcome=outcome)
            vr.atomic(path, saved)
        journal.index['files'][str(path)] = vr.sha(path)
        vr.seal(folder / 'request_index.json', journal.index)
        outcomes.append(dict(outcome, configuration_sha256=plan['configuration_sha256'], source_sha256=vr.digest(row)))
        base.old.emit(out, 'restoring_case_notes', processed=i + 1, planned=len(rows), pilot_only=pilot)
    products = base.p.materialize(rows, outcomes, plan['configuration_sha256'])
    if pilot:
        require(all(x['status'] == 'collected' for x in outcomes), 'Original case-note pilot schema gate failed; no resampling')
    target = out / ('pilot_materials' if pilot else 'materials')
    vr.atomic(target / 'cases.json', products['cases'])
    vr.atomic(target / 'case_summary.json', products['summary'])
    return products['cases']


def build_skills(out, plan, tokenizer, cases, *, pilot):
    folder = out / 'skills'; folder.mkdir(exist_ok=True)
    rows = vr.read(out / 'sources.json')
    case_by_source = {x['source_ids'][0]: x for x in cases}
    rows = [r for r in rows if r['id'] in case_by_source]
    class Journal(bench.ProcedureJournal):
        def messages(self, stage, payload):
            require(stage == 'extract_skill', 'Unexpected bounded-skill stage')
            return bounded.skill_messages(payload['case'])
    journal = Journal(folder, plan['configuration'], tokenizer); outcomes = []
    vr.EvidenceCache(folder / 'evidence_cache.json').check(journal.index['files'])
    for i, row in enumerate(rows):
        case = case_by_source[row['id']]
        require(not (out / 'STOP').exists(), 'STOP requested')
        path = base.old.outcome_path(folder, row)
        if path.exists():
            saved = vr.read(path)
            require(saved['source_sha256'] == vr.digest(row) and saved['case_sha256'] == vr.digest(case) and
                    saved['configuration_sha256'] == plan['configuration_sha256'], 'Skill source drift')
            outcome = saved['outcome']
        else:
            try:
                value = journal('extract_skill', row['id'], {'case': case})
                try: bounded.validate_skills(value)
                except (ValueError, TypeError, KeyError) as exc: raise base.old.tree.ContentFailure(str(exc)) from exc
                outcome = dict(id=row['id'], status='collected', value=value)
            except base.old.tree.ContentFailure as exc:
                outcome = dict(id=row['id'], status='content_failure', reason=str(exc))
            vr.atomic(path, dict(configuration_sha256=plan['configuration_sha256'], source_sha256=vr.digest(row),
                                case_sha256=vr.digest(case), outcome=outcome))
        journal.index['files'][str(path)] = vr.sha(path)
        vr.seal(folder / 'request_index.json', journal.index)
        outcomes.append(outcome)
        base.old.emit(out, 'restoring_bounded_skills', processed=i + 1, planned=len(cases), pilot_only=pilot)
    docs, summary = bounded.materialize_skills(rows, outcomes)
    followup.check_pilot(out, rows, plan['configuration_sha256'])
    target = out / ('pilot_materials' if pilot else 'materials')
    vr.atomic(target / 'skills.json', docs); vr.atomic(target / 'skill_summary.json', summary)
    return docs


def run(out, mode):
    require(mode in {'pilot', 'build'}, 'Formal evaluation is not released by this alignment adapter')
    with base.r.run_lock(out / 'active.lock'), base.r.run_lock(out.parent / 'organization_service_18083.lock'):
        plan = verify(out); c = plan['configuration']; exit_code = 1
        ready = vr.read(out / 'service/startup.json')
        require(ready['fixed_harmony_date'] == DATE, 'Wrong service date')
        actual = base.old.server_preflight(out, dict(c, model=base.r.EXPECTED_MODEL))
        require(actual == vr.read(Path(c['legacy']) / 'runtime.json'), 'Historical runtime package/model mismatch')
        base.old.bind_runtime(out / 'runtime.json', actual)
        (out / 'runner.pid').write_text(str(os.getpid()))
        try:
            base.runtime.wait_detached(out, plan['configuration_sha256'])
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(base.r.EXPECTED_MODEL, local_files_only=True, trust_remote_code=False)
            with configured(out, tokenizer), httpx.Client(trust_env=False) as client:
                if mode == 'build':
                    acceptance = vr.read(out / 'pilot_acceptance.json')
                    require(acceptance['passed'] and acceptance['configuration_sha256'] == plan['configuration_sha256'], 'Pilot not accepted')
                cases = build_cases(out, plan, tokenizer, pilot=mode == 'pilot')
                docs = build_skills(out, plan, tokenizer, cases, pilot=mode == 'pilot')
                require(bool(docs), 'No usable bounded skills; preserve failures')
                if mode == 'pilot':
                    memories = {'pure': None, 'rule_tree': base.p.TreeMemory(vr.read(Path(c['current']) / 'materials/catalog.json')),
                        'case_rag': base.p.FlatMemory(cases), 'flat_skills': base.p.FlatMemory(docs)}
                    results = []
                    for phase, rows in [('controls', c['controls']), ('pilot', vr.read(out / 'pilot_inputs.json'))]:
                        for i, row in enumerate(rows):
                            arms = c['arms'][i % 4:] + c['arms'][:i % 4]
                            for arm in arms:
                                require(not (out / 'STOP').exists(), 'STOP requested')
                                result = bench.run_condition(out, row, phase, arm, memories[arm], plan, tokenizer, client)
                                results.append(result)
                    require(all(base.runtime.healthy_result(r) and r['protocol_audit']['passed'] for r in results),
                            'Pilot has recorded protocol/content failures; no automatic bulk release')
                    vr.atomic(out / 'pilot_acceptance.json', dict(configuration_sha256=plan['configuration_sha256'], passed=True,
                        checks=len(results), source_pilot=len(cases), physical_quality_certified=False,
                        new_case_notes=len(cases), new_skill_items=len(docs),
                        scope='Old message chain, actual input date/model, complete retrieval packets, schema and evidence. No score threshold.'))
                else:
                    require(1000 <= len(docs) <= 2 * len(cases), 'Original bounded-skill count gate not met')
                base.old.emit(out, mode + '_complete', formal_released=False, blockers=c['formal_blockers'])
                exit_code = 0
        except BaseException as exc:
            base.old.emit(out, 'stopped', mode=mode, error_type=type(exc).__name__, error=str(exc)[:1200]); raise
        finally:
            (out / 'runner.exit_code').write_text(str(exit_code)); (out / 'runner.pid').unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'verify', 'pilot', 'build', 'confirm-detached'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--legacy', type=Path); p.add_argument('--current', type=Path)
    a = p.parse_args(); out = a.output.resolve()
    if a.mode == 'prepare': prepare(out, a.legacy.resolve(), a.current.resolve())
    elif a.mode == 'verify': print(verify(out)['configuration_sha256'])
    elif a.mode == 'confirm-detached': confirm(out)
    else: run(out, a.mode)


if __name__ == '__main__': main()
