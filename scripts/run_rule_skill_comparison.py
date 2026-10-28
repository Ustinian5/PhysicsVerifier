"""Build skills from the same 30B wrong answers and compare with the frozen tree.

All inference is explicit, server-local and detached. prepare/verify/score are
offline. Existing libraries, prompts and quality flags are never rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import httpx

from rule_framework import rule_skill_comparison as protocol
from rule_framework import bounded_tree as tree, verified_resume as vr
from scripts import reorganize_qwen30b_expansion as navigation
from scripts import run_built_catalog as comparison
from scripts import continuous_context_v3 as continuous
from scripts import rescore_frozen_bench as evaluation

b, p, r = navigation.b, navigation.p, navigation.r
runtime = comparison.runner
ROOT = b.ROOT
SOURCE_QUESTIONS = 4995
SKILL_SOURCES = 1726
TREE_RELEASE_SHA = 'f9be4a0e591a577432789ce69c2a0b86a2e08fac7e987e18252d9348dc5a92f4'
TREE_CONFIG_SHA = '8b0a22606f6de0c8269611e015fcd5c3abdd000cf127703400c377c9ec204a5c'
SOURCES = sorted(set(navigation.source_hashes()) | set(runtime.SOURCE_FILES) | {
    'rule_framework/rule_skill_comparison.py', 'scripts/run_rule_skill_comparison.py',
    'scripts/run_built_catalog.py', 'scripts/continuous_context_v3.py',
    'scripts/run_continuous_context_v3.py', 'scripts/resume_continuous_context_v3.py',
    'scripts/rescore_frozen_bench.py', 'scripts/evaluate_direct_baseline_comparison.py'})
ZERO_USAGE = dict(requests_started=0, accounted_input_tokens=0, accounted_output_tokens=0,
                  requests_without_usage=0, in_flight=0)


def source_hashes():
    return {name: vr.sha(ROOT / name) for name in SOURCES}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rules_asset(path, *, full=False):
    """Explicit experimental use of this audited, noisy release, not promotion."""
    path = Path(path).resolve()
    release = vr.read(path / 'release_manifest.json')
    require(vr.sha(path / 'release_manifest.json') == TREE_RELEASE_SHA and
            release['configuration_sha256'] == TREE_CONFIG_SHA, 'Unexpected rule-tree release')
    require(release['collection_only'] is True and release['human_gold'] is False and
            release['physical_quality_passed'] is False and release['usable_for_runtime'] is False,
            'Original quality/status flags changed')
    plan = navigation.verify(path, full=full)
    counts = release['counts']
    require(counts['input_candidates'] == 2408 and counts['placed_rules'] == 2065 and
            counts['unplaced_rules'] == 343 and vr.read(path / 'validation.json')['valid'],
            'Frozen rule counts/structure mismatch')
    require(plan['configuration']['model']['model'] == r.EXPECTED_MODEL and
            plan['configuration']['model']['reasoning_effort'] == 'medium', 'Unexpected construction model')
    return {'path': str(path), 'release_sha256': TREE_RELEASE_SHA,
            'configuration_sha256': TREE_CONFIG_SHA, 'catalog_sha256': vr.sha(path / 'catalog.json'),
            'counts': counts, 'permission': 'experimental_comparison_of_unmodified_noisy_library_only',
            'physical_quality_passed': False, 'human_gold': False}


def prepare(output, rule_tree):
    output.mkdir(parents=True, exist_ok=True)
    with r.run_lock(output / 'active.lock'):
        if (output / 'plan.json').exists():
            verify(output)
            print('Already prepared; zero model calls.')
            return
        asset = rules_asset(rule_tree, full=True)
        original = vr.read(rule_tree / 'plan.json')['configuration']
        source = Path(original['legacy_root']) / 'build/inputs.json'
        release = vr.read(rule_tree / 'release_manifest.json')
        require(release['files'].get(str(source)) == vr.sha(source), 'Unbound construction input')
        rows = vr.read(source)
        receipt = vr.read(rule_tree / 'isolation_receipt.json')
        require(len(rows) == SOURCE_QUESTIONS and p.digest(rows) == receipt['inputs_sha256'] and not receipt['excluded'],
                'Construction cohort/isolation mismatch')
        material_path = rule_tree / 'materials.json'
        require(release['files'].get(str(material_path)) == vr.sha(material_path), 'Unbound source provenance')
        materials = vr.read(material_path)
        sources = protocol.wrong_answer_sources(rows, materials)
        require(len(sources) == SKILL_SOURCES, 'Wrong-answer source count changed')
        original_taxonomy = materials['taxonomy']
        taxonomy = {'domains': [dict({k: d[k] for k in ('id', 'name', 'summary')},
            topics=[{k: t[k] for k in ('id', 'name', 'summary')} for t in d['topics']])
            for d in original_taxonomy['domains']]}
        require(len(tree.topic_list(taxonomy)) == 123, 'Fixed taxonomy changed')
        # The inherited isolation covers every visible question/context field.
        # New generated content is screened once again before publication.
        vr.atomic(output / 'sources.json', sources)
        vr.atomic(output / 'taxonomy.json', taxonomy)
        model = dict(original['model'], protocol=protocol.VERSION, max_requests=60000)
        config = {'protocol': protocol.VERSION, 'rules': asset, 'source_input': str(source),
            'source_input_sha256': vr.sha(source), 'sources_sha256': vr.sha(output / 'sources.json'),
            'source_materials': str(material_path), 'source_materials_sha256': vr.sha(material_path),
            'source_questions': len(sources), 'parent_complete_questions': len(rows),
            'source_selection': 'all 120B-error-found 30B answers before navigation; exact original question/answer/context; no old rule or audit text',
            'taxonomy_sha256': vr.sha(output / 'taxonomy.json'), 'model': model,
            'bench_sha256': original['bench_sha256'], 'benchmarks': original['benchmarks'],
            'source_hashes': source_hashes(), 'distill': protocol.DISTILL,
            'navigation_policy': tree.POLICY, 'navigation_prompts_sha256': vr.digest(tree.PROMPTS),
            'pilot_ids': [x['id'] for x in sources[:protocol.PILOT_SIZE]],
            'budget_limits': {'requests': 60000, 'input_tokens': 24000000000, 'output_tokens': 983040000},
            'prior_usage': ZERO_USAGE, 'semantic_content_retries': 0,
            'human_gold': False, 'independent_test': False,
            'construction_compute': 'same 30B wrong answers; 120B distills rules or skills; report actual tokens/calls, not equal construction compute',
            'hypothesis': 'compare rule and skill construction from identical 30B wrong answers, all semantic stages use 120B; no guaranteed winner'}
        vr.atomic(output / 'plan.json', {'configuration': config, 'configuration_sha256': vr.digest(config)})
        vr.atomic(output / 'execution_status.json', {'state': 'prepared_no_model_calls',
            'source_questions': len(sources), 'parent_complete_questions': len(rows)})
    print('Prepared 1726 complete 30B wrong answers; 12-source pilot precedes skill construction; zero model calls.')


def verify(output, *, full=False):
    plan = vr.read(output / 'plan.json')
    c = plan['configuration']
    require(plan['configuration_sha256'] == vr.digest(c) and c['protocol'] == protocol.VERSION and
            c['source_hashes'] == source_hashes(), 'Plan/code binding changed')
    for name, expected in [('sources.json', c['sources_sha256']), ('taxonomy.json', c['taxonomy_sha256'])]:
        require(vr.sha(output / name) == expected, 'Prepared input changed: ' + name)
    require(vr.sha(c['source_input']) == c['source_input_sha256'], 'Parent construction input changed')
    require(vr.sha(c['source_materials']) == c['source_materials_sha256'], 'Source provenance changed')
    require(rules_asset(Path(c['rules']['path']), full=full) == c['rules'], 'Rule-tree handoff changed')
    if (output / 'skills/request_index.json').exists():
        index = vr.unseal(output / 'skills/request_index.json')
        require(index['binding'] == plan['configuration_sha256'], 'Skill request ledger binding changed')
        vr.EvidenceCache(output / 'skills/evidence_cache.json').check(index['files'], full=full)
    return plan


class SkillJournal(navigation.TreeJournal):
    def supported_stages(self):
        return super().supported_stages() | {'extract_skill'}

    def messages(self, stage, payload):
        return (protocol.skill_messages(stage, payload) if stage == 'extract_skill'
                else super().messages(stage, payload))

    def decode_record(self, path, stage):
        if stage != 'extract_skill':
            return super().decode_record(path, stage)
        raw = vr.read(path)
        response = raw.get('response') or {}
        require(not raw.get('error') and raw.get('http_status') == 200 and
                response.get('model') == self.config['model'] and response.get('id'),
                'Transport/model identity failed; stop, never an empty skill')
        choices = response.get('choices')
        require(isinstance(choices, list) and len(choices) == 1, 'Invalid response envelope')
        if choices[0].get('finish_reason') == 'length':
            raise tree.ContentFailure('length_truncated; no resampling')
        require(choices[0].get('finish_reason') == 'stop', 'Unknown finish reason')
        text = b.harmony.decode_response(response, model=self.config['model'],
                    max_tokens=self.config['max_tokens'], native_tools=None)['content'].strip()
        if not text:
            raise tree.ContentFailure('empty_final; no resampling')
        return super().decode_record(path, stage)

    def __call__(self, stage, key, payload):
        cap = self.config['max_request_input_bytes'] if stage == 'extract_skill' else tree.POLICY['max_input_bytes']
        if len(json.dumps(self.messages(stage, payload), ensure_ascii=False).encode()) > cap:
            raise tree.ContentFailure('input_too_large; no truncation')
        try:
            # The inherited journal replays exact completed requests. Technical
            # failures stop; a later explicit resume observes the same cooldown
            # and three-attempt cap. Invalid content is never sampled again.
            return b.Journal.__call__(self, stage, key, payload)
        finally:
            self.record_job(stage, key)


def collect_one(journal, row):
    try:
        value = journal('extract_skill', row['id'], {'row': row})
        try:
            protocol.validate_skills(value)
        except (ValueError, TypeError, KeyError) as exc:
            raise tree.ContentFailure(str(exc)) from exc
    except tree.ContentFailure as exc:
        return {'id': row['id'], 'status': 'invalid_skills', 'reason': str(exc)}
    return {'id': row['id'], 'status': 'collected', 'value': value}


def outcome_path(folder, row):
    return folder / 'collected' / (vr.digest(row['id'])[:24] + '.json')


def read_outcomes(folder, rows, binding):
    outcomes, chain = [], ''
    for row in rows:
        record = vr.read(outcome_path(folder, row))
        require(record['configuration_sha256'] == binding and
                record['source_sha256'] == vr.digest(row) and record['outcome']['id'] == row['id'],
                'Collected outcome source/plan changed')
        outcomes.append(record['outcome'])
        chain = vr.digest([chain, vr.digest(record)])
    return outcomes, chain


def server_preflight(output, config):
    require(sys.platform == 'linux' and 'huhengyuan' in Path(os.getenv('CONDA_PREFIX', '')).parts,
            'Use the explicit personal server conda environment')
    require('huhengyuan' in output.parts, 'Use personal persistent storage')
    for cmd in (['df', '-h', str(output)], ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv']):
        print(r.subprocess.check_output(cmd, text=True), flush=True)
    require(config['api_base'] == 'http://127.0.0.1:18083/v1', 'Use the existing server-local service')
    navigation.old.auditor_ready(config)
    hashes = {name: vr.sha(Path(config['model']) / name) for name in runtime.MODEL_FILES}
    require(hashes == config['model_files'] == runtime.MODEL_FILES, 'Model weights/configuration changed')
    return {'model_files': hashes, 'packages': {n: r.package_version(n) for n in ('vllm', 'transformers', 'torch', 'httpx')}}


def bind_runtime(path, value):
    if path.exists():
        require(vr.read(path) == value, 'Runtime changed since initial execution')
    else:
        vr.atomic(path, value)


def emit(output, state, **extra):
    vr.atomic(output / 'execution_status.json', dict(state=state, unix=time.time(), **extra))


def build_skills(output, *, pilot):
    folder = output / 'skills'
    folder.mkdir(exist_ok=True)
    with r.run_lock(output / 'active.lock'), r.run_lock(folder / 'active.lock'):
        plan = verify(output)
        c, binding = plan['configuration'], plan['configuration_sha256']
        if (folder / 'release_manifest.json').exists():
            skill_asset(output, full=True)
            print('Skill library already completed; zero new model calls.')
            return
        if not pilot:
            require((folder / 'pilot_acceptance.json').exists() and
                    vr.read(folder / 'pilot_acceptance.json')['configuration_sha256'] == binding,
                    'Run and accept the fixed 12-question engineering pilot first')
        bind_runtime(folder / 'runtime.json', server_preflight(output, c['model']))
        (folder / 'runner.pid').write_text(str(os.getpid()))
        exit_code = 1
        try:
            b.wait_detached(folder, binding)
            journal = SkillJournal(folder, c)
            progress = vr.Progress(folder / 'collection_checkpoint.json', binding, {'cursor': 0, 'outcome_chain': ''})
            rows = vr.read(output / 'sources.json')
            stop = protocol.PILOT_SIZE if pilot else len(rows)
            while progress.state['cursor'] < stop:
                require(not (folder / 'STOP').exists(), 'STOP requested')
                row = rows[progress.state['cursor']]
                path = outcome_path(folder, row)
                if path.exists():
                    record = vr.read(path)
                    require(record['configuration_sha256'] == binding and record['source_sha256'] == vr.digest(row)
                            and record['outcome']['id'] == row['id'], 'Uncommitted outcome source changed')
                else:
                    record = {'configuration_sha256': binding, 'source_sha256': vr.digest(row),
                              'outcome': collect_one(journal, row)}
                    vr.atomic(path, record)
                journal.index['files'][str(path)] = vr.sha(path)
                vr.seal(folder / 'request_index.json', journal.index)
                progress.state['outcome_chain'] = vr.digest([progress.state['outcome_chain'], vr.digest(record)])
                progress.state['cursor'] += 1
                progress.save()
                emit(folder, 'collecting_skills', processed=progress.state['cursor'], planned=stop)
            if pilot:
                outcomes, _ = read_outcomes(folder, rows[:protocol.PILOT_SIZE], binding)
                vr.atomic(folder / 'pilot_summary.json', {'configuration_sha256': binding,
                    'questions': len(outcomes), 'statuses': dict(Counter(x['status'] for x in outcomes)),
                    'skills': sum(len(x.get('value', {}).get('skills', [])) for x in outcomes),
                    'physical_quality_passed': False, 'human_gold': False})
                emit(folder, 'pilot_completed_pending_engineering_acceptance')
            else:
                publish_skills(output, plan, journal)
            exit_code = 0
        except BaseException as exc:
            emit(folder, 'stopped', error_type=type(exc).__name__, error=str(exc)[:1200])
            raise
        finally:
            (folder / 'runner.exit_code').write_text(str(exit_code) + '\n')
            (folder / 'runner.pid').unlink(missing_ok=True)


def accept_pilot(output):
    with r.run_lock(output / 'active.lock'):
        plan = verify(output)
        folder = output / 'skills'
        saved = vr.unseal(folder / 'collection_checkpoint.json')
        questions = vr.read(output / 'sources.json')[:protocol.PILOT_SIZE]
        rows, _ = read_outcomes(folder, questions, plan['configuration_sha256'])
        summary = vr.read(folder / 'pilot_summary.json')
        vr.EvidenceCache(folder / 'evidence_cache.json').check(
            vr.unseal(folder / 'request_index.json')['files'], full=True)
        require(saved['binding'] == plan['configuration_sha256'] == summary['configuration_sha256'] and
                [x['id'] for x in rows] == plan['configuration']['pilot_ids'], 'Pilot inputs/configuration changed')
        require(summary['questions'] == protocol.PILOT_SIZE and summary['skills'] > 0 and
                all(x['status'] == 'collected' for x in rows),
                'Pilot serialization/collection failed; inspect original outputs on mainline, no blind retry')
        vr.atomic(folder / 'pilot_acceptance.json', {'configuration_sha256': plan['configuration_sha256'],
            'pilot_summary_sha256': vr.sha(folder / 'pilot_summary.json'),
            'pilot_outcomes': {str(outcome_path(folder, row)): vr.sha(outcome_path(folder, row)) for row in questions},
            'purpose': 'engineering_collection_only', 'physical_quality_passed': False,
            'human_gold': False, 'new_model_calls': 0})


def publish_skills(output, plan, journal):
    folder = output / 'skills'
    c, binding = plan['configuration'], plan['configuration_sha256']
    rows = vr.read(output / 'sources.json')
    # Collection writes one immutable outcome at a time. Summarize once, not
    # all source answers on every checkpoint or navigation request.
    prepared_names = ('materials', 'isolation', 'collection_summary')
    prepared = all(str(folder / (name + '.json')) in journal.index['files'] for name in prepared_names)
    if not prepared:
        outcomes, chain = read_outcomes(folder, rows, binding)
        checkpoint = vr.unseal(folder / 'collection_checkpoint.json')['state']
        require(checkpoint == {'cursor': len(rows), 'outcome_chain': chain}, 'Source outcome chain mismatch')
        heldout, files = b.benchmark_questions()
        require(files == c['benchmarks'] and p.digest(heldout) == c['bench_sha256'], 'Benchmark registry changed')
        visible = [dict(row, prediction=row['prediction'] + '\n' + json.dumps(outcome, ensure_ascii=False), answer='')
                   for row, outcome in zip(rows, outcomes)]
        eligible, excluded = p.isolate_sources(visible, heldout)
        vr.atomic(folder / 'isolation.json', {'generated_content_sha256': vr.digest(visible),
            'bench_sha256': c['bench_sha256'], 'excluded': excluded, 'eligible': len(eligible),
            'semantic_family_independence_proven': False})
        require(not excluded and len(eligible) == len(rows), 'Generated content overlaps bench; stop')
        materials = protocol.materialize(outcomes, vr.read(output / 'taxonomy.json'))
        require(materials['rules'], 'No skills available; cannot substitute an empty successful library')
        vr.atomic(folder / 'collection_summary.json', {'source_questions': len(rows),
            'outcome_chain': chain, 'collection_checkpoint_sha256': vr.sha(folder / 'collection_checkpoint.json'),
            'collection_statuses': dict(Counter(x['status'] for x in outcomes)),
            'source_assessments': dict(Counter(x['value']['source_assessment'] for x in outcomes if x['status'] == 'collected')),
            'valid_empty_skills': sum(not x['value']['skills'] for x in outcomes if x['status'] == 'collected'),
            'generated_skills_before_exact_dedup': sum(len(x.get('value', {}).get('skills', [])) for x in outcomes)})
        vr.atomic(folder / 'materials.json', materials)
        for name in prepared_names:
            journal.index['files'][str(folder / (name + '.json'))] = vr.sha(folder / (name + '.json'))
        vr.seal(folder / 'request_index.json', journal.index)
    else:
        for name in prepared_names:
            path = folder / (name + '.json')
            require(journal.index['files'].get(str(path)) == vr.sha(path), 'Unbound prepared skill material')
        materials = vr.read(folder / 'materials.json')
    receipt = vr.read(folder / 'isolation.json')
    require(not receipt['excluded'] and receipt['eligible'] == len(rows) and receipt['bench_sha256'] == c['bench_sha256'],
            'Generated content isolation failed')
    collection_summary = vr.read(folder / 'collection_summary.json')
    require(collection_summary['collection_checkpoint_sha256'] == vr.sha(folder / 'collection_checkpoint.json'),
            'Collection changed after skill preparation')
    progress = vr.Progress(folder / 'checkpoint.json', binding, tree.initial_state())
    engine = tree.Organizer(materials, progress, journal)
    while True:
        require(not (folder / 'STOP').exists(), 'STOP requested')
        emit(folder, 'organizing_skills', **navigation.progress_summary(progress.state, len(engine.ids)))
        if not engine.step():
            break
    products = engine.products()
    # Ensure all native fields still survive the shared navigation projection.
    source = {x['rule_id']: json.loads(x['check_logic']) for x in materials['rules']}
    source_text = {x['rule_id']: x['check_logic'] for x in materials['rules']}
    # The legacy leaf projector collapses literal whitespace even inside JSON
    # strings. Restore the original skill body mechanically, preserving all
    # mathematical spacing/text. No model edits or physical corrections occur.
    for domain in products['catalog']['domains']:
        for topic in domain['topics']:
            for leaf in topic['rules']:
                leaf['check_logic'] = source_text[leaf['rule_id']]
    corpus = r.Corpus(products['catalog'], tree=True)
    require(all(json.loads(x['check_logic']) == source[k] for k, x in corpus.leaves.items()), 'Lossy skill projection')
    for name, value in products.items():
        vr.atomic(folder / (name + '.json'), value)
    counts = dict(products['counts'], **{k: collection_summary[k] for k in
        ('source_questions', 'collection_statuses', 'source_assessments', 'valid_empty_skills', 'generated_skills_before_exact_dedup')})
    vr.atomic(folder / 'counts.json', counts)
    files = {str(path): vr.sha(path) for path in [output / 'plan.json', output / 'sources.json',
        output / 'taxonomy.json', *[folder / (n + '.json') for n in
        ['materials', 'isolation', 'runtime', 'collection_summary', 'collection_checkpoint', 'checkpoint', 'budget_checkpoint',
         'request_index', 'pilot_summary', 'pilot_acceptance', *products]] ]}
    files.update(vr.unseal(folder / 'request_index.json')['files'])
    vr.EvidenceCache(folder / 'evidence_cache.json').check(files, full=True)
    vr.atomic(folder / 'release_manifest.json', {'protocol': protocol.VERSION, 'configuration_sha256': binding,
        'files': files, 'counts': counts, 'human_gold': False, 'physical_quality_passed': False,
        'usable_for_runtime': False, 'collection_only': True,
        'semantic_models': {'skills': r.EXPECTED_MODEL, 'navigation': r.EXPECTED_MODEL},
        'source_generator': 'Qwen3-30B-A3B-Instruct-2507', 'source_is_human_gold': False,
        'external_teacher': None, 'embedding_calls': 0})
    emit(folder, 'completed_skill_collection', counts=counts)


def skill_asset(output, *, full=False):
    folder = output / 'skills'
    plan = vr.read(output / 'plan.json')
    release = vr.read(folder / 'release_manifest.json')
    require(release['configuration_sha256'] == plan['configuration_sha256'] and
            release['protocol'] == protocol.VERSION and release['human_gold'] is False and
            release['physical_quality_passed'] is False and release['usable_for_runtime'] is False and
            release['collection_only'] is True and release['embedding_calls'] == 0 and
            release['external_teacher'] is None and release['semantic_models'] ==
            {stage: r.EXPECTED_MODEL for stage in ('skills', 'navigation')} and
            release['source_generator'] == 'Qwen3-30B-A3B-Instruct-2507' and release['source_is_human_gold'] is False,
            'Skill release contract changed')
    require(release['counts']['source_questions'] == SKILL_SOURCES and release['counts']['placed_rules'] > 0 and
            release['counts']['all_candidates_preserved'] is True, 'Incomplete skill cohort/library')
    required = {str(output / n) for n in ('plan.json', 'sources.json', 'taxonomy.json')} | {
        str(folder / (n + '.json')) for n in ('materials', 'isolation', 'runtime', 'collection_checkpoint',
        'checkpoint', 'collection_summary', 'budget_checkpoint', 'request_index', 'pilot_summary', 'pilot_acceptance',
        'catalog', 'validation', 'topic_mapping', 'unplaced_rules', 'counts')}
    require(required <= set(release['files']), 'Skill release omits required evidence')
    vr.EvidenceCache(folder / 'evidence_cache.json').check(release['files'], full=full)
    require(vr.read(folder / 'counts.json') == release['counts'] and
            vr.read(folder / 'validation.json')['valid'] and not vr.read(folder / 'isolation.json')['excluded'],
            'Skill structure/isolation failed')
    return {'path': str(folder), 'release_sha256': vr.sha(folder / 'release_manifest.json'),
            'catalog_sha256': vr.sha(folder / 'catalog.json'), 'counts': release['counts'],
            'physical_quality_passed': False, 'human_gold': False}


def evaluation_configuration(output, *, full=False):
    plan = verify(output, full=full)
    skill = skill_asset(output, full=full)
    # Labels are loaded only by the evaluator/validator, never by a prompt builder.
    evaluation.reference_data(ROOT)
    rows = vr.read(runtime.INPUT)
    require(len(rows) == 50 and vr.sha(runtime.INPUT) == r.EXPECTED_INPUT_SHA256 and
            len({x['id'] for x in rows}) == 50 and
            all(not set(x) - {'id', 'question', 'prediction', 'context'} for x in rows), 'Evaluation inputs changed/leak labels')
    return {'protocol': protocol.VERSION, 'construction_configuration_sha256': plan['configuration_sha256'],
        'assets': {'rules': plan['configuration']['rules'], 'skills': skill},
        'input_sha256': vr.sha(runtime.INPUT), 'sample_ids': [x['id'] for x in rows],
        'model': r.EXPECTED_MODEL, 'model_files': runtime.MODEL_FILES,
        'api_base': 'http://127.0.0.1:18083/v1', 'reasoning_effort': 'medium', 'temperature': 0, 'top_p': 1,
        'limits': {**runtime.LIMITS, 'review_protocol': continuous.PROTOCOL},
        'base_prompt': protocol.BASE, 'knowledge_prompt': protocol.KNOWLEDGE,
        'source_hashes': source_hashes(), 'reference_version': evaluation.RELEASE_VERSION,
        'reference_sha256': {n: vr.sha(ROOT / n) for n in (evaluation.RELEASE, evaluation.REFERENCE)},
        'arms': list(protocol.ARMS), 'controls': runtime.CONTROLS,
        'tools': comparison.execution.runner.native_tools(True),
        'metric': 'frozen_new_reference_location_proxy_not_semantic_accuracy',
        'case_policy': {'seed': protocol.SEED, 'categories': ['rules_only_hits', 'skills_only_hits',
             'shared_miss', 'same_hits_different_findings'], 'physics_review': 'Codex assistant, not human gold'},
        'human_gold': False, 'independent_test': False}


def prepare_eval(output):
    folder = output / 'evaluation'
    folder.mkdir(exist_ok=True)
    with r.run_lock(output / 'active.lock'):
        c = evaluation_configuration(output, full=True)
        target = folder / 'plan.json'
        plan = {'configuration': c, 'configuration_sha256': vr.digest(c)}
        if target.exists():
            require(vr.read(target) == plan, 'Evaluation already frozen under different materials/protocol')
        else:
            vr.atomic(target, plan)
    print('Prepared two arms x full 50; no generation or scoring calls.')


def verify_eval(output):
    plan = vr.read(output / 'evaluation/plan.json')
    require(plan['configuration'] == evaluation_configuration(output) and
            plan['configuration_sha256'] == vr.digest(plan['configuration']), 'Evaluation binding changed')
    return plan


@contextmanager
def configured_evaluation():
    with comparison.execution.configured(), ExitStack() as stack:
        stack.enter_context(patch.object(continuous, 'BASE', protocol.BASE))
        stack.enter_context(patch.object(continuous, 'RULES', protocol.KNOWLEDGE))
        stack.enter_context(patch.object(continuous, 'progress_note', protocol.progress_note))
        stack.enter_context(patch.object(continuous, 'audit_final', protocol.audit_final))
        stack.enter_context(patch.object(comparison.execution.transport, 'decode_response', comparison.serialization.decode_response))
        yield


def collect_evaluation(folder, rows, binding):
    groups = {}
    for arm in protocol.ARMS:
        found = []
        for row in rows:
            short = r.hashlib.sha256(str(row['id']).encode()).hexdigest()[:16]
            path = folder / arm / 'samples/C4' / short / 'result.json'
            if path.exists():
                item = vr.read(path)
                require(item['configuration_sha256'] == binding and item['id'] == row['id'] and
                        item['verifier'] == 'C4', 'Paired result identity changed')
                runtime.audit_result(item)
                found.append(dict(item, knowledge_arm=arm, original_result_path=str(path),
                                  original_result_sha256=vr.sha(path)))
        groups[arm] = found
    return groups


def run_eval(output):
    folder = output / 'evaluation'
    with r.run_lock(output / 'active.lock'), r.run_lock(folder / 'active.lock'), configured_evaluation():
        plan = verify_eval(output)
        c, binding = plan['configuration'], plan['configuration_sha256']
        rows = vr.read(runtime.INPUT)
        existing = collect_evaluation(folder / 'formal', rows, binding)
        if all(len(items) == 50 for items in existing.values()):
            score(output)
            print('All 100 evaluations already recorded; zero generation calls.')
            return
        bind_runtime(folder / 'runtime.json', server_preflight(output, c))
        (folder / 'runner.pid').write_text(str(os.getpid()))
        exit_code = 1
        try:
            b.wait_detached(folder, binding)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(r.EXPECTED_MODEL, local_files_only=True, trust_remote_code=False)
            catalogs = {a: vr.read(Path(c['assets'][a]['path']) / 'catalog.json') for a in protocol.ARMS}
            with httpx.Client(trust_env=False) as client:
                for arm in protocol.ARMS:
                    runtime.native_io_probe(client, tokenizer, catalogs[arm], folder / 'engineering' / arm, c)
                for phase, samples in [('controls', runtime.CONTROLS), ('formal', rows)]:
                    for i, row in enumerate(samples):
                        order = protocol.ARMS[i % 2:] + protocol.ARMS[:i % 2]
                        for arm in order:
                            require(not (folder / 'STOP').exists(), 'STOP requested')
                            result = r.run_sample(row, arm='C4', catalog=catalogs[arm], tokenizer=tokenizer,
                                client=client, endpoint=c['api_base'] + '/chat/completions', model=r.EXPECTED_MODEL,
                                root=folder / phase / arm, signature=binding, limits=c['limits'])
                            runtime.audit_result(result)
                            emit(folder, phase, sample_id=row['id'], knowledge_arm=arm, result_status=result['status'])
                            require(runtime.healthy_result(result), 'Technical failure retained; inspect exact request before recovery')
                            if phase == 'controls':
                                require(result['protocol_audit']['passed'], 'Engineering quote/protocol failure')
                score(output)
            emit(folder, 'completed_evaluation', recorded_tasks=100)
            exit_code = 0
        except BaseException as exc:
            emit(folder, 'stopped', error_type=type(exc).__name__, error=str(exc)[:1200])
            raise
        finally:
            (folder / 'runner.exit_code').write_text(str(exit_code) + '\n')
            (folder / 'runner.pid').unlink(missing_ok=True)


def score(output):
    plan = verify_eval(output)
    folder = output / 'evaluation'
    rows = vr.read(runtime.INPUT)
    groups = collect_evaluation(folder / 'formal', rows, plan['configuration_sha256'])
    _, dataset, unknown = evaluation.reference_data(ROOT)
    scores = {}
    for arm, items in groups.items():
        if len(items) == 50:
            scores[arm] = evaluation.score(dataset, [dict(x, execution_failure=not runtime.healthy_result(x),
                source_status=x['status'], protocol_noncompliance=not x.get('protocol_audit', {}).get('passed', False))
                for x in items], unknown)
    report = {'configuration_sha256': plan['configuration_sha256'],
        'metric': plan['configuration']['metric'], 'reference_version': evaluation.RELEASE_VERSION,
        'human_gold': False, 'independent_test': False, 'scores_all_50': scores,
        'recorded_tasks': {a: len(v) for a, v in groups.items()},
        'library_use_and_compute': {a: comparison.trace_summary(v) for a, v in groups.items()}}
    if len(scores) == 2:
        report['rules_minus_skills_paired_bootstrap'] = evaluation.bootstrap(scores['rules'], scores['skills'])
        cases = protocol.case_selection(scores['rules'], scores['skills'])
        by_input = {x['id']: x for x in rows}
        by_reference = {x['id']: x for x in dataset}
        by_arm = {a: {x['id']: x for x in items} for a, items in groups.items()}
        cases['cases'] = [dict(x, input=by_input[x['id']], reference=by_reference[x['id']]['physics_error_gt'],
            unresolved_reference=unknown[x['id']],
            outputs={a: by_arm[a][x['id']] for a in protocol.ARMS},
            review={'actor': 'Codex assistant', 'status': 'pending', 'human_gold': False,
                    'same_physical_error': None, 'knowledge_contribution': None, 'basis': ''})
            for x in cases['selection']]
        review_path = folder / 'case_review.json'
        if review_path.exists():
            previous = vr.read(review_path)
            require(previous['configuration_sha256'] == plan['configuration_sha256'] and
                    previous['selection'] == cases['selection'], 'Existing case review selection changed')
        else:
            vr.atomic(review_path, dict(cases, configuration_sha256=plan['configuration_sha256']))
    vr.atomic(folder / 'metrics.json', report)
    return report


def confirm(output, phase):
    folder = output / phase
    ready = vr.read(folder / 'disconnect_ready.json')
    require(runtime.process_identity(ready['pid']) == ready['process_start'] and time.time() - ready['unix'] >= 5,
            'Reconnect in a fresh SSH session before confirmation')
    state = vr.read(folder / 'execution_status.json')
    require(state['state'] == 'waiting_disconnect_acceptance' and state['token'] == ready['token'], 'Runner is not waiting')
    plan = verify_eval(output) if phase == 'evaluation' else verify(output)
    require(ready['configuration_sha256'] == plan['configuration_sha256'], 'Ready plan changed')
    vr.atomic(folder / 'disconnect_acceptance.json', dict(ready, passed=True, confirmed_unix=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify', 'build-pilot', 'accept-pilot', 'build',
        'prepare-eval', 'run-eval', 'confirm-detached', 'score', 'status'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rule-tree', type=Path)
    parser.add_argument('--phase', choices=['skills', 'evaluation'], default='skills')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode == 'prepare':
        require(args.rule_tree is not None, '--rule-tree is required for preparation')
        prepare(output, args.rule_tree.resolve())
    elif args.mode == 'verify':
        verify(output, full=True)
        print('Full evidence verification passed; zero model calls.')
    elif args.mode in {'build-pilot', 'build'}:
        build_skills(output, pilot=args.mode == 'build-pilot')
    elif args.mode == 'accept-pilot':
        accept_pilot(output)
    elif args.mode == 'prepare-eval':
        prepare_eval(output)
    elif args.mode == 'run-eval':
        run_eval(output)
    elif args.mode == 'confirm-detached':
        confirm(output, args.phase)
    elif args.mode == 'score':
        report = score(output)
        print(json.dumps({a: v['summary'] for a, v in report['scores_all_50'].items()}, ensure_ascii=False))
    elif args.mode == 'status':
        path = output / args.phase / 'execution_status.json'
        print(json.dumps(vr.read(path) if path.exists() else vr.read(output / 'execution_status.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
