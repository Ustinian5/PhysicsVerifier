"""Import completed extraction once, then build a bounded, resumable 120B tree.

This is an explicit navigation protocol revision, not lossless replay of the
old clustering. Original answers, judgments, candidates, old groups and failed
requests remain immutable. Outputs live in the same batch's tree_v2 directory.
No inference in prepare/verify/status/confirm-detached. Model use requires a
detached server runner and fresh SSH acceptance. No external teacher/embedding.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

from rule_framework import bounded_tree as tree, verified_resume as vr
from scripts import run_qwen30b_expansion as old

b, p, r, e = old.b, old.p, old.r, old.e
NEW_SOURCES = ['rule_framework/bounded_tree.py', 'rule_framework/verified_resume.py',
               'scripts/reorganize_qwen30b_expansion.py']


def source_hashes():
    return {name: r.sha256_file(b.ROOT / name) for name in sorted(set(old.SOURCES + NEW_SOURCES))}


def emit(output, state, **extra):
    vr.atomic(output / 'execution_status.json', {'state': state, 'pid': os.getpid(),
              'updated_unix': time.time(), **extra})


def bind(files, path, expected=None):
    path = str(Path(path).absolute())
    value = expected or vr.sha(path)
    p.require(path not in files or files[path] == value, 'Conflicting evidence binding')
    files[path] = value


def add_release(files, root, name):
    release = b.read(root / name)
    bind(files, root / name)
    for relative, value in release['files'].items():
        path = Path(relative)
        p.require(not path.is_absolute() and '..' not in path.parts, 'Unsafe release path')
        bind(files, root / path, value)
    return release


def decode_legacy(raw, config):
    p.require(not raw.get('error') and raw.get('response', {}).get('model') == config['model'] and
              raw['response'].get('id'), 'Invalid original response identity')
    text = b.harmony.decode_response(raw['response'], model=config['model'],
             max_tokens=config['max_tokens'], native_tools=None)['content'].strip()
    if text.startswith('```json\n') and text.endswith('```'):
        text = text[8:-3].strip()
    try:
        return b.parse_json_escapes(e.escape_json_controls(text)[0], repair=True)[0]
    except (ValueError, json.JSONDecodeError) as exc:
        if not isinstance(exc, json.JSONDecodeError) and str(exc) != 'Ambiguous escaped control character':
            raise
        return {'_unparsed_response_text': text}


def saved_value(root, config, stage, key, payload, evidence):
    """Pure read: no Journal constructor, budget scan, model call, or new sidecar."""
    folder = root / ('arm_' + stage) / p.digest(key)[:24]
    receipt, prefix = folder / 'result.json', folder / 'prefix_replay.json'
    if receipt.exists():
        saved = b.read(receipt)
        relative = Path(saved['request_file'])
        p.require(relative.name == str(relative), 'Unsafe original request filename')
        path, expected = folder / relative, saved['request_sha256']
        bind(evidence, receipt)
    else:
        p.require(prefix.exists(), 'Missing completed extraction: ' + key)
        saved = b.read(prefix)
        path, expected = Path(saved['source']), saved['source_sha256']
        bind(evidence, prefix)
    p.require(vr.sha(path) == expected, 'Original raw response changed')
    bind(evidence, path, expected)
    raw = b.read(path)
    journal = e.Journal.__new__(e.Journal)
    body = r.request_body(config['model'], journal.messages(stage, payload), config['max_tokens'], backend='vllm')
    body.update(stream=False, return_token_ids=True, reasoning_effort=config['reasoning_effort'])
    body.pop('stream_options', None)
    p.require(raw['request'] == body, 'Original extraction request does not exactly match')
    value = decode_legacy(raw, config)
    p.require(value == saved['value'] if receipt.exists() else p.digest(value) == saved['value_sha256'],
              'Original parsed extraction differs')
    return value


def extraction_bundle(root, parent, config, evidence, on_progress=lambda *a: None):
    """Reconstruct only extraction; the large original registry is never rebuilt."""
    rows = b.read(root / 'build/inputs.json')
    taxonomy = {'domains': []}
    skeleton = b.read(parent / 'topic_skeleton.json')
    for domain in skeleton['domains']:
        part = saved_value(parent, config, 'taxonomy', domain['id'], {'domains': [domain]}, evidence)
        p.require(e.flow.experience_topic_skeleton(part) == {'domains': [domain]}, 'Taxonomy changed')
        taxonomy['domains'].extend(part['domains'])
    e.extraction._experience_topics(taxonomy)
    base_ids = {row['id'] for row in b.read(parent / 'inputs.json')}
    experiences, invalid = [], []
    for index, row in enumerate(rows):
        where = parent if row['id'] in base_ids else root / 'build'
        payload = {'sample': row, 'taxonomy': taxonomy, 'max_rules_per_sample': config['max_rules_per_sample']}
        draft = saved_value(where, config, 'extract_experience', row['id'], payload, evidence)
        try:
            normalized = e.extraction.normalize_120b_experience(draft, **payload)
        except ValueError as exc:
            draft = saved_value(where, config, 'repair_experience', row['id'],
                                {'original': payload, 'draft': draft, 'validation_error': str(exc)}, evidence)
            try:
                normalized = e.extraction.normalize_120b_experience(draft, **payload)
            except ValueError as final:
                invalid.append({'sample_id': row['id'], 'error': str(final), 'raw_value_sha256': p.digest(draft)})
                continue
        experiences.append(normalized)
        if index % 100 == 0:
            on_progress(index + 1, len(rows))
    distilled = e.extraction._build_distilled_library(experiences, min_count=1)
    checkpoint = b.read(root / 'build/extraction_checkpoint.json')
    p.require(checkpoint['semantic_experience_sha256'] == p.digest(experiences) and
              checkpoint['distilled_sha256'] == p.digest(distilled) and
              checkpoint['inputs_sha256'] == vr.sha(root / 'build/inputs.json'), 'Extraction checkpoint differs')
    support = defaultdict(list)
    for sample in experiences:
        domain, topic = sample['topic_guess']['domain'], sample['topic_guess']['topic']
        errors = {item['id']: item for item in sample['semantic_audit']['key_errors']}
        for rule in sample['experience_rules']:
            rid = 'exp_' + e.extraction._fingerprint_rule(rule, domain, topic)
            support[rid].extend({'sample_id': sample['sample_id'], 'source_error': errors[eid]}
                                for eid in rule['source_error_ids'])
    p.require(len(rows) == checkpoint['counts']['complete_inputs'] and
              len(invalid) == checkpoint['counts']['invalid_outputs'], 'Extraction accounting differs')
    for rule in distilled['rules']:
        p.require(set(rule['sample_ids']) == {v['sample_id'] for v in support[rule['rule_id']]}, 'Lost rule support')
    return {'taxonomy': taxonomy, 'rules': distilled['rules'], 'provenance': dict(support),
            'source_counts': checkpoint['counts'], 'invalid_outputs': invalid,
            'isolation_input_sha256': p.digest(rows), 'semantic_experience_sha256': p.digest(experiences)}


def ensure_idle(root):
    # The flock is authoritative. PID files can be stale but a live owner must
    # be inspected rather than killed or silently taken over.
    if (root / 'runner.pid').exists():
        pid = int((root / 'runner.pid').read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        raise ValueError('Original client is still alive; do not run two builders')


def prepare(root):
    """One full import, with resumable import phases; never start an LLM here."""
    output = root / 'tree_v2'
    output.mkdir(exist_ok=True)
    selection = b.read(root / 'selection.json')
    parent, corpus = Path(selection['parent_build']), Path(selection['base_corpus'])
    with r.run_lock(root / 'active.lock'), r.run_lock(parent / 'active.lock'), r.run_lock(output / 'active.lock'):
        ensure_idle(root)
        if (output / 'plan.json').exists():
            verify(output)
            print('Already prepared; no extraction replay or new model calls.')
            return
        start = time.monotonic()
        phase_file = output / 'import_checkpoint.json'
        cached = vr.unseal(phase_file) if phase_file.exists() else None
        if cached:
            p.require(cached['source_hashes'] == source_hashes(), 'Import implementation changed')
            vr.EvidenceCache(output / 'evidence_cache.json').check(cached['files'])
            plan = b.read(root / 'plan.json')
            evidence = cached['files']
        else:
            emit(output, 'verifying_legacy_once')
            plan = old.verify(root)  # Narrow original amendment chain, exactly once.
            config = plan['configuration']
            evidence = {}
            release = add_release(evidence, parent, 'release_manifest.json')
            generation = add_release(evidence, root, 'generation_release.json')
            add_release(evidence, corpus, 'release_manifest.json')
            checkpoint = b.read(root / 'build/extraction_checkpoint.json')
            p.require(release['configuration_sha256'] == config['parent_configuration_sha256'] and
                      generation['configuration_sha256'] == checkpoint['configuration_sha256'] == plan['configuration_sha256'] and
                      checkpoint['parent_release_sha256'] == vr.sha(parent / 'release_manifest.json'),
                      'Parent/generation changed')
            for name in ['plan.json', 'selection.json', 'bench_questions.json', 'build/extraction_checkpoint.json',
                         'build/inputs.json', old.CLUSTER_RECOVERY, old.MISSING_RECOVERY, old.PROMPT_RECOVERY]:
                bind(evidence, root / name)
            for name, expected in {**selection['source_files'], **selection['base_bindings']}.items():
                bind(evidence, name, expected)
            # Bind all old requests (including failures) for immutable accounting.
            for path in (root / 'build').glob('arm*/*/*.json'):
                bind(evidence, path)
            for name in ('source_release.json', 'isolation.json', 'topic_skeleton.json'):
                bind(evidence, parent / name)
            heldout, benchmarks = b.benchmark_questions()
            p.require(benchmarks == selection['benchmark_files'] and p.digest(heldout) == selection['bench_sha256'],
                      'Registered benchmark changed')
            for name, info in benchmarks.items():
                bind(evidence, b.ROOT / name, info['sha256'])
            stats = vr.EvidenceCache(output / 'evidence_cache.json').check(evidence, full=True)
            cached = {'files': evidence, 'source_hashes': source_hashes(), 'benchmarks': benchmarks,
                      'bench_sha256': p.digest(heldout), 'legacy_configuration_sha256': plan['configuration_sha256'],
                      'first_hash_check': stats, 'input_verified': False, 'bundle_ready': False}
            vr.seal(phase_file, cached)

        config = plan['configuration']
        if not cached['input_verified']:
            emit(output, 'validating_generation_and_isolation_once')
            _, baseline = old.released_corpus.load(corpus / 'release_manifest.json', corpus / 'answers.json',
                                                  full=False, complete_answers=True)
            for row in b.read(root / 'new_questions.json'):
                old.validate_generation(root, row, config['generator'])
            new = b.read(root / 'admitted_new_answers.json')
            rows = sorted(baseline + new, key=lambda x: p.digest([20261024, x['id']]))
            p.require(rows == b.read(root / 'build/inputs.json'), 'Construction cohort changed')
            # Isolation is computed once for the exact complete cohort, never on
            # each confirmation, restart, or individual page. No label access.
            eligible, excluded = p.isolate_sources(rows, b.read(root / 'bench_questions.json'))
            p.require(not excluded and len(eligible) == len(rows), 'Benchmark overlap')
            vr.atomic(output / 'isolation_receipt.json', {
                'inputs_sha256': p.digest(rows), 'bench_sha256': cached['bench_sha256'],
                'algorithm_sha256': vr.sha(b.ROOT / 'rule_framework/oss120b_construction.py'),
                'policy': p.ISOLATION_POLICY, 'rows': len(rows), 'excluded': [],
                'semantic_family_isolation_proven': False})
            bind(evidence, output / 'isolation_receipt.json')
            cached['input_verified'] = True
            vr.seal(phase_file, cached)

        if not cached['bundle_ready']:
            emit(output, 'importing_extraction_once')
            bundle = extraction_bundle(root, parent, config['construction'], evidence,
                         lambda n, total: emit(output, 'importing_extraction_once', completed=n, total=total))
            vr.atomic(output / 'materials.json', bundle)
            bind(evidence, output / 'materials.json')
            cached['bundle_ready'] = True
            vr.seal(phase_file, cached)
        bundle = b.read(output / 'materials.json')

        emit(output, 'binding_inherited_usage_once')
        pc = b.read(parent / 'plan.json')['configuration']
        ancestors = [parent]
        if pc.get('draft_prefix_parent'):
            ancestors.append(Path(pc['draft_prefix_parent']))
        # All previous construction usage, including failed attempts, is charged.
        budget = b.BatchBudget(root / 'build', {'requests': config['construction']['max_requests'],
                               'input_tokens': config['construction']['max_requests'] * config['construction']['max_request_input_bytes'],
                               'output_tokens': config['construction']['max_requests'] * config['construction']['max_tokens']},
                               prior_runs=tuple(ancestors))
        prior = {'requests_started': budget.requests, 'accounted_input_tokens': budget.input_tokens,
                 'accounted_output_tokens': budget.output_tokens, 'requests_without_usage': budget.unknown_usage,
                 'in_flight': 0}
        # Ancestor requests outside the parent manifest (e.g. the pilot) also
        # belong to the immutable accounting receipt.
        for ancestor in ancestors:
            for path in ancestor.glob('arm*/*/request_*.json'):
                bind(evidence, path)
        vr.atomic(output / 'import_manifest.json', {'files': evidence,
            'source_counts': bundle['source_counts'], 'legacy_configuration_sha256': plan['configuration_sha256'],
            'raw_candidates_unchanged': True, 'new_model_calls': 0})
        c = {'protocol': tree.VERSION, 'legacy_root': str(root), 'source_hashes': source_hashes(),
             'materials_sha256': vr.sha(output / 'materials.json'),
             'import_manifest_sha256': vr.sha(output / 'import_manifest.json'),
             'isolation_sha256': vr.sha(output / 'isolation_receipt.json'),
             'policy': tree.POLICY, 'prompts_sha256': vr.digest(tree.PROMPTS),
             'model': config['construction'], 'budget_limits': budget.limits, 'prior_usage': prior,
             'benchmarks': cached['benchmarks'], 'bench_sha256': cached['bench_sha256'],
             'collection_only': True, 'physical_quality_passed': False, 'usable_for_runtime': False,
             'new_protocol_old_cluster_decisions_not_reused': True}
        vr.atomic(output / 'plan.json', {'configuration': c, 'configuration_sha256': vr.digest(c)})
        verify(output)
        emit(output, 'prepared_new_navigation_protocol', input_candidates=len(bundle['rules']),
             preparation_seconds=round(time.monotonic() - start, 2), new_model_calls=0)


def verify(output, *, full=False):
    start = time.monotonic()
    plan = b.read(output / 'plan.json')
    c = plan['configuration']
    p.require(plan['configuration_sha256'] == vr.digest(c) and c['protocol'] == tree.VERSION,
              'Navigation plan changed')
    p.require(c['source_hashes'] == source_hashes() and c['policy'] == tree.POLICY and
              c['prompts_sha256'] == vr.digest(tree.PROMPTS), 'Navigation implementation/prompt changed')
    for name, key in [('materials.json', 'materials_sha256'), ('import_manifest.json', 'import_manifest_sha256'),
                      ('isolation_receipt.json', 'isolation_sha256')]:
        p.require(vr.sha(output / name) == c[key], 'Frozen navigation artifact changed: ' + name)
    heldout, benchmarks = b.benchmark_questions()
    p.require(benchmarks == c['benchmarks'] and p.digest(heldout) == c['bench_sha256'], 'Benchmark drift: stop')
    receipt = b.read(output / 'isolation_receipt.json')
    materials = b.read(output / 'materials.json')
    p.require(receipt['inputs_sha256'] == materials['isolation_input_sha256'] and
              receipt['bench_sha256'] == c['bench_sha256'] and receipt['excluded'] == [] and
              receipt['algorithm_sha256'] == vr.sha(b.ROOT / 'rule_framework/oss120b_construction.py'),
              'Isolation receipt no longer applies')
    evidence = dict(b.read(output / 'import_manifest.json')['files'])
    if (output / 'request_index.json').exists():
        index = vr.unseal(output / 'request_index.json')
        p.require(index['binding'] == plan['configuration_sha256'], 'Request index belongs to another plan')
        for name, expected in index['files'].items():
            bind(evidence, name, expected)
    if (output / 'release_manifest.json').exists():
        release = b.read(output / 'release_manifest.json')
        p.require(release['configuration_sha256'] == plan['configuration_sha256'], 'Release belongs to another plan')
        for name, expected in release['files'].items():
            bind(evidence, name, expected)
    stats = vr.EvidenceCache(output / 'evidence_cache.json').check(evidence, full=full)
    vr.atomic(output / 'verification.json', dict(stats, total_seconds=round(time.monotonic() - start, 4),
        isolation_recomputed=False, historical_model_outputs_replayed=0, verified_unix=time.time()))
    return plan


class TreeJournal(e.Journal):
    def __init__(self, output, c):
        # Avoid Journal.__init__ and its full raw transcript budget scan.
        self.output, self.config, self.retry = output, c['model'], True
        self.distill_requests = 0
        self.budget = vr.resumed_budget(b.BatchBudget, output, c['budget_limits'], c['prior_usage'])
        self.binding = vr.digest(c)
        self.index = vr.unseal(output / 'request_index.json') if (output / 'request_index.json').exists() else {
            'binding': self.binding, 'files': {}}
        p.require(self.index['binding'] == self.binding, 'Request ledger binding changed')

    def record_job(self, stage, key):
        folder = self.output / ('arm_' + stage) / p.digest(key)[:24]
        for path in folder.glob('*.json'):
            actual = vr.sha(path)
            name = str(path.absolute())
            p.require(name not in self.index['files'] or self.index['files'][name] == actual,
                      'Previously recorded model evidence changed')
            self.index['files'][name] = actual
        vr.seal(self.output / 'request_index.json', self.index)

    def supported_stages(self):
        return set(tree.PROMPTS) | {'complete_missing'}

    def messages(self, stage, payload):
        value = tree.messages(stage, payload)
        r.assert_no_control_tokens(value)
        return value

    def decode_record(self, path, stage):
        raw = b.read(path)
        p.require(not raw.get('error') and raw.get('http_status') == 200 and
                  raw.get('response', {}).get('model') == self.config['model'] and raw['response'].get('id'),
                  'Unknown identity or transport/protocol failure')
        choices = raw['response'].get('choices')
        p.require(isinstance(choices, list) and len(choices) == 1, 'Unknown response protocol')
        if choices[0].get('finish_reason') == 'length':
            raise tree.ContentFailure('length_truncated; original response retained, no resampling')
        p.require(choices[0].get('finish_reason') == 'stop', 'Unknown completion reason')
        value = decode_legacy(raw, self.config)
        if not isinstance(value, dict) or '_unparsed_response_text' in value:
            raise tree.ContentFailure('malformed_json; no semantic repair or resampling')
        return value

    def __call__(self, stage, key, payload):
        p.require(tree.input_size(stage, payload) <= tree.POLICY['max_input_bytes'], 'Input bound changed')
        while True:
            try:
                value = super().__call__(stage, key, payload)
                self.record_job(stage, key)
                return value
            except tree.ContentFailure:
                self.record_job(stage, key)
                raise
            except Exception:
                self.record_job(stage, key)
                paths = sorted((self.output / ('arm_' + stage) / p.digest(key)[:24]).glob('request_*.json'))
                if not paths or len(paths) >= self.config['attempts_per_job'] or self.budget.stopped:
                    raise
                record = b.read(paths[-1])
                if not b.retryable_transport_failure(record):
                    raise
                deadline = max(b.retry_after_deadline(record),
                    record.get('started_unix', 0) + record.get('elapsed_seconds', 0) + 300)
                while time.time() < deadline:
                    p.require(not (self.output / 'STOP').exists(), 'STOP requested')
                    emit(self.output, 'waiting_transport_cooldown', stage=stage, ready_unix=deadline)
                    time.sleep(min(5, max(.1, deadline - time.time())))


def progress_summary(state, total):
    return {'phase': state['phase'], 'candidates': total, 'topic_processed': state['topic_cursor'],
            'placed': len(state['placed']), 'unresolved': len(state['unresolved']),
            'groups': sum(map(len, state['groups'].values())), 'jobs_committed': state['jobs_committed'],
            'active_page': state['active']['pages'] if state['active'] else None}


def run(output):
    p.require(sys.platform == 'linux' and 'huhengyuan' in Path(os.getenv('CONDA_PREFIX', '')).parts,
              'Use the personal server conda environment')
    root = output.parent
    with r.run_lock(root / 'active.lock'), r.run_lock(output / 'active.lock'):
        ensure_idle(root)
        (output / 'runner.pid').write_text(str(os.getpid()))
        try:
            emit(output, 'verifying_resume')
            plan = verify(output)
            c = plan['configuration']
            if (output / 'release_manifest.json').exists():
                verify(output, full=True)
                print('Navigation collection already finished; no new calls.')
                return
            b.wait_detached(output, plan['configuration_sha256'])
            old.auditor_ready(c['model'])
            journal = TreeJournal(output, c)
            materials = b.read(output / 'materials.json')
            progress = vr.Progress(output / 'checkpoint.json', plan['configuration_sha256'], tree.initial_state())
            engine = tree.Organizer(materials, progress, journal)
            while True:
                p.require(not (output / 'STOP').exists(), 'STOP requested')
                emit(output, 'running', **progress_summary(progress.state, len(engine.ids)))
                if not engine.step():
                    break
            products = engine.products()
            emit(output, 'verifying_publication', **progress_summary(progress.state, len(engine.ids)))
            verify(output, full=True)
            files = dict(b.read(output / 'import_manifest.json')['files'])
            for name, value in products.items():
                vr.atomic(output / (name + '.json'), value)
            for path in output.glob('arm*/*/*.json'):
                bind(files, path)
            for name in ['materials', 'plan', 'import_manifest', 'isolation_receipt', 'checkpoint', 'budget_checkpoint', 'request_index', *products]:
                bind(files, output / (name + '.json'))
            vr.atomic(output / 'release_manifest.json', {'protocol': tree.VERSION,
                'configuration_sha256': plan['configuration_sha256'], 'files': files, 'counts': products['counts'],
                'collection_only': True, 'physical_quality_passed': False, 'usable_for_runtime': False,
                'human_gold': False, 'has_unresolved': bool(products['counts']['unplaced_rules'])})
            emit(output, 'completed_navigation_collection', counts=products['counts'])
        except BaseException as exc:
            emit(output, 'stopped', error_type=type(exc).__name__, error=str(exc)[:1200])
            raise
        finally:
            (output / 'runner.pid').unlink(missing_ok=True)


def confirm(output):
    # Confirmation is a handshake, not an excuse to scan 20,000 old files again.
    plan = b.read(output / 'plan.json')
    p.require(vr.digest(plan['configuration']) == plan['configuration_sha256'] and
              plan['configuration']['source_hashes'] == source_hashes(), 'Plan/source changed')
    ready = b.read(output / 'disconnect_ready.json')
    state = b.read(output / 'execution_status.json')
    p.require(ready['configuration_sha256'] == plan['configuration_sha256'] and
              b.existing.process_identity(ready['pid']) == ready['process_start'] and
              time.time() - ready['unix'] >= 5 and state['state'] == 'waiting_disconnect_acceptance' and
              state['token'] == ready['token'], 'Reconnect to this detached process after five seconds')
    vr.atomic(output / 'disconnect_acceptance.json', dict(ready, passed=True, confirmed_unix=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify', 'run', 'confirm-detached', 'status'])
    parser.add_argument('--root', type=Path, required=True, help='Existing cumulative-5000 output, not a new copy')
    parser.add_argument('--full', action='store_true', help='Force content hashes; used at import/publication')
    args = parser.parse_args()
    output = args.root / 'tree_v2'
    if args.mode == 'prepare':
        try:
            prepare(args.root)
        except BaseException as exc:
            status_path = output / 'execution_status.json'
            if status_path.exists() and b.read(status_path).get('pid') == os.getpid():
                emit(output, 'preparation_stopped', error_type=type(exc).__name__, error=str(exc)[:1200])
            raise
    elif args.mode == 'verify':
        print(json.dumps(verify(output, full=args.full)['configuration_sha256']))
    elif args.mode == 'run':
        run(output)
    elif args.mode == 'confirm-detached':
        confirm(output)
    else:
        print(json.dumps(b.read(output / 'execution_status.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
