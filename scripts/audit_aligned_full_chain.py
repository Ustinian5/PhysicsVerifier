"""Read-only audit of the completed combined-source/30B/old-v15 experiment.

The source subcommand streams the original local dataset. The server subcommand
checks frozen evidence and replays the original execution audit in memory,
without replacing any prior acceptance, labels, predictions or scores.
Additional findings are observations, never automatic score or protocol edits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unicodedata
from unittest.mock import patch

from scripts import review_aligned_step_scoring as review


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def key(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).casefold()


def progress(root, state, **details):
    review.write(root / 'offline_review/full_chain_progress.json',
                 dict(state=state, pid=os.getpid(), unix=time.time(), **details))


def source_audit(root, source, inventory, references):
    from scripts.combined_language_io import iter_rollout_batches
    from scripts.combined_language_samples import extract_question
    inv, ref = review.read(inventory), review.read(references)
    by_key = {key(row['question']): row['id'] for row in inv['questions']}
    assert len(by_key) == len(inv['questions']) == 4839
    expected = {row['id']: {json.dumps(v['value'], sort_keys=True, ensure_ascii=False): v['count']
                            for v in row['variants']} for row in ref['references']}
    actual = defaultdict(Counter)
    records, batches = 0, 0
    before = source.stat()
    for batch in iter_rollout_batches(source):
        batches += 1
        for sample in batch['samples']:
            sid = by_key[key(extract_question(sample))]
            actual[sid][json.dumps(sample.get('label'), sort_keys=True, ensure_ascii=False)] += 1
            records += 1
        if batches % 20 == 0:
            progress(root, 'source_stream', batches=batches, records=records)
    assert records == 197632 and dict(actual) == expected
    assert (before.st_size, before.st_mtime_ns) == (source.stat().st_size, source.stat().st_mtime_ns)
    source_sha = sha(source)
    assert source_sha == ref['source_sha256']
    report_path = root / 'offline_review/full_chain_audit.json'
    report = review.read(report_path) if report_path.exists() else {}
    if report.get('raw_source'):
        assert report['raw_source']['source_sha256'] == source_sha, 'Audit source changed'
    report.update(raw_source=dict(passed=True, source=str(source), source_sha256=source_sha,
        inventory_sha256=sha(inventory), references_sha256=sha(references),
        question_keys=list(by_key), records=records, unique_questions=len(by_key), batches=batches,
        reference_values_and_frequencies_equal=True, historical_predictions_or_rewards_used=False),
        audit_code_sha256=sha(__file__), model_calls=0, human_gold=False)
    review.write(report_path, report)
    progress(root, 'source_complete', records=records, questions=len(by_key))


def reference_origins(row, diagnostic, references):
    """Return literal/format matches, not a semantic assertion of contamination."""
    quote = diagnostic.get('evidence', {}).get('quote', '')
    student = review.locate_all(row['prediction'], quote)
    hits = []
    # Very short formulas can occur by chance. Keep them out of provenance claims.
    if len(review.normalize(quote)[0]) >= 24:
        for doc in references:
            spans = review.locate_all(doc['text'], quote)
            if spans:
                s = spans[0]
                hits.append(dict(reference_id=doc['id'], title=doc['title'],
                    source_span=s, context=doc['text'][max(0, s['start']-180):s['end']+180]))
    return dict(in_student=bool(student), reference_matches=hits)


def server_audit(root):
    assert Path('/proc').exists(), 'Formal execution replay is server-only'
    from scripts import audit_aligned_step_evaluation as execution
    from scripts import run_corrected_collection as collection
    from scripts import run_corrected_extraction as extraction
    from scripts import run_corrected_step_generation as generation
    from rule_framework import organization_followup as bounded
    out = root / 'offline_review'
    report = review.read(out / 'full_chain_audit.json')
    report['server_audit_code_sha256'] = sha(__file__)
    original_report_sha = sha(root / 'acceptance.json')
    old_progress = root / 'acceptance_progress.json'
    original_progress_sha = sha(old_progress) if old_progress.exists() else None
    review.verify_sources(root)
    final = review.read(out / 'metrics.json')
    assert sha(out / 'metrics.json') == review.read(out / 'acceptance.json')['metrics_sha256']
    for name, digest in review.read(out / 'review_frozen.json')['sha256'].items():
        assert sha(out / name) == digest
    base = root.parent.parent
    corrected = base / 'corrected'
    screen = corrected / 'source_screen'
    collect = corrected / 'collection'
    build = corrected / 'construction'
    pipe = corrected / 'pipeline'
    screen_config = review.read(screen / 'plan.json')['configuration']
    raw = report['raw_source']
    assert raw['source_sha256'] == screen_config['source_sha256']
    assert raw['inventory_sha256'] == screen_config['source_inventory_sha256']
    assert raw['references_sha256'] == sha(screen / 'references_private.json')
    question_keys = set(raw.pop('question_keys'))
    tasks = review.read(collect / 'tasks.json')
    assert Counter(t['role'] for t in tasks) == {'build': 2000, 'heldout': 702}
    assert len({t['row']['id'] for t in tasks}) == len(tasks)
    assert all(key(t['row']['question']) in question_keys for t in tasks)
    inputs = {r['id']: r for r in review.read(root / 'inputs.json')}
    assert len(inputs) == 100 and all(key(r['question']) in question_keys for r in inputs.values())
    assert all(not r.get('context') and set(r) <= {'id', 'question', 'prediction', 'context'} for r in inputs.values())
    heldout = review.read(root / 'bench_questions.json')
    build_keys = {key(t['row']['question']) for t in tasks if t['role'] == 'build'}
    overlap = sorted(build_keys & {key(r['question']) for r in heldout})
    assert not overlap
    collect_plan = collection.verify(collect)
    generation.verify(corrected / 'bench_generation')
    build_rows, rejected, source_evidence = extraction.build_rows(collect)
    assert build_rows == review.read(build / 'inputs.json')
    assert len(build_rows) == 1996 and len(rejected) == 4
    assert source_evidence == review.read(build / 'source_evidence.json')
    manifest = dict(review.read(pipe / 'source_evidence.json'))
    for name, digest in review.read(root / 'acceptance.json')['files'].items():
        if name in manifest:
            assert manifest[name] == digest, name
        manifest[name] = digest
    manifest.update(source_evidence)
    progress(root, 'frozen_evidence', total=len(manifest))
    def verify_file(item):
        name, expected = item
        assert sha(name) == expected, 'Evidence drift: ' + name
        return name
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n, _ in enumerate(pool.map(verify_file, manifest.items()), 1):
            if n % 1000 == 0:
                progress(root, 'frozen_evidence', checked=n, total=len(manifest))
    sources = review.read(root / 'sources.json')
    errors = {r['sample_id'] for r in review.read(build / 'semantic_experience.json')
              if r['status'] == 'ok' and r['semantic_audit']['status'] == 'error_found'}
    assert len(errors) == 756 and {r['id'] for r in sources} <= errors
    excluded = set(review.read(root / 'isolation.json')['excluded_source_ids'])
    assert len(sources) == 742 and errors - {r['id'] for r in sources} == excluded
    bundle = review.read(pipe / 'tree_bundle.json')
    catalog = review.read(root / 'catalog.json')
    leaves = [r for d in catalog['domains'] for t in d['topics'] for r in t['rules']]
    assert len(leaves) == 899
    assert all({s['sample_id'] for s in bundle['provenance'][r['rule_id']]} <= errors - excluded for r in leaves)
    screening = bounded.FamilyScreen(heldout)
    cases, skills = review.read(root / 'cases.json'), review.read(root / 'skills.json')
    assert (len(cases), len(skills)) == (742, 1477)
    assert all(set(d['source_ids']) <= errors - excluded for d in cases + skills)
    family_candidates = []
    for r in sources:
        found = screening.matches(r)
        if found: family_candidates.append(dict(source_id=r['id'], matches=found))
    report['source_chain'] = dict(original_membership=True, build_questions=2000,
        complete_build_answers=1996, truncated_answers=4, reserve=702, step_questions=100,
        exact_build_heldout_overlap=overlap, retained_source_family_candidates=family_candidates,
        error_sources=756, released_sources=742, excluded_sources=14,
        counts=dict(rag=742, skills=1477, tree=899), verified_frozen_files=len(manifest),
        question_only_generator=True, generator_config=collect_plan['configuration']['generator_config'],
        semantic_family_independence_proven=False)
    progress(root, 'replay_all_actual_requests')
    atomic = execution.vr.atomic
    def capture(path, value):
        path = Path(path)
        if path == root / 'acceptance.json':
            return
        if path == root / 'acceptance_progress.json':
            progress(root, 'replay_all_actual_requests', **{k:v for k,v in value.items() if k not in {'state','unix'}})
            return
        atomic(path, value)
    with patch.object(execution.vr, 'atomic', capture):
        replay = execution.audit(root)
    report['execution_replay'] = {k:v for k,v in replay.items() if k not in {'files', 'owned_service_released', 'scores'}}
    report['execution_replay']['files_count'] = len(replay['files'])
    report['execution_replay']['files_digest'] = review.value_hash(replay['files'])
    progress(root, 'diagnosis_provenance')
    evidence = review.read(out / 'evidence.json')['records']
    operator = review.read(out / 'operator_map.json')
    leaks, counts, mismatches, citations = [], Counter(), [], Counter()
    packets = {}
    for d in evidence:
        b = operator[d['review_id']]
        sid, arm = b['id'], b['arm']
        pair = sid, arm
        if pair not in packets:
            sample = root / 'formal' / arm / 'samples/C0p' / hashlib.sha256(sid.encode()).hexdigest()[:16]
            packets[pair] = review.read(sample / 'reference_packet.json')['packet']
        packet = packets[pair]
        origin = reference_origins(inputs[sid], d['original_diagnostic'], packet['references'])
        if origin['reference_matches'] and not origin['in_student']:
            counts[arm] += 1
            leaks.append(dict(review_id=d['review_id'], id=sid, arm=arm,
                diagnostic=d['original_diagnostic'], approved_student_spans=d['approved_spans'],
                frozen_disposition=d['disposition'], **origin))
        provided = {r['id'] for r in packet['references']}
        declared = set(d['original_diagnostic'].get('reference_ids', []))
        assert declared <= provided
        citations[arm + '/published'] += 1
        citations[arm + '/cited'] += bool(declared)
        if len(d['root_links']) > 1:
            mismatches.append(d['review_id'])
    report['diagnosis_provenance'] = dict(reference_only_quote_candidates=leaks,
        candidate_counts=dict(counts), citation_counts=dict(citations),
        multi_root_diagnoses=mismatches, requires_context_review=True)
    # Equal source spans in approved aliases mean the numerical overlap threshold
    # validates the manual mapping, not independently its physical correctness.
    matches = [m for g in final['profiles']['evidence_and_self_correction'].values()
               for row in g['details'] for m in row['matches']]
    report['review_scoring'] = dict(frozen_review_unchanged=True, matched_roots=len(matches),
        matches_with_identical_review_alias=sum(m['prediction_span']['start']==m['reference_span']['start'] and
            m['prediction_span']['end']==m['reference_span']['end'] for m in matches),
        requires_manual_root_direction_decision=True, independent_physical_gold=False)
    assert sha(root / 'acceptance.json') == original_report_sha
    assert (sha(old_progress) if old_progress.exists() else None) == original_progress_sha
    review.verify_sources(root)
    report.update(completed_unix=time.time(), original_records_unchanged=True,
                  scope='frozen engineering chain plus evidence-origin and scoring checks; physical claims need separate review')
    review.write(out / 'full_chain_audit.json', report)
    progress(root, 'completed', original_records_unchanged=True, model_calls=0)


def detail_audit(root):
    """Inspect additional raw stage records without repeating the full file scan."""
    assert Path('/proc').exists(), 'Run the formal evidence audit on the server'
    from scripts import run_corrected_step_generation as generation
    from scripts import run_corrected_annotations as annotations
    out = root / 'offline_review'
    report = review.read(out / 'full_chain_audit.json')
    assert report.get('original_records_unchanged') and report['execution_replay']['passed']
    metric_sha = sha(out / 'metrics.json')
    review.verify_sources(root)
    corrected = root.parent.parent / 'corrected'
    collection = corrected / 'collection'
    config = review.read(collection / 'plan.json')['configuration']['generator_config']
    generation_counts = Counter()
    for task in review.read(collection / 'tasks.json'):
        row = task['row']
        saved = review.read(collection / 'generated' / (row['id'] + '.json'))
        record = review.read(collection / saved['evidence']['file'])
        assert record['status'] == 'received'
        assert record['request'] == generation.request_body(row, config)
        raw = record['response']
        assert raw['model'] == saved['model'] == config['generator']
        assert len(raw['choices']) == 1 and raw['usage'] == saved['usage']
        choice = raw['choices'][0]
        assert choice['message']['content'] == saved['prediction']
        assert choice['finish_reason'] == saved['finish_reason']
        if saved['state'] == 'complete':
            generation.validate_import(row, config, saved, record)
        else:
            assert saved['state'] == 'incomplete' and choice['finish_reason'] == 'length'
        generation_counts[task['role'] + '/' + saved['state']] += 1
    progress(root, 'annotation_request_replay', generation_records=sum(generation_counts.values()))
    annotation_root = corrected / 'annotations'
    annotations.verify(annotation_root)
    teacher = annotations.teacher
    bindings = {r['id']: r for r in review.read(annotation_root / 'reference_bindings.json')}
    index = {r['id']: r['variants'] for r in review.read(corrected / 'source_screen/references_private.json')['references']}
    inputs = {r['id']: r for r in review.read(root / 'inputs.json')}
    annotation_rows = review.read(annotation_root / 'inputs.json')
    assert annotation_rows == review.read(annotation_root / 'annotation_inputs.json')
    assert {r['id'] for r in annotation_rows} == set(inputs)
    stage_counts = Counter()
    for row in annotation_rows:
        assert {k: row[k] for k in ('id','question','prediction','context')} == inputs[row['id']]
        binding = bindings[row['id']]
        assert binding['variants'] == index[binding['source_id']]
        assert row['answer'] == json.dumps(binding['variants'][binding['selected_variant']]['value'], ensure_ascii=False)
        for stage in teacher.STAGES:
            previous = teacher.previous_annotations(annotation_root, stage, row['id'])
            assert previous is not None
            teacher.validate_saved(annotation_root, stage, row, previous)
            stage_counts[stage] += 1
    assert stage_counts == dict.fromkeys(teacher.STAGES, 100)
    report['raw_stage_replay'] = dict(generation_records=dict(generation_counts),
        annotation_records=dict(stage_counts), annotation_model=teacher.MODEL, reasoning_effort='high',
        exact_request_response_and_export_binding=True, original_reference_values_preserved=True,
        no_verifier_results_in_annotation_requests=True, no_reference_answers_in_solver_requests=True,
        code_sha256=sha(__file__))
    progress(root, 'material_delivery_summary')
    rows = []
    summaries = {arm:Counter() for arm in ('pure','case_rag','flat_skills','rule_tree')}
    for arm, totals in summaries.items():
        for sid, source in inputs.items():
            folder = root/'formal'/arm/'samples/C0p'/hashlib.sha256(sid.encode()).hexdigest()[:16]
            saved = review.read(folder/'result.json')
            packet = review.read(folder/'reference_packet.json')['packet']
            docs, retrieval = packet['references'], packet['retrieval']
            totals['questions'] += 1
            if saved.get('failure_stage') == 'prepare':
                totals['prepare_failed'] += 1
                continue
            totals['check_attempted'] += 1
            totals['completed'] += saved['status'] == 'completed'
            totals['with_complete_material'] += bool(docs)
            totals['complete_items'] += len(docs)
            totals['ranked_items'] += len(retrieval['ranked'])
            totals['omitted_items'] += len(retrieval['omitted'])
            totals['no_material_despite_candidates'] += bool(retrieval['ranked']) and not docs
            totals['reference_tokens'] += retrieval['reference_tokens']
            if arm == 'case_rag' and sid in {
                'trainphysics_12118','cl_175_98235','trainphysics_7865','trainphysics_1899',
                'cl_129_50950','trainphysics_12582','trainphysics_12537','trainphysics_3860'}:
                rows.append(dict(id=sid, student_question=source['question'],
                    student_answer_open=source['prediction'][:500], student_answer_end=source['prediction'][-1600:],
                    diagnostics=saved['diagnostics'], provided_references=[dict(id=d['id'],title=d['title']) for d in docs],
                    reference_packet_sha256=sha(folder/'reference_packet.json')))
    report['material_delivery'] = {arm:dict(counts) for arm,counts in summaries.items()}
    report['wrong_target_contexts'] = rows
    # These are new audit observations, not edits to the frozen inclusion policy.
    # A numbered reference alone is not proof that a question is incomplete.
    dependency_pattern = re.compile(r'\b(?:part|question|problem|section)\s+[A-Za-z]?\s*[.\-]?\s*\d+(?:\.\d+)?', re.I)
    sources = review.read(root/'sources.json')
    candidates = []
    for row in sources:
        hits = [m for m in dependency_pattern.finditer(row['question'])
                if re.search(r'found|obtained|result|solution|from|given|use|consider|previous',
                             row['question'][max(0,m.start()-70):m.start()], re.I)]
        if hits:
            m = hits[0]
            candidates.append(dict(source_id=row['id'],
                excerpt=row['question'][max(0,m.start()-90):m.end()+100], status='requires_context_review'))
    sid = '9cb9022f6b09bf8a47ff9eb575b3753cb0ea937958a70c6223a63aed94e39e2d'
    source = next(r for r in sources if r['id']==sid)
    retained_cases = [d for d in review.read(root/'cases.json') if sid in d['source_ids']]
    retained_skills = [d for d in review.read(root/'skills.json') if sid in d['source_ids']]
    provenance = review.read(corrected/'pipeline/tree_bundle.json')['provenance']
    retained_rules = [r for d in review.read(root/'catalog.json')['domains'] for t in d['topics']
                      for r in t['rules'] if sid in {s['sample_id'] for s in provenance[r['rule_id']]}]
    affected_ids = {d['id'] for d in retained_cases+retained_skills}|{r['rule_id'] for r in retained_rules}
    delivered = []
    for packet_path in (root/'formal').glob('*/samples/C0p/*/reference_packet.json'):
        packet = review.read(packet_path)['packet']
        found = affected_ids & set(packet['retrieval']['provided_complete_ids'])
        if found:
            result = review.read(packet_path.with_name('result.json'))
            delivered.append(dict(arm=packet_path.parts[-5], id=result['id'], material_ids=sorted(found)))
    screening_decisions = []
    for path in (corrected/'source_screen/reviewed').glob('*.json'):
        for decision in review.read(path).get('items',[]):
            if decision['id']==sid:
                screening_decisions.append(dict(path=str(path),sha256=sha(path),decision=decision))
    report.setdefault('source_completeness_observation', {}).update(numbered_dependency_candidates=candidates,
        confirmed_example=dict(source_id=sid, question=source['question'], context=source.get('context',''),
            original_screen=screening_decisions, rag=retained_cases, skills=retained_skills, rules=retained_rules,
            formal_delivery=delivered, reviewer='Codex assistant', human_gold=False,
            finding='The question requires the result of C.4 and a magnetic-field setup that the supplied text does not contain. The RAG note itself records missing field direction, strength and charges/currents.',
            filter_limitation='Old dependency regex recognizes previous/preceding/above question/problem/part, not the numbered reference part C.4.',
            classification='statement completeness miss; not evidence of benchmark overlap',
            protocol_or_material_modified=False))
    assert sha(out/'metrics.json') == metric_sha
    review.verify_sources(root)
    report['detail_completed_unix'] = time.time()
    review.write(out/'full_chain_audit.json', report)
    progress(root,'completed',original_records_unchanged=True,model_calls=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['source','server','details'])
    p.add_argument('--root', type=Path, required=True)
    for arg in ('source','inventory','references'):
        p.add_argument('--'+arg, type=Path)
    p.add_argument('--await-release', action='store_true')
    args = p.parse_args()
    out = args.root / 'offline_review'
    with (out/'full_chain.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.await_release:
            progress(args.root, 'awaiting_reconnect')
            deadline = time.monotonic()+60
            while not (out/'ALLOW_FULL_AUDIT').exists():
                if time.monotonic()>deadline: raise TimeoutError('Reconnect missing')
                time.sleep(.5)
            (out/'ALLOW_FULL_AUDIT').unlink()
        try:
            if args.mode == 'source':
                source_audit(args.root,args.source,args.inventory,args.references)
            elif args.mode == 'server':
                server_audit(args.root)
            else:
                detail_audit(args.root)
        except Exception as exc:
            progress(args.root,'failed',error=type(exc).__name__+': '+str(exc))
            raise


if __name__ == '__main__':
    main()
