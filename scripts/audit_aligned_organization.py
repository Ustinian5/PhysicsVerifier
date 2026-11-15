"""Offline readiness audit of the restored old99 method; never runs inference.

Keep generated material, labels, and old scores immutable. The only outputs are
readiness_audit.json and a small readiness_progress.json in the supplied run.
Lexical family hits are review candidates, not automatic physical judgments.
"""
import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from scripts import run_aligned_organization as a
from scripts import reorganize_qwen30b_expansion as navigation

vr, require = a.vr, a.require


class Evidence:
    def __init__(self):
        self.files = {}

    def read_raw(self, path, expected=None):
        path = Path(path)
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        require(expected is None or actual == expected, 'Evidence changed: ' + str(path))
        require(str(path) not in self.files or self.files[str(path)] == actual,
                'Evidence changed during audit: ' + str(path))
        self.files[str(path)] = actual
        return data

    def read(self, path, expected=None):
        return json.loads(self.read_raw(path, expected))


def request_contract(record, messages, config, fingerprint):
    expected = dict(model=config['model'], messages=messages, temperature=0, top_p=1,
                    max_tokens=config['max_tokens'], stream=False, return_token_ids=True,
                    reasoning_effort='medium')
    require(record['request'] == expected, 'Model-visible request differs from the old method')
    require(record.get('endpoint') == '/chat/completions' and
            record.get('http_status') == 200 and not record.get('error'), 'Transport failure')
    require(record.get('client_metadata') == dict(build_stage='extract_skill', fingerprint=fingerprint),
            'Request/source fingerprint differs')
    raw = record['response']
    require(raw.get('model') == config['model'] and raw.get('id') and
            len(raw.get('choices', [])) == 1, 'Response identity differs')
    require(raw['choices'][0]['finish_reason'] in {'stop', 'length'}, 'Unknown finish reason')
    require(raw['usage']['completion_tokens'] <= config['max_tokens'], 'Output budget exceeded')
    require(len(raw['prompt_token_ids']) + config['max_tokens'] <= a.base.p.BUILD_LIMITS['context_window'],
            'Input context exceeds old limit')
    return raw


def verify_content_failure(record, reason, config):
    finish = record['response']['choices'][0]['finish_reason']
    if reason.startswith('length_truncated'):
        require(finish == 'length', 'Recorded truncation has another terminal state')
        return
    require(finish == 'stop', 'Non-truncation failure has another terminal state')
    if reason.startswith('malformed_json'):
        decoded = navigation.decode_legacy(record, config)
        require(not isinstance(decoded, dict) or '_unparsed_response_text' in decoded,
                'Recorded malformed content now parses under the same decoder')
    elif reason.startswith('missing_final'):
        try:
            navigation.decode_legacy(record, config)
        except ValueError as exc:
            require(str(exc) == 'Missing final message', 'Different decode failure')
        else:
            raise ValueError('Recorded missing final is not reproduced')
    else:
        raise ValueError('Unreviewed failure kind: ' + reason)


def stage_audit(root, stage, rows, cases, plan, tokenizer, evidence, progress):
    config, binding = plan['configuration']['model'], plan['configuration_sha256']
    folder = root / stage
    sealed = evidence.read(folder / 'request_index.json')
    require(vr.digest(sealed['value']) == sealed['sha256'], 'Broken request ledger')
    index = sealed['value']
    require(index['binding'] == binding, 'Wrong ledger binding')
    files, consumed, outcomes, counts, failures = index['files'], set(), [], Counter(), []
    for i, row in enumerate(rows):
        def read_bound(path):
            require(str(path) in files, 'Unindexed original: ' + str(path))
            consumed.add(str(path))
            return evidence.read(path, files[str(path)])
        saved = read_bound(a.base.old.outcome_path(folder, row))
        require(saved['configuration_sha256'] == binding and saved['source_sha256'] == vr.digest(row),
                'Outcome/input binding differs')
        outcome = saved['outcome']
        require(outcome['id'] == row['id'] and outcome['status'] in {'collected', 'content_failure'},
                'Unfinished or different source')
        if stage == 'experience':
            payload, messages = {'row': row}, a.base.p.build_messages(row)
            validate = a.base.p.validate_experience
        else:
            case = cases[row['id']]
            require(saved['case_sha256'] == vr.digest(case), 'Skill uses another case')
            payload, messages = {'case': case}, a.bounded.skill_messages(case)
            validate = a.bounded.validate_skills
        fingerprint = vr.digest(['extract_skill', row['id'], payload, config['model'], config])
        job = folder / 'arm_extract_skill' / vr.digest(row['id'])[:24]
        paths = sorted(job.glob('request_*.json'))
        require(len(paths) == 1, 'Expected the observed single-attempt cohort; inspect retries explicitly')
        record = read_bound(paths[0])
        raw = request_contract(record, messages, config, fingerprint)
        a.validate_effective_input(raw, tokenizer, a.DATE)
        counts['requests'] += 1
        counts['input_tokens'] += raw['usage']['prompt_tokens']
        counts['output_tokens'] += raw['usage']['completion_tokens']
        counts[outcome['status']] += 1
        counts['finish_' + raw['choices'][0]['finish_reason']] += 1
        if outcome['status'] == 'collected':
            require(raw['choices'][0]['finish_reason'] == 'stop', 'Truncation counted as success')
            decoded = navigation.decode_legacy(record, config)
            validate(decoded)
            receipt = read_bound(job / 'result.json')
            require(decoded == receipt['value'] == outcome['value'], 'Raw response/material mismatch')
            require(receipt['fingerprint'] == fingerprint and receipt['request_file'] == paths[0].name and
                    receipt['request_sha256'] == files[str(paths[0])], 'Decoded receipt drift')
            if stage == 'experience': counts[decoded['answer_status']] += 1
        else:
            require(outcome.get('reason'), 'Unexplained content failure')
            verify_content_failure(record, outcome['reason'], config)
            failures.append(dict(source_id=row['id'], reason=outcome['reason'], request=str(paths[0])))
        outcomes.append(dict(outcome, configuration_sha256=binding, source_sha256=vr.digest(row)))
        if (i + 1) % 100 == 0:
            progress(stage, checked=i + 1, total=len(rows))
    for name, sha in files.items():
        if name not in consumed: evidence.read(name, sha)
    actual = {str(p) for p in folder.glob('arm_extract_skill/*/request_*.json')}
    require(actual == {p for p in files if Path(p).name.startswith('request_')}, 'Unaccounted request')
    budget = evidence.read(folder / 'budget.json')
    require(budget['requests_started'] == counts['requests'] and
            budget['accounted_input_tokens'] == counts['input_tokens'] and
            budget['accounted_output_tokens'] == counts['output_tokens'] and
            budget['in_flight'] == budget['requests_without_usage'] == 0, 'Budget ledger mismatch')
    return outcomes, dict(counts, files_verified=len(files), failures=failures)


def provenance_audit(root, plan, rows, cases, skills, memory, evidence):
    current = Path(plan['configuration']['current'])
    parent = evidence.read(current / 'plan.json')['configuration']
    original = Path(parent['root'])
    split = evidence.read(original / 'source_screen/split.json')
    construction = evidence.read(original / 'construction/inputs.json')
    source_by_id = {r['id']: a.base.p.source_payload(r) for r in construction}
    build = {r['id']: r for r in split['build']}
    reserve = {r['id']: r for r in split['reserve']}
    require(len(build) == 2000 and len(reserve) == 702 and not set(build) & set(reserve), 'Pool sizes/IDs differ')
    require(len(construction) == 1996 and set(source_by_id) <= set(build), 'Construction source changed')
    require(all(r == source_by_id[r['id']] for r in rows), 'Case source differs from original 30B answer')
    require(all(r['question'] == build[r['id']]['question'] for r in construction), 'Build question changed')
    require({s for d in cases for s in d['source_ids']} == {r['id'] for r in rows}, 'RAG source cohort changed')
    require({s for d in skills for s in d['source_ids']} <= set(source_by_id), 'Skill source outside build pool')
    bundle = evidence.read(original / 'pipeline/tree_bundle.json')
    provenance = bundle['provenance']
    require(set(memory.corpus.leaves) <= set(provenance), 'Missing tree provenance')
    tree_sources = {v['sample_id'] for rid in memory.corpus.leaves for v in provenance[rid]}
    require(tree_sources <= {r['id'] for r in rows}, 'Tree source outside common error cohort')
    inputs = evidence.read(current / 'step_inputs.json')
    generated = evidence.read(original / 'bench_generation/inputs.json')
    require(inputs == generated and len(inputs) == 100, 'Bench does not use the frozen new 30B answers')
    refs = evidence.read(current / 'step_reference.json')
    require([a.base.p.source_payload(r) for r in refs] == inputs, 'Step labels refer to different answers')
    screen_plan = evidence.read(original / 'source_screen/plan.json')['configuration']
    label_release = evidence.read(current / 'step_label_release.json')
    return dict(build_count=len(build), reserve_count=len(reserve), complete_30b_answers=len(construction),
                source_count=len(rows), tree_source_count=len(tree_sources), source_inputs_match=True,
                step_inputs_match=True, source_sha256=screen_plan['source_sha256'],
                original_database_rescanned=False, label_release=label_release), original


def isolation_audit(root, current, rows, cases, skills, memory, evidence, progress):
    inputs = evidence.read(current / 'step_inputs.json')
    pilots = evidence.read(current / 'pilot_inputs.json')
    final = evidence.read(current / 'final_inputs.json')
    heldout = [dict(id=kind + ':' + r['id'], question=r['question'])
               for kind, cohort in [('step', inputs), ('pilot', pilots), ('final', final)] for r in cohort]
    external = evidence.read(root.parent / 'hipho_overlap_audit.json')
    external_rows = external['question_references']
    require(len(external_rows) == 128 and external['benchmark_sha256'] ==
            '6ca95621b3b37490fbf4d4092d82e0d4dd471b4d6c9fe6f3bf3b957f7e802a59', 'External heldout identity differs')
    raw = evidence.read_raw(external['benchmark_path'], external['benchmark_sha256'])
    original_rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    require(external_rows == [{k: row.get(k, '') for k in ('id', 'exam', 'question', 'context')}
                             for row in original_rows], 'External heldout reference rendering differs')
    for row in external_rows:
        heldout.append(dict(id='hipho:' + row['id'], question=row['question']))
        if row.get('context'):
            heldout.append(dict(id='hipho_context:' + row['id'], question=row['context'] + '\n' + row['question']))
    screen = a.bounded.FamilyScreen(heldout)
    case_by_id = {r['source_ids'][0]: r for r in cases}
    findings = []
    for i, row in enumerate(rows):
        matches = screen.matches(dict(row, prediction=case_by_id[row['id']]['text']))
        if matches: findings.append(dict(source_id=row['id'], matches=matches))
        if (i + 1) % 100 == 0: progress('family_screen', checked=i + 1, total=len(rows))
    derived = []
    for doc in skills + list(memory.memory.docs.values()):
        matches = screen.matches(dict(id=doc['id'], question='', prediction=doc['title'] + '\n' + doc['text']))
        if matches: derived.append(dict(item_id=doc['id'], kind=doc['kind'], matches=matches))
    previous = evidence.read(current / 'evaluation_audit.json')
    known = previous['family_split_counterexample']
    bad = known['build_id']
    exposure = dict(build_source=bad, test_source=known['test_id'], source_retained=bad in case_by_id,
        rag_ids=[r['id'] for r in cases if bad in r['source_ids']],
        skill_ids=[r['id'] for r in skills if bad in r['source_ids']],
        confirmed_same_family_rule=known['rule_id'],
        known_rule_retained=known['rule_id'] in memory.corpus.leaves,
        human_gold=False, official_data_modified=False)
    # Quantify affected derivations without pruning any source or rebuilding a tree.
    affected = {x['source_id'] for x in findings} | {bad}
    original = Path(evidence.read(current / 'plan.json')['configuration']['root'])
    bundle = evidence.read(original / 'pipeline/tree_bundle.json')
    impact = dict(provisional_only=True, requires_review=True, source_ids=sorted(affected),
        rag_ids=[d['id'] for d in cases if affected & set(d['source_ids'])],
        skill_ids=[d['id'] for d in skills if affected & set(d['source_ids'])],
        rule_ids=[rid for rid in memory.corpus.leaves
                  if affected & {s['sample_id'] for s in bundle['provenance'][rid]}])
    return dict(previous_v15_policy=a.bounded.ISOLATION_POLICY,
                old_v15_preparation_screen_not_called_by_alignment_adapter=True,
                heldout_counts=dict(step=len(inputs), pilot=len(pilots), final=len(final), hipho=len(external_rows)),
                lexical_source_candidates=findings, lexical_derived_candidates=derived,
                known_family_exposure=exposure, provisional_affected_materials=impact,
                classification='Lexical candidates require review; the prior confirmed family issue remains a separate finding.')


def preflight(root, current, cases, skills, memory, tokenizer, evidence):
    count = lambda m: len(tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True))
    inputs = evidence.read(current / 'step_inputs.json')
    # A legal maximum-length artificial query stresses the CHECK input format;
    # it is not a generated query or a scientific retrieval result.
    queries = dict(queries=['检' * 300] * 3)
    prepare_counts = [count(a.base.p.query_messages(row)) for row in inputs]
    check_counts = [count(a.base.p.continuation_messages(row, queries, [])) for row in inputs]
    require(max(prepare_counts) + a.base.p.QUERY_MAX_TOKENS <= 131072 and
            max(check_counts) + 8192 + 16384 <= 131072, 'A formal input exceeds the frozen context cap')
    memories = dict(pure=None, case_rag=a.base.p.FlatMemory(cases),
                    flat_skills=a.base.p.FlatMemory(skills), rule_tree=memory)
    packets = []
    for row in evidence.read(root / 'pilot_inputs.json'):
        prefix = evidence.read(root / 'queries' / vr.digest(row['id'])[:24] / 'result.json')
        require(prefix['input_sha256'] == vr.digest(row), 'Pilot query belongs to another input')
        for arm, mem in memories.items():
            packet = a.base.p.reference_packet(row, arm, prefix['queries'], mem, count)
            require(len(packet['references']) <= 6 and packet['retrieval']['reference_tokens'] <= 8192,
                    'Full-material packet exceeds old limits')
            if arm != 'pure':
                flat = memory.memory if arm == 'rule_tree' else mem
                require(all(item == flat.view(item['id']) for item in packet['references']),
                        'A reference was truncated or transformed')
            packets.append(dict(id=row['id'], arm=arm,
                supplied_items=len(packet['references']), reference_tokens=packet['retrieval']['reference_tokens'],
                omitted_whole_items=len(packet['retrieval']['omitted'])))
    return dict(formal_input_count=len(inputs), prepare_max_input_tokens=max(prepare_counts),
                stress_check_max_input_tokens=max(check_counts), additional_reference_cap=8192,
                reserved_output_cap=16384, context_cap=131072,
                pilot_packets_with_complete_current_materials=packets, new_model_calls=0,
                formal_queries_generated=False, scientific_predictions_generated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = args.output.resolve()
    started = time.time(); evidence = Evidence()
    def progress(stage, **fields):
        vr.atomic(root / 'readiness_progress.json', dict(stage=stage, unix=time.time(), **fields))
    with ExitStack() as stack:
        for path in [root / 'readiness_audit.lock', root / 'active.lock',
                     root.parent / 'organization_service_18083.lock']:
            stack.enter_context(a.base.r.run_lock(path))
        plan = a.verify(root)
        require(vr.read(root / 'execution_status.json')['state'] == 'build_complete' and
                (root / 'runner.exit_code').read_text().strip() == '0', 'Build is not complete')
        require(not (root / 'runner.pid').exists(), 'Runner is still present')
        evidence.read(root / 'plan.json')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.base.r.EXPECTED_MODEL, local_files_only=True,
                                                  trust_remote_code=False)
        # Same decoder stack, without the generation runner's evidence-writing wrapper.
        stack.enter_context(a.base.old.comparison.execution.configured())
        stack.enter_context(patch.object(a.recovery, 'ORIGINAL_DECODE', a.historical_transport().decode_response))
        stack.enter_context(patch.object(a.base.runtime.transport, 'decode_response', a.base.decode_response))
        def no_generation(*args, **kwargs):
            raise AssertionError('This audit must never issue a model request')
        stack.enter_context(patch('httpx.Client.post', no_generation))
        rows = evidence.read(root / 'sources.json')
        require(len(rows) == 756, 'Frozen source count changed')
        cases = evidence.read(root / 'materials/cases.json')
        skills = evidence.read(root / 'materials/skills.json')
        case_map = {d['source_ids'][0]: d for d in cases}
        progress('checking_requests')
        outcomes, case_info = stage_audit(root, 'experience', rows, case_map, plan, tokenizer, evidence, progress)
        products = a.base.p.materialize(rows, outcomes, plan['configuration_sha256'])
        require(products['cases'] == cases and products['summary'] == evidence.read(root / 'materials/case_summary.json'),
                'RAG rendering differs from the old method')
        outcomes, skill_info = stage_audit(root, 'skills', rows, case_map, plan, tokenizer, evidence, progress)
        docs, summary = a.bounded.materialize_skills(rows, outcomes)
        require(docs == skills and summary == evidence.read(root / 'materials/skill_summary.json'), 'Skill rendering differs')
        current = Path(plan['configuration']['current'])
        memory = a.base.p.TreeMemory(evidence.read(current / 'materials/catalog.json'))
        require(len(memory.corpus.leaves) == 915, 'Tree leaf count changed')
        a.base.p.FlatMemory(cases); a.base.p.FlatMemory(skills)
        source, original = provenance_audit(root, plan, rows, cases, skills, memory, evidence)
        progress('isolation')
        isolation = isolation_audit(root, current, rows, cases, skills, memory, evidence, progress)
        progress('full_material_input_preflight')
        input_check = preflight(root, current, cases, skills, memory, tokenizer, evidence)
        pilot = evidence.read(root / 'runtime_protocol_audit.json')
        relocation = evidence.read(root / 'pilot_lifecycle/manifest.json')['files']
        for name, sha in pilot['files'].items():
            evidence.read(relocation.get(name, {}).get('archived_path', name), sha)
        release = evidence.read(root / 'service/released_after_build.json')
        # The old wrapper loads its own historical 99 answers/labels. It cannot
        # be used as a drop-in scorer for the corrected 100-question bench.
        tree = ast.parse((a.ROOT / 'scripts/run_aligned_organization.py').read_text())
        funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        formal = dict(adapter_has_formal_or_score_function=bool(funcs & {'formal', 'score'}),
                      configured_scoring_zero_offsets=plan['configuration']['scoring_preserve_zero_offsets'],
                      old_bench_wrapper_hardcodes_99_and_old_labels=True,
                      formal_output_present=any(root.glob('formal*')))
        blockers = ['Step reference quote/ambiguity release is incomplete.',
                    'Known source-family conflict and incomplete statements remain unresolved.',
                    'Old v15 conservative family-screen/pruning is absent from the alignment adapter.',
                    'The adapter has no corrected-100 formal evaluation/scoring entry point.',
                    'The 600-case final-answer task has no same-task old99 baseline and has unresolved label issues.']
        report = dict(version='aligned_old99_readiness_v1', audited_unix=time.time(),
            configuration_sha256=plan['configuration_sha256'], new_model_calls=0,
            materials_and_recorded_execution_verified=True, clean_formal_evaluation_ready=False,
            cases=case_info, skills=skill_info, material_counts=dict(rag=len(cases), skills=len(skills), tree=915),
            source=source, isolation=isolation, formal_entry=formal, input_preflight=input_check,
            pilot=dict(checks=pilot['checks'], requests=pilot['actual_requests'], files_reverified=len(pilot['files'])),
            service_release=release, blockers=blockers, human_gold=False,
            physical_content_quality_certified=False, files=evidence.files,
            script_sha256=vr.sha(Path(__file__)), elapsed_seconds=time.time() - started)
        vr.atomic(root / 'readiness_audit.json', report)
        progress('complete', materials_verified=True, formal_ready=False)
        print(json.dumps(dict(material_counts=report['material_counts'],
              case_failures=case_info['content_failure'], skill_failures=skill_info['content_failure'],
              lexical_source_candidates=len(isolation['lexical_source_candidates']),
              lexical_derived_candidates=len(isolation['lexical_derived_candidates']),
              known_family_exposure=isolation['known_family_exposure'], formal_ready=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
