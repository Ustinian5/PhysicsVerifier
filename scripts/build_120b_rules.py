"""Prepare/run an isolated all-120B build from released or attributed answers.

No historical catalog is overwritten. prepare/verify are offline. run must be a
detached server process; confirm-detached is issued after reconnecting via SSH.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import uuid

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rule_framework import oss120b_construction as p
from rule_framework import released_corpus
from scripts import run_gpt_oss_selfcheck as r
from scripts import run_continuous_context_v2 as existing
from scripts import resume_continuous_context_v2 as harmony
from scripts.run_retrieval_comparison import (BatchBudget, Transport,
    retryable_transport_failure, retry_after_deadline)

SOURCES = ["rule_framework/oss120b_construction.py", "scripts/build_120b_rules.py",
           "scripts/run_gpt_oss_selfcheck.py", "scripts/run_retrieval_comparison.py",
           "scripts/gpt_oss_harmony_transport.py", "scripts/resume_continuous_context_v2.py",
           "scripts/response_stream.py", "scripts/validate_unified_catalog_structure.py",
           "core/rule_catalog_retrieval.py", "scripts/run_continuous_context_v2.py",
           "rule_framework/released_corpus.py", "scripts/rebuild_qwen30b_corpus.py"]
BENCHMARK_FILES = {
    # Label-free stems from the additional historical server eval/validation cohorts.
    "results/oss120b_rule_build/server_bench_questions.json": 1775,
    "results/bench_reannotation_99_20261011/inputs.json": 99,
    "results/retrieval_comparison_50_20260920/frozen50/inputs.json": 50,
    "data/derived/catalog_compare_seed20260508_test200_20260901/annotated_chain/error_eval_dataset_100.json": 100,
    "data/derived/catalog_compare_seed20260508_test200_20260901/qa_chain/combined_language_main_test.json": 200,
    "data/derived/catalog_compare_seed20260508_test200_20260901/qa_chain/combined_language_val_ablation.json": 80,
}
PREPARED_FILES = {"inputs.json", "isolation.json", "bench_questions.json", "plan.json", "active.lock",
                  "source_audits.json", "source_release.json"}
RELEASE_FILES = {"inputs.json", "isolation.json", "bench_questions.json", "plan.json",
                 "catalog.json", "provenance.json", "validation.json", "counts.json", "runtime.json"}
TOPIC_RECOVERY = "topic_assignment_recovery.json"
TOPIC_POLICY = "exact_topic_assignment_duplicates_only_v1"
REJECTION_RECOVERY = "distill_rejection_recovery.json"
REJECTION_POLICY = "duplicate_rejections_preserve_all_reasons_v1"
RECOVERIES = {
    'assign_topics': (TOPIC_RECOVERY, TOPIC_POLICY, 'Missing/duplicate imported topic member'),
    'distill': (REJECTION_RECOVERY, REJECTION_POLICY, 'All error IDs must occur exactly once in support/rejected'),
    'group_errors': ('group_id_recovery.json', 'unique_existing_group_hex_prefix_v1', 'Unknown group ID'),
    'group_duplicates': ('group_member_recovery.json', 'exact_group_assignment_duplicates_v1', 'Missing or duplicate group member'),
    'short_ids': ('short_id_protocol.json', 'request_local_group_ids_v1', 'Not an eligible group ID prefix'),
    'short_distill': ('short_distill_protocol.json', 'request_local_distillation_ids_v1', 'Invented source error'),
    'short_distill_extras': ('unknown_rejection_recovery.json', 'exclude_unknown_rejections_only_v1', 'All error IDs must occur exactly once in support/rejected'),
    'error_type_projection': ('error_type_projection.json', 'valid_type_list_first_item_v1', 'Invalid error type'),
    'json_escapes': ('json_escape_recovery.json', 'literal_invalid_json_escape_v1', 'Invalid \\escape: line 11 column 50 (char 300)'),
}
SHORT_ID_INSTRUCTION = ('\n本请求只使用局部编号：items.id为I1、I2等，已有groups.id为G1、G2等。'
    'assignments.item_id必须逐项原样引用I编号，group_id只能引用已有G编号或本次new_groups.key。'
    '新组key按new1、new2顺序命名。每个I编号恰好一次，不输出任何哈希编号；不改变归组判断标准。')
SHORT_DISTILL_INSTRUCTION = ('\n来源引用只使用本请求errors.id中的E1、E2等局部编号，不引用题号或编造哈希。'
    '必须把全部E编号划分为互不重叠的集合：每条规则的support是一组，rejected是其余编号。'
    '每个E编号在整个输出中恰好出现一次，不得被两条规则共用，也不得同时支持和拒绝。'
    '先确定不重叠的来源划分，再写规则；重叠机制只能保留一条由这些来源共同支持的规则。'
    '每条规则仍必须有至少两道不同题干模板的真实支持；无法满足时拒绝，不能复制来源凑数。'
    '不要改变原物理依据或为增加规则数量而合并不同机制。')


def short_distill_payload(payload):
    mapping = {f'E{i+1}': e['id'] for i, e in enumerate(payload['errors'])}
    p.require(len(set(mapping.values())) == len(mapping), 'Duplicate source error IDs')
    return dict(payload, errors=[dict(e, id=k) for k, e in zip(mapping, payload['errors'])]), {'errors': mapping}


def normalized_unknown_rejections(value, payload):
    """Quarantine nonexistent rejected sources; real coverage/support still exact."""
    p.require(set(value) == {'rules', 'rejected'} and isinstance(value['rejected'], list),
              'Unexpected distillation schema')
    expected = {e['id'] for e in payload['errors']}
    retained, excluded = [], []
    for index, entry in enumerate(value['rejected']):
        p.require(set(entry) == {'error_id', 'reason'}, 'Unexpected rejection fields')
        p.text_fields(entry, ('error_id', 'reason'))
        if entry['error_id'] in expected:
            retained.append(entry)
        else:
            excluded.append({'index': index, 'record': entry})
    normalized, duplicates = normalized_rejections(dict(value, rejected=retained), payload)
    return normalized, {'excluded_unknown_rejections': excluded, 'duplicate_rejection_indices': duplicates}


def project_error_types(value):
    """Project explicit valid pipe-separated metadata to its first listed label."""
    result = json.loads(json.dumps(value))
    p.require(isinstance(result.get('rules'), list), 'Bad distillation schema')
    changes = []
    for index, rule in enumerate(result['rules']):
        label = rule.get('error_type')
        p.require(isinstance(label, str), 'Invalid error type')
        if label in p.ERROR_TYPES:
            continue
        parts = [s.strip() for s in label.split('|')]
        p.require(len(parts) >= 2 and all(s in p.ERROR_TYPES for s in parts), 'Invalid error type')
        rule['error_type'] = parts[0]
        changes.append({'rule_index': index, 'original': label, 'labels': parts, 'projected': parts[0]})
    return result, changes


def restore_short_distill(value, payload, allow_unknown_rejections=False):
    local, mapping = short_distill_payload(payload)
    normalizer = normalized_unknown_rejections if allow_unknown_rejections else normalized_rejections
    normalized, removed = normalizer(value, local)
    result = dict(normalized, rules=[dict(rule, support=[dict(s, error_id=mapping['errors'][s['error_id']])
        for s in rule['support']]) for rule in normalized['rules']],
        rejected=[dict(e, error_id=mapping['errors'][e['error_id']]) for e in normalized['rejected']])
    p.distilled_rules(result, payload['errors'], payload['errors'][0]['topic_key'])
    return result, mapping, removed


def short_group_payload(payload):
    items = {f'I{i+1}': x['id'] for i, x in enumerate(payload['items'])}
    groups = {f'G{i+1}': x['id'] for i, x in enumerate(payload['groups'])}
    p.require(len(set(items.values())) == len(items) and len(set(groups.values())) == len(groups),
              'Duplicate persistent grouping IDs')
    data = dict(payload, items=[dict(x, id=k) for k, x in zip(items, payload['items'])],
                groups=[dict(x, id=k) for k, x in zip(groups, payload['groups'])])
    return data, {'items': items, 'groups': groups}


def restore_short_groups(value, payload):
    local, mapping = short_group_payload(payload)
    normalized, changes = normalized_group_duplicates(value, local)
    new_keys = {g['key'] for g in normalized['new_groups']}
    p.require(all(k == f'new{i+1}' for i, k in enumerate(g['key'] for g in normalized['new_groups'])),
              'Unexpected new short group key')
    result = dict(normalized, assignments=[{'item_id': mapping['items'][a['item_id']],
        'group_id': a['group_id'] if a['group_id'] in new_keys else mapping['groups'][a['group_id']]}
        for a in normalized['assignments']])
    normalized_groups(result, payload)
    return result, mapping, changes


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    if path.exists():
        p.require(read(path) == value, "Existing frozen artifact differs: " + str(path))
    else:
        r.write_json(path, value)


def source_hashes():
    return {s: r.sha256_file(ROOT / s) for s in SOURCES}


def normalized_topics(value, payload):
    """A lossless view of an ID->Topic map; never resolve conflicts or omissions."""
    p.require(set(value) == {'assignments'} and isinstance(value['assignments'], list),
              'Topic normalization requires the exact assignment schema')
    allowed = p.validate_taxonomy(payload['taxonomy'])
    expected = [x['id'] for x in payload['items']]
    p.require(len(expected) == len(set(expected)), 'Duplicate input topic member')
    unique, removed = {}, []
    for index, item in enumerate(value['assignments']):
        p.require(isinstance(item, dict) and set(item) == {'item_id', 'topic_key'},
                  'Unexpected topic assignment fields')
        p.text_fields(item, ('item_id', 'topic_key'))
        p.require(item['item_id'] in expected and item['topic_key'] in allowed,
                  'Unknown topic/member; normalization cannot invent a mapping')
        prior = unique.get(item['item_id'])
        if prior is not None:
            p.require(item == prior, 'Conflicting topic assignments cannot be normalized')
            removed.append(index)
        else:
            unique[item['item_id']] = item
    p.require(set(unique) == set(expected), 'Missing topic member cannot be normalized')
    return {'assignments': list(unique.values())}, removed


def normalized_rejections(value, payload):
    """Merge repeated rejection IDs, retaining every reason and every rule byte."""
    p.require(set(value) == {'rules', 'rejected'} and isinstance(value['rules'], list)
              and isinstance(value['rejected'], list), 'Unexpected distillation schema')
    reasons, removed = {}, []
    for index, item in enumerate(value['rejected']):
        p.require(isinstance(item, dict) and set(item) == {'error_id', 'reason'},
                  'Unexpected rejection fields')
        p.text_fields(item, ('error_id', 'reason'))
        if item['error_id'] in reasons:
            removed.append(index)
        reasons.setdefault(item['error_id'], [])
        if item['reason'] not in reasons[item['error_id']]:
            reasons[item['error_id']].append(item['reason'])
    normalized = {'rules': value['rules'], 'rejected': [
        {'error_id': eid, 'reason': '\n'.join(texts)} for eid, texts in reasons.items()]}
    # Support/rejection conflicts, missing IDs, reused support and two-question
    # violations still fail. No alternative output is sampled to fix them.
    p.distilled_rules(normalized, payload['errors'], payload['errors'][0]['topic_key'])
    return normalized, removed


def normalized_groups(value, payload):
    """Resolve only unique existing hash-ID prefixes; never infer a mechanism."""
    p.require(set(value) == {'new_groups', 'assignments'} and
              isinstance(value['new_groups'], list) and isinstance(value['assignments'], list),
              'Unexpected grouping schema')
    existing_ids = [g['id'] for g in payload['groups']]
    new = {}
    for g in value['new_groups']:
        p.text_fields(g, ('key', 'name', 'summary'))
        p.require(g['key'] not in new and g['key'] not in existing_ids, 'Duplicate/new group collision')
        new[g['key']] = g
    assignments, expansions = [], []
    for index, a in enumerate(value['assignments']):
        p.require(set(a) == {'item_id', 'group_id'}, 'Unexpected group assignment fields')
        p.text_fields(a, ('item_id', 'group_id'))
        gid = a['group_id']
        if gid not in existing_ids and gid not in new:
            suffix = gid[2:]
            p.require(gid.startswith('g_') and 7 <= len(suffix) < 20 and
                      all(c in '0123456789abcdef' for c in suffix), 'Not an eligible group ID prefix')
            matches = [g for g in existing_ids if g.startswith(gid)]
            p.require(len(matches) == 1, 'Unknown or ambiguous group ID prefix')
            expansions.append({'index': index, 'original': gid, 'resolved': matches[0]})
            gid = matches[0]
        assignments.append(dict(a, group_id=gid))
    p.require(Counter(a['item_id'] for a in assignments) == Counter(x['id'] for x in payload['items']),
              'Missing or duplicate group member')
    p.require(all(any(a['group_id'] == k for a in assignments) for k in new), 'Empty new group')
    return {'new_groups': value['new_groups'], 'assignments': assignments}, expansions


def normalized_group_duplicates(value, payload):
    """Remove only byte-equivalent assignment objects, keeping first occurrence."""
    p.require(isinstance(value.get('assignments'), list), 'Bad grouping schema')
    assignments, removed = [], []
    for index, a in enumerate(value['assignments']):
        if a in assignments:
            removed.append(index)
        else:
            assignments.append(a)
    normalized, expansions = normalized_groups(dict(value, assignments=assignments), payload)
    if not removed:
        return normalized, []  # Prefix-only cases retain their existing recovery receipt.
    return normalized, [{'removed_indices': removed, 'id_expansions': expansions}]


def normalized_view(stage, value, payload):
    return {'assign_topics': normalized_topics, 'distill': normalized_rejections,
            'group_errors': normalized_groups, 'group_duplicates': normalized_group_duplicates}[stage](value, payload)


def verify_sources(output, plan, proposed=None):
    """Keep the original plan immutable; narrowly bind an explicit adapter update."""
    config = plan['configuration']
    current = source_hashes()
    updates = [(output / name, read(output / name)) for name, _, _ in RECOVERIES.values()
               if (output / name).exists()]
    if proposed is not None:
        p.require(all(u['policy'] != proposed['policy'] for _, u in updates), 'Recovery already exists')
        updates.append((None, proposed))
    if not updates:
        p.require(config['source_hashes'] == current, 'Construction source changed since prepare')
        return
    previous, previous_sha = config['source_hashes'], None
    policies = {policy: error for _, policy, error in RECOVERIES.values()}
    for recovery_path, update in updates:
        p.require(config.get('corpus_release') and update.get('policy') in policies,
                  'Unsupported construction recovery')
        p.require(update['configuration_sha256'] == plan['configuration_sha256'] and
                  update['plan_sha256'] == r.sha256_file(output / 'plan.json') and
                  update['original_sources'] == previous and
                  update.get('previous_recovery_sha256') == previous_sha, 'Recovery source/plan binding changed')
        replacement = update['replacement_sources']
        changed = {n for n in replacement if replacement[n] != previous.get(n)}
        p.require(set(replacement) == set(previous) and changed == {'scripts/build_120b_rules.py'},
                  'Recovery may only update the construction request adapter')
        p.require(update['stopped_status'].get('error') == policies[update['policy']],
                  'Recovery lacks the original failure')
        for name, sha in update['prefix_files'].items():
            path = (output / name).resolve()
            p.require(path.is_relative_to(output.resolve()) and r.sha256_file(path) == sha,
                      'Frozen recovery prefix changed: ' + name)
        previous = replacement
        previous_sha = r.sha256_file(recovery_path) if recovery_path else None
    p.require(previous == current, 'Recovery source/plan binding changed')


def recover_structure(output, stage):
    """Register this engineering change offline; do not edit any raw/cache value."""
    with r.run_lock(output / 'active.lock'):
        filename, policy, error = RECOVERIES[stage]
        if (output / filename).exists():
            verify(output)
            print('Structural recovery already verified; zero model calls.'); return
        p.require(not (output / 'release_manifest.json').exists(), 'Completed builds are immutable')
        plan = read(output / 'plan.json'); config = plan['configuration']
        status = read(output / 'execution_status.json')
        p.require(config.get('corpus_release') and status.get('state') == 'stopped' and
                  status.get('error') == error, 'Expected a stopped released-corpus structural failure')
        budget = read(output / 'budget.json')
        p.require(budget.get('in_flight') == 0, 'Requests are still in flight')
        affected = []
        request_stage = ('distill' if stage in {'short_distill', 'short_distill_extras', 'error_type_projection'} else
                         'group_errors' if stage in {'group_duplicates', 'short_ids'} else stage)
        suffix = '_shortids' if stage in {'short_distill_extras', 'error_type_projection'} else ''
        if stage == 'json_escapes':
            for path in sorted(output.glob('arm*/*/request_*.json')):
                if (path.parent / 'result.json').exists():
                    continue
                record = read(path)
                if record.get('error'):
                    continue
                raw_stage = record.get('client_metadata', {}).get('build_stage')
                text = response_text(record, raw_stage, config)
                _, repaired, positions = parse_json_escapes(text, repair=True)
                if positions:
                    affected.append({'request_file': str(path.relative_to(output)),
                                     'literal_backslash_positions': positions,
                                     'normalized_text_sha256': p.digest(repaired)})
        for receipt in sorted(output.glob('arm_' + request_stage + suffix + '/*/result.json')):
            saved = read(receipt); raw_path = receipt.parent / saved['request_file']
            p.require(raw_path.parent == receipt.parent and
                      r.sha256_file(raw_path) == saved['request_sha256'], 'Cached raw evidence changed')
            record = read(raw_path)
            p.require(decode(record, request_stage, config) == saved['value'], 'Cached value changed')
            payload = json.loads(record['request']['messages'][-1]['content'])
            if stage == 'error_type_projection':
                projected, changes = project_error_types(saved['value'])
                normalized_unknown_rejections(projected, payload)
                if changes:
                    affected.append({'result_file': str(receipt.relative_to(output)), 'metadata_projection': changes})
                continue
            if stage == 'short_distill_extras':
                _, changes = normalized_unknown_rejections(saved['value'], payload)
                if changes['excluded_unknown_rejections']:
                    affected.append({'result_file': str(receipt.relative_to(output)), 'changes': changes})
                continue
            if stage in {'short_ids', 'short_distill'}:
                try:
                    (normalized_rejections if stage == 'short_distill' else
                     normalized_group_duplicates)(saved['value'], payload)
                except ValueError as exc:
                    p.require(str(exc) == error, 'Unrelated invalid output cannot be superseded')
                    affected.append({'result_file': str(receipt.relative_to(output)), 'error': str(exc)})
                continue
            _, removed = normalized_view(stage, saved['value'], payload)
            if removed:
                affected.append({'result_file': str(receipt.relative_to(output)),
                                 ('assignment_changes' if stage == 'group_duplicates' else
                                  'id_expansions' if stage == 'group_errors' else 'removed_indices'): removed})
        p.require(affected, 'No unambiguous structural normalization to recover')
        files = sorted(output.glob('arm*/*/*.json')) + sorted((output / 'controls').rglob('*.json'))
        files.append(output / 'runtime.json')
        prior = [output / name for name, _, _ in RECOVERIES.values() if (output / name).exists()]
        files.extend(prior)
        update = {'policy': policy, 'actor': 'Codex assistant', 'human_gold': False,
                  'configuration_sha256': plan['configuration_sha256'],
                  'plan_sha256': r.sha256_file(output / 'plan.json'),
                  'original_sources': read(prior[-1])['replacement_sources'] if prior else config['source_hashes'],
                  'replacement_sources': source_hashes(),
                  'prefix_files': {str(f.relative_to(output)): r.sha256_file(f) for f in files},
                  'previous_disconnect_acceptance': read(output / 'disconnect_acceptance.json'),
                  'stopped_status': status, 'budget_before': budget, 'affected_jobs': affected,
                  'new_model_calls': 0, 'registered_unix': time.time()}
        if prior:
            update['previous_recovery_sha256'] = r.sha256_file(prior[-1])
        if stage in {'short_ids', 'short_distill'}:
            instruction = SHORT_DISTILL_INSTRUCTION if stage == 'short_distill' else SHORT_ID_INSTRUCTION
            update['protocol_change'] = {'stages': ['distill'] if stage == 'short_distill' else ['group_errors', 'cluster_rules'],
                'instruction': instruction, 'instruction_sha256': p.digest(instruction),
                'successful_prefix_reused': True, 'superseded_invalid_outputs_retained': True,
                'selection': 'first structurally valid new-protocol output; no quality reranking',
                'same_prompt_as_original': False, 'model_and_budgets_unchanged': True}
        verify(output, proposed_source_update=update)
        write_once(output / filename, update)
        print(json.dumps({'state': 'structural_recovery_registered', 'stage': stage, 'affected_jobs': len(affected),
                          'new_model_calls': 0}))


def prompts(config=None):
    return p.CORPUS_PROMPTS if (config or {}).get('corpus_release') else p.PROMPTS


def model_roles(config=None):
    return {stage: r.EXPECTED_MODEL for stage in prompts(config)}


def construction_messages(stage, payload, config=None):
    """Encode data control markers without deleting/changing historical text."""
    selected_prompts = prompts(config)
    p.require(stage in selected_prompts and isinstance(payload, dict), "Invalid construction job")
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    # JSON decodes to exactly the original data; no raw Harmony token reaches
    # the tokenizer. System instructions still undergo the unchanged guard.
    data = data.replace("<|", "\\u003c|")
    p.require(json.loads(data) == payload, "Input serialization changed data")
    messages = [{"role": "system", "content": p.SYSTEM + "\n" + selected_prompts[stage]},
                {"role": "user", "content": data}]
    r.assert_no_control_tokens(messages)
    return messages


def portable_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def benchmark_questions(additional=()):
    """Only stems/context leave the isolation loader; labels and scores never do.

    The mandatory known benchmarks cannot be replaced by --exclude. Extra files
    can only expand the exclusion set. Missing/shortened cohorts fail closed.
    """
    paths = {portable_path(ROOT / name): count for name, count in BENCHMARK_FILES.items()}
    for name in additional:
        paths.setdefault(portable_path(ROOT / name), None)
    registry, files = {}, {}
    for name, count in sorted(paths.items()):
        path = ROOT / name
        rows = read(path)
        p.require(isinstance(rows, list) and rows, "Empty benchmark cohort: " + name)
        p.require(count is None or len(rows) == count, "Known benchmark cohort size changed: " + name)
        p.require(len({x["id"] for x in rows}) == len(rows), "Duplicate benchmark IDs: " + name)
        files[name] = {"sha256": r.sha256_file(path), "rows": len(rows)}
        for row in rows:
            p.text_fields(row, ("id", "question"))
            # Context is part of the question, not an answer. If present, exclude
            # both the stem and the combined statement rather than lose either.
            texts = [row["question"]]
            if row.get("context"):
                p.require(isinstance(row["context"], str), "Benchmark context must be text")
                texts.append(row["question"] + "\n" + row["context"])
            for text in texts:
                key = p.question_key(text)
                entry = registry.setdefault(key, {"id": "bench_" + p.digest(key)[:24],
                                                  "question": text, "aliases": []})
                alias = {"file": name, "sample_id": row["id"]}
                if alias not in entry["aliases"]:
                    entry["aliases"].append(alias)
    return sorted(registry.values(), key=lambda x: x["id"]), files


def isolation_artifacts(source, heldout, cohorts, size, seed, source_audits=None):
    visible = source if source_audits is None else [dict(x, context=json.dumps(
        {'context': x.get('context', ''), 'audit': source_audits[x['id']]}, ensure_ascii=False)) for x in source]
    eligible, excluded = p.isolate_sources(visible, heldout)
    if source_audits is not None:
        original = {x['id']: {k: x.get(k) or '' for k in p.INPUT_FIELDS} for x in source}
        eligible = [original[x['id']] for x in eligible]
    rows = p.select_sources(eligible, size, seed)
    priorities = ("normalized_exact", "numeric_template", "lexical_family", "embedded_benchmark_question")
    reasons = Counter(next(reason for reason in priorities if any(
        match["reason"] == reason for match in x["matches"])) for x in excluded)
    report = {"policy": p.ISOLATION_POLICY, "benchmark_files": cohorts,
        "source_count": len(source), "heldout_unique_questions": len(heldout),
        "eligible_count": len(eligible), "selected_count": len(rows),
        "eligible_distinct_templates": len({p.question_template(x["question"]) for x in eligible}),
        "selected_distinct_templates": len({p.question_template(x["question"]) for x in rows}),
        "excluded": excluded, "excluded_counts": dict(reasons), "selected_overlap_count": 0,
        "no_claim_of_complete_family_isolation": True}
    return rows, report


def prepare(args):
    source = read(args.source)
    additional = [portable_path(x) for x in (args.exclude or [])]
    heldout, cohorts = benchmark_questions(additional)
    p.require(isinstance(source, list) and source, "Nonempty construction source required")
    p.require(args.size >= 2 and args.distill_max_requests > 0, "Positive build size and distillation cap required")
    p.require(bool(args.source_model.strip()), "Declare the historical answer model; do not infer it from filenames")
    corpus_release = getattr(args, 'corpus_release', None)
    binding, source_audits = None, None
    if corpus_release:
        binding, source_audits = released_corpus.load(corpus_release, args.source, full=True)
        p.require(args.source_model.strip() == binding['generator'], 'Declared generator conflicts with release')
    # No heldout labels/scores; corpus mode reuses only released training claims.
    rows, isolation = isolation_artifacts(source, heldout, cohorts, args.size, args.seed, source_audits)
    mode = {'corpus_release': portable_path(corpus_release)} if corpus_release else {}
    config = {**mode, "protocol": p.CORPUS_VERSION if corpus_release else p.VERSION, "source_sha256": r.sha256_file(args.source),
              "source_file": portable_path(args.source), "additional_exclusions": additional,
              "benchmark_files": cohorts, "bench_questions_sha256": p.digest(heldout),
              "isolation_sha256": p.digest(isolation), "selected_sha256": p.digest(rows),
              "source_hashes": source_hashes(), "prompts_sha256": p.digest([p.SYSTEM, prompts(mode)]),
              "model": r.EXPECTED_MODEL, "model_files": existing.MODEL_FILES,
              "api_base": "http://127.0.0.1:18083/v1", "reasoning_effort": "medium",
              "model_roles": model_roles(mode), "external_teacher": None,
              "distill_max_requests": args.distill_max_requests,
              "input_serialization": "json_escape_control_markers_lossless_v1",
              "max_tokens": 16384, "max_request_input_bytes": 400000 if corpus_release else 100000, "attempts_per_job": 3,
              "max_requests": args.size * 5 + args.distill_max_requests + 20,
              "batch_size": 12, "distill_size": 6, "seed": args.seed,
              "source_prediction_origin": "existing_historical_answers_not_regenerated_by_120B",
              "source_prediction_model": args.source_model.strip(),
              "source_model_evidence": "operator_declaration_not_independently_verified_generation_trace",
              "historical_reference_model": "not_verified",
              "old_rules_imported": False, "embedding_model": None,
              "isolation": p.ISOLATION_POLICY,
              "budget_policy": "unknown usage retains full reservation; exact-identity only; no content reroll",
              "sample_ids": [x["id"] for x in rows]}
    if corpus_release:
        selected_audits = {x['id']: source_audits[x['id']] for x in rows}
        config.update(source_prediction_origin='fresh_30b_released_generation',
            source_model_evidence=binding['source_model_evidence'], historical_reference_model=None,
            source_release_sha256=p.digest(binding), source_audits_sha256=p.digest(selected_audits),
            engineering_control_requests=24, control_required=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with r.run_lock(args.output / "active.lock"):
        if getattr(args, "refresh_unstarted", False):
            # A preparation may be corrected before the first execution, never
            # used to erase failed attempts or overwrite a generated catalog.
            p.require({x.name for x in args.output.iterdir()} <= PREPARED_FILES,
                      "Cannot refresh: execution artifacts exist; retain them and return to mainline")
            for name in PREPARED_FILES - {"active.lock"}:
                (args.output / name).unlink(missing_ok=True)
        write_once(args.output / "inputs.json", rows)
        write_once(args.output / "isolation.json", isolation)
        write_once(args.output / "bench_questions.json", heldout)
        if corpus_release:
            write_once(args.output / 'source_release.json', binding)
            write_once(args.output / 'source_audits.json', selected_audits)
        plan = {"configuration": config, "configuration_sha256": p.digest(config)}
        write_once(args.output / "plan.json", plan)
    print(json.dumps({"state": "prepared_no_model_calls", "selected": len(rows), "excluded": len(isolation["excluded"]),
                      "heldout_unique_questions": len(heldout),
                      "distill_request_cap": args.distill_max_requests, "output": str(args.output)}, ensure_ascii=False))


def verify(output, *, proposed_source_update=None):
    plan = read(output / "plan.json"); config = plan["configuration"]
    p.require(p.digest(config) == plan["configuration_sha256"], "Plan hash mismatch")
    corpus_release = config.get('corpus_release')
    p.require(config["protocol"] == (p.CORPUS_VERSION if corpus_release else p.VERSION),
              "Historical protocol: preserve its evidence; prepare a separate all-120B run")
    verify_sources(output, plan, proposed_source_update)
    rows = read(output / "inputs.json")
    p.require(config["selected_sha256"] == p.digest(rows), "Build inputs changed")
    p.require(config["prompts_sha256"] == p.digest([p.SYSTEM, prompts(config)]), "Prompts changed")
    p.require(config["model"] == r.EXPECTED_MODEL and config["model_roles"] == model_roles(config)
              and config["external_teacher"] is None and config["reasoning_effort"] == "medium"
              and config.get('embedding_model', 'missing') is None and config.get('old_rules_imported') is False
              and config["api_base"] == "http://127.0.0.1:18083/v1",
              "All construction stages must use the frozen local 120B medium service")
    p.require(config["isolation"] == p.ISOLATION_POLICY, "Isolation policy changed")
    heldout, cohorts = benchmark_questions(config["additional_exclusions"])
    p.require(cohorts == config["benchmark_files"], "Known benchmark inputs changed")
    p.require(heldout == read(output / "bench_questions.json") and
              p.digest(heldout) == config["bench_questions_sha256"], "Benchmark exclusion registry changed")
    isolation = read(output / "isolation.json")
    p.require(p.digest(isolation) == config["isolation_sha256"], "Isolation evidence changed")
    source = ROOT / config["source_file"]
    p.require(r.sha256_file(source) == config["source_sha256"], "Construction source inputs changed")
    source_audits = None
    if corpus_release:
        binding, source_audits = released_corpus.load(ROOT / corpus_release, source)
        selected_audits = {x['id']: source_audits[x['id']] for x in rows}
        p.require(binding == read(output / 'source_release.json') and p.digest(binding) == config['source_release_sha256'],
                  'Corpus release binding changed')
        p.require(selected_audits == read(output / 'source_audits.json') and
                  p.digest(selected_audits) == config['source_audits_sha256'], 'Imported audit evidence changed')
    expected, expected_isolation = isolation_artifacts(read(source), heldout, cohorts, len(rows), config["seed"], source_audits)
    p.require(rows == expected and config["sample_ids"] == [x["id"] for x in rows] and
              isolation == expected_isolation, "Selection/isolation does not replay from frozen source")
    return plan


def response_text(record, stage, config):
    p.require(not record.get("error"), "Failed request cannot be accepted")
    raw = record.get("response") or {}
    p.require(stage in prompts(config), "Unregistered construction stage")
    model = config["model"]
    p.require(raw.get("model") == model and raw.get("id"), "Model identity mismatch")
    text = harmony.decode_response(raw, model=model, max_tokens=config["max_tokens"], native_tools=None)["content"]
    # Do not repair physical/schema content or invoke a model to fix it.
    if text.startswith("```json\n") and text.rstrip().endswith("```"):
        text = text[len("```json\n"):].rstrip()[:-3]
    return text


def parse_json_escapes(text, repair=False):
    try:
        return json.loads(text), text, []
    except json.JSONDecodeError as exc:
        if not repair or exc.msg != 'Invalid \\escape':
            raise
    pieces, positions, inside, index = [], [], False, 0
    while index < len(text):
        char = text[index]
        if char == '"':
            inside = not inside
        if inside and char == '\\' and index + 1 < len(text):
            following = text[index + 1]
            if following not in '"\\/bfnrtu':
                positions.append(index)
                pieces.append('\\')
            pieces.extend([char, following]); index += 2
        else:
            pieces.append(char); index += 1
    normalized = ''.join(pieces)
    return json.loads(normalized), normalized, positions


def decode(record, stage, config):
    value, _, _ = parse_json_escapes(response_text(record, stage, config))
    p.require(isinstance(value, dict), "Expected JSON object")
    return value


class Journal:
    def __init__(self, output, config, retry_transient=False):
        self.output, self.config, self.retry = output, config, retry_transient
        self.distill_requests = len(list(output.glob("arm_distill/*/request_*.json"))) + len(list(output.glob("arm_distill_shortids/*/request_*.json")))
        maxreq = config["max_requests"]
        self.budget = BatchBudget(output, {"requests": maxreq,
            "input_tokens": maxreq * config["max_request_input_bytes"], "output_tokens": maxreq * config["max_tokens"]},
            stop_file=output / "STOP")

    def decode_record(self, path, stage):
        policy_path = self.output / RECOVERIES['json_escapes'][0]
        if not policy_path.exists():
            return decode(read(path), stage, self.config)
        text = response_text(read(path), stage, self.config)
        value, normalized, positions = parse_json_escapes(text, repair=True)
        p.require(isinstance(value, dict), 'Expected JSON object')
        if positions:
            write_once(path.with_name('json_escape_view_' + path.stem + '.json'), {
                'recovery_sha256': r.sha256_file(policy_path), 'request_sha256': r.sha256_file(path),
                'raw_text_sha256': p.digest(text), 'normalized_text_sha256': p.digest(normalized),
                'literal_backslash_positions': positions, 'value_sha256': p.digest(value),
                'scope': 'JSON string serialization only; legal escapes and content unchanged', 'new_model_calls': 0})
        return value

    def result_view(self, stage, payload, receipt, value):
        if stage not in RECOVERIES:
            return value
        if stage == 'group_errors' and (self.output / RECOVERIES['group_duplicates'][0]).exists():
            _, changes = normalized_group_duplicates(value, payload)
            if changes:
                stage = 'group_duplicates'
        filename, policy, _ = RECOVERIES[stage]
        if not (self.output / filename).exists():
            return value
        normalized, removed = normalized_view(stage, value, payload)
        if removed:
            saved = read(receipt)
            write_once(receipt.parent / ('group_member_normalization.json' if stage == 'group_duplicates'
                                         else 'normalization.json'), {
                'policy': policy, 'configuration_sha256': p.digest(self.config),
                'recovery_sha256': r.sha256_file(self.output / filename),
                'result_sha256': r.sha256_file(receipt), 'request_sha256': saved['request_sha256'],
                'original_value_sha256': p.digest(value), 'normalized_value_sha256': p.digest(normalized),
                ('assignment_changes' if stage == 'group_duplicates' else
                 'id_expansions' if stage == 'group_errors' else 'removed_indices'): removed,
                'unique_members': (len(normalized['assignments'])
                    if stage in {'assign_topics', 'group_errors', 'group_duplicates'} else len(payload['errors'])),
                'physical_content_edited': False, 'new_model_calls': 0})
        return normalized

    def short_view(self, stage, payload, receipt, value):
        projection_path = self.output / RECOVERIES['error_type_projection'][0]
        if stage == 'distill' and projection_path.exists():
            value, projections = project_error_types(value)
            if projections:
                write_once(receipt.parent / 'error_type_projection.json', {
                    'recovery_sha256': r.sha256_file(projection_path), 'result_sha256': r.sha256_file(receipt),
                    'changes': projections, 'rule_text_and_support_edited': False, 'new_model_calls': 0})
        extra_path = self.output / RECOVERIES['short_distill_extras'][0]
        if stage == 'distill' and extra_path.exists():
            normalized, mapping, changes = restore_short_distill(value, payload, allow_unknown_rejections=True)
            if changes['excluded_unknown_rejections']:
                write_once(receipt.parent / 'unknown_rejections.json', {
                    'recovery_sha256': r.sha256_file(extra_path), 'result_sha256': r.sha256_file(receipt),
                    'normalized_value_sha256': p.digest(normalized), 'changes': changes,
                    'rule_content_edited': False, 'new_model_calls': 0})
            # Keep the mapping receipt compatible for outputs without extras.
            if not changes['excluded_unknown_rejections']:
                changes = changes['duplicate_rejection_indices']
        else:
            normalized, mapping, changes = (restore_short_distill if stage == 'distill' else restore_short_groups)(value, payload)
        write_once(receipt.parent / 'id_mapping.json', {
            'protocol_sha256': r.sha256_file(self.output / RECOVERIES['short_distill' if stage == 'distill' else 'short_ids'][0]),
            'result_sha256': r.sha256_file(receipt), 'mapping': mapping,
            'normalized_value_sha256': p.digest(normalized), 'lossless_changes': changes})
        return normalized

    def supported_stages(self):
        return prompts(self.config)

    def messages(self, stage, payload):
        return construction_messages(stage, payload, self.config)

    def __call__(self, stage, key, payload, _short=False):
        c = self.config
        p.require(stage in self.supported_stages(), "Unregistered construction stage")
        model = c["model"]
        protocol_path = self.output / RECOVERIES['short_distill' if stage == 'distill' else 'short_ids'][0]
        short_enabled = protocol_path.exists() and stage in {'group_errors', 'cluster_rules', 'distill'}
        p.require(not _short or short_enabled, 'Short IDs need an explicit protocol amendment')
        wire_payload = (short_distill_payload if stage == 'distill' else short_group_payload)(payload)[0] if _short else payload
        messages = self.messages(stage, wire_payload)
        if _short:
            messages[0]['content'] += SHORT_DISTILL_INSTRUCTION if stage == 'distill' else SHORT_ID_INSTRUCTION
        p.require(len(json.dumps(messages, ensure_ascii=False).encode()) <= c["max_request_input_bytes"],
                  "Input exceeds frozen cap; no truncation, redesign batching on mainline")
        original_folder = self.output / ("arm_" + stage) / p.digest(key)[:24]
        folder = self.output / ("arm_" + stage + '_shortids') / p.digest(key)[:24] if _short else original_folder
        fingerprint = p.digest([stage, key, payload, model, c] +
            ([r.sha256_file(protocol_path), wire_payload] if _short else []))
        receipt = folder / "result.json"
        if short_enabled and not _short and not receipt.exists() and not list(folder.glob('request_*.json')):
            return self(stage, key, payload, _short=True)
        if receipt.exists():
            saved = read(receipt)
            p.require(saved["fingerprint"] == fingerprint, "Cached job input mismatch")
            path = folder / saved["request_file"]
            p.require(r.sha256_file(path) == saved["request_sha256"], "Cached response evidence changed")
            p.require(self.decode_record(path, stage) == saved["value"], "Cached decoded value mismatch")
            if _short:
                return self.short_view(stage, payload, receipt, saved['value'])
            try:
                return self.result_view(stage, payload, receipt, saved["value"])
            except ValueError:
                authorized = {x['result_file'] for x in read(protocol_path)['affected_jobs']} if short_enabled else set()
                if str(receipt.relative_to(self.output)) not in authorized:
                    raise
                return self(stage, key, payload, _short=True)
        paths = sorted(folder.glob("request_*.json"))
        for path in paths:
            p.require(read(path).get("client_metadata", {}).get("fingerprint") == fingerprint,
                      "Prior request belongs to another input/configuration")
        if paths and not read(paths[-1]).get("error"):
            # Crash after a completed request: decode the saved response, never pay again.
            record = read(paths[-1]); value = self.decode_record(paths[-1], stage)
        else:
            if paths:
                p.require(self.retry and retryable_transport_failure(read(paths[-1])),
                          "Prior failure requires explicit eligible technical recovery; no content rerun")
                last = read(paths[-1])
                ready = max(retry_after_deadline(last),
                            last.get("started_unix", 0) + last.get("elapsed_seconds", 0) + 300)
                p.require(time.time() >= ready, f"Transient cooldown active until Unix {ready:.0f}; no new request")
            old_attempts = len(list(original_folder.glob('request_*.json'))) if _short else 0
            p.require(len(paths) + old_attempts < c["attempts_per_job"], "Technical attempt cap reached")
            if stage == "distill":
                p.require(self.distill_requests < c["distill_max_requests"], "Distillation request cap reached")
            transport = Transport(base=c["api_base"], key="local",
                output=folder, model=model, reasoning_effort=c["reasoning_effort"], batch_budget=self.budget,
                limits={"requests": c["attempts_per_job"], "input_tokens": 300000,
                        "output_tokens": c["attempts_per_job"] * c["max_tokens"], "per_response": c["max_tokens"],
                        "seconds": 1800, "request_timeout_seconds": 1200, "connect_timeout_seconds": 20})
            transport.requests = len(paths)
            transport.record_metadata = {"build_stage": stage, "fingerprint": fingerprint}
            r.write_json(self.output / "execution_status.json", {"state": "running", "stage": stage,
                "key": key, "model": model, "updated_unix": time.time()})
            if stage == "distill":
                self.distill_requests += 1
            body = r.request_body(model, messages, c["max_tokens"], backend="vllm")
            body.update(stream=False, return_token_ids=True, reasoning_effort=c["reasoning_effort"])
            body.pop("stream_options", None)
            transport.post("/chat/completions", body)
            paths = sorted(folder.glob("request_*.json")); record = read(paths[-1])
            value = self.decode_record(paths[-1], stage)
        p.require(record.get("client_metadata", {}).get("fingerprint") == fingerprint, "Saved request binding mismatch")
        r.write_json(receipt, {"fingerprint": fingerprint, "value": value, "request_file": paths[-1].name,
                              "request_sha256": r.sha256_file(paths[-1])})
        print(stage, key, "response recorded; downstream validation follows", flush=True)
        return self.short_view(stage, payload, receipt, value) if _short else self.result_view(stage, payload, receipt, value)


def wait_detached(output, signature):
    ready = {"pid": os.getpid(), "process_start": existing.process_identity(os.getpid()),
             "configuration_sha256": signature, "token": uuid.uuid4().hex, "unix": time.time()}
    r.write_json(output / "disconnect_ready.json", ready)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        p.require(not (output / "STOP").exists(), "STOP requested")
        r.write_json(output / "execution_status.json", dict(ready, state="waiting_disconnect_acceptance", updated_unix=time.time()))
        receipt = output / "disconnect_acceptance.json"
        if receipt.exists():
            accepted = read(receipt)
            if accepted.get("passed") and all(accepted.get(k) == v for k, v in ready.items()):
                return
        time.sleep(2)
    raise RuntimeError("No new SSH disconnect acceptance; zero generation requests")


def confirm_detached(output):
    ready = read(output / "disconnect_ready.json")
    p.require(existing.process_identity(ready["pid"]) == ready["process_start"], "Runner PID changed")
    p.require(time.time() - ready["unix"] >= 5, "Reconnect after at least five seconds")
    status = read(output / "execution_status.json")
    p.require(status["state"] == "waiting_disconnect_acceptance" and status["token"] == ready["token"], "Not waiting for this acceptance")
    p.require(verify(output)["configuration_sha256"] == ready["configuration_sha256"], "Plan changed")
    r.write_json(output / "disconnect_acceptance.json", dict(ready, passed=True, confirmed_unix=time.time()))


def engineering_controls(output, config, retry_transient=False):
    folder = output / 'controls'
    rows = [
        {'id': 'control_mass_division', 'question': '惯性系中质量2kg的质点受6N合力，求加速度。',
         'prediction': 'a=F*m=12 m/s^2', 'answer': '', 'context': ''},
        {'id': 'control_cart_acceleration', 'question': '质量为m的小车在光滑水平面上受水平合力F，写出其加速度。',
         'prediction': '由牛顿第二定律，加速度a=F*m。', 'answer': '', 'context': ''}]
    audits = {x['id']: {'state': 'error_found', 'human_gold': False, 'uncertain': [],
        'errors': [{'quote': x['prediction'], 'message': '把加速度错误写成合力乘质量。',
                    'correction': '惯性系、质量不变时，合力F=ma，因此a=F/m。', 'error_type': 'concept'}]} for x in rows}
    control_config = dict(config, max_requests=config['engineering_control_requests'], purpose='synthetic_engineering_control')
    built = p.build_catalog(rows, Journal(folder, control_config, retry_transient),
                            batch_size=config['batch_size'], distill_size=config['distill_size'], source_audits=audits)
    p.require(built['counts']['errors'] == 2 and built['counts']['rules'] == 1,
              'Engineering control did not retain both source claims in one rule')
    for name, value in built.items():
        write_once(folder / (name + '.json'), value)
    summary = {'configuration_sha256': p.digest(config), 'catalog_sha256': r.sha256_file(folder / 'catalog.json'),
               'counts': built['counts'], 'human_gold': False, 'scope': 'synthetic_engineering_only_not_benchmark'}
    write_once(folder / 'summary.json', summary)
    return summary


def run(output, retry_transient=False, control_only=False):
    plan = verify(output); c = plan["configuration"]
    p.require(not control_only or c.get('corpus_release'),
              '--control-only requires the released-corpus protocol; never run a historical batch instead')
    p.require(sys.platform == "linux" and "huhengyuan" in Path(os.getenv("CONDA_PREFIX", "")).parts,
              "Run in the personal server conda environment")
    p.require("huhengyuan" in output.parts, "Personal persistent server output required")
    for variable in ("NO_PROXY", "no_proxy"):
        os.environ[variable] = ",".join(filter(None, [os.environ.get(variable), "127.0.0.1", "localhost"]))
    with r.run_lock(output / "active.lock"):
        if (output / "release_manifest.json").exists():
            release = read(output / "release_manifest.json")
            p.require(release["configuration_sha256"] == plan["configuration_sha256"], "Completed release changed")
            for name, sha in release["files"].items():
                p.require(r.sha256_file(output / name) == sha, "Completed artifact changed")
            print("Build already completed; zero new requests."); return
        (output / "runner.pid").write_text(str(os.getpid()))
        try:
            for cmd in (["df", "-h", str(output)], ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv"]):
                print(r.subprocess.check_output(cmd, text=True), flush=True)
            with httpx.Client(trust_env=False) as client:
                response = client.get(c["api_base"] + "/models", timeout=10); response.raise_for_status()
                p.require(any(x["id"] == c["model"] and x.get("max_model_len", 0) >= 131072 for x in response.json()["data"]), "Existing service/model context mismatch")
            p.require({n: r.sha256_file(Path(c["model"]) / n) for n in c["model_files"]} == c["model_files"], "120B configuration/weight index mismatch")
            write_once(output / "runtime.json", {
                "model_files": c["model_files"],
                "packages": {n: r.package_version(n) for n in ("vllm", "transformers", "torch", "httpx")},
                "python": sys.version,
                "checkpoint_validation": "configuration, tokenizer, template and shard index hashes; not full weight-shard hashes",
            })
            wait_detached(output, plan["configuration_sha256"])
            source_audits = None
            if c.get('corpus_release'):
                summary = engineering_controls(output, c, retry_transient)
                if control_only:
                    r.write_json(output / 'execution_status.json', {'state': 'engineering_controls_awaiting_content_review',
                        'summary': summary, 'formal_model_requests': 0, 'updated_unix': time.time()})
                    return
                acceptance = read(output / 'controls' / 'acceptance.json')
                p.require(acceptance.get('passed') is True and acceptance.get('actor') == 'Codex assistant'
                          and acceptance.get('summary_sha256') == r.sha256_file(output / 'controls' / 'summary.json'),
                          'Engineering control content has not been reviewed')
                source_audits = read(output / 'source_audits.json')
            built = p.build_catalog(read(output / "inputs.json"), Journal(output, c, retry_transient),
                                    batch_size=c["batch_size"], distill_size=c["distill_size"], source_audits=source_audits)
            # Detect an input/benchmark edit during a long-running build before release.
            verify(output)
            for name, value in built.items():
                write_once(output / (name + ".json"), value)
            artifacts = sorted(RELEASE_FILES)
            if c.get('corpus_release'):
                artifacts += ['source_release.json', 'source_audits.json']
                artifacts += [str(x.relative_to(output)) for x in sorted((output / 'controls').rglob('*.json'))]
            artifacts += [name for name, _, _ in RECOVERIES.values() if (output / name).exists()]
            # Bind every stage result/raw request, including failures and unknown usage.
            artifacts += [str(x.relative_to(output)) for x in sorted(output.glob("arm*/*/*.json"))]
            release = {"configuration_sha256": plan["configuration_sha256"], "files": {n: r.sha256_file(output / n) for n in artifacts},
                       "status": "candidate_catalog_complete_not_default_release", "human_gold": False,
                       "online_model": c["model"], "model_roles": c["model_roles"], "external_teacher": None,
                       "source_prediction_model": c["source_prediction_model"],
                       "source_model_evidence": c["source_model_evidence"],
                       "usable_for_runtime": built["counts"]["rules"] > 0, "independent_test_ready": False,
                       "data_isolation": {"passed": True, "policy_version": p.ISOLATION_POLICY["version"],
                           "bench_questions_sha256": c["bench_questions_sha256"],
                           "isolation_sha256": c["isolation_sha256"], "selected_overlap_count": 0,
                           "complete_semantic_family_isolation_proven": False}}
            write_once(output / "release_manifest.json", release)
            r.write_json(output / "execution_status.json", {"state": "completed", "counts": built["counts"], "updated_unix": time.time()})
        except BaseException as exc:
            r.write_json(output / "execution_status.json", {"state": "stopped", "error_type": type(exc).__name__,
                "error": str(exc)[:500], "updated_unix": time.time()})
            raise
        finally:
            (output / "runner.pid").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "verify", "run", "confirm-detached", "recover-topic-duplicates", "recover-rejection-duplicates", "recover-group-prefixes", "recover-group-duplicates", "prepare-short-ids", "prepare-short-distill", "recover-unknown-rejections", "recover-error-types", "recover-json-escapes"):
        command = sub.add_parser(mode); command.add_argument("--output", type=Path, required=True)
        if mode == "prepare":
            command.add_argument("--source", type=Path, required=True,
                                 help="Explicitly select the historical answer file after verifying its model origin")
            command.add_argument("--source-model", required=True,
                                 help="Historical answer-model attribution, separate from the fixed 120B builder")
            command.add_argument('--corpus-release', type=Path,
                                 help='Verified fresh-corpus release; reuse accepted errors without another audit')
            command.add_argument("--exclude", type=Path, action="append",
                                 help="Additional exclusion file; cannot replace mandatory known benchmarks")
            command.add_argument("--size", type=int, default=60)
            command.add_argument("--seed", type=int, default=20261019)
            command.add_argument("--distill-max-requests", type=int, required=True)
            command.add_argument("--refresh-unstarted", action="store_true",
                                 help="Replace preparation only if no execution artifacts exist")
        if mode == "run":
            command.add_argument("--retry-transient", action="store_true")
            command.add_argument('--control-only', action='store_true', help='Stop after synthetic engineering controls')
    args = parser.parse_args(); args.output = args.output.resolve()
    if args.mode == "prepare": prepare(args)
    elif args.mode == "verify": print(json.dumps({"state": "verified_no_model_calls", "signature": verify(args.output)["configuration_sha256"]}))
    elif args.mode == "run": run(args.output, args.retry_transient, args.control_only)
    elif args.mode == 'recover-topic-duplicates': recover_structure(args.output, 'assign_topics')
    elif args.mode == 'recover-rejection-duplicates': recover_structure(args.output, 'distill')
    elif args.mode == 'recover-group-prefixes': recover_structure(args.output, 'group_errors')
    elif args.mode == 'recover-group-duplicates': recover_structure(args.output, 'group_duplicates')
    elif args.mode == 'prepare-short-ids': recover_structure(args.output, 'short_ids')
    elif args.mode == 'prepare-short-distill': recover_structure(args.output, 'short_distill')
    elif args.mode == 'recover-unknown-rejections': recover_structure(args.output, 'short_distill_extras')
    elif args.mode == 'recover-error-types': recover_structure(args.output, 'error_type_projection')
    elif args.mode == 'recover-json-escapes': recover_structure(args.output, 'json_escapes')
    else: confirm_detached(args.output)


if __name__ == "__main__":
    main()
