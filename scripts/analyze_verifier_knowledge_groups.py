"""Compare archived verifier outputs and replay full-case delivery without model calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from rule_framework import verified_resume as vr
from scripts import audit_final_answer_metrics as binary
from scripts import run_aligned_step_revision as step
from scripts import run_final_answer_revision as final


def checked_json(path, manifest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest[str(path)]:
        raise ValueError(f'Archived evidence changed: {path}')
    return json.loads(data)


def step_comparison(root):
    release = vr.read(root / 'acceptance.json')
    step.require(release['passed'], 'Step release was not accepted')
    manifest = release['files']
    groups = checked_json(root / 'offline_review/metrics.json', manifest)['groups']
    ids = [row['id'] for row in groups['pure']['details']]
    seed, draws = 20261121, 10000
    weights = np.random.default_rng(seed).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=draws)
    scores, hits, report = {}, {}, {}
    for arm in binary.ARMS:
        rows = {row['id']: row for row in groups[arm]['details']}
        step.require(len(rows) == len(ids) and set(rows) == set(ids), 'Step cohorts differ')
        hits[arm] = {m['root_id'] for row in rows.values() for m in row['matches']}
        summary = groups[arm]['summary']
        step.require(len(hits[arm]) == summary['hits'], 'Repeated reference root')
        pairs = np.array([(rows[sid]['hits'], rows[sid]['locatable'] + rows[sid]['gt']) for sid in ids])
        matched, denominator = (weights @ pairs).T
        scores[arm] = 2 * matched / denominator
        delta = 100 * (scores[arm] - scores['pure'])
        report[arm] = dict(precision=summary['precision'], recall=summary['recall'], f1=summary['f1'],
                           gained_roots=sorted(hits[arm] - hits['pure']), lost_roots=sorted(hits['pure'] - hits[arm]),
                           shared_roots=len(hits[arm] & hits['pure']),
                           empty_completed=sum(not row['failed'] and row['published'] == 0 for row in rows.values()),
                           paired_f1_delta_pp_ci95=np.quantile(delta, [.025, .975]).tolist())
    sizes = {arm: Counter() for arm in binary.ARMS}
    for task in checked_json(root / 'tasks.json', manifest):
        if task['action'] == 'retain_prepare_failure':
            continue
        arm = task['arm']
        _, path = step.result_for(root, task)
        messages = checked_json(path.parent / 'turn_02/request.json', manifest)['request']['messages']
        payload, _ = json.JSONDecoder().raw_decode(messages[-1]['content'])
        items = [item['id'] for item in payload['Reference material']]
        step.require(items == task['after_reference_ids'] and payload['stage'] == 'CHECK', 'Delivered material drift')
        sizes[arm][len(items)] += 1
    for arm, counts in sizes.items():
        report[arm]['delivered_item_histogram'] = {str(n): count for n, count in sorted(counts.items())}
    return dict(groups=report, root_count=groups['pure']['summary']['gt'], union_root_hits=len(set.union(*hits.values())),
                paired_bootstrap=dict(unit='question', draws=draws, seed=seed, interval='percentile'),
                acceptance_sha256=vr.sha(root / 'acceptance.json'))


def archived_choice_bound(root):
    audit = vr.read(root / 'final_metric_audit.json')
    manifest = audit['files']
    labels = {sid: v['verdict'] for sid, v in checked_json(root / 'final_labels_private.json', manifest).items()}
    decisions = {}
    for arm in binary.ARMS:
        decisions[arm] = {}
        for sid in labels:
            path = root / 'checks/final/formal' / arm / sid / 'result.json'
            result = checked_json(path, manifest)
            step.require(result['id'] == sid and result['arm'] == arm, 'Archived prediction mismatch')
            if result['status'] == 'complete' and result['value']['verdict'] in {'correct', 'incorrect'}:
                decisions[arm][sid] = result['value']['verdict']
    common = {sid: y for sid, y in labels.items() if all(sid in v for v in decisions.values())}
    step.require(len(common) == audit['common_binary_questions'], 'Archived common binary cohort changed')
    views = {}
    for selection in [('pure',), ('pure', 'case_rag'), ('pure', 'flat_skills'), binary.ARMS]:
        predictions = {}
        for sid, gold in common.items():
            options = [decisions[a][sid] for a in selection]
            verdict = gold if gold in options else options[0]
            predictions[sid] = dict(status='complete', value=dict(verdict=verdict))
        stats = binary.measure(common, predictions)
        views['+'.join(selection)] = dict(**stats['original_binary_view'], counts=stats['counts'],
                                          accuracy=stats['binary_supplement']['accuracy'])
    return dict(n=len(labels), common_binary=len(common), old_labels=dict(Counter(common.values())),
                paired_vs_pure=audit['paired_vs_pure'], best_choice_with_known_label=views,
                scope='Exactly one of the listed existing binary outputs per question, on the same common subset.',
                metric_audit_sha256=vr.sha(root / 'final_metric_audit.json'))


def replay_rag_candidates(root):
    from transformers import AutoTokenizer

    config = vr.read(root / 'plan.json')['configuration']
    accepted = vr.read(root / 'pilot_acceptance.json')
    step.require(accepted['passed'] and vr.sha(root / 'cases.json') == config['material_hashes']['cases']
                 and vr.sha(root / 'pilot_inputs.json') == config['pilot_sha256'], 'Pilot or case store changed')
    memory = final.base.p.FlatMemory(vr.read(root / 'cases.json'))
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True, trust_remote_code=False)
    token_cap = config['contract']['reference_token_cap']
    item_cap = config['contract']['max_items']
    context_cap = config['limits']['context_window'] - config['limits']['output_tokens_per_sample']
    rows = [r for r in vr.read(root / 'pilot_inputs.json') if not r['id'].startswith('control_')]
    step.require(len(rows) == 4, 'Unexpected development cohort')
    report = {}
    for row in rows:
        path = root / 'pilot/case_rag' / vr.digest(row['id'])[:24] / 'reference_packet.json'
        packet = checked_json(path, accepted['files'])
        queries = json.loads(packet['messages'][2]['content'])
        query = '\n'.join(queries['queries'])
        remaining = set(memory.ids)
        ranked = []
        for _ in range(4):
            batch = memory.search(query, limit=item_cap, allowed_ids=remaining)
            ranked.extend(item['id'] for item in batch)
            remaining.difference_update(item['id'] for item in batch)
        step.require(ranked[:item_cap] == [d['id'] for d in packet['retrieval']['ranked']], 'Top-six ranking drift')

        def token_count(references):
            messages = final.p.continuation_messages(row, queries, references, arm='case_rag')
            return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     reasoning_effort='medium'))

        base_tokens = token_count([])
        selected, first_six = [], []
        for rank, sid in enumerate(ranked, 1):
            proposed = selected + [memory.view(sid)]
            if len(proposed) <= item_cap and token_count(proposed) <= min(context_cap, base_tokens + token_cap):
                selected = proposed
            if rank == min(item_cap, len(ranked)):
                first_six = [d['id'] for d in selected]
        step.require(first_six == packet['retrieval']['provided_complete_ids'], 'Baseline delivery drift')
        report[row['id']] = dict(top6_delivered=first_six, top24_delivered=[d['id'] for d in selected],
                                 top24_material_tokens=token_count(selected) - base_tokens)
    return dict(rows=report, scope='CPU delivery only; frozen development queries, whole cases, no new verdicts.',
                pilot_acceptance_sha256=vr.sha(root / 'pilot_acceptance.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='qwen30b_step_verifier results root')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = dict(model_calls=0, analysis_sha256=vr.sha(__file__),
                  step100=step_comparison((root / 'aligned_old99/evaluation_revision').resolve()),
                  old600=archived_choice_bound((root / 'corrected/pipeline/check_prompt_revision_bilingual').resolve()),
                  development_rag=replay_rag_candidates((root / 'aligned_old99/final_revision').resolve()))
    final.immutable(args.output, report)
