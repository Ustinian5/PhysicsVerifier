"""Offline delivery/citation/paired-hit audit of a frozen four-arm batch.

Reads original requests, including failed generations. Never generates, changes
predictions, repairs quotations, or substitutes semantic judgments for scores.
Run with ``python -m scripts.analyze_organization_comparison RESULT_DIRECTORY``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

from rule_framework.verified_resume import atomic, digest, read, sha
from scripts import rescore_frozen_bench as scoring
from scripts.run_continuous_context_v2 import healthy_result

ARMS = ('pure', 'case_rag', 'flat_skills', 'rule_tree')
ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def evidence_path(root, recorded):
    """Map a receipt from the original server into this exact downloaded run."""
    parts = Path(recorded).parts
    if root.name not in parts and (root/'plan.json').exists():
        # New follow-ups may reuse a bound old PREPARE, never arbitrary external files.
        plan = read(root/'plan.json'); c = plan['configuration']
        require(digest(c) == plan['configuration_sha256'] and c.get('reused_queries') is True,
                'Unbound external evidence')
        prefix_path = root/'frozen_prefixes.json'
        expected = [h for name, h in c['files'].items() if name.endswith('/'+root.name+'/frozen_prefixes.json')]
        require(expected == [sha(prefix_path)], 'Frozen prefix manifest changed')
        allowed = {name: h for prefix in read(prefix_path).values() for name, h in prefix['files'].items()}
        require(recorded in allowed and c['files'].get(recorded) == allowed[recorded], 'Unbound external query')
        parent = Path(c['parent'])
        if not parent.exists(): parent = root.parent/parent.name
        require(parts.count(parent.name) == 1, 'Ambiguous external query path')
        suffix = parts[parts.index(parent.name)+1:]
        path = parent.joinpath(*suffix).resolve()
        require(suffix and suffix[0] == 'queries' and path.is_relative_to(parent.resolve()),
                'External evidence escapes frozen query directory')
        require(sha(path) == allowed[recorded], 'External query evidence changed')
        return path
    require(parts.count(root.name) == 1, 'Ambiguous/out-of-run evidence path')
    path = root.joinpath(*parts[parts.index(root.name) + 1:]).resolve()
    require(path.is_relative_to(root.resolve()), 'Evidence escapes run directory')
    return path


def sample_record(path, root, binding, model, evidence):
    result = read(path)
    packet_file = path.with_name('reference_packet.json')
    packet = read(packet_file)
    require(result['configuration_sha256'] == packet['configuration_sha256'] == binding,
            'Sample/packet configuration changed')
    p = packet['packet']
    reference_ids = [x['id'] for x in p['references']]
    require(len(set(reference_ids)) == len(reference_ids), 'Duplicate supplied ID')
    require(reference_ids == p['retrieval']['provided_complete_ids'], 'Delivery list drift')
    for item in result['trace']['response_evidence']:
        require(item['model_ids'] == [model], 'Recorded response identity changed')
        for key in ('request', 'response', 'raw_sse'):
            f = evidence_path(root, item[key + '_path'])
            name = str(f.relative_to(root)) if f.is_relative_to(root) else str(f)
            actual = evidence.setdefault(name, sha(f))
            require(actual == item[key + '_sha256'], 'Broken evidence hash: ' + str(f))
    # Some terminal native-format failures have supplemental usage receipts;
    # inspect actual turn files even if they were not appended to result.trace.
    turns = sorted(path.parent.glob('turn_*/request.json'))
    require(turns and turns[0].parent.name == 'turn_02', 'Missing first CHECK request')
    for f in turns:
        request = read(f)
        require(digest(request['request']) == request['request_sha256'], 'Request body hash drift')
        require(request['request']['model'] == model, 'Wrong requested model')
        response_path = f.with_name('response.json')
        response = read(response_path)
        require(response['model_ids'] == [model], 'Wrong actual model')
        raw = f.with_name('response.sse')
        require(sha(raw) == response['raw_sse_sha256'], 'Raw response hash drift')
        for source in (f, response_path, raw):
            evidence[str(source.relative_to(root))] = sha(source)
    first_request = read(turns[0])
    require(first_request['request']['messages'] == p['messages'],
            'Packet was not delivered verbatim in the actual CHECK request')
    first_path = path.with_name('first_diagnosis.json')
    first_diagnostics = read(first_path)['diagnostics'] if first_path.exists() else []
    diagnostics = result['diagnostics']
    citations = []
    for index, diagnostic in enumerate(diagnostics):
        refs = diagnostic.get('reference_ids', [])
        require(isinstance(refs, list) and all(isinstance(x, str) for x in refs), 'Invalid reference IDs')
        unsupplied = sorted(set(refs) - set(reference_ids))
        audit = result.get('protocol_audit', {})
        require(not unsupplied or (audit.get('passed') is False and
                'unsupplied_reference' in audit.get('violations', [])),
                'Final diagnostic cites an unsupplied reference without recorded violation')
        if refs:
            citations.append({'diagnostic_index': index, 'reference_ids': refs,
                              'unsupplied_reference_ids': unsupplied,
                              'message': diagnostic['message'],
                              'quote': diagnostic['evidence']['quote']})
    if first_path.exists():
        require([d.get('reference_ids', []) for d in first_diagnostics] ==
                [d.get('reference_ids', []) for d in diagnostics], 'Citation changed during quote repair')
        evidence[str(first_path.relative_to(root))] = sha(first_path)
    for f in (path, packet_file):
        evidence[str(f.relative_to(root))] = sha(f)
    return {
        'id': result['id'], 'status': result['status'], 'execution_failure': not healthy_result(result),
        'result_path': str(path.relative_to(root)), 'result_sha256': sha(path),
        'packet_sha256': sha(packet_file), 'input_sha256': packet['input_sha256'],
        'query_evidence_sha256': packet['query_evidence_sha256'],
        'shared_prefix_sha256': digest(p['messages'][:-1]),
        'check_request_body_sha256': first_request['request_sha256'],
        'diagnostics_sha256': digest(diagnostics),
        'provided_ids': reference_ids, 'provided_count': len(reference_ids),
        'reference_tokens': p['retrieval']['reference_tokens'],
        'omitted_count': len(p['retrieval']['omitted']),
        'eligible_leaf_count': p['retrieval'].get('eligible_leaf_count'),
        'published_diagnostic_count': len(diagnostics), 'citations': citations,
        'first_diagnosis_citation_count': sum(bool(d.get('reference_ids')) for d in first_diagnostics),
    }


def audit_inputs(root, config):
    """Resolve the exact released cohort, never silently use legacy 50 for 99."""
    from rule_framework import organization_bench99 as bench99
    from rule_framework import organization_followup as followup
    if config.get('protocol') in {bench99.VERSION, followup.VERSION}:
        for name in (bench99.INPUT, scoring.RELEASE, scoring.REFERENCE):
            hashes = [value for path, value in config['files'].items() if path.endswith('/'+name)]
            require(hashes == [sha(ROOT/name)], 'Frozen 99 reference/input dependency drift')
        rows, dataset, unknown = bench99.load_bench(ROOT)
        require(config['sample_count'] == 99 and sha(root/'inputs.json') == config['input_sha256'] and
                read(root/'inputs.json') == rows, 'Frozen 99 input drift')
        return rows, dataset, unknown, 'scores_all_99'
    require(sha(ROOT/scoring.INPUTS) == config['input_sha256'], 'Frozen input drift')
    for name, expected in config['reference_hashes'].items():
        require(sha(ROOT/name) == expected, 'Frozen scoring reference drift')
    _, dataset, unknown = scoring.reference_data(ROOT)
    return read(ROOT/scoring.INPUTS), dataset, unknown, 'scores_all_50'


def paired_summary(ids, assisted, pure, assisted_scores, pure_scores):
    """Partition observed location matches; this is NOT a causal decomposition."""
    rows = []
    for sid in ids:
        a, b = assisted_scores[sid], pure_scores[sid]
        gains = sorted(set(a['matched_claim_ids']) - set(b['matched_claim_ids']))
        losses = sorted(set(b['matched_claim_ids']) - set(a['matched_claim_ids']))
        rows.append({'id': sid, 'gained_claim_ids': gains, 'lost_claim_ids': losses,
                     'same_check_request': assisted[sid]['check_request_body_sha256'] ==
                                           pure[sid]['check_request_body_sha256'],
                     'same_diagnostics': assisted[sid]['diagnostics_sha256'] ==
                                         pure[sid]['diagnostics_sha256']})
    gains, losses = (sum(len(r[k]) for r in rows) for k in ('gained_claim_ids', 'lost_claim_ids'))
    return {'questions': len(ids), 'gained_location_matches': gains,
            'lost_location_matches': losses, 'net_location_matches': gains - losses,
            'identical_check_requests': sum(r['same_check_request'] for r in rows),
            'identical_diagnostics': sum(r['same_diagnostics'] for r in rows), 'details': rows,
            'assisted': scoring.summarize([assisted_scores[i] for i in ids]),
            'pure': scoring.summarize([pure_scores[i] for i in ids])}


def analyze(root):
    root = root.resolve()
    plan, metrics = read(root / 'plan.json'), read(root / 'metrics.json')
    config, binding = plan['configuration'], plan['configuration_sha256']
    require(digest(config) == binding == metrics['configuration_sha256'], 'Plan/metrics drift')
    rows, dataset, unknown, score_key = audit_inputs(root, config)
    inputs = {r['id']: r for r in rows}
    ids = config['sample_ids']
    require(len(ids) == len(set(ids)) and set(ids) == set(inputs), 'Duplicate/mismatched planned question')
    evidence, groups, scores, delivery = {}, {}, {}, {}
    for arm in ARMS:
        records = [sample_record(f, root, binding, config['model']['model'], evidence)
                   for f in sorted((root / 'formal' / arm).rglob('result.json'))]
        groups[arm] = {r['id']: r for r in records}
        require(len(records) == len(ids) and set(groups[arm]) == set(ids), 'Incomplete/duplicate arm')
        details = metrics[score_key][arm]['details']
        scores[arm] = {d['id']: d for d in details}
        require(len(details) == len(ids) and set(scores[arm]) == set(ids), 'Score IDs differ')
        require(scoring.summarize(details) == metrics[score_key][arm]['summary'],
                'Published summary does not aggregate its details')
        outputs = [read(root/groups[arm][sid]['result_path']) for sid in ids]
        rescored = scoring.score(dataset, [dict(x, execution_failure=not healthy_result(x),
            source_status=x['status'], protocol_noncompliance=not x.get('protocol_audit', {}).get('passed', False))
            for x in outputs], unknown)
        require(rescored == metrics[score_key][arm], 'Published scores differ from frozen outputs/reference')
        for sid, sample in groups[arm].items():
            require(sample['input_sha256'] == digest(inputs[sid]), 'Wrong frozen row binding')
            packet = read((root / sample['result_path']).with_name('reference_packet.json'))['packet']
            visible, _ = json.JSONDecoder().raw_decode(packet['messages'][1]['content'])
            require(visible == {k: inputs[sid].get(k, '') for k in ('question', 'prediction', 'context')},
                    'Actual visible question/answer differs from frozen input')
            detail = scores[arm][sid]
            require(sample['published_diagnostic_count'] == detail['published_diagnostic_count'],
                    'Published diagnostic count differs from result')
            require(sample['execution_failure'] == detail['execution_failure'], 'Failure drift')
            cited_indices = {d['diagnostic_index'] for d in sample['citations'] if not d['unsupplied_reference_ids']}
            sample['cited_location_match_count'] = len(cited_indices.intersection(detail['matched_finding_indices']))
        delivery[arm] = {
            'questions': len(records), 'with_material': sum(bool(r['provided_count']) for r in records),
            'total_items': sum(r['provided_count'] for r in records),
            'reference_tokens': sum(r['reference_tokens'] for r in records),
            'whole_items_omitted': sum(r['omitted_count'] for r in records),
            'item_count_distribution': dict(sorted(Counter(r['provided_count'] for r in records).items())),
            'published_diagnostics': sum(r['published_diagnostic_count'] for r in records),
            'questions_with_explicit_citation': sum(bool(r['citations']) for r in records),
            'explicitly_citing_diagnostics': sum(len(r['citations']) for r in records),
            'diagnostics_citing_only_supplied': sum(not c['unsupplied_reference_ids']
                                                   for r in records for c in r['citations']),
            'unsupplied_reference_citations': sum(bool(c['unsupplied_reference_ids'])
                                                for r in records for c in r['citations']),
            'cited_location_matches': sum(r['cited_location_match_count'] for r in records),
            'delivery_during_failed_output': sum(bool(r['provided_count']) and r['execution_failure']
                                                for r in records),
        }
    for sid in ids:
        for field in ('input_sha256', 'query_evidence_sha256', 'shared_prefix_sha256'):
            require(len({groups[a][sid][field] for a in ARMS}) == 1, 'Shared prefix differs')
    partitions = {}
    for arm in ARMS[1:]:
        subsets = {'all_'+str(len(ids)): ids,
            'with_material_both_successful': [i for i in ids if groups[arm][i]['provided_count'] and
                not scores[arm][i]['execution_failure'] and not scores['pure'][i]['execution_failure']],
            'with_material_any_output_failed': [i for i in ids if groups[arm][i]['provided_count'] and
                (scores[arm][i]['execution_failure'] or scores['pure'][i]['execution_failure'])],
            'without_material': [i for i in ids if not groups[arm][i]['provided_count']],
            'both_successful_sensitivity': [i for i in ids if not scores[arm][i]['execution_failure'] and
                not scores['pure'][i]['execution_failure']]}
        partitions[arm] = {name: paired_summary(subset, groups[arm], groups['pure'], scores[arm], scores['pure'])
                           for name, subset in subsets.items()}
    casepack = read(root / 'case_review.json')
    for case in casepack['cases']:
        for arm, output in case['outputs'].items():
            require(output == read(root / groups[arm][case['id']]['result_path']), 'Case output drift')
    pairwise = {a+'_vs_'+b: scoring.bootstrap(metrics[score_key][a], metrics[score_key][b])
                for a, b in combinations(reversed(ARMS), 2)}
    return {'configuration_sha256': binding, 'human_gold': False, 'independent_test': False,
        'sample_count': len(ids), 'original_scores_reproduced': True,
        'pairwise_comparisons': pairwise, 'pairwise_interpretation': 'Exploratory paired intervals; no multiple-comparison correction.',
        'metric': metrics['metric'], 'analysis_source_sha256': sha(Path(__file__)),
        'source_files': {f: sha(root / f) for f in ('plan.json', 'metrics.json', 'case_review.json')},
        'verified_evidence_files': len(evidence), 'verified_evidence_manifest_sha256': digest(evidence),
        'delivery': delivery, 'partitions_vs_pure': partitions, 'samples': groups,
        'case_review': {'entries': len(casepack['cases']), 'unique_questions': len({x['id'] for x in casepack['cases']}),
                       'statuses': dict(Counter(x['review']['status'] for x in casepack['cases'])),
                       'targeted_citation_reviews': len(casepack.get('citation_reviews', []))},
        'interpretation': [
            'Delivery is measured from actual request bodies, including failed model outputs.',
            'Citation is observable self-attribution, not proof of causality; no citation does not prove non-use.',
            'Matched claims are frozen location proxies, not semantic correctness judgments.',
            'Partitions are post-hoc descriptive analyses; original full-cohort scores remain primary.',
            'Identical empty-reference requests show repeated-output variation; they do not estimate the causal effect on other questions.',
            'Content/extraction/routing all differ, so this is not an isolated test of tree structure.',
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_directory', type=Path)
    args = parser.parse_args()
    report = analyze(args.result_directory)
    path = args.result_directory / 'material_usage_audit.json'
    atomic(path, report)
    print(path)
    for arm, values in report['delivery'].items():
        print(arm, values)


if __name__ == '__main__':
    main()
