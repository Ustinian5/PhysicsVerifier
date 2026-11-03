"""Independent follow-ups: bounded-skills 99-case comparison or 5x case replay.

prepare/verify/score are offline. run uses the existing detached server engine.
Completed protocols are imported, never edited; changes are scoped adapters.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import httpx

from rule_framework import organization_followup as protocol
from scripts import run_organization_bench99 as bench
from scripts import resume_organization_bench99 as native
from scripts import analyze_organization_comparison as audit

base, vr, require, ROOT = bench.base, bench.vr, bench.require, bench.ROOT
SOURCE_FILES = sorted(set(bench.SOURCE_FILES) | {
    'rule_framework/organization_followup.py', 'scripts/run_organization_followup.py',
    'scripts/resume_organization_bench99.py', 'scripts/analyze_organization_comparison.py'})
MATERIALS = ('sources', 'inputs', 'cases', 'catalog', 'isolation', 'bench_questions')
ORIGINAL_PILOT = bench.check_pilot
ORIGINAL_BUILD = bench.build


def source_hashes():
    return {name: vr.sha(ROOT/name) for name in SOURCE_FILES}


def frozen_prefix(parent, row, binding, files):
    path = parent/'queries'/vr.digest(row['id'])[:24]/'result.json'
    original = vr.read(path)
    require(original['configuration_sha256'] == binding and original['input_sha256'] == vr.digest(row),
            'Wrong frozen query prefix')
    value = copy.deepcopy(original); localized = {}
    files[str(path)] = vr.sha(path)
    for recorded, sha in original['files'].items():
        f = audit.evidence_path(parent, recorded)
        require(vr.sha(f) == sha, 'Frozen query evidence drift')
        localized[str(f)] = sha
    value['files'] = localized; files.update(localized)
    for key in ('request', 'response', 'raw_sse'):
        f = audit.evidence_path(parent, original['evidence'][key+'_path'])
        require(localized.get(str(f)) == original['evidence'][key+'_sha256'], 'Query evidence list drift')
        value['evidence'][key+'_path'] = str(f)
    request = vr.read(Path(value['evidence']['request_path']))
    response = vr.read(Path(value['evidence']['response_path']))
    require(vr.digest(request['request']) == request['request_sha256'] and
            request['request']['messages'] == base.p.query_messages(row) and
            request['request']['model'] == base.r.EXPECTED_MODEL and
            request['request']['max_tokens'] == base.p.QUERY_MAX_TOKENS and
            not response.get('error') and response['model_ids'] == [base.r.EXPECTED_MODEL] and
            response['finish_reasons'] == ['stop'] and response['usage'] == value['evidence']['usage'] and
            base.p.validate_queries(base.r.parse_json(response['content'])) == value['queries'],
            'Frozen query does not match current input/prompt/model')
    return value, original


def prepare(output, parent, task):
    require(output != parent and not output.is_relative_to(parent), 'Keep the completed batch immutable')
    output.mkdir(parents=True, exist_ok=True)
    with base.r.run_lock(output/'active.lock'):
        if (output/'plan.json').exists():
            plan = verify(output)
            require(plan['configuration']['task'] == task and plan['configuration']['parent'] == str(parent),
                    'Existing task/parent differs')
            return plan
        prior = vr.read(parent/'plan.json'); pc = prior['configuration']
        require(vr.digest(pc) == prior['configuration_sha256'] and pc['protocol'] == protocol.previous.VERSION,
                'Expected the completed unbounded procedural-skills 99-case protocol')
        require(vr.read(parent/'execution_status.json')['state'] == 'completed', 'Parent not completed')
        require(pc['model']['model'] == base.r.EXPECTED_MODEL and pc['model']['reasoning_effort'] == 'medium' and
                pc['model_files'] == base.runtime.MODEL_FILES and pc['check_prompt'] == base.p.STAGED_BASE and
                pc['controls'] == base.runtime.CONTROLS, 'Parent runtime/protocol changed')
        files = {str(parent/'plan.json'): vr.sha(parent/'plan.json')}
        for name in MATERIALS:
            path = parent/(name+'.json'); bench.bound_artifact(pc, path)
            files[str(path)] = vr.sha(path)
        rows, _, _ = protocol.load_bench(ROOT)
        require(rows == vr.read(parent/'inputs.json'), 'Published 99 inputs changed')
        sources, cases = vr.read(parent/'sources.json'), vr.read(parent/'cases.json')
        require(len(sources) == len(cases) == 1621 and
                {tuple(c['source_ids']) for c in cases} == {(s['id'],) for s in sources}, 'Source cohort changed')
        heldout, registry = base.old.b.benchmark_questions()
        require(heldout == vr.read(parent/'bench_questions.json'), 'Held-out registry changed; review isolation')
        dependencies = [ROOT/protocol.previous.INPUT, ROOT/base.evaluation.RELEASE, ROOT/base.evaluation.REFERENCE]
        dependencies += [(ROOT/base.evaluation.RELEASE).parent/n for n in vr.read(ROOT/base.evaluation.RELEASE)['files']]
        dependencies += [ROOT/n for n in registry]
        for f in dependencies:
            require([h for p, h in pc['files'].items() if p.endswith('/'+str(f.relative_to(ROOT)))] == [vr.sha(f)],
                    'Parent label or isolation dependency drift')
            files[str(f)] = vr.sha(f)
        c = copy.deepcopy(pc)
        c.update(protocol=protocol.VERSION if task == 'bounded99' else protocol.REPEAT_VERSION,
            task=task, parent=str(parent), parent_configuration_sha256=prior['configuration_sha256'],
            source_hashes=source_hashes(), files=files, prior_usage=base.old.ZERO_USAGE,
            reused_queries=True, independent_test=False, human_gold=False,
            query_policy='reuse exact old preparation; charge its tokens in every condition budget, zero new query calls',
            controls_policy='20 fresh checks for bounded99; case-repeat reuses passed historical controls and checks runtime identity')
        c['model']['protocol'] = c['protocol']
        if task == 'bounded99':
            # One preparation-time screen, never repeated on every checkpoint.
            screen = protocol.FamilyScreen(heldout); byid = {x['source_ids'][0]: x for x in cases}
            excluded = []
            for row in sources:
                matches = screen.matches(dict(row, prediction=byid[row['id']]['text']))
                if matches: excluded.append(dict(source_id=row['id'], matches=matches))
            excluded_ids = {x['source_id'] for x in excluded}
            sources = [x for x in sources if x['id'] not in excluded_ids]
            cases = [x for x in cases if x['source_ids'][0] not in excluded_ids]
            require(sources and len(sources) == len(cases), 'No isolated build sources')
            provenance_paths = [p for p in pc['files'] if p.endswith('/tree_v2/materials.json')]
            require(len(provenance_paths) == 1, 'Need unique bound original tree provenance')
            provenance_path = Path(provenance_paths[0])
            if not provenance_path.exists():
                provenance_path = ROOT/'results/qwen30b_expansion_5000/tree_v2/materials.json'
            require(vr.sha(provenance_path) == pc['files'][provenance_paths[0]], 'Tree provenance drift')
            files[str(provenance_path)] = vr.sha(provenance_path)
            catalog, tree_changes = protocol.previous.prune_catalog(vr.read(parent/'catalog.json'),
                vr.read(provenance_path)['provenance'], excluded_ids)
            base.p.TreeMemory(catalog)
            for name in ('inputs', 'bench_questions'):
                (output/(name+'.json')).write_bytes((parent/(name+'.json')).read_bytes())
            for name, value in [('sources', sources), ('cases', cases), ('catalog', catalog),
                    ('isolation', dict(parent_isolation_sha256=vr.sha(parent/'isolation.json'),
                        policy=protocol.ISOLATION_POLICY, original_sources=1621, kept_sources=len(sources),
                        excluded=excluded, tree_changes=tree_changes, human_gold=False,
                        semantic_family_independence_proven=False))]:
                vr.atomic(output/(name+'.json'), value)
            c.update(skill_prompt=protocol.SKILL_PROMPT, skill_input_policy=protocol.INPUT_POLICY,
                     skill_audit_prompt=protocol.AUDIT_PROMPT, extraction_passes=protocol.EXTRACTION_PASSES,
                     extraction_pass_output_tokens=protocol.PASS_OUTPUT_TOKENS,
                     skill_pilot_size=protocol.PILOT_SIZE, skill_development_size=12,
                     skill_pilot="fixed first 12 development plus next 12 construction validation sources; review all before scale-up",
                     skill_pilot_ids=[x['id'] for x in sources[:protocol.PILOT_SIZE]],
                     skills='one complete procedure by default; at most two per source',
                     max_skills_per_source=2, target_skill_count_is_not_a_quota=True,
                     minimum_skill_count_before_comparison=1000,
                     source_count=len(sources), isolation_policy=protocol.ISOLATION_POLICY,
                     source_exclusions_from_parent=sorted(excluded_ids),
                     order='build bounded skills, then fresh 99x4 checks with original shared queries',
                     planned_checks=396, planned_skill_sources=len(sources))
            query_rows = rows + base.runtime.CONTROLS
        else:
            rows = [x for x in rows if x['id'] == protocol.TARGET_ID]
            require(len(rows) == 1, 'Missing predefined replication case')
            vr.atomic(output/'inputs.json', rows)
            vr.atomic(output/'target_rubric.json', protocol.target_rubric())
            c.update(sample_count=1, sample_ids=[protocol.TARGET_ID], repeats=protocol.REPEATS,
                     planned_checks=20, planned_skill_sources=0,
                     order='five blocks, all four arms rotated; frozen original CHECK packets',
                     selected_positive_case=True, no_overall_precision_recall_f1=True)
            query_rows = rows
        prefixes, originals = {}, {}
        for row in query_rows:
            prefixes[row['id']], originals[row['id']] = frozen_prefix(parent, row, prior['configuration_sha256'], files)
        vr.atomic(output/'frozen_prefixes.json', prefixes)
        if task == 'case-repeat':
            packets, first_requests, evidence = {}, {}, {}
            for arm in base.p.ARMS:
                path = base.path_for(parent, 'formal', arm, rows[0])
                audit.sample_record(path, parent, prior['configuration_sha256'], base.r.EXPECTED_MODEL, evidence)
                record = vr.read(path.with_name('reference_packet.json'))
                require(record['input_sha256'] == vr.digest(rows[0]) and
                        record['query_evidence_sha256'] == vr.digest(originals[rows[0]['id']]), 'Packet query binding drift')
                packets[arm] = record['packet']
                first_requests[arm] = vr.read(path.parent/'turn_02/request.json')['request']
            files.update({str(parent/p): h for p, h in evidence.items()})
            require(vr.read(parent/'metrics.json')['all_engineering_controls_passed'], 'Original engineering controls failed')
            files[str(parent/'metrics.json')] = vr.sha(parent/'metrics.json')
            vr.atomic(output/'frozen_packets.json', packets)
            vr.atomic(output/'first_check_requests.json', first_requests)
        for f in output.glob('*.json'):
            if f.name != 'plan.json': files[str(f)] = vr.sha(f)
        if (parent/'runtime.json').exists(): files[str(parent/'runtime.json')] = vr.sha(parent/'runtime.json')
        c['input_sha256'] = vr.sha(output/'inputs.json')
        plan = dict(configuration=c, configuration_sha256=vr.digest(c))
        vr.atomic(output/'plan.json', plan)
        base.old.emit(output, 'prepared_no_model_calls', task=task, planned_checks=c['planned_checks'])
        return verify(output)


def verify(output, full=False):
    plan = vr.read(output/'plan.json'); c = plan['configuration']
    require(vr.digest(c) == plan['configuration_sha256'] and c['source_hashes'] == source_hashes(), 'Plan/code drift')
    require(c['task'] in {'bounded99', 'case-repeat'} and
            c['protocol'] == (protocol.VERSION if c['task'] == 'bounded99' else protocol.REPEAT_VERSION), 'Wrong protocol')
    vr.EvidenceCache(output/'evidence_cache.json').check(c['files'], full=full)
    require(c['check_prompt'] == base.p.STAGED_BASE and c['reference_token_cap'] == base.p.REFERENCE_TOKEN_CAP and
            c['max_items'] == base.p.MAX_ITEMS and c['query_max_tokens'] == base.p.QUERY_MAX_TOKENS and
            c['tree_branches'] == base.p.TREE_BRANCHES and vr.sha(output/'inputs.json') == c['input_sha256'],
            'Online prompt/budget/input drift')
    rows = vr.read(output/'inputs.json')
    require([r['id'] for r in rows] == c['sample_ids'] and len(rows) == c['sample_count'], 'Wrong cohort')
    if c['task'] == 'bounded99':
        require(len(rows) == 99 and c['skill_prompt'] == protocol.SKILL_PROMPT and c['max_skills_per_source'] == 2 and
                c['minimum_skill_count_before_comparison'] == 1000 and c['isolation_policy'] == protocol.ISOLATION_POLICY and
                c['skill_input_policy'] == protocol.INPUT_POLICY and
                c['skill_audit_prompt'] == protocol.AUDIT_PROMPT and
                c['extraction_passes'] == protocol.EXTRACTION_PASSES and
                c['extraction_pass_output_tokens'] == protocol.PASS_OUTPUT_TOKENS and
                c['skill_pilot_size'] == protocol.PILOT_SIZE and c['skill_development_size'] == 12 and
                c['skill_pilot_ids'] == [x['id'] for x in vr.read(output/'sources.json')[:protocol.PILOT_SIZE]],
                'Bounded skill protocol changed')
    else:
        require(c['repeats'] == 5 and c['sample_ids'] == [protocol.TARGET_ID] and
                vr.read(output/'target_rubric.json') == protocol.target_rubric(), 'Replication target/rubric changed')
    return plan


def check_pilot(output, rows, binding):
    ORIGINAL_PILOT(output, rows, binding)
    review = vr.read(output/'skills/pilot_review.json')
    byid = {x['id']: x for x in review['samples']}
    for row in rows[:base.p.PILOT_SIZE]:
        outcome = vr.read(base.old.outcome_path(output/'skills', row))['outcome']
        if outcome['status'] != 'collected': continue
        skills = protocol.validate_skills(outcome['value'])['skills']
        if skills:
            require(byid[row['id']].get('substeps_bundled') is True, 'Pilot must check complete-method granularity')
            if len(skills) == 2:
                require(byid[row['id']].get('second_method_independent') is True,
                        'Second method is not independent; revise protocol, never reroll this source')


def build(output, plan, tokenizer):
    memory = ORIGINAL_BUILD(output, plan, tokenizer)
    count = len(memory.docs)
    require(1000 <= count <= 2 * plan['configuration']['source_count'],
            f'Skill count {count} is outside the predefined comparable range; preserve outputs and return to mainline, no filler or resampling')
    return memory


class IndependentSkillJournal(bench.ProcedureJournal):
    def __init__(self, folder, config, tokenizer):
        super().__init__(folder, config, tokenizer)
        self.sources = {x['id']: x for x in vr.read(folder.parent/'sources.json')}
        require(protocol.EXTRACTION_PASSES * protocol.PASS_OUTPUT_TOKENS <= config['model']['max_tokens'],
                'Two-pass extraction exceeds original per-source output cap')
        self.config = dict(self.config, max_tokens=protocol.PASS_OUTPUT_TOKENS)

    def source(self, payload):
        ids = payload['case']['source_ids']
        require(len(ids) == 1 and ids[0] in self.sources, 'Unknown skill source')
        return self.sources[ids[0]]

    def messages(self, stage, payload):
        require(stage == 'extract_skill', 'Independent extraction only')
        if 'draft' in payload:
            return protocol.selected_evidence_messages(self.source(payload), payload['draft'])[0]
        return protocol.skill_messages(self.source(payload))

    def __call__(self, stage, key, payload):
        require(key == self.source(payload)['id'], 'Skill source/key mismatch')
        draft = super().__call__(stage, key + ':select', payload)
        try:
            _, visible = protocol.selected_evidence_messages(self.source(payload), draft)
        except (ValueError, TypeError, KeyError) as exc:
            raise base.old.tree.ContentFailure(str(exc)) from exc
        value = super().__call__(stage, key, dict(payload, draft=draft))
        try:
            require(set(value['source_review']['evidence_ids']) <= visible, 'Final evidence was not visible during distillation')
            return protocol.validate_source_evidence(value, self.source(payload))
        except (ValueError, TypeError, KeyError) as exc:
            raise base.old.tree.ContentFailure(str(exc)) from exc


@contextmanager
def adapted(output, plan):
    prefixes = vr.read(output/'frozen_prefixes.json')
    def reuse_query(unused_output, row, unused_plan, tokenizer, client):
        prefix = copy.deepcopy(prefixes[row['id']])
        require(prefix['input_sha256'] == vr.digest(row), 'Replayed query input changed')
        prefix['configuration_sha256'] = plan['configuration_sha256']
        return prefix
    with patch.object(base.p, 'PILOT_SIZE', protocol.PILOT_SIZE), \
         patch.object(bench, 'protocol', protocol), patch.object(bench, 'verify', verify), \
         patch.object(bench, 'ProcedureJournal', IndependentSkillJournal), \
         patch.object(bench, 'check_pilot', check_pilot), patch.object(bench, 'build', build), \
         patch.object(base, 'query', reuse_query), \
         patch.object(bench.recovery, 'inspect_failure', native.inspect_failure):
        yield


def repeat_path(output, repeat, arm, row):
    return base.path_for(output/f'repeat_{repeat:02d}', 'formal', arm, row)


class ExactFirstRequestClient:
    """Reject first-request drift before network I/O, including hidden parameters."""
    def __init__(self, client, expected):
        self.client, self.expected = client, expected

    def post(self, endpoint, *, json, **kwargs):
        if len(json['messages']) <= len(self.expected['messages']):
            require(json == self.expected, 'First CHECK request differs from frozen original')
        return self.client.post(endpoint, json=json, **kwargs)


def score_repeats(output, plan):
    row = vr.read(output/'inputs.json')[0]; binding = plan['configuration_sha256']
    records, mapping, actual_requests = [], {}, set()
    reused_requests = {x['evidence']['request_path'] for x in vr.read(output/'frozen_prefixes.json').values()}
    for repeat in range(1, protocol.REPEATS+1):
        for arm in base.p.ARMS:
            path = repeat_path(output, repeat, arm, row)
            if not path.exists(): continue
            value = vr.read(path); base.runtime.audit_result(value)
            require(value['configuration_sha256'] == binding, 'Replication binding changed')
            if not base.runtime.healthy_result(value): value = bench.recovery.retained_failure(path, output_amended=True)
            actual_requests.update(x['request_path'] for x in value['trace']['response_evidence'])
            if not base.runtime.healthy_result(value) and not base.complete_content_failure(value):
                actual_requests.add(bench.recovery.account(path)['request_path'])
            key = vr.digest([binding, arm, repeat])[:20]
            mapping[key] = dict(arm=arm, repeat=repeat, result_path=str(path), result_sha256=vr.sha(path))
            records.append(dict(id=key, result_sha256=vr.sha(path), execution_failure=not base.runtime.healthy_result(value),
                diagnostics=[dict(message=d['message'], evidence=d['evidence']) for d in value['diagnostics']],
                review=dict(actor='Codex assistant', status='pending', human_gold=False)))
    complete = len(records) == 20
    review_path = output/'target_review.json'
    # Export only once after all scheduled runs; never overwrite filled reviews.
    if complete and not review_path.exists():
        vr.atomic(review_path, dict(configuration_sha256=binding, input=row, rubric=protocol.target_rubric(),
            records=sorted(records, key=lambda x: x['id']), human_gold=False))
        vr.atomic(output/'operator_map.json', dict(configuration_sha256=binding, records=mapping))
    report = dict(configuration_sha256=binding, recorded_tasks=len(records), planned_tasks=20,
        target_scores=None, review_pending=True, metric='semantic_target_detection_not_overall_PRF',
        actual_new_check_requests=len(actual_requests-reused_requests),
        reused_query_requests=len(actual_requests & reused_requests),
        human_gold=False, independent_test=False, selected_positive_case=True)
    if complete and review_path.exists():
        review = vr.read(review_path)
        require(review['configuration_sha256'] == binding and review['human_gold'] is False and
                review['input'] == row and review['rubric'] == protocol.target_rubric(), 'Review binding changed')
        entries = {e['id']: e for e in review['records']}
        require(len(entries) == len(review['records']) == 20 and set(entries) == set(mapping), 'Review every scheduled run')
        totals = {arm: Counter() for arm in base.p.ARMS}; pending = 0
        for record in records:
            entry = entries[record['id']]; item = entry['review']; arm = mapping[record['id']]['arm']
            require({k: v for k, v in entry.items() if k != 'review'} ==
                    {k: v for k, v in record.items() if k != 'review'}, 'Review source output changed')
            if record['execution_failure']: totals[arm]['technical_failure'] += 1; continue
            if item['status'] == 'pending': pending += 1; continue
            require(item['status'] == 'completed' and item.get('actor') in {'Codex assistant', 'human'} and
                    item.get('human_gold') is False and isinstance(item.get('notes'), str) and item['notes'].strip(),
                    'Review needs explicit actor and substantive semantic justification')
            verdict = item.get('target_verdict')
            require(verdict in {'correct_detection', 'missed', 'incorrect_target_claim', 'unassessable'}, 'Bad target verdict')
            indices = item.get('diagnostic_indices', [])
            require(isinstance(indices, list) and all(type(i) is int and 0 <= i < len(record['diagnostics']) for i in indices),
                    'Bad diagnosis references')
            require(bool(indices) if verdict in {'correct_detection', 'incorrect_target_claim'} else True,
                    'Target judgment must cite actual diagnostics')
            totals[arm][verdict] += 1
        report.update(pending_reviews=pending, review_pending=bool(pending))
        if not pending:
            report['target_scores'] = {a: dict(counts=dict(t), scheduled=5,
                correct_detection_rate=t['correct_detection']/5) for a, t in totals.items()}
    vr.atomic(output/'metrics.json', report)
    return report


def score(output, plan=None):
    plan = plan or verify(output)
    with adapted(output, plan):
        if plan['configuration']['task'] == 'case-repeat': return score_repeats(output, plan)
        report = bench.score(output, plan)
    prefixes = vr.read(output/'frozen_prefixes.json')
    reused = {p['evidence']['request_path'] for p in prefixes.values()}
    # Old shared query requests are visible in traces, but did not consume new inference.
    paths = {e['request_path'] for f in (output/'formal').rglob('result.json')
             for e in vr.read(f)['trace']['response_evidence']}
    paths.update(x['request_path'] for u in report['usage'].values() for x in u['supplemental_failure_usage'])
    report.update(formal_reused_prefix_requests=len(paths & reused),
                  formal_new_check_requests=len(paths-reused),
                  accounting='per-condition budget includes shared prefix; old prefix requests are not new cost')
    vr.atomic(output/'metrics.json', report)
    return report


def run_repeats(output, plan):
    c = plan['configuration']
    with base.r.run_lock(output/'active.lock'), base.r.run_lock(output.parent/'organization_service_18083.lock'):
        verify(output)
        if vr.read(output/'execution_status.json')['state'] == 'completed': score(output, plan); return
        runtime = base.old.server_preflight(output, dict(c, model=base.r.EXPECTED_MODEL))
        require(runtime == vr.read(Path(c['parent'])/'runtime.json'), 'Replication runtime differs from original')
        base.old.bind_runtime(output/'runtime.json', runtime)
        (output/'runner.pid').write_text(str(os.getpid())); exit_code = 1
        try:
            base.old.b.wait_detached(output, plan['configuration_sha256'])
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(base.r.EXPECTED_MODEL, local_files_only=True, trust_remote_code=False)
            packets = vr.read(output/'frozen_packets.json'); expected = vr.read(output/'first_check_requests.json')
            row = vr.read(output/'inputs.json')[0]
            def frozen_packet(given, arm, queries, memory, count):
                require(given == row, 'Wrong replication input')
                return copy.deepcopy(packets[arm])
            with adapted(output, plan), base.old.comparison.execution.configured(), \
                 patch.object(base.p, 'reference_packet', frozen_packet), \
                 patch.object(base.runtime.transport, 'decode_response', base.decode_response), httpx.Client(trust_env=False) as client:
                for repeat in range(1, protocol.REPEATS+1):
                    arms = list(base.p.ARMS); offset = (repeat-1) % 4
                    for arm in arms[offset:] + arms[:offset]:
                        require(not (output/'STOP').exists(), 'STOP requested')
                        bench.run_condition(output/f'repeat_{repeat:02d}', row, 'formal', arm, None, plan, tokenizer,
                                            ExactFirstRequestClient(client, expected[arm]))
                        first = repeat_path(output, repeat, arm, row).parent/'turn_02/request.json'
                        require(vr.read(first)['request'] == expected[arm], 'First CHECK request differs from frozen original')
                    base.old.emit(output, 'replicating', completed_blocks=repeat, planned_blocks=5)
                score(output, plan)
            base.old.emit(output, 'completed', recorded_tasks=20, semantic_review='pending'); exit_code = 0
        except BaseException as exc:
            base.old.emit(output, 'stopped', error_type=type(exc).__name__, error=str(exc)[:1200]); raise
        finally:
            (output/'runner.exit_code').write_text(str(exit_code)+'\n'); (output/'runner.pid').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify', 'run', 'score', 'status', 'confirm-detached'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent-comparison', type=Path)
    parser.add_argument('--task', choices=['bounded99', 'case-repeat'])
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args(); output = args.output.resolve()
    if args.mode == 'prepare':
        require(args.parent_comparison is not None and args.task is not None, 'Specify task and completed 99-case parent')
        print(prepare(output, args.parent_comparison.resolve(), args.task)['configuration_sha256']); return
    if args.mode == 'status': print(json.dumps(vr.read(output/'execution_status.json'), ensure_ascii=False)); return
    plan = verify(output, args.full)
    if args.mode == 'verify': print(plan['configuration_sha256'])
    elif args.mode == 'score': print(json.dumps(score(output, plan).get('recorded_tasks'), ensure_ascii=False))
    elif args.mode == 'confirm-detached':
        with patch.object(base, 'verify', lambda path: plan): base.confirm(output)
    elif plan['configuration']['task'] == 'case-repeat': run_repeats(output, plan)
    else:
        with adapted(output, plan): bench.run(output)
        score(output, plan)


if __name__ == '__main__': main()
