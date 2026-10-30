"""Server-local four-way experience-organization comparison.

prepare/verify/score are offline. run executes pure first, then the shared
experience collection, then the three knowledge conditions. All generation is
detached, locked, metered and resumable; terminal content is never resampled.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from rule_framework import flat_skill_rag as p
from scripts import run_rule_skill_comparison as old
from scripts import continuous_context_v2 as quotes

vr, r, runtime, evaluation = old.vr, old.r, old.runtime, old.evaluation
ROOT, require = old.ROOT, p.require
SOURCE_FILES = sorted(set(old.SOURCES) | {'rule_framework/flat_skill_rag.py',
    'scripts/run_organization_comparison.py', 'scripts/resume_continuous_context_v3.py'})


def source_hashes():
    return {n: vr.sha(ROOT / n) for n in SOURCE_FILES}


def prepare(output, parent):
    output.mkdir(parents=True, exist_ok=True)
    with r.run_lock(output / 'active.lock'):
        if (output / 'plan.json').exists():
            verify(output)
            return
        prior = vr.read(parent / 'plan.json')
        c = prior['configuration']
        require(vr.digest(c) == prior['configuration_sha256'], 'Parent plan changed')
        rows = vr.read(parent / 'sources.json')
        require(vr.sha(parent / 'sources.json') == c['sources_sha256'] and len(rows) == 1726,
                'Need the unchanged complete cohort before tree selection')
        require(len({x['id'] for x in rows}) == len(rows), 'Duplicate source')
        asset = old.rules_asset(Path(c['rules']['path']))
        require(asset == c['rules'], 'New rule library changed')
        heldout, registry = old.b.benchmark_questions()
        require(registry == c['benchmarks'] and p.digest(heldout) == c['bench_sha256'], 'Bench registry drift')
        clean, excluded = old.p.isolate_sources([dict(x, answer='') for x in rows], heldout)
        require(not excluded and len(clean) == len(rows), 'Source overlap; no evaluation launch')
        rows = sorted([p.source_payload(x) for x in rows], key=lambda x: vr.digest([p.SEED, x['id']]))
        vr.atomic(output / 'sources.json', rows)
        runtime.frozen_rows()
        evaluation.reference_data(ROOT)
        config = {'protocol': p.VERSION, 'source_parent': str(parent),
            'source_parent_plan_sha256': vr.sha(parent / 'plan.json'),
            'source_parent_sources_sha256': c['sources_sha256'], 'sources_sha256': vr.sha(output / 'sources.json'),
            'source_count': len(rows), 'rules': asset, 'benchmarks': registry, 'bench_sha256': c['bench_sha256'],
            'model': {**{k: c['model'][k] for k in ('api_base', 'attempts_per_job', 'json_policy',
                'max_request_input_bytes', 'model', 'model_files', 'reasoning_effort')},
                'protocol': p.VERSION, 'max_tokens': p.BUILD_LIMITS['output_tokens_per_source'],
                'max_requests': 6000}, 'model_files': runtime.MODEL_FILES,
            'api_base': 'http://127.0.0.1:18083/v1', 'input_sha256': vr.sha(runtime.INPUT),
            'sample_ids': [x['id'] for x in runtime.frozen_rows()], 'arms': list(p.ARMS),
            'limits': {**runtime.LIMITS, 'review_protocol': old.continuous.PROTOCOL},
            'query_max_tokens': p.QUERY_MAX_TOKENS, 'reference_token_cap': p.REFERENCE_TOKEN_CAP,
            'max_items': p.MAX_ITEMS, 'tree_branches': p.TREE_BRANCHES,
            'build_prompt': p.BUILD_PROMPT, 'check_prompt': p.STAGED_BASE,
            'source_hashes': source_hashes(), 'controls': runtime.CONTROLS,
            'budget_limits': {'requests': 6000, 'input_tokens': 1800000000, 'output_tokens': 98304000},
            'prior_usage': old.ZERO_USAGE, 'reference_version': evaluation.RELEASE_VERSION,
            'reference_hashes': {n: vr.sha(ROOT / n) for n in (evaluation.RELEASE, evaluation.REFERENCE)},
            'shared_query': 'one label-free bilingual preparation per input; prefix replayed in all four arms',
            'order': 'pure first; build experiences; alternate the other three arms by sample index',
            'comparison_scope': 'whole organization/extraction/retrieval methods, not isolated tree effect; '
                'pure and assisted conditions run in different phases on the same pinned service',
            'failures': 'no content resampling; exact completed-prefix replay; unknown transport or identity stops',
            'human_gold': False, 'independent_test': False}
        vr.atomic(output / 'plan.json', {'configuration': config, 'configuration_sha256': vr.digest(config)})
        old.emit(output, 'prepared_no_model_calls', planned_checks=200, planned_sources=len(rows))


def verify(output):
    plan = vr.read(output / 'plan.json'); c = plan['configuration']
    require(vr.digest(c) == plan['configuration_sha256'] and c['protocol'] == p.VERSION,
            'Frozen configuration changed')
    require(c['source_hashes'] == source_hashes(), 'Code changed after freeze')
    require(c['sources_sha256'] == vr.sha(output / 'sources.json') and
            c['input_sha256'] == vr.sha(runtime.INPUT), 'Frozen inputs changed')
    parent = Path(c['source_parent'])
    require(vr.sha(parent / 'plan.json') == c['source_parent_plan_sha256'] and
            vr.sha(parent / 'sources.json') == c['source_parent_sources_sha256'], 'Source parent changed')
    require(old.rules_asset(Path(c['rules']['path'])) == c['rules'], 'Rule library changed')
    heldout, registry = old.b.benchmark_questions()
    require(registry == c['benchmarks'] and p.digest(heldout) == c['bench_sha256'], 'Bench registry changed')
    require(c['reference_hashes'] == {n: vr.sha(ROOT / n) for n in c['reference_hashes']}, 'Scoring reference changed')
    return plan


class ExperienceJournal(old.SkillJournal):
    def __init__(self, output, config, tokenizer):
        super().__init__(output, config)
        self.tokenizer = tokenizer

    def messages(self, stage, payload):
        require(stage == 'extract_skill', 'Only free experience extraction is allowed')
        return p.build_messages(payload['row'])

    def __call__(self, stage, key, payload):
        count = len(self.tokenizer.apply_chat_template(self.messages(stage, payload),
                    tokenize=True, add_generation_prompt=True))
        if count + self.config['max_tokens'] > p.BUILD_LIMITS['context_window']:
            raise old.tree.ContentFailure('input_too_large_for_context; no truncation')
        return super().__call__(stage, key, payload)


def build(output, plan, tokenizer):
    folder = output / 'experience'; folder.mkdir(exist_ok=True)
    c, binding = plan['configuration'], plan['configuration_sha256']
    if (folder / 'release_manifest.json').exists():
        return materials(output, plan)
    journal = ExperienceJournal(folder, c, tokenizer)
    if journal.index['files']:
        vr.EvidenceCache(folder / 'evidence_cache.json').check(journal.index['files'])
    progress = vr.Progress(folder / 'collection_checkpoint.json', binding, {'cursor': 0, 'outcome_chain': ''})
    rows = vr.read(output / 'sources.json')
    def accept_pilot():
        pilot, _ = old.read_outcomes(folder, rows[:p.PILOT_SIZE], binding)
        require(all(x['status'] == 'collected' for x in pilot), 'Engineering pilot content protocol failed')
        for item in pilot:
            p.validate_experience(item['value'])
        vr.atomic(folder / 'pilot_acceptance.json', {'configuration_sha256': binding,
            'sources': p.PILOT_SIZE, 'passed': True, 'scope': 'schema/identity/provenance only, not score selection'})
    if progress.state['cursor'] >= p.PILOT_SIZE and not (folder / 'pilot_acceptance.json').exists():
        accept_pilot()
    while progress.state['cursor'] < len(rows):
        require(not (output / 'STOP').exists(), 'STOP requested')
        row = rows[progress.state['cursor']]
        path = old.outcome_path(folder, row)
        if path.exists():
            record = vr.read(path)
            require(record['configuration_sha256'] == binding and record['source_sha256'] == vr.digest(row),
                    'Uncommitted outcome changed')
        else:
            try:
                value = journal('extract_skill', row['id'], {'row': row})
                try:
                    p.validate_experience(value)
                except (ValueError, TypeError, KeyError) as exc:
                    raise old.tree.ContentFailure(str(exc)) from exc
                outcome = {'id': row['id'], 'status': 'collected', 'value': value}
            except old.tree.ContentFailure as exc:
                outcome = {'id': row['id'], 'status': 'content_failure', 'reason': str(exc)}
            record = {'configuration_sha256': binding, 'source_sha256': vr.digest(row), 'outcome': outcome}
            vr.atomic(path, record)
        journal.index['files'][str(path)] = vr.sha(path)
        vr.seal(folder / 'request_index.json', journal.index)
        progress.state['outcome_chain'] = vr.digest([progress.state['outcome_chain'], vr.digest(record)])
        progress.state['cursor'] += 1; progress.save()
        old.emit(output, 'building_flat_experience', processed=progress.state['cursor'], planned=len(rows))
        if progress.state['cursor'] == p.PILOT_SIZE and not (folder / 'pilot_acceptance.json').exists():
            accept_pilot()
    outcomes, chain = old.read_outcomes(folder, rows, binding)
    require(chain == progress.state['outcome_chain'], 'Outcome chain changed')
    prepared = [dict(x, configuration_sha256=binding, source_sha256=vr.digest(row))
                for row, x in zip(rows, outcomes)]
    products = p.materialize(rows, prepared, binding)
    heldout, registry = old.b.benchmark_questions()
    visible = [dict(row, prediction=row['prediction']+'\n'+json.dumps(outcome, ensure_ascii=False), answer='')
               for row, outcome in zip(rows, outcomes)]
    clean, excluded = old.p.isolate_sources(visible, heldout)
    require(registry == c['benchmarks'] and not excluded and len(clean) == len(rows), 'Generated content overlap')
    vr.atomic(folder / 'isolation.json', {'bench_sha256': c['bench_sha256'], 'excluded': excluded,
        'eligible': len(clean), 'semantic_family_independence_proven': False})
    for name in ('skills', 'cases', 'summary'):
        vr.atomic(folder / (name+'.json'), products[name])
    files = {str(folder / (n+'.json')): vr.sha(folder / (n+'.json'))
             for n in ('skills', 'cases', 'summary', 'isolation', 'collection_checkpoint', 'request_index')}
    files.update(journal.index['files'])
    vr.atomic(folder / 'release_manifest.json', {'configuration_sha256': binding, 'files': files,
        'counts': products['summary'], 'human_gold': False, 'physical_quality_passed': False})
    return materials(output, plan)


def materials(output, plan):
    folder = output / 'experience'; release = vr.read(folder / 'release_manifest.json')
    require(release['configuration_sha256'] == plan['configuration_sha256'], 'Experience release changed')
    vr.EvidenceCache(folder / 'evidence_cache.json').check(release['files'])
    return {'flat_skills': p.FlatMemory(vr.read(folder / 'skills.json')),
            'case_rag': p.FlatMemory(vr.read(folder / 'cases.json'))}


def decode_response(raw, **kwargs):
    # The pinned engine sometimes labels an otherwise complete final as queries.
    # Only an exact schema-complete JSON label is normalized; text is unchanged.
    from vllm.entrypoints import harmony_utils
    original = harmony_utils.parse_output_into_messages
    def normalize(ids):
        parsed = original(ids); messages = []
        for m in parsed.messages:
            recipient = m.recipient
            if m.channel == 'final' and recipient == '<|constrain|>queries':
                p.validate_queries(json.loads(''.join(part.text for part in m.content)))
                recipient = '<|constrain|>json'
            messages.append(SimpleNamespace(channel=m.channel, recipient=recipient, content=m.content))
        return SimpleNamespace(messages=messages, current_content=parsed.current_content)
    with patch.object(harmony_utils, 'parse_output_into_messages', normalize):
        return old.comparison.serialization.decode_response(raw, **kwargs)


def query(output, row, plan, tokenizer, client):
    folder = output / 'queries' / vr.digest(row['id'])[:24]
    receipt = folder / 'result.json'; binding = plan['configuration_sha256']
    request, raw, response = [folder / n for n in ('request.json', 'raw.json', 'response.json')]
    if receipt.exists():
        value = vr.read(receipt)
        require(value['configuration_sha256'] == binding and value['input_sha256'] == vr.digest(row), 'Query binding changed')
        for path, sha in value['files'].items():
            require(vr.sha(Path(path)) == sha, 'Query evidence changed')
        return value
    messages = p.query_messages(row)
    if request.exists():
        require(response.exists() and raw.exists(), 'Incomplete prior query; inspect, never resample')
        saved = vr.read(request)['request']; record = vr.read(response)
        require(saved['messages'] == messages and saved['max_tokens'] == p.QUERY_MAX_TOKENS,
                'Prior query does not match frozen input')
    else:
        count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        require(count + p.QUERY_MAX_TOKENS <= p.LIMITS['context_window'], 'Query input too large')
        record = runtime.transport.chat_response(client, endpoint=plan['configuration']['api_base']+'/chat/completions',
            model=r.EXPECTED_MODEL, messages=messages, max_tokens=p.QUERY_MAX_TOKENS, timeout=1200,
            raw_path=raw, request_path=request, response_path=response)
    require(not record.get('error') and record.get('model_ids') == [r.EXPECTED_MODEL], 'Query transport/identity failure')
    queries = p.validate_queries(r.parse_json(record['content']))
    evidence = {'turn': 1, 'request_path': str(request), 'request_sha256': vr.sha(request),
        'raw_sse_path': str(raw), 'raw_sse_sha256': vr.sha(raw), 'response_path': str(response),
        'response_sha256': vr.sha(response), 'model_ids': record['model_ids'], 'finish_reasons': record['finish_reasons'],
        'usage': record['usage'], 'shared_preparation': True}
    value = {'configuration_sha256': binding, 'input_sha256': vr.digest(row), 'queries': queries,
             'evidence': evidence, 'files': {str(f): vr.sha(f) for f in (request, raw, response)}}
    vr.atomic(receipt, value)
    return value


def reference_audit(value, provided):
    violations = []
    for d in value['diagnostics']:
        ids = d.get('reference_ids', [])
        require(isinstance(ids, list) and all(isinstance(x, str) for x in ids), 'Invalid reference IDs')
        if set(ids) - set(provided): violations.append('unsupplied_reference')
    return {'passed': not violations, 'violations': sorted(set(violations)),
            'provided_complete_ids': list(provided), 'adherence_only': True}


@contextmanager
def checking(row, arm, packet, prefix):
    provided = packet['retrieval']['provided_complete_ids']
    def initial(*args, **kwargs):
        usage = prefix['evidence']['usage']
        return {'messages': packet['messages'], 'tool_events': [{'tool': 'provide_reference',
            'arguments': {'ids': provided}, 'result': packet['references']}], 'read_rule_ids': provided,
            'returned_chars': len(json.dumps(packet['references'], ensure_ascii=False)),
            'reserved_output_tokens': usage['completion_tokens'], 'input_tokens': usage['prompt_tokens'],
            'turn': 1, 'pending': None, 'response_evidence': [copy.deepcopy(prefix['evidence'])],
            'visible_output_tokens': 0, 'tree_access': False, 'started_unix': time.time(), 'sample_id': str(row['id'])}
    def audit(value, *args, **kwargs):
        r.validate_diagnostics(value)
        return reference_audit(value, provided)
    def finish(row, inner_arm, model, signature, state, corpus, value, sample_dir, *, consume_response, content=''):
        if consume_response:
            state['messages'].append({'role': 'assistant', 'content': content})
            state['turn'] += 1; state['pending'] = None
        r.write_json(sample_dir / 'state.json', state)
        result = {'id': row['id'], 'model': model, 'verifier': inner_arm, 'organization_arm': arm,
            'status': 'completed', 'checker_status': 'completed', 'diagnostics': copy.deepcopy(value['diagnostics']),
            'configuration_sha256': signature, 'protocol_audit': reference_audit(value, provided),
            'first_diagnosis_path': str(sample_dir/'first_diagnosis.json'),
            'first_diagnosis_sha256': vr.sha(sample_dir/'first_diagnosis.json'),
            'quote_repair': state.get('quote_repair', {'status': 'not_needed'}),
            'retrieval': packet['retrieval'], 'trace': {'model_requests': state['turn'],
                'input_tokens': state['input_tokens'], 'output_token_limit_reserved': state['reserved_output_tokens'],
                'tool_events': state['tool_events'], 'read_rule_ids': provided, 'tree_access': False,
                'response_evidence': state['response_evidence']}}
        r.write_json(sample_dir/'result.json', result)
        return result
    with patch.object(r, 'initial_state', initial), patch.object(old.continuous, 'audit_final', audit), \
         patch.object(old.continuous, 'finish', finish):
        yield


def path_for(output, phase, arm, row):
    return output / phase / arm / 'samples/C0p' / r.hashlib.sha256(str(row['id']).encode()).hexdigest()[:16] / 'result.json'


def run_condition(output, row, phase, arm, memory, plan, tokenizer, client):
    path = path_for(output, phase, arm, row)
    if path.exists():
        saved = vr.read(path)
        require(saved['configuration_sha256'] == plan['configuration_sha256'], 'Existing result plan changed')
        runtime.audit_result(saved)
        require(runtime.healthy_result(saved), 'Preserved failed result requires audited recovery')
        return saved
    prefix = query(output, row, plan, tokenizer, client)
    count = lambda m: len(tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True))
    packet = p.reference_packet(row, arm, prefix['queries'], memory, count)
    folder = path.parent; folder.mkdir(parents=True, exist_ok=True)
    packet_path = folder / 'reference_packet.json'
    packet_record = {'configuration_sha256': plan['configuration_sha256'], 'packet': packet,
                     'query_evidence_sha256': vr.digest(prefix), 'input_sha256': vr.digest(row)}
    if packet_path.exists(): require(vr.read(packet_path) == packet_record, 'Retrieval packet drift')
    else: vr.atomic(packet_path, packet_record)
    with checking(row, arm, packet, prefix):
        result = r.run_sample(row, arm='C0p', catalog={'domains': []}, tokenizer=tokenizer, client=client,
            endpoint=plan['configuration']['api_base']+'/chat/completions', model=r.EXPECTED_MODEL,
            root=output/phase/arm, signature=plan['configuration_sha256'], limits=plan['configuration']['limits'])
    runtime.audit_result(result)
    require(runtime.healthy_result(result), 'Technical failure retained; no reroll')
    require(phase != 'controls' or result['protocol_audit']['passed'], 'Engineering reference protocol failure')
    old.emit(output, phase, arm=arm, sample_id=row['id'], result_status=result['status'])
    return result


def score(output, plan=None):
    plan = plan or verify(output)
    rows = vr.read(runtime.INPUT); _, dataset, unknown = evaluation.reference_data(ROOT)
    groups, scores = {}, {}
    for arm in p.ARMS:
        groups[arm] = []
        for row in rows:
            path = path_for(output, 'formal', arm, row)
            if path.exists():
                value = vr.read(path)
                require(value['configuration_sha256'] == plan['configuration_sha256'], 'Result binding drift')
                runtime.audit_result(value)
                groups[arm].append(value)
        if len(groups[arm]) == 50:
            scores[arm] = evaluation.score(dataset, [dict(x, execution_failure=not runtime.healthy_result(x),
                source_status=x['status'], protocol_noncompliance=not x.get('protocol_audit',{}).get('passed',False))
                for x in groups[arm]], unknown)
    usage = {}
    for arm, items in groups.items():
        usage[arm] = {'completed_records': len(items),
            'questions_with_full_references': sum(bool(x.get('retrieval',{}).get('provided_complete_ids')) for x in items),
            'reference_tokens': sum(x.get('retrieval',{}).get('reference_tokens',0) for x in items),
            'distinct_actual_requests': len({e['request_path'] for x in items for e in x['trace']['response_evidence']})}
    report = {'configuration_sha256': plan['configuration_sha256'], 'scores_all_50': scores,
              'recorded_tasks': {a:len(v) for a,v in groups.items()}, 'usage': usage,
              'reference_version': evaluation.RELEASE_VERSION, 'metric': 'location_proxy_not_semantic_correctness',
              'human_gold': False, 'independent_test': False,
              'shared_preparation_requests_counted_once_in_batch': len({e['request_path'] for items in groups.values()
                   for x in items for e in x['trace']['response_evidence']}),
              'comparisons': {a: evaluation.bootstrap(v, scores['pure']) for a,v in scores.items()
                              if a != 'pure' and 'pure' in scores}}
    vr.atomic(output/'metrics.json',report)
    lines = ['条件 | 完成记录 | Precision | Recall | F1','--- | ---: | ---: | ---: | ---:']
    for arm in p.ARMS:
        summary = scores.get(arm,{}).get('summary')
        vals = ' | '.join(f'{100*summary[k]:.2f}%' for k in ('precision','recall','f1')) if summary else '待完成 | 待完成 | 待完成'
        lines.append(f'{arm} | {len(groups[arm])}/50 | {vals}')
    (output/'metrics.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if len(scores) == 4 and not (output/'case_review.json').exists():
        # Fixed, two-direction selection; do not pick only cases favoring rules.
        selections = []
        for alternative in ('pure','case_rag','flat_skills'):
            selected = old.protocol.case_selection(scores['rule_tree'],scores[alternative])
            selections.extend(dict(x, comparison=['rule_tree',alternative]) for x in selected['selection'])
        inputs = {x['id']:x for x in rows}; refs = {x['id']:x for x in dataset}
        by_arm = {a:{x['id']:x for x in items} for a,items in groups.items()}
        vr.atomic(output/'case_review.json', {'configuration_sha256':plan['configuration_sha256'],
            'human_gold':False,'cases':[dict(x,input=inputs[x['id']],reference=refs[x['id']]['physics_error_gt'],
            outputs={a:by_arm[a][x['id']] for a in p.ARMS},review={'actor':'Codex assistant','status':'pending',
            'angles':['information_preserved_or_lost','scope','tree_filtering','wrong_transfer','quote_only_change'],
            'human_gold':False}) for x in selections]})
    return report


def run(output):
    with r.run_lock(output/'active.lock'), r.run_lock(output.parent/'organization_service_18083.lock'):
        plan = verify(output); c=plan['configuration']; binding=plan['configuration_sha256']
        if vr.read(output/'execution_status.json').get('state') == 'completed':
            score(output,plan); return
        old.bind_runtime(output/'runtime.json',old.server_preflight(output,dict(c,model=r.EXPECTED_MODEL)))
        (output/'runner.pid').write_text(str(os.getpid())); exit_code=1
        try:
            old.b.wait_detached(output,binding)
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(r.EXPECTED_MODEL,local_files_only=True,trust_remote_code=False)
            rows=vr.read(runtime.INPUT)
            with old.comparison.execution.configured(), patch.object(runtime.transport,'decode_response',decode_response), \
                 httpx.Client(trust_env=False) as client:
                for phase,samples in [('controls',runtime.CONTROLS),('formal',rows)]:
                    for row in samples:
                        require(not (output/'STOP').exists(),'STOP requested')
                        run_condition(output,row,phase,'pure',None,plan,tokenizer,client)
                score(output,plan)
                memories=build(output,plan,tokenizer)
                memories['rule_tree']=p.TreeMemory(vr.read(Path(c['rules']['path'])/'catalog.json'))
                arms=['case_rag','flat_skills','rule_tree']
                for phase,samples in [('controls',runtime.CONTROLS),('formal',rows)]:
                    for i,row in enumerate(samples):
                        for arm in arms[i%3:]+arms[:i%3]:
                            require(not (output/'STOP').exists(),'STOP requested')
                            run_condition(output,row,phase,arm,memories[arm],plan,tokenizer,client)
                    score(output,plan)
            old.emit(output,'completed',recorded_tasks=200,case_review='pending');exit_code=0
        except BaseException as exc:
            old.emit(output,'stopped',error_type=type(exc).__name__,error=str(exc)[:1200]);raise
        finally:
            (output/'runner.exit_code').write_text(str(exit_code)+'\n')
            (output/'runner.pid').unlink(missing_ok=True)


def confirm(output):
    plan=verify(output);ready=vr.read(output/'disconnect_ready.json')
    require(runtime.process_identity(ready['pid'])==ready['process_start'] and time.time()-ready['unix']>=5,
            'Reconnect after detaching before confirmation')
    status=vr.read(output/'execution_status.json')
    require(status['state']=='waiting_disconnect_acceptance' and status['token']==ready['token'] and
            ready['configuration_sha256']==plan['configuration_sha256'],'Not waiting on this plan')
    vr.atomic(output/'disconnect_acceptance.json',dict(ready,passed=True,confirmed_unix=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','verify','run','confirm-detached','score','status'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-parent',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if args.mode=='prepare':
        require(args.source_parent is not None,'--source-parent required');prepare(out,args.source_parent.resolve())
    elif args.mode=='verify':print(verify(out)['configuration_sha256'])
    elif args.mode=='run':run(out)
    elif args.mode=='confirm-detached':confirm(out)
    elif args.mode=='score':print(json.dumps(score(out)['recorded_tasks']))
    else:print(json.dumps(vr.read(out/'execution_status.json'),ensure_ascii=False))


if __name__=='__main__':main()
