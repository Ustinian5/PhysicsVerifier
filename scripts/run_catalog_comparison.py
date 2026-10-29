"""Freeze and run old/new rule libraries with the historically validated C4 prompt.

Only ``run`` generates model calls. It runs inside the personal server conda
environment, under a lock, after a fresh SSH disconnect receipt. Existing
libraries, historical results and the C4 prompt implementation stay unchanged.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import httpx

from scripts import run_rule_skill_comparison as shared
from scripts.repair_skill_native_io import decode_tool_payload

vr, r, runtime = shared.vr, shared.r, shared.runtime
comparison, evaluation, continuous = shared.comparison, shared.evaluation, shared.continuous
require, ROOT = shared.require, shared.ROOT
PROTOCOL = 'old_new_catalog_frozen_c4_v1'
ARMS = ('old', 'new')
SELECTED_PLAN = ROOT / 'results/continuous_context_v3_20261013/recovery_002/plan.json'
SELECTED_PLAN_SHA = '9bac483fcd85917dca70c7857dcd1393c81cf39a5f525bdafe19d08f2e0b8770'
SOURCES = sorted(set(shared.SOURCES) | {'scripts/run_catalog_comparison.py',
                                      'scripts/repair_skill_native_io.py'})


@contextmanager
def configured():
    # C4's original prompts/progress/quote-only repair, plus the already audited
    # deterministic Harmony serialization compatibility from its recovery run.
    with comparison.execution.configured(), patch.object(
            runtime.transport, 'decode_response', comparison.serialization.decode_response):
        yield


def selected_protocol():
    require(vr.sha(SELECTED_PLAN) == SELECTED_PLAN_SHA, 'Historical selection evidence changed')
    c = vr.read(SELECTED_PLAN)['configuration']
    with configured():
        require(c['prompts'] == {'base': continuous.BASE, 'rules': continuous.RULES} and
                c['limits'] == runtime.LIMITS and c['controls'] == runtime.CONTROLS and
                c['tools'] == runtime.native_tools(True) and c['model_files'] == runtime.MODEL_FILES,
                'Current C4 implementation differs from the selected historical condition')
    return c


def configuration(rule_tree, *, full=False):
    selected = selected_protocol()
    rows = runtime.frozen_rows()  # Pins all 50 inputs and the original 1,123-leaf catalog.
    evaluation.reference_data(ROOT)
    return {
        'protocol': PROTOCOL, 'arms': list(ARMS),
        'assets': {
            'old': {'catalog_path': str(runtime.CATALOG), 'catalog_sha256': vr.sha(runtime.CATALOG),
                    'leaf_count': 1123, 'known_source_overlap_questions_in_50': 23,
                    'overlapping_questions_traced_to_leaves': 13, 'traced_leaves': 19,
                    'isolation_passed': False},
            'new': dict(shared.rules_asset(rule_tree, full=full),
                        catalog_path=str(rule_tree / 'catalog.json'))},
        'input_sha256': vr.sha(runtime.INPUT), 'sample_ids': [x['id'] for x in rows],
        'model': r.EXPECTED_MODEL, 'model_files': runtime.MODEL_FILES,
        'api_base': 'http://127.0.0.1:18083/v1', 'reasoning_effort': 'medium',
        'temperature': 0, 'top_p': 1, 'limits': selected['limits'],
        'prompts': selected['prompts'], 'tools': selected['tools'], 'controls': selected['controls'],
        'quote_repair': selected['quote_repair'],
        'selection': {'experiment': 'E39', 'historical_plan_sha256': SELECTED_PLAN_SHA,
            'basis': 'highest observed new-reference F1 among completed compatible 120B rule-assisted prompts',
            'precision': 54.054054054054056, 'recall': 29.850746268656714, 'f1': 38.46153846153846,
            'selection_on_development_set': True, 'historical_outputs_reused': False},
        'source_hashes': {n: vr.sha(ROOT / n) for n in SOURCES},
        'reference_version': evaluation.RELEASE_VERSION,
        'reference_sha256': {n: vr.sha(ROOT / n) for n in (evaluation.RELEASE, evaluation.REFERENCE)},
        'order': 'alternate old/new by frozen sample index, including engineering controls',
        'failure_policy': 'retain all failures; continue only an audited complete malformed JSON/action response; '
            'stop on transport, identity, unknown protocol or budget faults; never resample a terminal result',
        'metric': 'frozen_new_reference_location_proxy_not_semantic_accuracy',
        'comparison_scope': 'same online C4 setting; existing library packages differ in sources, size and contamination; '
            'not an isolated construction-method effect or independent paper test',
        'case_policy': {'seed': shared.protocol.SEED, 'categories': ['old_only_hits', 'new_only_hits',
            'shared_miss', 'same_hits_different_findings']},
        'human_gold': False, 'independent_test': False}


def prepare(output, rule_tree):
    output.mkdir(parents=True, exist_ok=True)
    with r.run_lock(output / 'active.lock'):
        if (output / 'plan.json').exists():
            verify(output)
            return
        # The release already has a full-hash acceptance. Reuse its sealed
        # manifest cache: check every file's identity/stat and hash any change,
        # while still validating current sources and the benchmark registry.
        # Do not re-read tens of thousands of unchanged construction responses.
        c = configuration(rule_tree)
        vr.atomic(output / 'plan.json', {'configuration': c, 'configuration_sha256': vr.digest(c)})
        shared.emit(output, 'prepared_no_model_calls', formal_tasks=100)


def verify(output):
    plan = vr.read(output / 'plan.json')
    c = plan['configuration']
    require(plan['configuration_sha256'] == vr.digest(c) and
            c == configuration(Path(c['assets']['new']['path'])), 'Frozen plan/materials/code changed')
    return plan


def result_path(folder, arm, sid):
    short = r.hashlib.sha256(str(sid).encode()).hexdigest()[:16]
    return folder / arm / 'samples/C4' / short / 'result.json'


def collect(folder, rows, binding):
    groups = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for row in rows:
            path = result_path(folder, arm, row['id'])
            if not path.exists():
                continue
            item = vr.read(path)
            require(item['configuration_sha256'] == binding and item['id'] == row['id'] and
                    item['verifier'] == 'C4', 'Paired result identity changed')
            runtime.audit_result(item)
            groups[arm].append(dict(item, catalog_arm=arm, original_result_path=str(path),
                                    original_result_sha256=vr.sha(path)))
    return groups


def terminal_content_failure(item, path):
    """Only a complete, identity-audited response is a terminal model-format miss."""
    if item.get('status') != 'failed':
        return False
    reason = item.get('failure_reason', '')
    allowed = (reason.startswith('JSONDecodeError:') or
               reason.startswith('ValueError: Model output is not valid JSON') or
               reason == 'ValueError: Model JSON must be final diagnostics or one action with arguments')
    evidence = item.get('trace', {}).get('response_evidence', [])
    if not allowed or not evidence:
        return False
    runtime.audit_result(item)
    pending = vr.read(path.parent / 'state.json').get('pending') or {}
    last = evidence[-1]
    return (pending.get('request_path') == last['request_path'] and
            pending.get('response_path') == last['response_path'] and
            last.get('finish_reasons') == ['stop'])


def native_io(client, tokenizer, catalog, output, config):
    """Validate full native tool payload by JSON round-trip, including LaTeX."""
    folder = output / 'native_io'
    receipt = folder / 'acceptance.json'
    if receipt.exists():
        saved = vr.read(receipt)
        require(saved['passed'] and saved['configuration_sha256'] == vr.digest(config), 'IO receipt changed')
        require(all(vr.sha(folder / n) == h for n, h in saved['artifacts'].items()), 'IO evidence changed')
        return
    require(not (folder / 'request.json').exists(), 'Inspect preserved IO attempt; no blind resend')
    corpus = r.Corpus(catalog, tree=True)
    rid = corpus.ids[0]
    messages = continuous.messages(runtime.CONTROLS[1], rules=True, tree_root=corpus.call(
        'read_tree', {'node_id': 'root', 'note': 'Initial tree context'}, max_chars=500000))
    messages.append({'role': 'user', 'content': f'Engineering IO probe only: call read_rules with rule_ids=["{rid}"]. This is not a physics evaluation.'})
    response = runtime.transport.chat_response(client, endpoint=config['api_base'] + '/chat/completions',
        model=r.EXPECTED_MODEL, messages=messages, max_tokens=4096, timeout=1200,
        raw_path=folder / 'raw.json', response_path=folder / 'response.json', request_path=folder / 'request.json',
        native_tools=runtime.native_tools(True), force_tool='read_rules')
    action = json.loads(response['content'])
    require(response['model_ids'] == [r.EXPECTED_MODEL] and response['finish_reasons'] == ['stop'] and
            action['action'] == 'read_rules' and action['arguments'].get('rule_ids') == [rid], 'IO identity/call failed')
    returned = corpus.call('read_rules', action['arguments'], max_chars=500000)
    call = response['tool_calls'][0]
    messages.extend([{'role': 'assistant', 'content': response['reasoning_content'], 'tool_calls': [call]},
                     {'role': 'tool', 'tool_call_id': call['id'], 'name': 'read_rules', 'content': r.canonical(returned)}])
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             tools=runtime.native_tools(True))
    require(continuous.BASE in rendered and continuous.RULES in rendered and
            (not response['reasoning_content'] or response['reasoning_content'] in rendered) and
            decode_tool_payload(rendered) == returned, 'Template lost prompt, context or full rule payload')
    vr.atomic(folder / 'history.json', {'messages': messages, 'rendered_sha256': vr.digest(rendered)})
    vr.atomic(receipt, {'passed': True, 'configuration_sha256': vr.digest(config),
        'usage': response['usage'], 'physics_accuracy_not_gate': True, 'decoded_full_payload_equal': True,
        'artifacts': {n: vr.sha(folder / n) for n in ('request.json', 'raw.json', 'response.json', 'history.json')}})


def run(output):
    with r.run_lock(output / 'active.lock'), configured():
        plan = verify(output)
        c, binding = plan['configuration'], plan['configuration_sha256']
        rows = runtime.frozen_rows()
        found = collect(output / 'formal', rows, binding)
        if all(len(items) == 50 for items in found.values()):
            score(output)
            print('All 100 tasks recorded; no new generation.')
            return
        shared.bind_runtime(output / 'runtime.json', shared.server_preflight(output, c))
        (output / 'runner.pid').write_text(str(os.getpid()))
        exit_code = 1
        try:
            shared.b.wait_detached(output, binding)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(r.EXPECTED_MODEL, local_files_only=True, trust_remote_code=False)
            catalogs = {a: vr.read(Path(c['assets'][a]['catalog_path'])) for a in ARMS}
            with httpx.Client(trust_env=False) as client:
                for arm in ARMS:
                    native_io(client, tokenizer, catalogs[arm], output / 'engineering' / arm, c)
                for phase, samples in [('controls', runtime.CONTROLS), ('formal', rows)]:
                    for i, row in enumerate(samples):
                        for arm in ARMS[i % 2:] + ARMS[:i % 2]:
                            require(not (output / 'STOP').exists(), 'STOP requested')
                            shared.emit(output, phase, sample_id=row['id'], catalog_arm=arm, index=i)
                            result = r.run_sample(row, arm='C4', catalog=catalogs[arm], tokenizer=tokenizer,
                                client=client, endpoint=c['api_base'] + '/chat/completions', model=r.EXPECTED_MODEL,
                                root=output / phase / arm, signature=binding, limits=c['limits'])
                            runtime.audit_result(result)
                            healthy = runtime.healthy_result(result)
                            if phase == 'controls':
                                require(healthy and result['protocol_audit']['passed'], 'Engineering protocol failure retained')
                            else:
                                terminal = terminal_content_failure(result, result_path(output / phase, arm, row['id']))
                                require(healthy or terminal, 'Unclassified failure retained; inspect before recovery')
                            shared.emit(output, phase, sample_id=row['id'], catalog_arm=arm, index=i,
                                        result_status=result['status'], result_saved=True)
                            print(f'{phase} {i + 1}/{len(samples)} {arm} {row["id"]}: {result["status"]}', flush=True)
                score(output)
            shared.emit(output, 'completed_evaluation', recorded_tasks=100)
            exit_code = 0
        except BaseException as exc:
            shared.emit(output, 'stopped', error_type=type(exc).__name__, error=str(exc)[:1200])
            raise
        finally:
            (output / 'runner.exit_code').write_text(str(exit_code) + '\n')
            (output / 'runner.pid').unlink(missing_ok=True)


def compute_summary(items):
    """Correct only the reporting copy of the legacy failed-final turn counter."""
    copied, adjustments = copy.deepcopy(items), []
    for item in copied:
        trace = item['trace']
        actual = max(trace['model_requests'], len(trace.get('response_evidence', [])))
        if actual != trace['model_requests']:
            require(item['status'] == 'failed', 'Unexpected successful request counter mismatch')
            adjustments.append({'id': item['id'], 'recorded': trace['model_requests'], 'actual': actual,
                                'basis': 'distinct hash-audited request/response evidence; original unchanged'})
            trace['model_requests'] = actual
    return dict(comparison.trace_summary(copied), request_counter_adjustments=adjustments)


def score(output):
    plan = verify(output)
    rows = runtime.frozen_rows()
    groups = collect(output / 'formal', rows, plan['configuration_sha256'])
    _, dataset, unknown = evaluation.reference_data(ROOT)
    scores = {}
    for arm, items in groups.items():
        if len(items) == 50:
            scores[arm] = evaluation.score(dataset, [dict(x, execution_failure=not runtime.healthy_result(x),
                source_status=x['status'], protocol_noncompliance=not x.get('protocol_audit', {}).get('passed', False))
                for x in items], unknown)
    report = {'configuration_sha256': plan['configuration_sha256'], 'metric': plan['configuration']['metric'],
        'reference_version': evaluation.RELEASE_VERSION, 'human_gold': False, 'independent_test': False,
        'scores_all_50': scores, 'recorded_tasks': {a: len(v) for a, v in groups.items()},
        'library_use_and_compute': {a: compute_summary(v) for a, v in groups.items()},
        'comparison_scope': plan['configuration']['comparison_scope']}
    if len(scores) == 2:
        report['new_minus_old_paired_bootstrap'] = evaluation.bootstrap(scores['new'], scores['old'])
        cases = shared.protocol.case_selection(scores['old'], scores['new'])
        # The shared selector is label-neutral; translate its original arm names.
        cases = json.loads(json.dumps(cases).replace('rules_only_hits', 'old_only_hits').replace(
            'skills_only_hits', 'new_only_hits'))
        by_input, by_ref = {x['id']: x for x in rows}, {x['id']: x for x in dataset}
        by_arm = {a: {x['id']: x for x in items} for a, items in groups.items()}
        cases['cases'] = [dict(x, input=by_input[x['id']], reference=by_ref[x['id']]['physics_error_gt'],
            unresolved_reference=unknown[x['id']], outputs={a: by_arm[a][x['id']] for a in ARMS},
            review={'actor': 'Codex assistant', 'status': 'pending', 'human_gold': False,
                    'same_physical_error': None, 'knowledge_contribution': None, 'basis': ''})
            for x in cases['selection']]
        review = output / 'case_review.json'
        if review.exists():
            previous = vr.read(review)
            require(previous['configuration_sha256'] == plan['configuration_sha256'] and
                    previous['selection'] == cases['selection'], 'Existing case review changed selection')
        else:
            vr.atomic(review, dict(cases, configuration_sha256=plan['configuration_sha256']))
    vr.atomic(output / 'metrics.json', report)
    return report


def confirm(output):
    ready = vr.read(output / 'disconnect_ready.json')
    require(runtime.process_identity(ready['pid']) == ready['process_start'] and time.time() - ready['unix'] >= 5,
            'Reconnect in a fresh SSH session before confirmation')
    state = vr.read(output / 'execution_status.json')
    require(state['state'] == 'waiting_disconnect_acceptance' and state['token'] == ready['token'], 'Runner not waiting')
    require(ready['configuration_sha256'] == verify(output)['configuration_sha256'], 'Plan changed')
    vr.atomic(output / 'disconnect_acceptance.json', dict(ready, passed=True, confirmed_unix=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify', 'run', 'confirm-detached', 'score', 'status'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rule-tree', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode == 'prepare':
        require(args.rule_tree is not None, '--rule-tree is required')
        prepare(output, args.rule_tree.resolve())
    elif args.mode == 'score':
        print(json.dumps({a: v['summary'] for a, v in score(output)['scores_all_50'].items()}, ensure_ascii=False))
    elif args.mode == 'status':
        print(json.dumps(vr.read(output / 'execution_status.json'), ensure_ascii=False))
    else:
        {'verify': verify, 'run': run, 'confirm-detached': confirm}[args.mode](output)


if __name__ == '__main__':
    main()
