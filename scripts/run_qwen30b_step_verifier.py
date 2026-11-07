"""Original-source 30B step-verifier experiment. No historical answer/GT reuse.

prepare is offline; generate is detached server-only and stops after the pilot.
The construction corpus is immutable: potentially related evaluation questions
are excluded BEFORE generating answers, never on the basis of model outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import threading
import time
import httpx

from scripts import rebuild_qwen30b_corpus as corpus
from scripts import run_organization_comparison as comparison
from rule_framework import organization_followup as knowledge

ROOT = Path(__file__).resolve().parents[1]
vr, require = comparison.vr, comparison.require
VERSION = 'original_source_30b_steps_v1'
SOURCE = 'data/derived/catalog_compare_seed20260508_test200_20260901/qa_chain/combined_language_main_test.json'
QUESTIONS = 'results/bench_reannotation_99_20261011/inputs.json'
BUILD = 'results/oss120b_experience_first_v5'
PILOT = 6


def solver_row(row):
    """Context belongs to the problem; answers/rubrics never reach the solver."""
    context = row.get('context', '')
    return {'id': row['id'], 'question': (context + '\n\n' if context else '') + row['question']}


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    with corpus.r.run_lock(output/'active.lock'):
        require(not (output/'plan.json').exists(), 'Already prepared; use verify, do not overwrite')
        candidates = vr.read(ROOT/QUESTIONS)
        original = {x['id']: x for x in vr.read(ROOT/SOURCE)}
        require(all(x['id'] in original and x['prediction'] == original[x['id']]['prediction']
                    and x['question'] == original[x['id']]['question'] for x in candidates),
                'Historical question/source mapping differs')
        # Drop the historical predictions and ALL historical error annotations.
        rows = [{k: x.get(k, '') for k in ('id', 'question', 'context', 'answer')} for x in candidates]
        screen = knowledge.FamilyScreen(rows)
        build_rows = vr.read(ROOT/BUILD/'inputs.json')
        excluded = {}
        for i, source in enumerate(build_rows):
            for match in screen.matches(source):
                if match['heldout_id'] in original:
                    excluded.setdefault(match['heldout_id'], []).append(dict(source_id=source['id'], **match))
            if i % 200 == 0:
                print('isolation', i, len(build_rows), flush=True)
        seen = set(); selected = []
        for row in sorted(rows, key=lambda x: vr.digest([20261107, x['id']])):
            key = knowledge.previous.isolation.question_template(row['question'])
            if row['id'] in excluded:
                continue
            if key in seen:
                excluded[row['id']] = [{'reason': 'duplicate_evaluation_template'}]
                continue
            # Pairwise heldout family screen, independent of labels or answers.
            relatives = knowledge.FamilyScreen(selected).matches(row) if selected else []
            if relatives:
                excluded[row['id']] = relatives
                continue
            seen.add(key); selected.append(row)
        require(len(selected) >= PILOT, 'Insufficient isolated pilot questions')
        inputs = [{k: x[k] for k in ('id','question','context')} for x in selected]
        references = [{'id': x['id'], 'historical_reference_answer': x['answer'],
                       'authority': 'unverified original reference, not a 30B label'} for x in selected]
        vr.atomic(output/'questions.json', inputs)
        vr.atomic(output/'reference_answers.json', references)
        vr.atomic(output/'isolation.json', {'screen': 'FamilyScreen shared-prose + numeric + lexical',
            'source_questions': len(build_rows), 'candidates': len(rows), 'retained': len(inputs),
            'excluded': excluded, 'independent_test': False,
            'reason': 'historical development questions, fresh 30B answers; lexical screening is not semantic proof'})
        files = [SOURCE, QUESTIONS, BUILD+'/inputs.json', BUILD+'/catalog.json',
                 BUILD+'/semantic_experience.json', BUILD+'/release_manifest.json',
                 'results/qwen30b_corpus/source_manifest.json', 'results/qwen30b_corpus/release_manifest.json']
        c = {'protocol': VERSION, 'generator': corpus.MODEL,
             'generator_api': 'http://127.0.0.1:18084/v1', 'auditor': corpus.r.EXPECTED_MODEL,
             'auditor_api': 'http://127.0.0.1:18083/v1', 'temperature': .7, 'top_p': .8, 'top_k': 20,
             'max_tokens': 16384, 'seed': 20261107, 'technical_attempts': 3,
             'retry_cooldown_seconds': 60, 'timeout_seconds': 1200,
             'reasoning_effort': 'medium', 'fixed_harmony_date': '2026-11-07',
             'pilot_ids': [x['id'] for x in inputs[:PILOT]], 'sample_ids': [x['id'] for x in inputs],
             'frozen_files': {n: vr.sha(output/n) for n in ('questions.json','reference_answers.json','isolation.json')},
             'source_files': {n: vr.sha(ROOT/n) for n in files},
             'solver_prompt': corpus.SOLVER, 'solver_context': 'original question and provided context only',
             'source_code': {n: vr.sha(ROOT/n) for n in (
                 'scripts/run_qwen30b_step_verifier.py','scripts/rebuild_qwen30b_corpus.py',
                 'rule_framework/organization_followup.py','rule_framework/organization_bench99.py')},
             'annotation': 'new independent 30B step labels required; no inference authorized by prepare',
             'score': 'step diagnosis matching; never relabel an old 4B span as 30B gold',
             'independent_test': False, 'human_gold': False}
        vr.atomic(output/'plan.json', {'configuration': c, 'configuration_sha256': vr.digest(c)})
        vr.atomic(output/'execution_status.json', {'state': 'prepared_no_calls', 'questions': len(inputs),
                                                  'pilot_questions': PILOT, 'reference_labels_ready': False})
        print(json.dumps({'retained': len(inputs), 'excluded': len(excluded), 'pilot': PILOT}))


def verify(output):
    plan = vr.read(output/'plan.json'); c = plan['configuration']
    require(c['protocol'] == VERSION and vr.digest(c) == plan['configuration_sha256'], 'Plan drift')
    for name, sha in c['frozen_files'].items():
        require(vr.sha(output/name) == sha, 'Input drift: '+name)
    for name, sha in c['source_code'].items():
        if vr.sha(ROOT/name) != sha:
            recovery=vr.read(output/'generation_recovery.json')
            require(name=='scripts/run_qwen30b_step_verifier.py' and
                    recovery['configuration_sha256']==plan['configuration_sha256'] and
                    recovery['previous_sha256']==sha and recovery['replacement_sha256']==vr.sha(ROOT/name)
                    and recovery['requests_before_recovery']==0 and
                    recovery['policy']=='generation_readiness_only_remove_unused_auditor_dependency',
                    'Code drift: '+name)
    require(c['solver_prompt'] == corpus.SOLVER, 'Solver prompt drift')
    return plan


def export_generated(output, plan):
    refs = {r['id']: r for r in vr.read(output/'reference_answers.json')}
    inputs, annotations, failed = [], [], []
    for row in vr.read(output/'questions.json'):
        path = output/'generated'/(row['id']+'.json')
        if not path.exists(): continue
        record = vr.read(path); corpus.evidence_valid(output, record)
        require(record['model'] == plan['configuration']['generator'] and
                record['signature'] == corpus.p.digest([solver_row(row), plan['configuration']]),
                'Generated answer binding differs')
        if record['state'] != 'complete':
            failed.append({'id': row['id'], 'state': record['state']}); continue
        item = dict(row, prediction=record['prediction'])
        inputs.append(item)
        annotations.append(dict(item, answer=refs[row['id']]['historical_reference_answer']))
    vr.atomic(output/'inputs.json', inputs)  # checker sees no answers or labels
    vr.atomic(output/'annotation_inputs.json', annotations)
    vr.atomic(output/'generation_summary.json', {'complete': len(inputs), 'failed': failed,
                                               'labels_created': 0, 'human_gold': False})
    return inputs


def confirm(output):
    plan = verify(output); ready = vr.read(output/'disconnect_ready.json')
    require(corpus.runtime.process_identity(ready['pid']) == ready['process_start'] and
            time.time()-ready['unix'] >= 5 and ready['configuration_sha256'] == plan['configuration_sha256'],
            'Reconnect to accept the same live process')
    vr.atomic(output/'disconnect_acceptance.json', dict(ready, passed=True, confirmed_unix=time.time()))


def generate(output, full=False):
    require(Path('/proc').exists(), 'Generation is server-only')
    with corpus.r.run_lock(output/'active.lock'), corpus.r.run_lock(output.parent/'qwen30b_step_generation.lock'):
        plan = verify(output); c = plan['configuration']
        if full:
            approval = vr.read(output/'pilot_acceptance.json')
            require(approval.get('configuration_sha256') == plan['configuration_sha256'] and
                    approval.get('passed') is True, 'Pilot engineering acceptance required')
        (output/'runner.pid').write_text(str(os.getpid()))
        code = 1
        try:
            corpus.runtime.wait_detached(output, plan['configuration_sha256'])
            deadline=time.monotonic()+14400
            with httpx.Client(trust_env=False,timeout=10) as client:
                while True:
                    require(not (output/'STOP').exists(),'STOP requested')
                    try: response=client.get(c['generator_api']+'/models')
                    except httpx.TransportError:
                        require(time.monotonic()<deadline,'30B readiness deadline')
                        time.sleep(10);continue
                    response.raise_for_status()
                    require(any(x['id']==c['generator'] and x.get('max_model_len',0)>=32768 and
                                x.get('root')=='/slow_share2/huhengyuan/models/Qwen3-30B-A3B-Instruct-2507'
                                for x in response.json()['data']),'30B service identity/context drift')
                    break
            journal = corpus.Journal(output, c, threading.Event())
            rows = vr.read(output/'questions.json')
            if not full: rows = [x for x in rows if x['id'] in c['pilot_ids']]
            for row in rows:
                record = corpus.generate(journal, solver_row(row))
                vr.atomic(output/'execution_status.json', {'state': 'generating', 'last_id': row['id'],
                    'last_state': record['state'], 'updated_unix': time.time()})
            export_generated(output, plan)
            vr.atomic(output/'execution_status.json', {'state': 'generation_complete' if full else 'pilot_generated',
                'reference_labels_ready': False, 'updated_unix': time.time()})
            code = 0
        except BaseException as exc:
            vr.atomic(output/'execution_status.json', {'state': 'stopped', 'error': str(exc)[:1000]})
            raise
        finally:
            (output/'runner.exit_code').write_text(str(code))
            (output/'runner.pid').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','verify','generate','confirm-detached','export'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args(); output = args.output.resolve()
    if args.command == 'prepare': prepare(output)
    elif args.command == 'verify': verify(output); print('verified')
    elif args.command == 'confirm-detached': confirm(output)
    elif args.command == 'generate': generate(output, args.full)
    else: export_generated(output, verify(output))


if __name__ == '__main__':
    main()
