"""Generation only for the approved combined-source 30B benchmark.

Never resumes the old library/checking coordinator. Imported answers require
byte-equivalent request bodies and hash-bound original response evidence.
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time

import httpx

from scripts import rebuild_qwen30b_corpus as corpus
from rule_framework.oss120b_construction import question_key

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'combined_source_30b_bench_generation_v1'
CONFIG_KEYS = ('generator', 'generator_api', 'auditor_api', 'temperature', 'top_p',
               'top_k', 'max_tokens', 'seed', 'technical_attempts',
               'retry_cooldown_seconds', 'timeout_seconds')
SOURCES = ('scripts/run_corrected_step_generation.py', 'scripts/rebuild_qwen30b_corpus.py',
           'scripts/run_gpt_oss_selfcheck.py', 'scripts/run_continuous_context_v2.py',
           'rule_framework/oss120b_construction.py')
read, write, require = corpus.read, corpus.r.write_json, corpus.p.require
sha, digest = corpus.r.sha256_file, corpus.p.digest


def request_body(row, c):
    return dict(model=c['generator'], messages=corpus.solver_messages(row),
                temperature=c['temperature'], top_p=c['top_p'], top_k=c['top_k'],
                max_tokens=c['max_tokens'], stream=False,
                seed=int(digest([c['seed'], row['id']])[:8], 16) % (2**31))


def validate_import(row, c, record, raw):
    require(record['id'] == row['id'] and record['state'] == 'complete', 'Incomplete/wrong answer')
    body = request_body(row, c)
    require(raw['status'] == 'received' and raw['request'] == body, 'Prior generation request differs')
    response = raw['response']
    require(response['model'] == c['generator'] == record['model'], 'Model identity differs')
    require(len(response['choices']) == 1, 'Unexpected choices')
    choice = response['choices'][0]
    require(choice['finish_reason'] == 'stop' == record['finish_reason'] and
            choice['message']['content'] == record['prediction'] and bool(record['prediction'].strip()),
            'Saved answer differs from response')
    require(response['usage'] == record['usage'] and
            0 < response['usage']['completion_tokens'] <= c['max_tokens'], 'Usage differs')


def prepare(out, prior, inventory):
    require(not (out/'plan.json').exists(), 'Already prepared; verify or resume')
    rows = read(prior/'questions.json')
    require(len(rows) == 100 and len({x['id'] for x in rows}) == 100, 'Expected frozen 100 questions')
    source = {question_key(x['question']) for x in read(inventory)['questions']}
    require(all(question_key(x['question']) in source and not x.get('context') for x in rows),
            'Question not present in combined source or unverified context')
    rows = [dict(id=x['id'], question=x['question']) for x in rows]
    previous = read(prior/'plan.json')['configuration']['generator_config']
    require(previous['generator'] == corpus.MODEL and previous['solver_prompt'] == corpus.SOLVER,
            'Solver model/prompt drift')
    c = {k: previous[k] for k in CONFIG_KEYS}
    imports = []
    for row in rows:
        path = prior/'generated'/f"{row['id']}.json"
        if not path.exists():
            continue
        saved = read(path)
        require(saved['signature'] == digest([dict(row, question=row['question']), previous]),
                'Prior answer configuration binding differs')
        corpus.evidence_valid(prior, saved)
        evidence = prior/saved['evidence']['file']
        validate_import(row, c, saved, read(evidence))
        imports.append(dict(id=row['id'], record=os.path.relpath(path, out),
                            sha256=sha(path), evidence=os.path.relpath(evidence, out),
                            evidence_sha256=sha(evidence)))
    plan = dict(protocol=VERSION, configuration=c, solver_prompt=corpus.SOLVER,
                questions_sha256=digest(rows), imports=imports,
                source_inventory_sha256=sha(inventory), source_file='data/combined_language_only.json',
                code_hashes={n: sha(ROOT/n) for n in SOURCES}, downstream_calls_authorized=False)
    with corpus.r.run_lock(out/'active.lock'):
        write(out/'questions.json', rows)
        write(out/'plan.json', dict(configuration=plan, configuration_sha256=digest(plan)))
        write(out/'execution_status.json', dict(state='prepared_no_calls', imported=len(imports),
                                               total=100, remaining=100-len(imports)))
    return verify(out)


def verify(out):
    envelope = read(out/'plan.json'); plan = envelope['configuration']
    require(plan['protocol'] == VERSION and digest(plan) == envelope['configuration_sha256'], 'Plan drift')
    require(plan['solver_prompt'] == corpus.SOLVER and
            digest(read(out/'questions.json')) == plan['questions_sha256'], 'Frozen input/prompt drift')
    require(all(sha(ROOT/n) == h for n, h in plan['code_hashes'].items()), 'Source code drift')
    rows = {r['id']: r for r in read(out/'questions.json')}
    for item in plan['imports']:
        require(sha(out/item['record']) == item['sha256'] and
                sha(out/item['evidence']) == item['evidence_sha256'], 'Imported evidence changed')
        validate_import(rows[item['id']], plan['configuration'], read(out/item['record']),
                        read(out/item['evidence']))
    return envelope


def confirm(out):
    plan = verify(out); ready = read(out/'disconnect_ready.json')
    require(corpus.runtime.process_identity(ready['pid']) == ready['process_start'] and
            ready['configuration_sha256'] == plan['configuration_sha256'] and
            time.time()-ready['unix'] >= 5, 'Fresh SSH disconnect confirmation required')
    write(out/'disconnect_acceptance.json', dict(ready, passed=True, confirmed_unix=time.time()))


def run(out):
    require(Path('/proc').exists(), 'Server-only generation')
    with corpus.r.run_lock(out/'active.lock'), corpus.r.run_lock(out.parent/'generation.lock'):
        envelope = verify(out); plan = envelope['configuration']; c = plan['configuration']
        (out/'runner.pid').write_text(str(os.getpid())); code = 1
        try:
            require(not (out/'STOP').exists(), 'STOP requested')
            corpus.runtime.wait_detached(out, envelope['configuration_sha256'])
            deadline = time.monotonic()+14400
            with httpx.Client(trust_env=False, timeout=10) as client:
                while True:
                    require(not (out/'STOP').exists(), 'STOP requested')
                    try:
                        response = client.get(c['generator_api']+'/models')
                        response.raise_for_status()
                        data = response.json()['data']
                        require(any(x['id'] == c['generator'] and x.get('max_model_len', 0) >= 32768
                                    and x.get('root') == '/slow_share2/huhengyuan/models/Qwen3-30B-A3B-Instruct-2507'
                                    for x in data), 'Service model/context mismatch')
                        break
                    except httpx.TransportError:
                        require(time.monotonic() < deadline, 'Service readiness timed out')
                        write(out/'execution_status.json', dict(state='waiting_model', updated_unix=time.time()))
                        time.sleep(10)
            imports = {r['id']: r for r in plan['imports']}
            journal = corpus.Journal(out, c, threading.Event()); result = []; failures = []
            for row in read(out/'questions.json'):
                require(not (out/'STOP').exists(), 'STOP requested')
                if row['id'] in imports:
                    saved = read(out/imports[row['id']]['record'])
                else:
                    saved = corpus.generate(journal, row)
                if saved['state'] == 'complete':
                    result.append(dict(row, context='', prediction=saved['prediction']))
                else:
                    failures.append(dict(id=row['id'], state=saved['state']))
                write(out/'inputs.json', result)
                write(out/'execution_status.json', dict(state='generating', completed=len(result),
                      failed=len(failures), imported=len(imports), total=100, updated_unix=time.time()))
            write(out/'generation_summary.json', dict(complete=len(result), failures=failures,
                  imported=len(imports), total=100, downstream_started=False))
            write(out/'execution_status.json', dict(state='generation_complete' if not failures else 'generation_with_failures',
                  completed=len(result), failed=len(failures), imported=len(imports), total=100, updated_unix=time.time()))
            code = 0
        except BaseException as exc:
            write(out/'execution_status.json', dict(state='stopped', error=str(exc)[:1000], updated_unix=time.time()))
            raise
        finally:
            (out/'runner.exit_code').write_text(str(code)); (out/'runner.pid').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify', 'run', 'confirm-detached'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--inventory', type=Path)
    args = parser.parse_args(); out = args.output.resolve()
    if args.mode == 'prepare':
        require(args.prior is not None and args.inventory is not None, 'Prior/inventory required')
        plan = prepare(out, args.prior.resolve(), args.inventory.resolve())
        print(json.dumps(dict(imported=len(plan['configuration']['imports']), total=100)))
    elif args.mode == 'verify': verify(out); print('verified')
    elif args.mode == 'confirm-detached': confirm(out)
    else: run(out)


if __name__ == '__main__': main()
