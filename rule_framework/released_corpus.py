"""Read-only, evidence-checked handoff from the fresh 30B corpus to construction."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import hashlib
from pathlib import Path
import stat
import threading
import time

from rule_framework import oss120b_construction as p


def load(release_path, source_path, *, full=False, complete_answers=False):
    # Lazy import: the corpus runner imports build_120b_rules for shared IO.
    from scripts import rebuild_qwen30b_corpus as corpus
    from scripts import run_gpt_oss_selfcheck as r
    root = Path(release_path).resolve().parent
    release_path, source_path = Path(release_path).resolve(), Path(source_path).resolve()
    release = json.loads(release_path.read_text())
    p.require(release.get('protocol') == corpus.VERSION and release.get('human_gold') is False
              and release.get('counts', {}).get('state') == 'completed', 'Corpus is not a completed machine release')
    source_name = 'answers.json' if complete_answers else 'wrong_answers.json'
    p.require(source_path == root / source_name, 'Use the exact released source without substitution')
    files = release['files']
    checked_dirs = {Path('.')}
    directory_guard = threading.Lock()
    verified_json = {}
    cache_guard = threading.Lock()

    def safe_file(name):
        relative = Path(name)
        p.require(not relative.is_absolute() and '..' not in relative.parts and name in files,
                  'Unbound/outside corpus artifact: ' + name)
        # Resolving every ancestor for every file causes >100,000 NFS metadata
        # round trips. Validate each distinct directory once, refusing symlinks.
        for parent in reversed(relative.parents):
            with directory_guard:
                known = parent in checked_dirs
            if not known:
                p.require(stat.S_ISDIR((root / parent).lstat().st_mode), 'Corpus parent is not a real directory')
                with directory_guard:
                    checked_dirs.add(parent)
        path = root / relative
        p.require(stat.S_ISREG(path.lstat().st_mode), 'Corpus artifact is not a regular file')
        return path

    def checked(name):
        if complete_answers:
            with cache_guard:
                if name in verified_json:
                    return verified_json[name]
        data = safe_file(name).read_bytes()
        p.require(hashlib.sha256(data).hexdigest() == files[name], 'Corpus artifact changed: ' + name)
        value = json.loads(data)
        if complete_answers:
            with cache_guard:
                verified_json[name] = value
        return value

    if full:
        # Bounded read concurrency avoids serial NFS metadata latency; no writes.
        active, guard = {}, threading.Lock()
        def checksum(name):
            with guard:
                active[name] = time.monotonic()
            try:
                if complete_answers:
                    checked(name)  # hash and parse once, then reuse verified RAM
                else:
                    path = safe_file(name)
                    p.require(r.sha256_file(path) == files[name],
                              'Corpus release checksum failed: ' + name)
            finally:
                with guard:
                    active.pop(name, None)
        # The new flow does not consume old audit judgments. Verify every
        # generation attempt/result plus its frozen inputs, not unused audit
        # payloads; the old full wrong-answer handoff remains unchanged.
        names_to_check = [n for n in files if not complete_answers or
            n in {'plan.json', 'physics_inputs.json', 'answers.json'} or
            n.startswith(('generated/', 'raw/generate/'))]
        with ThreadPoolExecutor(max_workers=64 if complete_answers else 32) as pool:
            pending = {pool.submit(checksum, name) for name in names_to_check}
            last = time.monotonic()
            while pending:
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                if time.monotonic() - last >= 30 or not pending:
                    with guard:
                        blocked = {k: round(time.monotonic() - v) for k, v in active.items()}
                    print(json.dumps({'source_checksums': len(names_to_check)-len(pending), 'total': len(names_to_check),
                                      'reading_seconds': blocked}), flush=True)
                    last = time.monotonic()
    plan = checked('plan.json'); config = plan['configuration']
    p.require(p.digest(config) == plan['configuration_sha256'] == release['configuration_sha256'],
              'Corpus configuration mismatch')
    p.require(config['generator'] == release['generator'] == 'Qwen3-30B-A3B-Instruct-2507'
              and config['auditor'] == release['auditor'] == r.EXPECTED_MODEL
              and config['reasoning_effort'] == 'medium' and config['samples_per_question'] == 1
              and config['human_gold'] is False and config['old_answers_used'] is False
              and config['reference_answers_used'] is False, 'Corpus model/source policy changed')
    p.require(config['prompts_sha256'] == p.digest([corpus.SOLVER, corpus.AUDITOR, corpus.DOMAIN_SCREEN]),
              'Corpus prompt decoder no longer matches its frozen version')
    inputs = {x['id']: x for x in checked('physics_inputs.json')}
    rows = checked(source_name)
    expected_count = release['counts']['generation']['complete'] if complete_answers else release['counts']['wrong_answers']
    if complete_answers:
        with ThreadPoolExecutor(max_workers=64 if complete_answers else 8) as pool:
            generations = list(pool.map(lambda sid: checked('generated/' + sid + '.json'), inputs))
        p.require({x['id'] for x in generations if x['state'] == 'complete'} == {x['id'] for x in rows},
                  'Complete-answer cohort was shortened or substituted')
    p.require(len(rows) == expected_count and
              len({x['id'] for x in rows}) == len(rows), 'Corpus candidate count/IDs mismatch')

    def validate(row):
        sid = row['id']; p.require(sid in inputs, 'Candidate outside frozen physics cohort')
        original = inputs[sid]
        p.require(row['question'] == original['question'] and not row.get('answer')
                  and not row.get('context'), 'Candidate changed question or includes external reference')
        gen = checked('generated/' + sid + '.json')
        audit = None if complete_answers else checked('audited/' + sid + '.json')
        p.require(gen['id'] == sid and gen['state'] == 'complete'
                  and gen['model'] == config['generator'] and gen['prediction'] == row['prediction']
                  and gen['signature'] == p.digest([original, config]), 'Generator result/input mismatch')
        if not complete_answers:
            p.require(audit['id'] == sid and audit['state'] == 'error_found'
                      and audit['model'] == config['auditor'] and audit['human_gold'] is False
                      and audit['signature'] == p.digest([original, row['prediction'], config]),
                      'Auditor result/input mismatch')
        for result in ((gen,) if complete_answers else (gen, audit)):
            evidence = result['evidence']; record = checked(evidence['file'])
            p.require(evidence['sha256'] == files[evidence['file']]
                      and record['status'] == 'received'
                      and record['response']['model'] == result['model'], 'Raw response/identity mismatch')
            if result is gen:
                expected = {'model': config['generator'], 'messages': corpus.solver_messages(original),
                    'temperature': config['temperature'], 'top_p': config['top_p'], 'top_k': config['top_k'],
                    'max_tokens': config['max_tokens'], 'stream': False,
                    'seed': int(p.digest([config['seed'], sid])[:8], 16) % (2**31)}
                p.require(record['request'] == expected, 'Generation request changed')
                choice = record['response']['choices'][0]
                p.require(choice['finish_reason'] == 'stop' and
                          choice['message']['content'] == row['prediction'], 'Generation text changed/truncated')
            else:
                expected = r.request_body(config['auditor'], corpus.audit_messages(original, row['prediction']),
                                          config['max_tokens'], backend='vllm')
                expected.update(stream=False, return_token_ids=True, reasoning_effort='medium')
                expected.pop('stream_options', None)
                p.require(record['request'] == expected, 'Audit request changed')
                decoded = corpus.harmony.decode_response(record['response'], model=config['auditor'],
                    max_tokens=config['max_tokens'], native_tools=None)['content'].strip()
                if decoded.startswith('```json\n') and decoded.endswith('```'):
                    decoded = decoded[8:-3].strip()
                raw = json.loads(decoded)
                p.require(raw == audit['raw_judgment'], 'Audit decoding changed')
                actual = corpus.classify_audit(raw, row['prediction'])
                p.require(all(audit[k] == v for k, v in actual.items()), 'Admitted claims changed')
        if complete_answers:
            return sid, {'generation_evidence': gen['evidence'], 'prediction_sha256': p.digest(row['prediction'])}
        # Only accepted claims/uncertainty reach construction; rejected raw claims never do.
        return sid, {k: audit[k] for k in ('state', 'errors', 'uncertain', 'human_gold', 'evidence')}

    with ThreadPoolExecutor(max_workers=64 if complete_answers else 8) as pool:
        audits = dict(pool.map(validate, rows))
    if complete_answers:
        return {'release_sha256': r.sha256_file(release_path),
            'configuration_sha256': release['configuration_sha256'], 'answers_sha256': files[source_name],
            'generator': config['generator'], 'complete_answers': len(rows), 'bound_files': len(files),
            'generation_evidence_sha256': p.digest(audits), 'human_gold': False,
            'source_model_evidence': 'all_complete_generation_requests_and_responses_verified',
            'verification_scope': 'all complete answer generation inputs, results and raw responses; old judgments excluded',
            'claim_policy': 'fresh_continuous_review_no_old_judgment_imported'}, rows
    binding = {'release_sha256': r.sha256_file(release_path),
        'configuration_sha256': release['configuration_sha256'],
        'wrong_answers_sha256': files['wrong_answers.json'], 'generator': config['generator'],
        'auditor': config['auditor'], 'source_audits_sha256': p.digest(audits),
        'candidate_questions': len(rows), 'accepted_claims': sum(len(x['errors']) for x in audits.values()),
        'bound_files': len(files), 'human_gold': False,
        'source_model_evidence': 'released_generation_and_audit_requests_verified',
        'claim_policy': 'reuse_only_admitted_claims_no_new_audit_no_rejected_claim_repair'}
    return binding, audits
