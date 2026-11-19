"""Revise final-answer labels, then evaluate the current four knowledge arms.

Server-only, detached and locked.  Original generations, labels and 2400 old
outputs are never edited.  Unresolved label disagreements gate publication;
classification failures remain failures rather than fabricated negative labels.
"""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import httpx

from rule_framework import final_answer_revision as p
from scripts import run_aligned_step_revision as step
from scripts import run_final_answer_labels as old_labels
from scripts.audit_final_answer_metrics import measure
from scripts.run_qwen30b_step_checks import validate_effective_input

a = step.old.a
base, vr, require = a.base, a.vr, a.require
ARMS = base.p.ARMS
RUNNER = 'scripts/run_final_answer_revision.py'
PREPARED_CONFIGURATION = '9aab6152e80c8a1920e38867a64b5657c8f7482e2107b8b8e303fe43de7b965e'
PREPARED_RUNNER = 'e4eac6bf581324ffaa9f97505d39fd56d9ae654e091255a1485708180ee3d1fd'


def immutable(path, value):
    if path.exists():
        require(vr.read(path) == value, 'Refusing to overwrite frozen evidence: ' + str(path))
    else:
        vr.atomic(path, value)


def bind(files, path):
    path = Path(path)
    files[str(path)] = vr.sha(path)
    return vr.read(path)


def source_hashes():
    values = step.source_files()
    for n in ('rule_framework/final_answer_revision.py', 'rule_framework/final_answer_protocol.py',
              'scripts/run_final_answer_revision.py', 'scripts/audit_final_answer_metrics.py',
              'scripts/run_final_answer_labels.py'):
        values[n] = vr.sha(a.ROOT / n)
    return values


def control_rows():
    rows = [dict(x, context='') for x in old_labels.CONTROLS]
    rows.append(dict(id='control_reference_conflict', context='',
        question='A body travels 100 metres in 10 seconds at constant speed. Find its speed.',
        prediction='The final speed is 10 m/s.', reference=['20 m/s'], expected='uncertain'))
    rows.append(dict(id='control_missing_topology', context='',
        question='In the missing circuit diagram, find the voltage gain in terms of R1 and R2.',
        prediction='The gain is 1+R2/R1.', reference=['1+R2/R1'], expected='uncertain'))
    return rows


def prepare(out, root):
    require(Path('/proc').exists(), 'Formal preparation runs on the server')
    require(not (out / 'plan.json').exists(), 'Revision already prepared')
    current = root / 'aligned_old99/evaluation_revision'
    cp = step.verify(current)['configuration']
    archive = root / 'corrected/pipeline/check_prompt_revision_bilingual'
    src = root / 'corrected'
    files = {}
    questions = bind(files, src / 'final_labels/questions.json')
    refs = {x['id']: x for x in bind(files, src / 'final_labels/references_private.json')}
    lc = bind(files, src / 'final_labels/plan.json')['configuration']
    old_rows = bind(files, archive / 'final_inputs.json')
    old_gold = bind(files, archive / 'final_labels_private.json')
    bind(files, archive / 'final_metrics.json')
    rows = []
    for question in questions:
        sid = question['id']; generation = bind(files, src / 'collection/generated' / (sid + '.json'))
        require(generation['id'] == sid and
                generation['signature'] == vr.digest([question, lc['generator_config']]) and
                generation['model'] == lc['generator_config']['generator'], '30B input/model binding mismatch')
        ev = generation['evidence']; path = src / 'collection' / ev['file']
        require(vr.sha(path) == ev['sha256'], '30B raw evidence drift'); files[str(path)] = ev['sha256']
        row = dict(id=sid, question=question['question'], prediction=generation['prediction'], context='',
                   reference=refs[sid]['reference'], generation_status=generation['state'],
                   question_sha256=vr.digest(question['question']), answer_sha256=vr.digest(generation['prediction']),
                   reference_sha256=vr.digest(refs[sid]['reference']))
        label_path = src / 'final_labels/labels' / (sid + '.json')
        if label_path.exists():
            label = bind(files, label_path)
            raw = src / 'final_labels' / label['evidence']['file']
            require(vr.sha(raw) == label['evidence']['sha256'], 'Original label evidence drift')
            files[str(raw)] = label['evidence']['sha256']
            require(label['signature'] == vr.digest([dict(question, prediction=row['prediction']), row['reference'], lc]),
                    'Original label binds another answer/reference')
            row['old_verdict'] = label['judgment']['verdict'] if label['state'] == 'complete' else None
        if generation['state'] != 'complete':
            row['eligibility_failure'] = 'original_30b_generation_incomplete'
        elif not row['reference'] or any(not s.strip() for s in row['reference']):
            row['eligibility_failure'] = 'original_reference_missing'
        rows.append(row)
    require(len(rows) == 702 and len({r['id'] for r in rows}) == 702, 'Frozen candidate inventory drift')
    byid = {r['id']: r for r in rows}
    require(len(old_rows) == 600 and all(p.public(r) == p.public(byid[r['id']]) for r in old_rows),
            'Old 600 use different 30B answers')
    require(all(old_gold[sid]['verdict'] == byid[sid]['old_verdict'] for sid in old_gold), 'Old gold mismatch')
    for name in ('cases', 'skills', 'catalog'):
        value = bind(files, current / (name + '.json'))
        immutable(out / (name + '.json'), value)
    for name in ('plan', 'source_quarantine', 'acceptance'):
        bind(files, current / (name + '.json'))
    for name in ('sources', 'isolation', 'bench_questions'):
        bind(files, current.parent / 'evaluation' / (name + '.json'))
    for name in ('split', 'family_components'):
        bind(files, src / 'source_screen' / (name + '.json'))
    # Remaining original reserve rows must be covered by the same source filter
    # before they can replace a quarantined old benchmark item.
    heldout = [dict(id='final:' + r['id'], question=r['question']) for r in rows]
    heldout += vr.read(current.parent / 'evaluation/bench_questions.json')
    screen = a.bounded.FamilyScreen(heldout)
    cases = vr.read(out / 'cases.json'); skills = vr.read(out / 'skills.json')
    sources = vr.read(current.parent / 'evaluation/sources.json')
    existing_ids = {sid for d in cases for sid in d['source_ids']}
    source_map = {r['id']: r for r in sources}
    candidate_overlaps = []
    for sid in sorted(existing_ids):
        material = [d['text'] for d in cases + skills if sid in d['source_ids']]
        matches = screen.matches(dict(source_map[sid], prediction='\n'.join(material)))
        if matches:
            candidate_overlaps.append(dict(source_id=sid, matches=matches))
    tree = base.p.TreeMemory(vr.read(out / 'catalog.json'))
    for rid, rule in tree.corpus.leaves.items():
        matches = screen.matches(dict(id=rid, question='', prediction=json.dumps(rule, ensure_ascii=False)))
        if matches:
            candidate_overlaps.append(dict(rule_id=rid, matches=matches))
    immutable(out / 'isolation_candidates.json', dict(candidates=candidate_overlaps,
        policy=a.bounded.ISOLATION_POLICY, scope='702 candidate questions plus existing heldouts against retained source/case/skill texts and all rule leaves',
        semantic_family_independence_proven=False, known_watermelon_source_absent=(step.old.KNOWN_SOURCE not in existing_ids)))
    require(step.old.KNOWN_SOURCE not in existing_ids, 'Known exposed source still retained')
    immutable(out / 'candidates.json', rows)
    pilot = copy.deepcopy(cp['pilot_inputs'])
    require(len(pilot) == 6 and not {r['id'] for r in pilot} & set(byid), 'Development/test overlap')
    immutable(out / 'pilot_inputs.json', pilot)
    cfg = dict(protocol=p.VERSION, root=str(root), archive=str(archive), material_parent=str(current),
        source_hashes=source_hashes(), dependencies=files,
        candidates_sha256=vr.sha(out/'candidates.json'), pilot_sha256=vr.sha(out/'pilot_inputs.json'),
        material_hashes={name: vr.sha(out/(name+'.json')) for name in ('cases','skills','catalog')},
        isolation_sha256=vr.sha(out/'isolation_candidates.json'), contract=p.contract(),
        limits=cp['limits'], api_base=cp['api_base'], model=base.r.EXPECTED_MODEL,
        fixed_harmony_date=a.DATE, controls=control_rows(), quota_per_class=300,
        service_lock=str(root/'organization_service_18083.lock'), human_gold=False,
        review_policy='All eligible original candidates without an explicit assistant disposition receive one blind 120B reference audit; '
                      'different or uncertain verdicts require explicit assistant adjudication before release.',
        classification_failure_policy='Retain without resampling; report coverage separately.')
    immutable(out/'plan.json', dict(configuration=cfg, configuration_sha256=vr.digest(cfg)))
    base.old.emit(out, 'prepared_review_pending', candidates=len(rows),
                  preexcluded=sum(bool(r.get('eligibility_failure')) for r in rows),
                  isolation_candidates=len(candidate_overlaps), model_calls=0)


def execution_plan(out, *, register=False, prepared=None):
    plan = vr.read(out/'plan.json') if prepared is None else prepared
    c = plan['configuration']
    require(vr.digest(c) == plan['configuration_sha256'] and c['protocol'] == p.VERSION, 'Revision plan drift')
    require(c['contract'] == p.contract(), 'Revision protocol drift')
    live = source_hashes()
    if (out/'replay_repair.json').is_file():
        from scripts.final_answer_replay_binding import bound_generation_sources
        live = bound_generation_sources(out, plan, live)
    receipt = out/'execution_repair.json'
    if c['source_hashes'] == live:
        require(not register and not receipt.exists(), 'Unexpected repair receipt for unchanged code')
        return plan
    expected = {**live, RUNNER: PREPARED_RUNNER}
    require(plan['configuration_sha256'] == PREPARED_CONFIGURATION and c['source_hashes'] == expected,
            'Code drift outside the authorized final-answer runner repair')
    effective = dict(c, source_hashes=live)
    signature = vr.digest(effective)
    binding = dict(preparation_sha256=vr.sha(out/'plan.json'), parent_configuration_sha256=PREPARED_CONFIGURATION,
                   configuration_sha256=signature, runner_before=PREPARED_RUNNER, runner_after=live[RUNNER],
                   repairs=['cumulative_wall_seconds', 'released_dispositions_sha256'])
    if register and not receipt.exists():
        for name in ('pilot','formal','review','review_controls','inputs.json','dispositions.json',
                     'labels_private.json','bench_release.json','pilot_acceptance.json','metrics.json'):
            require(not (out/name).exists(), 'Cannot rebind after execution or publication: ' + name)
        immutable(receipt, binding)
    require(receipt.is_file() and vr.read(receipt) == binding, 'Missing or changed execution repair receipt')
    return dict(configuration=effective, configuration_sha256=signature)


def verify(out, *, register=False):
    # Verify all immutable inputs before issuing an execution-only amendment.
    original = vr.read(out/'plan.json'); c = original['configuration']
    require(vr.digest(c) == original['configuration_sha256'], 'Revision plan drift')
    for key, name in [('candidates_sha256','candidates.json'),('pilot_sha256','pilot_inputs.json'),
                      ('isolation_sha256','isolation_candidates.json')]:
        require(c[key] == vr.sha(out/name), 'Frozen data drift: ' + name)
    require(c['material_hashes'] == {n:vr.sha(out/(n+'.json')) for n in c['material_hashes']}, 'Materials changed')
    vr.EvidenceCache(out/'evidence_cache.json').check(c['dependencies'])
    return execution_plan(out, register=register, prepared=original)


def read_overrides(out, candidates):
    path = out/'assistant_review.json'
    if not path.exists(): return {}
    review = vr.read(path)
    require(review['human_gold'] is False and review['reviewer'] == 'Codex assistant', 'Misidentified review authority')
    byid = {r['id']:r for r in candidates}; values = {}
    for d in review['decisions']:
        require(d['id'] in byid and d['id'] not in values, 'Unknown/duplicate adjudication')
        row = byid[d['id']]
        require(all(d[k] == row[k] for k in ('question_sha256','answer_sha256','reference_sha256')),
                'Assistant review uses different question/answer/reference')
        p.decision(row, None, d)
        values[d['id']] = d
    return values


def dispositions(out):
    plan = verify(out)
    candidates = vr.read(out/'candidates.json'); overrides = read_overrides(out, candidates); result = {}
    for row in candidates:
        path = out/'review'/row['id']/'result.json'
        reviewed = vr.read(path) if path.exists() else None
        if reviewed:
            reviewed = saved_request(path.parent, p.review_messages(row,row['reference']),
                                     plan['configuration']['limits']['max_tokens_per_response'], p.validate_review, plan)
        result[row['id']] = p.decision(row, reviewed, overrides.get(row['id']))
    return result


def seconds(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid elapsed time evidence')
    return value


@contextmanager
def request_deadline(duration):
    # HTTP read timeouts restart after each chunk. The single-threaded runner
    # also needs an absolute deadline when the service keeps sending bytes.
    def expired(*_):
        raise TimeoutError('Final-answer wall-time budget exhausted during request')
    require(duration > 0, 'Final-answer wall-time budget exhausted before request')
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'Another request timer is active')
    previous = signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, duration)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def request_budget(folder, binding, available, limits, *, setup=None):
    path = folder/'request_budget.json'
    if setup is not None:
        timeout = max(0, min(limits['request_timeout_seconds'], available - setup))
        immutable(path, dict(binding=binding, available_seconds=available, setup_seconds=setup, timeout_seconds=timeout))
    require(path.is_file(), 'Missing request wall-time evidence')
    budget = vr.read(path)
    timeout = max(0, min(limits['request_timeout_seconds'], available - seconds(budget['setup_seconds'])))
    require(budget['binding'] == binding and budget['available_seconds'] == available and
            budget['timeout_seconds'] == timeout, 'Request wall-time budget drift')
    require(timeout > 0, 'Cumulative wall-time budget exhausted before request')
    return budget


def check_charged_time(budget, response_elapsed, charged):
    require(seconds(response_elapsed) <= budget['timeout_seconds'] and
            budget['setup_seconds'] + response_elapsed <= seconds(charged) <= budget['available_seconds'],
            'Actual cumulative wall-time budget exceeded')


def saved_request(folder, messages, max_tokens, validator, plan, *, available=None):
    value = vr.read(folder/'result.json')
    binding = vr.digest([messages,max_tokens,plan['configuration_sha256']])
    require(value['binding'] == binding and value['input_sha256'] == vr.digest(messages) and
            value['configuration_sha256'] == plan['configuration_sha256'], 'Completed request binding drift')
    evidence = ['request.json','raw.json','response.json']
    if available is not None:
        evidence.append('request_budget.json')
    require(set(value['files']) == {str(folder/n) for n in evidence},
            'Incomplete request evidence set')
    for fn, sha in value['files'].items(): require(vr.sha(Path(fn)) == sha, 'Completed request evidence drift')
    record = vr.read(folder/'request.json'); req = record['request']
    require(record['request_sha256'] == base.r.sha256_value(req) and
            req['messages'] == messages and req['max_tokens'] == max_tokens and
            req['model'] == plan['configuration']['model'] and req['temperature'] == 0 and
            req['top_p'] == 1 and req['reasoning_effort'] == 'medium' and not req.get('tools'),
            'Replayed request mismatch')
    raw = vr.read(folder/'raw.json'); response = vr.read(folder/'response.json')
    require(seconds(response['elapsed_seconds']) == value['elapsed_seconds'], 'Response elapsed time drift')
    if available is not None:
        budget = request_budget(folder, binding, available, plan['configuration']['limits'])
        check_charged_time(budget, response['elapsed_seconds'], value['wall_seconds'])
    require(response['http_status'] == 200 and raw['model'] == req['model'] and
            response['model_ids'] == [req['model']] and response['raw_sse_sha256'] == vr.sha(folder/'raw.json') and
            value['usage'] == raw['usage'] and value['effective_input']['sha256'] == vr.digest(raw['prompt_token_ids']),
            'Cached identity/usage/effective-input drift')
    if value['status'] == 'complete':
        require(not response.get('error') and validator(base.r.parse_json(response['content'])) == value['value'],
                'Cached result differs from original model output')
    else:
        require(value['status'] == 'content_failure' and value['value'] is None and value['reason'],
                'Unclassified cached failure')
    return value


def request_once(out, folder, messages, max_tokens, validator, plan, tokenizer, client, *, available=None, started=None):
    entry = time.monotonic()
    if started is None:
        started = entry
    c = plan['configuration']; folder.mkdir(parents=True, exist_ok=True)
    binding = vr.digest([messages,max_tokens,plan['configuration_sha256']])
    target = folder/'result.json'
    if target.exists():
        return saved_request(folder,messages,max_tokens,validator,plan,available=available)
    request, raw_path, response = [folder/n for n in ('request.json','raw.json','response.json')]
    cached = request.exists()
    if cached:
        require(raw_path.exists() and response.exists(), 'Interrupted request lacks evidence; never blindly resample')
        if available is not None:
            budget = request_budget(folder, binding, available, c['limits'])
    else:
        require(not (out/'STOP').exists(), 'STOP requested')
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, reasoning_effort='medium')
        estimated = len(tokenizer.encode(rendered, add_special_tokens=False))
        require(estimated+max_tokens <= c['limits']['context_window'], 'Input does not fit; no truncation')
        timeout = c['limits']['request_timeout_seconds']
        if available is not None:
            if (folder/'request_budget.json').exists():
                request_budget(folder, binding, available, c['limits'])
                raise ValueError('Interrupted request budget without a request; inspect before retry')
            budget = request_budget(folder, binding, available, c['limits'], setup=time.monotonic()-started)
            timeout = min(budget['timeout_seconds'], available - (time.monotonic()-started))
        try:
            with request_deadline(timeout) if available is not None else nullcontext():
                base.runtime.transport.chat_response(client, endpoint=c['api_base']+'/chat/completions',
                    model=c['model'], messages=messages, max_tokens=max_tokens, timeout=timeout,
                    raw_path=raw_path, request_path=request, response_path=response)
        except TimeoutError:
            raise
        except Exception:
            if not all(f.exists() for f in (request,raw_path,response)):
                raise
    validation_started = time.monotonic()
    req, raw, rec = vr.read(request), vr.read(raw_path), vr.read(response)
    response_elapsed = seconds(rec['elapsed_seconds'])
    body = req['request']
    require(req['request_sha256'] == base.r.sha256_value(body) and body['messages'] == messages and
            body['max_tokens'] == max_tokens and body['model'] == c['model'] and
            body['temperature'] == 0 and body['top_p'] == 1 and body['reasoning_effort'] == 'medium' and
            not body.get('tools'), 'Actual request deviates from frozen protocol')
    require(rec.get('http_status') == 200 and raw.get('model') == c['model'] and
            rec.get('model_ids') == [c['model']] and len(raw.get('choices',[])) == 1,
            'HTTP/identity/envelope failure; inspect before continuing')
    require(rec.get('raw_sse_sha256') == vr.sha(raw_path), 'Raw response evidence drift')
    effective = validate_effective_input(raw, tokenizer, c['fixed_harmony_date'])
    actual = tokenizer.decode(raw['prompt_token_ids'],skip_special_tokens=False)
    require(all(m['content'] in actual for m in messages), 'Lost model-visible input or shared prefix')
    usage = raw.get('usage') or {}
    require(type(usage.get('prompt_tokens')) is int and type(usage.get('completion_tokens')) is int and
            0 < usage['completion_tokens'] <= max_tokens and
            usage['prompt_tokens'] == effective['token_count'] and
            usage.get('total_tokens') == usage['prompt_tokens'] + usage['completion_tokens'], 'Invalid actual usage')
    require(usage['prompt_tokens']+max_tokens <= c['limits']['context_window'], 'Actual context exceeds budget')
    state, value, reason = 'complete', None, None
    if rec.get('error'):
        known = ('Incomplete model output','No authoritative final/tool termination',
                 'Incomplete Harmony message','Missing final message')
        if raw['choices'][0].get('finish_reason') == 'length' or any(s in rec['error'] for s in known):
            state, reason = 'content_failure', rec['error']
        else:
            raise ValueError('Unclassified decoder/protocol error: '+rec['error'])
    else:
        try:
            value = validator(base.r.parse_json(rec['content']))
        except (ValueError, TypeError, KeyError) as exc:
            state, reason = 'content_failure', type(exc).__name__+': '+str(exc)
    result = dict(binding=binding, input_sha256=vr.digest(messages), configuration_sha256=plan['configuration_sha256'],
        status=state, value=value, reason=reason, usage=usage, effective_input=effective,
        files={str(fn):vr.sha(fn) for fn in (request,raw_path,response)}, elapsed_seconds=response_elapsed,
        resampled=False, human_gold=False)
    if available is not None:
        now = time.monotonic()
        # Retain the original setup and request durations when recovering the
        # raw response. Downtime and time spent on a different arm do not count.
        measured = budget['setup_seconds'] + response_elapsed + (now - validation_started)
        result['wall_seconds'] = measured if cached else max(measured, now - started)
        check_charged_time(budget, response_elapsed, result['wall_seconds'])
        result['files'][str(folder/'request_budget.json')] = vr.sha(folder/'request_budget.json')
    immutable(target,result)
    return result


def run_reviews(out, plan, tokenizer, client):
    rows = vr.read(out/'candidates.json'); overrides = read_overrides(out,rows)
    limit = plan['configuration']['limits']['max_tokens_per_response']
    for row in plan['configuration']['controls']:
        got = request_once(out,out/'review_controls'/row['id'],p.review_messages(row,row['reference']),limit,
                           p.validate_review,plan,tokenizer,client)
        require(got['status']=='complete' and got['value']['verdict']==row['expected'], 'Reference audit control failed')
    immutable(out/'review_controls_passed.json',dict(passed=True,configuration_sha256=plan['configuration_sha256'],
        files={str(f):vr.sha(f) for f in (out/'review_controls').glob('*/result.json')},
        scope='Synthetic protocol checks; not benchmark-score selection'))
    for i,row in enumerate(rows):
        if row.get('eligibility_failure') or row['id'] in overrides:
            continue
        request_once(out,out/'review'/row['id'],p.review_messages(row,row['reference']),limit,
                     p.validate_review,plan,tokenizer,client)
        base.old.emit(out,'reviewing',processed=i+1,planned=len(rows))
    values = dispositions(out)
    vr.atomic(out/'review_status.json',dict(counts=dict(Counter(d['state'] for d in values.values())),
        pending=[d for d in values.values() if d['state'].startswith('pending')],human_gold=False))
    base.old.emit(out,'review_complete_pending_release',counts=dict(Counter(d['state'] for d in values.values())))


def freeze_bench(out):
    plan=verify(out); rows=vr.read(out/'candidates.json'); choices=dispositions(out)
    isolation=vr.read(out/'isolation_candidates.json')
    require(not isolation['candidates'], 'New source-family candidates require review before publishing')
    selected=p.select_balanced(rows,choices,plan['configuration']['quota_per_class'])
    labels={r['id']:choices[r['id']] for r in selected}
    records=[out/'plan.json',out/'candidates.json',out/'isolation_candidates.json']
    if (out/'execution_repair.json').exists():records.append(out/'execution_repair.json')
    if (out/'assistant_review.json').exists():records.append(out/'assistant_review.json')
    for path in (out/'review').glob('*/result.json'):
        records.append(path); records.extend(Path(f) for f in vr.read(path)['files'])
    require((out/'review_controls_passed.json').exists(),'Reference audit controls not passed')
    controls = vr.read(out/'review_controls_passed.json')
    require(controls['passed'] and controls['configuration_sha256'] == plan['configuration_sha256'], 'Wrong control receipt')
    for path, sha in controls['files'].items():
        require(vr.sha(Path(path)) == sha, 'Control receipt drift')
    for row in plan['configuration']['controls']:
        folder=out/'review_controls'/row['id']
        got=saved_request(folder,p.review_messages(row,row['reference']),
            plan['configuration']['limits']['max_tokens_per_response'],p.validate_review,plan)
        require(got['status']=='complete' and got['value']['verdict']==row['expected'],'Control not accepted')
        records.append(folder/'result.json'); records.extend(Path(f) for f in got['files'])
    records.append(out/'review_controls_passed.json')
    immutable(out/'inputs.json',selected);immutable(out/'labels_private.json',labels)
    immutable(out/'dispositions.json',choices)
    immutable(out/'bench_release.json',dict(configuration_sha256=plan['configuration_sha256'],
        human_gold=False,counts=dict(Counter(x['verdict'] for x in labels.values())),
        pool_counts=dict(Counter(d['state'] for d in choices.values())),
        inputs_sha256=vr.sha(out/'inputs.json'),labels_sha256=vr.sha(out/'labels_private.json'),
        dispositions_sha256=vr.sha(out/'dispositions.json'),
        files={str(f):vr.sha(f) for f in records},all_physics_certified=False,
        selection='Original candidate order within corrected label classes; no verifier predictions used.'))
    base.old.emit(out,'bench_released',samples=len(selected),counts=dict(Counter(x['verdict'] for x in labels.values())))


def released(out, plan):
    release=vr.read(out/'bench_release.json')
    require(release['configuration_sha256']==plan['configuration_sha256'], 'Released benchmark configuration changed')
    for path,digest in release['files'].items():require(vr.sha(Path(path))==digest,'Released review changed')
    require(vr.sha(out/'inputs.json')==release['inputs_sha256'] and
            vr.sha(out/'labels_private.json')==release['labels_sha256'],'Released benchmark changed')
    path = out/'dispositions.json'
    require(path.is_file() and 'dispositions_sha256' in release, 'Missing released dispositions binding')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == release['dispositions_sha256'], 'Released dispositions changed')
    revised = json.loads(raw)
    rows = vr.read(out/'candidates.json')
    selected = p.select_balanced(rows, revised, plan['configuration']['quota_per_class'])
    require(selected == vr.read(out/'inputs.json') and {r['id']: revised[r['id']] for r in selected}
            == vr.read(out/'labels_private.json'), 'Released cohort differs from its dispositions')
    return revised


def stores(out):
    return dict(pure=None, case_rag=base.p.FlatMemory(vr.read(out/'cases.json')),
                flat_skills=base.p.FlatMemory(vr.read(out/'skills.json')),
                rule_tree=base.p.TreeMemory(vr.read(out/'catalog.json')))


def run_question(out, row, phase, plan, tokenizer, client, memories):
    started = time.monotonic()
    limits=plan['configuration']['limits']; key=vr.digest(row['id'])[:24]
    prefix=request_once(out,out/phase/'queries'/key,p.query_messages(row),base.p.QUERY_MAX_TOKENS,
                        base.p.validate_queries,plan,tokenizer,client,
                        available=limits['wall_seconds_per_sample'],started=started)
    # Same shared, label-free prefix is replayed and charged to all four arms.
    count=lambda m:len(tokenizer.apply_chat_template(m,tokenize=True,add_generation_prompt=True,reasoning_effort='medium'))
    for arm in ARMS:
        folder=out/phase/arm/key; path=folder/'result.json'
        if path.exists():
            saved=vr.read(path)
            verify_arm(row,arm,path,saved,prefix,plan)
            continue
        started = time.monotonic()
        charged = prefix['wall_seconds']
        files=dict(prefix['files']);files[str(out/phase/'queries'/key/'result.json')]=vr.sha(out/phase/'queries'/key/'result.json')
        usage=dict(prefix['usage']); retrieval=dict(provided_complete_ids=[],reference_tokens=0)
        value=None;status='content_failure';reason='Shared PREPARE failed';responses=1
        if prefix['status']=='complete':
            available = limits['wall_seconds_per_sample'] - charged
            require(available > 0, 'Cumulative wall-time budget exhausted before CHECK')
            packet=p.reference_packet(row,arm,prefix['value'],memories[arm],count)
            immutable(folder/'reference_packet.json',packet)
            files[str(folder/'reference_packet.json')]=vr.sha(folder/'reference_packet.json')
            retrieval=packet['retrieval']
            remaining=limits['output_tokens_per_sample']-usage['completion_tokens']
            require(remaining>0 and count(packet['messages'])+usage['prompt_tokens']<=limits['input_tokens_per_sample'],
                    'Cumulative input/output budget exceeded')
            check=request_once(out,folder/'check',packet['messages'],min(remaining,limits['max_tokens_per_response']),
                p.original.validate_verdict,plan,tokenizer,client,available=available,started=started)
            charged += check['wall_seconds']
            files.update(check['files']);files[str(folder/'check/result.json')]=vr.sha(folder/'check/result.json')
            status,value,reason=check['status'],check['value'],check['reason'];responses+=1
            usage={k:usage[k]+check['usage'][k] for k in ('prompt_tokens','completion_tokens','total_tokens')}
        require(seconds(charged)<=limits['wall_seconds_per_sample'], 'Actual cumulative wall-time budget exceeded')
        require(responses<=limits['max_requests_per_sample'] and usage['completion_tokens']<=limits['output_tokens_per_sample']
                and usage['prompt_tokens']<=limits['input_tokens_per_sample'],'Actual cumulative budget exceeded')
        immutable(path,dict(id=row['id'],arm=arm,status=status,value=value,reason=reason,usage=usage,
            actual_response_count=responses,wall_seconds=charged,retrieval=retrieval,files=files,human_gold=False,
            input_sha256=vr.digest(p.public(row)),configuration_sha256=plan['configuration_sha256']))


def run_checks(out, plan, tokenizer, client, pilot):
    memories=stores(out)
    rows=vr.read(out/('pilot_inputs.json' if pilot else 'inputs.json'))
    if not pilot:
        released(out,plan)
        accepted=vr.read(out/'pilot_acceptance.json')
        require(accepted['passed'] and accepted['configuration_sha256']==plan['configuration_sha256'],'Pilot not accepted')
        for path,digest in accepted['files'].items():require(vr.sha(Path(path))==digest,'Pilot evidence drift')
    phase='pilot' if pilot else 'formal'
    for i,row in enumerate(rows):
        require(not (out/'STOP').exists(),'STOP requested')
        run_question(out,p.public(row),phase,plan,tokenizer,client,memories)
        base.old.emit(out,phase,completed_questions=i+1,planned_questions=len(rows))
    if pilot:
        files={}; delivery={arm:0 for arm in ARMS}
        expected={x['id']:x['expected'] for x in old_labels.CONTROLS[:2]}
        for row in rows:
            for arm in ARMS:
                path=out/'pilot'/arm/vr.digest(row['id'])[:24]/'result.json';value=vr.read(path)
                require(value['status']=='complete','Pilot output-schema failure')
                provided=value['retrieval']['provided_complete_ids']
                require(len(provided)<=base.p.MAX_ITEMS and
                        value['retrieval']['reference_tokens']<=base.p.REFERENCE_TOKEN_CAP,
                        'Pilot material budget exceeded')
                if provided:delivery[arm]+=1
                if row['id'] in expected:
                    require(value['value']['verdict']==expected[row['id']],'Synthetic verdict control failed')
                files[str(path)]=vr.sha(path);files.update(value['files'])
        require(delivery['pure']==0 and all(delivery[arm]>0 for arm in ARMS if arm!='pure'),
                'Knowledge delivery not exercised by the fixed pilot; inspect without changing the test set')
        immutable(out/'pilot_acceptance.json',dict(passed=True,configuration_sha256=plan['configuration_sha256'],
            files=files,questions_with_material=delivery,
            scope='Fixed six development inputs; model/input/budget/schema/material delivery, no score optimization'))
    else:
        score(out)


def verify_arm(row, arm, path, value, prefix, plan):
    limits = plan['configuration']['limits']
    require(value['id'] == row['id'] and value['arm'] == arm and
            value['configuration_sha256'] == plan['configuration_sha256'] and
            value['input_sha256'] == vr.digest(p.public(row)), 'Wrong result/cohort binding')
    query_result = path.parents[2]/'queries'/path.parent.name/'result.json'
    files = dict(prefix['files'])
    files[str(query_result)] = vr.sha(query_result)
    charged = prefix['wall_seconds']
    usage = prefix['usage']
    responses = 1
    if prefix['status'] == 'complete':
        packet_path = path.parent/'reference_packet.json'
        packet = bind(files, packet_path)
        # FlatMemory.view sends id, title, text. Atomic JSON sorts the stored object keys.
        # Restore the send order only when replaying the archived packet.
        fields = ('id', 'title', 'text')
        references = packet['references']
        require(isinstance(references, list) and all(isinstance(ref, dict) and set(ref) == set(fields)
                for ref in references), 'Invalid stored full-material fields')
        sent = [{k: ref[k] for k in fields} for ref in references]
        expected = p.continuation_messages(row, prefix['value'], sent, arm=arm)
        require(packet['messages'] == expected and packet['retrieval'] == value['retrieval']
                and (arm != 'pure' or not packet['references']), 'Wrong shared prefix or supplied materials')
        limit = min(limits['max_tokens_per_response'], limits['output_tokens_per_sample'] - usage['completion_tokens'])
        check = saved_request(path.parent/'check', expected, limit, p.original.validate_verdict, plan,
                              available=limits['wall_seconds_per_sample'] - charged)
        require(all(check[k] == value[k] for k in ('status','value','reason')),
                'Published verdict differs from actual CHECK response')
        charged += check['wall_seconds']
        usage = {k: usage[k] + check['usage'][k] for k in ('prompt_tokens','completion_tokens','total_tokens')}
        responses += 1
        files.update(check['files'])
        check_result = path.parent/'check/result.json'
        files[str(check_result)] = vr.sha(check_result)
    else:
        require(value['status'] == 'content_failure' and value['value'] is None and
                value['reason'] == 'Shared PREPARE failed', 'Failed preparation was silently reclassified')
    require(value['files'] == files, 'Result evidence changed')
    require(seconds(value['wall_seconds']) == charged <= limits['wall_seconds_per_sample'],
            'Incorrect cumulative wall-time charge')
    require(value['actual_response_count'] == responses <= limits['max_requests_per_sample'] and
            value['usage'] == usage and usage['prompt_tokens'] <= limits['input_tokens_per_sample'] and
            usage['completion_tokens'] <= limits['output_tokens_per_sample'], 'Incorrect charged usage')


def score(out):
    plan=verify(out);released(out,plan)
    rows=vr.read(out/'inputs.json');labels={sid:d['verdict'] for sid,d in vr.read(out/'labels_private.json').items()}
    groups={};files={}
    for arm in ARMS:
        predictions={}
        for row in rows:
            path=out/'formal'/arm/vr.digest(row['id'])[:24]/'result.json'
            value=bind(files,path)
            prefix=saved_request(out/'formal/queries'/vr.digest(row['id'])[:24],p.query_messages(row),
                                 base.p.QUERY_MAX_TOKENS,base.p.validate_queries,plan,
                                 available=plan['configuration']['limits']['wall_seconds_per_sample'])
            verify_arm(row,arm,path,value,prefix,plan)
            files.update(value['files']);predictions[row['id']]=value
        groups[arm]=measure(labels,predictions)
    immutable(out/'metrics.json',dict(groups=groups,positive_class=p.POSITIVE_CLASS,
        confusion=p.contract()['confusion'],model_calls_for_scoring=0,human_gold=False,
        benchmark_sha256=vr.sha(out/'bench_release.json'),files=files))
    base.old.emit(out,'completed',groups={a:g['original_binary_view'] for a,g in groups.items()})


def rescore_archive(out):
    plan=verify(out);revised=released(out,plan);archive=Path(plan['configuration']['archive'])
    old=vr.read(archive/'final_labels_private.json')
    ids=[sid for sid in old if revised[sid]['state']=='eligible']
    before={sid:old[sid]['verdict'] for sid in ids};after={sid:revised[sid]['verdict'] for sid in ids}
    frozen=vr.read(archive/'final_metrics.json')['source_results'];groups={};files={}
    for arm in ARMS:
        predictions={}
        for sid in ids:
            rel=f'checks/final/formal/{arm}/{sid}/result.json';path=archive/rel
            require(vr.sha(path)==frozen[rel],'Archived result hash mismatch')
            value=bind(files,path);require(value['id']==sid and value['arm']==arm,'Archived identity mismatch')
            predictions[sid]=value
        groups[arm]=dict(old_labels_same_subset=measure(before,predictions),
                        revised_labels_same_subset=measure(after,predictions))
    immutable(out/'archived_rescore.json',dict(n=len(ids),excluded_old_ids=sorted(set(old)-set(ids)),
        changed_labels=[dict(id=sid,before=before[sid],after=after[sid]) for sid in ids if before[sid]!=after[sid]],
        groups=groups,files=files,new_model_calls=0,human_gold=False,
        scope='Same surviving old questions/outputs, compare only old versus revised labels; not a new-protocol run.',
        benchmark_sha256=vr.sha(out/'bench_release.json')))


def confirm(out, signature):
    ready=vr.read(out/'disconnect_ready.json')
    require(base.runtime.process_identity(ready['pid'])==ready['process_start'] and
            time.time()-ready['unix']>=5 and ready['configuration_sha256']==signature,
            'Fresh reconnect required')
    vr.atomic(out/'disconnect_acceptance.json',dict(ready,passed=True,confirmed_unix=time.time()))


def prepare_signature():
    return vr.digest(dict(stage='prepare_final_revision',sources=source_hashes(),contract=p.contract()))


def repair_signature(out):
    return vr.digest(['register-repair', vr.sha(out/'plan.json'), source_hashes()])


def online(out, mode):
    require(Path('/proc').exists(),'Server only')
    plan=verify(out);c=plan['configuration'];code=1
    with base.r.run_lock(out/'active.lock'),base.r.run_lock(Path(c['service_lock'])):
        (out/'runner.pid').write_text(str(os.getpid()))
        try:
            base.runtime.wait_detached(out,plan['configuration_sha256'])
            require(not (out/'STOP').exists(),'STOP requested')
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True,trust_remote_code=False)
            with httpx.Client(trust_env=False,timeout=10) as client:
                response=client.get(c['api_base']+'/models');response.raise_for_status()
                require(any(m['id']==c['model'] for m in response.json()['data']),'Model service mismatch')
                with a.configured(out,tok):
                    if mode=='review':run_reviews(out,plan,tok,client)
                    else:run_checks(out,plan,tok,client,pilot=mode=='pilot')
            code=0
        except BaseException as exc:
            base.old.emit(out,'stopped',error=type(exc).__name__+': '+str(exc));raise
        finally:
            (out/'runner.exit_code').write_text(str(code));(out/'runner.pid').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','verify','review','freeze','rescore-archive','pilot','run','score','confirm-detached','status','register-repair','confirm-repair'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--root',type=Path)
    parser.add_argument('--await-release',action='store_true',help='Require fresh reconnect acceptance for detached offline work')
    args=parser.parse_args();out=args.output.resolve()
    require(Path('/proc').exists(),'Formal preparation/execution/scoring runs on server')
    if args.mode=='confirm-detached':
        confirm(out,verify(out)['configuration_sha256'] if (out/'plan.json').exists() else prepare_signature())
    elif args.mode=='confirm-repair':confirm(out/'repair_validation',repair_signature(out))
    elif args.mode=='verify':verify(out);print('verified')
    elif args.mode=='status':verify(out);print(dict(Counter(x['state'] for x in dispositions(out).values())))
    elif args.mode in {'review','pilot','run'}:online(out,args.mode)
    else:
        require(args.await_release,'Start offline work detached with --await-release, then confirm from a new SSH session')
        with base.r.run_lock(out/'active.lock'):
            control = out/'repair_validation' if args.mode=='register-repair' else out
            control.mkdir(parents=True, exist_ok=True)
            code=1;(control/'runner.pid').write_text(str(os.getpid()))
            try:
                if args.mode=='register-repair':signature=repair_signature(out)
                elif args.mode=='prepare':signature=prepare_signature()
                else:signature=verify(out)['configuration_sha256']
                base.runtime.wait_detached(control,signature)
                require(not (out/'STOP').exists(),'STOP requested')
                if args.mode=='prepare':
                    require(args.root is not None,'--root is required for prepare')
                    prepare(out,args.root.resolve())
                elif args.mode=='register-repair':
                    effective=verify(out,register=True)
                    base.old.emit(control,'repair_bound',configuration_sha256=effective['configuration_sha256'],model_calls=0)
                elif args.mode=='freeze':freeze_bench(out)
                elif args.mode=='rescore-archive':rescore_archive(out)
                else:score(out)
                code=0
            except BaseException as exc:
                base.old.emit(control,'stopped',error=type(exc).__name__+': '+str(exc));raise
            finally:
                (control/'runner.exit_code').write_text(str(code));(control/'runner.pid').unlink(missing_ok=True)


if __name__=='__main__':main()
