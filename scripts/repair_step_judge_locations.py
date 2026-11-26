"""Locate evidence for the 31 approved failed step judgments and replay the combined score."""
import argparse
from functools import partial
import json
from pathlib import Path
import time

import httpx

from scripts import run_step_match_judge as judge

vr, require, write = judge.vr, judge.require, judge.write
MATCH = judge.match
APPROVED_JUDGE = '20d154e0b4e94c500cc08776f16cce25e75510c386c89bf02e6c260053084238'
INITIAL_STATUS_SHA = 'cb6796bcbb06eac76416742a805277b4316c96ad205376443d267f106b78fdf0'
QUOTE_ERRORS = {'ValueError: Judge evidence is not a verbatim passage in the student solution',
                'ValueError: A claimed root match needs a source passage'}
PROMPT = """Locate the statement in this student's numbered solution that the published diagnosis actually criticizes. The previous judge classified the claim, but its quoted evidence could not be verified. Select the shortest continuous range of original lines that shows the claim in context. Read later corrections. A shared topic, a correct reference statement or an unrelated occurrence is insufficient. For an earlier same_root judgment, the passage must support the stated cause and correction direction for each proposed root. If the evidence is absent or you cannot identify a sufficient passage, return null for both line numbers.

Return only one JSON object with inclusive, one-based line numbers: {"start_line":3,"end_line":4} or {"start_line":null,"end_line":null}. Do not copy the passage or issue a new root classification."""


def location_messages(original_messages, draft):
    original = json.loads(original_messages[1]['content'])
    solution = original.pop('student_solution')
    original['numbered_student_solution'] = [[i, text] for i, text in enumerate(solution.splitlines(), 1)]
    original['earlier_judgment'] = {k: draft[k] for k in ('disposition', 'root_ids')}
    return [dict(role='system', content=PROMPT), dict(role='user', content=json.dumps(
        original, ensure_ascii=False, sort_keys=True).replace('<|', '\\u003c|'))]


def selected_span(answer, value):
    require(isinstance(value, dict) and set(value) == {'start_line', 'end_line'}, 'Unexpected location fields')
    first, last = value['start_line'], value['end_line']
    if first is None and last is None:
        return None
    lines = answer.splitlines(keepends=True)
    require(type(first) is int and type(last) is int and 1 <= first <= last <= len(lines), 'Invalid line range')
    start = sum(map(len, lines[:first - 1]))
    end = start + sum(map(len, lines[first - 1:last]))
    require(answer[start:end].strip(), 'Location contains only whitespace')
    return MATCH.location.source_span(answer, start, end)


def validate_location(answer, value):
    selected_span(answer, value)
    return value


def prepare(published):
    initial = published / 'step_judge'
    out = initial / 'location_repair'
    initial_status = vr.read(initial / 'status.json')
    require(vr.sha(initial / 'status.json') == INITIAL_STATUS_SHA and initial_status['state'] == 'incomplete'
            and initial_status['configuration_sha256'] == APPROVED_JUDGE and not initial_status['missing'],
            'This amendment is bound to the original 31 failed judgments')
    checker = vr.read(published / 'plan.json')['configuration']
    data = judge.production(Path(checker['step_benchmark']), published)
    parent, bindings, jobs = judge.prepare(initial, *data)
    require(parent['configuration_sha256'] == APPROVED_JUDGE, 'Different first-pass judge')
    config = parent['configuration']
    allowed = initial_status['content_failures']
    drafts, kept, evidence = {}, {}, {}
    for key, (row, messages) in jobs.items():
        folder = initial / 'judgments' / key
        validator = partial(MATCH.validate, answer=row['prediction'], roots={r['root_id'] for r in row['roots']})
        saved = judge.transport.saved_request(folder, messages, judge.MAX_OUTPUT, validator, parent,
                    available=config['limits']['request_timeout_seconds'])
        evidence[key] = vr.sha(folder / 'result.json')
        if key not in allowed:
            require(saved['status'] == 'complete', 'Unexpected failed original judgment')
            kept[key] = saved['value']
            continue
        require(saved['status'] == 'content_failure' and saved['reason'] == allowed[key] and allowed[key] in QUOTE_ERRORS,
                'Only original citation failures may be repaired')
        draft = judge.base.r.parse_json(vr.read(folder / 'response.json')['content'])
        try:
            validator(draft)
        except ValueError as exc:
            require('ValueError: ' + str(exc) == allowed[key], 'Different original validation failure')
        else:
            raise ValueError('Original judgment already valid')
        drafts[key] = draft
    require(set(drafts) == set(allowed) and len(drafts) == 31 and len(kept) == 32, 'Wrong repair cohort')
    cfg = {k: config[k] for k in ('model', 'api_base', 'fixed_harmony_date', 'max_tokens', 'limits',
                                  'temperature', 'top_p', 'reasoning_effort')}
    cfg.update(parent_judge_sha256=APPROVED_JUDGE, first_pass_status_sha256=INITIAL_STATUS_SHA,
               original_results=evidence, authorized_keys=sorted(drafts), prompt=PROMPT,
               code_sha256=vr.sha(Path(__file__)), prior_publication_sha256=config['publication_sha256'])
    plan = dict(configuration=cfg, configuration_sha256=vr.digest(cfg))
    write(out / 'plan.json', plan)
    return out, plan, data, bindings, jobs, kept, drafts


def merge_location(row, draft, lines):
    span = selected_span(row['prediction'], lines)
    # A proposed match still needs source evidence under the original judge contract.
    if span is None and draft['disposition'] == 'same_root':
        effective = dict(disposition='unclear', root_ids=[], source_quote='',
                         rationale='No source passage supporting the proposed match was located.')
    else:
        effective = dict(draft, source_quote=span['quote'] if span else '')
    return MATCH.validate(effective, row['prediction'], {r['root_id'] for r in row['roots']}), span


def score_row(row, result, decisions, spans):
    score = MATCH.score_row(row, result, decisions)
    # A repeated passage is pinned to the exact occurrence selected by the repair judge.
    if not any(span and row['prediction'].count(span['quote']) > 1 for span in spans):
        return score
    reviewed = []
    for index, (diagnosis, decision, span) in enumerate(zip(result['diagnostics'], decisions, spans)):
        item = MATCH.reviewed(row['id'], index, row['prediction'], diagnosis, decision)
        if span and row['prediction'].count(span['quote']) > 1:
            original = MATCH.location.locate_all(row['prediction'], diagnosis['evidence']['quote'])
            aligned = [s for s in original if MATCH.location.overlap(s, span) >= 0]
            item['approved_spans'] = aligned or [span]
            for link in item['root_links']:
                link['acceptable_source_spans'] = item['approved_spans']
        reviewed.append(item)
    states = {g['error_id']: {'state': 'active_error'} for g in row['gt']}
    matches = MATCH.location.assign(row['gt'], reviewed, states, True)
    return dict(score, matches=matches, hits=len(matches),
                unresolved_overlap=MATCH.location.unresolved_overlaps(reviewed, matches, row['unresolved']))


def evaluate(published, prepared, *, client=None, tokenizer=None):
    out, plan, data, bindings, jobs, kept, drafts = prepared
    rows, results, coverage, source = data
    completed, spans, selections, invalid, missing = dict(kept), {}, {}, {}, []
    config = plan['configuration']
    for key in config['authorized_keys']:
        row, original = jobs[key]
        messages = location_messages(original, drafts[key])
        folder = out / 'calls' / key
        validator = partial(validate_location, row['prediction'])
        args = (folder, messages, config['max_tokens'], validator, plan)
        timeout = config['limits']['request_timeout_seconds']
        if (folder / 'result.json').exists():
            saved = judge.transport.saved_request(*args, available=timeout)
        elif client is None:
            missing.append(key)
            continue
        else:
            require(not (out / 'STOP').exists() and not (published / 'STOP').exists(), 'STOP requested')
            saved = judge.transport.request_once(out, *args, tokenizer, client, available=timeout)
        if saved['status'] == 'complete':
            completed[key], spans[key] = merge_location(row, drafts[key], saved['value'])
            selections[key] = dict(lines=saved['value'], source_span=spans[key], original_disposition=drafts[key]['disposition'],
                                   effective_disposition=completed[key]['disposition'], effective_root_ids=completed[key]['root_ids'])
        else:
            invalid[key] = saved['reason']
        if client is not None:
            vr.atomic(out / 'progress.json', dict(terminal=len(selections)+len(invalid), total=len(drafts), invalid=len(invalid)))
    status = dict(configuration_sha256=plan['configuration_sha256'], original_valid_reused=len(kept), requested=len(drafts),
                  valid=len(selections), located=sum(s is not None for s in spans.values()), invalid=invalid, missing=missing)
    if invalid or missing:
        vr.atomic(out / 'status.json', dict(status, state='incomplete'))
        raise ValueError(f'Location repair incomplete: {len(missing)} missing, {len(invalid)} invalid')
    by_id = {r['id']: r for r in rows}
    details = {a: [] for a in source['arms']}
    for b in bindings:
        sid, arm, keys = b['id'], b['arm'], b['keys']
        details[arm].append(score_row(by_id[sid], results[sid, arm], [completed[k] for k in keys], [spans.get(k) for k in keys]))
    report = dict(protocol=MATCH.VERSION, configuration_sha256=plan['configuration_sha256'], human_gold=False,
                  coverage=coverage, judge=dict(valid=len(completed), published_diagnoses=sum(len(b['keys']) for b in bindings),
                  unique_diagnoses=len(jobs), first_pass_valid=len(kept), location_repair=status),
                  recorded_tasks={a: len(ds) for a, ds in details.items()},
                  scores={a: MATCH.summarize(ds) for a, ds in details.items()}, details_sha256=vr.digest(details))
    write(out / 'locations.json', selections)
    write(out / 'details.json', details)
    write(published / 'metrics.json', report)
    vr.atomic(out / 'status.json', dict(status, state='complete', metrics_sha256=vr.sha(published / 'metrics.json')))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'confirm', 'score'))
    parser.add_argument('--published', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(4, 5, 6, 7), default=4)
    args = parser.parse_args()
    published = args.published.resolve()
    out = published / 'step_judge/location_repair'
    if args.command == 'confirm':
        judge.node.node()
        return judge.node.confirm(out, vr.read(out / 'plan.json')['configuration_sha256'])
    with judge.base.r.run_lock(published / 'active.lock'), judge.base.r.run_lock(published / 'step_judge/runner.lock'):
        prepared = prepare(published)
        if args.command == 'prepare':
            return
        if args.command == 'score':
            return evaluate(published, prepared)
        judge.node.node()
        signature = prepared[1]['configuration_sha256']
        try:
            if not all((out / 'calls' / k / 'result.json').is_file() for k in prepared[-1]):
                judge.base.runtime.wait_detached(out, signature)
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(judge.node.host.HOT / 'models/gpt-oss-120b', local_files_only=True)
                with judge.node.service(out, args.gpu, out / 'STOP', signature) as endpoint, httpx.Client(trust_env=False) as client:
                    require(endpoint == prepared[1]['configuration']['api_base'], 'Wrong judge endpoint')
                    evaluate(published, prepared, client=client, tokenizer=tok)
            report = evaluate(published, prepared)
            vr.atomic(out / 'exit.json', dict(code=0, offline_replay=True, valid=report['judge']['valid'], unix=time.time()))
        except Exception as exc:
            vr.atomic(out / 'exit.json', dict(code=1, error=str(exc), unix=time.time()))
            raise


if __name__ == '__main__':
    main()
