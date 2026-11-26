"""Retest the two first-review flat libraries using the published final-answer BM25 protocol."""
from collections import Counter
from pathlib import Path
import time

import argparse

import httpx

from scripts import frozen_bm25_sources as sources
from scripts import serve_frozen_bm25 as node
from scripts import run_final_answer_revision as final
from scripts import run_node14_final_pilot as host

vr, require = final.vr, final.require
ARMS = final.ARMS
FLAT_ARMS = ('case_rag', 'flat_skills')
RESULTS = host.HOT / 'results/qwen30b_step_verifier'
OUTPUT = sources.OUTPUT / 'final'
FORMAL = RESULTS / 'aligned_old99/final_revision_balanced'
BENCH = 'ff89b8ef9f0fc9d894f06051057e634849db572e27e34f970dcc8273ec28af91'
METRICS = 'a06537e7366d1f7eaac24defab02815c5682719a10f14c5c28b299cb71dd5683'
PILOT_RECEIPT = 'ad06cf0edb6d15bbef7e6eedfef486bd2bd3630c7f547f89e52eb59065fba993'


def definition():
    parent = host.OUT
    old = final.execution_plan(parent)
    cfg = old['configuration']
    require(old['configuration_sha256'] == host.CONFIG and cfg['fixed_harmony_date'] == '2026-11-03'
            and cfg['contract'] == final.p.contract(), 'Different parent checking protocol')
    receipt = parent / 'pilot_acceptance.json'
    require(vr.sha(receipt) == PILOT_RECEIPT and vr.read(receipt)['passed'], 'Wrong development release')
    samples = parent / 'pilot_inputs.json'
    require(vr.sha(samples) == host.PILOT == cfg['pilot_sha256'], 'Different six development inputs')
    rows = vr.read(samples)
    controls = {x['id']: x['expected'] for x in cfg['controls'][:2]}
    require([r['id'] for r in rows] == [*final.a.PILOT_IDS, *controls], 'Development question order changed')
    accepted = vr.read(sources.OUTPUT / 'sources.json')['groups']['n2000']
    library = Path(accepted['material_path'])
    require(all(vr.sha(library / n) == h for n, h in accepted['files'].items())
            and vr.sha(parent / 'catalog.json') == sources.TREE, 'Published material changed')
    effective = dict(model=cfg['model'], api_base=cfg['api_base'], limits=cfg['limits'],
                     fixed_harmony_date=cfg['fixed_harmony_date'], contract=cfg['contract'],
                     parent_configuration_sha256=host.CONFIG, sources_sha256=vr.sha(sources.OUTPUT / 'sources.json'),
                     measured_tree_sha256=sources.TREE, changed_arms=list(FLAT_ARMS),
                     material_hashes=accepted['files'],
                     arms=list(ARMS), phase='pilot', input_sha256=host.PILOT, parent_acceptance_sha256=PILOT_RECEIPT)
    plan = dict(configuration=effective, configuration_sha256=vr.digest(effective))
    return parent, old, rows, controls, library, plan


def eligible_for_reuse(packet, old_body, model, max_tokens):
    body = final.base.r.request_body(model, packet['messages'], max_tokens, 'vllm', None)
    body.update(stream=False, return_token_ids=True, reasoning_effort='medium')
    body.pop('stream_options', None)
    return body == old_body


def load_memories(library):
    return dict(pure=None, case_rag=final.base.p.FlatMemory(vr.read(library / 'cases.json')),
                flat_skills=final.base.p.FlatMemory(vr.read(library / 'skills.json')))


def prefix_record(parent, old, phase, row):
    from functools import partial
    english = final.en.active(old['configuration'])
    key = vr.digest(row['id'])[:24]
    folder = parent / phase / 'queries' / key
    prefix = final.saved_request(folder, final.p.query_messages(row, english=english), final.base.p.QUERY_MAX_TOKENS,
                                 partial(final.base.p.validate_queries, english=english), old,
                                 available=old['configuration']['limits']['wall_seconds_per_sample'])
    return folder, prefix


def evaluate_row(out, parent, old, phase, row, plan, memories, tokenizer, client, source_files):
    key = vr.digest(row['id'])[:24]
    limits = plan['configuration']['limits']
    english = final.en.active(plan['configuration'])
    require(english == final.en.active(old['configuration']), 'Incompatible shared PREPARE language')
    reference = final.check_exam_answer(row, plan['configuration'])
    require(reference == final.check_exam_answer(row, old['configuration']), 'Incompatible exam reference for baseline reuse')
    prefix = None
    def count(messages):
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 reasoning_effort='medium'))
    results = {}
    for arm in ARMS:
        staged = final.base.p.uses_preparation(arm, english=english)
        if staged and prefix is None:
            folder, prefix = prefix_record(parent, old, phase, row)
        old_dir = parent / phase / arm / key
        old_result = old_dir / 'result.json'
        require(vr.sha(old_result) == source_files[str(old_result)], 'Unbound original arm outcome')
        previous = vr.read(old_result)
        final.verify_arm(row, arm, old_result, previous, prefix if staged else None, old)
        dest = out / phase / arm / key
        if arm not in FLAT_ARMS:
            retrieval = previous['retrieval']
            result = dict(id=row['id'], arm=arm, source='frozen_baseline', status=previous['status'],
                          value=previous['value'], provided=retrieval['provided_complete_ids'],
                          material_tokens=retrieval['reference_tokens'], original_result=str(old_result),
                          original_sha256=source_files[str(old_result)], configuration_sha256=plan['configuration_sha256'])
            final.immutable(dest / 'result.json', result)
            results[arm] = result
            continue
        started = time.monotonic()
        evidence = dict(prefix['files'])
        evidence[str(folder / 'result.json')] = vr.sha(folder / 'result.json')
        status, value, kind = 'content_failure', None, 'shared_prepare_failure'
        delivery = dict(provided_complete_ids=[], reference_tokens=0)
        reason, usage, wall = 'Shared PREPARE failed', dict(prefix['usage']), prefix['wall_seconds']
        if prefix['status'] == 'complete':
            packet = final.p.reference_packet(row, arm, prefix['value'], memories[arm], count, english=english,
                                              official_reference=reference)
            delivery = packet['retrieval']
            require(len(delivery['provided_complete_ids']) <= final.base.p.MAX_ITEMS and
                    delivery['reference_tokens'] <= final.base.p.REFERENCE_TOKEN_CAP, 'Material budget exceeded')
            final.immutable(dest / 'reference_packet.json', packet)
            evidence[str(dest / 'reference_packet.json')] = vr.sha(dest / 'reference_packet.json')
            maximum = min(limits['output_tokens_per_sample'] - usage['completion_tokens'], limits['max_tokens_per_response'])
            require(maximum > 0 and count(packet['messages']) + usage['prompt_tokens'] <= limits['input_tokens_per_sample'],
                    'Cumulative input/output budget exhausted')
            available = limits['wall_seconds_per_sample'] - wall
            require(available > 0, 'Cumulative wall time exhausted')
            body = vr.read(old_dir / 'check/request.json')['request']
            reuse = eligible_for_reuse(packet, body, plan['configuration']['model'], maximum)
            if reuse:
                check_dir = old_dir / 'check'
                check = final.saved_request(check_dir, packet['messages'], maximum, final.p.original.validate_verdict,
                                            old, available=available)
                kind = 'identical_check'
            else:
                check_dir = dest / 'check'
                require(client is not None or (check_dir / 'result.json').is_file(), 'Missing new CHECK')
                check = final.request_once(out, check_dir, packet['messages'], maximum, final.p.original.validate_verdict,
                                           plan, tokenizer, client, available=available, started=started)
                kind = 'new_check'
            evidence[str(check_dir / 'result.json')] = vr.sha(check_dir / 'result.json')
            evidence.update(check['files'])
            status, value, reason = check['status'], check['value'], check['reason']
            usage = {k: usage[k] + check['usage'][k] for k in ('prompt_tokens','completion_tokens','total_tokens')}
            wall += check['wall_seconds']
        require(wall <= limits['wall_seconds_per_sample'] and usage['prompt_tokens'] <= limits['input_tokens_per_sample']
                and usage['completion_tokens'] <= limits['output_tokens_per_sample']
                and 1 + (prefix['status'] == 'complete') <= limits['max_requests_per_sample'],
                'Actual cumulative check budget exceeded')
        result = dict(id=row['id'], arm=arm, status=status, value=value, reason=reason, source=kind, usage=usage,
                      wall_seconds=wall, provided=delivery['provided_complete_ids'], material_tokens=delivery['reference_tokens'],
                      configuration_sha256=plan['configuration_sha256'], input_sha256=vr.digest(final.p.public(row)),
                      files=evidence, human_gold=False)
        final.immutable(dest / 'result.json', result)
        results[arm] = result
    return results


def pilot(out, inputs, tokenizer, client=None):
    parent, old, rows, controls, library, plan = inputs
    final.immutable(out / 'pilot_plan.json', plan)
    files = vr.read(parent / 'pilot_acceptance.json')['files']
    memories = load_memories(library)
    records, counts, checks = {}, Counter(), {}
    for number, row in enumerate(rows, 1):
        require(not (out / 'STOP').exists(), 'STOP requested')
        group = evaluate_row(out, parent, old, 'pilot', row, plan, memories, tokenizer, client, files)
        records[row['id']] = {}
        for arm, value in group.items():
            counts[value['source']] += 1
            verdict = value['value']['verdict'] if value['status'] == 'complete' else value['status']
            records[row['id']][arm] = dict(verdict=verdict, material_count=len(value['provided']),
                                           material_tokens=value['material_tokens'], source=value['source'])
            if row['id'] in controls:
                checks[arm + '/' + row['id']] = verdict == controls[row['id']]
        if client is not None:
            vr.atomic(out / 'pilot_progress.json', dict(completed=number, total=len(rows), counts=dict(counts)))
    good = all(checks.values()) and len(checks) == 8 and all(
        v['verdict'] in ('correct','incorrect','uncertain') for r in records.values() for v in r.values())
    delivery = {a: sum(r[a]['material_count'] > 0 for r in records.values()) for a in ARMS}
    require(delivery['pure'] == 0, 'Materials reached the baseline')
    report = dict(phase='pilot', questions=len(rows), configuration_sha256=plan['configuration_sha256'],
                  passed=good and delivery['case_rag'] > 0 and delivery['flat_skills'] > 0,
                  controls=controls, all_controls_passed=all(checks.values()), rows=records,
                  material_coverage=delivery, new_check_calls=counts['new_check'],
                  identical_checks=counts['identical_check'], frozen_baselines=counts['frozen_baseline'], shared_prepare_failures=counts['shared_prepare_failure'],
                  shared_prepares_reused=len(rows), real_questions_with_independent_final_label=0, human_gold=False)
    final.immutable(out / 'pilot_report.json', report)
    return report


def formal_definition(pilot_inputs):
    parent = FORMAL
    old = final.execution_plan(parent)
    require(vr.sha(parent / 'bench_release.json') == BENCH and vr.sha(parent / 'metrics.json') == METRICS
            and old['configuration_sha256'] == host.CONFIG
            and vr.sha(parent / 'catalog.json') == sources.TREE, 'Different final-answer release')
    bench = vr.read(parent / 'bench_release.json')
    require(bench['counts'] == {'correct': 237, 'incorrect': 237}
            and vr.sha(parent / 'inputs.json') == bench['inputs_sha256']
            and vr.sha(parent / 'labels_private.json') == bench['labels_sha256']
            and vr.sha(parent / 'dispositions.json') == bench['dispositions_sha256'], 'Changed cohort or labels')
    all_rows = vr.read(parent / 'inputs.json')
    ids = {row['id'] for row in all_rows}
    require(len(ids) == len(all_rows) == 474 and sources.FINAL_HOLD <= ids, 'Changed coverage')
    rows = [row for row in all_rows if row['id'] not in sources.FINAL_HOLD]
    configuration = dict(pilot_inputs[-1]['configuration'], phase='formal', input_sha256=bench['inputs_sha256'],
                         labels_sha256=bench['labels_sha256'], parent_acceptance_sha256=METRICS,
                         benchmark_sha256=BENCH, scope_hold=sorted(sources.FINAL_HOLD))
    plan = dict(configuration=configuration, configuration_sha256=vr.digest(configuration))
    return rows, old, plan


def formal(rows, old, plan, library, tokenizer, client):
    final.immutable(OUTPUT / 'formal_plan.json', plan)
    final.immutable(OUTPUT / 'coverage.json', dict(registered=474, eligible=len(rows), held=sorted(sources.FINAL_HOLD),
                                                   original_order=True, replacements=0))
    archived = vr.read(FORMAL / 'metrics.json')['files']
    memories = load_memories(library)
    distribution = Counter()
    for i, row in enumerate(rows, 1):
        require(not (sources.OUTPUT / 'STOP').exists(), 'STOP requested')
        group = evaluate_row(OUTPUT, FORMAL, old, 'formal', row, plan, memories, tokenizer, client, archived)
        distribution.update(x['source'] for x in group.values())
        vr.atomic(OUTPUT / 'formal_progress.json', dict(questions=i, total=len(rows), sources=distribution, unix=time.time()))
    final.immutable(OUTPUT / 'formal_complete.json', dict(questions=len(rows), sources=distribution,
                    configuration_sha256=plan['configuration_sha256']))


def score(rows, plan):
    require(Path('/proc').is_dir(), 'Compute official metrics on node14')
    labels = {sid: rec['verdict'] for sid, rec in vr.read(FORMAL / 'labels_private.json').items()
              if sid not in sources.FINAL_HOLD}
    require(Counter(labels.values()) == {'correct': 233, 'incorrect': 235}, 'Changed common cohort')
    groups, files = {}, {}
    for arm in ARMS:
        predictions = {}
        for row in rows:
            path = OUTPUT / 'formal' / arm / vr.digest(row['id'])[:24] / 'result.json'
            value = final.bind(files, path)
            require(value['id'] == row['id'] and value['arm'] == arm
                    and value['configuration_sha256'] == plan['configuration_sha256'], 'Mixed result')
            if value['source'] == 'frozen_baseline':
                require(vr.sha(value['original_result']) == value['original_sha256'], 'Frozen baseline drift')
            else:
                require(all(vr.sha(p) == h for p, h in value['files'].items()), 'Actual CHECK evidence drift')
            predictions[row['id']] = value
        groups[arm] = final.measure(labels, predictions)
    final.immutable(OUTPUT / 'metrics.json', dict(registered=474, common_questions=len(rows), groups=groups,
                    held=sorted(sources.FINAL_HOLD), positive_class='incorrect', human_gold=False,
                    configuration_sha256=plan['configuration_sha256'], files=files, scoring_model_calls=0))


def run(attempt, gpu):
    node.node()
    with final.base.r.run_lock(OUTPUT / 'runner.lock'):
        inputs = definition()
        rows, old, plan = formal_definition(inputs)
        final.immutable(OUTPUT / 'formal_plan.json', plan)
        final.base.runtime.wait_detached(attempt, plan['configuration_sha256'])
        step_root = sources.OUTPUT / 'step'
        while not ((step_root / 'formal_complete.json').is_file() and any(
                vr.read(p).get('code') == 0 and (p.parent / 'release.json').is_file()
                for p in (step_root / 'attempts').glob('*/exit.json'))):
            require(not (sources.OUTPUT / 'STOP').exists(), 'STOP while waiting for steps')
            vr.atomic(attempt / 'queue.json', dict(state='waiting_for_step_checks', unix=time.time()))
            time.sleep(20)
        from scripts import run_frozen_step_bm25 as step
        finished = vr.read(step.ROOT / 'formal_complete.json')
        require(finished['passed'] and finished['tasks'] == 400
                and finished['configuration_sha256'] == step.verify()['configuration_sha256'], 'Incomplete preceding steps')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(host.HOT / 'models/gpt-oss-120b', local_files_only=True)
        with node.service(attempt, gpu, sources.OUTPUT / 'STOP', plan['configuration_sha256']) as endpoint:
            require(endpoint == plan['configuration']['api_base'], 'Unexpected endpoint')
            with httpx.Client(trust_env=False) as client:
                acceptance = pilot(OUTPUT, inputs, tokenizer, client)
                require(acceptance['passed'], 'Final-answer pilot did not pass')
                formal(rows, old, plan, inputs[-2], tokenizer, client)
        score(rows, plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'confirm'])
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=4)
    args = parser.parse_args()
    attempt = OUTPUT / 'attempts' / str(args.attempt)
    if args.action == 'confirm':
        signature = vr.read(OUTPUT / 'formal_plan.json')['configuration_sha256']
        node.confirm(attempt, signature)
    else:
        attempt.mkdir(parents=True, exist_ok=True)
        try:
            run(attempt, args.gpu)
            vr.atomic(attempt / 'exit.json', dict(code=0, unix=time.time()))
        except Exception as error:
            vr.atomic(attempt / 'exit.json', dict(code=1, error=str(error), unix=time.time()))
            raise
