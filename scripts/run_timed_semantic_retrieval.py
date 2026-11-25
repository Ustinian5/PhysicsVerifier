"""Run BM25 + independent 120B index retrieval, RRF, reranking and final-answer CHECK."""
import argparse
import json
import time
from pathlib import Path
from statistics import mean
from unittest.mock import patch

import httpx

from rule_framework import rank_output
from rule_framework import semantic_index_retrieval as semantic
from scripts import timed_retrieval_baseline as base
from scripts import timed_retrieval_cohort as cohort
from scripts import timed_retrieval_runtime as runtime

b, vr, require = cohort.original, cohort.vr, cohort.require
NAME = 'semantic_120b_bounded_rank'
PRIOR = 'semantic_120b'
MATERIAL_LIMIT = 8192
RESERVED_CHECK_OUTPUT = 4096
PROCESSOR = Path(rank_output.__file__).with_name('vllm_rank_output.py')


class SelectionFailure(Exception):
    """A metered content failure or exhausted selection budget."""


def design(root):
    out = root/cohort.RESULT
    registry = vr.read(out/'cohort.json')
    require(vr.sha(out/'cohort.json') == 'd6873f993b7050517e76cc8bef45d514dd5e9260e1fc318c09f1e59e9b2a3ed0'
            and registry['inputs_sha256'] == vr.sha(out/'inputs.json') and vr.digest(registry['ids']) == cohort.IDS,
            'Cohort mismatch')
    prior = out/PRIOR/'design.json'
    require(vr.sha(prior) == 'ac3d31dc6a7e4a23d8747559506f6d302889aac055fe09b71d697774dfba1190',
            'Unbound original ten-question design')
    config = dict(vr.read(prior), parent_design_sha256=vr.sha(prior), sample_size=50, acceptance_prefix=10,
        acceptance='Replay the first ten frozen inputs without new calls. Each arm completes its semantic route/scan, fusion/selection and reaches CHECK. No grade threshold.',
        repair='Only ranking requests opt into bounded Harmony analysis and a legal, unique ID list. Preserve the original model-visible messages, cards, page cuts, BM25, RRF, tree Top2, rerank Top6 and CHECK.',
        selector_analysis_boundary=rank_output.ANALYSIS_BUDGET,
        budgets='Separately account all preview tokens. Delivered whole texts <=8192 and <=6. Original cumulative input/output/request/time budgets, 4096 output tokens reserved for CHECK.',
        code={Path(p).name:vr.sha(p) for p in (__file__,cohort.__file__,base.__file__,runtime.__file__,
                                                 semantic.__file__,rank_output.__file__,PROCESSOR)})
    b.final.immutable(out/NAME/'design.json', config)
    return vr.digest(config)


def model_text(request, payload):
    data = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('<|', '\\u003c|')
    return [dict(role='system', content=semantic.ROLE), dict(role='user', content=request+'\n'+data)]


def timing_prefix(rows):
    keys = ('lookup_seconds', 'prepare_seconds', 'check_seconds', 'pipeline_seconds', 'after_prepare_seconds')
    averages = {k: mean(v[k] for v in rows) for k in keys}
    return dict(n=len(rows), means=averages, projected_468_seconds=468*averages['pipeline_seconds'],
                projected_50_seconds=50*averages['pipeline_seconds'], question_ids=[v['id'] for v in rows])


def selector_call(budget, dest, messages, cap, shape, history, stage_start):
    # The immutable CHECK transport is also used for historical PREPARE replay. Add the
    # selector-only wire field before that transport hashes, saves and sends this request.
    wire = shape.wire()
    engine = b.final.base.r
    original = engine.request_body

    def rank_request(*args, **kwargs):
        return dict(original(*args, **kwargs), vllm_xargs=wire)

    with patch.object(engine, 'request_body', rank_request):
        result = budget.call(dest, messages, cap, shape, history, stage_start)
    request = vr.read(dest/'request.json')['request']
    require(request.get('vllm_xargs') == wire, 'Rank constraints differ from the recorded request')
    encoding = {}
    if result['status'] == 'complete':
        tokens = vr.read(dest/'raw.json')['choices'][0]['token_ids']
        grammar = rank_output.RankGrammar(shape, rank_output.number_tokens(budget.tokenizer), request['max_tokens'])
        encoding = grammar.audit(tokens, result['value'])
    return result, encoding


def evaluate(row, arm, dest, old, plan, index, tok, client, stamp, *, replay=False):
    q = row['id']
    folder = dest/'rows'/q
    result_path = folder/'result.json'
    if result_path.is_file() and not replay:
        result = vr.read(result_path)
        require(result['id'] == q and result['signature'] == stamp, 'Different completed pipeline')
        return result
    if replay:
        require(result_path.is_file(), 'Missing terminal record')
    _, prep = b.prefix_record(b.FORMAL, old, 'formal', row)
    limits = plan['configuration']['limits']
    budget = base.previous.Budget(dest, plan, tok, client)
    history, files, detailed = [prep], {}, {}
    query = (prep['value'] or {}).get('queries', [])
    preview_tokens, full_tokens = 0, 0
    lookup, check_seconds = 0., 0.
    status, value, reason = 'content_failure', None, 'shared_prepare_failure'
    delivered = []
    stage_start = time.monotonic()

    def call(key, prompt, payload, shape):
        nonlocal preview_tokens, lookup, stage_start
        if replay:
            require((folder/key/'result.json').is_file(), 'Missing archived selection stage')
        messages = model_text(prompt, payload)
        field = 'children' if 'children' in payload else 'candidates'
        blank = model_text(prompt, dict(payload, **{field: []}))
        material_tokens = budget.count(messages) - budget.count(blank)
        require(material_tokens >= 0, 'Negative preview size')
        remaining = limits['output_tokens_per_sample'] - sum(x['usage']['completion_tokens'] for x in history)
        cap = min(4096 if key == 'rerank' else 2048, remaining-RESERVED_CHECK_OUTPUT)
        if cap <= 0 or len(history)+1 >= limits['max_requests_per_sample']:
            raise SelectionFailure('retrieval_budget_exhausted')
        result, encoding = selector_call(budget, folder/key, messages, cap, shape, history, stage_start)
        stage_start = time.monotonic()
        history.append(result)
        files[key] = vr.sha(folder/key/'result.json')
        preview_tokens += material_tokens
        lookup += result['wall_seconds']
        detailed[key] = dict(seconds=result['wall_seconds'], preview_tokens=material_tokens,
                             prompt_tokens=result['usage']['prompt_tokens'], encoding=encoding,
                             output_tokens=result['usage']['completion_tokens'], status=result['status'])
        if result['status'] != 'complete':
            raise SelectionFailure(key+': '+str(result['reason']))
        return result['value']

    if prep['status'] == 'complete':
        try:
            selected, trace = index.fused_selection(query, call)
            messages, delivered, full_tokens = base.pack(row, arm, prep['value'], selected, index, budget.count, limits)
            selection = dict(trace=trace, provided=delivered, preview_tokens=preview_tokens,
                             full_tokens=full_tokens, signature=stamp, query=query)
            if replay:
                require(vr.read(folder/'selection.json') == selection and (folder/'check/result.json').is_file(),
                        'Selection or CHECK replay mismatch')
            else:
                b.final.immutable(folder/'selection.json', selection)
            files['selection'] = vr.sha(folder/'selection.json')
            check = budget.call(folder/'check', messages, limits['max_tokens_per_response'],
                                b.final.p.original.validate_verdict, history, stage_start)
            require('vllm_xargs' not in vr.read(folder/'check/request.json')['request'], 'CHECK must use original decoding')
            files['check'] = vr.sha(folder/'check/result.json')
            setup = vr.read(folder/'check/request_budget.json')['setup_seconds']
            lookup += setup
            check_seconds = check['wall_seconds']-setup
            status, value, reason = check['status'], check['value'], check['reason']
            history.append(check)
        except SelectionFailure as exc:
            reason = str(exc)
    pipeline = prep['wall_seconds']+lookup+check_seconds
    usage = {k:sum(r['usage'][k] for r in history) for k in ('prompt_tokens','completion_tokens')}
    require(pipeline <= limits['wall_seconds_per_sample'] and len(history) <= limits['max_requests_per_sample']
            and usage['prompt_tokens'] <= limits['input_tokens_per_sample']
            and usage['completion_tokens'] <= limits['output_tokens_per_sample'], 'Cumulative budget exceeded')
    result = dict(id=q, signature=stamp, status=status, value=value, reason=reason, provided=delivered,
                  full_tokens=full_tokens, preview_tokens=preview_tokens, stage_times=detailed,
                  model_calls=len(history), usage=usage, lookup_seconds=lookup, prepare_seconds=prep['wall_seconds'],
                  check_seconds=check_seconds, pipeline_seconds=pipeline, after_prepare_seconds=lookup+check_seconds,
                  files=files)
    b.final.immutable(result_path, result)
    return result


def expansion_gate(out, stamp):
    release = vr.read(out/'acceptance_10.json')
    require(release['passed'] and release['signature'] == stamp and release['cohort'] == cohort.IDS
            and release['metrics_sha256'] == vr.sha(out/'metrics_10.json'), 'Missing repaired ten-question acceptance')
    for arm, hashes in release['accepted_rows'].items():
        require(set(hashes) == set(release['ids']) and all(
            vr.sha(out/arm/'rows'/q/'result.json') == digest for q,digest in hashes.items()),
            'Accepted row changed')
    require(set(release['accepted_rows']) == set(cohort.ARMS) and
            release['ids'] == vr.read(out.parent/'cohort.json')['ids'][:10], 'Acceptance coverage changed')


def run(folder, arm, stop_at, stamp):
    from transformers import AutoTokenizer
    root = b.RESULTS
    out = root/cohort.RESULT/NAME
    dest = out/arm
    if stop_at == 50:
        expansion_gate(out, stamp)
    rows = vr.read(out.parent/'inputs.json')[:stop_at]
    require(any(not (dest/'rows'/r['id']/'result.json').exists() for r in rows), 'Requested cohort already complete')
    tok = AutoTokenizer.from_pretrained(b.host.HOT/'models/gpt-oss-120b', local_files_only=True)
    rank_output.number_tokens(tok)
    index = base.load_index(root, arm)
    old = b.final.execution_plan(b.FORMAL)
    plan = runtime.checking_plan(arm, stamp)
    complete = []
    started = time.time()
    with runtime.large_service(folder, arm, stamp, dest/'STOP', rank_output=True), httpx.Client(trust_env=False) as client:
        ready = time.time()
        for n,row in enumerate(rows,1):
            require(not (dest/'STOP').exists(), 'STOP requested')
            complete.append(evaluate(row, arm, dest, old, plan, index, tok, client, stamp))
            progress = dict(visited=n, completed=len(complete), target=stop_at, unix=time.time(), signature=stamp)
            vr.atomic(dest/'progress.json', progress)
            if n in (5,10,50):
                b.final.immutable(dest/f'timing_{n}.json', timing_prefix(complete))
                b.final.immutable(folder/f'wall_{n}.json', dict(progress, worker_start_unix=started,
                                  service_ready_unix=ready, elapsed_since_ready_seconds=time.time()-ready))
    b.final.immutable(folder/'complete.json', dict(n=stop_at, signature=stamp, finished_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'confirm'))
    parser.add_argument('arm', choices=cohort.ARMS, nargs='?')
    parser.add_argument('--stop-at', type=int, choices=(10,50), default=10)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--root', type=Path, default=b.RESULTS)
    args = parser.parse_args()
    signature = design(args.root)
    if args.action != 'prepare':
        require(args.root == b.RESULTS and args.arm in cohort.ARMS, 'A node14 arm is required')
        dest = args.root/cohort.RESULT/NAME/args.arm
        runtime.detached(dest, args.attempt, signature, lambda folder:run(folder,args.arm,args.stop_at,signature),
                         args.action == 'confirm')
