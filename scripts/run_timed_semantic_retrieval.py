"""Time independent model and BM25 retrieval, fusion, reranking and final-answer decisions."""
import argparse
import json
import time
from pathlib import Path
from statistics import mean

import httpx

from rule_framework import semantic_index_retrieval as semantic
from scripts import timed_retrieval_baseline as base
from scripts import timed_retrieval_cohort as cohort
from scripts import timed_retrieval_runtime as runtime

b, vr, require = cohort.original, cohort.vr, cohort.require
MODEL = 'gpt-oss-120b'
NAME = 'semantic_120b'
MATERIAL_LIMIT = 8192
RESERVED_CHECK_OUTPUT = 4096


class SelectionFailure(Exception):
    """An invalid model response or an exhausted request/output budget."""


class IndexBudgetBoundary(Exception):
    """The next call or delivery would exceed the earlier combined material allowance."""


def design(root):
    out = root / cohort.RESULT
    require(vr.read(out/'cohort.json')['inputs_sha256'] == vr.sha(out/'inputs.json'), 'Cohort mismatch')
    config = dict(
        cohort=cohort.IDS, sources=cohort.SOURCES, base_sha256=cohort.CURRENT,
        original_tree=b.sources.TREE, eligible_leaf_count=883, original_leaf_count=895,
        material_reads='All flat cards, semantic Top2 per tree parent, original BM25 routing. Source holds excluded before model exposure.',
        query='The two routes independently use the same frozen 120B PREPARE queries.',
        representation='Native title up to 72 characters and 96 characters of case review, skill text or leaf logic.',
        first_page=semantic.FIRST_PAGE_SIZE, page_size=semantic.PAGE_SIZE, lexical_limit=24,
        semantic_limit=24, rrf_k=60, fused_limit=24, selector_model=MODEL, checker_model=MODEL,
        effort='medium', semantic_response_cap=2048, rerank_response_cap=4096,
        budgets='Whole delivered texts <=8192 tokens, <=6 items. Original cumulative request/input/output/time limits; reserve 4096 output for CHECK. Combined-budget mode pauses before the next over-budget preview or whole-text delivery. Separately accounted previews require explicit authorization. Reuse byte-identical prefixes only.',
        primary='Final P/R/F1 on valid binary decisions. Incorrect frozen 30B answers are positive. Report failures and abstentions.',
        timing='Sequential per dedicated GPU, simultaneous across arms. Timed lookup includes indexing requests and packing. Compute-equivalent end-to-end includes verified shared PREPARE. Measure cold start and active worker wall separately.',
        library_hash=vr.sha(root/'shared_experience/n2000/sources.json'),
        code={Path(p).name: vr.sha(p) for p in (__file__, cohort.__file__, base.__file__, runtime.__file__, semantic.__file__)})
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


def evaluate(row, arm, dest, old, plan, index, tok, client, stamp, separate):
    q = row['id']
    folder = dest/'rows'/q
    result_path = folder/'result.json'
    if result_path.is_file():
        result = vr.read(result_path)
        require(result['id'] == q and result['signature'] == stamp, 'Different completed pipeline')
        return result
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

    def call(key, prompt, payload, validator):
        nonlocal preview_tokens, lookup, stage_start
        messages = model_text(prompt, payload)
        field = 'children' if 'children' in payload else 'candidates'
        blank = model_text(prompt, dict(payload, **{field: []}))
        material_tokens = budget.count(messages) - budget.count(blank)
        require(material_tokens >= 0, 'Negative preview size')
        if not separate and preview_tokens + material_tokens > MATERIAL_LIMIT:
            raise IndexBudgetBoundary(key)
        remaining = limits['output_tokens_per_sample'] - sum(x['usage']['completion_tokens'] for x in history)
        cap = min(4096 if key == 'rerank' else 2048, remaining-RESERVED_CHECK_OUTPUT)
        if cap <= 0 or len(history)+1 >= limits['max_requests_per_sample']:
            raise SelectionFailure('retrieval_budget_exhausted')
        result = budget.call(folder/key, messages, cap, validator, history, stage_start)
        stage_start = time.monotonic()
        history.append(result)
        files[key] = vr.sha(folder/key/'result.json')
        preview_tokens += material_tokens
        lookup += result['wall_seconds']
        detailed[key] = dict(seconds=result['wall_seconds'], preview_tokens=material_tokens,
                             prompt_tokens=result['usage']['prompt_tokens'],
                             output_tokens=result['usage']['completion_tokens'], status=result['status'])
        if result['status'] != 'complete':
            raise SelectionFailure(key+': '+str(result['reason']))
        return result['value']

    if prep['status'] == 'complete':
        try:
            selected, trace = index.fused_selection(query, call)
            messages, delivered, full_tokens = base.pack(row, arm, prep['value'], selected, index, budget.count, limits)
            if not separate and preview_tokens+full_tokens > MATERIAL_LIMIT:
                raise IndexBudgetBoundary('full_text_delivery')
            b.final.immutable(folder/'selection.json', dict(trace=trace, provided=delivered,
                              preview_tokens=preview_tokens, full_tokens=full_tokens, signature=stamp, query=query))
            files['selection'] = vr.sha(folder/'selection.json')
            check = budget.call(folder/'check', messages, limits['max_tokens_per_response'],
                                b.final.p.original.validate_verdict, history, stage_start)
            files['check'] = vr.sha(folder/'check/result.json')
            setup = vr.read(folder/'check/request_budget.json')['setup_seconds']
            lookup += setup
            check_seconds = check['wall_seconds']-setup
            status, value, reason = check['status'], check['value'], check['reason']
        except IndexBudgetBoundary as exc:
            evidence = dict(id=q, signature=stamp, blocked_stage=str(exc), preview_tokens=preview_tokens,
                            selected_full_tokens=full_tokens, stages=detailed, files=files)
            b.final.immutable(folder/'combined_budget_boundary.json', evidence)
            return None
        except SelectionFailure as exc:
            reason = str(exc)
    pipeline = prep['wall_seconds']+lookup+check_seconds
    require(pipeline <= limits['wall_seconds_per_sample'], 'Cumulative time exceeded')
    result = dict(id=q, signature=stamp, status=status, value=value, reason=reason, provided=delivered,
                  full_tokens=full_tokens, preview_tokens=preview_tokens, stage_times=detailed,
                  lookup_seconds=lookup, prepare_seconds=prep['wall_seconds'], check_seconds=check_seconds,
                  pipeline_seconds=pipeline, after_prepare_seconds=lookup+check_seconds, files=files)
    b.final.immutable(result_path, result)
    return result


def run(folder, arm, stop_at, stamp, separate):
    from transformers import AutoTokenizer
    root = b.RESULTS
    out = root/cohort.RESULT
    dest = out/NAME/arm
    rows = vr.read(out/'inputs.json')[:stop_at]
    require(vr.sha(out/'inputs.json') == vr.read(out/'cohort.json')['inputs_sha256'], 'Input drift')
    require(any(not (dest/'rows'/r['id']/'result.json').exists() for r in rows), 'Prefix already complete')
    tok = AutoTokenizer.from_pretrained(b.host.HOT/'models/gpt-oss-120b', local_files_only=True)
    index = base.load_index(root, arm)
    old = b.final.execution_plan(b.FORMAL)
    plan = runtime.checking_plan(arm, stamp)
    complete, pending = [], []
    visit_start = time.time()
    with runtime.large_service(folder, arm, stamp, dest/'STOP'), httpx.Client(trust_env=False) as client:
        ready = time.time()
        for n, row in enumerate(rows, 1):
            require(not (dest/'STOP').exists(), 'STOP requested')
            result = evaluate(row, arm, dest, old, plan, index, tok, client, stamp, separate)
            if result is None:
                pending.append(row['id'])
            else:
                complete.append(result)
            progress = dict(visited=n, completed=len(complete), pending=len(pending), target=stop_at,
                            separate_preview_accounting=separate, unix=time.time(), signature=stamp)
            vr.atomic(dest/'progress.json', progress)
            if n in (5, 10, 50):
                if len(complete) == n:
                    b.final.immutable(dest/f'timing_{n}.json', timing_prefix(complete))
                b.final.immutable(folder/f'wall_{n}.json', dict(progress, worker_start_unix=visit_start,
                                  service_ready_unix=ready, elapsed_since_ready_seconds=time.time()-ready))
    b.final.immutable(folder/'complete.json', dict(n=stop_at, completed=len(complete), pending=pending,
                      signature=stamp, separate_preview_accounting=separate, finished_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'confirm'))
    parser.add_argument('arm', choices=cohort.ARMS, nargs='?')
    parser.add_argument('--stop-at', type=int, choices=(5, 10, 50), default=10)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--index-accounting', choices=('combined', 'separate'), default='combined')
    parser.add_argument('--root', type=Path, default=b.RESULTS)
    args = parser.parse_args()
    signature = design(args.root)
    if args.action != 'prepare':
        require(args.root == b.RESULTS and args.arm in cohort.ARMS, 'A node14 arm is required')
        dest = args.root/cohort.RESULT/NAME/args.arm
        runtime.detached(dest, args.attempt, signature,
            lambda folder: run(folder, args.arm, args.stop_at, signature, args.index_accounting == 'separate'),
            args.action == 'confirm')
