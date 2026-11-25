"""Prepare paired step-verifier inputs from the frozen checkpoint and retrieval trial."""
import time

from scripts import run_frozen_step_bm25 as frozen
from scripts import run_timed_semantic_retrieval as timed

vr, require = timed.vr, timed.require
ROOT = timed.b.RESULTS / 'retrieval_fusion_dev/step99_matched'


def prefix_for(parent, row):
    prefix, path = frozen.step.prefix_at(parent, row)
    require(all(vr.sha(p) == h for p, h in prefix['files'].items()), 'Different frozen PREPARE')
    ev = prefix['evidence']
    request = vr.read(ev['request_path'])['request']
    require(request['messages'] == frozen.base.p.query_messages(row) and
            request['model'] == frozen.base.r.EXPECTED_MODEL and
            request['max_tokens'] == frozen.base.p.QUERY_MAX_TOKENS, 'Different PREPARE request')
    prepared = dict(usage=ev['usage'], wall_seconds=vr.read(ev['response_path'])['elapsed_seconds'])
    return prefix, vr.sha(path), prepared


def select(folder, prefix, prepared, arm, index, budget):
    queries = prefix['queries']['queries']
    history = [prepared]
    stages = {}
    start = time.monotonic()
    preview = 0

    def rank(key, prompt, payload, shape):
        nonlocal preview, start
        require(not (ROOT / 'STOP').exists(), 'STOP requested')
        messages = timed.model_text(prompt, payload)
        field = 'children' if 'children' in payload else 'candidates'
        blank = timed.model_text(prompt, dict(payload, **{field: []}))
        preview += budget.count(messages) - budget.count(blank)
        limits = budget.limits
        left = limits['output_tokens_per_sample'] - sum(v['usage']['completion_tokens'] for v in history)
        cap = min(4096 if key == 'rerank' else 2048, left - timed.RESERVED_CHECK_OUTPUT)
        if cap <= 0 or len(history) + 1 >= limits['max_requests_per_sample']:
            raise timed.SelectionFailure('selection_budget_exhausted')
        result, _ = timed.selector_call(budget, folder / key, messages, cap, shape, history, start)
        start = time.monotonic()
        history.append(result)
        stages[key] = vr.sha(folder / key / 'result.json')
        if result['status'] != 'complete':
            raise timed.SelectionFailure(key + ': ' + str(result['reason']))
        return result['value']

    failure = None
    try:
        chosen, trace = index.fused_selection(queries, rank)
    except timed.SelectionFailure as error:
        failure, chosen, trace = str(error), [], None
    lexical, routes = index.lexical('\n'.join(queries))
    selection = dict(arm=arm, query=queries, bm25=lexical[:6], fusion=chosen, trace=trace,
                     lexical_routes=routes, failure=failure, preview_tokens=preview, stages=stages)
    charges = dict(requests=len(stages), seconds=sum(s['wall_seconds'] for s in history[1:]),
                   **{k: sum(s['usage'][k] for s in history[1:]) for k in ('prompt_tokens', 'completion_tokens')})
    return dict(selection, charges=charges)


def packet(row, arm, prefix, ranked, index, count, limits):
    messages_for = frozen.step.continuation if arm == 'case_rag' else frozen.step.ORIGINAL_CONTINUATION
    def messages(ids):
        return messages_for(row, prefix['queries'], [index.store.view(i) for i in ids])
    base = count(messages([]))
    limit = limits['context_window'] - limits['output_tokens_per_sample']
    require(base <= limit and len(set(ranked)) == len(ranked) <= 6, 'Invalid material packet')
    provided = []
    for i in ranked:
        require(i in index.cards, 'Outside approved material scope')
        size = count(messages(provided + [i]))
        if size <= limit and size - base <= timed.MATERIAL_LIMIT:
            provided.append(i)
    result = messages(provided)
    return dict(messages=result, references=[index.store.view(i) for i in provided],
                retrieval=dict(provided_complete_ids=provided, reference_tokens=count(result)-base,
                               candidate_ids=ranked))


def check_limits(limits, prepared, charges, mode):
    # Both CHECKs keep the same available output. Only fusion incurs selector time/inputs/calls.
    reduced = dict(limits, output_tokens_per_sample=limits['output_tokens_per_sample']-charges['completion_tokens'],
                   wall_seconds_per_sample=limits['wall_seconds_per_sample']-prepared['wall_seconds'])
    if mode == 'fusion':
        reduced['max_requests_per_sample'] -= charges['requests']
        reduced['input_tokens_per_sample'] -= charges['prompt_tokens']
        reduced['wall_seconds_per_sample'] -= charges['seconds']
    return reduced
