"""Time the released BM25 lookup on the frozen sample and bind its original CHECK times."""
from pathlib import Path
from statistics import mean
import time

from scripts import timed_retrieval_cohort as cohort
from scripts import confirm_rag_tree_retrieval as previous
from rule_framework import semantic_index_retrieval as semantic

b, vr, require = cohort.original, cohort.vr, cohort.require


def load_index(root, arm):
    meta = root / 'shared_experience/n2000'
    require(vr.sha(root/cohort.BM25/'sources.json') == cohort.SOURCES and
            vr.sha(meta/'sources.json') == '83bef4d56d790af2aced7c10bfd5e7dc2d42f4905e08b2395ac5894471d2fb7f'
            and vr.sha(root/cohort.BENCH/'catalog.json') == b.sources.TREE, 'Foreign library release')
    spec = vr.read(root/cohort.BM25/'sources.json')['groups']['n2000']
    hashes = dict(spec['files'], **{'catalog.json':'901054712673f89c146233830d0c57bc8a31ec2e95e65963b9e4b3afed63ce42'})
    require(all(vr.sha(meta/'materials'/k) == h for k,h in hashes.items()), 'Materials changed')
    allowed = vr.read(meta/'sources.json')['leaf_errors']
    index = semantic.Index(arm, meta/'materials', vr.read(root/cohort.BENCH/'catalog.json'), allowed)
    require(len(index.ids) == {'case_rag':729,'flat_skills':1452,'rule_tree':883}[arm], 'Unexpected eligible index size')
    return index


def count_messages(tokenizer, messages):
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, reasoning_effort='medium'))


def pack(row, arm, queries, ids, index, count, limits, *, english=False):
    def messages(chosen):
        return b.final.p.continuation_messages(row, queries, [index.store.view(i) for i in chosen], arm=arm, english=english)
    base = count(messages([]))
    cap = limits['context_window'] - limits['output_tokens_per_sample']
    require(base <= cap and len(ids) <= 6 and len(ids) == len(set(ids)), 'Unpackable or duplicated candidates')
    chosen = []
    for item in ids:
        require(item in index.cards, 'Item outside approved release')
        n = count(messages(chosen+[item]))
        if n-base <= 8192 and n <= cap:
            chosen.append(item)
    result = messages(chosen)
    return result, chosen, count(result)-base


def bm25_lookup(row, arm, queries, index, count, limits, *, english=False):
    if english:
        b.final.base.p.validate_queries(queries, english=True)
    query = '\n'.join(queries['queries'])
    if index.tree is None:
        ranked = index.store.search(query)
    else:
        ranked, trace = index.tree.retrieve(query)
        if any(hit['id'] not in index.cards for hit in ranked):
            eligible = {i for route in trace['routes'] for i in route.get('leaf_ids', []) if i in index.cards}
            ranked = index.store.search(query, allowed_ids=eligible)
    return pack(row,arm,queries,[x['id'] for x in ranked],index,count,limits, english=english)


def archived_check(arm, q, control, root):
    k = vr.digest(q)[:24]
    current = root/cohort.BM25/'final/formal'/arm/k/'result.json'
    require(vr.sha(current) == control['file_sha256'], 'Published row changed')
    data = vr.read(current)
    if data['source'] == 'frozen_baseline':
        require(vr.sha(data['original_result']) == data['original_sha256'], 'Original control changed')
        data = vr.read(data['original_result'])
    paths = [Path(p) for p in data['files'] if p.endswith('/check/result.json')]
    if not paths:
        return None
    require(len(paths)==1 and vr.sha(paths[0]) == data['files'][str(paths[0])], 'Missing archived CHECK')
    return vr.read(paths[0])


def run():
    from transformers import AutoTokenizer
    root = b.RESULTS
    out = root/cohort.RESULT
    rows = cohort.freeze(root)
    control = vr.read(out/'bm25_baseline.json')
    require(control['cohort'] == cohort.IDS and control['source_sha256'] == cohort.CURRENT, 'Baseline drift')
    tok = AutoTokenizer.from_pretrained(b.host.HOT/'models/gpt-oss-120b',local_files_only=True)
    old = b.final.execution_plan(b.FORMAL)
    limits = old['configuration']['limits']
    results = {}
    for arm in cohort.ARMS:
        index = load_index(root,arm)
        group = {}
        for row in rows:
            q = row['id']
            _, prep = b.prefix_record(b.FORMAL,old,'formal',row)
            ref = control['rows'][arm][q]
            saved = archived_check(arm,q,ref,root)
            before = time.monotonic()
            ids = []
            if prep['status']=='complete':
                _, ids, _ = bm25_lookup(row,arm,prep['value'],index,lambda m:count_messages(tok,m),limits)
            lookup = time.monotonic()-before
            require(ids == ref['delivered'] or arm=='rule_tree' and any(i not in index.cards for i in ref['delivered']),
                    'Unexpected divergence from frozen BM25')
            group[q] = dict(prepare_seconds=prep['elapsed_seconds'], lookup_seconds=lookup,
                            archived_check_seconds=saved['elapsed_seconds'] if saved else 0,
                            same_delivery=ids==ref['delivered'], eligible_ids=ids)
        results[arm]=group
        vr.atomic(out/'bm25_timing_progress.json',dict(arms=list(results)))
    summary = {}
    qs = [r['id'] for r in rows]
    for n in (5,10,50):
        summary[str(n)] = {}
        for arm,group in results.items():
            subset=[group[q] for q in qs[:n]]
            summary[str(n)][arm]={k:mean(d[k] for d in subset) for k in ('lookup_seconds','prepare_seconds','archived_check_seconds')}
    b.final.immutable(out/'bm25_timing.json',dict(rows=results,prefixes=summary,benchmark=cohort.IDS,
            note='Fresh CPU BM25+packing on node14; PREPARE and CHECK elapsed seconds are archived model calls.'))


if __name__ == '__main__':
    b.node.node()
    with b.final.base.r.run_lock(b.RESULTS/cohort.RESULT/'bm25_timing.lock'):
        if not (b.RESULTS/cohort.RESULT/'bm25_timing.json').is_file():
            run()
