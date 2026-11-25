"""Small, isolated comparison of delivered skills. No verifier decisions or bench labels."""
import argparse
from itertools import zip_longest
import json
from pathlib import Path
import time
from unittest.mock import patch

import httpx
from rule_framework import flat_skill_rag as flat
from rule_framework import verifier_english as en
from scripts import gpt_oss_harmony_transport as transport
from scripts import serve_frozen_bm25 as node
from scripts.run_qwen30b_step_checks import validate_effective_input

host, vr, require = node.host, node.vr, node.require
ROOT = host.HOT / 'results/qwen30b_step_verifier'
OUT = ROOT / 'retrieval_fusion_dev'
PORT = 18245
GROUP = 'n1000_seed1247261125'
REWRITE = ('Return JSON {"query":"..."} only. Generate one concise bilingual Chinese-English library query '
           'naming transferable physical relations, assumptions and useful alternative terminology for checking '
           'this problem. Generalize away names and numbers. Do not solve, judge or presuppose the answer.')
RANK = ('Rank the candidate skills for checking the CURRENT problem. Consider the specific relation, '
        'assumptions and applicability, not just the subject. Return JSON {"ids":[...]} using up to six '
        'displayed integer IDs in descending usefulness. Empty is valid. No diagnosis or explanations.')
RUBRIC = (
    'Read each delivered whole skill without ranks or method. Grade 1 only for an actionable valid relation '
    'or check transferable to a necessary step under the stated assumptions. Topic overlap, generic advice, '
    'wrong physics and incompatible mechanisms score 0. Primary: relevant whole items delivered per real '
    'question (six slots); secondary: any relevant hit and reciprocal rank of first relevant item. '
    'Report controls separately. Candidate for replication if superior on >=3/4 real questions by primary '
    'endpoint with no decrease on remaining questions, compared with frozen BM25. Do not infer population gain.')
SKILL_SHA = '6e11ffe22058f74ee317e7342f84e01d31ffa30595c024a6eb37c1d81efee6d2'


def rewrite_messages(row, prefix, *, english=False):
    require(len(prefix) == 3 and prefix[:2] == host.final.p.query_messages(row, english=english)
            and prefix[2]['role'] == 'assistant', 'Incompatible shared PREPARE for query rewrite')
    flat.validate_queries(json.loads(prefix[2]['content']), english=english)
    return [*prefix, dict(role='user', content=en.REWRITE if english else REWRITE)]


def top24(memory, query, allowed_ids=None):
    remaining = set(memory.ids if allowed_ids is None else allowed_ids)
    ids = []
    while len(ids) < 24:
        hits = memory.search(query, allowed_ids=remaining)
        if not hits:
            break
        batch = [h['id'] for h in hits]
        ids.extend(batch)
        remaining.difference_update(batch)
    return ids[:24]


def merge(left, right):
    alternating = list(dict.fromkeys(k for pair in zip_longest(left, right) for k in pair if k))
    scores = {key: 0.0 for key in alternating}
    for ranking in (left, right):
        for rank, key in enumerate(ranking, 1):
            scores[key] += 1 / (60 + rank)
    return alternating[:24], sorted(alternating, key=lambda k: -scores[k])[:24]


def call(folder, client, url, config, messages, tokenizer):
    folder.mkdir(exist_ok=True, parents=True)
    result = folder / 'response.json'
    if result.exists():
        require(vr.read(folder / 'request.json')['request']['messages'] == messages, 'Resume input drift')
        response = vr.read(result)
    else:
        require(not (folder / 'request.json').exists(), 'Interrupted request; inspect recorded evidence')
        response = transport.chat_response(client, endpoint=url+'/chat/completions', model=config['model'],
            messages=messages, max_tokens=4096, timeout=600, raw_path=folder/'raw.json',
            request_path=folder/'request.json', response_path=result)
    raw = vr.read(folder/'raw.json')
    validate_effective_input(raw, tokenizer, node.CHECK_DATE)
    require(response['model_ids'] == [config['model']] and response.get('content'), 'Missing model output')
    try:
        value = json.loads(response['content'])
    except ValueError:
        value = None
    return value, response['elapsed_seconds']


def run_row(row, folder, store, tokenizer, client, url, cfg, acceptance):
    sid = row['id']
    pkt = host.OUT/'pilot/flat_skills'/vr.digest(sid)[:24]/'reference_packet.json'
    require(vr.sha(pkt) in {v for k,v in acceptance['files'].items() if k.endswith(str(pkt.relative_to(host.OUT)))},
            'Wrong pilot packet')
    prefix = vr.read(pkt)['messages'][:3]
    english = en.active(cfg)
    rewrite = rewrite_messages(row, prefix, english=english)
    queries = json.loads(prefix[2]['content'])
    count = lambda ms: len(tokenizer.apply_chat_template(ms, tokenize=True, add_generation_prompt=True, reasoning_effort='medium'))
    def size(keys):
        return count(host.final.p.continuation_messages(row, queries, [store.view(k) for k in keys],
                                                        arm='flat_skills', english=english))
    base = size([])
    def pack(keys, preview=0):
        chosen = []
        for key in keys[:6]:
            if size(chosen+[key])-base+preview <= 8192:
                chosen.append(key)
        return chosen
    first = top24(store, '\n'.join(queries['queries']))
    methods = dict(bm25=pack(first))
    value, secs = call(folder/'relation', client, url, cfg, rewrite, tokenizer)
    valid = isinstance(value,dict) and isinstance(value.get('query'),str) and bool(value['query'].strip())
    if english and valid:
        try:
            en.require_english_query(value['query'])
        except ValueError:
            valid = False
    record = dict(id=sid, methods=methods, first=first, relation_valid=valid, rewrite_seconds=secs)
    if valid:
        second = top24(store,value['query'])
        alternating, fused = merge(first,second)
        methods.update(relation_only=pack(second), alternating=pack(alternating), rrf=pack(fused))
        history = rewrite+[dict(role='assistant',content=json.dumps(value, ensure_ascii=False))]
        def prompt(cards):
            return history+[dict(role='user',content=RANK+'\n'+json.dumps(cards,ensure_ascii=False).replace('<|','\\u003c|'))]
        # Stable opaque numbers carry no fusion score. Keep each candidate's native title and opening excerpt.
        cards = [dict(id=i+1,title=store.docs[k]['title'],excerpt=store.docs[k]['text'][:160]) for i,k in enumerate(fused)]
        cards = sorted(cards,key=lambda c: vr.digest([sid,c['id']]))
        preview = max(0,count(prompt(cards))-count(prompt([])))
        require(preview < 8192, 'Candidate previews exceed material budget')
        value, rank_seconds = call(folder/'rerank',client,url,cfg,prompt(cards),tokenizer)
        ids = value.get('ids') if isinstance(value,dict) else None
        ok = isinstance(ids,list) and len(ids)<=6 and all(type(i) is int and 1<=i<=len(fused) for i in ids) and len(ids)==len(set(ids))
        if ok:
            methods['rrf_rerank'] = pack([fused[i-1] for i in ids], preview)
        record.update(second=second,merged=fused,rank_valid=ok,rank_seconds=rank_seconds,preview_tokens=preview)
    record['full_tokens'] = {k:size(v)-base for k,v in methods.items()}
    vr.atomic(folder/'outcome.json',record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','confirm'])
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args()
    folder=OUT/'attempts'/str(args.attempt)
    cfg=vr.read(host.OUT/'plan.json')['configuration']
    signature=vr.digest([vr.sha(__file__),host.PILOT,SKILL_SHA,PORT])
    if args.action=='confirm':
        return node.confirm(folder,signature)
    node.node()
    folder.mkdir(exist_ok=False,parents=True)
    try:
        with host.final.base.r.run_lock(OUT/'active.lock'):
            require(not (OUT/'summary.json').exists(), 'Run already complete')
            path=ROOT/'shared_experience'/GROUP/'materials/skills.json'
            require(vr.sha(path)==SKILL_SHA and vr.sha(host.OUT/'pilot_inputs.json')==host.PILOT,'Wrong frozen inputs')
            require(vr.read(ROOT/'library_source_rerun_20261124/bm25_n2000/sources.json')['groups'][GROUP]['files']['skills.json']==SKILL_SHA,'Wrong source release')
            rows=vr.read(host.OUT/'pilot_inputs.json')
            require(len(rows)==6,'Need all six controls and real inputs')
            host.final.immutable(OUT/'design.json',dict(signature=signature,source=SKILL_SHA,sample=host.PILOT,
                rubric=RUBRIC,query_prompt=REWRITE,rank_prompt=RANK,cutoffs=[24,6],rrf_constant=60))
            host.final.base.runtime.wait_detached(folder,signature)
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(host.HOT/'models/gpt-oss-120b',local_files_only=True)
            store=flat.FlatMemory(vr.read(path))
            original=host.service_script
            def launch(config):
                script,paths=original(config)
                require(script.count('18083')==2,'Unexpected launcher ports')
                return script.replace('18083',str(PORT)),paths
            acceptance=vr.read(host.OUT/'pilot_acceptance.json')
            outcomes=[]
            with patch.object(host,'PORT',PORT), patch.object(host,'service_script',launch), node.service(folder,5,OUT/'STOP',signature) as url, httpx.Client(trust_env=False) as client:
                for row in rows:
                    dest=OUT/'questions'/row['id']
                    outcomes.append(run_row(row,dest,store,tok,client,url,cfg,acceptance))
                    vr.atomic(folder/'progress.json',dict(completed=len(outcomes),unix=time.time()))
            blind={}
            for result in outcomes:
                ids={key for values in result['methods'].values() for key in values}
                blind[result['id']]=[{k:store.docs[key][k] for k in ('id','title','text')}
                    for key in sorted(ids,key=lambda k:vr.digest(['blind',result['id'],k]))]
            vr.atomic(OUT/'blind.json',blind)
            vr.atomic(OUT/'summary.json',outcomes)
        vr.atomic(folder/'exit.json',dict(code=0))
    except Exception as exc:
        vr.atomic(folder/'exit.json',dict(code=1,error=str(exc)))
        raise


if __name__=='__main__':
    main()
