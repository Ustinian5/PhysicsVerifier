"""Paired RAG and tree comparisons with the fixed 26-question skills cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time
from unittest.mock import patch

import httpx
from scripts import confirm_skill_retrieval as previous

final, host, pilot, roles = previous.final, previous.host, previous.pilot, previous.roles
node, vr, require = previous.node, previous.vr, previous.require
PARENT = previous.REL
RESULT = PARENT.parent/'expanded_rag_tree'
GPUS = {'case_rag': 5, 'rule_tree': 6, 'pure': 7}
PORTS = {'case_rag': 18246, 'rule_tree': 18247, 'pure': 18248}
INPUT_HASH = 'd333bdd73845ba0a40cf8eeb46b90547c351f47f894b37ed52220e8279eec055'
SUMMARY_HASH = '3e066228c883d5c684fd3e50bedf73146c27d4528942863bb50aecc7378cc0a1'
TREE_HASH = '36a7be73e87ea00cfbaed223a3e177255b03e91c7349be5550b9700fffeca587'
RAG_HASH = 'f3dcfab7d57a87598c2642a4ebf8ffbb1f00c24bad5aea4fdccd71f41ff568f3'
DIAGNOSTICS = {
    '1820d217a0349ab84701b2cd819fa1c1929a8d9a4f46a7dd3acd2b8ee389950f': 'order',
    'c246c23aece32bd94d2211e3d018386b400d46395df71e8748f879936292023e': 'spring',
}
NOUN = {'case_rag':'historical cases', 'rule_tree':'rule-tree leaves'}


def library(root):
    return root/'shared_experience'/pilot.GROUP


def prepare(root):
    parent = root/PARENT
    out = root/RESULT
    require(vr.sha(parent/'inputs.json')==INPUT_HASH and vr.sha(parent/'summary.json')==SUMMARY_HASH,
            'Changed development cohort or earlier outcomes')
    rows = vr.read(parent/'inputs.json')
    design = vr.read(parent/'design.json')
    require(len(rows)==26 and design['classes']=={'correct':26} and design['input_sha256']==INPUT_HASH,
            'Cohort differs from the skills confirmation')
    lib = library(root)
    receipt = vr.read(lib/'construction_acceptance.json')
    manifest = vr.read(lib/'sources.json')
    require(receipt['passed'] and receipt['counts']['drawn']==1000
            and vr.sha(lib/'sources.json')==receipt['sources_sha256']
            and vr.sha(lib/'material_isolation.json')==receipt['material_isolation_sha256'],
            'Unverified development library')
    for name,h in [('cases',RAG_HASH),('skills',pilot.SKILL_SHA),('catalog',TREE_HASH)]:
        require(vr.sha(lib/'materials'/(name+'.json'))==receipt['materials'][name+'.json']==h,
                'Wrong material release')
    screen = previous.FamilyScreen(rows)
    ids = {r['id'] for r in rows}
    cases = vr.read(lib/'materials/cases.json')
    tree = pilot.flat.TreeMemory(vr.read(lib/'materials/catalog.json'))
    good = set(manifest['common_reviews'])
    require(len(cases)==355 and {s for c in cases for s in c['source_ids']}==good
            and not good.intersection(manifest['source_hold'])
            and len(tree.corpus.leaves)==446 and set(manifest['leaf_errors'])==set(tree.corpus.leaves)
            and all(all(s in good for s,_ in refs) for refs in manifest['leaf_errors'].values()),
            'Tree or case lineage mismatch')
    surfaces = [*cases,*tree.memory.docs.values(),*tree.nodes.docs.values()]
    for d in surfaces:
        hits = screen.matches(dict(id=d['id'],question=d['title'],prediction=d['text']))
        require(not any(h['heldout_id'] in ids for h in hits),'Development item appears in material '+d['id'])
    stop = 'Select a maximum of six ranked items, admit whole texts in order, charge previews plus texts <=8192 tokens.'
    record = dict(input_sha256=INPUT_HASH, prior_summary_sha256=SUMMARY_HASH, prior_design=vr.sha(parent/'design.json'),
        rubric_sha256=vr.sha(parent/'rubric.json'), material_receipt=vr.sha(lib/'construction_acceptance.json'),
        files=receipt['materials'], group=pilot.GROUP, methods=list(previous.METHODS), role=roles.ROLE,
        rank={a:pilot.RANK.replace('candidate skills','candidate '+v) for a,v in NOUN.items()},
        packing=stop, tree='Per query run original positive Top2 routes. Admit terminal leaves only. '
        'For RRF fuse two within-route positive leaf rankings (k=60), no flat fallback.',
        primary='Useful whole texts per question, including zero deliveries. Reuse the 26 question-specific rubrics. '
        'Two-sided paired sign tests for both new methods vs BM25 within each library. Holm over those four tests.',
        secondary='Report exact paired false-positive tests on common binary decisions, coverage and abstentions. '
        'Same 26 question families for all comparisons. Correct class only.',
        diagnostic='All 26 get a no-material control. Two previously observed skills errors also use the archived '
        'RRF CHECK cap, history and prompt, comparing its original delivery to empty, original-query BM25 and '
        'relation-ranked delivery. Reuse exact archived bodies. This case analysis is post hoc.',
        prior_checkpoint=host.CONFIG, human_gold=False)
    final.immutable(out/'design.json',record)
    return record


def run_signature(out, arm):
    return vr.digest([vr.sha(__file__),vr.sha(out/'design.json'),arm,
                      vr.sha(roles.__file__),vr.sha(pilot.__file__)])


def make_plan(parent,arm,out):
    config = vr.read(host.OUT/'plan.json')['configuration']
    prior_config = dict(config,api_base=f'http://127.0.0.1:{pilot.PORT}/v1')
    prior_stamp = previous.signature(parent)
    require(vr.read(parent/'attempts/1/disconnect_acceptance.json')['configuration_sha256']==prior_stamp,
            'Earlier run signature changed')
    before = dict(configuration=prior_config,configuration_sha256=vr.digest([prior_config,prior_stamp]))
    cfg = dict(config,api_base=f'http://127.0.0.1:{PORTS[arm]}/v1')
    return before,dict(configuration=cfg,configuration_sha256=vr.digest([cfg,run_signature(out,arm)]))


def prepared(row,parent,old):
    folder = parent/'rows'/row['id']
    messages = final.p.query_messages(row)
    wall = old['configuration']['limits']['wall_seconds_per_sample']
    prep = final.saved_request(folder/'prepare',messages,pilot.flat.QUERY_MAX_TOKENS,
                               pilot.flat.validate_queries,old,available=wall)
    require(prep['status']=='complete','Frozen shared PREPARE missing')
    prefix = messages+[dict(role='assistant',content=json.dumps(prep['value'],ensure_ascii=False))]
    query_messages = prefix+[dict(role='user',content=pilot.REWRITE)]
    rewrite = final.saved_request(folder/'rewrite',query_messages,4096,previous.query,old,
                                  available=wall-prep['wall_seconds'])
    require(rewrite['status']=='complete','Frozen shared rewrite missing')
    rank_prefix = [dict(role='system',content=roles.ROLE),*query_messages[1:],
                   dict(role='assistant',content=json.dumps(rewrite['value'],ensure_ascii=False))]
    return prep,rewrite,rank_prefix


def candidates(store,tree,query):
    if tree is None:
        return pilot.top24(store,query),None
    top,trace = tree.retrieve(query)
    eligible = {i for route in trace['routes'] for i in route.get('leaf_ids',[])}
    ranked = pilot.top24(store,query,eligible)
    require(ranked[:6]==[h['id'] for h in top] and len(eligible)==trace['eligible_leaf_count'],
            'Tree scope differs from original hierarchy')
    return ranked,trace


class Budget:
    def __init__(self,out,plan,tokenizer,client):
        self.out,self.plan,self.tokenizer,self.client = out,plan,tokenizer,client
        self.limits = plan['configuration']['limits']

    def count(self,messages):
        return len(self.tokenizer.apply_chat_template(messages,tokenize=True,
                   add_generation_prompt=True,reasoning_effort='medium'))

    def call(self,folder,messages,cap,validator,history,started):
        lim = self.limits
        maximum = min(cap,lim['output_tokens_per_sample']-sum(v['usage']['completion_tokens'] for v in history))
        available = lim['wall_seconds_per_sample']-sum(v['wall_seconds'] for v in history)
        require(maximum>0 and available>0 and len(history)<lim['max_requests_per_sample'] and
                sum(v['usage']['prompt_tokens'] for v in history)+self.count(messages)<=lim['input_tokens_per_sample'],
                'Cumulative budget exhausted')
        return final.request_once(self.out,folder,messages,maximum,validator,self.plan,self.tokenizer,
                                  self.client,available=available,started=started)


def case_diagnostics(row,parent,folder,old,budget,prep,rewrite,store):
    archived = parent/'rows'/row['id']
    prior = vr.read(archived/'delivery.json')
    require(prior==vr.read(parent/'summary.json')[row['id']],'Incorrect archived delivery')
    wall = budget.limits['wall_seconds_per_sample']-prep['wall_seconds']-rewrite['wall_seconds']
    ranked_folder = archived/'rrf/rank'
    rank_record = vr.read(ranked_folder/'request.json')['request']
    cards = json.loads(rank_record['messages'][-1]['content'].split('\n',1)[1])
    ranked = final.saved_request(ranked_folder,rank_record['messages'],rank_record['max_tokens'],
                                 lambda v:previous.rank(v,len(cards)),old,available=wall)
    history = [prep,rewrite,ranked]
    original = vr.read(archived/'rrf/check/request.json')['request']
    cap = original['max_tokens']
    variants = {'archived_rrf':prior['rrf']['ids'],'empty':[],'original_bm25':prior['bm25']['ids'],
                'relation':prior['relation']['ids']}
    if DIAGNOSTICS[row['id']]=='order':
        left,right = variants['relation'],variants['archived_rrf']
        require(left==[right[1],right[0],*right[2:]],'Not a first-two swap')
    else:
        # Test the gravity-focused pair within the original relation list; keep every other item and its order.
        hints = [i for i in variants['relation'] if store.docs[i]['title'] in (
                 '投影重力得到沿任意路径的加速度','检查重势能参考高度与垂直位移的计算')]
        require(len(hints)==2,'Expected two previously noted gravity hints')
        variants['relation_without_gravity_pair']=[i for i in variants['relation'] if i not in hints]
    results = {}
    for name,ids in variants.items():
        messages = final.p.continuation_messages(row,prep['value'],[store.view(i) for i in ids],arm='flat_skills')
        require(messages[:-1]==original['messages'][:-1], 'Counterfactual prefix mismatch')
        if name=='archived_rrf':
            record = final.saved_request(archived/'rrf/check',messages,cap,final.p.original.validate_verdict,
                                         old,available=wall-ranked['wall_seconds'])
            origin = archived/'rrf/check'
        else:
            origin = folder/'diagnostics'/name
            record = budget.call(origin,messages,cap,final.p.original.validate_verdict,history,time.monotonic())
        body = vr.read(origin/'request.json')['request']
        require(dict(body,messages=original['messages'])==original and body['messages']==messages,
                'Counterfactual changed the request apart from delivered material')
        results[name]=dict(ids=ids,status=record['status'],value=record['value'],cap=cap,
                           request_sha256=vr.sha(origin/'request.json'),result_sha256=vr.sha(origin/'result.json'))
    return results


def initialize(root,arm):
    from transformers import AutoTokenizer
    global worker_state
    out = root/RESULT
    before,plan = make_plan(root/PARENT,arm,out)
    tree = None
    lib = library(root)/'materials'
    if arm=='rule_tree':
        tree=pilot.flat.TreeMemory(vr.read(lib/'catalog.json'))
        store=tree.memory
    else:
        store=pilot.flat.FlatMemory(vr.read(lib/('cases.json' if arm=='case_rag' else 'skills.json')))
    tok=AutoTokenizer.from_pretrained(host.HOT/'models/gpt-oss-120b',local_files_only=True)
    worker_state=(root,arm,tree,store,before,Budget(out/arm,plan,tok,httpx.Client(trust_env=False)))


def evaluate(row):
    root,arm,tree,store,old,budget = worker_state
    parent=root/PARENT
    folder = budget.out/'rows'/row['id']
    prep,rewrite,rank_prefix=prepared(row,parent,old)
    queries=prep['value']
    def messages(ids):
        return final.p.continuation_messages(row,queries,[store.view(i) for i in ids],arm=arm)
    empty_size=budget.count(messages([]))
    if arm=='pure':
        check=budget.call(folder/'check',messages([]),budget.limits['max_tokens_per_response'],
                          final.p.original.validate_verdict,[prep],time.monotonic())
        result=dict(status=check['status'],value=check['value'])
        if row['id'] in DIAGNOSTICS:
            result['diagnostics']=case_diagnostics(row,parent,folder,old,budget,prep,rewrite,store)
        final.immutable(folder/'delivery.json',result)
        return result
    outcome={}
    for method in previous.METHODS:
        started=time.monotonic()
        history=[prep]
        preview=0
        routes={}
        if method in ('bm25','rrf'):
            first,routes['original']=candidates(store,tree,'\n'.join(queries['queries']))
        if method=='bm25':
            pool=first
            ordered=pool[:6]
        else:
            second,routes['relation']=candidates(store,tree,rewrite['value']['query'])
            pool=second if method=='relation' else pilot.merge(first,second)[1]
            history.append(rewrite)
            ordered=[]
            if pool:
                prompt=pilot.RANK.replace('candidate skills','candidate '+NOUN[arm])
                request,blank=roles.single_pool_request([*rank_prefix,{}],pool,row['id'],store,prompt)
                preview=budget.count(request)-budget.count(blank)
                require(0<=preview<=8192,'Candidate previews exceed material budget')
                ranked=budget.call(folder/method/'rank',request,4096,lambda v:previous.rank(v,len(pool)),history,started)
                history.append(ranked)
                if ranked['status']!='complete':
                    outcome[method]=dict(ids=[],status='rank_failure',verdict=None,pool=pool,preview_tokens=preview,
                                         routes=routes,full_tokens=0,extra_model_seconds=sum(a['elapsed_seconds'] for a in history[1:]))
                    continue
                ordered=[pool[i-1] for i in ranked['value']['ids']]
                started=time.monotonic()
        ids,omitted=[],[]
        for i in ordered:
            size=budget.count(messages([*ids,i]))
            if size-empty_size+preview<=8192 and size<=budget.limits['context_window']-budget.limits['output_tokens_per_sample']:
                ids.append(i)
            else:
                omitted.append(i)
        check_messages=messages(ids)
        if method=='bm25':
            frozen=final.p.reference_packet(row,arm,queries,tree if tree else store,budget.count)
            require(check_messages==frozen['messages'] and ids==frozen['retrieval']['provided_complete_ids'],
                    'New baseline differs from frozen reference_packet')
        check=budget.call(folder/method/'check',check_messages,budget.limits['max_tokens_per_response'],
                          final.p.original.validate_verdict,history,started)
        outcome[method]=dict(ids=ids,requested=ordered,omitted=omitted,status=check['status'],
            verdict=(check['value'] or {}).get('verdict'),pool=pool,routes=routes,preview_tokens=preview,
            full_tokens=budget.count(check_messages)-empty_size, extra_model_seconds=sum(a['elapsed_seconds'] for a in history[1:]))
    final.immutable(folder/'delivery.json',outcome)
    return outcome


def run(root,arm,attempt):
    node.node()
    out=root/RESULT
    armout=out/arm
    folder=armout/'attempts'/str(attempt)
    folder.mkdir(exist_ok=False,parents=True)
    stamp=run_signature(out,arm)
    try:
        with host.final.base.r.run_lock(armout/'active.lock'):
            require(not (armout/'summary.json').exists(),'Already complete')
            require(prepare(root)==vr.read(out/'design.json'),'Development sources changed')
            host.final.base.runtime.wait_detached(folder,stamp)
            original=host.service_script
            def launch(cfg):
                script,paths=original(cfg)
                require(script.count('18083')==2,'Frozen launcher changed')
                return script.replace('18083',str(PORTS[arm])),paths
            rows=vr.read(root/PARENT/'inputs.json')
            summary,blind={},{}
            if arm!='pure':
                initialize(root,arm)
                store=worker_state[3]
            with patch.object(host,'PORT',PORTS[arm]),patch.object(host,'service_script',launch),node.service(
                    folder,GPUS[arm],armout/'STOP',stamp),ProcessPoolExecutor(max_workers=3,
                    mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(root,arm)) as pool:
                for row,result in zip(rows,pool.map(evaluate,rows)):
                    k=row['id']
                    summary[k]=result
                    if arm!='pure':
                        ids={i for v in result.values() for i in v['ids']}
                        blind[k]=[store.view(i) for i in sorted(ids,key=lambda i:vr.digest([k,'blind',i]))]
                    vr.atomic(folder/'progress.json',dict(completed=len(summary),total=len(rows),unix=time.time()))
            if arm!='pure':
                final.immutable(armout/'blind.json',blind)
            final.immutable(armout/'summary.json',summary)
        vr.atomic(folder/'exit.json',dict(code=0))
    except Exception as exc:
        vr.atomic(folder/'exit.json',dict(code=1,error=str(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','confirm'))
    parser.add_argument('arm',choices=tuple(GPUS),nargs='?')
    parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--root',type=Path,default=pilot.ROOT)
    args=parser.parse_args()
    if args.action=='prepare':
        prepare(args.root)
    elif args.arm is None:
        parser.error('An arm is required for this action')
    elif args.action=='confirm':
        out=args.root/RESULT
        node.confirm(out/args.arm/'attempts'/str(args.attempt),run_signature(out,args.arm))
    else:
        run(args.root,args.arm,args.attempt)
