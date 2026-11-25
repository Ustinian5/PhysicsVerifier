"""Paired skills confirmation outside the construction, pilot and formal question families."""
import argparse
from functools import partial
from rule_framework import verifier_english as en
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
from pathlib import Path
from unittest.mock import patch

import httpx
from rule_framework.organization_followup import FamilyScreen
from scripts import pilot_retrieval_fusion as pilot
from scripts import replay_retrieval_rank_role as roles

host, node, vr, require = pilot.host, pilot.node, pilot.vr, pilot.require
final = host.final
METHODS = ('bm25','relation','rrf')
REL = Path('retrieval_fusion_dev/expanded')
PARENT = Path('aligned_old99/final_revision_balanced')


def prepare(root):
    out = root/REL
    def read(name):
        return vr.read(root/name)
    plan = read(PARENT/'plan.json')['configuration']
    candidates = read(PARENT/'candidates.json')
    release = read(PARENT/'bench_release.json')
    require(vr.sha(root/PARENT/'candidates.json')==plan['candidates_sha256']
            and vr.sha(root/PARENT/'dispositions.json')==release['dispositions_sha256'],'Candidate or label drift')
    gold = read(PARENT/'dispositions.json')
    by_id = {x['id']:x for x in candidates}
    old = read(REL/'old600_ids.json')
    require(len(set(old['ids']))==600 and old['sha256'] in plan['dependencies'].values(),'Unbound older benchmark')
    split = read('corrected/source_screen/split.json')
    fam = {k:g['id'] for g in read('corrected/source_screen/family_components.json')['components'] for k in g['members']}
    blocked = set(old['ids'])
    protected = []
    for name in (PARENT/'inputs.json','aligned_old99/final_revision/inputs.json','full/bench_questions.json',
                 'aligned_old99/final_revision/pilot_inputs.json','questions.json'):
        rows = read(name)
        blocked.update(r['id'] for r in rows)
        protected.extend(rows)
    protected.extend(by_id[k] for k in old['ids'])
    protected.extend(dict(r,question=r.get('context','')+'\n'+r['question'])
                     for r in read('hipho_overlap_audit.json')['question_references'])
    protected.extend(split['build'])
    blocked.update(r['id'] for r in split['build'])
    occupied = {fam[k] for k in blocked if k in fam}
    eligible = [r for r in candidates if gold[r['id']]['state']=='eligible' and fam[r['id']] not in occupied]
    screen = FamilyScreen(eligible)
    ids = {r['id'] for r in eligible}
    exclude = {}
    library = root/'shared_experience/n2000/materials'
    for name in ('cases','skills'):
        protected.extend(dict(id=d['id'],question='',prediction=d['text']) for d in vr.read(library/(name+'.json')))
    for r in protected:
        for h in screen.matches(r):
            k = h['heldout_id']
            if k in ids: exclude.setdefault(k,[]).append([r['id'],h['reason']])
    # Screen candidate visible text in the reverse direction as well.
    held = {r['id']:r for r in protected if r['question'].strip()}
    reverse = FamilyScreen(list(held.values()))
    for r in eligible:
        hits = reverse.matches(r)
        if hits: exclude.setdefault(r['id'],[]).extend([h['heldout_id'],h['reason']] for h in hits)
    selected = []
    for r in sorted(eligible,key=lambda x: x['id']):
        k = r['id']
        if k in exclude or fam[k] in occupied: continue
        occupied.add(fam[k])
        selected.append(final.p.public(r))
    require(selected and all(gold[r['id']]['verdict']=='correct' for r in selected),'Unexpected cohort class')
    final.immutable(out/'inputs.json',selected)
    final.immutable(out/'design.json',dict(input_sha256=vr.sha(out/'inputs.json'),family_candidates=len(eligible),
        exclusion=exclude,classes={'correct':len(selected)},dispositions_sha256=release['dispositions_sha256'],
        library=pilot.SKILL_SHA,methods=METHODS,rank_prompt=pilot.RANK,rewrite=en.REWRITE,role=roles.ROLE,
        prompt_profile=en.PROFILE, prompt_sha256=vr.sha(en.__file__),
        cohort='All eligible reserve families outside current/retired benches and construction; one per family.',
        primary='Useful whole skills in six available slots per question; paired two-sided exact sign test.',
        test_order='Relation vs BM25 first, relation vs RRF only if the first comparison passes at 0.05.',
        secondary='False positives, abstentions and failures on the frozen correct answers; no F1 or sensitivity.',
        relevance='Grade pooled full texts for a correct relation directly applicable to a required step. '
                  'Generic hints, incompatible assumptions and contradictory core instructions score zero.',
        old600_sha256=old['sha256'],human_gold=False))
    print('screened',len(eligible),'isolated',len(selected),flush=True)


def query(value, *, english=False):
    require(isinstance(value,dict) and set(value)=={'query'} and isinstance(value['query'],str)
            and value['query'].strip(),'Invalid relation query')
    if english:
        en.require_english_query(value['query'])
    return value


def rank(value,n):
    ids = value.get('ids') if isinstance(value,dict) else None
    require(isinstance(ids,list) and len(ids)<=6 and all(type(i) is int and 1<=i<=n for i in ids)
            and len(ids)==len(set(ids)),'Invalid candidate selection')
    return value


def evaluate(row, out, plan, memory, tokenizer, client):
    folder = out/'rows'/row['id']
    limits = plan['configuration']['limits']
    english = en.active(plan['configuration'])
    def count(messages):
        return len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,reasoning_effort='medium'))
    def call(key,messages,maximum,validator,history,started=None):
        used = sum(a['usage']['completion_tokens'] for a in history)
        available = limits['wall_seconds_per_sample']-sum(a['wall_seconds'] for a in history)
        remaining = min(maximum,limits['output_tokens_per_sample']-used)
        require(remaining>0 and available>0 and len(history)<limits['max_requests_per_sample']
                and sum(a['usage']['prompt_tokens'] for a in history)+count(messages)<=limits['input_tokens_per_sample'],
                'Development budget exhausted')
        return final.request_once(out,folder/key,messages,remaining,validator,plan,tokenizer,client,available=available,started=started)
    results = {}
    first_messages = final.p.query_messages(row, english=english)
    prep = call('prepare',first_messages,pilot.flat.QUERY_MAX_TOKENS,partial(pilot.flat.validate_queries, english=english),[])
    if prep['status']!='complete':
        return {m:dict(status='prepare_failure',ids=[]) for m in METHODS}
    queries = prep['value']
    prefix = first_messages+[dict(role='assistant',content=json.dumps(queries,ensure_ascii=False))]
    rewrite_messages = pilot.rewrite_messages(row, prefix, english=english)
    rewrite = call('rewrite',rewrite_messages,4096,partial(query, english=english),[prep])
    def msgs(ids):
        return final.p.continuation_messages(row,queries,[memory.view(i) for i in ids],arm='flat_skills', english=english)
    empty_size = count(msgs([]))
    for m in METHODS:
        started = time.monotonic()
        first = pilot.top24(memory,'\n'.join(queries['queries']))
        preview = 0
        history = [prep]
        if m=='bm25':
            indices = first[:6]
        else:
            if rewrite['status']!='complete':
                results[m]=dict(ids=[],status='rewrite_failure')
                continue
            history.append(rewrite)
            second = pilot.top24(memory,rewrite['value']['query'])
            pool = second if m=='relation' else pilot.merge(first,second)[1]
            rank_history = [dict(role='system',content=roles.ROLE),*rewrite_messages[1:],
                            dict(role='assistant',content=json.dumps(rewrite['value'],ensure_ascii=False))]
            cards,empty = roles.single_pool_request([*rank_history,{}],pool,row['id'],memory)
            preview = count(cards)-count(empty)
            require(0<=preview<=8192,'Candidate previews exceed budget')
            ranked = call(m+'/rank',cards,4096,lambda v:rank(v,len(pool)),history,started)
            history.append(ranked)
            started = time.monotonic()
            if ranked['status']!='complete':
                results[m]=dict(ids=[],status='rank_failure')
                continue
            indices = [pool[i-1] for i in ranked['value']['ids']]
        ids = []
        for k in indices:
            size = count(msgs(ids+[k]))
            if size-empty_size+preview<=8192 and size<=limits['context_window']-limits['output_tokens_per_sample']:
                ids.append(k)
        decision = call(m+'/check',msgs(ids),limits['max_tokens_per_response'],final.p.original.validate_verdict,history,started)
        results[m]=dict(ids=ids,status=decision['status'],verdict=(decision['value'] or {}).get('verdict'),
                        preview_tokens=preview,full_tokens=count(msgs(ids))-empty_size,
                        extra_model_seconds=sum(a['elapsed_seconds'] for a in history[1:]))
    return results


def signature(out):
    return vr.digest([vr.sha(__file__),vr.sha(out/'design.json'),vr.sha(out/'inputs.json')])


def setup_worker(source, out, plan):
    from transformers import AutoTokenizer
    global worker_args
    tokenizer = AutoTokenizer.from_pretrained(host.HOT/'models/gpt-oss-120b',local_files_only=True)
    worker_args = (out,plan,pilot.flat.FlatMemory(vr.read(source)),tokenizer,httpx.Client(trust_env=False))


def evaluate_worker(row):
    results = evaluate(row,*worker_args)
    final.immutable(worker_args[0]/'rows'/row['id']/'delivery.json', results)
    return results


def run(out, attempt):
    node.node()
    folder = out/'attempts'/str(attempt)
    folder.mkdir(exist_ok=False,parents=True)
    try:
        with host.final.base.r.run_lock(out/'active.lock'):
            require(not (out/'summary.json').exists(),'Already completed')
            stamp = signature(out)
            host.final.base.runtime.wait_detached(folder,stamp)
            config = dict(vr.read(host.OUT/'plan.json')['configuration'],api_base=f'http://127.0.0.1:{pilot.PORT}/v1')
            design = vr.read(out/'design.json')
            if en.active(design):
                require(design['rewrite'] == en.REWRITE and design['prompt_sha256'] == vr.sha(en.__file__), 'English rewrite drift')
                config['prompt_profile'] = en.PROFILE
            plan = dict(configuration=config,configuration_sha256=vr.digest([config,stamp]))
            require(design['input_sha256']==vr.sha(out/'inputs.json'),'Cohort drift')
            rows = vr.read(out/'inputs.json')
            source = pilot.ROOT/'shared_experience'/pilot.GROUP/'materials/skills.json'
            require(vr.sha(source)==design['library']==pilot.SKILL_SHA,'Skills changed')
            memory = pilot.flat.FlatMemory(vr.read(source))
            original = host.service_script
            def launch(cfg):
                script,paths = original(cfg)
                require(script.count('18083')==2,'Launcher changed')
                return script.replace('18083',str(pilot.PORT)),paths
            summary,blind = {},{}
            with patch.object(host,'PORT',pilot.PORT), patch.object(host,'service_script',launch), node.service(
                    folder,5,out/'STOP',stamp), ProcessPoolExecutor(max_workers=3,
                        mp_context=multiprocessing.get_context('spawn'),initializer=setup_worker,
                        initargs=(source,out,plan)) as pool_workers:
                for row,result in zip(rows,pool_workers.map(evaluate_worker,rows)):
                    k = row['id']
                    summary[k]=result
                    pool={i for m in METHODS for i in result[m]['ids']}
                    blind[k]=[{key:memory.docs[i][key] for key in ('id','title','text')}
                              for i in sorted(pool,key=lambda i:vr.digest([k,'blind',i]))]
                    vr.atomic(folder/'progress.json',dict(completed=len(summary),total=len(rows),time=time.time()))
            vr.atomic(out/'blind.json',blind)
            vr.atomic(out/'summary.json',summary)
        vr.atomic(folder/'exit.json',dict(code=0))
    except Exception as exc:
        vr.atomic(folder/'exit.json',dict(code=1,error=str(exc)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','confirm'))
    p.add_argument('--root',type=Path,default=pilot.ROOT)
    p.add_argument('--attempt',type=int,default=1)
    args=p.parse_args()
    out=args.root/REL
    if args.action=='prepare':
        prepare(args.root)
    elif args.action=='confirm':
        node.confirm(out/'attempts'/str(args.attempt),signature(out))
    else:
        run(out,args.attempt)
