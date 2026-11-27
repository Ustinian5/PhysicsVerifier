"""Paired CPU replay of the frozen English verifier; original collections are read-only."""
from collections import Counter
from itertools import cycle
from pathlib import Path
import json
import argparse

from rule_framework import flat_skill_rag as mem
from rule_framework import verifier_english as en
from scripts import run_structured_final as structured
from scripts import timed_retrieval_cohort as family
from scripts import run_organization_comparison as step

final, vr = structured.final, structured.vr
BASE = structured.WORK.parent
OUT = BASE/'bm25f_50'
ARMS = ('case_rag','flat_skills','rule_tree')


def fields(doc):
    title, text = doc['title'], doc['text']
    if doc['kind']=='skill':
        return [(1,title),(2,text)]
    if doc['kind']=='case':
        problem, rest = text.removeprefix(en.QUESTION).split(en.ANSWER)
        answer, audit = rest.split(en.REVIEW)
        return [(1,title),(2,problem),(1,answer),(3,audit)]
    rule = json.loads(text)
    return [(1,title+' '+json.dumps({k:v for k,v in rule.items() if k not in ('trigger','check_logic')},ensure_ascii=False)),
            (2,rule['trigger']), (2,rule['check_logic'])]


class FieldSearch:
    def __init__(self, store):
        self.store = store
        self.counts = {sid:[(w, Counter(mem.tokens(text))) for w,text in fields(doc)] for sid,doc in store.docs.items()}
        n = len(store.ids)
        lengths = [sum(sum(c.values()) for _,c in xs)/n for xs in zip(*self.counts.values())]
        self.fields = {sid:[(weight/(.25+.75*sum(counts.values())/max(avg,1)),counts)
                            for (weight,counts),avg in zip(parts,lengths)] for sid,parts in self.counts.items()}

    def search(self, query, limit=6, allowed_ids=None):
        terms = set(mem.tokens(query))
        ranked = []
        for sid in self.store.ids if allowed_ids is None else sorted(allowed_ids):
            score = 0.
            for t in terms:
                tf = sum(scale*c[t] for scale,c in self.fields[sid])
                if tf:
                    score += self.store.idf[t]*2.2*tf/(1.2+tf)
            if score>0:
                ranked.append((score,sid))
        ranked.sort(key=lambda pair:(-pair[0],pair[1]))
        return [dict(id=sid,score=round(score,6),title=self.store.docs[sid]['title'],
                     excerpt=self.store.docs[sid]['text'][:240]) for score,sid in ranked[:limit]]


def inputs():
    q = structured.OUT
    s = BASE/'step_formal'
    cfg = vr.read(q/'plan.json')['configuration']
    rows = final.eligible_inputs(vr.read(q/'inputs.json'),cfg)
    labels = vr.read(q/'labels_private.json')
    fp = family.original.RESULTS/family.FAMILIES
    assert vr.sha(fp)==family.FAMILIES_SHA
    families = {q:g['id'] for g in vr.read(fp)['components'] for q in g['members']}
    order = sorted(rows,key=lambda x:vr.digest(['BM25F-20261127',x['id']]))
    selected, used = [],set()
    for label in cycle(('incorrect','correct')):
        row=next(r for r in order if labels[r['id']]['verdict']==label and families[r['id']] not in used)
        selected.append(row)
        used.add(families[row['id']])
        if len(selected)==50: break
    eligible=vr.read(s/'plan.json')['configuration']['sample_ids']
    byid={r['id']:r for r in vr.read(s/'inputs.json')}
    return [(q,selected),(s,[byid[i] for i in eligible[:50]])]


def run():
    from transformers import AutoTokenizer
    stores=final.stores(structured.OUT)
    alternate=final.stores(structured.OUT)
    for arm in ARMS:
        store=alternate[arm].memory if arm=='rule_tree' else alternate[arm]
        store.search=FieldSearch(store).search
    tok=AutoTokenizer.from_pretrained(structured.node.host.HOT/'models/gpt-oss-120b',local_files_only=True)
    cohorts=inputs()
    final.immutable(OUT/'cohort.json',{source.name:[r['id'] for r in rows] for source,rows in cohorts})
    for source,rows in cohorts:
        is_final=source==structured.OUT
        for row in rows:
            sid=row['id']
            key=vr.digest(sid)[:24]
            for arm in ARMS:
                archive=source/'formal'/arm/key if is_final else step.path_for(source,'formal',arm,row).parent
                qdir=(source/'formal' if is_final else source)/'queries'/key
                prefix=vr.read(qdir/'result.json') if (qdir/'result.json').exists() else None
                dest=OUT/source.name/sid/arm
                if prefix is None or not (archive/'reference_packet.json').exists():
                    result=vr.read(archive/'result.json')
                    assert result.get('reason')=='Shared PREPARE failed' or result.get('failure_stage')=='prepare'
                    final.immutable(dest/'receipt.json',dict(retained_failure=vr.sha(archive/'result.json')))
                    continue
                opts={'reasoning_effort':'medium'} if is_final else {}
                def count(m):
                    return len(tok.apply_chat_template(m,tokenize=True,add_generation_prompt=True,**opts))
                make=final.p.reference_packet if is_final else mem.reference_packet
                queries=prefix['value' if is_final else 'queries']
                old=make(row,arm,queries,stores[arm],count,english=True)
                archived=vr.read(archive/'reference_packet.json')
                archived=archived if is_final else archived['packet']
                actual=vr.read(archive/('check/request.json' if is_final else 'turn_02/request.json'))['request']
                assert old['messages']==archived['messages']==actual['messages'], ('Baseline mismatch',source.name,sid,arm)
                new=make(row,arm,queries,alternate[arm],count,english=True)
                final.immutable(dest/'packet.json',new)
                final.immutable(dest/'receipt.json',dict(old=old['retrieval']['provided_complete_ids'],
                    new=new['retrieval']['provided_complete_ids'],changed=new['messages']!=old['messages'],
                    original_result=str(archive/'result.json'),source_sha256=vr.sha(archive/'reference_packet.json'),
                    packet_sha256=vr.sha(dest/'packet.json')))
    final.immutable(OUT/'cpu_complete.json',dict(pairs=300,script_sha256=vr.sha(__file__)))

if __name__=='__main__':
    from scripts import run_continuous_context_v2 as runtime
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','confirm']); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.action=='confirm':
        from unittest.mock import patch
        with patch.object(runtime,'validate_plan',return_value={'configuration_sha256':vr.sha(__file__)}):
            runtime.confirm_detached(OUT)
    else:
        with final.base.r.run_lock(OUT/'cpu.lock'):
            runtime.wait_detached(OUT,vr.sha(__file__))
            run()
