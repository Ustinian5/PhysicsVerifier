"""Extract fresh experiences from the corrected, isolated 2000-question cohort.

Reuse the original per-answer audit+rule prompt and one structural repair.
No old corpus, reference labels, embedding or separate checker is invoked.
This stage exports raw experiences and deduplicated candidates, not a tree.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import threading
import time
import httpx

from scripts import run_corrected_collection as c
from scripts import run_experience_first_120b as old
from scripts.run_qwen30b_step_checks import validate_effective_input

read, write, sha, digest, require = c.read, c.write, c.sha, c.digest, c.require


def build_rows(source):
    plan = c.verify(source)
    tasks = read(source/'tasks.json')
    require(read(source/'execution_status.json')['state'] == 'generation_finished', 'Generation not finished')
    rows, rejected, evidence = [], [], {}
    for task in tasks:
        if task['role'] != 'build':
            continue
        q = task['row']; path = source/'generated'/(q['id']+'.json'); value = read(path)
        require(value['signature'] == digest([q, plan['configuration']['generator_config']]), 'Generation drift')
        c.corpus.evidence_valid(source, value)
        evidence[str(path)] = sha(path)
        evidence[str(source/value['evidence']['file'])] = value['evidence']['sha256']
        if value['state'] != 'complete':
            rejected.append(dict(id=q['id'], state=value['state'], finish_reason=value['finish_reason']))
            continue
        rows.append(dict(id=q['id'], question=q['question'], prediction=value['prediction'], context='', answer=''))
    require(len(rows)+len(rejected) == 2000, 'Wrong build population')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate source')
    return rows, rejected, evidence


def prepare(out, source):
    require(not (out/'plan.json').exists(), 'Already prepared')
    rows, rejected, evidence = build_rows(source)
    skeleton = old.flow.experience_topic_skeleton(read(c.ROOT/'catalogs/rules_catalog_top_down.json'))
    cfg = dict(model=c.corpus.r.EXPECTED_MODEL, auditor_api='http://127.0.0.1:18083/v1',
        max_tokens=16384, max_request_input_bytes=400000, reasoning_effort='medium', timeout_seconds=1200,
        technical_attempts=3, retry_cooldown_seconds=60, max_rules_per_sample=2,
        structural_repair_rounds=1, fixed_date='2026-11-07', pilot_size=12,
        source_plan_sha256=sha(source/'plan.json'), source=str(source),
        inputs_sha256=digest(rows), skeleton_sha256=digest(skeleton), evidence_sha256=digest(evidence),
        sources={n:sha(c.ROOT/n) for n in sorted(set(list(old.SOURCES)+list(c.SOURCES)+
            ['scripts/run_corrected_extraction.py','scripts/run_qwen30b_step_checks.py']))},
        model_files=old.b.existing.MODEL_FILES, human_gold=False,
        policy='Frozen build order; one continuous audit+extraction; content failures retained; no resampling')
    for name, value in [('inputs',rows),('generation_failures',rejected),('topic_skeleton',skeleton),('source_evidence',evidence)]:
        write(out/(name+'.json'), value)
    write(out/'plan.json',dict(configuration=cfg,configuration_sha256=digest(cfg)))
    write(out/'execution_status.json',dict(state='prepared',complete_inputs=len(rows),generation_failures=len(rejected)))


def verify(out):
    plan=read(out/'plan.json');cfg=plan['configuration']
    require(digest(cfg)==plan['configuration_sha256'], 'Plan drift')
    require(all(sha(c.ROOT/n)==h for n,h in cfg['sources'].items()),'Code drift')
    for name,key in [('inputs','inputs_sha256'),('topic_skeleton','skeleton_sha256'),('source_evidence','evidence_sha256')]:
        require(digest(read(out/(name+'.json')))==cfg[key], 'Input drift: '+name)
    require(sha(Path(cfg['source'])/'plan.json')==cfg['source_plan_sha256'], 'Source plan drift')
    return plan


def confirm(out):
    p=verify(out);ready=read(out/'disconnect_ready.json')
    require(c.corpus.runtime.process_identity(ready['pid'])==ready['process_start'] and
        time.time()-ready['unix']>=5 and ready['configuration_sha256']==p['configuration_sha256'], 'Fresh reconnect required')
    write(out/'disconnect_acceptance.json',dict(ready,passed=True,confirmed_unix=time.time()))


def call(out, journal, tokenizer, stage, key, payload):
    cfg=journal.config
    messenger=old.Journal.__new__(old.Journal)
    messages=messenger.messages(stage,payload)
    body=c.corpus.r.request_body(cfg['model'],messages,cfg['max_tokens'],backend='vllm')
    body.update(stream=False,return_token_ids=True,reasoning_effort='medium');body.pop('stream_options',None)
    require(len(json.dumps(body,ensure_ascii=False).encode())<=cfg['max_request_input_bytes'],'Request byte budget exceeded')
    raw,evidence=journal.call(stage,key,body)
    effective=validate_effective_input(raw,tokenizer,cfg['fixed_date'])
    text=c.corpus.harmony.decode_response(raw,model=cfg['model'],max_tokens=cfg['max_tokens'],native_tools=None)['content'].strip()
    if text.startswith('```json\n') and text.endswith('```'):text=text[8:-3].strip()
    try:
        escaped,controls=old.escape_json_controls(text)
        value,_,positions=old.b.parse_json_escapes(escaped,repair=True)
        require(isinstance(value,dict),'Expected JSON object')
    except (ValueError,TypeError):
        value={'_unparsed_response_text':text};controls=[];positions=[]
    write(out/'decoded'/stage/(digest(key)+'.json'),dict(value=value,evidence=evidence,
        effective_input=effective,literal_control_positions=controls,literal_backslash_positions=positions))
    return value,evidence


def run(out):
    require(Path('/proc').exists(),'Server only')
    with c.corpus.r.run_lock(out/'active.lock'),c.corpus.r.run_lock(out.parent/'organization_service_18083.lock'):
        plan=verify(out);cfg=plan['configuration'];code=1
        (out/'runner.pid').write_text(str(os.getpid()))
        try:
            c.corpus.runtime.wait_detached(out,plan['configuration_sha256'])
            for n,h in cfg['model_files'].items():require(sha(Path(cfg['model'])/n)==h,'Model configuration drift')
            for n,h in read(out/'source_evidence.json').items():require(sha(Path(n))==h,'Source evidence changed')
            with httpx.Client(trust_env=False,timeout=20) as client:
                response=client.get(cfg['auditor_api']+'/models');response.raise_for_status()
                require(any(x['id']==cfg['model'] and x.get('max_model_len',0)>=131072 for x in response.json()['data']),
                    'Unexpected model/context service')
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
            journal=c.corpus.Journal(out,cfg,threading.Event());taxonomy={'domains':[]}
            for domain in read(out/'topic_skeleton.json')['domains']:
                write(out/'execution_status.json',dict(state='taxonomy',domain=domain['id'],updated_unix=time.time()))
                value,_=call(out,journal,tokenizer,'taxonomy',domain['id'],{'domains':[domain]})
                old.extraction._experience_topics(value)
                require(old.flow.experience_topic_skeleton(value)=={'domains':[domain]},'Ontology changed')
                taxonomy['domains'].extend(value['domains'])
            write(out/'taxonomy.json',taxonomy)
            rows=read(out/'inputs.json');valid=[];invalid=[];counts=Counter()
            for i,row in enumerate(rows):
                require(not(out/'STOP').exists(),'STOP requested')
                path=out/'experiences'/(row['id']+'.json')
                payload=dict(sample=row,taxonomy=taxonomy,max_rules_per_sample=cfg['max_rules_per_sample'])
                signature=digest([payload,cfg])
                if path.exists():
                    saved=read(path);require(saved['signature']==signature,'Saved experience drift')
                    for ev in saved['evidence']:require(sha(out/ev['file'])==ev['sha256'],'Response drift')
                else:
                    draft,ev=call(out,journal,tokenizer,'extract_experience',row['id'],payload);evidence=[ev]
                    try:value=old.extraction.normalize_120b_experience(draft,**payload);error=None
                    except ValueError as exc:
                        draft,ev=call(out,journal,tokenizer,'repair_experience',row['id'],dict(original=payload,draft=draft,validation_error=str(exc)));evidence.append(ev)
                        try:value=old.extraction.normalize_120b_experience(draft,**payload);error=None
                        except ValueError as last:value=None;error=str(last)
                    saved=dict(id=row['id'],signature=signature,value=value,error=error,evidence=evidence,human_gold=False)
                    write(path,saved)
                if saved['value'] is None:invalid.append(dict(id=row['id'],error=saved['error']));counts['invalid_output']+=1
                else:valid.append(saved['value']);counts[saved['value']['semantic_audit']['status']]+=1
                write(out/'execution_status.json',dict(state='extracting',processed=i+1,total=len(rows),counts=dict(counts),updated_unix=time.time()))
                if i+1==cfg['pilot_size']:
                    write(out/'pilot_acceptance.json',dict(passed=bool(valid),scope='engineering_only_not_physical_quality',
                        configuration_sha256=plan['configuration_sha256'],counts=dict(counts),
                        files={str((out/'experiences'/(x['id']+'.json')).relative_to(out)):sha(out/'experiences'/(x['id']+'.json')) for x in rows[:i+1]}))
                    require(bool(valid),'All pilot outputs failed structure validation; inspect before continuing')
            write(out/'semantic_experience.json',valid);write(out/'invalid_outputs.json',invalid)
            distilled=old.extraction._build_distilled_library(valid,min_count=1)
            write(out/'distilled.json',distilled)
            write(out/'execution_status.json',dict(state='extraction_complete',processed=len(rows),counts=dict(counts),
                rules=len(distilled['rules']),tree_complete=False,skills_complete=False,updated_unix=time.time()))
            code=0
        except BaseException as exc:
            write(out/'execution_status.json',dict(state='stopped',error=str(exc),updated_unix=time.time()));raise
        finally:
            (out/'runner.exit_code').write_text(str(code));(out/'runner.pid').unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','verify','run','confirm-detached'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path)
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='prepare':prepare(out,a.source.resolve())
    elif a.mode=='verify':verify(out)
    elif a.mode=='confirm-detached':confirm(out)
    else:run(out)


if __name__=='__main__':main()
