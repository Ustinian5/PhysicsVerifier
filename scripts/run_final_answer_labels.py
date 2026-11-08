"""Reference-assisted 120B labels for the frozen held-out 30B candidate pool.

Does not run a tested verifier or transfer labels into construction. Stops at a
small engineering pilot before the independently authorized full labeling run.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import threading
import time

import httpx
from scripts import run_corrected_collection as collection
from scripts.run_qwen30b_step_checks import validate_effective_input
from rule_framework import final_answer_protocol as protocol

read,write,require,sha,digest=collection.read,collection.write,collection.require,collection.sha,collection.digest
corpus=collection.corpus;ROOT=collection.ROOT
SOURCES=tuple(sorted(set(collection.SOURCES+('scripts/run_final_answer_labels.py',
    'rule_framework/final_answer_protocol.py','scripts/run_qwen30b_step_checks.py',
    'scripts/resume_continuous_context_v2.py','scripts/gpt_oss_harmony_transport.py'))))
REFERENCE_REVIEWS={
 'fc543632eb5751b1cb737bc3f56d7b95413fc983a4ffbe1a89674e5df5696e13':'First original formula has balanced grouping; second is malformed LaTex. Retain both originals, select first without editing.',
 '4d04252a8c7e9ce925490079aa1879a0893ba5353592b8dd504b758ff221b038':'Both state 110 J/K; only LaTex unit formatting differs.',
 'd680b2cacf0d0dc125c700d2170fceb3be9223f1279084852cebceb94f366198':'Identical formula with or without boxed wrapper.',
 'e0df2b4933171e6871e23fe0bb556991959af10c5917d0d0060cf0b568b6e9c3':'Identical omega relation; math delimiter versus boxed wrapper.',
 'aed201c74b94c50976d6f352a7d9c1954de2a48b2855b3138b48bf5b3342d174':'Identical hyperbolic tangent formula with or without boxed wrapper.',
}
CONTROLS=[
 dict(id='control_equivalent_units',question='A body travels 100 metres in 10 seconds at constant speed. Find its speed in metres per second.',prediction='The speed is 36 km/h, equivalent to 10 m/s.',reference=['10 m/s'],expected='correct'),
 dict(id='control_wrong_final',question='A body travels 100 metres in 10 seconds at constant speed. Find its speed.',prediction='The speed is 20 m/s.',reference=['10 m/s'],expected='incorrect'),
 dict(id='control_intermediate_error',question='A body travels 100 metres in 10 seconds at constant speed. Find its speed.',prediction='First I compute 100/10=20. My final answer is 10 m/s.',reference=['10 m/s'],expected='correct'),
 dict(id='control_missing_part',question='A body travels 100 metres in 10 seconds at constant speed. Give (a) its speed in m/s and (b) its distance in the next 5 seconds.',prediction='(a) The speed is 10 m/s. I do not give an answer to (b).',reference=['(a) 10 m/s; (b) 50 m'],expected='incorrect'),
]


def prepare(out,source,screen):
    require(not(out/'plan.json').exists(),'Already prepared')
    gen=collection.verify(source);split=read(screen/'split.json')
    refs={r['id']:r['variants'] for r in read(screen/'references_private.json')['references']}
    bound=[]
    for row in split['reserve']:
        variants=refs[row['id']]
        require(len(variants)==1 or row['id'] in REFERENCE_REVIEWS,'Unreviewed reference variants')
        chosen=variants[0]['value']
        require(isinstance(chosen,list) and chosen and all(isinstance(x,str) for x in chosen),'Unsupported reference format')
        bound.append(dict(id=row['id'],variants=variants,selected_variant=0,reference=chosen,
            reference_complete=all(x.strip() for x in chosen),
            resolution=REFERENCE_REVIEWS.get(row['id'],'Single original reference variant'),human_gold=False))
    c=dict(protocol=protocol.VERSION,source=os.path.relpath(source,out),source_plan_sha256=sha(source/'plan.json'),
        questions_sha256=digest(split['reserve']),references_sha256=digest(bound),
        source_reference_sha256=sha(screen/'references_private.json'),controls_sha256=digest(CONTROLS),
        code_hashes={n:sha(ROOT/n) for n in SOURCES},generator_config=gen['configuration']['generator_config'],
        auditor=corpus.r.EXPECTED_MODEL,auditor_api='http://127.0.0.1:18083/v1',generator_api='http://127.0.0.1:18084/v1',
        technical_attempts=3,retry_cooldown_seconds=60,timeout_seconds=1200,max_tokens=16384,
        reasoning_effort='medium',fixed_harmony_date='2026-11-07',pilot_size=6,human_gold=False)
    write(out/'questions.json',split['reserve']);write(out/'references_private.json',bound)
    write(out/'plan.json',dict(configuration=c,configuration_sha256=digest(c)))
    write(out/'execution_status.json',dict(state='prepared_no_calls',candidates=len(bound),target=600))


def verify(out):
    p=read(out/'plan.json');c=p['configuration']
    require(digest(c)==p['configuration_sha256'] and c['protocol']==protocol.VERSION,'Plan drift')
    require(all(sha(ROOT/n)==h for n,h in c['code_hashes'].items()),'Code drift')
    require(digest(read(out/'questions.json'))==c['questions_sha256'] and
            digest(read(out/'references_private.json'))==c['references_sha256'] and digest(CONTROLS)==c['controls_sha256'],'Input drift')
    require(sha(out/c['source']/'plan.json')==c['source_plan_sha256'],'Generation plan drift')
    return p


def confirm(out):
    p=verify(out);ready=read(out/'disconnect_ready.json')
    require(corpus.runtime.process_identity(ready['pid'])==ready['process_start'] and
            ready['configuration_sha256']==p['configuration_sha256'] and time.time()-ready['unix']>=5,'Fresh reconnect required')
    write(out/'disconnect_acceptance.json',dict(ready,passed=True,confirmed_unix=time.time()))


def judge(out,journal,row,reference,tokenizer,stage='labels'):
    c=journal.config;signature=digest([row,reference,c]);path=out/stage/(row['id']+'.json')
    if path.exists():
        value=read(path);require(value['signature']==signature,'Existing label binding drift');corpus.evidence_valid(out,value);return value
    body=corpus.r.request_body(c['auditor'],protocol.label_messages(row,reference),c['max_tokens'],backend='vllm')
    body.update(stream=False,return_token_ids=True,reasoning_effort='medium');body.pop('stream_options',None)
    raw,evidence=journal.call(stage,row['id'],body)
    effective=validate_effective_input(raw,tokenizer,c['fixed_harmony_date'])
    try:
        text=corpus.harmony.decode_response(raw,model=c['auditor'],max_tokens=c['max_tokens'],native_tools=None)['content'].strip()
        if text.startswith('```json\n') and text.endswith('```'):text=text[8:-3].strip()
        verdict=protocol.validate_verdict(json.loads(text));state='complete';error=None
    except (ValueError,KeyError,TypeError) as exc:verdict=None;state='content_failure';error=str(exc)
    value=dict(id=row['id'],signature=signature,state=state,judgment=verdict,error=error,
        evidence=evidence,effective_input=effective,usage=raw.get('usage'),human_gold=False)
    write(path,value);return value


def run(out,full):
    require(Path('/proc').exists(),'Server only')
    with corpus.r.run_lock(out/'active.lock'),corpus.r.run_lock(out.parent/'organization_service_18083.lock'):
        p=verify(out);c=p['configuration'];code=1
        if full:
            accepted=read(out/'pilot_acceptance.json')
            require(accepted['passed'] and accepted['configuration_sha256']==p['configuration_sha256'] and
                all(sha(out/n)==h for n,h in accepted['files'].items()),'Pilot not accepted')
        (out/'runner.pid').write_text(str(os.getpid()))
        try:
            require(not(out/'STOP').exists(),'STOP requested');corpus.runtime.wait_detached(out,p['configuration_sha256'])
            with httpx.Client(trust_env=False,timeout=10) as client:
                response=client.get(c['auditor_api']+'/models');response.raise_for_status()
                require(any(x['id']==c['auditor'] and x.get('max_model_len',0)>=131072 for x in response.json()['data']),'120B identity/context mismatch')
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(c['auditor'],local_files_only=True)
            journal=corpus.Journal(out,c,threading.Event())
            for row in CONTROLS:
                value=judge(out,journal,row,row['reference'],tokenizer,'controls')
                require(value['state']=='complete' and value['judgment']['verdict']==row['expected'],'Final-answer engineering control failed')
            questions=read(out/'questions.json');refs={r['id']:r for r in read(out/'references_private.json')}
            counts=Counter();source=out/c['source']
            for q in questions if full else questions[:c['pilot_size']]:
                if not refs[q['id']]['reference_complete']:
                    write(out/'reference_holds'/(q['id']+'.json'),dict(id=q['id'],state='reference_incomplete',
                        reason='Original reference contains blank entries; do not infer missing answers or issue a gold label.',
                        reference_sha256=digest(refs[q['id']])))
                    counts['reference_incomplete']+=1;continue
                generated=source/'generated'/(q['id']+'.json')
                while not generated.exists():
                    require(not(out/'STOP').exists(),'STOP requested')
                    state=read(source/'execution_status.json')['state']
                    require(state not in {'stopped','generation_finished'},'Generator ended before required candidate; no replacement')
                    write(out/'execution_status.json',dict(state='waiting_answer',id=q['id'],counts=dict(counts),updated_unix=time.time()))
                    time.sleep(5)
                saved=read(generated);corpus.evidence_valid(source,saved)
                require(saved['signature']==digest([q,c['generator_config']]) and saved['model']==c['generator_config']['generator'],'Generated answer drift')
                require(not(out/'STOP').exists(),'STOP requested')
                if saved['state']!='complete':
                    write(out/'generation_failures'/(q['id']+'.json'),dict(id=q['id'],state=saved['state'],generation_sha256=sha(generated)))
                    counts['generation_failure']+=1;continue
                value=judge(out,journal,dict(q,prediction=saved['prediction']),refs[q['id']]['reference'],tokenizer)
                counts[value['judgment']['verdict'] if value['state']=='complete' else 'content_failure']+=1
                write(out/'execution_status.json',dict(state='labeling',counts=dict(counts),updated_unix=time.time()))
            write(out/'execution_status.json',dict(state='labeling_complete' if full else 'pilot_pending_review',counts=dict(counts),
                candidates=len(questions),target=600,benchmark_frozen=False,updated_unix=time.time()))
            code=0
        except BaseException as exc:
            write(out/'execution_status.json',dict(state='stopped',error=str(exc)[:1000],updated_unix=time.time()));raise
        finally:(out/'runner.exit_code').write_text(str(code));(out/'runner.pid').unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','verify','run','confirm-detached'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--screen',type=Path);p.add_argument('--full',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='prepare':prepare(out,a.source.resolve(),a.screen.resolve())
    elif a.mode=='verify':verify(out);print('verified')
    elif a.mode=='confirm-detached':confirm(out)
    else:run(out,a.full)


if __name__=='__main__':main()
