"""Detached cumulative-5000 collection using frozen old generators/builders.

The running v5 code is not edited. Existing answers and exact v5 requests are
reused, including empty and invalid content. No historical audit is rerun.
The final catalog is explicitly an unverified candidate collection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import httpx

from scripts import run_experience_first_120b as e
from scripts import rebuild_qwen30b_corpus as g
from scripts import prepare_qwen30b_expansion as prep
from rule_framework import released_corpus

b, p, r = e.b, e.p, e.r
VERSION = 'qwen30b_cumulative5000_experience_v1'
SOURCES = sorted(set(e.SOURCES + g.SOURCES + ['scripts/prepare_qwen30b_expansion.py',
                                 'scripts/run_qwen30b_expansion.py',
                                 'scripts/start_qwen30b_expansion_service.sh']))


def hashes():
    return {name:r.sha256_file(b.ROOT/name) for name in SOURCES}


def initialize(output):
    selection = b.read(output/'selection.json')
    base = Path(selection['base_corpus']); parent = Path(selection['parent_build'])
    gc = b.read(base/'plan.json')['configuration']; pc = b.read(parent/'plan.json')['configuration']
    generator = {k:gc[k] for k in ('generator','generator_api','auditor','auditor_api','temperature',
        'top_p','top_k','max_tokens','seed','samples_per_question','generator_workers','reasoning_effort',
        'domain_batch_size','domain_max_tokens','technical_attempts','timeout_seconds','retry_cooldown_seconds')}
    generator['selection_sha256'] = p.digest(selection)
    construction = {k:pc[k] for k in ('model','model_files','api_base','reasoning_effort','max_tokens',
        'max_request_input_bytes','attempts_per_job','batch_size','max_rules_per_sample','json_policy')}
    construction.update(distill_max_requests=0,max_requests=3*(4*5000+36),protocol=VERSION,
                        selection_sha256=p.digest(selection))
    c = {'protocol':VERSION,'selection_sha256':p.digest(selection),'source_hashes':hashes(),
        'generator':generator,'construction':construction,'parent_configuration_sha256':p.digest(pc),
        'target_questions':5000,'new_questions_needed':selection['new_questions_needed'],
        'full_context_structure_repairs':1,'collection_only':True,'usable_for_runtime':False,
        'human_gold':False,'embedding_model':None,'external_teacher':None}
    b.write_once(output/'plan.json',{'configuration':c,'configuration_sha256':p.digest(c)})
    return verify(output)


def verify(output, replay=False):
    plan = b.read(output/'plan.json'); c = plan['configuration']; s = b.read(output/'selection.json')
    p.require(p.digest(c)==plan['configuration_sha256'] and c['protocol']==VERSION and
              c['source_hashes']==hashes() and c['target_questions']==5000, 'Frozen code/configuration changed')
    p.require(p.digest(s)==c['selection_sha256'],'Selection changed')
    for name,sha in {**s['source_files'],**s['base_bindings']}.items():
        p.require(r.sha256_file(Path(name))==sha,'Source binding changed: '+name)
    for name,key in [('reserve_questions','reserve_sha256'),('isolation','isolation_sha256'),('bench_questions','bench_sha256')]:
        p.require(p.digest(b.read(output/(name+'.json')))==s[key],name+' changed')
    heldout,cohorts=b.benchmark_questions()
    p.require(cohorts==s['benchmark_files'] and p.digest(heldout)==s['bench_sha256'],'Bench registry drift')
    p.require(c['construction']['model']==r.EXPECTED_MODEL and c['construction']['reasoning_effort']=='medium' and
              c['generator']['generator']==g.MODEL and c['generator']['samples_per_question']==1,'Model identity changed')
    if replay:
        source=next(Path(n) for n in s['source_files'] if Path(n).name=='question_pool.json')
        selected,isolation=prep.select_new(b.read(source),heldout,b.read(Path(s['base_corpus'])/'inputs.json'),
                                          limit=s['reserve_questions'],seed=s['seed'])
        p.require(p.digest(selected)==s['reserve_sha256'] and p.digest(isolation)==s['isolation_sha256'],
                  'Source isolation does not replay')
    return plan


def checked_release(root, name):
    release=b.read(root/name)
    for rel,sha in release['files'].items():
        path=Path(rel)
        p.require(not path.is_absolute() and '..' not in path.parts,'Unsafe release path')
        p.require(r.sha256_file(root/path)==sha,'Released evidence changed: '+rel)
    return release


def parent_ready(output):
    s=b.read(output/'selection.json');parent=Path(s['parent_build'])
    p.require((parent/'release_manifest.json').is_file(),'Current v5 build has not completed; no expansion inference')
    release=checked_release(parent,'release_manifest.json')
    p.require(release['protocol']==e.VERSION and release['configuration_sha256']==b.read(parent/'plan.json')['configuration_sha256']
              and release['collection_only'] is True and release['usable_for_runtime'] is False,'Unexpected parent release')
    e.verify(parent,source_check=True)
    b.write_once(output/'parent_release_binding.json',{'parent':str(parent),
        'release_sha256':r.sha256_file(parent/'release_manifest.json'),'configuration_sha256':release['configuration_sha256'],
        'counts':release['counts'],'physical_quality_passed':False})
    return parent


def model_ready(model, base, context, root=None):
    p.require(base.startswith('http://127.0.0.1:'),'Only local inference endpoints are allowed')
    with httpx.Client(trust_env=False,timeout=20) as client:
        response=client.get(base+'/models');response.raise_for_status()
    p.require(any(x['id']==model and x.get('max_model_len',0)>=context and
                  (root is None or x.get('root')==root) for x in response.json()['data']),
              'Wrong local checkpoint or context capacity')


def auditor_ready(config):
    model_ready(config['model'],config['api_base'],131072,config['model'])
    p.require({n:r.sha256_file(Path(config['model'])/n) for n in config['model_files']}==
              config['model_files'],'120B metadata changed')


def status(output,state,**kwargs):
    r.write_json(output/'execution_status.json',{'state':state,'updated_unix':time.time(),**kwargs})


def freeze_new_questions(reserve, decisions, needed):
    # Only domain outcomes are used. Generation/checking results never select rows.
    chosen=[x for x in reserve if decisions.get(x['id'],{}).get('domain')=='physics'][:needed]
    p.require(len(chosen)==needed,f'Insufficient physics questions: {len(chosen)}/{needed}; retain shortfall, no generation')
    return chosen


def validate_generation(output,row,config):
    value=b.read(output/'generated'/(row['id']+'.json'))
    p.require(value['signature']==p.digest([row,config]) and value['id']==row['id'],'Generation binding changed')
    evidence=value['evidence']; raw_path=output/evidence['file']
    p.require(raw_path.resolve().is_relative_to(output.resolve()) and r.sha256_file(raw_path)==evidence['sha256'],
              'Generation raw evidence changed')
    raw=b.read(raw_path)
    expected={'model':config['generator'],'messages':g.solver_messages(row),'temperature':config['temperature'],
        'top_p':config['top_p'],'top_k':config['top_k'],'max_tokens':config['max_tokens'],'stream':False,
        'seed':int(p.digest([config['seed'],row['id']])[:8],16)%(2**31)}
    p.require(raw['request']==expected and raw['status']=='received' and
              raw['response']['model']==config['generator']==value['model'],'Generator identity/request changed')
    choices=raw['response']['choices'];p.require(len(choices)==1,'Unexpected choices')
    choice=choices[0];text=choice.get('message',{}).get('content')
    state='complete' if choice.get('finish_reason')=='stop' and isinstance(text,str) and text.strip() else 'incomplete'
    p.require(value['state']==state and value['prediction']==(text or '') and value['finish_reason']==choice.get('finish_reason'),
              'Stored answer differs from original response')
    return value


def collect(output,c):
    path=output/'generation_release.json'
    if path.exists():
        checked_release(output,path.name);release_generator(output);return
    gc=c['generator']; s=b.read(output/'selection.json');stop=threading.Event()
    p.require(gc['auditor']==c['construction']['model'] and
              gc['auditor_api']==c['construction']['api_base'],'Domain/build model differs')
    auditor_ready(c['construction'])
    journal=g.Journal(output,gc,stop); controls=g.Journal(output/'controls',gc,stop)
    control_rows=[{'id':'control_physics','question':'A 2 kg body feels a net 6 N force. Find its acceleration.'},
                  {'id':'control_math','question':'Find all positive integers n such that n squared plus one is prime.'}]
    controlled=g.screen_domains(controls,control_rows,'domain_control')
    p.require({x['id']:x['domain'] for x in controlled['items']}==
              {'control_physics':'physics','control_math':'non_physics'},'Domain engineering control failed')
    reserve=b.read(output/'reserve_questions.json');decisions={};needed=s['new_questions_needed']
    for start in range(0,len(reserve),gc['domain_batch_size']):
        part=reserve[start:start+gc['domain_batch_size']]
        result=g.screen_domains(journal,part,'batch_'+str(start));decisions.update({x['id']:x for x in result['items']})
        status(output,'domain_screening',screened=len(decisions),counts=dict(Counter(x['domain'] for x in decisions.values())))
        if sum(x['domain']=='physics' for x in decisions.values())>=needed:break
    chosen=freeze_new_questions(reserve,decisions,needed)
    b.write_once(output/'domain_decisions.json',decisions);b.write_once(output/'new_questions.json',chosen)
    b.write_once(output/'cohort_freeze.json',{'selection_sha256':p.digest(s),'new_questions_sha256':p.digest(chosen),
        'target_questions':5000,'existing_complete':s['baseline_complete_questions'],'new_attempts':len(chosen),
        'no_replacement_after_generation':True,'surplus_domain_passes_not_generated':sum(x['domain']=='physics' for x in decisions.values())-needed})
    missing=[x for x in chosen if not (output/'generated'/(x['id']+'.json')).exists()]
    if missing:
        model_ready(gc['generator'],gc['generator_api'],32768,'/slow_share2/huhengyuan/models/'+g.MODEL)
        checkpoint=b.read(output/'service/checkpoint_manifest.json')
        p.require(checkpoint['complete'] and checkpoint['destination']=='/slow_share2/huhengyuan/models/'+g.MODEL,
                  '30B checkpoint not verified by personal launcher')
        for name,entry in checkpoint['files'].items():
            fp=Path(checkpoint['destination'])/name
            p.require(fp.stat().st_size==entry['bytes'] and fp.stat().st_mtime<=checkpoint['completed_unix'],'30B checkpoint changed')
            if not name.endswith('.safetensors'):p.require(r.sha256_file(fp)==entry['sha256'],'30B metadata changed')
        with ThreadPoolExecutor(max_workers=gc['generator_workers']) as pool:
            pending=[];cursor=iter(missing)
            for _ in range(gc['generator_workers']):
                row=next(cursor,None)
                if row:pending.append(pool.submit(g.generate,journal,row))
            completed=len(chosen)-len(missing)
            try:
                while pending:
                    pending.pop(0).result();completed+=1
                    status(output,'generating',completed=completed,total=len(chosen))
                    row=next(cursor,None)
                    if row:pending.append(pool.submit(g.generate,journal,row))
            except BaseException:
                stop.set();raise
    values=[validate_generation(output,row,gc) for row in chosen]
    answers=[{**row,'prediction':v['prediction'],'answer':'','context':''} for row,v in zip(chosen,values) if v['state']=='complete']
    kept,excluded=p.isolate_sources(answers,b.read(output/'bench_questions.json'))
    b.write_once(output/'new_answers.json',answers);b.write_once(output/'post_generation_exclusions.json',excluded)
    b.write_once(output/'admitted_new_answers.json',kept)
    files=['new_questions.json','cohort_freeze.json','domain_decisions.json','new_answers.json',
           'admitted_new_answers.json','post_generation_exclusions.json']
    files += [str(x.relative_to(output)) for directory in ['raw','screened','generated','controls','service']
              for x in sorted((output/directory).rglob('*.json')) if x.name not in {'execution_status.json'}]
    raw=[b.read(x) for x in [*output.glob('raw/*/*/request_*.json'),*output.glob('controls/raw/*/*/request_*.json')]]
    counts={'target_questions':5000,'new_attempted':len(chosen),'generation':dict(Counter(v['state'] for v in values)),
            'post_generation_excluded':len(excluded),'new_admitted':len(kept),'domain_counts':dict(Counter(x['domain'] for x in decisions.values())),
            'actual_requests':len(raw),'unknown_usage':sum(x.get('usage') is None for x in raw)}
    b.write_once(path,{'protocol':VERSION,'configuration_sha256':p.digest(c),
        'generator':g.MODEL,'human_gold':False,'counts':counts,
        'files':{name:r.sha256_file(output/name) for name in files},'admitted_source_sha256':p.digest(kept)})
    release_generator(output)
    status(output,'generation_completed',counts=counts)


def release_generator(output):
    """Signal only the live child of this expansion's recorded service launcher."""
    receipt=output/'service/released_after_generation.json';pid_file=output/'service/launcher.pid'
    if receipt.exists() or not pid_file.exists():return
    parent=int(pid_file.read_text());root=Path('/proc')/str(parent)
    if not root.exists():return
    args=(root/'cmdline').read_bytes().split(b'\0')
    p.require(any(x.endswith(b'/start_qwen30b_expansion_service.sh') for x in args) and
              str(output).encode() in args,'Service launcher is not this expansion; no signal sent')
    matches=[]
    for child in (root/'task'/str(parent)/'children').read_text().split():
        args=(Path('/proc')/child/'cmdline').read_bytes().split(b'\0')
        if (b'vllm.entrypoints.openai.api_server' in args and b'18084' in args and
            b'/slow_share2/huhengyuan/models/Qwen3-30B-A3B-Instruct-2507' in args and
            args[0]==b'/home/visitor/huhengyuan/envs/physicsverifier/bin/python'):
            matches.append((int(child),args))
    p.require(len(matches)==1,'Cannot identify own service child; no signal sent')
    pid,args=matches[0]
    proof={'pid':pid,'process_start':b.existing.process_identity(pid),'parent':parent,
           'command_sha256':p.digest([x.decode() for x in args]),'signal':'SIGTERM','unix':time.time()}
    os.kill(pid,signal.SIGTERM);r.write_json(receipt,proof)


def parent_value(journal,parent,stage,key,payload):
    folder=parent/('arm_'+stage)/p.digest(key)[:24];receipt=folder/'result.json';prefix=folder/'prefix_replay.json'
    if receipt.exists():
        saved=b.read(receipt);raw_path=folder/saved['request_file'];expected_sha=saved['request_sha256']
    elif prefix.exists():
        saved=b.read(prefix);raw_path=Path(saved['source']);expected_sha=saved['source_sha256']
    else:
        return False,None
    p.require(r.sha256_file(raw_path)==expected_sha,'Parent request changed')
    record=b.read(raw_path);c=journal.config
    body=r.request_body(c['model'],journal.messages(stage,payload),c['max_tokens'],backend='vllm')
    body.update(stream=False,return_token_ids=True,reasoning_effort=c['reasoning_effort']);body.pop('stream_options',None)
    p.require(record['request']==body,'Parent request differs; no regeneration or approximate replay')
    raw=record.get('response') or {}
    p.require(not record.get('error') and raw.get('model')==c['model'] and raw.get('id'),'Invalid parent identity')
    text=b.harmony.decode_response(raw,model=c['model'],max_tokens=c['max_tokens'],native_tools=None)['content'].strip()
    if text.startswith('```json\n') and text.endswith('```'):text=text[8:-3].strip()
    try:value=b.parse_json_escapes(e.escape_json_controls(text)[0],repair=True)[0]
    except json.JSONDecodeError:value={'_unparsed_response_text':text}
    p.require((value==saved['value']) if receipt.exists() else (p.digest(value)==saved['value_sha256']),
              'Parent parsed result changed')
    b.write_once(journal.output/('arm_'+stage)/p.digest(key)[:24]/'prefix_replay.json',
        {'source':str(raw_path),'source_sha256':expected_sha,'value_sha256':p.digest(value),'new_model_calls':0})
    return True,value


def build(output,c,parent):
    target=output/'build';target.mkdir(exist_ok=True)
    if (target/'release_manifest.json').exists():checked_release(target,'release_manifest.json');return
    generation=checked_release(output,'generation_release.json');s=b.read(output/'selection.json')
    p.require(generation['configuration_sha256']==p.digest(c),'Generation protocol changed')
    baseline=b.read(Path(s['base_corpus'])/'answers.json');new=b.read(output/'admitted_new_answers.json')
    p.require(p.digest(new)==generation['admitted_source_sha256'],'New answers changed')
    for row in b.read(output/'new_questions.json'):validate_generation(output,row,c['generator'])
    rows=baseline+new;p.require(len({x['id'] for x in rows})==len(rows),'Duplicate generation ID')
    rows.sort(key=lambda x:p.digest([20261024,x['id']]))
    b.write_once(target/'inputs.json',rows)
    config=c['construction'];journal=e.Journal(target,config,True)
    ancestors=[parent];pc=b.read(parent/'plan.json')['configuration']
    if pc.get('draft_prefix_parent'):ancestors.append(Path(pc['draft_prefix_parent']))
    journal.budget=b.BatchBudget(target,journal.budget.limits,prior_runs=tuple(ancestors),stop_file=output/'STOP')
    baseline_ids={x['id'] for x in baseline}
    def raw_call(stage,key,payload):
        if stage=='taxonomy' or (stage in {'extract_experience','repair_experience'} and key in baseline_ids):
            found,value=parent_value(journal,parent,stage,key,payload)
            p.require(found,'Missing completed base result; do not rerun old rows')
            return value
        while True:
            p.require(not (output/'STOP').exists(),'STOP requested')
            try:return journal(stage,key,payload)
            except Exception:
                paths=sorted((target/('arm_'+stage)/p.digest(key)[:24]).glob('request_*.json'))
                if not paths or len(paths)>=config['attempts_per_job'] or not b.retryable_transport_failure(b.read(paths[-1])):raise
                last=b.read(paths[-1]);deadline=max(b.retry_after_deadline(last),last.get('started_unix',0)+last.get('elapsed_seconds',0)+300)
                while time.time()<deadline:
                    p.require(not (output/'STOP').exists(),'STOP requested')
                    status(output,'waiting_transient',stage=stage,key=key,ready_unix=deadline)
                    time.sleep(min(10,max(.1,deadline-time.time())))
    def call(stage,key,payload):
        draft=raw_call(stage,key,payload)
        if stage=='extract_experience':
            try:e.extraction.normalize_120b_experience(draft,**payload)
            except ValueError as exc:return raw_call('repair_experience',key,{'original':payload,'draft':draft,'validation_error':str(exc)})
        return draft
    auditor_ready(config)
    built=e.flow.build_120b_experience_catalog(rows,b.read(output/'bench_questions.json'),call,
        topic_skeleton=b.read(parent/'topic_skeleton.json'),max_rules_per_sample=config['max_rules_per_sample'],
        batch_size=config['batch_size'],retain_invalid=True)
    verify(output)
    for name,value in built.items():b.write_once(target/(name+'.json'),value)
    b.write_once(target/'release_manifest.json',{'protocol':VERSION,'configuration_sha256':p.digest(c),
        'status':'unverified_candidates_collected_quality_not_accepted','collection_only':True,
        'physical_quality_passed':False,'usable_for_runtime':False,'human_gold':False,'counts':built['counts'],
        'parent_release_sha256':r.sha256_file(parent/'release_manifest.json'),
        'generation_release_sha256':r.sha256_file(output/'generation_release.json'),
        'files':{str(x.relative_to(target)):r.sha256_file(x) for x in sorted(target.rglob('*.json'))
                 if x.name not in {'execution_status.json','release_manifest.json'}}})
    status(output,'completed_candidate_collection',counts=built['counts'])


def run(output,phase):
    plan=verify(output);c=plan['configuration'];s=b.read(output/'selection.json')
    p.require(sys.platform=='linux' and 'huhengyuan' in Path(os.getenv('CONDA_PREFIX','')).parts and
              'huhengyuan' in output.resolve().parts,'Use personal persistent server environment/output')
    for name in ('NO_PROXY','no_proxy'):os.environ[name]='127.0.0.1,localhost'
    with r.run_lock(output/'active.lock'),r.run_lock(Path(s['parent_build'])/'active.lock'):
        (output/'runner.pid').write_text(str(os.getpid()))
        try:
            if (output/'build/release_manifest.json').exists():
                checked_release(output/'build','release_manifest.json')
                print('Already completed; zero new requests.');return
            parent=parent_ready(output)
            verify(output,replay=True)
            # Verify every complete original generation before any new inference.
            released_corpus.load(Path(s['base_corpus'])/'release_manifest.json',Path(s['base_corpus'])/'answers.json',full=True,complete_answers=True)
            b.wait_detached(output,plan['configuration_sha256'])
            if not (output/'runtime.json').exists():
                b.write_once(output/'runtime.json',{'python':sys.version,
                    'packages':{n:r.package_version(n) for n in ('vllm','transformers','torch','httpx')},'started_unix':time.time()})
            if phase in {'collect','all'}:collect(output,c)
            if phase in {'build','all'}:build(output,c,parent)
        except BaseException as exc:
            status(output,'stopped',phase=phase,error_type=type(exc).__name__,error=str(exc)[:1200]);raise
        finally:(output/'runner.pid').unlink(missing_ok=True)


def confirm(output):
    plan=verify(output);ready=b.read(output/'disconnect_ready.json');state=b.read(output/'execution_status.json')
    p.require(b.existing.process_identity(ready['pid'])==ready['process_start'] and time.time()-ready['unix']>=5,
              'Reconnect after five seconds to the same process')
    p.require(state['state']=='waiting_disconnect_acceptance' and state['token']==ready['token'] and
              ready['configuration_sha256']==plan['configuration_sha256'],'Not awaiting this confirmation')
    r.write_json(output/'disconnect_acceptance.json',dict(ready,passed=True,confirmed_unix=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','verify','run','confirm-detached','status'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['collect','build','all'],default='all')
    a=parser.parse_args()
    if a.mode=='prepare':print(json.dumps(initialize(a.output)['configuration_sha256']))
    elif a.mode=='verify':print(json.dumps(verify(a.output)['configuration_sha256']))
    elif a.mode=='run':run(a.output,a.phase)
    elif a.mode=='confirm-detached':confirm(a.output)
    else:print(json.dumps(b.read(a.output/'execution_status.json'),ensure_ascii=False))


if __name__=='__main__':main()
