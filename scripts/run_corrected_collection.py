"""Generate the frozen, isolated build and final-answer candidate answers.

Generation only: never invokes the stopped legacy coordinator or reads labels.
"""
import argparse
from collections import Counter
from itertools import zip_longest
import os
from pathlib import Path
import threading
import time

import httpx
from scripts import run_corrected_step_generation as prior

corpus=prior.corpus
read,write,require,sha,digest=prior.read,prior.write,prior.require,prior.sha,prior.digest
ROOT=prior.ROOT
VERSION='corrected_combined_collection_v1'
SOURCES=tuple(sorted(set(prior.SOURCES+('scripts/run_corrected_collection.py',))))


def ordered_tasks(split):
    build,reserve=split['build'],split['reserve']
    require(len(build)==2000 and len(reserve)>=600,'Wrong frozen population')
    ids=[r['id'] for r in build+reserve]
    require(len(set(ids))==len(ids),'Build/test ID overlap')
    tasks=[]
    for pair in zip_longest(build,reserve):
        for role,row in zip(('build','heldout'),pair):
            if row is not None:
                tasks.append(dict(role=role,row={k:row[k] for k in ('id','question')}))
    return tasks


def prepare(out,screen,bench):
    require(not (out/'plan.json').exists(),'Already prepared; verify or resume')
    parent=prior.verify(bench);split=read(screen/'split.json');tasks=ordered_tasks(split)
    require(split['build_target']==2000 and split['benchmark_target']==600 and
            split['source_plan_sha256']==sha(screen/'plan.json') and
            split['family_sha256']==sha(screen/'family_components.json'),'Split binding changed')
    families=read(screen/'family_components.json')['components']
    a={x['id'] for x in split['build']};b={x['id'] for x in split['reserve']}
    for family in families:
        require(not(a.intersection(family['members']) and b.intersection(family['members'])),
                'Family crosses build/heldout boundary')
    old={prior.question_key(r['question']) for r in read(bench/'questions.json')}
    require(not any(prior.question_key(t['row']['question']) in old for t in tasks),'Step bench overlap')
    c=parent['configuration']['configuration']
    plan=dict(protocol=VERSION,generator_config=c,solver_prompt=corpus.SOLVER,
        tasks_sha256=digest(tasks),split_path=os.path.relpath(screen/'split.json',out),
        split_sha256=sha(screen/'split.json'),family_path=os.path.relpath(screen/'family_components.json',out),
        family_sha256=sha(screen/'family_components.json'),parent_plan_sha256=sha(bench/'plan.json'),
        code_hashes={n:sha(ROOT/n) for n in SOURCES},
        roles='30B generation only; question only; all heldout answers forbidden for construction',
        scheduling='alternate build/heldout in frozen order; no outcome-dependent sampling')
    with corpus.r.run_lock(out/'active.lock'):
        write(out/'tasks.json',tasks)
        write(out/'plan.json',dict(configuration=plan,configuration_sha256=digest(plan)))
        write(out/'execution_status.json',dict(state='prepared_no_calls',total=len(tasks),build=2000,heldout=len(b)))


def verify(out):
    env=read(out/'plan.json');p=env['configuration']
    require(p['protocol']==VERSION and digest(p)==env['configuration_sha256'],'Plan drift')
    require(p['solver_prompt']==corpus.SOLVER and all(sha(ROOT/n)==h for n,h in p['code_hashes'].items()),'Code/prompt drift')
    require(sha(out/p['split_path'])==p['split_sha256'] and sha(out/p['family_path'])==p['family_sha256'],'Split drift')
    require(digest(read(out/'tasks.json'))==p['tasks_sha256'],'Task drift')
    return env


def confirm(out):
    p=verify(out);ready=read(out/'disconnect_ready.json')
    require(corpus.runtime.process_identity(ready['pid'])==ready['process_start'] and
            ready['configuration_sha256']==p['configuration_sha256'] and time.time()-ready['unix']>=5,
            'Fresh SSH confirmation required')
    write(out/'disconnect_acceptance.json',dict(ready,passed=True,confirmed_unix=time.time()))


def run(out):
    require(Path('/proc').exists(),'Run on server')
    with corpus.r.run_lock(out/'active.lock'),corpus.r.run_lock(out.parent/'generation.lock'):
        env=verify(out);p=env['configuration'];c=p['generator_config'];code=1
        (out/'runner.pid').write_text(str(os.getpid()))
        try:
            require(not(out/'STOP').exists(),'STOP requested')
            corpus.runtime.wait_detached(out,env['configuration_sha256'])
            with httpx.Client(trust_env=False,timeout=10) as client:
                response=client.get(c['generator_api']+'/models');response.raise_for_status()
                require(any(x['id']==c['generator'] and x.get('max_model_len',0)>=32768 and
                    x.get('root')=='/slow_share2/huhengyuan/models/Qwen3-30B-A3B-Instruct-2507'
                    for x in response.json()['data']),'30B service identity/context mismatch')
            journal=corpus.Journal(out,c,threading.Event());counts=Counter()
            tasks=read(out/'tasks.json');by_role={'build':[],'heldout':[]}
            for task in tasks:
                require(not(out/'STOP').exists(),'STOP requested')
                saved=corpus.generate(journal,task['row']);role=task['role']
                counts[role+'/'+saved['state']]+=1
                if saved['state']=='complete':
                    by_role[role].append(dict(task['row'],context='',prediction=saved['prediction']))
                write(out/(role+'_inputs.json'),by_role[role])
                write(out/'execution_status.json',dict(state='generating',counts=dict(counts),total=len(tasks),
                    processed=sum(counts.values()),updated_unix=time.time(),downstream_started=False))
            write(out/'execution_status.json',dict(state='generation_finished',counts=dict(counts),
                processed=sum(counts.values()),total=len(tasks),updated_unix=time.time(),downstream_started=False))
            code=0
        except BaseException as exc:
            write(out/'execution_status.json',dict(state='stopped',error=str(exc)[:1000],updated_unix=time.time()))
            raise
        finally:
            (out/'runner.exit_code').write_text(str(code));(out/'runner.pid').unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','verify','run','confirm-detached'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--screen',type=Path);p.add_argument('--bench',type=Path)
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='prepare':
        require(a.screen is not None and a.bench is not None,'Source paths required')
        prepare(out,a.screen.resolve(),a.bench.resolve())
    elif a.mode=='verify':verify(out);print('verified')
    elif a.mode=='confirm-detached':confirm(out)
    else:run(out)


if __name__=='__main__':main()
