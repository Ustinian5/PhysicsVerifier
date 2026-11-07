"""Detached continuation of the frozen original-source 30B experiment.

Wait for the already running answer generator, run independent Sol labels and
the four 120B checking arms, then perform semantic evaluation on the server.
This orchestrates existing protocols; it cannot select cases or change prompts.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import httpx

from scripts import run_qwen30b_step_full as full
from scripts import score_step_semantic as semantic
from scripts import run_bench_reannotation as teacher

vr, require = full.vr, full.require
PYTHON = '/home/visitor/huhengyuan/envs/physicsverifier/bin/python'
SERVICE_SCRIPTS = {'qwen': 'scripts/start_qwen30b_step_service.sh',
                   'oss': 'scripts/start_oss120b_step_service.sh'}


def phase(out, state, **fields):
    vr.atomic(out/'pipeline/execution_status.json', dict(state=state, updated_unix=time.time(), **fields))


def verify(out):
    full_plan = full.verify(out, acceptance=True)
    plan = vr.read(out/'pipeline/plan.json')
    require(plan['full_plan_sha256'] == vr.sha(out/'plan.json') and
            plan['full_configuration_sha256'] == full_plan['configuration_sha256'], 'Pipeline/full plan drift')
    for name, expected in plan['code_hashes'].items():
        require(vr.sha(full.ROOT/name) == expected, 'Pipeline source drift: '+name)
    return plan


def prepare(out, generator_pid, service_pid):
    require(Path('/proc').exists(), 'Pipeline executes on server only')
    full.verify(out, acceptance=True)
    target = out/'pipeline'; target.mkdir(exist_ok=True)
    with full.base.r.run_lock(target/'active.lock'):
        require(not (target/'plan.json').exists(), 'Pipeline already prepared; resume existing plan')
        require(str(out).encode() in Path(f'/proc/{generator_pid}/cmdline').read_bytes(), 'Wrong generator PID')
        require(SERVICE_SCRIPTS['qwen'].encode() in Path(f'/proc/{service_pid}/cmdline').read_bytes(), 'Wrong service PID')
        files = {'scripts/run_step_full_pipeline.py', 'scripts/run_bench_reannotation.py',
                 'scripts/bench_reannotation_runtime.py', 'scripts/score_step_semantic.py',
                 'scripts/response_stream.py', *SERVICE_SCRIPTS.values()}
        vr.atomic(target/'plan.json', {'full_plan_sha256':vr.sha(out/'plan.json'),
            'full_configuration_sha256':vr.read(out/'plan.json')['configuration_sha256'],
            'code_hashes':{n:vr.sha(full.ROOT/n) for n in sorted(files)},
            'generator':{'pid':generator_pid,'start':full.base.runtime.process_identity(generator_pid)},
            'qwen_service':{'pid':service_pid,'start':full.base.runtime.process_identity(service_pid)},
            'endpoint_sha256':vr.digest(os.environ['OPENAI_BASE_URL']), 'created_unix':time.time(),
            'python_executable':sys.executable,
            'runtime_pythonpath':os.environ.get('PYTHONPATH',''),
            'roles':'30B answers; 120B skills/checks; Sol high independent labels and semantic evaluation',
            'retry_policy':'existing per-stage budgets and exact saved-success replay only'})
    return verify(out)


def confirm(out):
    verify(out); target = out/'pipeline'; ready = vr.read(target/'disconnect_ready.json')
    require(full.base.runtime.process_identity(ready['pid']) == ready['process_start'] and
            ready['configuration_sha256'] == vr.sha(target/'plan.json') and
            time.time()-ready['unix'] >= 5, 'Fresh SSH confirmation of detached coordinator required')
    vr.atomic(target/'disconnect_acceptance.json',dict(ready,passed=True,confirmed_unix=time.time()))


def stage_acceptance(out, signature):
    """A later stage runs in the SAME previously detached, confirmed process."""
    plan = verify(out); target = out/'pipeline'
    ready, accepted = (vr.read(target/n) for n in ('disconnect_ready.json','disconnect_acceptance.json'))
    require(signature == plan['full_configuration_sha256'] and accepted.get('passed') and
            all(accepted.get(k)==v for k,v in ready.items()) and ready['pid']==os.getpid() and
            ready['process_start']==full.base.runtime.process_identity(os.getpid()),
            'Checking stage is not the confirmed server coordinator')
    vr.atomic(target/'checking_stage_acceptance.json',dict(accepted,stage='four_arm_checks',
        full_configuration_sha256=signature, checked_unix=time.time()))


def alive(record):
    try: return full.base.runtime.process_identity(record['pid']) == record['start']
    except (FileNotFoundError, ProcessLookupError): return False


def release_service(record, kind):
    """Only stop our known detached launcher group, never a recycled/shared PID."""
    if not alive(record): return
    pid = record['pid']
    require(SERVICE_SCRIPTS[kind].encode() in Path(f'/proc/{pid}/cmdline').read_bytes() and
            os.getpgid(pid)==pid, 'Service ownership/group changed')
    os.killpg(pid, signal.SIGTERM)


def launch(args, log, env=None):
    with log.open('ab') as stream:
        return subprocess.Popen(args, cwd=full.ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            env=env or os.environ.copy())


def wait_answers(out, plan):
    deadline = time.monotonic()+48*3600
    while time.monotonic()<deadline:
        require(not (out/'STOP').exists(), 'STOP requested')
        if alive(plan['generator']):
            phase(out,'waiting_30b_generation',completed=len(list((out/'generated').glob('*.json'))))
            time.sleep(15);continue
        summary=out/'generation_summary.json'
        completed=vr.read(summary) if summary.exists() else {}
        # A later checking stage shares runner.exit_code; do not mistake its failure
        # for a failed generator when the complete answer receipt already exists.
        require(completed.get('complete')==100 and not completed.get('failures') or
                (out/'runner.exit_code').read_text().strip()=='0', 'Answer generator stopped with failure')
        rows = full.export(out, full.verify(out, acceptance=True))
        require(len(rows)==100, 'Incomplete answer batch')
        return
    raise RuntimeError('Answer wait deadline reached')


def link_stop(out, target):
    path=target/'STOP'
    if path.is_symlink():
        require(path.resolve()==(out/'STOP').resolve(),'Child STOP points outside this batch')
    elif not path.exists():
        path.symlink_to(os.path.relpath(out/'STOP',target))


def start_labels(out):
    full.prepare_annotations(out, full.ROOT/'.env')
    target = out/'annotations'
    link_stop(out,target)
    rows, manifest = vr.read(target/'inputs.json'), vr.read(target/'manifest.json')
    if teacher.audit(target,rows,manifest)['completed_tasks']==400: return None
    if (target/'runner.pid').exists():
        pid=int((target/'runner.pid').read_text())
        try: os.kill(pid,0)
        except ProcessLookupError: pass
        else: raise RuntimeError('A label runner already exists; inspect rather than duplicate')
    return launch([PYTHON,'-u','-m','scripts.run_bench_reannotation',
        '--source',str(out/'annotation_inputs.json'),'--output',str(target),
        '--env',str(full.ROOT/'.env'),'--mode','pilot','--expected-count','100',
        '--pilot-size','100','--recover-technical'], target/'runner.log')


def start_checker_service(out):
    target = out.parent/'oss_service'
    # The existing launcher itself also checks GPU UUID/free memory, port and weights.
    used=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used',
                                  '--format=csv,noheader,nounits'],text=True)
    require(int(used.strip())<100, 'GPU1 unavailable; do not displace other users')
    with httpx.Client(trust_env=False,timeout=3) as client:
        try: response=client.get('http://127.0.0.1:18083/v1/models')
        except httpx.TransportError: pass
        else: raise RuntimeError('Port18083 already has a service; do not adopt or duplicate it')
    service = launch(['bash',str(full.ROOT/SERVICE_SCRIPTS['oss'])],target/'runner.log')
    record={'pid':service.pid,'start':full.base.runtime.process_identity(service.pid)}
    vr.atomic(out/'pipeline/oss_service.json',record)
    deadline=time.monotonic()+1800
    with httpx.Client(trust_env=False,timeout=5) as client:
        while time.monotonic()<deadline:
            require(not (out/'STOP').exists(), 'STOP requested')
            require(service.poll() is None,'120B service exited during loading')
            try:
                response=client.get('http://127.0.0.1:18083/v1/models');response.raise_for_status()
                require(any(x['id']==full.base.r.EXPECTED_MODEL for x in response.json()['data']),
                        'Wrong120B service identity')
                return record
            except httpx.TransportError: pass
            phase(out,'loading_120b_service',service_pid=service.pid);time.sleep(10)
    raise RuntimeError('120B service readiness deadline reached')


def run(out):
    require(Path('/proc').exists(),'Server only')
    target=out/'pipeline'; code=1; service=None
    with full.base.r.run_lock(target/'active.lock'):
        plan=verify(out)
        require(sys.executable==plan['python_executable'] and
                os.environ.get('PYTHONPATH','')==plan['runtime_pythonpath'], 'Pipeline environment drift')
        require(vr.digest(os.environ['OPENAI_BASE_URL'])==plan['endpoint_sha256'], 'Teacher endpoint changed')
        require(bool(os.environ.get('OPENAI_API_KEY')), 'Missing independent-label credential')
        (target/'runner.pid').write_text(str(os.getpid()))
        try:
            full.base.runtime.wait_detached(target,vr.sha(target/'plan.json'))
            wait_answers(out,plan)
            labels=start_labels(out)
            release_service(plan['qwen_service'],'qwen')
            phase(out,'labels_started_preparing_checker',label_pid=labels.pid if labels else None)
            try: service=start_checker_service(out)
            except BaseException:
                owned=target/'oss_service.json'
                if owned.exists():release_service(vr.read(owned),'oss')
                raise
            phase(out,'building_skills_then_four_arm_checks',label_pid=labels.pid if labels else None)
            with patch.object(full.base.runtime,'wait_detached',lambda path,sig: stage_acceptance(out,sig)):
                full.run_checks(out)
            release_service(service,'oss');service=None
            if labels:
                while labels.poll() is None:
                    require(not (out/'STOP').exists(),'STOP requested')
                    phase(out,'waiting_independent_labels',label_pid=labels.pid);time.sleep(15)
                require(labels.returncode==0,'Independent labeling stopped; keep all saved tasks')
            sem=out/'semantic';endpoint=os.environ['OPENAI_BASE_URL']
            if not (sem/'manifest.json').exists():
                semantic.prepare(sem,out/'checks',out/'annotations',endpoint)
            link_stop(out,sem)
            phase(out,'independent_semantic_matching')
            semantic.run(sem,endpoint,os.environ['OPENAI_API_KEY'])
            semantic.score(sem)
            result=vr.read(sem/'metrics.json')
            phase(out,'complete' if result['primary_metrics_ready'] else 'completed_needs_adjudication',
                  primary_metrics_ready=result['primary_metrics_ready'],metrics='semantic/metrics.json')
            code=0
        except BaseException as exc:
            phase(out,'stopped',error_type=type(exc).__name__,error=str(exc)[:1000]);raise
        finally:
            if service: release_service(service,'oss')
            (target/'runner.exit_code').write_text(str(code));(target/'runner.pid').unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','confirm-detached','verify'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--generator-pid',type=int);p.add_argument('--service-pid',type=int)
    a=p.parse_args();out=a.output.resolve()
    if a.command=='prepare':prepare(out,a.generator_pid,a.service_pid)
    elif a.command=='run':run(out)
    elif a.command=='confirm-detached':confirm(out)
    else:verify(out)
