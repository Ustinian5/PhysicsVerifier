"""Run one independent small-library organizer in a detached node14 process."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx

from scripts import run_node14_final_pilot as host
from scripts import run_verifier_library_scale as scale


vr, require = host.vr, host.require
RESULTS = host.HOT/'results/qwen30b_step_verifier'


def launcher(config, port):
    script, _ = host.service_script(config)
    for old, new in [("s.bind(('127.0.0.1',18083))", f"s.bind(('127.0.0.1',{port}))"),
                     ('--port 18083 ', f'--port {port} '),
                     ('-m scripts.serve_fixed_date_vllm', '-m scripts.serve_continuous_context_vllm')]:
        require(script.count(old) == 1, 'Unexpected original launcher')
        script = script.replace(old,new)
    return script


def available_card(snapshot, gpu):
    require(gpu in (4,5,6,7), 'GPU outside authorized indices')
    return host.idle_card(dict(snapshot, gpus=[g for g in snapshot['gpus'] if int(g[0])==gpu]))


def service_port(size, seed):
    size_index = scale.BUILD_SIZES.index(size)
    if seed in scale.RANDOM_SEEDS:
        return 18160 + 10 * size_index + scale.RANDOM_SEEDS.index(seed)
    return 18100 + 10 * size_index + seed % 100


def run(out, folder, gpu, plan):
    from transformers import AutoTokenizer

    if (out/'construction_acceptance.json').exists():
        require(vr.read(out/'construction_acceptance.json')['configuration_sha256']==plan['configuration_sha256'],
                'Existing release belongs to another run')
        return
    host.set_environment()
    c = plan['configuration']
    port = service_port(c['source_count'], c['seed'])
    # Each detached supervisor process owns exactly one port; reuse the audited
    # PID, process-group, in-flight-request and release checks for that port.
    host.PORT = port
    url = f'http://127.0.0.1:{port}'
    transport = dict(c['transport'], auditor_api=url+'/v1')
    accepted = vr.read(host.CONTROL/'environment.json')
    expected = vr.read(host.OUT.parent/'evaluation/runtime.json')['packages']
    require(accepted['verified'] and accepted['prefix']==str(host.ENV) and
            all(importlib.metadata.version(k)==v for k,v in expected.items()), 'Wrong personal runtime')
    weight_proof = host.CONTROL/'weights.json'
    prior = vr.read(host.CONTROL/'attempts/2/deployment.json')
    frozen = vr.read(host.OUT/'plan.json')['configuration']
    require(vr.sha(weight_proof)==prior['weights_receipt_sha256'] and
            frozen['model']==transport['model'] and transport['fixed_date']=='2026-11-07',
            'Wrong construction checkpoint or Harmony date')
    script = launcher(frozen, port)
    with host.final.base.r.run_lock(out/'active.lock'):
        host.final.base.runtime.wait_detached(folder, plan['configuration_sha256'])
        require(not (out/'STOP').exists(), 'STOP requested')
        scale.verify(out)
        progress = out/'tree_progress.json'
        if progress.exists() and vr.unseal(progress)['state']['phase']=='done':
            # Re-materialize conserved leaves after a crash between the final
            # checkpoint and file publication; this phase makes no model call.
            scale.build(out, plan, None, transport)
            scale.accept(out, plan)
            return
        vr.atomic(folder/'execution_status.json', dict(state='verifying_weights', gpu=gpu, unix=time.time()))
        redirect = host.weight_location(frozen, folder, prior['weights_receipt_sha256'])
        tokenizer = AutoTokenizer.from_pretrained(transport['model'],local_files_only=True,trust_remote_code=False)
        with host.final.base.r.run_lock(host.HOT/f'locks/organization_service_{port}.lock'), \
             host.final.base.r.run_lock(host.HOT/f'locks/library_gpu_{gpu}.lock'):
            while True:
                require(not (out/'STOP').exists(), 'STOP requested')
                before = host.resources()
                card = available_card(before, gpu)
                if card and not host.sockets(['-Hltn']):
                    break
                vr.atomic(folder/'execution_status.json',dict(state='waiting_for_gpu_or_port',gpu=gpu,unix=time.time()))
                time.sleep(20)
            index, uuid = card
            env = dict(os.environ, STEP_OSS_GPU_INDEX=index, STEP_OSS_GPU_UUID=uuid,
                       STEP_OSS_RUN_ROOT=str(folder/'service'), STEP_OSS_CACHE_ROOT=str(host.HOT/'cache/gptoss120b'),
                       VLLM_MODEL_REDIRECT_PATH=str(redirect), PV_HARMONY_DATE=transport['fixed_date'])
            with (folder/'service.log').open('ab') as log:
                service = subprocess.Popen(['/bin/bash','-c',script], cwd=host.ROOT, env=env, start_new_session=True,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            ready = False
            record = dict(pid=service.pid, start=host.final.base.runtime.process_identity(service.pid), gpu_uuid=uuid,
                          gpu_index=index, port=port, preflight=before, fixed_date=transport['fixed_date'],
                          model=transport['model'], weights_receipt_sha256=prior['weights_receipt_sha256'],
                          configuration_sha256=plan['configuration_sha256'])
            try:
                vr.atomic(folder/'deployment.json',record)
                vr.atomic(folder/'execution_status.json',dict(state='starting_service',gpu=gpu,unix=time.time()))
                with httpx.Client(trust_env=False,timeout=10) as client:
                    while not ready:
                        require(service.poll() is None and not (out/'STOP').exists(),
                                'Service exited or STOP requested before readiness')
                        if host.owned_listener(service.pid):
                            try:
                                response = client.get(url+'/v1/models')
                                if response.is_success:
                                    require([m['id'] for m in response.json()['data']]==[transport['model']],
                                            'Service model identity changed')
                                    ready = True
                                    break
                            except httpx.TransportError:
                                pass
                        require(time.time()-(folder/'service.log').stat().st_mtime < 3600, 'Startup stopped making progress')
                        time.sleep(3)
                require(vr.read(folder/'service/startup.json')['fixed_harmony_date']==transport['fixed_date'],
                        'Deployed Harmony date changed')
                vr.atomic(folder/'execution_status.json',dict(state='building_tree',gpu=gpu,unix=time.time()))
                scale.build(out, plan, tokenizer, transport)
            finally:
                host.release(service, record, folder, ready, url)
        scale.accept(out, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run','confirm'))
    parser.add_argument('--seed', type=int, required=True, choices=scale.SEEDS)
    parser.add_argument('--size', type=int, choices=scale.BUILD_SIZES, default=250)
    parser.add_argument('--gpu', type=int, required=True, choices=range(4,8))
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    require(socket.gethostname().split('.')[0]=='node14' and Path(sys.prefix)==host.ENV and args.attempt>0,
            'Unexpected host, conda environment or attempt')
    out = RESULTS/'library_scale'/f'n{args.size}_seed{args.seed}'
    folder = out/'attempts'/str(args.attempt)
    plan = scale.verify(out)
    require((plan['configuration']['source_count'], plan['configuration']['seed']) == (args.size, args.seed),
            'Library size or seed differs from the published plan')
    if args.mode=='confirm':
        ready = vr.read(folder/'disconnect_ready.json')
        require(ready['configuration_sha256']==plan['configuration_sha256'] and time.time()-ready['unix']>=5
                and host.final.base.runtime.process_identity(ready['pid'])==ready['process_start'],
                'Fresh SSH confirmation does not match the waiting process')
        vr.atomic(folder/'disconnect_acceptance.json',dict(ready,passed=True,confirmed_unix=time.time()))
        return
    require(not (folder/'disconnect_ready.json').exists(),'Use a fresh attempt number to resume')
    folder.mkdir(parents=True,exist_ok=True)
    try:
        run(out, folder, args.gpu, plan)
    except Exception as error:
        vr.atomic(folder/'execution_status.json',dict(state='failed',error=str(error),unix=time.time()))
        raise
    vr.atomic(folder/'execution_status.json',dict(state='completed',unix=time.time()))
    scale.construction.emit(out,'completed',acceptance_sha256=vr.sha(out/'construction_acceptance.json'))


if __name__=='__main__':
    main()
