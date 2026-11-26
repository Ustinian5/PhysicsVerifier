"""Own a real reviewed-library inference service on an available node14 GPU."""
from contextlib import contextmanager
import importlib.metadata
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx

from scripts import run_node14_final_pilot as host

vr, require = host.vr, host.require
CHECK_DATE = '2026-11-03'


def node():
    require(socket.gethostname().split('.')[0] == 'node14' and Path(sys.prefix) == host.ENV,
            'Run in the personal node14 conda environment')
    host.set_environment()


def confirm(folder, signature):
    ready = vr.read(folder / 'disconnect_ready.json')
    require(ready['configuration_sha256'] == signature and time.time() - ready['unix'] >= 5
            and host.final.base.runtime.process_identity(ready['pid']) == ready['process_start'],
            'Detached supervisor identity changed')
    vr.atomic(folder / 'disconnect_acceptance.json', dict(ready, passed=True, confirmed_unix=time.time()))


@contextmanager
def service(folder, gpu, stop_file, signature, *, fixed_date=CHECK_DATE):
    """The caller opens this context only when a verified new request is ready."""
    require(gpu in (4, 5, 6, 7), 'Unexpected stage or GPU')
    node()
    port = host.PORT
    cfg = vr.read(host.OUT / 'plan.json')['configuration']
    expected = vr.read(host.OUT.parent / 'evaluation/runtime.json')['packages']
    environment = vr.read(host.CONTROL / 'environment.json')
    require(environment['verified'] and environment['prefix'] == str(host.ENV)
            and all(importlib.metadata.version(k) == v for k, v in expected.items()), 'Runtime changed')
    proof = vr.read(host.CONTROL / 'attempts/2/deployment.json')['weights_receipt_sha256']
    redirect = host.weight_location(cfg, folder, proof)
    script, _ = host.service_script(cfg)
    require(fixed_date in (CHECK_DATE, '2026-11-07'), 'Unrecognized verifier service date')
    date = fixed_date
    url = f'http://127.0.0.1:{port}'
    lock = host.final.base.r.run_lock
    with lock(host.HOT / f'locks/organization_service_{port}.lock'), lock(host.HOT / f'locks/library_gpu_{gpu}.lock'):
        while True:
            require(not stop_file.exists(), 'STOP requested')
            snapshot = host.resources()
            card = host.idle_card(dict(snapshot, gpus=[v for v in snapshot['gpus'] if int(v[0]) == gpu]))
            if card and not host.sockets(['-Hltn']):
                break
            vr.atomic(folder / 'service_status.json', dict(state='waiting_for_gpu_or_port', gpu=gpu, unix=time.time()))
            time.sleep(20)
        index, uuid = card
        env = dict(os.environ, STEP_OSS_GPU_INDEX=index, STEP_OSS_GPU_UUID=uuid,
                   STEP_OSS_RUN_ROOT=str(folder / 'service'), STEP_OSS_CACHE_ROOT=str(host.HOT / 'cache/gptoss120b'),
                   VLLM_MODEL_REDIRECT_PATH=str(redirect), PV_HARMONY_DATE=date)
        with (folder / 'service.log').open('ab') as log:
            proc = subprocess.Popen(['/bin/bash', '-c', script], cwd=host.ROOT, env=env, start_new_session=True,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        ready = False
        record = dict(pid=proc.pid, start=host.final.base.runtime.process_identity(proc.pid), gpu_uuid=uuid,
                      gpu_index=index, port=port, preflight=snapshot, model=cfg['model'], fixed_date=date,
                      weights_receipt_sha256=proof, configuration_sha256=signature)
        try:
            vr.atomic(folder / 'deployment.json', record)
            with httpx.Client(trust_env=False, timeout=10) as client:
                while True:
                    require(not stop_file.exists() and proc.poll() is None, 'STOP or service exit before readiness')
                    if host.owned_listener(proc.pid):
                        try:
                            response = client.get(url + '/v1/models')
                            if response.is_success:
                                require([m['id'] for m in response.json()['data']] == [cfg['model']],
                                        'Unexpected deployed model')
                                ready = True
                                break
                        except httpx.TransportError:
                            pass
                    require(time.time() - (folder / 'service.log').stat().st_mtime < 3600, 'Service startup stalled')
                    time.sleep(3)
                require(vr.read(folder / 'service/startup.json')['fixed_harmony_date'] == date, 'Wrong Harmony date')
            yield url + '/v1'
        finally:
            host.release(proc, record, folder, ready, url)
