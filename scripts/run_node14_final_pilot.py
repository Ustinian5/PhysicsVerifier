"""Run the frozen six-question pilot on node14 with a private service and locks."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

import httpx

from scripts import run_final_answer_revision as final


HOT = Path('/data0/zengzheni/PhysicsVerifier-huhengyuan')
ROOT = Path('/home/zengzheni/PhysicsVerifier/huhengyuan')
ENV = HOT / 'envs/physicsverifier'
OUT = HOT / 'results/qwen30b_step_verifier/aligned_old99/final_revision'
CONTROL = OUT / 'node14_pilot'
LOCK = HOT / 'locks/organization_service_18083.lock'
PORT = 18083
CONFIG = '3482ba6eedd144a5984bb7f9cdf89a170cceb9d5c82546778d8de99a004bfc4f'
PILOT = '618991c554aebc6d2875aab871eaa6a8713e6089008534b7718e586254385855'
LAUNCHER = 'scripts/start_oss120b_step_service.sh'
vr, require = final.vr, final.require


def service_script(config):
    path = ROOT / LAUNCHER
    require(vr.sha(path) == config['source_hashes'][LAUNCHER], 'Frozen launcher changed')
    script = path.read_text()
    paths = {
        'ENV=/home/visitor/huhengyuan/envs/physicsverifier': f'ENV={ENV}',
        'export PYTHONPATH=/home/visitor/huhengyuan/envs/vllm_gptoss120b/lib/python3.10/site-packages':
            f'export PYTHONPATH={ENV}/runtime/site-packages',
        'export TIKTOKEN_ENCODINGS_BASE=/home/visitor/huhengyuan/.cache/gptoss120b/harmony':
            f'export TIKTOKEN_ENCODINGS_BASE={ENV}/share/harmony',
        'cd /home/visitor/huhengyuan/PhysicsVerifier': f'cd {ROOT}',
    }
    for old, new in paths.items():
        require(script.count(old + '\n') == 1, 'Unexpected launcher path: ' + old)
        script = script.replace(old + '\n', new + '\n')
    return script, paths


def set_environment():
    cache = HOT / 'cache/gptoss120b'
    locations = dict(XDG_CACHE_HOME=cache, HF_HOME=cache / 'huggingface', TRITON_CACHE_DIR=cache / 'triton',
                     TORCH_HOME=cache / 'torch', TORCHINDUCTOR_CACHE_DIR=cache / 'inductor',
                     TORCH_EXTENSIONS_DIR=cache / 'extensions', CUDA_CACHE_PATH=cache / 'cuda',
                     VLLM_CACHE_ROOT=cache / 'vllm', TMPDIR=cache / 'tmp', FLASHINFER_WORKSPACE_BASE=cache / 'flashinfer')
    for key, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(CONDA_PREFIX=str(ENV), PYTHONPATH=str(ENV / 'runtime/site-packages'), PYTHONNOUSERSITE='1',
                      PYTHONDONTWRITEBYTECODE='1', PATH=str(ENV / 'bin') + ':' + os.environ['PATH'],
                      TIKTOKEN_ENCODINGS_BASE=str(ENV / 'share/harmony'), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', NO_PROXY='127.0.0.1,localhost')


def nvidia(query, fields):
    result = subprocess.check_output(
        ['nvidia-smi', f'--query-{query}={fields}', '--format=csv,noheader,nounits'], text=True)
    return [list(map(str.strip, line.split(','))) for line in result.splitlines() if line.strip()]


def resources():
    gpus = nvidia('gpu', 'index,uuid,name,memory.total,memory.used,utilization.gpu')
    processes = nvidia('compute-apps', 'gpu_uuid,pid')
    return dict(gpus=gpus, compute_processes=processes, disk_free_bytes=shutil.disk_usage(HOT).free)


def idle_card(snapshot):
    busy = {uuid for uuid, _ in snapshot['compute_processes']}
    for index, uuid, name, total, used, utilization in snapshot['gpus']:
        if ('A800' in name and int(total) >= 80000 and uuid not in busy
                and int(used) < 100 and int(utilization) == 0):
            return index, uuid
    return None


def sockets(options):
    return subprocess.check_output(['ss', *options, f'( sport = :{PORT} )'], text=True).strip()


def owned_listener(pid):
    listeners = sockets(['-Hltnp'])
    pids = [int(p) for p in re.findall(r'pid=(\d+)', listeners)]
    return bool(pids) and all(os.getpgid(p) == pid for p in pids)


def request_counts(client, url):
    response = client.get(url + '/metrics')
    response.raise_for_status()
    counts = {}
    for key in ('running', 'waiting'):
        values = re.findall(r'^vllm:num_requests_' + key + r'(?:\{[^\n]*?\})?\s+([\d.eE+-]+)\s*$',
                            response.text, re.MULTILINE)
        require(values, 'Missing service request gauge: ' + key)
        counts[key] = sum(map(float, values))
    return counts


def release(service, record, folder, ready, url):
    idle = {}
    if service.poll() is None:
        pid = service.pid
        identity = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
        require(final.base.runtime.process_identity(pid) == record['start'] and os.getpgid(pid) == pid
                and os.fsencode('STEP_OSS_RUN_ROOT=' + str(folder / 'service')) in identity,
                'Service process identity changed')
        if ready or owned_listener(pid):
            while True:
                require(owned_listener(pid), 'Service port ownership changed')
                with httpx.Client(trust_env=False, timeout=10) as client:
                    idle = request_counts(client, url)
                if not any(idle.values()) and not sockets(['-Htn', 'state', 'established']):
                    break
                time.sleep(3)
        os.killpg(pid, signal.SIGTERM)
        service.wait(timeout=150)
    deadline = time.monotonic() + 150
    while True:
        snapshot = resources()
        gpu_pids = [pid for uuid, pid in snapshot['compute_processes'] if uuid == record['gpu_uuid']]
        ours = []
        for pid in gpu_pids:
            proc_env = Path(f'/proc/{pid}/environ')
            try:
                if os.fsencode('STEP_OSS_RUN_ROOT=' + str(folder / 'service')) in proc_env.read_bytes().split(b'\0'):
                    ours.append(pid)
            except (PermissionError, FileNotFoundError, ProcessLookupError):
                continue
        if not ours and not owned_listener(service.pid):
            break
        require(time.monotonic() < deadline, 'Owned GPU process or listener still active')
        time.sleep(2)
    vr.atomic(folder / 'release.json', dict(unix=time.time(), gauges=idle, exit_code=service.returncode,
              owned_gpu_pids=ours, gpu_snapshot=snapshot, port_listeners=sockets(['-Hltnp'])))


def audit_results(plan, tokenizer, folder):
    rows = vr.read(OUT / 'pilot_inputs.json')
    pilot = OUT / 'pilot'
    memories = final.stores(OUT)
    groups = {arm: dict(complete=0, with_material=0, max_items=0, max_material_tokens=0,
                        max_wall_seconds=0, controls={}, failures={}) for arm in final.ARMS}
    requests = {}
    controls = {item['id']: item['expected'] for item in final.old_labels.CONTROLS[:2]}
    def count(messages):
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, reasoning_effort='medium'))
    def actual_input(path, result):
        raw = vr.read(path / 'raw.json')
        actual = final.validate_effective_input(raw, tokenizer, plan['configuration']['fixed_harmony_date'])
        messages = vr.read(path / 'request.json')['request']['messages']
        decoded = tokenizer.decode(raw['prompt_token_ids'], skip_special_tokens=False)
        require(actual == result['effective_input'] and all(m['content'] in decoded for m in messages),
                'Effective input differs from the saved messages')
        requests[str(path.relative_to(pilot))] = vr.sha(path / 'result.json')
    for row in rows:
        key = vr.digest(row['id'])[:24]
        query = pilot / 'queries' / key
        prefix = final.saved_request(query, final.p.query_messages(row), final.base.p.QUERY_MAX_TOKENS,
                                     final.base.p.validate_queries, plan,
                                     available=plan['configuration']['limits']['wall_seconds_per_sample'])
        actual_input(query, prefix)
        for arm in final.ARMS:
            target = pilot / arm / key
            value = vr.read(target / 'result.json')
            final.verify_arm(row, arm, target / 'result.json', value, prefix, plan)
            if prefix['status'] == 'complete':
                packet = final.p.reference_packet(row, arm, prefix['value'], memories[arm], count)
                require(packet == vr.read(target / 'reference_packet.json'), 'Replayed retrieval differs')
                actual_input(target / 'check', vr.read(target / 'check/result.json'))
            summary = groups[arm]
            retrieval = value['retrieval']
            items = len(retrieval['provided_complete_ids'])
            summary['with_material'] += int(items > 0)
            summary['max_items'] = max(summary['max_items'], items)
            summary['max_material_tokens'] = max(summary['max_material_tokens'], retrieval['reference_tokens'])
            summary['max_wall_seconds'] = max(summary['max_wall_seconds'], value['wall_seconds'])
            if value['status'] == 'complete':
                summary['complete'] += 1
            else:
                summary['failures'][row['id']] = value['reason']
            if row['id'] in controls:
                summary['controls'][row['id']] = dict(expected=controls[row['id']],
                    actual=value['value']['verdict'] if value['value'] else None)
    receipt = OUT / 'pilot_acceptance.json'
    if receipt.exists():
        accepted = vr.read(receipt)
        require(accepted['passed'] and accepted['configuration_sha256'] == CONFIG
                and all(vr.sha(Path(p)) == h for p, h in accepted['files'].items()), 'Native acceptance drift')
    vr.atomic(folder / 'audit.json', dict(questions=[r['id'] for r in rows], groups=groups, requests=requests,
              accepted=receipt.exists(), configuration_sha256=CONFIG, offline_model_calls=0))


def run(folder, plan, signature):
    from transformers import AutoTokenizer

    cfg = plan['configuration']
    require(socket.gethostname().split('.')[0] == 'node14' and Path(sys.prefix) == ENV, 'Wrong host or conda prefix')
    require(final.a.ROOT == ROOT and cfg['api_base'] == f'http://127.0.0.1:{PORT}/v1', 'Wrong checkout or endpoint')
    require(not (OUT / 'pilot_acceptance.json').exists(), 'Fixed pilot already accepted')
    env_receipt = vr.read(CONTROL / 'environment.json')
    packages = vr.read(OUT.parent / 'evaluation/runtime.json')['packages']
    require(env_receipt['verified'] and env_receipt['prefix'] == str(ENV)
            and {k: importlib.metadata.version(k) for k in packages} == packages, 'Runtime differs')
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], local_files_only=True, trust_remote_code=False)
    final.a.historical_transport()
    script, paths = service_script(cfg)
    url = cfg['api_base'].removesuffix('/v1')
    with final.base.r.run_lock(OUT / 'active.lock'), final.base.r.run_lock(LOCK):
        final.base.runtime.wait_detached(folder, signature)
        while True:
            require(not (OUT / 'STOP').exists(), 'STOP requested')
            snapshot = resources()
            card = idle_card(snapshot)
            if card is not None and not sockets(['-Hltn']):
                break
            vr.atomic(folder / 'execution_status.json', dict(state='waiting_for_free_gpu_or_port', unix=time.time(), **snapshot))
            time.sleep(30)
        index, uuid = card
        env = dict(os.environ, STEP_OSS_GPU_INDEX=index, STEP_OSS_GPU_UUID=uuid,
                   STEP_OSS_RUN_ROOT=str(folder / 'service'), STEP_OSS_CACHE_ROOT=str(HOT / 'cache/gptoss120b'),
                   PV_HARMONY_DATE=cfg['fixed_harmony_date'])
        with (folder / 'service.log').open('ab') as log:
            service = subprocess.Popen(['/bin/bash', '-c', script], cwd=ROOT, env=env, start_new_session=True,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        ready = False
        record = dict(pid=service.pid, start=final.base.runtime.process_identity(service.pid), gpu_uuid=uuid,
                      gpu_index=index, preflight=snapshot, port=PORT, source_lock=cfg['service_lock'],
                      active_lock=str(LOCK), source_launcher_sha256=cfg['source_hashes'][LAUNCHER],
                      wrapper_sha256=vr.sha(Path(__file__)), infrastructure_paths=paths,
                      configuration_sha256=plan['configuration_sha256'], environment_sha256=vr.sha(CONTROL / 'environment.json'))
        try:
            vr.atomic(folder / 'deployment.json', record)
            vr.atomic(folder / 'execution_status.json', dict(state='starting_service', unix=time.time(), pid=service.pid))
            start = time.monotonic()
            with httpx.Client(trust_env=False, timeout=10) as client:
                while True:
                    require(service.poll() is None, 'Service exited before readiness; see service.log')
                    require(time.monotonic() - start < 3600, 'Service startup timed out')
                    require(not (OUT / 'STOP').exists(), 'STOP requested')
                    if owned_listener(service.pid):
                        try:
                            response = client.get(cfg['api_base'] + '/models')
                            if response.is_success:
                                require([m['id'] for m in response.json()['data']] == [cfg['model']], 'Model identity mismatch')
                                ready = True
                                break
                        except httpx.TransportError:
                            pass
                    time.sleep(3)
                require(vr.read(folder / 'service/startup.json')['fixed_harmony_date'] == cfg['fixed_harmony_date'],
                        'Service date mismatch')
                vr.atomic(folder / 'execution_status.json', dict(state='running_fixed_pilot', unix=time.time(), pid=service.pid))
                with final.a.configured(OUT, tokenizer):
                    final.run_checks(OUT, plan, tokenizer, client, pilot=True)
        finally:
            release(service, record, folder, ready, url)
            if all(len(list((OUT / 'pilot' / arm).glob('*/result.json'))) == 6 for arm in final.ARMS):
                audit_results(plan, tokenizer, folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'confirm'))
    parser.add_argument('--attempt', required=True, type=int)
    args = parser.parse_args()
    require(args.attempt > 0 and socket.gethostname().split('.')[0] == 'node14', 'Invalid node14 attempt')
    set_environment()
    folder = CONTROL / 'attempts' / str(args.attempt)
    plan = final.verify(OUT)
    require(plan['configuration_sha256'] == CONFIG and plan['configuration']['pilot_sha256'] == PILOT,
            'Wrong frozen configuration or pilot')
    signature = vr.digest([CONFIG, PILOT, str(LOCK), vr.sha(Path(__file__)), vr.sha(CONTROL / 'environment.json')])
    if args.mode == 'confirm':
        final.confirm(folder, signature)
        return
    folder.mkdir(parents=True, exist_ok=False)
    code = 1
    def stop(_signal, _frame):
        raise InterruptedError('Supervisor received SIGTERM')
    signal.signal(signal.SIGTERM, stop)
    try:
        run(folder, plan, signature)
        vr.atomic(folder / 'execution_status.json', dict(state='passed', unix=time.time()))
        code = 0
    except BaseException as exc:
        vr.atomic(folder / 'execution_status.json', dict(state='stopped', unix=time.time(), error=str(exc)))
        raise
    finally:
        (folder / 'runner.exit_code').write_text(str(code))


if __name__ == '__main__':
    main()
