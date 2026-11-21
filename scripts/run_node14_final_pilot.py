"""Host the frozen final-answer pilot or formal batch on node14 with a private service."""
import argparse
from collections import Counter
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


def weight_location(config, folder, receipt_hash):
    if receipt_hash is None:
        return None
    receipt = CONTROL / 'weights.json'
    require(vr.sha(receipt) == receipt_hash, 'Weight transfer receipt changed')
    proof = vr.read(receipt)
    local = HOT / 'models/gpt-oss-120b'
    require(proof['verified'] and proof['source'] == config['model'] and proof['local'] == str(local),
            'Weight transfer belongs to another checkpoint or storage path')
    files = proof['files']
    require(set(files) == {p.name for p in local.iterdir()} and all((local / n).is_file() for n in files),
            'Missing or unexpected hot checkpoint files')
    require(set(vr.read(local / 'model.safetensors.index.json')['weight_map'].values()) == set(proof['shards'])
            == {n for n in files if n.endswith('.safetensors')}, 'Checkpoint shard inventory changed')
    require(all(files[n]['sha256'] == h for n, h in final.base.runtime.MODEL_FILES.items()), 'Model metadata drift')
    for name, entry in files.items():
        stat = (local / name).stat()
        require((stat.st_size, stat.st_mtime_ns) == (entry['bytes'], entry['mtime_ns']), 'Hot checkpoint stat drift')
    vr.EvidenceCache(folder / 'weight_cache.json').check(
        {str(local / n): v['sha256'] for n, v in files.items()}, full=True, workers=3)
    redirect = folder / 'model_location.json'
    final.immutable(redirect, {config['model']: str(local)})
    return redirect


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
    try:
        return bool(pids) and all(os.getpgid(p) == pid for p in pids)
    except ProcessLookupError:
        return False


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


def owns_gpu_process(pid, leader, run_root):
    try:
        # vLLM sets the worker title and can erase its original /proc environ.
        if os.getpgid(int(pid)) == leader and os.getsid(int(pid)) == leader:
            return True
        return os.fsencode('STEP_OSS_RUN_ROOT=' + str(run_root)) in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        return False


def session_workers(leader):
    result = subprocess.run(['ps', '--sid', str(leader), '-o', 'pid=,stat='], text=True, capture_output=True)
    require(result.returncode in (0, 1) and not result.stderr, 'Cannot inspect service session')
    workers = {}
    for line in result.stdout.splitlines():
        pid, state = line.split()
        if state.startswith('Z'):
            continue
        try:
            if os.getsid(int(pid)) == leader:
                workers[pid] = final.base.runtime.process_identity(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return workers


def release(service, record, folder, ready, url):
    idle = {}
    leader = service.pid
    run_root = folder / 'service'
    if service.poll() is None:
        pid = service.pid
        identity = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
        require(final.base.runtime.process_identity(pid) == record['start'] and os.getpgid(pid) == pid
                and os.fsencode('STEP_OSS_RUN_ROOT=' + str(folder / 'service')) in identity,
                'Service process identity changed')
    if owned_listener(leader) or (ready and service.poll() is None):
        while True:
            require(owned_listener(leader), 'Service port ownership changed')
            with httpx.Client(trust_env=False, timeout=10) as client:
                idle = request_counts(client, url)
            if not any(idle.values()) and not sockets(['-Htn', 'state', 'established']):
                break
            time.sleep(3)
    tracked = session_workers(leader)
    if tracked:
        try:
            os.killpg(leader, signal.SIGTERM)
        except ProcessLookupError:
            pass
    grace = time.monotonic() + 45
    forced = {}
    while True:
        service.poll()
        snapshot = resources()
        gpu_pids = [pid for uuid, pid in snapshot['compute_processes'] if uuid == record['gpu_uuid']]
        ours = []
        for pid in gpu_pids:
            if owns_gpu_process(pid, leader, run_root):
                ours.append(pid)
            elif pid in tracked:
                try:
                    if final.base.runtime.process_identity(pid) == tracked[pid]:
                        ours.append(pid)
                except (FileNotFoundError, ProcessLookupError):
                    ours.append(pid)  # The driver still lists the process that just exited.
        workers = session_workers(leader)
        tracked.update(workers)
        if not ours and not workers and not owned_listener(leader):
            break
        if time.monotonic() >= grace:
            for pid in set(ours).union(workers):
                try:
                    start = final.base.runtime.process_identity(pid)
                    owned = workers.get(pid) == start or owns_gpu_process(pid, leader, run_root)
                    if forced.get(pid) != start and owned and tracked.get(pid, start) == start:
                        os.kill(int(pid), signal.SIGKILL)
                        forced[pid] = start
                except (FileNotFoundError, ProcessLookupError):
                    continue
        time.sleep(2)
    vr.atomic(folder / 'release.json', dict(unix=time.time(), gauges=idle, exit_code=service.returncode,
              owned_gpu_pids=ours, session_workers=workers, tracked_processes=tracked, forced_pids=forced,
              gpu_snapshot=snapshot, port_listeners=sockets(['-Hltnp'])))


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
    final.immutable(folder / 'audit.json', dict(questions=[r['id'] for r in rows], groups=groups, requests=requests,
                    accepted=receipt.exists(), configuration_sha256=CONFIG, offline_model_calls=0))


def offline_replay(folder, confirm):
    from scripts import final_answer_replay_binding as repair
    control = folder / 'replay_validation'
    control.mkdir(exist_ok=True)
    acceptance = OUT / 'pilot_acceptance.json'
    signature = vr.digest([vr.sha(acceptance), vr.sha(OUT / 'execution_repair.json'),
                           vr.sha(final.a.ROOT / final.RUNNER), vr.sha(Path(repair.__file__)), vr.sha(Path(__file__))])
    if confirm:
        final.confirm(control, signature)
        return
    with final.base.r.run_lock(OUT / 'active.lock'):
        code = 1
        try:
            final.base.runtime.wait_detached(control, signature)
            frozen = vr.read(acceptance)
            require(frozen['configuration_sha256'] == CONFIG and frozen['passed'], 'Missing frozen native acceptance')
            require(all(vr.sha(Path(k)) == v for k, v in frozen['files'].items()), 'Original pilot evidence drift')
            repair.register(OUT)
            plan = final.verify(OUT)
            require(plan['configuration_sha256'] == CONFIG, 'Archived configuration changed')
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(plan['configuration']['model'], local_files_only=True,
                                                       trust_remote_code=False)
            audit_results(plan, tokenizer, folder)
            def no_request(_):
                raise AssertionError('Offline replay attempted a model call')
            mismatches = {arm: [] for arm in final.ARMS}
            rows = vr.read(OUT / 'pilot_inputs.json')
            with httpx.Client(transport=httpx.MockTransport(no_request)) as client:
                for row in rows:
                    key = vr.digest(row['id'])[:24]
                    prefix = vr.read(OUT / 'pilot/queries' / key / 'result.json')
                    for arm in final.ARMS:
                        packet = vr.read(OUT / 'pilot' / arm / key / 'reference_packet.json')
                        legacy = final.p.continuation_messages(row, prefix['value'], packet['references'], arm=arm)
                        if legacy != packet['messages']:
                            mismatches[arm].append(row['id'])
                    final.run_question(OUT, final.p.public(row), 'pilot', plan, tokenizer, client, {})
            require(all(vr.sha(Path(k)) == v for k, v in frozen['files'].items()), 'Pilot changed during offline replay')
            final.immutable(control / 'validation.json', dict(passed=True, archived_evidence_files=len(frozen['files']),
                            native_acceptance_sha256=vr.sha(acceptance), replay_binding_sha256=vr.sha(OUT / 'replay_repair.json'),
                            results_sha256=vr.sha(folder / 'audit.json'), legacy_material_order_mismatches=mismatches,
                            resumed_arm_results=len(rows)*len(final.ARMS), new_model_calls=0))
            vr.atomic(control / 'execution_status.json', dict(state='passed', unix=time.time()))
            code = 0
        except BaseException as exc:
            vr.atomic(control / 'execution_status.json', dict(state='stopped', unix=time.time(), error=str(exc)))
            raise
        finally:
            (control / 'validation.exit_code').write_text(str(code))


def batch_inputs(out, config, weights_hash):
    """Bind the accepted pilot, its replay and the labels authorized for this attempt."""
    attempt = out / 'node14_pilot/attempts/2'
    acceptance = out / 'pilot_acceptance.json'
    replay = attempt / 'replay_validation/validation.json'
    accepted, validation = vr.read(acceptance), vr.read(replay)
    require(accepted['passed'] and accepted['configuration_sha256'] == config
            and all(vr.sha(Path(p)) == h for p, h in accepted['files'].items()), 'Frozen pilot evidence changed')
    require(validation['passed'] and validation['resumed_arm_results'] == 24
            and validation['archived_evidence_files'] == len(accepted['files'])
            and validation['native_acceptance_sha256'] == vr.sha(acceptance)
            and validation['replay_binding_sha256'] == vr.sha(out / 'replay_repair.json')
            and validation['results_sha256'] == vr.sha(attempt / 'audit.json'), 'Pilot replay evidence changed')
    deployment = vr.read(attempt / 'deployment.json')
    require(deployment['configuration_sha256'] == config and weights_hash
            and deployment['weights_receipt_sha256'] == weights_hash
            and deployment['environment_sha256'] == vr.sha(out / 'node14_pilot/environment.json'),
            'Formal weights or environment differ from the accepted pilot')
    require(not vr.read(out / 'isolation_candidates.json')['candidates'], 'Unresolved source overlap')
    decisions = out / 'assistant_review.json'
    final.read_overrides(out, vr.read(out / 'candidates.json'))
    inputs = dict(pilot=vr.sha(acceptance), replay=vr.sha(replay), adjudications=vr.sha(decisions))
    amendment = out / final.cohort.AMENDMENT
    if amendment.is_file():
        inputs['cohort_amendment'] = vr.sha(amendment)
    return inputs


def batch_stage(out, inputs):
    require(vr.sha(out / 'assistant_review.json') == inputs['adjudications'],
            'Assistant adjudications changed during this attempt')
    amendment = out / final.cohort.AMENDMENT
    require((vr.sha(amendment) if amendment.is_file() else None) == inputs.get('cohort_amendment'),
            'Cohort amendment changed during this attempt')
    if not (out / 'review_controls_passed.json').is_file():
        return 'review'
    decisions = final.dispositions(out)
    counts = Counter(d['state'] for d in decisions.values())
    if counts['pending_review']:
        return 'review'
    if counts['pending_adjudication']:
        final.base.old.emit(out, 'awaiting_adjudication', counts=dict(counts),
                            pending=[sid for sid, d in decisions.items() if d['state'] == 'pending_adjudication'])
        return 'awaiting_adjudication'
    eligible = Counter(d['verdict'] for d in decisions.values() if d['state'] == 'eligible')
    final.base.old.emit(out, 'freezing_benchmark', counts=dict(counts), eligible_by_label=dict(eligible))
    final.freeze_bench(out)
    final.rescore_archive(out)
    if out.name == 'final_revision_balanced':
        final.immutable(out / 'prior_request_reuse.json', final.cohort.reuse_prior_requests(out))
    if (out / 'metrics.json').is_file():
        final.score(out)
        return 'completed'
    return 'formal'


def run(folder, plan, signature, weights_hash, *, batch=None):
    from transformers import AutoTokenizer

    cfg = plan['configuration']
    require(socket.gethostname().split('.')[0] == 'node14' and Path(sys.prefix) == ENV, 'Wrong host or conda prefix')
    require(final.a.ROOT == ROOT and cfg['api_base'] == f'http://127.0.0.1:{PORT}/v1', 'Wrong checkout or endpoint')
    if batch is None:
        require(not (OUT / 'pilot_acceptance.json').exists(), 'Fixed pilot already accepted')
    env_receipt = vr.read(CONTROL / 'environment.json')
    packages = vr.read(OUT.parent / 'evaluation/runtime.json')['packages']
    require(env_receipt['verified'] and env_receipt['prefix'] == str(ENV)
            and {k: importlib.metadata.version(k) for k in packages} == packages, 'Runtime differs')
    final.a.historical_transport()
    script, paths = service_script(cfg)
    url = cfg['api_base'].removesuffix('/v1')
    with final.base.r.run_lock(OUT / 'active.lock'), final.base.r.run_lock(LOCK):
        final.base.runtime.wait_detached(folder, signature)
        require(not (OUT / 'STOP').exists(), 'STOP requested')
        phase = 'pilot'
        if batch is not None:
            phase = batch_stage(OUT, batch)
            if phase in ('awaiting_adjudication', 'completed'):
                return phase
        vr.atomic(folder / 'execution_status.json', dict(state='verifying_weights', stage=phase, unix=time.time()))
        redirect = weight_location(cfg, folder, weights_hash)
        tokenizer = AutoTokenizer.from_pretrained(cfg['model'], local_files_only=True, trust_remote_code=False)
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
        env.pop('VLLM_MODEL_REDIRECT_PATH', None)
        if redirect is not None:
            env['VLLM_MODEL_REDIRECT_PATH'] = str(redirect)
        with (folder / 'service.log').open('ab') as log:
            service = subprocess.Popen(['/bin/bash', '-c', script], cwd=ROOT, env=env, start_new_session=True,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        ready = False
        record = dict(pid=service.pid, start=final.base.runtime.process_identity(service.pid), gpu_uuid=uuid,
                      gpu_index=index, preflight=snapshot, port=PORT, source_lock=cfg['service_lock'],
                      active_lock=str(LOCK), source_launcher_sha256=cfg['source_hashes'][LAUNCHER],
                      wrapper_sha256=vr.sha(Path(__file__)), infrastructure_paths=paths,
                      configuration_sha256=plan['configuration_sha256'], environment_sha256=vr.sha(CONTROL / 'environment.json'),
                      weights_receipt_sha256=weights_hash, storage_redirect=vr.read(redirect) if redirect else {})
        if batch is not None:
            record.update(batch_inputs=batch, result_lock=str(OUT / 'active.lock'), start_stage=phase)
        try:
            vr.atomic(folder / 'deployment.json', record)
            vr.atomic(folder / 'execution_status.json', dict(state='starting_service', unix=time.time(), pid=service.pid))
            last_progress = time.monotonic()
            log_stamp = None
            with httpx.Client(trust_env=False, timeout=10) as client:
                while True:
                    require(service.poll() is None, 'Service exited before readiness; see service.log')
                    stamp = (folder / 'service.log').stat().st_mtime_ns
                    if stamp != log_stamp:
                        last_progress, log_stamp = time.monotonic(), stamp
                    require(time.monotonic() - last_progress < 3600, 'No startup progress for one hour')
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
                with final.a.configured(OUT, tokenizer):
                    vr.atomic(folder / 'execution_status.json', dict(state='running_' + phase, unix=time.time(), pid=service.pid))
                    if phase == 'review':
                        final.run_reviews(OUT, plan, tokenizer, client)
                        phase = batch_stage(OUT, batch)
                        if phase == 'awaiting_adjudication':
                            return phase
                        require(phase == 'formal', 'Candidate review did not release a formal cohort')
                        vr.atomic(folder / 'execution_status.json', dict(state='running_formal', unix=time.time(), pid=service.pid))
                    final.run_checks(OUT, plan, tokenizer, client, pilot=phase == 'pilot')
                    return 'passed' if phase == 'pilot' else 'completed'
        finally:
            release(service, record, folder, ready, url)
            if batch is None and all(len(list((OUT / 'pilot' / arm).glob('*/result.json'))) == 6 for arm in final.ARMS):
                audit_results(plan, tokenizer, folder)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'confirm', 'audit', 'confirm-audit', 'batch', 'confirm-batch'))
    parser.add_argument('--attempt', required=True, type=int)
    parser.add_argument('--output', type=Path, help='Server result root for the separately adjudicated release')
    args = parser.parse_args()
    require(args.attempt > 0 and socket.gethostname().split('.')[0] == 'node14', 'Invalid node14 attempt')
    set_environment()
    is_batch = args.mode in ('batch', 'confirm-batch')
    if args.output is not None:
        expected = HOT / 'results/qwen30b_step_verifier/aligned_old99/final_revision_balanced'
        require(is_batch and args.output.resolve() == expected and final.cohort.read_amendment(expected).get('supersedes'),
                'Unknown amended batch result root')
        OUT = expected
    folder = (OUT / 'node14_formal' if is_batch else CONTROL) / 'attempts' / str(args.attempt)
    if args.mode in ('audit', 'confirm-audit'):
        offline_replay(folder, confirm=args.mode == 'confirm-audit')
        return
    plan = final.verify(OUT)
    require(plan['configuration_sha256'] == CONFIG and plan['configuration']['pilot_sha256'] == PILOT,
            'Wrong frozen configuration or pilot')
    weights = CONTROL / 'weights.json'
    weights_hash = vr.sha(weights) if weights.is_file() else None
    binding = [CONFIG, PILOT, str(LOCK), vr.sha(Path(__file__)), vr.sha(CONTROL / 'environment.json'), weights_hash]
    batch = batch_inputs(OUT, plan['configuration_sha256'], weights_hash) if is_batch else None
    if is_batch:
        binding.extend(['batch', batch, str(folder)])
    signature = vr.digest(binding)
    if args.mode in ('confirm', 'confirm-batch'):
        final.confirm(folder, signature)
        return
    folder.mkdir(parents=True, exist_ok=False)
    code = 1
    def stop(_signal, _frame):
        raise InterruptedError('Supervisor received SIGTERM')
    signal.signal(signal.SIGTERM, stop)
    try:
        state = run(folder, plan, signature, weights_hash, batch=batch)
        if state == 'completed' and is_batch and batch.get('cohort_amendment'):
            from scripts.analyze_final_answer_sample_size import run as measure_stability
            measure_stability(OUT)
        vr.atomic(folder / 'execution_status.json', dict(state=state, unix=time.time()))
        code = 0
    except BaseException as exc:
        vr.atomic(folder / 'execution_status.json', dict(state='stopped', unix=time.time(), error=str(exc)))
        raise
    finally:
        (folder / 'runner.exit_code').write_text(str(code))


if __name__ == '__main__':
    main()
