"""Adopt the new judge at the library-to-evaluation boundary of the running November 26 release."""
import os
from pathlib import Path
import time
from unittest.mock import patch

import httpx

from scripts import run_reference_guided_verifier as workflow
from scripts import run_step_line_judge as line

node, vr, require = workflow.node, workflow.vr, workflow.require
ROOT = line.ROOT
ADOPTION = ROOT/'adoption.json'
PARENT = '737f3258bc9e44249d2d0b65a055e4f7a190849bfdeb72b299c34e97410eb3f0'
CHANGED = {'scripts/run_reference_guided_verifier.py', 'scripts/run_aligned_step_evaluation.py',
           'scripts/reuse_step_baseline.py'}
NEW = {'rule_framework/step_match_lines.py', 'scripts/run_step_line_judge.py', 'scripts/step_judge_handoff.py'}


def plan():
    original = vr.read(workflow.WORK/'workflow.json')
    parent = original['configuration']
    require(original['configuration_sha256'] == vr.digest(parent) == PARENT and
            parent['judge_location_policy'] == 'stop', 'Different original workflow')
    current = workflow.configuration('lines_once', parent['review_gpus'])
    require(set(current['code']) == set(parent['code']) | NEW and
            all(current['code'][n] == val for n,val in parent['code'].items() if n not in CHANGED) and
            dict(current, code=parent['code'], judge_location_policy='stop') == parent,
            'Amendment changes a construction, verifier or benchmark dependency')
    current.update(parent_workflow_sha256=PARENT, judge_protocol_sha256=vr.digest(line.contract()),
                   pure_published_diagnoses=63, pure_checker_requests=0, acceptance_threshold=None)
    return dict(configuration=current, configuration_sha256=vr.digest(current))


def verify():
    current = plan()
    require(vr.read(ADOPTION) == current, 'Amended judge workflow changed')
    return current


def boundary_marker(current):
    return dict(parent=PARENT, adoption_sha256=current['configuration_sha256'], after='accepted_libraries')


def prepare():
    node.node()
    with workflow.final.base.r.run_lock(ROOT/'adoption.lock'):
        current = plan()
        workflow.connect_stop(ROOT, workflow.WORK)
        step = workflow.WORK/'step'
        require(not (step/'plan.json').exists() and not (step/'STOP').is_symlink(), 'Old step phase already started')
        line.judge.write(ADOPTION, current)
        line.judge.write(step/'STOP', boundary_marker(current))
        line.bind(ROOT/'pure', line.publication(workflow.WORK/'step_pure'), ROOT)
        line.bind(ROOT/'controls', line.controls(), ROOT)
        line.audit_pure(ROOT, measured=False)


def take_boundary(current):
    step = workflow.WORK/'step'
    marker = step/'STOP'
    receipt = ROOT/'handoff.json'
    require(not (workflow.WORK/'STOP').exists(), 'Workflow was stopped by its owner')
    if not receipt.is_file():
        prior_exit = vr.read(workflow.WORK/'workflow_exit.json')
        prior_status = vr.read(workflow.WORK/'workflow_status.json')
        require(prior_exit['configuration_sha256'] == PARENT and prior_exit['code'] == 1 and
                prior_status['stage'] == 'stopped' and prior_status['reason'] == 'STOP requested' and
                not marker.is_symlink() and vr.read(marker) == boundary_marker(current) and
                not (step/'plan.json').exists(), 'Unexpected original supervisor exit')
        library = workflow.WORK/'libraries'
        workflow.step.released_libraries(library,{})
        line.judge.write(receipt, dict(boundary_marker(current), parent_exit=prior_exit, parent_status=prior_status,
                                      libraries_sha256=vr.sha(library/'construction_acceptance.json')))
    else:
        accepted = vr.read(receipt)
        require(all(accepted[k] == v for k,v in boundary_marker(current).items()) and
                accepted['libraries_sha256'] == vr.sha(workflow.WORK/'libraries/construction_acceptance.json'),
                'Unexpected previous handoff')
    if marker.exists() and not marker.is_symlink():
        require(vr.read(marker) == boundary_marker(current), 'Preserve the separately requested step stop')
        marker.unlink()


def measure_on_available_card(signature, endpoint):
    original = node.host.service_script
    def launcher(cfg):
        script, paths = original(cfg)
        require(script.count('18083') == 2, 'Unexpected judge launcher port')
        return script.replace('18083', str(line.PORT)), paths

    with patch.object(node.host, 'PORT', line.PORT), patch.object(node.host, 'service_script', launcher):
        while True:
            require(not (ROOT/'STOP').exists(), 'STOP requested')
            snapshot = node.host.resources()
            candidates = [v for v in snapshot['gpus'] if int(v[0]) in (5,6,7)]
            card = node.host.idle_card(dict(snapshot, gpus=candidates))
            if card and not node.host.sockets(['-Hltn']):
                started = False
                try:
                    with node.service(ROOT, int(card[0]), ROOT/'STOP', signature) as actual:
                        started = True
                        require(actual == endpoint, 'Unexpected judge endpoint')
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(node.host.HOT/'models/gpt-oss-120b', local_files_only=True)
                        with httpx.Client(trust_env=False) as client:
                            line.measure_pure(ROOT, endpoint=endpoint, client=client, tokenizer=tokenizer)
                    return
                except RuntimeError as exc:
                    # An exiting construction replica can still hold its lock after releasing its GPU memory.
                    if started or str(exc) != 'Another runner holds the experiment lock':
                        raise
            vr.atomic(ROOT/'progress.json', dict(state='waiting_for_free_judge_gpu', available_indices=[5,6,7],
                                                  unix=time.time()))
            time.sleep(15)


def bootstrap(signature):
    endpoint = f'http://127.0.0.1:{line.PORT}/v1'
    with workflow.final.base.r.run_lock(ROOT/'judgments.lock'):
        if not (ROOT/'pure/first_pass.json').is_file():
            measure_on_available_card(signature, endpoint)
        live_hash = vr.sha(ROOT/'pure/first_pass.json')
        replay = line.measure_pure(ROOT, endpoint=endpoint)
        line.audit_pure(ROOT, measured=True)
        require(live_hash == vr.sha(ROOT/'pure/first_pass.json'), 'Offline pure first-pass replay differs')
        release = vr.read(ROOT/'release.json')
        require(not release['owned_gpu_pids'] and not release['session_workers'] and not release['port_listeners'],
                'Incomplete judge service release')
        line.judge.write(ROOT/'pure/offline_replay.json', dict(first_pass_sha256=live_hash, new_model_calls=0,
                         unique_judgments=replay['unique']['total'], service_released=True))


def resume():
    node.node()
    with workflow.final.base.r.run_lock(ROOT/'supervisor.lock'):
        current = verify()
        signature = current['configuration_sha256']
        workflow.final.base.runtime.wait_detached(ROOT, signature)
        try:
            bootstrap(signature)
            if not (ROOT/'handoff.json').is_file():
                vr.atomic(ROOT/'progress.json',dict(state='pure_measured_waiting_for_accepted_libraries',unix=time.time()))
                parent = vr.read(workflow.WORK/'disconnect_acceptance.json')
                while True:
                    require(not (ROOT/'STOP').exists(), 'STOP requested')
                    try:
                        alive = workflow.final.base.runtime.process_identity(parent['pid']) == parent['process_start']
                        if alive:
                            alive = (Path(f"/proc/{parent['pid']}/stat").read_text().rsplit(') ',1)[1].split()[0] != 'Z')
                    except (FileNotFoundError, ProcessLookupError):
                        alive = False
                    if not alive:
                        break
                    time.sleep(10)
            vr.atomic(ROOT/'progress.json',dict(state='continuing_evaluations',pid=os.getpid(),unix=time.time()))
            return workflow.run(adopted=True)
        except BaseException as exc:
            vr.atomic(ROOT/'progress.json',dict(state='stopped',reason=str(exc),unix=time.time()))
            raise
