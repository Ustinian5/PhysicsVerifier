"""Bind the archived online configuration to the exact replay-only reader repair."""
import ast
import hashlib
from pathlib import Path
import subprocess

from rule_framework import verified_resume as vr

ROOT = Path(__file__).resolve().parents[1]
RUNNER = 'scripts/run_final_answer_revision.py'
ONLINE_COMMIT = '6e4598b4d55282417e29e23ebbd885865fcadccd'
ONLINE_RUNNER = '42b66bdf47270033260f17d8e475528c7d384632f4bd799ee458a7d628aca10c'
REPLAY_RUNNER = '86763eac00f9b33a806fc104e1665a830e357c90afe89b6ccb7497ebf8a85d4c'
CONFIG = '3482ba6eedd144a5984bb7f9cdf89a170cceb9d5c82546778d8de99a004bfc4f'
CHANGED = ('execution_plan', 'verify_arm')


def binding(out, plan, live):
    from scripts import run_final_answer_revision as f
    original = vr.read(out / 'execution_repair.json')
    f.require(live[RUNNER] == REPLAY_RUNNER and original['runner_after'] == ONLINE_RUNNER
              and original['configuration_sha256'] == CONFIG
              and original['parent_configuration_sha256'] == plan['configuration_sha256'] == f.PREPARED_CONFIGURATION
              and original['preparation_sha256'] == vr.sha(out / 'plan.json'), 'Unknown replay repair lineage')
    online = dict(live)
    online[RUNNER] = ONLINE_RUNNER
    f.require(vr.digest(dict(plan['configuration'], source_hashes=online)) == CONFIG,
              'Source drift outside the registered replay correction')
    return dict(configuration_sha256=CONFIG, preparation_sha256=original['preparation_sha256'],
                execution_repair_sha256=vr.sha(out / 'execution_repair.json'), online_runner_sha256=ONLINE_RUNNER,
                replay_runner_sha256=REPLAY_RUNNER, binding_code_sha256=vr.sha(Path(__file__)),
                changed_functions=list(CHANGED), scope='Frozen full-material key order on readback only')


def bound_generation_sources(out, plan, live):
    from scripts import run_final_answer_revision as f
    f.require(vr.read(out / 'replay_repair.json') == binding(out, plan, live), 'Changed replay repair binding')
    # Existing request hashes use the configuration that actually issued the requests.
    return {**live, RUNNER: ONLINE_RUNNER}


def register(out):
    from scripts import run_final_answer_revision as f
    archived = subprocess.check_output(['git', 'show', ONLINE_COMMIT + ':' + RUNNER], cwd=ROOT)
    f.require(hashlib.sha256(archived).hexdigest() == ONLINE_RUNNER, 'Archived online source drift')
    def unmodified_nodes(source):
        tree = ast.parse(source)
        changed = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in CHANGED]
        f.require(sorted(changed) == sorted(CHANGED), 'Unexpected replay repair structure')
        return [ast.dump(n) for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in CHANGED)]
    f.require(unmodified_nodes(archived) == unmodified_nodes((ROOT / RUNNER).read_bytes()),
              'Generation, retrieval, scoring or other runner code changed')
    receipt = binding(out, vr.read(out / 'plan.json'), f.source_hashes())
    f.immutable(out / 'replay_repair.json', receipt)
