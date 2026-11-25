"""Run an isolated, detached timed verifier stage on an available node14 GPU."""
from contextlib import contextmanager
from unittest.mock import patch

from scripts import timed_retrieval_cohort as cohort

b, vr, require = cohort.original, cohort.vr, cohort.require
PORTS = {'case_rag':18250,'flat_skills':18251,'rule_tree':18252}
GPUS = {'case_rag':5,'flat_skills':6,'rule_tree':7}


@contextmanager
def large_service(folder, arm, signature, stop, *, rank_output=False):
    port = PORTS[arm]
    original_launcher = b.host.service_script
    def launcher(config):
        script, paths = original_launcher(config)
        require(script.count('18083') == 2, 'Unexpected launcher port')
        if rank_output:
            flag = '--disable-log-requests'
            require(script.count(flag) == 1, 'Unexpected model launcher')
            processor = 'rule_framework.vllm_rank_output:RankedIDsProcessor'
            script = script.replace(flag, f'--logits-processors {processor} {flag}')
        return script.replace('18083',str(port)), paths
    with patch.object(b.host,'PORT',port), patch.object(b.host,'service_script',launcher), b.node.service(
            folder,GPUS[arm],stop,signature) as endpoint:
        yield endpoint


def checking_plan(arm, signature):
    cfg = dict(vr.read(b.host.OUT/'plan.json')['configuration'],api_base=f'http://127.0.0.1:{PORTS[arm]}/v1')
    return dict(configuration=cfg, configuration_sha256=vr.digest([cfg,signature]))


def detached(directory, attempt, signature, action, confirm):
    b.node.node()
    folder = directory/'attempts'/str(attempt)
    if confirm:
        b.node.confirm(folder,signature)
        return
    folder.mkdir(parents=True,exist_ok=False)
    try:
        with b.final.base.r.run_lock(directory/'active.lock'):
            b.final.base.runtime.wait_detached(folder,signature)
            action(folder)
        vr.atomic(folder/'exit.json',dict(code=0))
    except Exception as exc:
        vr.atomic(folder/'exit.json',dict(code=1,error=str(exc)))
        raise
