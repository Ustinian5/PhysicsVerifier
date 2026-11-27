"""Run the fixed six and, after acceptance, the frozen 467-question comparison."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import shlex
import time
from unittest.mock import patch

import httpx

from rule_framework import final_output_format as fmt
from scripts import reference_review_pool as pool
from scripts import run_final_answer_revision as final
from scripts import serve_frozen_bm25 as node

vr, require = final.vr, final.require
WORK = node.host.HOT/'results/qwen30b_step_verifier/english_blind_verifier/formal'
PREVIOUS = WORK/'final'
OUT = WORK/'final_structured'
PORT = 18250
PLUGIN = Path(fmt.__file__).with_name('vllm_final_reasoning.py')
AUDIT = '39353f275813b2ed7eeec6fd1aca8573c3682d7edc1382689d8818c5cb603e28'
LIBRARIES = '848a86142eb6359173d92b42c7e56d071755200f1c199e39eb1ec6de60163360'


def configuration():
    original = final.verify(PREVIOUS)['configuration']
    require(vr.sha(PREVIOUS/'pilot_schema_audit.json') == AUDIT and
            vr.sha(WORK/'libraries/construction_acceptance.json') == LIBRARIES, 'Different pilot or libraries')
    return dict(original, api_base=f'http://127.0.0.1:{PORT}/v1',
                output_formats=dict(query=fmt.QUERY, verdict=fmt.VERDICT, policy='current_harmony_final_body',
                                    backend='xgrammar', parent_plan_sha256=vr.sha(PREVIOUS/'plan.json'),
                                    pilot_audit_sha256=AUDIT, library_acceptance_sha256=LIBRARIES,
                                    code={p.name: vr.sha(p) for p in (Path(__file__),Path(fmt.__file__),PLUGIN,
                                                                      Path(pool.__file__),Path(node.__file__))}))


def verify():
    plan = final.verify(OUT)
    require(plan['configuration'] == configuration(), 'Structured output release changed')
    return plan


def prepare():
    from scripts.run_reference_guided_verifier import connect_stop
    with final.base.r.run_lock(OUT/'active.lock'):
        connect_stop(OUT, WORK)
        cfg = configuration()
        for name, digest in cfg['released_files'].items():
            src = (PREVIOUS/(name+'.json')).resolve(strict=True)
            link = OUT/(name+'.json')
            if not link.is_symlink():
                link.symlink_to(src)
            require(link.resolve(strict=True) == src and vr.sha(link) == digest, 'Changed released input')
        final.immutable(OUT/'plan.json', dict(configuration=cfg,configuration_sha256=vr.digest(cfg)))
        return verify()


def schema_for(folder):
    if folder.parent.name == 'queries':
        return fmt.QUERY
    require(folder.name == 'check', 'Unexpected question-level call')
    return fmt.VERDICT


@contextmanager
def constrained_transport():
    transport = final.base.runtime.transport
    ordinary_send, ordinary_saved = transport.chat_response, final.saved_request
    build = transport.r.request_body

    def send(*args, **kwargs):
        file = Path(kwargs['request_path'])
        schema = schema_for(file.parent)
        def request(*args, **kwargs):
            return dict(build(*args, **kwargs), structured_outputs={'json': schema})
        with patch.object(transport.r, 'request_body', request):
            result = ordinary_send(*args, **kwargs)
        require(vr.read(file)['request'].get('structured_outputs') == {'json':schema}, 'Missing generation grammar')
        return result

    def saved(folder, *args, **kwargs):
        require(vr.read(folder/'request.json')['request'].get('structured_outputs') == {'json':schema_for(folder)},
                'Stored request has a different output schema')
        return ordinary_saved(folder,*args,**kwargs)

    with patch.object(transport,'chat_response',send), patch.object(final,'saved_request',saved):
        yield


@contextmanager
def service(plan, gpu):
    original = node.host.service_script
    def launcher(cfg):
        script, paths = original(cfg)
        require(script.rstrip().endswith('--disable-log-requests'), 'Different verifier service launcher')
        flags = (' --structured-outputs-config ' + shlex.quote('{"backend":"xgrammar","enable_in_reasoning":false}')
                 + ' --reasoning-parser-plugin ' + shlex.quote(str(PLUGIN)) + ' --reasoning-parser pv_final_harmony')
        return script.rstrip() + flags + '\n', paths
    with patch.object(node.host,'service_script',launcher), pool.replica_service(
            OUT,gpu,PORT,OUT/'STOP',plan['configuration_sha256'],plan['configuration']['fixed_harmony_date']) as url:
        require(url == plan['configuration']['api_base'], 'Different verifier endpoint')
        yield


def run(gpu):
    from transformers import AutoTokenizer
    with final.base.r.run_lock(OUT/'active.lock'):
        plan = verify()
        token = plan['configuration_sha256']
        code = 1
        try:
            final.base.runtime.wait_detached(OUT,token)
            with constrained_transport():
                if not (OUT/'metrics.json').is_file():
                    tok = AutoTokenizer.from_pretrained(node.host.HOT/'models/gpt-oss-120b',local_files_only=True)
                    with service(plan,gpu), httpx.Client(trust_env=False) as client:
                        if not (OUT/'pilot_acceptance.json').is_file():
                            final.run_checks(OUT,plan,tok,client,pilot=True)
                        final.run_checks(OUT,plan,tok,client,pilot=False)
                final.score(OUT)
            code = 0
        except BaseException as exc:
            final.base.old.emit(OUT,'stopped',reason=str(exc))
            raise
        finally:
            vr.atomic(OUT/'structured_exit.json',dict(code=code,configuration_sha256=token,unix=time.time()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare','confirm','run','score'))
    p.add_argument('--gpu',type=int,choices=(4,5,6,7),default=5)
    args = p.parse_args()
    node.node()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'confirm':
        node.confirm(OUT,verify()['configuration_sha256'])
    elif args.command == 'score':
        verify()
        with constrained_transport():
            final.score(OUT)
    else:
        run(args.gpu)


if __name__ == '__main__':
    main()
