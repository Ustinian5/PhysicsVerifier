"""Build the ordered tree and independent skills concurrently from the same frozen sources."""
import multiprocessing
import os

from scripts import reference_review_pool as pool
from scripts import run_corrected_pipeline as materials

vr, require = materials.vr, materials.require


def skill_worker(out, signature, gpu, offset, stride, stop, parent):
    folder = out/'replicas'/f'gpu{gpu}'
    folder.mkdir(parents=True, exist_ok=True)
    owner = pool.StopRequested(out, stop, parent)
    try:
        plan = vr.read(out/'plan.json')
        cfg = plan['configuration']
        require(vr.digest(cfg) == plan['configuration_sha256'] == signature and
                vr.sha(out/'cases.json') == cfg['frozen_files']['cases.json'], 'Different skill sources')
        cases = vr.read(out/'cases.json')[offset::stride]
        port = 18245 + gpu
        materials.immutable(folder/'assignment.json',dict(configuration_sha256=signature, gpu=gpu,port=port,
                                                          ids=[c['id'] for c in cases]))
        pending = [c for c in cases if not (out/'skill_outcomes'/(c['id']+'.json')).is_file()]
        if not pending:
            return
        with pool.replica_service(folder,gpu,port,owner,signature,cfg['transport']['fixed_date']) as endpoint:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(pool.node.host.HOT/'models/gpt-oss-120b',local_files_only=True)
            journal = materials.c.corpus.Journal(out,dict(cfg['transport'],auditor_api=endpoint),stop)
            for i,case in enumerate(pending):
                require(not owner.exists(), 'Skill worker stopped')
                materials.skill_outcome(out,cfg,case,journal,tokenizer)
                vr.atomic(folder/'progress.json',dict(total=len(cases),completed=len(cases)-len(pending)+i+1))
    except BaseException as exc:
        vr.atomic(folder/'worker_failure.json',dict(reason=str(exc),configuration_sha256=signature))
        stop.set()
        raise


def build(out, plan, gpus, inference):
    from transformers import AutoTokenizer
    cfg, sig = plan['configuration'], plan['configuration_sha256']
    date = cfg['transport']['fixed_date']
    ctx = multiprocessing.get_context('spawn')
    stop, workers = ctx.Event(), []
    try:
        for offset,gpu in enumerate(gpus[1:]):
            worker = ctx.Process(target=skill_worker,args=(out,sig,gpu,offset,len(gpus)-1,stop,os.getpid()))
            worker.start()
            workers.append(worker)
        tree_files = [out/'materials'/name for name in ('catalog.json','tree_report.json')]
        if len(gpus) == 1 or not all(p.is_file() for p in tree_files):
            tokenizer = AutoTokenizer.from_pretrained(pool.node.host.HOT/'models/gpt-oss-120b',local_files_only=True)
            with inference(out,date,sig):
                if len(gpus) == 1:
                    materials.build_materials(out,cfg,tokenizer)
                    return
                materials.build_tree(out,cfg,tokenizer,stop)
        for worker in workers:
            worker.join()
            require(worker.exitcode == 0, 'Skill worker failed; see its replica receipt')
        cases = vr.read(out/'cases.json')
        materials.publish_skills(out,cases,[materials.skill_outcome(out,cfg,c) for c in cases])
    finally:
        stop.set()
        for worker in workers:
            worker.join()
