"""Distribute independent frozen first reviews across privately owned node14 services."""
import multiprocessing
import os
from contextlib import contextmanager
from unittest.mock import patch

from scripts import run_corrected_extraction as extract
from scripts import serve_frozen_bm25 as node


class StopRequested:
    def __init__(self, out, aborted, parent):
        self.file, self.aborted, self.parent = out/'STOP', aborted, parent

    def exists(self):
        return self.file.exists() or self.aborted.is_set() or os.getppid() != self.parent


@contextmanager
def replica_service(folder, gpu, port, stop, signature, date):
    original = node.host.service_script

    def launcher(config):
        script, paths = original(config)
        node.require(script.count('18083') == 2, 'Unexpected frozen service port')
        return script.replace('18083', str(port)), paths

    with patch.object(node.host, 'PORT', port), patch.object(node.host, 'service_script', launcher), \
            node.service(folder, gpu, stop, signature, fixed_date=date) as url:
        yield url


def replica(out, gpu, positions, gate, aborted, parent, signature):
    folder = out/'replicas'/f'gpu{gpu}'
    folder.mkdir(parents=True, exist_ok=True)
    stop = StopRequested(out, aborted, parent)
    try:
        plan = extract.verify(out)
        node.require(plan['configuration_sha256'] == signature, 'Different replica construction plan')
        cfg = plan['configuration']
        rows = extract.read(out/'inputs.json')
        taxonomy = extract.read(out/'taxonomy.json')
        assigned = [rows[i]['id'] for i in positions]
        port = 18245 + gpu
        node.host.final.immutable(folder/'assignment.json',
                                  dict(ids=assigned, gpu=gpu, port=port, configuration_sha256=signature))
        if all((out/'experiences'/(sid+'.json')).is_file() for sid in assigned):
            return
        with replica_service(folder, gpu, port, stop, signature, cfg['fixed_date']) as url:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(node.host.HOT/'models/gpt-oss-120b', local_files_only=True)
            journal = extract.c.corpus.Journal(out, dict(cfg, auditor_api=url), aborted)
            for count, position in enumerate(positions, 1):
                if position >= cfg['pilot_size']:
                    while not gate.wait(0.5):
                        node.require(not stop.exists(), 'First-review pilot stopped')
                node.require(not stop.exists(), 'First-review replica stopped')
                row = rows[position]
                extract.review_one(out, row, taxonomy, cfg, journal, tokenizer)
                extract.write(folder/'progress.json', dict(completed=count, total=len(assigned), last_id=row['id']))
    except BaseException as exc:
        extract.write(folder/'failure.json', dict(error=str(exc), pid=os.getpid()))
        aborted.set()
        raise


def ordered_reviews(out, rows, taxonomy, cfg, journal, tokenizer, *, gpus):
    if len(gpus) == 1:
        yield from extract.serial_reviews(out, rows, taxonomy, cfg, journal, tokenizer)
        return
    ctx = multiprocessing.get_context('spawn')
    gate, aborted = ctx.Event(), ctx.Event()
    workers = {}
    try:
        for offset, gpu in enumerate(gpus[1:], 1):
            worker = ctx.Process(target=replica, args=(out, gpu, range(offset, len(rows), len(gpus)),
                                 gate, aborted, os.getpid(), extract.digest(cfg)))
            worker.start()
            workers[gpu] = worker
        for index, row in enumerate(rows):
            gpu = gpus[index % len(gpus)]
            if gpu != gpus[0]:
                path = out/'experiences'/(row['id']+'.json')
                while not path.exists():
                    node.require(workers[gpu].is_alive() and not aborted.wait(0.5) and not (out/'STOP').exists(),
                                 f'First-review replica interrupted. See replicas/gpu{gpu}')
            node.require(not aborted.is_set(), 'First-review replica failed. See replicas')
            yield extract.review_one(out, row, taxonomy, cfg, journal, tokenizer, saved_only=gpu != gpus[0])
            if index + 1 == cfg['pilot_size']:
                receipt = extract.read(out/'pilot_acceptance.json')
                node.require(receipt['passed'] and receipt['configuration_sha256'] == extract.digest(cfg),
                             'Fixed first-review pilot has not passed')
                gate.set()
        for gpu, worker in workers.items():
            worker.join()
            node.require(worker.exitcode == 0, f'First-review replica GPU{gpu} failed')
    finally:
        aborted.set()
        for worker in workers.values():
            worker.join()
