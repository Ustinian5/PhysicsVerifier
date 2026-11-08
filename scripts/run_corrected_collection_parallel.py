"""Bounded batching around the unchanged frozen per-question generator."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
from pathlib import Path
import threading
import time
import httpx
from scripts import run_corrected_collection as base


def run(out, workers, limit):
    base.require(Path('/proc').exists() and workers in (1, 2, 4), 'Server only; bounded concurrency')
    with base.corpus.r.run_lock(out/'active.lock'), base.corpus.r.run_lock(out.parent/'generation.lock'):
        env=base.verify(out); c=env['configuration']['generator_config']; tasks=base.read(out/'tasks.json')
        stop=threading.Event(); journal=base.corpus.Journal(out,c,stop); saved={}; code=1
        (out/'runner.pid').write_text(str(os.getpid()))
        def publish(state):
            counts=Counter(); roles={'build':[], 'heldout':[]}
            for t in tasks:
                v=saved.get(t['row']['id'])
                if v is None: continue
                counts[t['role']+'/'+v['state']]+=1
                if v['state']=='complete': roles[t['role']].append(dict(t['row'],context='',prediction=v['prediction']))
            for role,rows in roles.items(): base.write(out/(role+'_inputs.json'), rows)
            base.write(out/'execution_status.json',dict(state=state,counts=dict(counts),processed=len(saved),total=len(tasks),workers=workers,updated_unix=time.time(),downstream_started=False))
        try:
            base.require(not(out/'STOP').exists(),'STOP requested')
            base.corpus.runtime.wait_detached(out,env['configuration_sha256'])
            with httpx.Client(trust_env=False,timeout=10) as client:
                response=client.get(c['generator_api']+'/models');response.raise_for_status()
                base.require(any(m['id']==c['generator'] and m.get('max_model_len',0)>=32768 and m.get('root')=='/slow_share2/huhengyuan/models/Qwen3-30B-A3B-Instruct-2507' for m in response.json()['data']), 'Service identity/context mismatch')
            pending=[]
            for t in tasks:
                if (out/'generated'/(t['row']['id']+'.json')).exists():
                    saved[t['row']['id']]=base.corpus.generate(journal,t['row'])
                else: pending.append(t)
            if limit: pending=pending[:limit]
            binding=dict(pid=os.getpid(),workers=workers,limit=limit,started_unix=time.time(),configuration_sha256=env['configuration_sha256'],runner_sha256=base.sha(Path(__file__)),reused=len(saved),task_ids=[t['row']['id'] for t in pending],policy='Frozen task order; unchanged request bodies and seeds; batches do not combine question contexts; no result selection.')
            base.write(out/'scheduling'/f'{os.getpid()}.json',binding)
            publish('generating')
            with ThreadPoolExecutor(max_workers=workers) as pool:
                queue=iter(pending); futures={}; errors=[]
                def fill():
                    while len(futures)<workers and not errors and not(out/'STOP').exists():
                        t=next(queue,None)
                        if t is None: break
                        futures[pool.submit(base.corpus.generate,journal,t['row'])]=t
                fill()
                while futures:
                    done,_=wait(futures,return_when=FIRST_COMPLETED)
                    for future in done:
                        t=futures.pop(future)
                        try: saved[t['row']['id']]=future.result()
                        except Exception as exc:
                            stop.set();errors.append(str(exc))
                    publish('draining_failure' if errors else 'generating')
                    fill()
                base.require(not errors,'Batch failure; in-flight responses preserved: '+ '; '.join(errors))
                base.require(not(out/'STOP').exists(),'STOP requested')
            publish('scheduling_pilot_complete' if limit and len(saved)<len(tasks) else 'generation_finished');code=0
        except BaseException as exc:
            base.write(out/'execution_status.json',dict(state='stopped',error=str(exc)[:1000],updated_unix=time.time(),workers=workers));raise
        finally:
            (out/'runner.exit_code').write_text(str(code));(out/'runner.pid').unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();base.require(a.limit>=0,'Negative limit');run(a.output.resolve(),a.workers,a.limit)

if __name__=='__main__':main()
