"""Run paired, isolated BM25F CHECKs using the already replayed native packets."""
import argparse
from pathlib import Path
from unittest.mock import patch
import time

import httpx
from scripts import run_bm25f_retrieval as replay
from scripts import step_check_format as fmt
from scripts import run_step_line_judge as lines
from scripts import step_judge_release as release

st, final, vr = replay.structured, replay.final, replay.vr
write = final.immutable
base = replay.step


def run(kind, gpu, out, stamp):
    from transformers import AutoTokenizer
    root = st.OUT if kind=='final' else replay.BASE/'step_formal'
    config = vr.read(root/'plan.json')['configuration']
    port = 18256 if kind=='final' else lines.PORT
    cfg = dict(config,api_base=f'http://127.0.0.1:{port}/v1')
    plan = dict(configuration=cfg,configuration_sha256=stamp)
    native_rows = {r['id']:r for r in vr.read(root/'inputs.json')}
    ids = vr.read(replay.OUT/'cohort.json')[root.name]
    tok = AutoTokenizer.from_pretrained(st.node.host.HOT/'models/gpt-oss-120b',local_files_only=True)
    published = {'bm25':{},'bm25f':{}}
    from contextlib import ExitStack
    with ExitStack() as stack:
        client = stack.enter_context(httpx.Client(trust_env=False))
        if kind=='final':
            stack.enter_context(patch.object(st,'OUT',out))
            stack.enter_context(patch.object(st,'PORT',port))
            stack.enter_context(st.constrained_transport())
            stack.enter_context(st.service(plan,gpu))
        else:
            url=stack.enter_context(st.pool.replica_service(out,gpu,port,out/'STOP',stamp,cfg['fixed_harmony_date']))
            assert url==cfg['api_base']
            stack.enter_context(fmt.step.a.configured(out,tok))
            stack.enter_context(fmt.checks(out,dict(phase='unused',group='',id=''),None))
        for i,sid in enumerate(ids,1):
            row=native_rows[sid]
            key=vr.digest(sid)[:24]
            for arm in replay.ARMS:
                assert not (out/'STOP').exists() and not (replay.OUT/'STOP').exists()
                folder=replay.OUT/root.name/sid/arm
                receipt=vr.read(folder/'receipt.json')
                path=(root/'formal'/arm/key/'result.json' if kind=='final' else base.path_for(root,'formal',arm,row))
                original=vr.read(path)
                published['bm25'][sid,arm]=path
                current=path
                if receipt.get('changed'):
                    assert vr.sha(folder/'packet.json')==receipt['packet_sha256']
                    packet=vr.read(folder/'packet.json')
                    if kind=='final':
                        prefix=vr.read(root/'formal/queries'/key/'result.json')
                        archived=vr.read(path.parent/'check/request.json')['request']
                        cap=archived['max_tokens']
                        limits=cfg['limits']
                        count=len(tok.apply_chat_template(packet['messages'],tokenize=True,add_generation_prompt=True,reasoning_effort='medium'))
                        assert count+prefix['usage']['prompt_tokens']<=limits['input_tokens_per_sample']
                        assert cap==min(limits['output_tokens_per_sample']-prefix['usage']['completion_tokens'],limits['max_tokens_per_response'])
                        dest=out/'checks'/sid/arm/'check'
                        final.request_once(out,dest,packet['messages'],cap,final.p.original.validate_verdict,plan,tok,client,
                                           available=limits['wall_seconds_per_sample']-prefix['wall_seconds'])
                        actual=vr.read(dest/'request.json')['request']
                        assert {k:v for k,v in actual.items() if k!='messages'}=={k:v for k,v in archived.items() if k!='messages'}
                        current=dest/'result.json'
                    else:
                        prefix=vr.read(root/'queries'/key/'result.json')
                        with patch.object(base,'query',return_value=prefix),patch.object(base.p,'reference_packet',return_value=packet):
                            fmt.step.check_published(out,row,'formal',arm,None,plan,tok,client)
                        current=base.path_for(out,'formal',arm,row)
                        old=vr.read(path.parent/'turn_02/request.json')['request']
                        new=vr.read(current.parent/'turn_02/request.json')['request']
                        assert new['messages']==packet['messages'] and {k:v for k,v in new.items() if k!='messages'}=={k:v for k,v in old.items() if k!='messages'}
                elif 'retained_failure' in receipt:
                    assert vr.sha(path)==receipt['retained_failure']
                published['bm25f'][sid,arm]=current
                vr.atomic(out/'progress.json',dict(question=i,arm=arm,id=sid))
        if kind=='step':
            rows,_=lines.judge.benchmark(Path(cfg['step_benchmark']))
            rows=[r for r in rows if r['id'] in ids]
            shared=out/'judge_cache'
            official=replay.BASE/'step_judge_lines'
            for mode,items in published.items():
                results={}
                for k,p in items.items():
                    v=vr.read(p)
                    results[k]=dict(sha256=vr.sha(p),source=str(p),failed=not base.runtime.healthy_result(v),
                                    status=v['status'],diagnostics=v['diagnostics'])
                data=(rows,results,dict(evaluated_questions=len(rows),evaluated_active_roots=sum(len(r['gt']) for r in rows)),
                      dict(arms=list(replay.ARMS),cohort_sha256=vr.sha(replay.OUT/'cohort.json')))
                dest=out/'judge'/mode
                _,_,jobs=lines.bind(dest,data,shared)
                for key in jobs:
                    prior=official/'judgments'/key
                    link=shared/'judgments'/key
                    if (prior/'result.json').is_file() and not link.exists():
                        link.parent.mkdir(parents=True,exist_ok=True)
                        link.symlink_to(prior,target_is_directory=True)
                report=lines.evaluate(dest,data,shared,endpoint=cfg['api_base'],tokenizer=tok,client=client)
                write(dest/'acceptance.json',release.assess(report))
    if kind=='final':
        labels=vr.read(root/'labels_private.json')
        truth={i:labels[i]['verdict'] for i in ids}
        scores={mode:{arm:final.measure(truth,{i:vr.read(items[i,arm]) for i in ids}) for arm in replay.ARMS}
                for mode,items in published.items()}
        write(out/'metrics.json',scores)
    write(out/'complete.json',dict(questions=len(ids),configuration_sha256=stamp))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('run','confirm'));p.add_argument('kind',choices=('final','step'))
    p.add_argument('gpu',type=int,choices=(5,6,7));args=p.parse_args()
    st.node.node()
    out=replay.OUT/(args.kind+'_checks');out.mkdir(parents=True,exist_ok=True)
    stamp=vr.digest([vr.sha(__file__),vr.sha(replay.__file__),vr.sha(replay.OUT/'cohort.json'),args.kind])
    if args.action=='confirm':
        st.node.confirm(out,stamp)
    else:
        with base.r.run_lock(out/'active.lock'):
            base.runtime.wait_detached(out,stamp)
            while not (replay.OUT/'cpu_complete.json').exists():
                assert not (out/'STOP').exists() and not (replay.OUT/'STOP').exists()
                time.sleep(3)
            assert vr.read(replay.OUT/'cpu_complete.json')['script_sha256']==vr.sha(replay.__file__)
            run(args.kind,args.gpu,out,stamp)
