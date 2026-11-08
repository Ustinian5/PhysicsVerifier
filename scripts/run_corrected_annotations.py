"""Independent Sol high annotation of the completed, corrected 30B cohort."""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from dotenv import load_dotenv
from scripts import run_corrected_step_generation as generation
from scripts import run_bench_reannotation as teacher

read,write,require,sha,digest=generation.read,generation.write,generation.require,generation.sha,generation.digest
ROOT=generation.ROOT
SOURCES=('scripts/run_corrected_annotations.py','scripts/run_bench_reannotation.py',
         'scripts/bench_reannotation_runtime.py','scripts/run_retrieval_comparison.py','scripts/response_stream.py')


def prepare(out,bench,screen,inventory,env):
    require(not(out/'binding.json').exists(),'Already prepared')
    parent=generation.verify(bench);c=parent['configuration'];rows=read(bench/'inputs.json')
    require(len(rows)==100 and read(bench/'execution_status.json')['state']=='generation_complete','100 answers incomplete')
    questions={r['id']:r for r in read(bench/'questions.json')}
    imports={r['id']:r for r in c['imports']}
    ids={generation.question_key(r['question']):r['id'] for r in read(inventory)['questions']}
    refs={r['id']:r['variants'] for r in read(screen/'references_private.json')['references']}
    labeled=[];evidence={};reference_bindings=[]
    for row in rows:
        sid=row['id'];q=questions[sid]
        path=bench/imports[sid]['record'] if sid in imports else bench/'generated'/(sid+'.json')
        saved=read(path);rawpath=bench/imports[sid]['evidence'] if sid in imports else bench/saved['evidence']['file']
        if sid not in imports:
            require(saved['signature']==digest([q,c['configuration']]) and sha(rawpath)==saved['evidence']['sha256'],
                    'Generated answer binding changed')
        generation.validate_import(q,c['configuration'],saved,read(rawpath))
        require(row['question']==q['question'] and row['prediction']==saved['prediction'] and not row.get('context'),'Export differs from answer')
        evidence[os.path.relpath(path,out)]=sha(path);evidence[os.path.relpath(rawpath,out)]=sha(rawpath)
        source_id=ids[generation.question_key(q['question'])];variants=refs[source_id]
        if len(variants)>1:
            require([v['value'] for v in variants]==[['0.28 m/s'],['0.28']], 'Unreviewed multiple step references')
            reason='Same numerical value; first original variant explicitly supplies velocity units; no reference rewriting.'
        else:reason='Single original reference variant.'
        chosen=variants[0]['value']
        require(isinstance(chosen,list) and chosen and all(isinstance(x,str) for x in chosen)
                and any(x.strip() for x in chosen),'Missing original reference')
        labeled.append(dict(id=sid,question=q['question'],context='',prediction=row['prediction'],answer=json.dumps(chosen,ensure_ascii=False)))
        reference_bindings.append(dict(id=sid,source_id=source_id,variants=variants,selected_variant=0,reason=reason,
            partial_reference=any(not x.strip() for x in chosen),human_gold=False))
    write(out/'annotation_inputs.json',labeled);write(out/'reference_bindings.json',reference_bindings)
    load_dotenv(env,override=False)
    teacher.prepare(out/'annotation_inputs.json',out,os.environ['OPENAI_BASE_URL'],expected_count=100,pilot_size=100)
    binding=dict(protocol='corrected_sol_step_annotations_v1',human_gold=False,
        parent_plan_sha256=sha(bench/'plan.json'),reference_index_sha256=sha(screen/'references_private.json'),
        inputs_sha256=sha(out/'annotation_inputs.json'),reference_bindings_sha256=sha(out/'reference_bindings.json'),
        manifest_sha256=sha(out/'manifest.json'),evidence=evidence,code_hashes={n:sha(ROOT/n) for n in SOURCES},
        downstream_scoring='offline original location matcher only; no semantic matching API')
    write(out/'binding.json',binding)


def verify(out):
    b=read(out/'binding.json')
    require(all(sha(ROOT/n)==h for n,h in b['code_hashes'].items()),'Annotation code drift')
    for name,key in [('annotation_inputs.json','inputs_sha256'),('reference_bindings.json','reference_bindings_sha256'),('manifest.json','manifest_sha256')]:
        require(sha(out/name)==b[key],'Annotation input drift')
    require(all(sha(out/n)==h for n,h in b['evidence'].items()),'Generation evidence drift')
    return sha(out/'binding.json')


def confirm(out):
    signature=verify(out);ready=read(out/'disconnect_ready.json')
    require(generation.corpus.runtime.process_identity(ready['pid'])==ready['process_start'] and
        ready['configuration_sha256']==signature and time.time()-ready['unix']>=5,'Reconnect required')
    write(out/'disconnect_acceptance.json',dict(ready,passed=True,confirmed_unix=time.time()))


def run(out,env):
    require(Path('/proc').exists(),'Server only')
    with generation.corpus.r.run_lock(out/'coordinator.lock'):
        signature=verify(out)
        require(not(out/'STOP').exists(),'STOP requested')
        generation.corpus.runtime.wait_detached(out,signature)
        code=1
        try:
            code=teacher.run(SimpleNamespace(source=out/'annotation_inputs.json',output=out,env=env,
                mode='pilot',expected_count=100,pilot_size=100,recover_technical=True))
        finally:(out/'coordinator.exit_code').write_text(str(code))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','verify','run','confirm-detached']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bench',type=Path);p.add_argument('--screen',type=Path);p.add_argument('--inventory',type=Path)
    p.add_argument('--env',type=Path,default=ROOT/'.env')
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='prepare':prepare(out,a.bench.resolve(),a.screen.resolve(),a.inventory.resolve(),a.env.resolve())
    elif a.mode=='verify':verify(out);print('verified')
    elif a.mode=='confirm-detached':confirm(out)
    else:run(out,a.env.resolve())


if __name__=='__main__':main()
