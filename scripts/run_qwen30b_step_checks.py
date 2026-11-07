"""Two-arm engineering acceptance of fresh 30B step diagnoses.

Reuses the continuous checker and original-source v5 tree. No reference labels
are loaded during prepare/run; annotation and scoring are separate jobs.
"""
import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch

import httpx

from scripts import run_qwen30b_step_verifier as cohort
from scripts import run_organization_comparison as base

vr, require = base.vr, base.require
VERSION = 'qwen30b_original_v5_step_checks_v1'


def code_hashes():
    names=set(base.SOURCE_FILES)|{'scripts/run_qwen30b_step_checks.py',
        'scripts/serve_fixed_date_vllm.py','scripts/start_oss120b_step_service.sh'}
    return {n: vr.sha(base.ROOT/n) for n in sorted(names)}


def prepare(output, source, catalog):
    output.mkdir(parents=True,exist_ok=True)
    with base.r.run_lock(output/'active.lock'):
        require(not (output/'plan.json').exists(),'Already prepared; do not overwrite')
        parent=cohort.verify(source); pc=parent['configuration']
        require(vr.sha(catalog)==pc['source_files'][cohort.BUILD+'/catalog.json'],'Wrong v5 catalog')
        isolation=vr.read(source/'catalog_isolation.json')
        require(isolation.get('passed') is True and isolation['catalog_sha256']==vr.sha(catalog)
                and isolation['questions_sha256']==vr.sha(source/'questions.json'),
                'Catalog visible-text isolation has not passed for this cohort')
        rows=vr.read(source/'inputs.json')
        require([x['id'] for x in rows]==pc['pilot_ids'],'Pilot must contain all six fresh complete answers')
        require(all(set(x)=={'id','question','context','prediction'} for x in rows),'Checker input includes extra fields')
        source_receipts={}
        for row in rows:
            path=source/'generated'/(row['id']+'.json');saved=vr.read(path)
            cohort.corpus.evidence_valid(source,saved)
            require(saved['model']==pc['generator'] and saved['prediction']==row['prediction'],'Answer identity drift')
            source_receipts[str(path)]=vr.sha(path)
        vr.atomic(output/'inputs.json',rows)
        vr.atomic(output/'catalog.json',vr.read(catalog))
        base.p.TreeMemory(vr.read(catalog))
        c={'protocol':VERSION,'source_parent':str(source),'source_plan_sha256':vr.sha(source/'plan.json'),
           'catalog_isolation_sha256':vr.sha(source/'catalog_isolation.json'),
           'source_receipts':source_receipts,'generator':pc['generator'],
           'source_hashes':code_hashes(),'inputs_sha256':vr.sha(output/'inputs.json'),
           'catalog_sha256':vr.sha(output/'catalog.json'),'api_base':'http://127.0.0.1:18083/v1',
           'model':base.r.EXPECTED_MODEL,'model_files':base.runtime.MODEL_FILES,
           'fixed_harmony_date':pc['fixed_harmony_date'],'reasoning_effort':'medium',
           'limits':base.runtime.LIMITS,'check_prompt':base.p.STAGED_BASE,
           'query_max_tokens':base.p.QUERY_MAX_TOKENS,'reference_token_cap':base.p.REFERENCE_TOKEN_CAP,
           'arms':['pure','rule_tree'],'sample_ids':[x['id'] for x in rows],
           'mode':'engineering_pilot_only','planned_checks':len(rows)*2,
           'human_gold':False,'independent_test':False}
        vr.atomic(output/'plan.json',{'configuration':c,'configuration_sha256':vr.digest(c)})
        base.old.emit(output,'prepared_no_model_calls',planned_checks=len(rows)*2)


def verify(output):
    plan=vr.read(output/'plan.json');c=plan['configuration']
    require(vr.digest(c)==plan['configuration_sha256'] and c['protocol']==VERSION,'Plan drift')
    require(c['source_hashes']==code_hashes(),'Code drift')
    require(vr.sha(output/'inputs.json')==c['inputs_sha256'] and
            vr.sha(output/'catalog.json')==c['catalog_sha256'],'Material drift')
    require(vr.sha(Path(c['source_parent'])/'plan.json')==c['source_plan_sha256'],'Generation plan drift')
    require(vr.sha(Path(c['source_parent'])/'catalog_isolation.json')==c['catalog_isolation_sha256'],
            'Catalog isolation evidence drift')
    for name,sha in c['source_receipts'].items():require(vr.sha(Path(name))==sha,'30B evidence drift')
    return plan


def confirm(output):
    plan=verify(output);ready=vr.read(output/'disconnect_ready.json')
    require(base.runtime.process_identity(ready['pid'])==ready['process_start'] and
            time.time()-ready['unix']>=5 and ready['configuration_sha256']==plan['configuration_sha256'],
            'Fresh SSH confirmation required')
    vr.atomic(output/'disconnect_acceptance.json',dict(ready,passed=True,confirmed_unix=time.time()))


def validate_effective_input(raw, tokenizer, date):
    tokens=raw.get('prompt_token_ids')
    require(isinstance(tokens,list) and tokens and all(type(x) is int for x in tokens),
            'Server must return actual input tokens')
    header=tokenizer.decode(tokens[:256],skip_special_tokens=False).split('[im_end]',1)[0]
    require('Current date: '+date in header and 'Reasoning: medium' in header,
            'Actual Harmony date/reasoning differs from frozen protocol')
    return {'sha256':vr.digest(tokens),'token_count':len(tokens),'system_header':header}


def run(output):
    with base.r.run_lock(output/'active.lock'),base.r.run_lock(output.parent/'organization_service_18083.lock'):
        plan=verify(output);c=plan['configuration'];code=1
        # Read the startup receipt from the personal fixed-date service, not a client date string.
        service=output.parent/'oss_service/startup.json'
        require(vr.read(service)['fixed_harmony_date']==c['fixed_harmony_date'],'Fixed-date service not verified')
        base.old.bind_runtime(output/'runtime.json',base.old.server_preflight(output,c))
        (output/'runner.pid').write_text(str(os.getpid()))
        try:
            base.runtime.wait_detached(output,plan['configuration_sha256'])
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(c['model'],local_files_only=True,trust_remote_code=False)
            memories={'pure':None,'rule_tree':base.p.TreeMemory(vr.read(output/'catalog.json'))}
            def decode(raw,**kwargs):
                receipt=validate_effective_input(raw,tokenizer,c['fixed_harmony_date'])
                vr.atomic(output/'effective_inputs'/(receipt['sha256']+'.json'),receipt)
                return base.decode_response(raw,**kwargs)
            with base.old.comparison.execution.configured(),patch.object(base.runtime.transport,'decode_response',decode),httpx.Client(trust_env=False) as client:
                for phase,rows in [('controls',base.runtime.CONTROLS[:2]),('formal',vr.read(output/'inputs.json'))]:
                    for i,row in enumerate(rows):
                        for arm in c['arms'][i%2:]+c['arms'][:i%2]:
                            require(not (output/'STOP').exists(),'STOP requested')
                            base.run_condition(output,row,phase,arm,memories[arm],plan,tokenizer,client)
            base.old.emit(output,'pilot_checks_complete_pending_independent_labels',
                          recorded_tasks=c['planned_checks'],physical_metrics_available=False)
            code=0
        except BaseException as exc:
            base.old.emit(output,'stopped',error_type=type(exc).__name__,error=str(exc)[:1000]);raise
        finally:
            (output/'runner.exit_code').write_text(str(code));(output/'runner.pid').unlink(missing_ok=True)


def reference_rows(rows, annotations):
    """New-answer-only location reference; uncertain/unlocated claims stay visible."""
    expected={x['id']:x for x in rows}
    require(set(annotations)==set(expected),'Reference cohort differs')
    dataset=[];unresolved={};counts={'confirmed':0,'unresolved':0,'confirmed_unlocated':0}
    for row in rows:
        gt=[];unknown=[]
        for error in annotations[row['id']]['errors']:
            require(error['status'] in ('confirmed','unresolved'),'Unknown reference status')
            quote=error['quote'];unique=bool(quote) and row['prediction'].count(quote)==1
            start=row['prediction'].index(quote) if unique else -1
            item={'error_id':row['id']+'/'+error['root_id'],'error_text':error['claim'],
                  'answer_quote':quote,'start_char':start,'end_char':start+len(quote) if unique else -1,
                  'span_valid':unique,'paragraph_index':-1,'paragraph_valid':False}
            (gt if error['status']=='confirmed' else unknown).append(item)
            counts[error['status']]+=1
            if error['status']=='confirmed' and not unique:counts['confirmed_unlocated']+=1
        dataset.append(dict(row,physics_error_gt=gt));unresolved[row['id']]=unknown
    return dataset,unresolved,counts


def score(output, annotations):
    from scripts import run_bench_reannotation as labeling
    plan=verify(output);c=plan['configuration'];rows=vr.read(output/'inputs.json')
    labeled=vr.read(annotations/'inputs.json');manifest=vr.read(annotations/'manifest.json')
    require(manifest['model']==labeling.MODEL and manifest['reasoning_effort']=='high','Wrong label model')
    require({x['id']:{k:x.get(k,'') for k in ('id','question','context','prediction')} for x in labeled}
            =={x['id']:x for x in rows},'Annotations belong to a different answer/cohort')
    audit=labeling.audit(annotations,labeled,manifest)
    require(audit['fully_annotated_questions']==len(rows) and
            all(audit['completed'][stage]==len(rows) for stage in labeling.STAGES),'Four-pass annotation incomplete')
    finals={x['id']:vr.read(annotations/'arm_final'/x['id']/'result.json')['annotation'] for x in rows}
    dataset,unknown,counts=reference_rows(rows,finals);scores={};cases=[];evidence={}
    for arm in c['arms']:
        outputs=[]
        for row in rows:
            path=base.path_for(output,'formal',arm,row)
            result=vr.read(path);base.runtime.audit_result(result)
            require(result['configuration_sha256']==plan['configuration_sha256'],'Diagnosis plan drift')
            evidence[str(path)]=vr.sha(path)
            outputs.append(dict(result,execution_failure=not base.runtime.healthy_result(result),
                                source_status=result['status']))
        scores[arm]=base.evaluation.score(dataset,outputs,unknown,preserve_zero_offsets=True)
    for row in rows:
        cases.append({'input':row,'sol_reference':finals[row['id']],
            'outputs':{arm:vr.read(base.path_for(output,'formal',arm,row)) for arm in c['arms']},
            'matches':{arm:next(x for x in scores[arm]['details'] if x['id']==row['id']) for arm in c['arms']},
            'assistant_semantic_review':'pending'})
    vr.atomic(output/'metrics_step_location_proxy.json',{'configuration_sha256':plan['configuration_sha256'],
        'metric':'one-to-one location proxy, not semantic correctness','iou':0.5,'coverage':0.6,
        'preserve_zero_offsets':True,
        'reference':'new 30B answers, Sol high four-pass machine annotations; semantic acceptance pending',
        'reference_counts':counts,'reference_manifest_sha256':vr.sha(annotations/'manifest.json'),
        'reference_evidence':audit['preserved_results'],'prediction_files':evidence,'scores':scores,
        'human_gold':False,'independent_test':False,'pilot_only':True})
    vr.atomic(output/'pilot_cases.json',{'cases':cases,'human_gold':False,'review_status':'pending'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','verify','run','confirm-detached','score'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--catalog',type=Path)
    p.add_argument('--annotations',type=Path)
    a=p.parse_args();out=a.output.resolve()
    if a.command=='prepare':prepare(out,a.source.resolve(),a.catalog.resolve())
    elif a.command=='verify':verify(out)
    elif a.command=='run':run(out)
    elif a.command=='score':score(out,a.annotations.resolve())
    else:confirm(out)


if __name__=='__main__':main()
