"""Approved source quarantine and RAG target binding; reuse exact frozen inputs.

The parent run remains immutable. A new result may reference a parent result
only when the complete CHECK input and budget contract are identical. All
other execution uses the unchanged v15 model/decoder/checking engine.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import re
import time
from unittest.mock import patch

import httpx

from scripts import run_aligned_step_evaluation as old
from scripts import resume_aligned_step_evaluation as recovery
from scripts import audit_aligned_full_chain as provenance
from scripts import review_aligned_step_scoring as review

base, vr, require = old.base, old.vr, old.require
VERSION = 'aligned_source_quarantine_rag_target_v1'
EXCLUDED = {
    '9cb9022f6b09bf8a47ff9eb575b3753cb0ea937958a70c6223a63aed94e39e2d': 'confirmed_missing_C4_and_field_conditions',
    '45527c4f9f3526a8f61d50050bbc873a511c8e482f8ad78046224c1bcbc9539b': 'confirmed_missing_prior_dynamics_and_M_f_definitions',
    'a1278ed5174fc7a4e8019839ef355d46eac27c0f545afb4f44ce3b683f5f807b': 'quarantined_unresolved_prior_radius_model_not_confirmed_error',
}
RAG_BINDING = '''Only diagnose the current student solution: the `prediction` field in the first user message.
Everything under `Reference material` is historical reference data, including its problem,
original 30B answer and machine check note. These are not the current student's answer.
An error or off-topic statement in a reference is not a diagnosis of the current answer.
Use historical material only to derive checks that apply to the current problem and solution.
Bind every diagnosis to the current solution. Copy evidence.quote from that solution;
never substitute text from a reference or invent a passage describing an omission.
Continue checking issues outside the references and return the existing diagnostics JSON.'''
ORIGINAL_CONTINUATION = base.p.continuation_messages


def numbered_dependencies(text):
    """Candidate detector only: numbered references do not establish missing data."""
    pattern = r'\b(?:part|question|problem)\s+(?:[A-Z]\s*\.?\s*)?\d+(?:\.[A-Z0-9]+)*\b'
    return [dict(text=m.group(), start=m.start(), end=m.end(), disposition='review_required')
            for m in re.finditer(pattern, text, re.I)]


def continuation(row, queries, references):
    messages = ORIGINAL_CONTINUATION(row, queries, references)
    messages[-1]['content'] += '\n\n' + RAG_BINDING
    return messages


def packet(row, arm, prefix, memory, count):
    if arm == 'case_rag':
        with patch.object(base.p, 'continuation_messages', continuation):
            return base.p.reference_packet(row, arm, prefix['queries'], memory, count)
    return base.p.reference_packet(row, arm, prefix['queries'], memory, count)


def contract(value, prefix, count, limits):
    """Fingerprint only model-visible data and scientific budgets, not BM25 scores."""
    return dict(messages=value['messages'], model=base.r.EXPECTED_MODEL,
        reasoning_effort='medium', temperature=0, top_p=1, tools=None,
        prompt_tokens=count(value['messages']), harmony_date=old.a.DATE,
        prefix_usage=prefix['evidence']['usage'], limits=limits,
        remaining_output=limits['output_tokens_per_sample']-prefix['evidence']['usage']['completion_tokens'])


def source_files():
    files = old.code_hashes()
    for name in ('scripts/run_aligned_step_revision.py', 'scripts/resume_aligned_step_evaluation.py',
                 'scripts/audit_aligned_full_chain.py', 'scripts/review_aligned_step_scoring.py'):
        files[name] = vr.sha(old.a.ROOT / name)
    return files


def memories(out):
    return dict(pure=None, case_rag=base.p.FlatMemory(vr.read(out/'cases.json')),
                flat_skills=base.p.FlatMemory(vr.read(out/'skills.json')),
                rule_tree=base.p.TreeMemory(vr.read(out/'catalog.json')))


def prefix_at(parent, row):
    path = parent/'queries'/vr.digest(row['id'])[:24]/'result.json'
    value = vr.read(path)
    require(value['input_sha256'] == vr.digest(row), 'Parent PREPARE uses different input')
    base.p.validate_queries(value['queries'])
    return value, path


def bind(files, path):
    path = Path(path); digest = vr.sha(path)
    require(str(path) not in files or files[str(path)] == digest, 'Conflicting immutable evidence')
    files[str(path)] = digest
    return vr.read(path)


def prepare(out, parent):
    require(Path('/proc').exists(), 'Server-only preparation with the original tokenizer')
    require(not (out/'plan.json').exists(), 'Never overwrite a frozen revision')
    prior = old.verify(parent); pc = prior['configuration']; files = {}
    audit = bind(files,parent/'offline_review/full_chain_audit.json')
    require(audit['original_records_unchanged'] and audit['model_calls'] == 0, 'Parent audit missing')
    rows = bind(files,parent/'inputs.json'); require(len(rows)==100,'Not the frozen 100')
    sources = bind(files,parent/'sources.json'); cases=bind(files,parent/'cases.json')
    skills=bind(files,parent/'skills.json'); catalog=bind(files,parent/'catalog.json')
    require(set(EXCLUDED) <= {r['id'] for r in sources}, 'Approved exclusion absent')
    current=Path(pc['current']); current_plan=bind(files,current/'plan.json')
    bundle=bind(files,Path(current_plan['configuration']['root'])/'pipeline/tree_bundle.json')
    catalog, changes=old.a.bounded.previous.prune_catalog(catalog,bundle['provenance'],set(EXCLUDED))
    kept_cases=[d for d in cases if not set(d['source_ids']) & set(EXCLUDED)]
    kept_skills=[d for d in skills if not set(d['source_ids']) & set(EXCLUDED)]
    for name,value in [('cases',kept_cases),('skills',kept_skills),('catalog',catalog)]:
        vr.atomic(out/(name+'.json'),value)
    source_review = audit['source_completeness_observation']['candidate_context_review']
    vr.atomic(out/'source_quarantine.json',dict(excluded=EXCLUDED, original_sources=len(sources),
        kept_sources=len(sources)-len(EXCLUDED), removed_cases=[d['id'] for d in cases if d not in kept_cases],
        removed_skills=[d['id'] for d in skills if d not in kept_skills], tree_changes=changes,
        numbered_candidates=[dict(source_id=r['id'], matches=numbered_dependencies(r['question']))
                             for r in sources if numbered_dependencies(r['question'])],
        existing_context_review=source_review, detector_auto_excludes=False,
        regenerated_sources=0, new_model_calls=0, source_replacements=0, human_gold=False))
    pilots=bind(files,current/'pilot_inputs.json')
    require(len(pilots)==6 and not {r['id'] for r in pilots}&{r['id'] for r in rows},'Fixed pilot drift')
    # Label definitions are frozen before any new predictions exist.
    for name in ('plan','reference','unresolved','label_release','metrics'):
        bind(files,parent/(name+'.json'))
    for name in ('protocol','labels','labels_frozen','review_frozen','evidence','operator_map','metrics'):
        bind(files,parent/'offline_review'/(name+'.json'))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(base.r.EXPECTED_MODEL,local_files_only=True,trust_remote_code=False)
    count=lambda m:len(tokenizer.apply_chat_template(m,tokenize=True,add_generation_prompt=True))
    stores=memories(out); tasks=[]
    for row in rows:
        query_path=parent/'queries'/vr.digest(row['id'])[:24]/'result.json'
        prefix=None
        if query_path.exists():
            prefix,_=prefix_at(parent,row); bind(files,query_path)
            require(prefix['configuration_sha256']==prior['configuration_sha256'],'Prefix protocol mismatch')
            files.update(prefix['files'])
        for arm in base.p.ARMS:
            path=base.path_for(parent,'formal',arm,row); result=bind(files,path)
            require(result['id']==row['id'] and result.get('organization_arm',arm)==arm and
                    ('organization_arm' in result or result['status']=='failed') and
                    result['configuration_sha256']==prior['configuration_sha256'],'Parent result binding drift')
            old_packet=bind(files,path.with_name('reference_packet.json'))['packet']
            for event in result['trace']['response_evidence']:
                for key,hashkey in [('request_path','request_sha256'),('response_path','response_sha256'),
                                   ('raw_sse_path','raw_sse_sha256')]:
                    require(event[key] not in files or files[event[key]]==event[hashkey],'Evidence conflict')
                    files[event[key]]=event[hashkey]
            task=dict(id=row['id'],arm=arm,parent_result=str(path),parent_result_sha256=files[str(path)],
                      input_sha256=vr.digest(row))
            if prefix is None:
                require(result.get('failure_stage')=='prepare' and not result['diagnostics'],'Missing successful prefix')
                failure=bind(files,query_path.with_name(recovery.FAILURE)); files.update(failure['files'])
                task.update(action='retain_prepare_failure',reason='No resampling of the original content failure')
            else:
                new=packet(row,arm,prefix,stores[arm],count)
                before=contract(old_packet,prefix,count,pc['limits']); after=contract(new,prefix,count,pc['limits'])
                request_path=path.parent/'turn_02/request.json'; actual=bind(files,request_path)['request']
                require(actual['messages']==before['messages'] and actual['model']==before['model'] and
                        actual['temperature']==0 and actual['top_p']==1 and actual['reasoning_effort']=='medium' and
                        not actual.get('tools'),'Parent actual CHECK differs from packet')
                require(actual['max_tokens']==min(pc['limits']['max_tokens_per_response'],before['remaining_output']),
                        'Parent CHECK output budget differs')
                # The parent audit already checks actual Harmony tokens. Preserve
                # its raw evidence hash; never replace actual counts with estimates.
                task.update(action='reuse' if before==after else 'run_check',
                    before_contract_sha256=vr.digest(before),after_contract_sha256=vr.digest(after),
                    prompt_tokens_before=before['prompt_tokens'],prompt_tokens_after=after['prompt_tokens'],
                    before_reference_ids=old_packet['retrieval']['provided_complete_ids'],
                    after_reference_ids=new['retrieval']['provided_complete_ids'])
                vr.atomic(out/'packets'/arm/(vr.digest(row['id'])[:24]+'.json'),new)
            tasks.append(task)
    require(all(t['action']!='run_check' for t in tasks if t['arm']=='pure'),'Pure contract drift')
    require(sum(t['action']=='run_check' and t['arm']=='case_rag' for t in tasks)==97,'RAG task inventory drift')
    vr.EvidenceCache(out/'evidence_cache.json').check(files,full=True)
    vr.atomic(out/'tasks.json',tasks)
    cfg=copy.deepcopy(pc)
    cfg.update(protocol=VERSION,parent=str(parent),parent_configuration_sha256=prior['configuration_sha256'],
        source_hashes=source_files(),dependencies=files,rag_binding=RAG_BINDING,excluded_sources=EXCLUDED,
        pilot_inputs=pilots,pilot_policy='Fixed six development rows, four arms, one trial; no score-based selection.',
        service_lock=str(parent.parent.parent/'organization_service_18083.lock'),
        published_files={str(p.relative_to(out)):vr.sha(p) for p in out.rglob('*.json')
                         if p.name!='evidence_cache.json'},
        frozen_label_states=dict(active_error=164,self_corrected=16,unresolved=2),
        score_policy=review.POLICY,final_answer_task='Only separately recompute the original frozen 600 results.')
    vr.atomic(out/'plan.json',dict(configuration=cfg,configuration_sha256=vr.digest(cfg)))
    base.old.emit(out,'prepared',tasks=dict(Counter(t['action'] for t in tasks)),new_model_calls=0,
                  materials=dict(rag=len(kept_cases),skills=len(kept_skills),tree=len(stores['rule_tree'].corpus.leaves)))
    return verify(out)


def verify(out):
    plan=vr.read(out/'plan.json'); c=plan['configuration']
    require(c['protocol']==VERSION and vr.digest(c)==plan['configuration_sha256'],'Revision plan drift')
    require(c['source_hashes']==source_files() and c['rag_binding']==RAG_BINDING and c['excluded_sources']==EXCLUDED,
            'Revision code/approved policy drift')
    vr.EvidenceCache(out/'evidence_cache.json').check(c['dependencies'])
    for name,digest in c['published_files'].items():
        require(vr.sha(out/name)==digest,'Frozen revision input drift: '+name)
    return plan


def confirm(out):
    plan=verify(out); ready=vr.read(out/'disconnect_ready.json')
    require(base.runtime.process_identity(ready['pid'])==ready['process_start'] and
            time.time()-ready['unix']>=5 and ready['configuration_sha256']==plan['configuration_sha256'],
            'Fresh reconnect verification required')
    vr.atomic(out/'disconnect_acceptance.json',dict(ready,passed=True,confirmed_unix=time.time()))


@contextmanager
def revised(out,plan,tokenizer):
    parent=Path(plan['configuration']['parent']); original_query=base.query
    original_packet=base.p.reference_packet
    def query(root,row,current,tok,client):
        if (parent/'queries'/vr.digest(row['id'])[:24]/'result.json').exists():
            value,_=prefix_at(parent,row)
            value=copy.deepcopy(value);value['configuration_sha256']=plan['configuration_sha256']
            return value
        return original_query(root,row,current,tok,client)
    def changed_packet(row,arm,queries,memory,count):
        if arm=='case_rag':
            with patch.object(base.p,'continuation_messages',continuation):
                value=original_packet(row,arm,queries,memory,count)
        else:value=original_packet(row,arm,queries,memory,count)
        frozen=out/'packets'/arm/(vr.digest(row['id'])[:24]+'.json')
        if frozen.exists():require(value==vr.read(frozen),'Actual packet changed before any model request')
        return value
    with old.a.configured(out,tokenizer),patch.object(base,'query',query), \
         patch.object(base.p,'reference_packet',changed_packet):
        yield


def result_for(out,task):
    if task['action'] in {'reuse','retain_prepare_failure'}:
        path=Path(task['parent_result'])
        require(vr.sha(path)==task['parent_result_sha256'],'Reused result drift')
    else:
        path=base.path_for(out,'formal',task['arm'],dict(id=task['id']))
    return vr.read(path),path


def origin_report(row,result,refs):
    records=[]
    for i,d in enumerate(result['diagnostics']):
        origin=provenance.reference_origins(row,d,refs)
        kind=('both' if origin['reference_matches'] else 'current') if origin['in_student'] else \
             ('historical_only' if origin['reference_matches'] else 'unlocated')
        records.append(dict(index=i,origin=kind,**origin))
    return dict(records=records,diagnostics_modified=False,automatic_physical_judgments=False)


def run(out):
    require(Path('/proc').exists(),'Inference must execute on the server')
    plan=verify(out);c=plan['configuration'];parent=Path(c['parent'])
    with base.r.run_lock(out/'active.lock'),base.r.run_lock(Path(c['service_lock'])):
        require(vr.read(out/'service/startup.json')['fixed_harmony_date']==old.a.DATE,'Service date drift')
        runtime=base.old.server_preflight(out,dict(c,model=base.r.EXPECTED_MODEL))
        require(runtime==vr.read(parent/'runtime.json'),'Model or runtime changed')
        base.old.bind_runtime(out/'runtime.json',runtime)
        (out/'runner.pid').write_text(str(os.getpid()));exit_code=1
        try:
            base.runtime.wait_detached(out,plan['configuration_sha256'])
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(base.r.EXPECTED_MODEL,local_files_only=True,trust_remote_code=False)
            stores=memories(out);rows={r['id']:r for r in vr.read(parent/'inputs.json')}
            with revised(out,plan,tokenizer),httpx.Client(trust_env=False) as client:
                for i,row in enumerate(c['pilot_inputs']):
                    for arm in c['arms'][i%4:]+c['arms'][:i%4]:
                        require(not (out/'STOP').exists(),'STOP requested')
                        result=old.a.bench.run_condition(out,row,'pilot',arm,stores[arm],plan,tokenizer,client)
                        require(base.runtime.healthy_result(result) and result['protocol_audit']['passed'],
                                'Pilot engineering failure; no resampling')
                # A human/assistant evidence review of actual pilot outputs is
                # required once; it is not a metric-based tuning opportunity.
                gate=out/'pilot_acceptance.json'
                if not gate.exists():
                    base.old.emit(out,'awaiting_pilot_review',pilot_tasks=24,formal_new_calls=0)
                    deadline=time.monotonic()+300
                    while not gate.exists() and time.monotonic()<deadline:
                        require(not (out/'STOP').exists(),'STOP requested during pilot review')
                        time.sleep(2)
                    if not gate.exists():exit_code=0;return
                acceptance=vr.read(gate)
                require(acceptance['passed'] and acceptance['configuration_sha256']==plan['configuration_sha256'],
                        'Pilot review not accepted')
                for name,digest in acceptance['files'].items():
                    require(vr.sha(name)==digest,'Reviewed pilot evidence changed')
                tasks=vr.read(out/'tasks.json')
                for i,task in enumerate(tasks):
                    row=rows[task['id']];arm=task['arm']
                    require(not (out/'STOP').exists(),'STOP requested')
                    if task['action']=='run_check':
                        result=old.a.bench.run_condition(out,row,'formal',arm,stores[arm],plan,tokenizer,client)
                        provided=vr.read(base.path_for(out,'formal',arm,row).with_name('reference_packet.json'))['packet']
                        frozen=vr.read(out/'packets'/arm/(vr.digest(row['id'])[:24]+'.json'))
                        require(provided==frozen,'Live input differs from frozen replay')
                    else:
                        result,_=result_for(out,task)
                        provided=vr.read(Path(task['parent_result']).with_name('reference_packet.json'))['packet']
                        receipt=out/'reused'/arm/(vr.digest(row['id'])[:24]+'.json')
                        if receipt.exists():require(vr.read(receipt)==task,'Reuse receipt drift')
                        else:vr.atomic(receipt,task)
                    evidence=origin_report(row,result,provided['references'] if 'references' in provided else [])
                    vr.atomic(out/'origins'/arm/(vr.digest(row['id'])[:24]+'.json'),evidence)
                    base.old.emit(out,'formal',processed=i+1,total=400,arm=arm,sample_id=row['id'],
                                  action=task['action'],result_status=result['status'])
            verify(out)
            base.old.emit(out,'collection_complete',recorded_tasks=400,scoring='new_diagnosis_review_pending')
            exit_code=0
        except BaseException as exc:
            base.old.emit(out,'stopped',reason=str(exc));raise
        finally:
            (out/'runner.exit_code').write_text(str(exit_code))
            (out/'runner.pid').unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','verify','run','confirm-detached'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--parent',type=Path)
    args=p.parse_args();out=args.output.resolve()
    if args.command=='prepare':
        out.mkdir(parents=True,exist_ok=True)
        with base.r.run_lock(out/'active.lock'):
            prepare(out,args.parent.resolve())
    elif args.command=='verify':verify(out)
    elif args.command=='confirm-detached':confirm(out)
    else:run(out)


if __name__=='__main__':main()
