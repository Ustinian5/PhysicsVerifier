"""Resume the approved corrected experiment through both offline score tables.

One detached server coordinator owns port 18083's experiment lock. Stage
completion advances automatically; immutable inputs, model identity, isolation
and pilot engineering failures remain hard stops. No Codex automation required.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import threading
import time

import httpx
from rule_framework import bounded_tree as tree
from rule_framework import corrected_verifier as protocol
from rule_framework import organization_followup as skills
from rule_framework import verified_resume as vr
from rule_framework.flat_skill_rag import FlatMemory, TreeMemory
from scripts import audit_hipho_overlap as isolation
from scripts import run_corrected_annotations as annotations
from scripts import run_corrected_extraction as extraction
from scripts import run_final_answer_labels as final_labels
from scripts import run_corrected_checks as checking
from scripts import rescore_frozen_bench as scoring
from scripts.run_qwen30b_step_checks import reference_rows, validate_effective_input

c = extraction.c
require = c.require
SOURCES = ('scripts/run_corrected_pipeline.py','scripts/run_corrected_checks.py',
    'rule_framework/corrected_verifier.py','rule_framework/bounded_tree.py','rule_framework/verified_resume.py',
    'rule_framework/flat_skill_rag.py','rule_framework/organization_followup.py','rule_framework/organization_bench99.py',
    'rule_framework/final_answer_protocol.py','scripts/rescore_frozen_bench.py',
    'scripts/continuous_context_v2.py','scripts/run_continuous_context_v2.py',
    'scripts/run_gpt_oss_selfcheck.py','scripts/gpt_oss_harmony_transport.py',
    'scripts/resume_continuous_context_v2.py','scripts/audit_hipho_overlap.py',
    'scripts/serve_continuous_context_vllm.py')
PILOT_IDS = ('cl_202_125689','cl_212_135843','cl_196_119740','cl_174_97192')


def public(row):
    return {k:row.get(k,'') for k in ('id','question','context','prediction')}


def immutable(path, value):
    if path.exists():
        require(vr.read(path)==value, 'Frozen output drift: '+str(path))
    else:
        vr.atomic(path,value)


def balanced(questions, outcomes, quota=300):
    counts=Counter();selected=[]
    for q in questions:
        outcome=outcomes.get(q['id'])
        if not outcome or outcome['state']!='complete':continue
        label=outcome['judgment']['verdict']
        if label in {'correct','incorrect'} and counts[label]<quota:
            selected.append(q);counts[label]+=1
    require(counts=={'correct':quota,'incorrect':quota}, 'Balanced final-answer quota unavailable')
    return selected


def prepare(out, root):
    require(Path('/proc').exists(),'Prepare against server originals')
    require(not(out/'plan.json').exists(),'Already prepared; use verify/run, never overwrite')
    source=root/'construction'
    extraction.verify(source)
    require(vr.read(source/'execution_status.json')['state']=='extraction_complete','Extraction not complete')
    annotations.verify(root/'annotations');final_labels.verify(root/'final_labels')
    files={}
    def bind(path):
        files[str(path.resolve())]=vr.sha(path)
        return vr.read(path)
    for n in ['plan.json','inputs.json','semantic_experience.json','distilled.json','taxonomy.json',
              'source_evidence.json','generation_failures.json','invalid_outputs.json']:
        bind(source/n)
    files.update(vr.read(source/'source_evidence.json'))
    valid=[]
    for row in vr.read(source/'inputs.json'):
        record=bind(source/'experiences'/(row['id']+'.json'))
        for ev in record['evidence']:
            files[str(source/ev['file'])]=ev['sha256']
        if record['value'] is not None:valid.append(record['value'])
    require(valid==vr.read(source/'semantic_experience.json'),'Experience aggregate drift')
    distilled=extraction.old.extraction._build_distilled_library(valid,min_count=1)
    require(distilled==vr.read(source/'distilled.json'),'Rule aggregate drift')
    rows={x['id']:x for x in vr.read(source/'inputs.json')}; cases=[];support=defaultdict(list)
    for exp in valid:
        sid=exp['sample_id'];row=rows[sid];audit=exp['semantic_audit']
        if audit['status']=='error_found':
            cases.append(dict(id='case_'+sid,kind='case',title=' '.join(row['question'].split())[:160],
                text='历史题目\n'+row['question']+'\n\n原30B回答（可能有错，不是正确范例）\n'+row['prediction']+
                '\n\n120B原始核查说明\n'+json.dumps(audit,ensure_ascii=False),source_ids=[sid]))
        errors={e['id']:e for e in audit['key_errors']}
        for rule in exp['experience_rules']:
            rid='exp_'+extraction.old.extraction._fingerprint_rule(rule,exp['topic_guess']['domain'],exp['topic_guess']['topic'])
            support[rid].extend(dict(sample_id=sid,source_error=errors[eid]) for eid in rule['source_error_ids'])
    error_ids={x['source_ids'][0] for x in cases}
    for rule in distilled['rules']:
        require(set(rule['sample_ids'])=={x['sample_id'] for x in support[rule['rule_id']]} and
                set(rule['sample_ids'])<=error_ids,'Rule source mismatch')
    bundle=dict(taxonomy=vr.read(source/'taxonomy.json'),rules=distilled['rules'],provenance=dict(support),
                human_gold=False,source_counts=vr.read(source/'execution_status.json')['counts'])
    step_rows=bind(root/'bench_generation/inputs.json')
    labeled=bind(root/'annotations/annotation_inputs.json')
    require(list(map(public,step_rows))==list(map(public,labeled)) and len(step_rows)==100,'Step answer cohort drift')
    bind(root/'annotations/binding.json');bind(root/'annotations/manifest.json')
    quality=bind(root/'annotations/quality_audit.json')
    require(quality['completed_tasks']==400 and quality['fully_annotated_questions']==100,'Step labels unfinished')
    finals={}
    # Revalidate all four label rounds from their original responses, not merely
    # trust an aggregate "400". Do not edit model physics or erase review flags.
    teacher=annotations.teacher
    for stage in teacher.STAGES:
        for row in labeled:
            previous=teacher.previous_annotations(root/'annotations',stage,row['id'])
            saved=teacher.validate_saved(root/'annotations',stage,row,previous)
            bind(root/'annotations'/('arm_'+stage)/row['id']/'result.json')
            files[str(root/'annotations'/saved['response_record'])]=saved['response_record_sha256']
            if stage=='final':finals[row['id']]=saved['annotation']
    dataset,unresolved,counts=reference_rows(step_rows,finals)
    label_release=dict(human_gold=False,counts=counts,review_flags=quality['human_review_queue'],
        status='four_round_machine_reference_with_explicit_unresolved_and_unlocatable_items',
        physical_independent_review_complete=False,
        policy='Final Sol physics unchanged; only uniquely verbatim quotes receive offsets; unresolved labels separate. Offline location proxy, not semantic accuracy.')
    questions=bind(root/'final_labels/questions.json');label_config=bind(root/'final_labels/plan.json')['configuration']
    references={x['id']:x for x in bind(root/'final_labels/references_private.json')}
    outcomes={}
    for q in questions:
        path=root/'final_labels/labels'/(q['id']+'.json')
        if path.exists():outcomes[q['id']]=bind(path)
    selected=balanced(questions,outcomes); final_rows=[];gold={}
    for q in selected:
        generated=bind(root/'collection/generated'/(q['id']+'.json'))
        c.corpus.evidence_valid(root/'collection',generated)
        files[str(root/'collection'/generated['evidence']['file'])]=generated['evidence']['sha256']
        require(generated['state']=='complete' and generated['signature']==c.digest([q,label_config['generator_config']]),'Final generation drift')
        row=dict(q,prediction=generated['prediction']);label=outcomes[q['id']]
        require(label['signature']==c.digest([row,references[q['id']]['reference'],label_config]),'Final label input drift')
        c.corpus.evidence_valid(root/'final_labels',label)
        files[str(root/'final_labels'/label['evidence']['file'])]=label['evidence']['sha256']
        final_rows.append(public(row));gold[q['id']]=dict(verdict=label['judgment']['verdict'],human_gold=False,
            original_label_sha256=vr.sha(root/'final_labels/labels'/(q['id']+'.json')))
    pilot_source=bind(root.parent/'inputs.json')
    pilot=[public(x) for x in pilot_source if x['id'] in PILOT_IDS]
    require({x['id'] for x in pilot}==set(PILOT_IDS),'Missing original complete-condition pilot')
    controls=[public(x) for x in final_labels.CONTROLS[:2]]
    pilot+=controls
    split=bind(root/'source_screen/split.json')
    bind(root/'source_screen/family_components.json')
    hipho=bind(root.parent/'hipho_overlap_audit.json')
    require(hipho['status']=='screening_and_assistant_review_complete_no_confirmed_overlap','HiPhO source audit unresolved')
    require(vr.sha(Path(hipho['benchmark_path']))==hipho['benchmark_sha256'],'External benchmark changed')
    files[hipho['benchmark_path']]=hipho['benchmark_sha256']
    require(set(rows)<= {x['id'] for x in split['build']} and
            {x['id'] for x in final_rows} <= {x['id'] for x in split['reserve']},'Cohort escaped frozen split')
    outputs={'tree_bundle':bundle,'cases':cases,'step_inputs':step_rows,'step_reference':dataset,
             'step_unresolved':unresolved,'step_label_release':label_release,
             'final_inputs':final_rows,'final_labels_private':gold,'pilot_inputs':pilot,
             'heldout_questions':list(map(public,step_rows))+list(map(public,split['reserve']))+list(map(public,pilot_source))}
    for name,value in outputs.items():immutable(out/(name+'.json'),value)
    transport=dict(model=c.corpus.r.EXPECTED_MODEL,auditor_api='http://127.0.0.1:18083/v1',
        max_tokens=16384,technical_attempts=3,retry_cooldown_seconds=60,timeout_seconds=1200,
        reasoning_effort='medium',fixed_date='2026-11-07')
    cfg=dict(protocol=protocol.VERSION,root=str(root),transport=transport,
        model_files=extraction.old.b.existing.MODEL_FILES,limits=protocol.LIMITS,
        tree_policy=tree.POLICY,arms=list(protocol.ARMS),human_gold=False,
        auto_advance='User 2026-11-10: continue through both experiments; automatic engineering-only pilot gate',
        code={n:vr.sha(c.ROOT/n) for n in SOURCES},
        frozen_files={name+'.json':vr.sha(out/(name+'.json')) for name in outputs},
        evidence_digest=vr.digest(files),planned_step_checks=400,planned_final_checks=2400,
        pilot_checks_per_task=24,unknown_usage_policy='Per online case, reserve full advertised allowance; never restart its budget')
    immutable(out/'source_evidence.json',files)
    vr.EvidenceCache(out/'source_verification.json').check(files,full=True)
    immutable(out/'plan.json',dict(configuration=cfg,configuration_sha256=vr.digest(cfg)))
    emit(out,'prepared',cases=len(cases),rules=len(distilled['rules']),step=100,final=600)


def verify(out, full=False):
    plan=vr.read(out/'plan.json');cfg=plan['configuration']
    require(vr.digest(cfg)==plan['configuration_sha256'],'Pipeline plan drift')
    effective_code=cfg['code']
    repair_path=out/'local_rendering_repair.json'
    if repair_path.exists():
        repair=vr.read(repair_path)
        require(repair['configuration_sha256']==plan['configuration_sha256'],'Runtime repair binding drift')
        require(set(repair['code'])==set(effective_code),'Runtime repair file set drift')
        allowed={'scripts/run_corrected_pipeline.py','scripts/run_corrected_checks.py'}
        require(all(n in allowed or repair['code'][n]==h for n,h in effective_code.items()),
                'Runtime repair changed unrelated implementation')
        require(all(vr.sha(out/n)==h for n,h in repair['retained_requests'].items()),'Runtime repair raw drift')
        effective_code=repair['code']
    require(all(vr.sha(c.ROOT/n)==h for n,h in effective_code.items()),'Pipeline code drift')
    require(all(vr.sha(out/n)==h for n,h in cfg['frozen_files'].items()),'Pipeline input drift')
    evidence=vr.read(out/'source_evidence.json')
    require(vr.digest(evidence)==cfg['evidence_digest'],'Evidence manifest drift')
    for path,h in cfg.get('parent_bindings',{}).items():
        require(vr.sha(Path(path))==h,'Parent experiment evidence drift: '+path)
    cache=Path(cfg['parent_pipeline']) if cfg.get('check_prompt_revision') else out
    vr.EvidenceCache(cache/'source_verification.json').check(evidence,full=full)
    return plan


def repair_local_rendering(out):
    """One bounded engineering repair; preserve signatures, bodies and budgets."""
    plan=vr.read(out/'plan.json');cfg=plan['configuration'];root=Path(cfg['root'])
    require(cfg.get('check_prompt_revision')==protocol.CHECK_REVISION,'Not the authorized checking revision')
    with c.corpus.r.run_lock(out/'active.lock'),c.corpus.r.run_lock(root/'organization_service_18083.lock'):
        if (out/'local_rendering_repair.json').exists():return verify(out)
        require(not(out/'runner.pid').exists(),'Stop coordinator before repair')
        require(vr.read(out/'execution_status.json').get('error')=='Template dropped visible context',
                'This repair only addresses the recorded local tool-JSON guard failure')
        require(not list((out/'checks/step/formal').rglob('result.json')) and
                not list((out/'checks/final').rglob('result.json')),'Repair is pre-formal only')
        allowed={'scripts/run_corrected_pipeline.py','scripts/run_corrected_checks.py'}
        require(all(n in allowed or vr.sha(c.ROOT/n)==h for n,h in cfg['code'].items()),'Unrelated runtime change')
        raw={str(f.relative_to(out)):vr.sha(f) for f in (out/'checks').rglob('request_*.json')}
        require(bool(raw),'Missing successful prefix to preserve')
        vr.atomic(out/'local_rendering_repair.json',dict(configuration_sha256=plan['configuration_sha256'],
            retained_requests=raw,code={n:vr.sha(c.ROOT/n) for n in SOURCES},
            reason='HF local template JSON-encodes tool strings. Decode that layer for exact context validation; actual request and deployed-input validation unchanged.',
            request_parameters_changed=False,case_budget_reset=False,created_unix=time.time()))
        verify(out)


def prepare_check_revision(out,parent,revision=protocol.CHECK_REVISION):
    """User-approved suffix only; retain original construction and pilot records."""
    require(Path('/proc').exists(),'Prepare against server originals')
    names={protocol.CHECK_REVISION:'check_prompt_revision',protocol.LANGUAGE_REVISION:'check_prompt_revision_language',
           protocol.BILINGUAL_REVISION:'check_prompt_revision_bilingual'}
    require(revision in names and out==parent/names[revision],'Use the registered checking revision')
    old=vr.read(parent/'plan.json');cfg=old['configuration']
    require(vr.digest(cfg)==old['configuration_sha256'],'Parent plan drift')
    root=Path(cfg['root'])
    with c.corpus.r.run_lock(parent/'active.lock'),c.corpus.r.run_lock(root/'organization_service_18083.lock'):
        if (out/'plan.json').exists():return verify(out)
        require(not(parent/'runner.pid').exists(),'Parent coordinator must be stopped')
        require(not(parent/'STOP').exists(),'Parent STOP remains in force')
        # These are the only three implementation files needed for this change.
        allowed={'scripts/run_corrected_pipeline.py','scripts/run_corrected_checks.py',
                 'rule_framework/corrected_verifier.py'}
        require(all(n in allowed or vr.sha(c.ROOT/n)==h for n,h in cfg['code'].items()),
                'Unrelated frozen implementation changed')
        require(not list((parent/'checks/step/formal').glob('*/*/result.json')) and
                not list((parent/'checks/final').rglob('result.json')),'Formal/new final checks already exist')
        require(all(vr.sha(parent/n)==h for n,h in cfg['frozen_files'].items()),'Parent frozen data drift')
        evidence=vr.read(parent/'source_evidence.json')
        require(vr.digest(evidence)==cfg['evidence_digest'],'Parent evidence manifest drift')
        vr.EvidenceCache(parent/'source_verification.json').check(evidence)
        accept_materials(parent,cfg)  # existing receipt only; no construction/API
        pilot=vr.read(parent/'step_pilot_acceptance.json')
        require(not pilot['passed'] and not pilot['blocked'],'Unexpected failed pilot boundary')
        require(all(v['completed']==6 and v['read']==v['searched']==0 for v in pilot['per_arm'].values()),
                'This revision is authorized only for the recorded zero-lookup pilot')
        bindings={str(parent/n):vr.sha(parent/n) for n in
            ['plan.json','construction_acceptance.json','material_isolation.json','step_pilot_acceptance.json']}
        bindings.update({str(parent/n):h for n,h in vr.read(parent/'construction_acceptance.json')['files'].items()})
        for name,h in pilot['output_files'].items():
            path=parent/name;require(vr.sha(path)==h,'Original pilot changed');bindings[str(path)]=h
            result=vr.read(path)
            for ev in result['evidence']:
                raw=path.parent/ev['file'];require(vr.sha(raw)==ev['sha256'],'Pilot raw changed')
                bindings[str(raw)]=ev['sha256']
        probe=parent/'native_io'/vr.sha(c.ROOT/'scripts/serve_continuous_context_vllm.py')[:16]/'acceptance.json'
        receipt=vr.read(probe)
        require(receipt['passed'] and receipt['configuration_sha256']==vr.digest(cfg),'Native IO acceptance drift')
        bindings[str(probe)]=vr.sha(probe)
        for name,h in receipt['files'].items():
            require(vr.sha(probe.parent/name)==h,'Native IO raw drift');bindings[str(probe.parent/name)]=h
        if revision in {protocol.LANGUAGE_REVISION,protocol.BILINGUAL_REVISION}:
            previous=parent/('check_prompt_revision_language' if revision==protocol.BILINGUAL_REVISION else 'check_prompt_revision')
            require(not(previous/'runner.pid').exists(),'Previous checking revision still active')
            require(not(previous/'STOP').exists(),'Previous STOP remains in force')
            previous_plan=vr.read(previous/'plan.json');previous_cfg=previous_plan['configuration']
            require(vr.digest(previous_cfg)==previous_plan['configuration_sha256'],'Previous plan digest drift')
            for n,h in previous_cfg['parent_bindings'].items():
                require(vr.sha(Path(n))==h,'Earlier bound evidence drift');bindings[n]=h
            old_pilot=vr.read(previous/'step_pilot_acceptance.json')
            require(not old_pilot['passed'] and not old_pilot['blocked'],'Unexpected preceding pilot outcome')
            require(old_pilot['per_arm']['rule_tree']['read']==0,'Expected recorded missing tree reads')
            require(not list((previous/'checks/step/formal').rglob('result.json')) and
                    not list((previous/'checks/final').rglob('result.json')),'Previous formal checks already started')
            records=['plan.json','step_pilot_acceptance.json']+(
                ['routing_failure_audit.json'] if revision==protocol.BILINGUAL_REVISION else
                ['local_rendering_repair.json','lookup_language_audit.json'])
            for n in records:
                bindings[str(previous/n)]=vr.sha(previous/n)
            for n,h in old_pilot['output_files'].items():
                f=previous/n;require(vr.sha(f)==h,'Previous pilot result drift');bindings[str(f)]=h
                for ev in vr.read(f)['evidence']:
                    raw=f.parent/ev['file'];require(vr.sha(raw)==ev['sha256'],'Previous pilot raw drift')
                    bindings[str(raw)]=ev['sha256']
        out.mkdir(parents=True,exist_ok=True)
        for name in list(cfg['frozen_files'])+['source_evidence.json','construction_acceptance.json','material_isolation.json','materials','service']:
            link=out/name
            if link.is_symlink():require(link.resolve()==(parent/name).resolve(),'Wrong inherited link')
            else:require(not link.exists(),'Unexpected revision file');link.symlink_to(parent/name)
        pure_link=out/'checks/step/pilot/pure';pure_link.parent.mkdir(parents=True,exist_ok=True)
        if pure_link.is_symlink():require(pure_link.resolve()==(parent/'checks/step/pilot/pure').resolve(),'Wrong pure link')
        else:require(not pure_link.exists(),'Unexpected pure output');pure_link.symlink_to(parent/'checks/step/pilot/pure')
        updated=dict(cfg,code={n:vr.sha(c.ROOT/n) for n in SOURCES},
            check_prompt_revision=revision,parent_pipeline=str(parent),parent_bindings=bindings,
            pure_pilot_condition=check_condition(parent,cfg),
            prompt_change=dict(authorization='User 2026-11-11: adopt minimal suffix; pilot then automatically continue',
                appended_user_suffix=protocol.REQUIRED_LOOKUP_SUFFIX,arms=list(protocol.ARMS[1:]),
                unchanged='Original prompts intact; pure unchanged; tool_choice auto; same model, retrieval, materials, budget and scoring'))
        if revision==protocol.LANGUAGE_REVISION:
            updated['prompt_change'].update(authorization='User 2026-11-11: adopt Chinese tree queries and full-read relevance instructions; pilot then continue automatically',
                additional_knowledge_suffix=protocol.READ_SUFFIX,additional_tree_suffix=protocol.TREE_LANGUAGE_SUFFIX,
                previous_check_revision=str(parent/'check_prompt_revision'))
        elif revision==protocol.BILINGUAL_REVISION:
            require(previous_cfg['check_prompt_revision']==protocol.LANGUAGE_REVISION,'Wrong preceding prompt revision')
            updated['prompt_change'].update(authorization='User 2026-11-11: bilingual tree queries with physical domain; reuse other three pilot arms; pilot then automatically continue',
                additional_knowledge_suffix=protocol.READ_SUFFIX,additional_tree_suffix=protocol.TREE_BILINGUAL_SUFFIX,
                previous_check_revision=str(previous),changed_from_previous=['rule_tree'])
            updated['pilot_reuse_conditions']={'pure':updated['pure_pilot_condition']}
            for arm in ('case_rag','flat_skills'):
                for row in vr.read(parent/'pilot_inputs.json'):
                    require(protocol.messages(row,arm,'step',revision=previous_cfg['check_prompt_revision'])==
                            protocol.messages(row,arm,'step',revision=revision),'Cannot reuse changed pilot input')
                link=out/'checks/step/pilot'/arm
                if link.is_symlink():require(link.resolve()==(previous/'checks/step/pilot'/arm).resolve(),'Wrong reuse link')
                else:require(not link.exists(),'Unexpected reuse output');link.symlink_to(previous/'checks/step/pilot'/arm)
                updated['pilot_reuse_conditions'][arm]=check_condition(previous,previous_cfg)
        immutable(out/'plan.json',dict(configuration=updated,configuration_sha256=vr.digest(updated)))
        verify(out)
        emit(out,'prepared_check_prompt_revision',retained_original_pilots=24,reused_pure_pilots=6,
             reused_pilot_arms=list(updated.get('pilot_reuse_conditions',{'pure':None})),construction_model_calls=0)


def refresh_unstarted(out):
    """Only a prepared, never-called pipeline may bind implementation fixes."""
    require(not list(out.rglob('request_*.json')) and not (out/'runner.pid').exists() and
            not (out/'tree_progress.json').exists(),'Cannot refresh a started pipeline')
    with c.corpus.r.run_lock(out/'active.lock'):
        plan=vr.read(out/'plan.json');cfg=plan['configuration']
        require(vr.digest(cfg)==plan['configuration_sha256'],'Unstarted plan drift')
        require(all(vr.sha(out/n)==h for n,h in cfg['frozen_files'].items()),'Unstarted inputs changed')
        cfg['code']={n:vr.sha(c.ROOT/n) for n in SOURCES}
        vr.atomic(out/'plan.json',dict(configuration=cfg,configuration_sha256=vr.digest(cfg)))
        verify(out)


def repair_engineering(out):
    """Bind the transport repair before any construction/checking API calls."""
    require(not(out/'runner.pid').exists() and not(out/'tree_progress.json').exists(),
            'Stop and inspect started scientific stages before changing implementation')
    require(all('native_io' in x.relative_to(out).parts for x in out.rglob('request_*.json')),
            'Repair is limited to pre-scientific native IO acceptance')
    with c.corpus.r.run_lock(out/'active.lock'):
        plan=vr.read(out/'plan.json');cfg=plan['configuration']
        require(vr.digest(cfg)==plan['configuration_sha256'],'Prior plan changed')
        require(all(vr.sha(out/n)==h for n,h in cfg['frozen_files'].items()),'Frozen data changed')
        evidence={str(x.relative_to(out)):vr.sha(x) for x in out.rglob('request_*.json')}
        old_plan=json.loads(json.dumps(plan));cfg['code']={n:vr.sha(c.ROOT/n) for n in SOURCES}
        new=dict(configuration=cfg,configuration_sha256=vr.digest(cfg))
        record=dict(original_plan=old_plan,new_configuration_sha256=new['configuration_sha256'],
            retained_failed_probe_files=evidence,scientific_calls_before_repair=0,
            reason='Native vLLM dropped assistant.content preceding tool calls; personal process adapter restores analysis channel without changing model, prompt, tools or limits.')
        repair_path=out/'engineering_repair.json'
        if repair_path.exists():
            prior=vr.read(repair_path)
            require(prior['new_configuration_sha256']==old_plan['configuration_sha256'], 'Repair chain broken')
            prior.setdefault('amendments',[]).append(dict(previous_configuration_sha256=old_plan['configuration_sha256'],
                next_configuration_sha256=new['configuration_sha256'],code=cfg['code'],
                reason='Pre-scientific engineering completion: service readiness and derived-material guards; no scientific requests, inputs, prompts or budgets changed'))
            prior['new_configuration_sha256']=new['configuration_sha256'];record=prior
        vr.atomic(repair_path,record)
        vr.atomic(out/'plan.json',new)
        verify(out)


def emit(out, stage, **values):
    vr.atomic(out/'execution_status.json',dict(state=stage,updated_unix=time.time(),**values))
    print(json.dumps(dict(state=stage,**values),ensure_ascii=False),flush=True)


def confirm(out):
    plan=verify(out);ready=vr.read(out/'disconnect_ready.json')
    require(c.corpus.runtime.process_identity(ready['pid'])==ready['process_start'] and
        ready['configuration_sha256']==plan['configuration_sha256'] and time.time()-ready['unix']>=5,'Reconnect verification required')
    vr.atomic(out/'disconnect_acceptance.json',dict(ready,passed=True,confirmed_unix=time.time()))


def model_json(out,journal,tokenizer,stage,key,messages):
    cfg=journal.config
    body=c.corpus.r.request_body(cfg['model'],messages,cfg['max_tokens'],backend='vllm')
    body.update(stream=False,return_token_ids=True,reasoning_effort='medium');body.pop('stream_options',None)
    require(checking.token_count(tokenizer,messages,None)+cfg['max_tokens']<=131072,'Construction request too large')
    raw,ev=journal.call(stage,key,body)
    effective=validate_effective_input(raw,tokenizer,cfg['fixed_date'])
    try:
        text=c.corpus.harmony.decode_response(raw,model=cfg['model'],max_tokens=cfg['max_tokens'],native_tools=None)['content'].strip()
    except ValueError as exc:
        if str(exc) not in {'Incomplete model output','No authoritative final/tool termination',
                            'Incomplete Harmony message','Missing final message'}:raise
        value={'_content_failure':str(exc)};edits={}
    else:
        try:value,edits=checking.parsed(text)
        except (ValueError,TypeError) as exc:value={'_content_failure':str(exc)};edits={}
    vr.atomic(out/'decoded'/stage/(vr.digest(key)+'.json'),dict(value=value,evidence=ev,effective_input=effective,serialization=edits))
    return value,ev


def native_probe(out,cfg,tokenizer):
    """Engineering IO only, before lengthy construction; no benchmark outputs."""
    folder=out/'native_io'/vr.sha(c.ROOT/'scripts/serve_continuous_context_vllm.py')[:16]
    receipt=folder/'acceptance.json'
    if receipt.exists():
        saved=vr.read(receipt)
        require(saved['configuration_sha256']==vr.digest(cfg) and saved['passed'],'Probe binding drift')
        require(all(vr.sha(folder/n)==h for n,h in saved['files'].items()),'Probe raw drift')
        return
    case=min(vr.read(out/'cases.json'),key=lambda d:len(d['text']))
    history=[dict(role='system',content='Engineering tool transport test only. Follow the requested tool call exactly. No physics evaluation.'),
        dict(role='user',content='Call read_memory with id='+case['id']+'.')]
    journal=c.corpus.Journal(folder,cfg['transport'],threading.Event());evidence={}
    for i,name in enumerate(('read_memory','search_memory')):
        body=c.corpus.r.request_body(cfg['transport']['model'],history,4096,backend='vllm',native_tools=protocol.tools())
        body.update(stream=False,return_token_ids=True,reasoning_effort='medium',
                    tool_choice=dict(type='function',function=dict(name=name)))
        body.pop('stream_options',None)
        raw,ev=journal.call('probe',str(i),body);evidence[ev['file']]=ev['sha256']
        validate_effective_input(raw,tokenizer,cfg['transport']['fixed_date'])
        actual=tokenizer.decode(raw['prompt_token_ids'],skip_special_tokens=False)
        require(all((m.get('content') or '') in actual for m in history),'Native history lost text/analysis')
        decoded=c.corpus.harmony.decode_response(raw,model=cfg['transport']['model'],max_tokens=4096,native_tools=protocol.tools())
        require(bool(decoded.get('tool_calls')),'Forced native tool missing')
        call=decoded['tool_calls'][0];args=json.loads(call['function']['arguments'])
        require(call['function']['name']==name and args==({'id':case['id']} if i==0 else {'query':'unit conversion','offset':0}),
                'Native tool arguments did not round trip')
        history.append(protocol.assistant_tool_message(decoded))
        history.append(dict(role='tool',name=name,tool_call_id=call['id'],
            content=json.dumps({k:case[k] for k in ('id','title','text')} if i==0 else {'items':[]},ensure_ascii=False).replace('<|','\\u003c|')))
        if i==0:history.append(dict(role='user',content='Now call search_memory with query="unit conversion" and offset=0.'))
    vr.atomic(folder/'history.json',history);evidence['history.json']=vr.sha(folder/'history.json')
    immutable(receipt,dict(passed=True,configuration_sha256=vr.digest(cfg),files=evidence,
        scope='native tool names/arguments, full tool text and preceding analysis in actual deployed prompt'))


def build_materials(out,cfg,tokenizer):
    journal=c.corpus.Journal(out,cfg['transport'],threading.Event())
    bundle=vr.read(out/'tree_bundle.json')
    progress=vr.Progress(out/'tree_progress.json',vr.digest([bundle,cfg]),tree.initial_state())
    def call(stage,key,payload):
        return model_json(out,journal,tokenizer,'tree_'+stage,key,tree.messages(stage,payload))[0]
    organizer=tree.Organizer(bundle,progress,call)
    while organizer.state['phase']!='done':
        require(not(out/'STOP').exists(),'STOP requested')
        organizer.step()
        emit(out,'building_tree',phase=organizer.state['phase'],jobs=organizer.state['jobs_committed'],
             placed=len(organizer.state['placed']),unresolved=len(organizer.state['unresolved']))
    products=organizer.products()
    immutable(out/'materials/catalog.json',products['catalog'])
    immutable(out/'materials/tree_report.json',{k:v for k,v in products.items() if k!='catalog'})
    cases=vr.read(out/'cases.json');outcomes=[]
    for i,case in enumerate(cases):
        require(not(out/'STOP').exists(),'STOP requested')
        path=out/'skill_outcomes'/(case['id']+'.json');signature=vr.digest([case,cfg])
        if path.exists():
            saved=vr.read(path);require(saved['signature']==signature,'Skill source drift')
            require(vr.sha(out/saved['evidence']['file'])==saved['evidence']['sha256'],'Skill raw drift')
        else:
            value,ev=model_json(out,journal,tokenizer,'skills',case['id'],skills.skill_messages(case))
            try:skills.validate_skills(value);state='collected';reason=None
            except ValueError as exc:state='content_failure';reason=str(exc)
            saved=dict(id=case['id'],signature=signature,status=state,value=value,reason=reason,evidence=ev)
            vr.atomic(path,saved)
        outcomes.append(saved)
        emit(out,'extracting_flat_skills',processed=i+1,total=len(cases),content_failures=sum(x['status']=='content_failure' for x in outcomes))
    docs,counts=skills.materialize_skills(cases,outcomes)
    # Convert case-document IDs back to actual question provenance, not labels.
    origins={x['id']:x['source_ids'] for x in cases}
    for d in docs:d['source_ids']=sorted({sid for cid in d['source_ids'] for sid in origins[cid]})
    immutable(out/'materials/skills.json',docs);immutable(out/'materials/skills_report.json',counts)
    immutable(out/'materials/cases.json',cases)
    return products


def accept_materials(out,cfg):
    receipt=out/'construction_acceptance.json'
    paths=[out/'materials'/n for n in ['catalog.json','tree_report.json','skills.json','skills_report.json','cases.json']]
    hashes={str(x.relative_to(out)):vr.sha(x) for x in paths}
    if receipt.exists():
        saved=vr.read(receipt);require(saved['passed'] and saved['files']==hashes,'Material acceptance drift');return
    verify(out,full=True)
    bundle=vr.read(out/'tree_bundle.json');catalog=vr.read(paths[0]);report=vr.read(paths[1])
    require(report['counts']['all_candidates_preserved'] and report['counts']['placed_rules']>0,'No usable conserved tree')
    cases=vr.read(out/'materials/cases.json');flat=vr.read(out/'materials/skills.json')
    require(bool(cases) and bool(flat),'Empty auxiliary store; inspect construction failure')
    allowed={sid for d in cases for sid in d['source_ids']}
    require(all(set(d['source_ids'])<=allowed for d in flat),'Skill provenance escaped common experience pool')
    root=Path(cfg['root']);hipho=vr.read(root.parent/'hipho_overlap_audit.json')
    external=isolation.Audit(hipho['question_references'])
    internal=isolation.Audit(vr.read(out/'heldout_questions.json'))
    tree_memory=TreeMemory(catalog)
    # Screen visible material, including generated text, once at publication.
    # Already reviewed HiPhO generic lexical source matches remain explicit.
    reviewed={(x['source_id'],hid) for x in hipho['candidate_review']['decisions']
              if x['cohort']=='current_build_2000' and x['decision']=='no_same_question_evidence_generic_lexical_match'
              for hid in x['heldout_ids']}
    sources={r['rule_id']:r['sample_ids'] for r in bundle['rules']}
    rule_docs=[dict(d,source_ids=sources[d['id']]) for d in tree_memory.memory.docs.values()]
    def descendants(nid):
        node=tree_memory.corpus.nodes[nid]
        return set(node.get('rule_ids',[])) | {rid for child in node.get('children',[]) for rid in descendants(child)}
    node_docs=[dict(d,source_ids=sorted({sid for rid in descendants(nid) for sid in sources[rid]}))
               for nid,d in tree_memory.nodes.docs.items()]
    checks=[]
    for doc in cases+flat+rule_docs+node_docs:
        row=dict(id=doc['id'],question='',context='',prediction=doc['title']+'\n'+doc['text'],answer='')
        own=internal.matches(row);ext=external.matches(row)
        # Prior generic source matches do not approve newly generated passages.
        # Only RAG's verbatim original fields may inherit the source review;
        # its fresh audit prose and all distilled/navigation text are re-screened.
        if doc['kind']=='case':
            fresh=doc['text'].split('\n\n120B原始核查说明\n',1)[1]
            fresh_hits=external.matches(dict(row,prediction=fresh))
            new=[h for h in ext if not any((sid,h['heldout_id']) in reviewed for sid in doc['source_ids'])]+fresh_hits
        else:new=ext
        if own or ext:checks.append(dict(id=doc['id'],source_ids=doc['source_ids'],internal=own,external=ext,unreviewed_external=new))
    unresolved=[x for x in checks if x['internal'] or x['unreviewed_external']]
    vr.atomic(out/'material_isolation.json',dict(matches=checks,unresolved=unresolved,
        source_audit_sha256=vr.sha(root.parent/'hipho_overlap_audit.json'),semantic_independence_proven=False))
    require(not unresolved,'New derived-material overlap candidates require review; frozen cohorts unchanged')
    immutable(receipt,dict(passed=True,files=hashes,tree_counts=report['counts'],cases=len(cases),skills=len(flat),
        common_error_sources=len(allowed),human_gold=False,physical_quality_certified=False,
        scope='source/structure/conservation/isolation/engineering; automatic model content errors retained',
        isolation_sha256=vr.sha(out/'material_isolation.json')))


def memory_set(out):
    return dict(pure=None,case_rag=FlatMemory(vr.read(out/'materials/cases.json')),
        flat_skills=FlatMemory(vr.read(out/'materials/skills.json')),
        rule_tree=TreeMemory(vr.read(out/'materials/catalog.json')))


def wait_service(out,cfg):
    """Startup is a dependency, not a failed experiment or repeated model call."""
    deadline=time.monotonic()+7200
    while True:
        require(not(out/'STOP').exists(),'STOP requested')
        try:
            with httpx.Client(trust_env=False,timeout=10) as client:
                response=client.get(cfg['transport']['auditor_api']+'/models')
                if response.status_code not in {502,503,504}:response.raise_for_status()
                if response.status_code==200:
                    require(any(x['id']==cfg['transport']['model'] and x.get('max_model_len',0)>=131072
                        for x in response.json()['data']),'Model/context identity mismatch')
                    return
        except httpx.TransportError:
            pass
        startup=out/'service/startup.json'
        require(startup.exists(),'No recorded personal service to wait for')
        pid=vr.read(startup)['pid']
        cmd=Path(f'/proc/{pid}/cmdline')
        require(cmd.exists() and b'scripts.serve_continuous_context_vllm' in cmd.read_bytes(),
                'Personal model service exited; do not restart or claim success blindly')
        require(time.monotonic()<deadline,'Personal service loading exceeded two hours')
        emit(out,'waiting_personal_service',service_pid=pid,scientific_checks_started=False,readiness_only=True)
        time.sleep(30)


def check_condition(out,cfg):
    condition=dict(transport=cfg['transport'],limits=cfg['limits'],protocol=cfg['protocol'],
        material_receipt_sha256=vr.sha(out/'construction_acceptance.json'),code=cfg['code'])
    if cfg.get('check_prompt_revision'):condition['check_prompt_revision']=cfg['check_prompt_revision']
    return condition


def checks(out,cfg,tokenizer,task,phase,rows,memories):
    condition=check_condition(out,cfg)
    results={a:[] for a in protocol.ARMS}
    for i,row in enumerate(rows):
        arms=protocol.ARMS[i%4:]+protocol.ARMS[:i%4]
        for arm in arms:
            require(not(out/'STOP').exists(),'STOP requested')
            if cfg.get('parent_pipeline'):
                require(not(Path(cfg['parent_pipeline'])/'STOP').exists(),'Parent STOP requested')
            active_condition=cfg['pure_pilot_condition'] if (cfg.get('check_prompt_revision') and
                task=='step' and phase=='pilot' and arm=='pure') else condition
            if task=='step' and phase=='pilot':
                active_condition=cfg.get('pilot_reuse_conditions',{}).get(arm,active_condition)
            result=checking.run_case(out/'checks'/task/phase/arm/row['id'],row,arm,task,memories[arm],active_condition,tokenizer)
            results[arm].append(result)
            emit(out,'checking_'+task+'_'+phase,question_index=i+1,total=len(rows),arm=arm,outcome=result['status'])
    if phase=='pilot':
        # Content/physics errors are observations, not reasons to select a better
        # output. A budget that prevents an actual complete run needs a decision.
        blocked=[dict(id=r['id'],arm=a,status=r['status']) for a,rs in results.items() for r in rs
                 if r['status'] in {'budget_exhausted','technical_failure'}]
        per_arm={a:dict(completed=sum(r['status']=='complete' for r in rs),
            searched=sum(r['protocol_audit']['searches']>0 for r in rs),
            read=sum(bool(r['protocol_audit']['reads']) for r in rs)) for a,rs in results.items()}
        passed=not blocked and all(v['completed'] for v in per_arm.values()) and all(
            v['read']>0 and v['searched']>0 for a,v in per_arm.items() if a!='pure')
        vr.atomic(out/(task+'_pilot_acceptance.json'),dict(passed=passed,blocked=blocked,per_arm=per_arm,
            scope='engineering only; physics accuracy and score superiority not gates',
            output_files={str((out/'checks'/task/phase/a/r['id']/'result.json').relative_to(out)):
                vr.sha(out/'checks'/task/phase/a/r['id']/'result.json') for a,rs in results.items() for r in rs}))
        require(passed,'Pilot engineering/budget boundary did not pass; no silent budget/protocol change')
    return results


def score(out,task,rows,results):
    all_scores={}
    for arm,values in results.items():
        if task=='final':
            all_scores[arm]=protocol.final_scores(rows,vr.read(out/'final_labels_private.json'),{x['id']:x for x in values})
        else:
            pred=[dict(id=x['id'],diagnostics=x['value']['diagnostics'] if x['value'] else [],
                execution_failure=x['status']!='complete',source_status=x['status'],
                protocol_noncompliance=x['protocol_audit']['protocol_noncompliance']) for x in values]
            all_scores[arm]=scoring.score(vr.read(out/'step_reference.json'),pred,vr.read(out/'step_unresolved.json'),preserve_zero_offsets=True)
        all_scores[arm]['execution']=dict(outcomes=dict(Counter(x['status'] for x in values)),
            protocol_noncompliance=sum(x['protocol_audit']['protocol_noncompliance'] for x in values),
            requests=sum(x['usage']['requests'] for x in values),
            input_tokens=sum(x['usage']['input_tokens'] for x in values),
            output_tokens_or_unknown_reservation=sum(x['usage']['output_tokens_or_unknown_reservation'] for x in values))
    vr.atomic(out/(task+'_metrics.json'),dict(task=task,human_gold=False,groups=all_scores,
        source_results={str((out/'checks'/task/'formal'/a/r['id']/'result.json').relative_to(out)):
            vr.sha(out/'checks'/task/'formal'/a/r['id']/'result.json') for a,rs in results.items() for r in rs},
        reference_release='step_label_release.json' if task=='step' else 'final_labels_private.json'))
    # Deterministic examples: first two disagreements in frozen input order,
    # not selected for a preferred method. Preserve raw output JSON and all arms.
    indexed={a:{x['id']:x for x in rs} for a,rs in results.items()};examples=[]
    for row in rows:
        outputs={a:indexed[a][row['id']] for a in protocol.ARMS}
        keys=[x['value'].get('verdict') if task=='final' and x['value'] else
              len(x['value']['diagnostics']) if x['value'] else None for x in outputs.values()]
        if len(set(keys))>1:
            examples.append(dict(input=row,outputs=outputs))
        if len(examples)==2:break
    vr.atomic(out/(task+'_examples.json'),dict(selection='first two count/verdict disagreements in frozen order',examples=examples))


def run(out):
    require(Path('/proc').exists(),'Formal execution is server-only')
    root=Path(vr.read(out/'plan.json')['configuration']['root'])
    with c.corpus.r.run_lock(out/'active.lock'),c.corpus.r.run_lock(root/'organization_service_18083.lock'):
        plan=verify(out);cfg=plan['configuration'];code=1
        (out/'runner.pid').write_text(str(os.getpid()))
        try:
            c.corpus.runtime.wait_detached(out,plan['configuration_sha256'])
            for name,h in cfg['model_files'].items():require(vr.sha(Path(cfg['transport']['model'])/name)==h,'Model files drift')
            wait_service(out,cfg)
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(cfg['transport']['model'],local_files_only=True)
            require(not(out/'STOP').exists(),'STOP requested')
            if not cfg.get('check_prompt_revision'):
                emit(out,'native_transport_probe');native_probe(out,cfg,tokenizer)
                build_materials(out,cfg,tokenizer)
            else:
                require(not(Path(cfg['parent_pipeline'])/'STOP').exists(),'Parent STOP requested')
                emit(out,'reuse_frozen_materials_and_native_io',construction_model_calls=0)
            emit(out,'accepting_materials');accept_materials(out,cfg)
            memories=memory_set(out)
            for task in ('step','final'):
                checks(out,cfg,tokenizer,task,'pilot',vr.read(out/'pilot_inputs.json'),memories)
                rows=vr.read(out/(task+'_inputs.json'))
                results=checks(out,cfg,tokenizer,task,'formal',rows,memories)
                score(out,task,rows,results)
                emit(out,task+'_scored',questions=len(rows))
            verify(out,full=True)
            failures={t:{a:v['execution']['outcomes'] for a,v in vr.read(out/(t+'_metrics.json'))['groups'].items()}
                      for t in ('step','final')}
            emit(out,'both_experiments_processed',outcomes=failures,
                 note='Inspect coverage, retained failures and machine-label flags before scientific claims')
            code=0
        except BaseException as exc:
            emit(out,'stopped',error_type=type(exc).__name__,error=str(exc)[:2000]);raise
        finally:
            (out/'runner.exit_code').write_text(str(code));(out/'runner.pid').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','verify','run','confirm-detached','refresh-unstarted','repair-engineering','prepare-check-revision','repair-local-rendering'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--root',type=Path)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--revision',choices=[protocol.CHECK_REVISION,protocol.LANGUAGE_REVISION,protocol.BILINGUAL_REVISION],default=protocol.CHECK_REVISION)
    args=parser.parse_args();out=args.output.resolve()
    if args.mode=='prepare':prepare(out,args.root.resolve())
    elif args.mode=='prepare-check-revision':prepare_check_revision(out,args.parent.resolve(),args.revision)
    elif args.mode=='repair-local-rendering':repair_local_rendering(out)
    elif args.mode=='verify':verify(out);print('verified')
    elif args.mode=='confirm-detached':confirm(out)
    elif args.mode=='refresh-unstarted':refresh_unstarted(out)
    elif args.mode=='repair-engineering':repair_engineering(out)
    else:run(out)


if __name__=='__main__':main()
