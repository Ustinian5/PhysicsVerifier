"""Prepare/run/score the 99-question original-case / procedural-skill comparison.

prepare and verify are offline. Inference runs only on the personal server,
detached with reconnection acceptance. Reuses the validated 50-question checking
engine, native-format failure accounting, and unchanged location scorer.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import httpx

from rule_framework import organization_bench99 as protocol
from scripts import run_organization_comparison as base
from scripts import resume_organization_comparison as recovery

vr, require, ROOT = base.vr, base.require, base.ROOT
SOURCE_FILES = sorted(set(base.SOURCE_FILES) | {'rule_framework/organization_bench99.py',
    'scripts/run_organization_bench99.py', 'scripts/resume_organization_comparison.py'})


def source_hashes():
    return {name: vr.sha(ROOT / name) for name in SOURCE_FILES}


def bound_artifact(manifest, path):
    """Resolve a unique published artifact by suffix across the local/server mirror."""
    matches = [sha for name, sha in manifest['files'].items()
               if name.endswith('/' + path.parent.name + '/' + path.name)]
    require(matches == [vr.sha(path)], 'Missing, ambiguous or changed published artifact: ' + str(path))


def prepare(output, parent, rule_tree):
    output.mkdir(parents=True, exist_ok=True)
    with base.r.run_lock(output / 'active.lock'):
        if (output / 'plan.json').exists(): return verify(output)
        prior = vr.read(parent / 'plan.json'); pc = prior['configuration']
        require(vr.digest(pc) == prior['configuration_sha256'] and pc['protocol'] == base.p.VERSION,
                'Expected the completed original-case comparison')
        require(pc['model']['model']==base.r.EXPECTED_MODEL and pc['model']['reasoning_effort']=='medium' and
                pc['model_files']==base.runtime.MODEL_FILES, 'Unexpected parent model/settings')
        require(vr.read(parent / 'execution_status.json')['state'] == 'completed', 'Parent is not completed')
        require(vr.sha(parent / 'sources.json') == pc['sources_sha256'], 'Source cohort drift')
        rows = vr.read(parent / 'sources.json')
        require(len(rows) == 1726 and len({r['id'] for r in rows}) == len(rows), 'Wrong original source cohort')
        release = vr.read(parent / 'experience/release_manifest.json')
        require(release['configuration_sha256'] == prior['configuration_sha256'], 'Wrong case release')
        cases_path = parent / 'experience/cases.json'; bound_artifact(release, cases_path)
        cases = vr.read(cases_path)
        require(len(cases) == len(rows) and {tuple(c['source_ids']) for c in cases} == {(r['id'],) for r in rows},
                'Need exactly one original case for every source')
        for row in rows:
            case = next(c for c in cases if c['source_ids'] == [row['id']])
            require(case['kind'] == 'case' and case['id'] == 'case_' + vr.digest(row)[:24], 'Case source binding drift')
        tree_release = vr.read(rule_tree / 'release_manifest.json')
        require(vr.sha(rule_tree / 'catalog.json') == pc['rules']['catalog_sha256'] and
                vr.sha(rule_tree / 'release_manifest.json') == pc['rules']['release_sha256'], 'Tree drift')
        bound_artifact(tree_release, rule_tree / 'materials.json')
        inputs, _, _ = protocol.load_bench(ROOT)
        heldout, registry = base.old.b.benchmark_questions()
        require(registry == pc['benchmarks'], 'Known benchmark registry changed; review before new preparation')
        screen = protocol.FamilyScreen(heldout)
        clean, excluded = screen.partition(rows)
        # Case notes can also contain a benchmark statement. Preserve all originals;
        # conservative exclusions apply equally to RAG, skills and rule provenance.
        excluded_ids = {x['source_id'] for x in excluded}
        case_by_source = {c['source_ids'][0]: c for c in cases}
        for row in clean:
            match = screen.matches(dict(row, prediction=case_by_source[row['id']]['text']))
            if match:
                excluded.append(dict(source_id=row['id'], matches=match, field='original_case_memory'))
                excluded_ids.add(row['id'])
        clean = sorted((r for r in clean if r['id'] not in excluded_ids), key=lambda r: vr.digest([protocol.SEED, r['id']]))
        require(clean and set(protocol.KNOWN_FAMILY['source_ids']) <= excluded_ids, 'Known family not excluded')
        kept_ids = {r['id'] for r in clean}
        clean_cases = [c for c in cases if c['source_ids'][0] in kept_ids]
        catalog, tree_changes = protocol.prune_catalog(vr.read(rule_tree / 'catalog.json'),
            vr.read(rule_tree / 'materials.json')['provenance'], excluded_ids)
        base.p.TreeMemory(catalog)  # Validate the exact serving representation.
        isolation = dict(policy=protocol.FAMILY_POLICY, original_sources=len(rows), kept_sources=len(clean),
            excluded=excluded, tree_changes=tree_changes, semantic_family_independence_proven=False,
            limitation='Known sibling and conservative text exclusions; not a complete semantic-family or pretraining audit.',
            human_gold=False, reference_answers_used=False)
        for name, value in [('sources', clean), ('inputs', inputs), ('cases', clean_cases),
                            ('catalog', catalog), ('isolation', isolation), ('bench_questions', heldout)]:
            vr.atomic(output / (name+'.json'), value)
        dependencies = [parent/'plan.json', parent/'sources.json', parent/'experience/release_manifest.json',
                        cases_path, rule_tree/'release_manifest.json', rule_tree/'catalog.json', rule_tree/'materials.json',
                        ROOT/protocol.INPUT, ROOT/base.evaluation.RELEASE, ROOT/base.evaluation.REFERENCE]
        ref_release = vr.read(ROOT/base.evaluation.RELEASE)
        dependencies += [(ROOT/base.evaluation.RELEASE).parent / n for n in ref_release['files']]
        dependencies += [ROOT / n for n in registry]
        files = {str(p.resolve()): vr.sha(p) for p in dependencies}
        files.update({str((output/(n+'.json')).resolve()): vr.sha(output/(n+'.json'))
                      for n in ('sources','inputs','cases','catalog','isolation','bench_questions')})
        c = dict(protocol=protocol.VERSION, model=copy.deepcopy(pc['model']), api_base=pc['api_base'],
            model_files=pc['model_files'], limits=pc['limits'], sample_count=99,
            sample_ids=[x['id'] for x in inputs], input_sha256=vr.sha(output/'inputs.json'),
            source_count=len(clean), parent_configuration_sha256=prior['configuration_sha256'],
            reference_version=base.evaluation.RELEASE_VERSION, reference_counts={'confirmed':402,'locatable':397,'unresolved':7},
            arms=list(base.p.ARMS), skill_prompt=protocol.SKILL_PROMPT, check_prompt=base.p.STAGED_BASE,
            query_max_tokens=base.p.QUERY_MAX_TOKENS, reference_token_cap=base.p.REFERENCE_TOKEN_CAP,
            max_items=base.p.MAX_ITEMS, tree_branches=base.p.TREE_BRANCHES,
            source_hashes=source_hashes(), files=files, controls=base.runtime.CONTROLS,
            budget_limits=pc['budget_limits'], prior_usage=base.old.ZERO_USAGE,
            order='build procedural skills; per-question rotation of all four arms including pure',
            skill_pilot='first 12 sources, semantic method-distinction review before remaining generation; no bench score access',
            rag='byte-identical original cases with source-family exclusions; no new case analysis',
            skills='flat independent natural-language procedures, not domain/trigger/operator records',
            failure_policy='known complete malformed/missing final retained; never semantic resampling; unknown stops',
            human_gold=False, independent_test=False, semantic_family_independence_proven=False)
        c['model'].update(protocol=protocol.VERSION, max_requests=6000)
        plan = dict(configuration=c, configuration_sha256=vr.digest(c))
        vr.atomic(output/'plan.json', plan)
        base.old.emit(output, 'prepared_no_model_calls', planned_checks=396, planned_sources=len(clean),
                      excluded_sources=len(excluded_ids), rules=tree_changes['kept_leaves'])
        return plan


def verify(output, full=False):
    plan = vr.read(output/'plan.json'); c = plan['configuration']
    require(c['protocol'] == protocol.VERSION and vr.digest(c) == plan['configuration_sha256'], 'Plan drift')
    require(c['source_hashes'] == source_hashes(), 'Code drift; do not silently adapt a running experiment')
    vr.EvidenceCache(output/'evidence_cache.json').check(c['files'], full=full)
    rows = vr.read(output/'inputs.json')
    require(len(rows)==c['sample_count']==99 and [r['id'] for r in rows]==c['sample_ids'], '99 input drift')
    require(c['skill_prompt']==protocol.SKILL_PROMPT and c['check_prompt']==base.p.STAGED_BASE and
            c['reference_token_cap']==base.p.REFERENCE_TOKEN_CAP and c['max_items']==base.p.MAX_ITEMS and
            c['query_max_tokens']==base.p.QUERY_MAX_TOKENS and c['tree_branches']==base.p.TREE_BRANCHES,
            'Frozen prompt/budget drift')
    return plan


class ProcedureJournal(base.old.SkillJournal):
    def __init__(self, folder, config, tokenizer):
        super().__init__(folder, config); self.tokenizer = tokenizer

    def messages(self, stage, payload):
        require(stage=='extract_skill', 'Procedural extraction only')
        return protocol.skill_messages(payload['case'])

    def decode_record(self, path, stage):
        try: return super().decode_record(path, stage)
        except ValueError as exc:
            raw = vr.read(path); response = raw.get('response') or {}
            if (str(exc)=='Missing final message' and raw.get('http_status')==200 and not raw.get('error') and
                response.get('model')==self.config['model'] and len(response.get('choices',[]))==1 and
                response['choices'][0].get('finish_reason')=='stop'):
                raise base.old.tree.ContentFailure('missing_final; no resampling') from exc
            raise

    def __call__(self, stage, key, payload):
        n=len(self.tokenizer.apply_chat_template(self.messages(stage,payload),tokenize=True,add_generation_prompt=True))
        if n+self.config['max_tokens'] > base.p.BUILD_LIMITS['context_window']:
            raise base.old.tree.ContentFailure('input_too_large; no truncation')
        return super().__call__(stage,key,payload)


class PilotReviewRequired(RuntimeError):
    pass


def check_pilot(output, rows, binding):
    rows=rows[:min(base.p.PILOT_SIZE,len(rows))];path=output/'skills/pilot_review.json'
    if not path.exists():raise PilotReviewRequired('Review the fixed skill pilot before continuing')
    review=vr.read(path)
    require(review.get('configuration_sha256')==binding and review.get('human_gold') is False and
            review.get('actor') in {'Codex assistant','human'} and review.get('decision')=='continue',
            'Invalid pilot review; do not use benchmark scores or impersonate human gold')
    entries=review.get('samples',[])
    require(len(entries)==len(rows) and {e['id'] for e in entries}=={r['id'] for r in rows},'Review every fixed pilot source')
    byid={e['id']:e for e in entries};accepted=0
    for row in rows:
        file=base.old.outcome_path(output/'skills',row);record=vr.read(file);entry=byid[row['id']]
        require(entry['outcome_sha256']==vr.sha(file) and isinstance(entry.get('notes'),str) and entry['notes'].strip(),
                'Pilot review needs bound output and concrete justification')
        outcome=record['outcome'];verdict=entry.get('method_distinction')
        if outcome['status']=='content_failure':require(verdict=='unassessable_output_failure','Failed output is not a valid skill')
        elif not outcome['value']['skills']:require(verdict=='valid_empty','Empty output must remain empty')
        else:
            require(verdict=='acceptable','Methods need mainline revision; no automatic regeneration')
            accepted+=1
    require(accepted>0,'Pilot contains no assessable procedural skills; review the protocol')


def build(output, plan, tokenizer):
    folder=output/'skills';folder.mkdir(exist_ok=True)
    release_path=folder/'release_manifest.json'; binding=plan['configuration_sha256']
    if release_path.exists():
        release=vr.read(release_path);require(release['configuration_sha256']==binding,'Skill release drift')
        vr.EvidenceCache(folder/'evidence_cache.json').check(release['files'])
        return base.p.FlatMemory(vr.read(folder/'skills.json'))
    rows=vr.read(output/'sources.json'); cases={x['source_ids'][0]:x for x in vr.read(output/'cases.json')}
    journal=ProcedureJournal(folder,plan['configuration'],tokenizer)
    vr.EvidenceCache(folder/'evidence_cache.json').check(journal.index['files'])
    progress=vr.Progress(folder/'collection_checkpoint.json',binding,{'cursor':0,'outcome_chain':''})
    pilot_count=min(base.p.PILOT_SIZE,len(rows))
    if progress.state['cursor']>=pilot_count:check_pilot(output,rows,binding)
    while progress.state['cursor']<len(rows):
        require(not (output/'STOP').exists(),'STOP requested')
        row=rows[progress.state['cursor']];path=base.old.outcome_path(folder,row)
        if path.exists():
            record=vr.read(path)
            require(record['configuration_sha256']==binding and record['source_sha256']==vr.digest(row),'Source drift')
        else:
            try:
                value=journal('extract_skill',row['id'],{'case':cases[row['id']]})
                try:protocol.validate_skills(value)
                except (ValueError,TypeError,KeyError) as exc:raise base.old.tree.ContentFailure(str(exc)) from exc
                outcome=dict(id=row['id'],status='collected',value=value)
            except base.old.tree.ContentFailure as exc:
                outcome=dict(id=row['id'],status='content_failure',reason=str(exc))
            record=dict(configuration_sha256=binding,source_sha256=vr.digest(row),outcome=outcome)
            vr.atomic(path,record)
        journal.index['files'][str(path)]=vr.sha(path);vr.seal(folder/'request_index.json',journal.index)
        progress.state['outcome_chain']=vr.digest([progress.state['outcome_chain'],vr.digest(record)])
        progress.state['cursor']+=1;progress.save()
        if progress.state['cursor']==pilot_count:
            pilot,_=base.old.read_outcomes(folder,rows[:progress.state['cursor']],binding)
            vr.atomic(folder/'pilot_summary.json',dict(configuration_sha256=binding,
                counts=dict(Counter(x['status'] for x in pilot)),semantic_quality_verified=False,
                policy='schema observations only; failures retained, no selection by score'))
            check_pilot(output,rows,binding)
        base.old.emit(output,'building_procedural_skills',processed=progress.state['cursor'],planned=len(rows))
    outcomes,chain=base.old.read_outcomes(folder,rows,binding)
    require(chain==progress.state['outcome_chain'],'Outcome chain changed')
    skills,counts=protocol.materialize_skills(rows,outcomes)
    screen=protocol.FamilyScreen(vr.read(output/'bench_questions.json'))
    by_source={r['id']:dict(r) for r in rows}
    for item in skills:
        for sid in item['source_ids']:by_source[sid]['prediction']+='\n'+item['title']+'\n'+item['text']
    _,excluded=screen.partition(list(by_source.values()))
    require(not excluded,'Generated content includes held-out material; retain evidence and stop')
    vr.atomic(folder/'skills.json',skills);vr.atomic(folder/'summary.json',counts)
    files=dict(journal.index['files'])
    files.update({str(folder/name):vr.sha(folder/name) for name in
                  ('skills.json','summary.json','request_index.json','collection_checkpoint.json',
                   'pilot_summary.json','pilot_review.json')})
    vr.atomic(release_path,dict(configuration_sha256=binding,files=files,counts=counts,human_gold=False))
    return base.p.FlatMemory(skills)


def run_condition(output,row,phase,arm,memory,plan,tokenizer,client):
    try:return base.run_condition(output,row,phase,arm,memory,plan,tokenizer,client)
    except ValueError:
        path=base.path_for(output,phase,arm,row)
        if not path.exists():raise
        value=recovery.retained_failure(path,output_amended=True)
        require(value['configuration_sha256']==plan['configuration_sha256'],'Failure binding changed')
        base.old.emit(output,phase,arm=arm,sample_id=row['id'],result_status='failed_content_preserved')
        return value


def score(output,plan=None):
    plan=plan or verify(output);rows,dataset,unknown=protocol.load_bench(ROOT)
    require(rows==vr.read(output/'inputs.json'),'Scoring inputs differ from generation')
    groups,scores,usage={}, {}, {};all_requests=set()
    for arm in base.p.ARMS:
        items=[];provided=reference_tokens=0;requests=set();supplemental=[]
        for row in rows:
            path=base.path_for(output,'formal',arm,row)
            if not path.exists():continue
            x=vr.read(path);base.runtime.audit_result(x)
            require(x['configuration_sha256']==plan['configuration_sha256'],'Mixed experiment outputs')
            if not base.runtime.healthy_result(x):
                x=recovery.retained_failure(path,output_amended=True)
                if not base.complete_content_failure(x):
                    extra=recovery.account(path);supplemental.append(extra);requests.add(extra['request_path'])
            packet=vr.read(path.parent/'reference_packet.json')['packet']['retrieval']
            provided+=bool(packet['provided_complete_ids']);reference_tokens+=packet['reference_tokens']
            requests.update(e['request_path'] for e in x['trace']['response_evidence']);items.append(x)
        groups[arm]=items;all_requests.update(requests)
        usage[arm]=dict(completed_records=len(items),questions_with_full_references=provided,
            reference_tokens=reference_tokens,distinct_actual_requests=len(requests),supplemental_failure_usage=supplemental)
        if len(items)==99:
            scores[arm]=base.evaluation.score(dataset,[dict(x,execution_failure=not base.runtime.healthy_result(x),
                source_status=x['status'],protocol_noncompliance=not x.get('protocol_audit',{}).get('passed',False))
                for x in items],unknown)
    controls=recovery.control_summary(output)
    report=dict(configuration_sha256=plan['configuration_sha256'],sample_count=99,scores_all_99=scores,
        recorded_tasks={a:len(v) for a,v in groups.items()},usage=usage,reference_version=base.evaluation.RELEASE_VERSION,
        engineering_controls=controls,all_engineering_controls_passed=all(v['all_passed'] for v in controls.values()),
        formal_actual_requests_shared_prefix_counted_once=len(all_requests),
        metric='location_proxy_not_semantic_correctness',human_gold=False,independent_test=False,
        comparisons={a:base.evaluation.bootstrap(s,scores['pure']) for a,s in scores.items() if a!='pure' and 'pure'in scores})
    vr.atomic(output/'metrics.json',report)
    lines=['条件 | 完成记录 | Precision | Recall | F1','--- | ---: | ---: | ---: | ---:']
    for a in base.p.ARMS:
        s=scores.get(a,{}).get('summary')
        values=' | '.join(f'{100*s[k]:.2f}%' for k in ('precision','recall','f1')) if s else '待完成 | 待完成 | 待完成'
        lines.append(f'{a} | {len(groups[a])}/99 | {values}')
    (output/'metrics.md').write_text('\n'.join(lines)+'\n')
    if len(scores)==4 and not (output/'case_review.json').exists():
        selections=[]
        for alternative in ('pure','case_rag','flat_skills'):
            selected=base.old.protocol.case_selection(scores['rule_tree'],scores[alternative])
            selections.extend(dict(x,comparison=['rule_tree',alternative]) for x in selected['selection'])
        byid={r['id']:r for r in rows};refs={r['id']:r['physics_error_gt'] for r in dataset}
        outputs={a:{r['id']:r for r in items} for a,items in groups.items()}
        vr.atomic(output/'case_review.json',dict(configuration_sha256=plan['configuration_sha256'],human_gold=False,
            cases=[dict(x,input=byid[x['id']],reference=refs[x['id']],outputs={a:outputs[a][x['id']] for a in base.p.ARMS},
                        review=dict(actor='Codex assistant',status='pending',human_gold=False)) for x in selections]))
    return report


def run(output):
    with base.r.run_lock(output/'active.lock'),base.r.run_lock(output.parent/'organization_service_18083.lock'):
        plan=verify(output);c=plan['configuration']
        if vr.read(output/'execution_status.json')['state']=='completed':score(output,plan);return
        base.old.bind_runtime(output/'runtime.json',base.old.server_preflight(output,dict(c,model=base.r.EXPECTED_MODEL)))
        (output/'runner.pid').write_text(str(os.getpid()));exit_code=1
        try:
            base.old.b.wait_detached(output,plan['configuration_sha256'])
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(base.r.EXPECTED_MODEL,local_files_only=True,trust_remote_code=False)
            memory={'flat_skills':build(output,plan,tokenizer),
                    'case_rag':base.p.FlatMemory(vr.read(output/'cases.json')),
                    'rule_tree':base.p.TreeMemory(vr.read(output/'catalog.json')),'pure':None}
            with base.old.comparison.execution.configured(),patch.object(base.runtime.transport,'decode_response',base.decode_response), \
                 httpx.Client(trust_env=False) as client:
                for phase,rows in [('controls',base.runtime.CONTROLS),('formal',vr.read(output/'inputs.json'))]:
                    for i,row in enumerate(rows):
                        arms=list(base.p.ARMS);arms=arms[i%4:]+arms[:i%4]
                        for arm in arms:
                            require(not (output/'STOP').exists(),'STOP requested')
                            run_condition(output,row,phase,arm,memory[arm],plan,tokenizer,client)
                        base.old.emit(output,phase,completed_questions=i+1,planned_questions=len(rows))
                    score(output,plan)
            base.old.emit(output,'completed',recorded_tasks=396,case_review='pending');exit_code=0
        except PilotReviewRequired as exc:
            base.old.emit(output,'awaiting_skill_pilot_review',reason=str(exc),
                          review_path=str(output/'skills/pilot_review.json'),planned_checks=396);exit_code=0
        except BaseException as exc:
            base.old.emit(output,'stopped',error_type=type(exc).__name__,error=str(exc)[:1200]);raise
        finally:
            (output/'runner.exit_code').write_text(str(exit_code)+'\n');(output/'runner.pid').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','verify','run','score','status','confirm-detached'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent-comparison',type=Path)
    parser.add_argument('--rule-tree',type=Path)
    parser.add_argument('--full',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if args.mode=='prepare':
        require(args.parent_comparison is not None and args.rule_tree is not None,'Parent comparison and rule tree required')
        plan=prepare(out,args.parent_comparison.resolve(),args.rule_tree.resolve());print(plan['configuration_sha256'])
    elif args.mode=='verify':print(verify(out,args.full)['configuration_sha256'])
    elif args.mode=='run':run(out)
    elif args.mode=='score':print(json.dumps(score(out)['recorded_tasks']))
    elif args.mode=='confirm-detached':
        plan=verify(out)
        with patch.object(base,'verify',lambda path:plan):base.confirm(out)
    else:print(json.dumps(vr.read(out/'execution_status.json'),ensure_ascii=False))


if __name__=='__main__':main()
