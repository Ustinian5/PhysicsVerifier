"""Compare a direct-LLM baseline against frozen retrieval-proxy metrics, without API calls.

Reconstructs the original quote-only GT representation on purpose. A later reviewed
location policy must rescore every arm together, not silently improve this baseline.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_physics_eval_sets import _locate_quote_span
from scripts.run_retrieval_comparison import write_json
from scripts.experiment_manifest import sha256_file


def reference_dataset(root, reference):
    for field, name in [('blind_review_sha256','review/blind_review.json'),
                        ('operator_mapping_sha256','review/operator_map.json'),
                        ('frozen_inputs_sha256','frozen50/inputs.json')]:
        if sha256_file(root / name) != reference['sample'][field]:
            raise ValueError('Frozen scoring source changed: ' + name)
    packet=json.loads((root/'review/blind_review.json').read_text())
    mapping=json.loads((root/'review/operator_map.json').read_text())
    rows=json.loads((root/'frozen50/inputs.json').read_text())
    indexed={row['id']:dict(row,physics_error_gt=[]) for row in rows}
    for claim in packet['reference_claims']:
        sid=mapping['reference_claims'][claim['claim_id']]['sample_id']
        row=indexed[sid]; quote=claim['reported_quote']
        if claim['reported_offset_matches_original']:
            loc=dict(start_char=claim['reported_start_char'],end_char=claim['reported_end_char'],span_valid=True)
        else:
            loc=_locate_quote_span(row['prediction'],quote)
        row['physics_error_gt'].append(dict(error_id=claim['claim_id'],error_text=claim['unreviewed_error_claim'],
            answer_quote=quote,start_char=loc['start_char'],end_char=loc['end_char'],span_valid=loc['span_valid'],
            paragraph_index=-1,paragraph_valid=False))
    # Confirm not merely GT counts, but exact locations of all previously matched claims.
    gt={g['error_id']:g for row in indexed.values() for g in row['physics_error_gt']}
    for arm in reference['paired_49'].values():
        for row in arm['details']:
            for match in row['location_matches']:
                item=gt[match['gt_error_id']]
                if [item['start_char'],item['end_char']] != match['gt_span']:
                    raise ValueError('GT span reconstruction differs from frozen proxy')
    return list(indexed.values())


def evaluate(dataset, predictions, tmp, name):
    ds=tmp/(name+'_dataset.json');pred=tmp/(name+'_predictions.json');audit=tmp/'empty_audit.json';out=tmp/(name+'_metrics.json')
    write_json(ds,dataset);write_json(pred,predictions);write_json(audit,[])
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_physics_eval_sets.py'),'--dataset',str(ds),
        '--results',str(pred),'--audit',str(audit),'--output',str(out),'--match-mode','location',
        '--location-iou-threshold','0.5','--location-coverage-threshold','0.6'],check=True,stdout=subprocess.DEVNULL)
    return json.loads(out.read_text())



def paired_bootstrap(direct, prior, repetitions=20000, seed=20260924):
    ids=[r['id'] for r in direct['details']]
    base={r['id']:r for r in direct['details']}
    matrices=[]
    arms=list(prior)
    for group in [direct,*[prior[a] for a in arms]]:
        mapping={r['id']:r for r in group['details']}
        if set(mapping)!=set(ids):raise ValueError('Bootstrap groups have different samples')
        values=[]
        for sid in ids:
            r=mapping[sid]
            if r['gt_error_count']!=base[sid]['gt_error_count']:raise ValueError('Bootstrap GT differs')
            values.append((r['matched_error_count'],r['pred_locatable_finding_count'],r['gt_error_count']))
        matrices.append(values)
    rng=random.Random(seed);diffs=[[] for _ in arms]
    for _ in range(repetitions):
        chosen=rng.choices(range(len(ids)),k=len(ids));scores=[]
        for matrix in matrices:
            tp=sum(matrix[i][0] for i in chosen);den=sum(matrix[i][1]+matrix[i][2] for i in chosen)
            scores.append(200*tp/den if den else 0.)
        for i in range(len(arms)):diffs[i].append(scores[0]-scores[i+1])
    def percentile(values,q):
        values=sorted(values);position=(len(values)-1)*q;i=int(position);fraction=position-i
        return values[i]*(1-fraction)+values[min(i+1,len(values)-1)]*fraction
    return dict(unit='question-level paired resampling',generator='Python random.Random.choices',seed=seed,
        replicates=repetitions,delta_f1_percentage_points_ci95={arms[i]:[round(percentile(d,.025),2),round(percentile(d,.975),2)] for i,d in enumerate(diffs)})


def all_diagnostics_sensitivity(metrics, predictions):
    ids={r['id'] for r in metrics['details']}
    total=sum(len(p['diagnostics']) for p in predictions if p['id'] in ids)
    s=metrics['summary'];tp=s['matched_pred_locatable'];recall=s['recall']
    precision=tp/total if total else 0.
    return dict(total_diagnostics=total,locatable_diagnostics=tp+s['location_unmatched_pred_findings'],
        unlocatable_diagnostics=total-tp-s['location_unmatched_pred_findings'],precision=precision,recall=recall,
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        interpretation='Count unlocatable predictions as unmatched; not physical adjudication')


def matched_claim_changes(pure, tree):
    def matches(metrics):
        return {(r['id'],m['gt_error_id']) for r in metrics['details'] for m in r['location_matches']}
    a,b=matches(pure),matches(tree)
    return dict(shared=len(a&b),gained=len(b-a),lost=len(a-b),
        gained_claims=[dict(sample_id=s,claim_id=c) for s,c in sorted(b-a)],
        lost_claims=[dict(sample_id=s,claim_id=c) for s,c in sorted(a-b)],
        interpretation='Changes in matched reference locations, not adjudicated physics errors')


def score_tree_review(root, ref, ds, paired, tmp):
    from scripts.run_llm_checker_baseline import validate_response_diagnostics
    from scripts.run_retrieval_comparison import digest
    run=root/'terra_tree_direct';manifest=json.loads((run/'manifest.json').read_text())
    config={k:v for k,v in manifest.items() if k not in {'configuration_sha256','script_sha256'}}
    if digest(config)!=manifest['configuration_sha256'] or manifest['input_sha256']!=sha256_file(root/'frozen50/inputs.json'):
        raise ValueError('Tree-review configuration or input mismatch')
    pure_path=root/'terra_direct/results.json';tree_path=run/'results.json'
    pure=json.loads(pure_path.read_text());tree=json.loads(tree_path.read_text())
    old=json.loads((root/'terra_direct/metrics_automatic_proxy.json').read_text())
    if old['source_baseline_sha256']!=sha256_file(pure_path):raise ValueError('Pure comparator changed')
    ids={r['id'] for r in ds}
    all_response_ids=[];tools=Counter();usage={};tree_navigation=set();leaf_read=set()
    for predictions in [pure,tree]:
        label='terra_tree_self_check' if predictions is tree else 'pure_terra'
        usage[label]=dict(input_tokens=0,output_tokens=0)
        if len(predictions)!=50 or {r['id'] for r in predictions}!=ids or any(r.get('checker_status')!='completed' for r in predictions):
            raise ValueError('Both direct groups must complete the same 50; failures cannot be scored as empty')
        for row in predictions:
            if predictions is tree and row['configuration_sha256']!=manifest['configuration_sha256']:
                raise ValueError('Tree sample protocol mismatch')
            evidence=row.get('accepted_response_evidence') if predictions is tree else [row['response_evidence']]
            if not evidence:raise ValueError('Missing response evidence')
            final=None;sample_output_tokens=0
            for e in evidence:
                path=Path(e['path'])
                if sha256_file(path)!=e['sha256']:raise ValueError('Response hash mismatch')
                record=json.loads(path.read_text());data=record.get('response') or {}
                if (record.get('error') or record.get('http_status')!=200 or data.get('model')!='gpt-5.6-terra'
                        or (data.get('reasoning') or {}).get('effort')!='medium' or data.get('status')!='completed' or not data.get('id')):
                    raise ValueError('Response identity/status mismatch')
                all_response_ids.append(data['id']);final=data
                for k in ['input_tokens','output_tokens']:
                    value=(data.get('usage') or {}).get(k)
                    if not isinstance(value,int) or value<0:raise ValueError('Missing accepted usage')
                    usage[label][k]+=value
                    if k=='output_tokens':sample_output_tokens+=value
                if predictions is tree:
                    for call in data.get('output',[]):
                        if call.get('type')=='function_call':
                            tools[call.get('name')]+=1
                            if call.get('name')=='read_tree':tree_navigation.add(row['id'])
            if sample_output_tokens>16384:raise ValueError('Accepted sample exceeds the shared output budget')
            if predictions is tree:
                if row.get('review_trace',{}).get('read_rule_ids'):leaf_read.add(row['id'])
                calls=[o for o in final['output'] if o.get('type')=='function_call']
                if len(calls)!=1 or calls[0].get('name')!='submit_diagnostics':raise ValueError('Missing terminal diagnosis')
                final=dict(status='completed',output=[dict(type='message',content=[dict(type='output_text',text=calls[0]['arguments'])])])
            if validate_response_diagnostics(final)!=row['diagnostics']:raise ValueError('Saved diagnosis differs from raw response')
    if len(all_response_ids)!=len(set(all_response_ids)):raise ValueError('Accepted response IDs are not unique')
    scores={};sensitivities={};bootstrap={};changes={}
    for name,dataset in [('all_50',ds),('paired_49',paired)]:
        pure_score=evaluate(dataset,pure,tmp,'pure_'+name)
        if pure_score['summary']!=old[name]['summary']:raise ValueError('Pure baseline scores failed exact reproduction')
        tree_score=evaluate(dataset,tree,tmp,'tree_'+name)
        scores[name]={'pure_terra':pure_score,'terra_tree_self_check':tree_score}
        sensitivities[name]={'pure_terra':all_diagnostics_sensitivity(pure_score,pure),
            'terra_tree_self_check':all_diagnostics_sensitivity(tree_score,tree)}
        bootstrap[name]=paired_bootstrap(tree_score,{'pure_terra':pure_score})
        changes[name]=matched_claim_changes(pure_score,tree_score)
    requests=[json.loads(p.read_text()) for p in (run/'arm6').glob('*/request_*.json')]
    execution=dict(planned=50,completed=50,actual_requests=len(requests),
        rejected_model_identity=sum(r.get('error')=='Provider model identity mismatch' for r in requests),
        http_status_counts=dict(Counter(str(r.get('http_status')) for r in requests)),
        accepted_responses=len(all_response_ids)-50,budget=json.loads((run/'budget.json').read_text()))
    execution.update(accepted_usage=usage,tool_calls=dict(tools),samples_read_full_leaves=len(leaf_read),
        samples_requested_tree_navigation=len(tree_navigation),initial_tree_root_provided_to_all=50,
        cost_note='Accepted usage excludes paid failures/rejected identities; full technical accounting remains in budget.json')
    result=dict(status='provisional_automatic_location_match_unreviewed_gt',
        comparison_scope='Terra medium direct versus autonomous rule-tree self-check; no Qwen; primary all 50',
        source_tree_sha256=sha256_file(tree_path),source_pure_sha256=sha256_file(pure_path),
        manifest_sha256=sha256_file(run/'manifest.json'),reference_metrics_sha256=sha256_file(root/'metrics_automatic_proxy_location_unreviewed.json'),
        script_sha256=sha256_file(Path(__file__)),evaluator_sha256=ref['sample']['evaluator_sha256'],scoring=ref['scoring'],
        execution=execution,**scores,all_diagnostics_denominator_sensitivity=sensitivities,paired_bootstrap=bootstrap,
        matched_claim_changes=changes)
    write_json(run/'metrics_automatic_proxy.json',result)
    print(json.dumps({k:v['summary'] for k,v in scores['all_50'].items()},ensure_ascii=False))


def score_leaf_review(root, ref, ds, paired, tmp):
    """Validate and score the autonomous leaves-only condition against pure Terra and arm 6."""
    from scripts.run_llm_checker_baseline import validate_response_diagnostics, leaf_review_tools, leaf_review_messages
    from scripts.run_retrieval_comparison import digest
    frozen_rows=json.loads((root/'frozen50/inputs.json').read_text())
    catalog_path=ROOT/'catalogs/rules_unified_3000.json'
    run=root/'terra_leaf_direct';manifest=json.loads((run/'manifest.json').read_text())
    config={k:v for k,v in manifest.items() if k not in {'configuration_sha256','script_sha256'}}
    if (digest(config)!=manifest['configuration_sha256'] or
            manifest['input_sha256']!=sha256_file(root/'frozen50/inputs.json') or
            manifest.get('selection_sha256')!=sha256_file(root/'frozen50/manifest.json') or
            manifest.get('catalog_sha256')!=sha256_file(catalog_path) or
            manifest.get('messages_sha256')!=digest([leaf_review_messages(r) for r in frozen_rows]) or
            manifest.get('checker')!='terra_leaf_self_check' or manifest.get('tree_view_provided') is not False or
            manifest.get('tree_tool_available') is not False or
            any(t.get('name')=='read_tree' for t in manifest.get('tools', []))):
        raise ValueError('Leaves-only configuration, tool set, or frozen input mismatch')
    if manifest.get('tools')!=leaf_review_tools():
        raise ValueError('Leaves-only tool definitions differ from the frozen protocol')
    pure=json.loads((root/'terra_direct/results.json').read_text())
    leaf=json.loads((run/'results.json').read_text())
    tree_run=root/'terra_tree_direct';tree_manifest=json.loads((tree_run/'manifest.json').read_text())
    tree_config={k:v for k,v in tree_manifest.items() if k not in {'configuration_sha256','script_sha256'}}
    if (digest(tree_config)!=tree_manifest.get('configuration_sha256') or tree_manifest.get('checker')!='terra_tree_self_check'
            or tree_manifest.get('input_sha256')!=manifest.get('input_sha256')
            or tree_manifest.get('catalog_sha256')!=manifest.get('catalog_sha256')):
        raise ValueError('Existing tree comparator configuration mismatch')
    tree=json.loads((tree_run/'results.json').read_text())
    old=json.loads((root/'terra_direct/metrics_automatic_proxy.json').read_text())
    ids={r['id'] for r in ds};all_response_ids=[];tool_counts=Counter();usage={}
    groups=[('pure_terra',pure,None,None),('terra_leaf_self_check',leaf,(run,'arm7'),manifest),
        ('terra_tree_self_check',tree,(tree_run,'arm6'),tree_manifest)]
    for label,predictions,rule_source,rule_manifest in groups:
        usage[label]=dict(input_tokens=0,output_tokens=0)
        if len(predictions)!=50 or {r['id'] for r in predictions}!=ids or any(r.get('checker_status')!='completed' for r in predictions):
            raise ValueError(label+' must complete the same 50 samples; failures cannot be scored as empty')
        for row in predictions:
            if rule_source and row.get('configuration_sha256')!=rule_manifest.get('configuration_sha256'):
                raise ValueError(label+' sample protocol mismatch')
            evidence=row.get('accepted_response_evidence') if rule_source else [row.get('response_evidence')]
            if not evidence or any(not e for e in evidence):raise ValueError(label+' missing response evidence')
            final=None;sample_output_tokens=0
            for e in evidence:
                path=Path(e['path'])
                if sha256_file(path)!=e['sha256']:raise ValueError(label+' response hash mismatch')
                record=json.loads(path.read_text());data=record.get('response') or {}
                if (record.get('error') or record.get('http_status')!=200 or data.get('model')!='gpt-5.6-terra'
                        or (data.get('reasoning') or {}).get('effort')!='medium' or data.get('status')!='completed' or not data.get('id')):
                    raise ValueError(label+' response identity/status mismatch')
                all_response_ids.append(data['id']);final=data
                for k in ['input_tokens','output_tokens']:
                    value=(data.get('usage') or {}).get(k)
                    if not isinstance(value,int) or value<0:raise ValueError(label+' missing accepted usage')
                    usage[label][k]+=value
                    if k=='output_tokens':sample_output_tokens+=value
                for output in data.get('output',[]):
                    if output.get('type')=='function_call':
                        name=output.get('name');tool_counts[label+':'+str(name)]+=1
                        if label=='terra_leaf_self_check' and name=='read_tree':
                            raise ValueError('Leaves-only run invoked read_tree')
            if sample_output_tokens>16384:raise ValueError(label+' exceeded per-sample cumulative output budget')
            if rule_source:
                calls=[o for o in final.get('output',[]) if o.get('type')=='function_call']
                if len(calls)!=1 or calls[0].get('name')!='submit_diagnostics':
                    raise ValueError(label+' missing terminal diagnosis')
                final=dict(status='completed',output=[dict(type='message',content=[dict(type='output_text',text=calls[0]['arguments'])])])
                saved=json.loads((rule_source[0]/'samples'/(digest(row['id'])[:16]+'.json')).read_text())
                if label=='terra_leaf_self_check':
                    for attempt in sorted((run/'arm7').glob(digest(row['id'])[:16]+'_*/tools.json')):
                        events=json.loads(attempt.read_text())
                        if any(e.get('tool')=='read_tree' for e in events):
                            raise ValueError('Leaves-only trace records tree access')
                        if any(e.get('tool') not in {'search_rules','list_rules','read_rules'} for e in events):
                            raise ValueError('Leaves-only trace contains an unsupported tool')
            if validate_response_diagnostics(final)!=row['diagnostics']:
                raise ValueError(label+' stored diagnostics differ from raw final response')
    if len(all_response_ids)!=len(set(all_response_ids)):raise ValueError('Accepted provider response IDs are not unique')
    scores={};sensitivities={};bootstrap={};changes={}
    for name,dataset in [('all_50',ds),('paired_49',paired)]:
        arm_scores={label:evaluate(dataset,predictions,tmp,label+'_'+name) for label,predictions,_,_ in groups}
        if arm_scores['pure_terra']['summary']!=old[name]['summary']:
            raise ValueError('Pure baseline scores failed exact reproduction')
        scores[name]=arm_scores
        sensitivities[name]={label:all_diagnostics_sensitivity(arm_scores[label],predictions)
            for label,predictions,_,_ in groups}
        bootstrap[name]=paired_bootstrap(arm_scores['terra_leaf_self_check'],{
            'pure_terra':arm_scores['pure_terra'],'terra_tree_self_check':arm_scores['terra_tree_self_check']})
        changes[name]={
            'versus_pure_terra':matched_claim_changes(arm_scores['pure_terra'],arm_scores['terra_leaf_self_check']),
            'versus_terra_tree_self_check':matched_claim_changes(arm_scores['terra_tree_self_check'],arm_scores['terra_leaf_self_check'])}
    requests=[json.loads(p.read_text()) for p in (run/'arm7').glob('*/request_*.json')]
    execution=dict(planned=50,completed=50,failed=0,actual_requests=len(requests),
        rejected_model_identity=sum(r.get('error')=='Provider model identity mismatch' for r in requests),
        http_status_counts=dict(Counter(str(r.get('http_status')) for r in requests)),
        budget=json.loads((run/'budget.json').read_text()),accepted_usage=usage,tool_calls=dict(tool_counts),
        cost_note='Accepted usage excludes paid failures/rejected identities; full conservative accounting remains in budget.json')
    result=dict(status='provisional_automatic_location_match_unreviewed_gt',
        comparison_scope='Same Terra medium self-check on pure, leaves-only, and tree-plus-leaves conditions; primary all 50',
        source_leaf_sha256=sha256_file(run/'results.json'),source_tree_sha256=sha256_file(tree_run/'results.json'),
        source_pure_sha256=sha256_file(root/'terra_direct/results.json'),manifest_sha256=sha256_file(run/'manifest.json'),
        reference_metrics_sha256=sha256_file(root/'metrics_automatic_proxy_location_unreviewed.json'),
        script_sha256=sha256_file(Path(__file__)),evaluator_sha256=ref['sample']['evaluator_sha256'],
        scoring=ref['scoring'],execution=execution,**scores,all_diagnostics_denominator_sensitivity=sensitivities,
        paired_bootstrap=bootstrap,matched_claim_changes=changes)
    write_json(run/'metrics_automatic_proxy.json',result)
    write_json(run/'execution_status.json',dict(leaves_only_audit=dict(checked_unix=time.time(),planned=50,completed=50,
        actual_requests=len(requests),model='gpt-5.6-terra',reasoning_effort='medium',tree_view_provided=False,
        read_tree_tool_available=False,read_tree_calls=0,leaf_tools_only_verified=True,
        frozen_input_selection_catalog_hashes_verified=True,accepted_response_hashes_and_payloads_verified=True,
        all_three_groups_scored_with_same_location_proxy=True,independent_physics_adjudication=False,
        results_sha256=sha256_file(run/'results.json'),metrics_sha256=sha256_file(run/'metrics_automatic_proxy.json'))))
    print(json.dumps({k:{label:v['summary'] for label,v in arm_scores.items()}
        for k,arm_scores in scores.items()},ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment-root',type=Path,required=True)
    ap.add_argument('--verify-reference-only',action='store_true')
    ap.add_argument('--tree-review',action='store_true',help='Score autonomous Terra rule-tree checking versus pure Terra')
    ap.add_argument('--leaf-review',action='store_true',help='Score autonomous Terra leaves-only checking versus pure Terra and tree review')
    args=ap.parse_args();root=args.experiment_root
    refpath=root/'metrics_automatic_proxy_location_unreviewed.json';ref=json.loads(refpath.read_text())
    if sha256_file(ROOT/'scripts/evaluate_physics_eval_sets.py') != ref['sample']['evaluator_sha256']:
        raise ValueError('Evaluator changed since original comparison')
    ds=reference_dataset(root,ref)
    paired_ids=[row['id'] for row in ref['paired_49']['1']['details']]
    paired=[row for row in ds if row['id'] in set(paired_ids)]
    n=sum(len(row['physics_error_gt']) for row in paired)
    located=sum(g['span_valid'] for row in paired for g in row['physics_error_gt'])
    if (len(paired),n,located)!=(49,184,127):raise ValueError('Unexpected paired reference counts')
    with tempfile.TemporaryDirectory(prefix='.score_',dir=root) as td:
        tmp=Path(td);verified={}
        for arm in ['1','2','3','4']:
            saved={}
            for run in ['formal_checker','formal_checker_retry_001']:
                for path in sorted((root/run/('arm'+arm)).glob('*/result.json')):
                    row=json.loads(path.read_text())
                    if row['status']=='completed' and row['id'] not in saved:saved[row['id']]=row['trace']
            scored=evaluate(paired,list(saved.values()),tmp,'arm'+arm)
            old=ref['paired_49'][arm]['summary'];new=scored['summary']
            for key in ['scored_size','total_gt_errors','total_gt_locatable_errors','matched_gt_errors','precision','recall','f1']:
                if old[key]!=new[key]:raise ValueError(f'Original arm {arm} mismatch in {key}: {old[key]} vs {new[key]}')
            verified[arm]=True
        print('Reproduced all four original paired metrics exactly.',flush=True)
        if args.verify_reference_only:return
        if args.tree_review:return score_tree_review(root,ref,ds,paired,tmp)
        if args.leaf_review:return score_leaf_review(root,ref,ds,paired,tmp)
        baseline=root/'terra_direct/results.json';pred=json.loads(baseline.read_text())
        if len(pred)!=50 or len({x['id'] for x in pred})!=50 or any(x.get('checker_status')!='completed' for x in pred):
            raise ValueError('Baseline is not 50/50 successful; do not silently shrink the common 49')
        if {p['id'] for p in pred}!={r['id'] for r in ds}:raise ValueError('Baseline IDs differ from frozen inputs')
        response_ids=[]
        for item in pred:
            evidence=item['response_evidence'];path=Path(evidence['path'])
            if sha256_file(path)!=evidence['sha256']:raise ValueError('Baseline response hash mismatch')
            raw=json.loads(path.read_text());response=raw.get('response') or {}
            if (raw.get('error') or response.get('model')!='gpt-5.6-terra' or
                    (response.get('reasoning') or {}).get('effort')!='medium' or response.get('status')!='completed'):
                raise ValueError('Baseline response identity/status mismatch')
            response_ids.append(response['id'])
        if len(set(response_ids))!=50:raise ValueError('Baseline provider response IDs are not unique')
        requests=[json.loads(p.read_text()) for p in (root/'terra_direct/arm5').glob('*/request_*.json')]
        execution=dict(planned=50,completed=50,failed=0,actual_requests=len(requests),
            rejected_model_identity=sum(r.get('error')=='Provider model identity mismatch' for r in requests),
            http_status_counts=dict(Counter(str(r.get('http_status')) for r in requests)),
            accepted_response_ids_unique=True,budget=json.loads((root/'terra_direct/budget.json').read_text()))
        common=evaluate(paired,pred,tmp,'direct_paired49');full=evaluate(ds,pred,tmp,'direct_all50')
        sensitivity={}
        for name,metrics,ids in [('paired_49',common,set(paired_ids)),('all_50',full,{r['id'] for r in ds})]:
            total=sum(len(p['diagnostics']) for p in pred if p['id'] in ids)
            summary=metrics['summary'];tp=summary['matched_pred_locatable']
            locatable=tp+summary['location_unmatched_pred_findings']
            recall=summary['recall'];precision=tp/total if total else 0.
            sensitivity[name]=dict(total_diagnostics=total,locatable_diagnostics=locatable,
                unlocatable_diagnostics=total-locatable,precision=precision,recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                interpretation='Sensitivity only: count all unlocatable predictions as unmatched; no new physical adjudication')
        result=dict(status='provisional_automatic_location_match_unreviewed_gt',model='gpt-5.6-terra',reasoning_effort='medium',
            source_baseline_sha256=sha256_file(baseline),reference_metrics_sha256=sha256_file(refpath),
            script_sha256=sha256_file(Path(__file__)),evaluator_sha256=ref['sample']['evaluator_sha256'],
            paired_ids=paired_ids,paired_gt_errors=n,paired_gt_locatable=located,reference_arms_reproduced=verified,
            scoring=ref['scoring'],comparison_scope='direct Terra versus retrieval plus Qwen checker; not a rule-only ablation',
            execution=execution,paired_49=common,all_50=full,all_diagnostics_denominator_sensitivity=sensitivity,
            paired_bootstrap=paired_bootstrap(common,ref['paired_49']),prior_arms={a:d['summary'] for a,d in ref['paired_49'].items()})
        write_json(root/'terra_direct/metrics_automatic_proxy.json',result)
        print(json.dumps(common['summary'],ensure_ascii=False))


if __name__=='__main__':main()
