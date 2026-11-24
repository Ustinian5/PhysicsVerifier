"""Score two retrieval representations and paired downstream errors on frozen correct answers."""
import argparse
from collections import Counter
from math import comb
from pathlib import Path

from scripts import confirm_rag_tree_retrieval as trial
from scripts.score_retrieval_fusion import score_confirmation

vr, require = trial.vr, trial.require


def paired_false_positives(base, other):
    require(set(base)==set(other),'Unpaired questions')
    improved = worsened = joint = 0
    for q,a in base.items():
        b=other[q]
        if a in ('correct','incorrect') and b in ('correct','incorrect'):
            joint+=1
            improved+=int(a=='incorrect' and b=='correct')
            worsened+=int(a=='correct' and b=='incorrect')
    changed=improved+worsened
    p=min(1,2*sum(comb(changed,k) for k in range(min(improved,worsened)+1))/2**changed)
    return dict(joint_binary=joint,removed_fp=improved,added_fp=worsened,unchanged=joint-changed,
                exact_p=p)


def compare_final_requests(folder, rows):
    """Post hoc comparison of the final CHECK bodies for relation and RRF."""
    same_messages = 0
    same_body = 0
    flipped = {}
    for q, arms in rows.items():
        methods = ('relation', 'rrf')
        if any(arms[m]['status'] != 'complete' for m in methods):
            continue
        paths = [folder/'rows'/q/m/'check' for m in methods]
        bodies = [vr.read(p/'request.json')['request'] for p in paths]
        if bodies[0]['messages'] != bodies[1]['messages']:
            continue
        require(arms['relation']['ids'] == arms['rrf']['ids'], 'Identical CHECK messages with different material IDs')
        same_messages += 1
        same_body += int(bodies[0] == bodies[1])
        if arms['relation']['verdict'] == arms['rrf']['verdict']:
            continue
        keys = bodies[0].keys() | bodies[1].keys()
        differing_fields = sorted(k for k in keys if bodies[0].get(k) != bodies[1].get(k))
        responses = {}
        for method, path, body in zip(methods, paths, bodies):
            result = vr.read(path/'result.json')
            raw = vr.read(path/'raw.json')
            responses[method] = dict(verdict=result['value']['verdict'], max_tokens=body['max_tokens'],
                                     generated=result['usage']['completion_tokens'], finish=raw['choices'][0]['finish_reason'])
        flipped[q] = dict(delivered=len(arms['relation']['ids']), fields=differing_fields, responses=responses)
    return dict(same_messages=same_messages, same_body=same_body, verdict_flips=flipped)


def score(root):
    out=root/trial.RESULT
    cohort=root/trial.PARENT
    design=vr.read(out/'design.json')
    require(vr.sha(cohort/'summary.json')==design['prior_summary_sha256'] and
            vr.sha(cohort/'rubric.json')==design['rubric_sha256'],'Prior evidence changed')
    old=score_confirmation(cohort)
    require(old==vr.read(cohort/'metrics.json'),'Skills outcome changed')
    control=vr.read(out/'pure/summary.json')
    pure={q:(v['value']['verdict'] if v['status']=='complete' else 'failure') for q,v in control.items()}
    require(len(pure)==26 and set(pure)=={r['id'] for r in vr.read(cohort/'inputs.json')},'Pure control mismatch')
    report=dict(n=len(pure),pure=dict(Counter(pure.values())),arms={},diagnostics={
                k:v['diagnostics'] for k,v in control.items() if 'diagnostics' in v},human_gold=False)
    pvalues={}
    for arm in ('flat_skills','case_rag','rule_tree'):
        folder=cohort if arm=='flat_skills' else out/arm
        is_new=arm!='flat_skills'
        new=score_confirmation(folder,cohort,out) if is_new else old
        rows=vr.read(folder/'summary.json')
        grades=vr.read(folder/'relevance.json')
        verdicts={m:{} for m in trial.previous.METHODS}
        false_positives={}
        for m,counts in new['groups'].items():
            operational=Counter()
            false_positives[m]={}
            for q,arms in rows.items():
                d=arms[m]
                require(len(set(d['ids']))==len(d['ids'])<=6 and
                        d.get('preview_tokens',0)+d.get('full_tokens',0)<=8192,'Budget or delivery mismatch')
                verdicts[m][q]=d.get('verdict') if d['status']=='complete' else 'failure'
                if verdicts[m][q]=='incorrect':
                    false_positives[m][q]={'delivered':len(d['ids']),
                                           'useful':sum(grades[q][i] for i in d['ids']),
                                           'pure_verdict':pure[q]}
                if is_new and d['status']!='rank_failure':
                    require(set(d['requested'])==set(d['ids'])|set(d['omitted']) and
                            not set(d['ids']).intersection(d['omitted']),'Invalid packing evidence')
                    operational.update(no_candidate=int(not d['pool']),empty_selection=int(bool(d['pool']) and not d['requested']),
                                       all_selected_exceeded_budget=int(bool(d['requested']) and not d['ids']),
                                       omitted_whole_texts=len(d['omitted']))
                operational.update(zero_delivery=int(not d['ids']))
            counts.update(operational)
        report['arms'][arm]=dict(retrieval=new,false_positives=false_positives,fp_vs_bm25={m:paired_false_positives(verdicts['bm25'],v)
            for m,v in verdicts.items() if m!='bm25'}, fp_relation_vs_rrf=paired_false_positives(verdicts['relation'],verdicts['rrf']),
            fp_vs_pure={m:paired_false_positives(pure,v) for m,v in verdicts.items()},
            posthoc_relation_rrf_request_comparison=compare_final_requests(folder,rows))
        if is_new:
            for m in ('relation','rrf'):
                pvalues[arm+'/'+m]=new['comparisons'][m+'_vs_bm25']['exact_two_sided_sign_p']
    adjusted={}
    largest=0
    for i,(key,pvalue) in enumerate(sorted(pvalues.items(),key=lambda kv:kv[1])):
        largest=max(largest,min(1,(len(pvalues)-i)*pvalue))
        adjusted[key]=dict(raw=pvalue,holm=largest)
    report['new_arms_retrieval_sign_tests']=adjusted
    report['hashes']={n:vr.sha(out/n) for n in ('design.json','pure/summary.json')}
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=trial.pilot.ROOT)
    args=p.parse_args()
    vr.atomic(args.root/trial.RESULT/'metrics.json',score(args.root))
