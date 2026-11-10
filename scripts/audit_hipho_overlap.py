"""Read-only source/family audit against an explicitly supplied HiPhO file.

Reuses the current conservative isolation screen; never calls a model, changes
frozen cohorts, or treats lexical candidates as confirmed duplicate questions.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from rule_framework.organization_followup import FamilyScreen, ISOLATION_POLICY
from rule_framework.organization_bench99 import FAMILY_POLICY
from rule_framework.oss120b_construction import question_key, question_template, shingles, require


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Audit:
    def __init__(self, refs):
        require(bool(refs) and len({r['id'] for r in refs}) == len(refs), 'Invalid benchmark IDs')
        require(all(isinstance(r.get('question'),str) and r['question'].strip() for r in refs), 'Missing benchmark question')
        self.refs={r['id']:r for r in refs}
        self.question_screen=FamilyScreen(refs)
        self.context_screen=FamilyScreen([dict(r,question=(r.get('context') or '')+'\n'+r['question']) for r in refs])
        self.cache={}

    def matches(self, row):
        visible={k:row.get(k) or '' for k in ('question','context','prediction','answer')}
        require(all(isinstance(v,str) for v in visible.values()), 'Non-text source field')
        key=json.dumps(visible,sort_keys=True,ensure_ascii=False)
        if key in self.cache:return self.cache[key]
        hits=[]
        for view,screen in [('question',self.question_screen),('context_and_question',self.context_screen)]:
            for hit in screen.matches(dict(visible,id=row['id'])):
                sid=hit['heldout_id']
                if sid not in self.refs:continue  # Old known-family / unresolved-parent sentinels are not HiPhO matches.
                reference=self.refs[sid]['question'] if view=='question' else (self.refs[sid].get('context') or '')+'\n'+self.refs[sid]['question']
                reason=hit['reason']
                if question_key(visible['question'])==question_key(reference):reason='normalized_exact'
                elif question_template(visible['question'])==question_template(reference):reason='numeric_template'
                a,b=shingles(visible['question']),shingles(reference);n=len(a&b)
                hits.append(dict(hit,reason=reason,reference_view=view,
                                 jaccard=n/max(1,len(a|b)),containment=n/max(1,min(len(a),len(b)))))
        self.cache[key]=hits
        return hits

    def cohort(self, name, rows):
        require(len({r['id'] for r in rows})==len(rows),'Duplicate cohort IDs')
        matches=[dict(source_id=r['id'],matches=hits) for r in rows if (hits:=self.matches(r))]
        exact={r['source_id'] for r in matches if any(h['reason']=='normalized_exact' for h in r['matches'])}
        numeric={r['source_id'] for r in matches if any(h['reason']=='numeric_template' for h in r['matches'])}
        result=dict(rows=len(rows),normalized_exact_sources=len(exact),numeric_only_sources=len(numeric-exact),
                    any_screen_hit_sources=len(matches),matches=matches,
                    meaning='Conservative screening candidates; non-exact hits require inspection, not automatic relabeling')
        print(name, {k:v for k,v in result.items() if k not in ('matches','meaning')},flush=True)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--legacy-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.results_root;step=root/'qwen30b_step_verifier';current=step/'corrected'
    files={}
    def load(path):
        value=read(path);files[str(path)]=sha(path);return value
    raw=[json.loads(line) for line in a.benchmark.read_text().splitlines() if line.strip()]
    files[str(a.benchmark)]=sha(a.benchmark)
    refs=[{k:r.get(k,'') for k in ('id','exam','question','context')} for r in raw]
    audit=Audit(refs)
    inventory=load(step/'source_question_inventory.json')['questions']
    contexts={r['id']:'\n'.join(r.get('contexts',[])) for r in inventory}
    split=load(current/'source_screen/split.json')
    actual=load(current/'construction/inputs.json');actual_by_id={r['id']:r for r in actual}
    build=[dict(r,context=contexts.get(r['id'],''),prediction=actual_by_id.get(r['id'],{}).get('prediction',''),answer='') for r in split['build']]
    cohorts={'current_build_2000':build,
             'current_reserve_702':[dict(r,context=contexts.get(r['id'],''),prediction='',answer='') for r in split['reserve']],
             'current_step_bench_100':load(step/'full/bench_questions.json'),
             'historical_v5_2626':load(a.legacy_root/'inputs.json'),
             'historical_expansion':load(root/'qwen30b_expansion_5000/build/inputs.json')}
    results={name:audit.cohort(name,rows) for name,rows in cohorts.items()}
    derived={}
    for name,path in [('historical_v5_2626',a.legacy_root/'distilled.json'),
                      ('historical_expansion',root/'qwen30b_expansion_5000/tree_v2/materials.json')]:
        rules=load(path)['rules'];affected={r['source_id'] for r in results[name]['matches']}
        source_ids={r['id'] for r in cohorts[name]}
        unsupported=[r['rule_id'] for r in rules if not set(r['sample_ids'])<=source_ids]
        derived[name]=dict(rule_count=len(rules),unrecognized_source_rule_ids=unsupported,
                          candidate_affected_rule_ids=[r['rule_id'] for r in rules if set(r['sample_ids'])&affected])
    paths=sorted((current/'construction/experiences').glob('*.json'))
    affected={r['source_id'] for r in results['current_build_2000']['matches']}
    states=Counter();candidate_affected=[];rules=0;unknown=[]
    for path in paths:
        value=load(path);sid=value['id'];result=value['value']
        if sid not in actual_by_id:unknown.append(sid)
        states[result['semantic_audit']['status'] if result else 'invalid_output']+=1
        if result:rules+=len(result['experience_rules'])
        if sid in affected:candidate_affected.append(sid)
    derived['current_in_progress_snapshot']=dict(processed=len(paths),counts=dict(states),raw_rules=rules,
        candidate_affected_source_ids=candidate_affected,unrecognized_source_ids=unknown,
        scope='Snapshot of immutable completed items; future extraction is not included')
    code=Path(__file__).resolve()
    report=dict(status='screening_complete_review_candidates',checked_at=datetime.now(timezone.utc).isoformat(),
        benchmark_path=str(a.benchmark),benchmark_rows=len(refs),benchmark_sha256=files[str(a.benchmark)],
        question_references=refs,cohorts=results,derived=derived,files=files,
        policies=dict(family=FAMILY_POLICY,shared_prose=ISOLATION_POLICY),
        code_sha256=sha(code),new_model_calls=0,frozen_inputs_changed=False,
        semantic_family_independence_proven=False,
        limitations=['Question/context/visible source-answer text and source lineage checked; lexical hits need review.',
                    'No exhaustive translated/paraphrased or pretraining contamination claim.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');tmp.replace(a.output)
    print('saved',a.output,flush=True)


if __name__=='__main__':main()
