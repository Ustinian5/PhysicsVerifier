"""Cache source-family links once; freeze an explicitly sized build/held-out split.

No model calls, answers, references or correctness scores influence the split.
The existing conservative lexical/shared-prose policy is reused unchanged.
"""
import argparse
from collections import defaultdict
from pathlib import Path

from scripts import run_combined_source_screen as review
from rule_framework.organization_followup import FamilyScreen

read, write, require, digest, sha = review.read, review.write, review.require, review.digest, review.sha


def components(rows, progress=None):
    parent = {r['id']: r['id'] for r in rows}
    require(len(parent) == len(rows), 'Duplicate source IDs')

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    screen = FamilyScreen(rows); links = []
    for i, row in enumerate(rows):
        for hit in screen.matches(row):
            other = hit['heldout_id']
            if other not in parent: continue  # incompleteness sentinel, not a family ID
            a, b = find(row['id']), find(other)
            if a != b:
                parent[max(a,b)] = min(a,b)
                links.append(dict(source_id=row['id'], **hit))
        if progress and i % 100 == 0: progress(i, len(rows), len(links))
    grouped = defaultdict(list)
    for sid in sorted(parent): grouped[find(sid)].append(sid)
    return dict(components=[dict(id=k, members=v) for k,v in sorted(grouped.items())], links=links)


def choose(groups, eligible, size=2500, seed=20261107):
    """Keep a family entirely on one side, never split it to fill a quota."""
    selected, reserve, count = [], [], 0
    for group in sorted(groups, key=lambda g: digest([seed,g['id']])):
        members = [x for x in group['members'] if x in eligible]
        if not members: continue
        if count + len(members) <= size:
            selected.extend(members); count += len(members)
        else: reserve.extend(members)
    require(count == size, f'Whole-family selection supplies {count}, requested {size}; no silent shrink')
    require(len(reserve) >= 600, 'Fewer than 600 eligible held-out candidates; no silent shrink')
    return selected, reserve


def family_cache(out, inventory_path):
    plan = review.verify(out); rows = read(out/'questions.json')
    inventory = read(inventory_path)
    require(sha(inventory_path) == plan['configuration']['source_inventory_sha256'], 'Inventory drift')
    by_id = {r['id']:r for r in inventory['questions']}
    rows = [dict(r,context='\n'.join(by_id[r['id']]['contexts']),prediction='',answer='') for r in rows]
    code = {p:sha(review.ROOT/p) for p in ('scripts/partition_combined_source.py',
        'rule_framework/organization_followup.py','rule_framework/organization_bench99.py',
        'rule_framework/oss120b_construction.py')}
    signature = digest([rows,code]); path = out/'family_components.json'
    if path.exists():
        value=read(path);require(value['signature']==signature,'Existing family cache differs');return value
    result = components(rows, lambda *x:print('family_screen',*x,flush=True))
    value=dict(signature=signature,source_plan_sha256=sha(out/'plan.json'),code_hashes=code,
               semantic_family_independence_proven=False,**result)
    write(path,value);return value


def freeze(out, build_size):
    require(2000 <= build_size <= 3000, 'Build size outside the user-authorized 2000–3000 range')
    plan=review.verify(out);cache=read(out/'family_components.json')
    require(cache['source_plan_sha256']==sha(out/'plan.json'),'Family cache input drift')
    require(all(sha(review.ROOT/p)==h for p,h in cache['code_hashes'].items()),'Family cache implementation drift')
    require(read(out/'execution_status.json')['state']=='source_review_complete','Review not complete')
    rows=read(out/'questions.json');eligible=set();decisions={};evidence={}
    for i,batch in enumerate(read(out/'batches.json')):
        p=out/'reviewed'/f'{i:04d}.json';r=read(p)
        require(r['signature']==digest([batch,plan['configuration']]),'Review binding drift')
        review.corpus.evidence_valid(out,r);evidence[str(p.relative_to(out))]=sha(p)
        if r['state']=='complete':
            for x in review.validate({'items':r['items']},batch):
                decisions[x['id']]=x
                if x['domain']=='physics' and x['self_contained']=='yes':eligible.add(x['id'])
        else:
            require(r['state']=='content_failure','Unfinished source review')
            for x in batch:decisions[x['id']]=dict(domain='invalid',self_contained='uncertain',reason=r['error'])
    acceptance=read(out/'pilot_acceptance.json')
    require(acceptance['passed'] is True and acceptance['configuration_sha256']==plan['configuration_sha256']
            and acceptance['reviewed_sha256']==sha(out/'reviewed/0000.json'),'Source acceptance drift')
    for issue in acceptance.get('known_source_issues',[]):
        require(issue['id'] in decisions,'Unknown source review correction')
        if issue['action']=='exclude_before_split':
            eligible.discard(issue['id'])
            decisions[issue['id']]=dict(id=issue['id'],domain=decisions[issue['id']]['domain'],
                self_contained='no',reason=issue['reason'],reviewer='Codex assistant',human_gold=False)
        else:
            require(issue['action']=='reference_alignment_verified','Unresolved source issue before split')
    require(len(eligible) >= build_size+600,
            f'Eligible questions: {len(eligible)}; build {build_size} leaves {len(eligible)-build_size}, need at least 600. No automatic shrink.')
    build,reserve=choose(cache['components'],eligible,size=build_size);by_id={r['id']:r for r in rows}
    payload=dict(protocol='combined_reserved_family_split_v2',seed=20261107,build_target=build_size,
        source_plan_sha256=sha(out/'plan.json'),family_sha256=sha(out/'family_components.json'),
        review_evidence=evidence,pilot_acceptance_sha256=sha(out/'pilot_acceptance.json'),build=[by_id[x] for x in build],
        reserve=[by_id[x] for x in sorted(reserve,key=lambda x:digest([20261107,'heldout',x]))],
        excluded=[dict(decisions[r['id']],id=r['id']) for r in rows if r['id'] not in eligible],
        benchmark_target=600,benchmark_frozen=False,balanced_class_quota_guaranteed=False,
        counts=dict(candidates=len(rows),eligible=len(eligible),build=len(build),reserve=len(reserve),
                    excluded=len(rows)-len(eligible)),
        semantic_family_independence_proven=False)
    path=out/'split.json'
    if path.exists():require(read(path)==payload,'Frozen split changed')
    else:write(path,payload)
    print('build',len(build),'reserve',len(reserve),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['families','freeze']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inventory',type=Path)
    p.add_argument('--build-size',type=int,help='Explicit authorized build size; required for freeze')
    a=p.parse_args();out=a.output.resolve()
    with review.corpus.r.run_lock(out/'partition.lock'):
        if a.mode=='families':
            require(a.inventory is not None,'Inventory required');v=family_cache(out,a.inventory)
            print('components',len(v['components']))
        else:
            require(a.build_size is not None,'freeze requires explicit --build-size; never shrink implicitly')
            freeze(out,a.build_size)


if __name__=='__main__':main()
