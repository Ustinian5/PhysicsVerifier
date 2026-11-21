"""Measure development material fit and nested construction inventories without model calls."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts import run_final_answer_revision as final
from scripts.analyze_verifier_knowledge_groups import checked_json


SEEDS = (20261121, 20261122, 20261123)
SIZES = (250, 500, 1000, 2000)
DEVELOPMENT_ID = 'cl_202_125689'
vr, require = final.vr, final.require


def family_levels(build, reserve, components, sizes, seed):
    families = []
    seen = set()
    for item in components:
        family = set(item['members'])
        members = family & build
        if members:
            require(not family & reserve and not members & seen, 'Source family crosses splits or repeats')
            seen.update(members)
            key = hashlib.sha256(f"{seed}:{item['id']}".encode()).digest()
            families.append((key, item['id'], members))
    require(seen == build, 'Missing construction sources')
    families.sort(key=lambda item: item[:2])
    selected = set()
    for n in sizes:
        for _, _, members in families:
            if members.isdisjoint(selected) and len(members) <= n - len(selected):
                selected.update(members)
            if len(selected) == n:
                break
        require(len(selected) == n, 'Cannot form the requested size without splitting a family')
        yield n, selected.copy()


def development_fit(root, row, accepted, cfg, tokenizer, arm, name):
    docs = checked_json(root / (name + '.json'), {str(root / (name + '.json')): cfg['material_hashes'][name]})
    store = final.base.p.FlatMemory(docs)
    path = root / 'pilot' / arm / vr.digest(row['id'])[:24] / 'reference_packet.json'
    packet = checked_json(path, accepted['files'])
    queries = json.loads(packet['messages'][2]['content'])

    def size(references):
        messages = final.p.continuation_messages(row, queries, references, arm=arm)
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 reasoning_effort='medium'))

    empty = size([])
    cap = min(empty + cfg['contract']['reference_token_cap'],
              cfg['limits']['context_window'] - cfg['limits']['output_tokens_per_sample'])
    can_fit = {sid for sid in store.ids if size([store.view(sid)]) <= cap}
    top = [v['id'] for v in store.search('\n'.join(queries['queries']))]
    require(top == [v['id'] for v in packet['retrieval']['ranked']], 'Frozen development ranking changed')
    delivered = packet['retrieval']['provided_complete_ids']
    require(set(delivered) <= can_fit, 'Inconsistent budget or tokenizer')
    sources = {d['id']: set(d['source_ids']) for d in docs}
    evidence = dict(total=len(docs), fits_alone=len(can_fit), top6_fit_alone=sum(sid in can_fit for sid in top),
                    delivered=delivered, packet_sha256=vr.sha(path))
    return sources, can_fit, evidence


def run(root, output):
    from transformers import AutoTokenizer

    base = root / 'aligned_old99/final_revision'
    plan = vr.read(base / 'plan.json')
    cfg = plan['configuration']
    require(vr.digest(cfg) == plan['configuration_sha256'], 'Frozen preparation changed')
    location = Path(cfg['root']) / 'corrected/source_screen'
    screen = root / 'corrected/source_screen'
    def source(name):
        path = screen / name
        return checked_json(path, {str(path): cfg['dependencies'][str(location / name)]})
    split = source('split.json')
    components = source('family_components.json')['components']
    build = {q['id'] for q in split['build']}
    reserve = {q['id'] for q in split['reserve']}
    require(len(build) == 2000 and len(reserve) == 702 and not build & reserve, 'Construction split changed')
    pilots = base / 'pilot_inputs.json'
    rows = checked_json(pilots, {str(pilots): cfg['pilot_sha256']})
    row, = (r for r in rows if r['id'] == DEVELOPMENT_ID)
    accepted = vr.read(base / 'pilot_acceptance.json')
    require(accepted['passed'] and accepted['configuration_sha256'] == vr.read(base / 'execution_repair.json')['configuration_sha256'],
            'Pilot does not match the frozen execution plan')
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], local_files_only=True, trust_remote_code=False)
    inventories = {}
    for arm, name in [('case_rag', 'cases'), ('flat_skills', 'skills')]:
        inventory = development_fit(base, row, accepted, cfg, tokenizer, arm, name)
        require(all(len(ids) == 1 and ids <= build for ids in inventory[0].values()),
                'Subsetting the published flat items would require rebuilding shared sources')
        inventories[arm] = inventory
    levels = []
    for seed in SEEDS:
        for n, members in family_levels(build, reserve, components, SIZES, seed):
            groups = {}
            for arm, (sources, can_fit, _) in inventories.items():
                items = {sid for sid, origin in sources.items() if origin <= members}
                groups[arm] = dict(items=len(items), fits_alone=len(items & can_fit))
            levels.append(dict(seed=seed, source_questions=n, source_ids_sha256=vr.digest(sorted(members)), groups=groups))
    final.immutable(output, dict(development_id=DEVELOPMENT_ID, model_calls=0,
                    scope='Published flat items from nested construction families; individual fit on one fixed development input, no verifier scores.',
                    plan_sha256=vr.sha(base / 'plan.json'), split_sha256=vr.sha(screen / 'split.json'),
                    families_sha256=vr.sha(screen / 'family_components.json'), pilot_sha256=vr.sha(base / 'pilot_acceptance.json'),
                    analysis_sha256=vr.sha(Path(__file__)), development={a: t[2] for a, t in inventories.items()}, levels=levels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
