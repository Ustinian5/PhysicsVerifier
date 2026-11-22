"""Build independent libraries from a frozen family draw and source-local original outcomes."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import subprocess
import threading

from rule_framework import verified_resume as vr
from scripts import audit_verifier_lineage as lineage
from scripts import run_corrected_pipeline as construction
from scripts import run_final_answer_revision as final
from scripts.audit_verifier_library_capacity import family_levels, SEEDS as ORIGINAL_SEEDS, SIZES
from scripts.analyze_verifier_knowledge_groups import checked_json


BUILD_SIZES = SIZES[:-1]  # The full 2000-source release is already frozen.
CAPACITY_HASH = '08973fc205ad87e943d98f12f646e5caa7170104ca27477d004c2b2448e79a5c'
ARCHIVED_BUILDS = {
    '837ad123a44a6ece8bc272a82dda79427c7829d019d580a843a3b54c2fe7b311': '5c2ccfa599fbd2be7abe66fd4646e7870989f7a4',
    '1e1c41ab344be9ba63663c8da1adf9895d7508c0dd11837dd8e5bc9787d65e73': 'f13f9ea0ab172f20285f24a119b2625d8c9e398b',
    '31d1cfaaab75be9ebc87628c6988a1c7732704b6e52132f0270818640e793bd0': 'f13f9ea0ab172f20285f24a119b2625d8c9e398b',
}
SOURCE_HOLD = {
    '4659f0d1c30facf22ee86b66b6edf9835b2c9eedab4d2cbb04c1de685ad7711c',
    'cbbfe15ffced7df58b85d2b6cbc27409df63baf4721cd4b039766a1ed9308e80',
    'cfb5fc2b61b5c1374817bdec9b662fde48d2af32fd9bcfbdcb670a12c19b93c5',
}
require, immutable = final.require, final.immutable
SEED_MANIFEST = final.a.ROOT/'experiments/library_scale_random_seeds.json'
SEED_MANIFEST_HASH = 'baa11425db66fc66413f2f7dac2f9114aefdeadafc588624f6bbf3d3136cc1c8'
require(vr.sha(SEED_MANIFEST) == SEED_MANIFEST_HASH, 'Random-seed registration changed')
RANDOM_REGISTRATION = vr.read(SEED_MANIFEST)
RANDOM_SEEDS = tuple(RANDOM_REGISTRATION['seeds'])
SEEDS = ORIGINAL_SEEDS + RANDOM_SEEDS
BUILD_FILES = ('scripts/run_verifier_library_scale.py', 'scripts/run_node14_library_scale.py',
               'scripts/audit_verifier_library_capacity.py', 'scripts/run_corrected_pipeline.py',
               'rule_framework/bounded_tree.py', 'scripts/generate_experience_rules.py',
               'rule_framework/organization_bench99.py', 'rule_framework/flat_skill_rag.py',
               'scripts/rebuild_qwen30b_corpus.py', 'scripts/run_node14_final_pilot.py')


def source_draw(split, families, seed, apply_hold, size=250):
    build = {r['id'] for r in split['build']}
    reserve = {r['id'] for r in split['reserve']}
    require(seed in SEEDS and size in BUILD_SIZES and len(build) == 2000 and len(reserve) == 702
            and not build & reserve, 'Unexpected seed, size or frozen population')
    levels = (level for level in BUILD_SIZES if level <= size)
    chosen = next(ids for n, ids in family_levels(build, reserve, families, levels, seed) if n == size)
    held = SOURCE_HOLD & chosen
    require(not held or apply_hold, 'The selected draw requires a source disposition before preparation')
    return [r for r in split['build'] if r['id'] in chosen], held


def local_rules(experiences, eligible):
    extract = construction.extraction.old.extraction
    rules = extract._build_distilled_library(experiences, min_count=1)['rules']
    support = defaultdict(list)
    for exp in experiences:
        sid = exp['sample_id']
        require(sid in eligible, 'Rule source is outside this draw')
        errors = {e['id']: e for e in exp['semantic_audit']['key_errors']}
        for rule in exp['experience_rules']:
            subject = exp['topic_guess']
            rid = 'exp_' + extract._fingerprint_rule(rule, subject['domain'], subject['topic'])
            support[rid].extend(dict(sample_id=sid, source_error=errors[k]) for k in rule['source_error_ids'])
    require(set(support) == {r['rule_id'] for r in rules}, 'Candidate accounting mismatch')
    for rule in rules:
        require(set(rule['sample_ids']) == {e['sample_id'] for e in support[rule['rule_id']]},
                'Rule support differs from local original audits')
    return rules, dict(support)


def local_flat(aligned, parent, manifest, source_rows):
    cases, skill_outcomes = [], []
    for row in source_rows:
        case_file = final.base.old.outcome_path(aligned / 'experience', row)
        note = checked_json(case_file, manifest)
        binding = dict(configuration_sha256=parent, source_sha256=vr.digest(row))
        require(all(note[k] == v for k, v in binding.items()), 'Case source or request changed')
        case, = final.base.p.materialize([row], [dict(note['outcome'], **binding)], parent)['cases']
        skill_file = final.base.old.outcome_path(aligned / 'skills', row)
        skill = checked_json(skill_file, manifest)
        require(all(skill[k] == v for k, v in binding.items()) and skill['case_sha256'] == vr.digest(case)
                and skill['outcome']['id'] == row['id'], 'Skill does not derive from this complete case')
        cases.append(case)
        skill_outcomes.append(skill['outcome'])
    skills, counts = final.a.bounded.materialize_skills(source_rows, skill_outcomes)
    return sorted(cases, key=lambda case: case['source_ids'][0]), skills, counts


def prepare(root, out, seed, apply_hold, size=250):
    if (out/'construction_acceptance.json').is_file():
        config = verify(out)['configuration']
        require((config['source_count'], config['seed']) == (size, seed), 'Wrong published library')
        return config
    inventory_path = root / 'knowledge_library_capacity.json'
    require(vr.sha(inventory_path) == CAPACITY_HASH, 'Frozen sampling receipt changed')
    inventory = vr.read(inventory_path)
    pilot = root / 'aligned_old99/final_revision/plan.json'
    require(vr.sha(pilot) == inventory['plan_sha256'], 'Frozen material release changed')
    cfg = vr.read(pilot)['configuration']
    archive = Path(cfg['root'])
    aligned = archive / 'aligned_old99'
    corrected = archive / 'corrected'
    ext = corrected / 'construction'
    pipe = corrected / 'pipeline'
    deps = cfg['dependencies']
    split = checked_json(corrected / 'source_screen/split.json', deps)
    families = checked_json(corrected / 'source_screen/family_components.json', deps)['components']
    rows, held = source_draw(split, families, seed, apply_hold, size)
    ids = {r['id'] for r in rows}
    levels = inventory['levels']
    if seed in RANDOM_SEEDS:
        require(tuple(RANDOM_REGISTRATION['sizes']) == BUILD_SIZES and
                all(RANDOM_REGISTRATION[k] == inventory[k] for k in ('families_sha256','split_sha256')) and
                vr.sha(final.a.ROOT/'scripts/audit_verifier_library_capacity.py') == RANDOM_REGISTRATION['sampling_code_sha256'],
                'Random draw does not use the frozen families, population or sampler')
        levels = RANDOM_REGISTRATION['draws']
    level, = (v for v in levels if v['seed'] == seed and v['source_questions'] == size)
    require(vr.digest(sorted(ids)) == level['source_ids_sha256'], 'Draw differs from the frozen inventory')
    excluded = set(checked_json(aligned/'evaluation/isolation.json', deps)['excluded_source_ids'])
    excluded.update(checked_json(aligned/'evaluation_revision/source_quarantine.json', deps)['excluded'])
    active = ids - held - excluded

    ec = vr.read(aligned/'evaluation/plan.json')
    require(vr.digest(ec['configuration']) == ec['configuration_sha256'], 'Flat-source plan changed')
    ef = ec['configuration']['dependencies']
    flat_binding = checked_json(aligned/'plan.json', ef)['configuration_sha256']
    pc = vr.read(pipe/'plan.json')
    cc = pc['configuration']
    require(vr.digest(cc) == pc['configuration_sha256'] and cc['tree_policy'] == construction.tree.POLICY and
            cc['frozen_files']['tree_bundle.json'] == ef[str(pipe/'tree_bundle.json')], 'Tree protocol changed')
    evidence = vr.read(pipe/'source_evidence.json')
    require(vr.digest(evidence) == cc['evidence_digest'], 'Original construction evidence changed')
    orig = checked_json(ext/'plan.json', evidence)['configuration']
    taxonomy = checked_json(ext/'taxonomy.json', evidence)
    inputs = lineage.indexed(checked_json(ext/'inputs.json', evidence))
    gen = corrected / 'collection'
    generator = vr.read(gen/'plan.json')
    require(vr.sha(gen/'plan.json') == orig['source_plan_sha256'], 'Frozen 30B generation plan changed')
    generation_config = generator['configuration']['generator_config']
    selected, flat_rows, disposition, receipts = [], [], {}, {}
    for row in rows:
        sid = row['id']
        if sid not in active:
            disposition[sid] = 'new_source_hold' if sid in held else 'prior_source_quarantine'
            continue
        path = gen/'generated'/(sid+'.json')
        answer = checked_json(path, evidence)
        raw = gen / answer['evidence']['file']
        require(evidence[str(raw)] == answer['evidence']['sha256'], 'Wrong generation evidence')
        lineage.solver_output(row, generation_config, answer, raw)
        receipts[str(path)] = evidence[str(path)]
        receipts[str(raw)] = evidence[str(raw)]
        if answer['state'] != 'complete':
            disposition[sid] = 'generation_' + answer['state']
            continue
        sample = inputs[sid]
        require(sample == dict(row, prediction=answer['prediction'], context='', answer=''), '30B input changed')
        path = ext/'experiences'/(sid+'.json')
        item = checked_json(path, evidence)
        payload = dict(sample=sample, taxonomy=taxonomy, max_rules_per_sample=orig['max_rules_per_sample'])
        require(item['id'] == sid and item['signature'] == construction.extraction.digest([payload, orig]),
                'Extraction is not a source-local request on this answer')
        receipts[str(path)] = evidence[str(path)]
        for ev in item['evidence']:
            key = str(ext/ev['file'])
            require(evidence[key] == ev['sha256'], 'Wrong extraction response binding')
            receipts[key] = ev['sha256']
        value = item['value']
        if value is None:
            disposition[sid] = 'extraction_content_failure'
            continue
        require(value['sample_id'] == sid, 'Experience belongs to another question')
        disposition[sid] = value['semantic_audit']['status']
        selected.append(value)
        if disposition[sid] == 'error_found':
            source = final.base.p.source_payload(sample)
            flat_rows.append(source)
            for kind in ('experience', 'skills'):
                key = str(final.base.old.outcome_path(aligned/kind, source))
                receipts[key] = ef[key]

    require(set(disposition) == ids, 'Incomplete source dispositions')
    vr.EvidenceCache(out/'source_cache.json').check(receipts, full=True)
    cases, skills, skill_counts = local_flat(aligned, flat_binding, ef, flat_rows)
    expectations = {}
    for name, docs in [('cases', cases), ('skills', skills)]:
        full = checked_json(aligned/'evaluation_revision'/(name+'.json'), deps)
        require(all(len(d['source_ids']) == 1 for d in full), 'Published flat material has cross-source support')
        original = [d for d in full if set(d['source_ids']) <= active]
        require(sorted(docs, key=lambda d:d['id']) == sorted(original, key=lambda d:d['id']),
                'Source-local reproduction differs from previously accepted flat material')
        require(all(set(d['source_ids']) <= active for d in docs), 'Out-of-draw material')
        expectations[name] = len(docs)
        immutable(out/'materials'/(name+'.json'), docs)
    rules, provenance = local_rules(selected, active)
    bundle = dict(taxonomy=taxonomy, rules=rules, provenance=provenance)
    immutable(out/'tree_bundle.json', bundle)
    immutable(out/'sources.json', dict(original_draw=[r['id'] for r in rows], dispositions=disposition,
              previous_quarantine=sorted(ids & excluded), new_hold=sorted(held)))
    immutable(out/'original_evidence.json', receipts)
    heldout = checked_json(aligned/'evaluation/bench_questions.json', deps)
    heldout += [dict(id='final:'+r['id'], question=r['question']) for r in split['reserve']]
    immutable(out/'heldout_questions.json', heldout)
    build_files = {name: vr.sha(final.a.ROOT/name) for name in BUILD_FILES}
    require(build_files['rule_framework/bounded_tree.py'] == cc['code']['rule_framework/bounded_tree.py'],
            'Frozen tree implementation changed')
    name = 'scripts/run_corrected_pipeline.py'
    snapshot = subprocess.check_output(['git','show','3fad274:'+name], cwd=final.a.ROOT)
    require(hashlib.sha256(snapshot).hexdigest() == cc['code'][name], 'Wrong original construction revision')
    def tree_functions(code):
        return {n.name:ast.dump(n) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)
                and n.name in {'model_json','build_materials'}}
    require(tree_functions(snapshot)==tree_functions((final.a.ROOT/name).read_bytes()), 'Tree call semantics changed')
    names = ('sources.json','tree_bundle.json','original_evidence.json','heldout_questions.json',
             'materials/cases.json','materials/skills.json')
    config = dict(seed=seed, source_count=size, source_ids_sha256=level['source_ids_sha256'], protocol=pc['configuration_sha256'],
                  source_release_sha256=inventory['plan_sha256'], transport=cc['transport'], tree_policy=cc['tree_policy'],
                  build_files=build_files, files={n:vr.sha(out/n) for n in names}, source_counts=dict(Counter(disposition.values())),
                  flat_counts=expectations, skill_outcomes=skill_counts, candidates=len(rules), fresh_tree=True)
    if seed in RANDOM_SEEDS:
        config['seed_manifest_sha256'] = SEED_MANIFEST_HASH
    immutable(out/'plan.json', dict(configuration=config, configuration_sha256=vr.digest(config)))
    return config


def verify(out):
    plan = vr.read(out/'plan.json')
    c = plan['configuration']
    if c['seed'] in RANDOM_SEEDS:
        require(c.get('seed_manifest_sha256') == vr.sha(SEED_MANIFEST) == SEED_MANIFEST_HASH,
                'Random-seed registration differs from the plan')
    require(vr.digest(c) == plan['configuration_sha256'] and all(vr.sha(out/n) == h for n,h in c['files'].items())
            and set(c['build_files']) == set(BUILD_FILES), 'Local library binding changed')
    receipt = out/'construction_acceptance.json'
    if receipt.exists():
        accepted = vr.read(receipt)
        require(accepted['passed'] and accepted['configuration_sha256'] == plan['configuration_sha256']
                and set(accepted['materials']) == {'cases.json', 'skills.json', 'catalog.json', 'tree_report.json'}
                and all(vr.sha(out/'materials'/n) == h for n, h in accepted['materials'].items())
                and vr.sha(out/'material_isolation.json') == accepted['material_isolation_sha256'],
                'Published library changed')
    changed = {n: h for n, h in c['build_files'].items() if vr.sha(final.a.ROOT/n) != h}
    if changed:
        # Only an already published, byte-identical library may bind archived code.
        commit = ARCHIVED_BUILDS.get(vr.sha(receipt)) if receipt.is_file() else None
        require(commit
                and set(changed) <= {'scripts/run_verifier_library_scale.py', 'scripts/run_node14_library_scale.py'},
                'Local library binding changed')
        for name, expected in changed.items():
            archived = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=final.a.ROOT)
            require(hashlib.sha256(archived).hexdigest() == expected, 'Archived build source changed')
    return plan


def build(out, plan, tokenizer, transport):
    cfg = plan['configuration']
    bundle = vr.read(out/'tree_bundle.json')
    progress = vr.Progress(out/'tree_progress.json', plan['configuration_sha256'], construction.tree.initial_state())
    journal = construction.c.corpus.Journal(out, transport, threading.Event())
    def call(stage, key, payload):
        return construction.model_json(out, journal, tokenizer, 'tree_'+stage, key, construction.tree.messages(stage,payload))[0]
    organizer = construction.tree.Organizer(bundle, progress, call)
    while organizer.state['phase'] != 'done':
        require(not (out/'STOP').exists(), 'STOP requested')
        organizer.step()
        construction.emit(out,'building_tree',phase=organizer.state['phase'],jobs=organizer.state['jobs_committed'],
                          candidates=cfg['candidates'],placed=len(organizer.state['placed']))
    products = organizer.products()
    immutable(out/'materials/catalog.json', products['catalog'])
    immutable(out/'materials/tree_report.json', {k:v for k,v in products.items() if k!='catalog'})


def accept(out, plan):
    verify(out)
    cfg = plan['configuration']
    bundle = vr.read(out/'tree_bundle.json')
    sources = {sid for sid,state in vr.read(out/'sources.json')['dispositions'].items() if state=='error_found'}
    rules = {r['rule_id']:set(r['sample_ids']) for r in bundle['rules']}
    require(all(s and s <= sources for s in rules.values()), 'Independent tree support escaped the draw')
    materials = out/'materials'
    tree = final.base.p.TreeMemory(vr.read(materials/'catalog.json'))
    tree_report = vr.read(materials/'tree_report.json')
    leaves = set(tree.corpus.leaves)
    unresolved = {r['rule']['rule_id'] for r in tree_report['unplaced_rules']}
    require(leaves and leaves.isdisjoint(unresolved) and leaves | unresolved == rules.keys()
            and tree_report['counts']['all_candidates_preserved'], 'Missing tree candidates')
    screen = final.a.bounded.FamilyScreen(vr.read(out/'heldout_questions.json'))
    matches = []
    for store in (tree.memory, tree.nodes):
        for sid in store.ids:
            doc = store.view(sid)
            found = screen.matches(dict(id=sid, question='', prediction=doc['title']+'\n'+doc['text']))
            if found:
                matches.append(dict(id=sid, matches=found))
    immutable(out/'material_isolation.json', dict(tree_matches=matches, flat_material='Identical source-local records from screened release',
              heldout_sha256=cfg['files']['heldout_questions.json']))
    require(not matches, 'New tree text matches a held-out question; see material_isolation.json')
    source_files = vr.read(out/'original_evidence.json')
    vr.EvidenceCache(out/'source_cache.json').check(source_files, full=True)
    outputs = {n:vr.sha(materials/n) for n in ('cases.json','skills.json','catalog.json','tree_report.json')}
    requests = {}
    for path in sorted((out/'decoded').glob('tree_*/*.json')):
        ev = vr.read(path)['evidence']
        require(vr.sha(out/ev['file'])==ev['sha256'], 'Tree response evidence changed')
        requests[ev['file']] = ev['sha256']
    require(bool(requests), 'Fresh tree has no model responses')
    immutable(out/'construction_acceptance.json',dict(passed=True, configuration_sha256=plan['configuration_sha256'],
              materials=outputs, cases=cfg['flat_counts']['cases'], skills=cfg['flat_counts']['skills'],
              tree_counts=tree_report['counts'], tree_responses=len(requests), response_files=requests,
              source_counts=cfg['source_counts'], material_isolation_sha256=vr.sha(out/'material_isolation.json'),
              human_gold=False, scope='Independent source draw, original source-local calls, fresh tree; no benchmark evaluation'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='node14 qwen30b_step_verifier results')
    parser.add_argument('--seed', required=True, type=int, choices=SEEDS)
    parser.add_argument('--size', type=int, choices=BUILD_SIZES, default=250)
    parser.add_argument('--apply-source-hold', action='store_true', help='Apply the confirmed disposition to affected draws')
    args = parser.parse_args()
    root = args.root.resolve()
    directory = root/'library_scale'/f'n{args.size}_seed{args.seed}'
    directory.mkdir(exist_ok=True, parents=True)
    with final.base.r.run_lock(directory/'active.lock'):
        prepare(root, directory, args.seed, args.apply_source_hold, args.size)
