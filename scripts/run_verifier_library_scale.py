"""Resolve a shared-review scale library and verify its original independent tree receipts."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import subprocess

from rule_framework import verified_resume as vr
from scripts import run_corrected_pipeline as construction
from scripts import run_final_answer_revision as final
from scripts.audit_verifier_library_capacity import family_levels, SEEDS as ORIGINAL_SEEDS, SIZES


BUILD_SIZES = SIZES[:-1]  # The full 2000-source release is already frozen.
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
        commit = ARCHIVED_BUILDS.get(vr.sha(receipt), '060cbc6f001be3d1547d5f5ba61fdfa2075d9f23') if receipt.is_file() else None
        require(commit
                and set(changed) <= {'scripts/run_verifier_library_scale.py', 'scripts/run_node14_library_scale.py'},
                'Local library binding changed')
        for name, expected in changed.items():
            archived = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=final.a.ROOT)
            require(hashlib.sha256(archived).hexdigest() == expected, 'Archived build source changed')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='node14 qwen30b_step_verifier results')
    parser.add_argument('--seed', required=True, type=int, choices=(20261122, *RANDOM_SEEDS))
    parser.add_argument('--size', type=int, choices=BUILD_SIZES, default=250)
    parser.add_argument('--apply-source-hold', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    from scripts.release_shared_experience import ensure_publication
    from rule_framework.shared_experience import library_path
    ensure_publication(root)
    print(library_path(root, f'n{args.size}_seed{args.seed}') / 'materials')
