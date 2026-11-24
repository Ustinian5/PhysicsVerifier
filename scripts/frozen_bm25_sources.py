"""Bind the two first-review libraries to the published BM25 comparison. Read-only inputs."""
from pathlib import Path

from rule_framework import shared_experience as shared
from scripts import rerun_reviewed_libraries as rebuild
from scripts import run_final_answer_revision as final
from scripts import run_node14_final_pilot as host

vr, require = final.vr, final.require
RESULTS = host.HOT / 'results/qwen30b_step_verifier'
OUTPUT = RESULTS / shared.REVIEWED_DIRECTORY / 'bm25_n2000'
TREE = '76ab53f2b97831d267dae24a3fdf3287d33f57ff900f7e1bc74a5dd607dbaf2b'
STEP_HOLD = {'cl_191_114424'}
FINAL_HOLD = {
    '29f903a2100995e8543bd13ff9dc024863ae75ecff292c7498542189520230df',
    'c5d88750ee5ccbde362fa4f7d18003b3e3d1790dbd8fb69a1aae23120550fd6e',
    'e6229419a6d028cc7af63c97b773062c620c89be1f6d360aac9733367e11769a',
    '1566ceaddcc0cd437806489cdb8df823b8464e497acd1dde111bd367105ad48e',
    'ca96a8d3c895890406bae4c823a3c779e866d4609df2edd373bcdb54a7733a22',
    'e04a4b2c2c5fedaf8e45633c603d3ce21ddc3ad087de90b124826cb63358b09c',
}


def freeze(root=RESULTS, out=OUTPUT):
    prior = root / 'aligned_old99/evaluation_revision'
    current = root / shared.REVIEWED_DIRECTORY
    source = root / shared.DIRECTORY
    index = shared.release_index(source)
    ledgers = [vr.read(base / 'source_reviews.json') for base in (source, current)]
    groups = {}
    for name in index['libraries']:
        directory = shared.library_path(root, name)
        reviewed = shared.reviewed_flat_library(root, name)
        a, b = [vr.read(p / 'materials/cases.json') for p in (directory, reviewed)]
        first = vr.read(directory / 'sources.json')
        selected = set(first['common_reviews'])
        require(len(a) == len(selected) and {d['source_ids'][0] for d in a} == selected
                and all(ledgers[0][s] == ledgers[1][s] for s in selected), 'First-review evidence changed')
        require(a == [d for d in b if d['source_ids'][0] in selected], 'First-review RAG text differs')
        skills = vr.read(directory / 'materials/skills.json')
        require(skills == rebuild.project_skills(vr.read(reviewed / 'materials/skills.json'), selected),
                'Skills differ from source-local first-review extraction')
        groups[name] = dict(drawn=len(first['draw']), cases=len(a), skills=len(skills),
                            material_path=str(directory / 'materials'),
                            files={k: vr.sha(directory / 'materials' / k) for k in ['cases.json', 'skills.json']})
        if name == 'n2000':
            historical = {d['source_ids'][0] for d in vr.read(prior / 'cases.json')}
            require(len(historical) == 739 and historical - set(first['source_hold']) == selected
                    and len(selected) == 729, 'Published source membership differs')
    require(len(groups) == 13 and all(vr.sha(root / 'aligned_old99' / d / 'catalog.json') == TREE
            for d in ('evaluation_revision', 'final_revision_balanced')), 'Frozen tree or registered groups changed')
    record = dict(groups=groups, measured_tree_sha256=TREE, measured_leaves=895, original_sources=739,
                  first_review_date='2026-11-07', matched_sources=729, held_sources=10,
                  construction_calls=0, new_trees=0, step_hold=sorted(STEP_HOLD), final_hold=sorted(FINAL_HOLD),
                  release_sha256=shared.RELEASE_SHA256, reviewed_release_sha256=shared.REVIEWED_FLAT_SHA256)
    final.immutable(out / 'sources.json', record)
    return record


if __name__ == '__main__':
    with final.base.r.run_lock(OUTPUT / 'source.lock'):
        report = freeze()
        print('Verified source records and original materials for', len(report['groups']), 'libraries.', flush=True)
