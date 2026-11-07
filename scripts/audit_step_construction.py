"""Read-only construction provenance audit for the corrected 30B experiment.

No generation, no physical relabeling and no automatic runtime acceptance.
Checks original-source membership, every reused 30B generation, immutable 120B
request evidence, leaf provenance, and the current held-out question faces.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from rule_framework import released_corpus
from scripts import run_qwen30b_step_full as full

vr, require, knowledge = full.vr, full.require, full.knowledge


def membership(pool, rows):
    index = {r['id']: r for r in pool}
    require(len(index) == len(pool), 'Duplicate original source IDs')
    for row in rows:
        require(row['id'] in index and row['question'] == index[row['id']]['question'] and
                not row.get('answer') and not row.get('context'),
                'Construction source is external, altered or includes a reference')


def same_answers(generated, selected):
    """Construction may use its own frozen order, never a changed answer."""
    left, right = ({r['id']: r for r in rows} for rows in (generated, selected))
    require(len(left) == len(generated) and len(right) == len(selected) and left == right,
            'Library did not use exactly the verified complete 30B answers')


def check_artifact(item):
    path, expected = item
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, 'Construction evidence changed: '+str(path))
    stage = None
    if path.name.startswith('request_') and path.suffix == '.json':
        record = json.loads(data); request = record.get('request', {})
        require(request.get('model') == full.base.r.EXPECTED_MODEL and
                request.get('reasoning_effort') == 'medium', 'Construction requested a different model/effort')
        if record.get('response', {}).get('model'):
            require(record['response']['model'] == full.base.r.EXPECTED_MODEL, 'Construction response model differs')
        stage = record.get('client_metadata', {}).get('build_stage')
        require(stage in {'taxonomy','extract_experience','repair_experience','cluster_rules'},
                'Unapproved construction call stage; cannot certify no embedding/teacher')
    return stage


def audit(output, corpus, library, bench):
    output.parent.mkdir(parents=True, exist_ok=True)
    with full.base.r.run_lock(output.with_suffix('.lock')):
        def state(phase, **extra):
            vr.atomic(output, dict(state=phase, updated_unix=time.time(), model_calls=0,
                                  human_gold=False, formal_release_accepted=False, **extra))
        try:
            state('checking_original_source_and_30b_generation')
            source = vr.read(corpus/'source_manifest.json')
            require(source['sha256'] == 'ff31782395fffce1e367e29cdbd4ac1a68abf7b7ea4d4363541218e40c7c9b8b' and
                    source['exported_rows'] == 6501 and vr.sha(Path(source['source_path'])) == source['sha256'],
                    'Original source file changed')
            binding, rows = released_corpus.load(corpus/'release_manifest.json', corpus/'answers.json', complete_answers=True)
            membership(vr.read(corpus/'question_pool.json'), rows)
            require(binding == vr.read(library/'source_release.json'), 'Generation binding changed')
            selected = vr.read(library/'inputs.json')
            same_answers(rows, selected)
            release = vr.read(library/'release_manifest.json'); plan = vr.read(library/'plan.json')
            c = plan['configuration']
            require(vr.digest(c) == plan['configuration_sha256'] == release['configuration_sha256'] and
                    vr.digest(selected) == c['selected_sha256'] and
                    c['source_prediction_model'] == full.corpus.MODEL and c['external_teacher'] is None and
                    c['embedding_model'] is None and not c['old_rules_imported'] and not c['old_judgments_imported'],
                    'Library protocol is not the corrected source/model route')
            artifacts = {library/n: h for n,h in release['files'].items()}
            parent = vr.read(library/'parent_preparation.json')
            artifacts.update({Path(parent['parent'])/n: h for n,h in parent['files'].items()})
            artifacts.update({Path(parent['parent'])/n: h for n,h in parent.get('prefix_files', {}).items()})
            stages = Counter()
            state('checking_bound_120b_construction', generation_count=len(rows), artifact_count=len(artifacts))
            with ThreadPoolExecutor(max_workers=16) as executor:
                for i, stage in enumerate(executor.map(check_artifact, artifacts.items()), 1):
                    if stage: stages[stage] += 1
                    if i % 250 == 0: state('checking_bound_120b_construction', checked=i, total=len(artifacts), request_stages=dict(stages))
            cases = vr.read(library/'semantic_experience.json')
            errors = {x['sample_id'] for x in cases if x['status']=='ok' and x['semantic_audit']['status']=='error_found'}
            provenance = vr.read(library/'provenance.json'); catalog = vr.read(library/'catalog.json')
            ids = {r['rule_id'] for d in catalog['domains'] for t in d['topics'] for r in t['rules']}
            require(ids == {x['rule_id'] for x in provenance} and all(set(x['sample_ids']) <= errors for x in provenance),
                    'Rule leaf sources are outside verified error experiences')
            require(catalog['metadata']['embedding_calls'] == 0 and catalog['metadata']['historical_seed_rules'] == 0,
                    'Old leaf or embedding import')
            held = vr.read(bench); held_ids = {x['id'] for x in held}; screen = knowledge.FamilyScreen(held)
            overlaps = []
            state('checking_current_heldout_visible_materials', generation_count=len(rows), request_stages=dict(stages))
            for row in full.visible_materials(library):
                hits = [h for h in screen.matches(row) if h['heldout_id'] in held_ids]
                if hits: overlaps.append(dict(source_id=row['id'], matches=hits))
            sources = {str(p): vr.sha(p) for p in [corpus/'source_manifest.json', corpus/'question_pool.json',
                corpus/'release_manifest.json', library/'plan.json', library/'release_manifest.json',
                library/'source_release.json', library/'catalog.json', library/'provenance.json', bench]}
            state('audit_complete' if not overlaps else 'heldout_overlap', provenance_passed=True,
                  heldout_isolation_passed=not overlaps, overlaps=overlaps, original_source_only=True,
                  original_records=6501, verified_complete_30b_answers=len(rows),
                  error_experiences=len(errors), rule_leaves=len(ids), heldout_questions=len(held),
                  construction_requests_by_stage=dict(stages), artifacts_verified=len(artifacts),
                  generation_binding=binding, source_files=sources,
                  note='Provenance only; not a 5000-question build or a physical quality certification. Formal material scope remains a separate decision.')
        except BaseException as exc:
            state('stopped', error_type=type(exc).__name__, error=str(exc)[:1500]); raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('output','corpus','library','bench'): p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();audit(a.output.resolve(),a.corpus.resolve(),a.library.resolve(),a.bench.resolve())
