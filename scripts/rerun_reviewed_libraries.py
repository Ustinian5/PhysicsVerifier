"""Rebuild RAG and flat skills from the frozen first review, after source-only adjudication."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from collections import Counter

from rule_framework import organization_followup as bounded
from rule_framework import shared_experience as shared
from rule_framework import verified_resume as vr
from scripts import release_shared_experience as original
from scripts import run_verifier_library_scale as scale


REVIEW = Path(__file__).resolve().parents[1] / 'experiments/library_source_adjudication.json'
REVIEW_SHA = '1f785ae37498813885539203278d13975d306fc322e5400fb939caf14cdcb3f8'
PARENT_SHA = 'e088c9322e74d0b263de7a009bc1df40e614b5eb95863e839687fce74daf9bf8'
PARENT = 'shared_experience'
require = original.require


def checked_inputs(root):
    review = original.checked(REVIEW, REVIEW_SHA)
    old = root / PARENT
    require(vr.sha(old / 'release.json') == PARENT_SHA, 'Wrong parent publication')
    parent = vr.unseal(old / 'release.json')
    origin = original.inputs(root)
    archive = origin['pipe'].parent
    first_isolation = archive.parent / 'aligned_old99/evaluation/isolation.json'
    quarantine = first_isolation.parent.parent / 'evaluation_revision/source_quarantine.json'
    original.checked(archive / 'source_screen/split.json', review['split_sha256'])
    original.checked(archive / 'source_screen/family_components.json', review['families_sha256'])
    isolation = original.checked(first_isolation, review['initial_isolation_sha256'])
    later = original.checked(quarantine, review['later_source_quarantine_sha256'])
    phrase = review['parent_phrase_review']
    sentinel = dict(heldout_id='unresolved_parent', reason='dependent_question_parent_not_resolved')
    require({i['source_id'] for i in isolation['excluded'] if i['matches'] == [sentinel]} == set(phrase)
            and len(isolation['excluded']) == len(phrase), 'Different phrase exclusions')
    require(set(parent['source_hold']) == set(review['new_scope_hold']) == original.HELD,
            'Out-of-scope source decisions changed')
    kept_old = original.checked(old / 'source_reviews.json', parent['source_reviews_sha256'])
    require(origin['eligible'] == set(kept_old), 'Previous source cohort changed')
    heldout_counts = Counter(row['id'].split(':', 1)[0] for row in origin['heldout'])
    require(heldout_counts == {'step': 100, 'pilot': 6, 'hipho': 128, 'hipho_context': 128, 'final': 1302},
            'Incomplete held-out registry')
    restored = {sid for sid, d in phrase.items() if d['status'] == 'eligible'}
    reserved = {x['id'] for x in origin['split']['reserve']}
    build = {x['id']: x['question'] for x in origin['split']['build']}
    for sid, d in phrase.items():
        question = origin['samples'][sid]['question']
        require(d['status'] in {'eligible', 'missing_input'} and d['reason']
                and hashlib.sha256(question.encode()).hexdigest() == d['question_sha256']
                and question == build[sid], 'Reviewed question or status changed')
    for family in origin['families']:
        members = set(family['members'])
        require(not (members & restored and members & reserved), 'Restored family in reserve set')
    require(restored <= origin['excluded'] and not restored.intersection(later['excluded'])
            and isolation['known_family_exclusion']['build_source'] not in restored,
            'Restoration overlaps a separate exclusion')
    origin['eligible'] = origin['eligible'] | restored
    return origin, review, restored, kept_old


def check_matches(matches, source_id, restored):
    for match in matches:
        require(source_id in restored and match == dict(heldout_id='unresolved_parent',
                reason='dependent_question_parent_not_resolved'), f'Held-out evidence for {source_id}: {match}')


def rebuild_case(sample, audit):
    sid = sample['id']
    return dict(id='case_' + sid, kind='case', source_ids=[sid],
                title=' '.join(sample['question'].split())[:160],
                text='历史题目\n' + sample['question'] + '\n\n原30B回答（可能有错，不是正确范例）\n'
                     + sample['prediction'] + '\n\n120B原始核查说明\n' + json.dumps(audit, ensure_ascii=False))


def prepare_sources(root, work):
    from transformers import AutoTokenizer

    origin, review, restored, previous = checked_inputs(root)
    manifest = work / 'source_replay.json'
    if manifest.exists():
        cached = vr.unseal(manifest)
        require(cached['adjudication_sha256'] == REVIEW_SHA and cached['parent_sha256'] == PARENT_SHA
                and all(vr.sha(work / name) == h for name, h in cached['files'].items()),
                'Source replay changed')
        vr.atomic(work / 'progress.json', dict(stage='source_replay', completed=cached['source_count'],
                                             total=cached['source_count']))
        return cached
    tokenizer = AutoTokenizer.from_pretrained(origin['first']['model'], local_files_only=True, trust_remote_code=False)
    screen = bounded.FamilyScreen(origin['heldout'])
    ledger, outcomes, cases = {}, {}, {}
    with tempfile.TemporaryDirectory(dir=work) as scratch:
        for i, sid in enumerate(sorted(origin['eligible']), 1):
            ledger[sid], outcomes[sid] = original.replay_source(origin, sid, tokenizer, Path(scratch))
            sample = origin['samples'][sid]
            exp = original.bound(origin['ext'] / 'experiences' / (sid + '.json'), origin['evidence'])['value']
            cases[sid] = rebuild_case(sample, exp['semantic_audit'])
            require(cases[sid] == origin['cases'][sid] and vr.digest(cases[sid]) == ledger[sid]['case_sha256'],
                    'Reconstructed RAG differs from the case supplied to the skills extractor')
            if sid in previous:
                require(previous[sid] == ledger[sid], 'Common first review or skill parents changed')
            texts = [cases[sid]['text']]
            for item in outcomes[sid].get('value', {}).get('skills', []):
                texts.extend([item['title'], item['text']])
            check_matches(screen.matches(dict(id=sid, question=sample['question'], prediction='\n'.join(texts))),
                          sid, restored)
            if i == 1 or i % 25 == 0:
                vr.atomic(work / 'progress.json', dict(stage='source_replay', completed=i, total=len(origin['eligible'])))
    products = {'source_reviews.json': ledger, 'flat_cases.json': cases, 'skill_outcomes.json': outcomes}
    for name, value in products.items():
        vr.atomic(work / name, value)
    receipt = dict(parent_sha256=PARENT_SHA, adjudication_sha256=REVIEW_SHA, restored=sorted(restored),
                   reviewed=len(review['parent_phrase_review']), source_count=len(ledger),
                   first_review_responses=sum(len(v['first_responses']) for v in ledger.values()),
                   skill_responses=len(outcomes), skill_outcomes=dict(Counter(v['status'] for v in outcomes.values())),
                   heldout_sha256=vr.digest(origin['heldout']), files={n: vr.sha(work / n) for n in products},
                   new_model_calls=0)
    vr.seal(manifest, receipt)
    vr.atomic(work / 'progress.json', dict(stage='source_replay', completed=len(ledger), total=len(ledger)))
    return receipt


def registered_draws(root, origin):
    yield 'n2000', [row['id'] for row in origin['split']['build']], origin['pipe']
    for seed in (20261122, *scale.RANDOM_SEEDS):
        for size in scale.BUILD_SIZES:
            name = f'n{size}_seed{seed}'
            parent = root / 'library_scale' / name
            plan = scale.verify(parent)['configuration']
            accepted = vr.read(parent / 'tree_replay.json')
            require(accepted['passed'] and accepted['acceptance_sha256'] ==
                    vr.sha(parent / 'construction_acceptance.json'), 'Unverified independent tree')
            rows, _ = scale.source_draw(origin['split'], origin['families'], seed, True, size)
            draw = [r['id'] for r in rows]
            require(draw == vr.read(parent / 'sources.json')['original_draw']
                    and vr.digest(sorted(draw)) == plan['source_ids_sha256'], 'Registered draw changed')
            yield name, draw, parent


def group_inputs(root, work, mode):
    require(mode in ('reviewed', 'existing'), 'Unknown group source selection')
    origin, _, restored, previous = checked_inputs(root)
    replay = prepare_sources(root, work)
    store = {name: original.checked(work / name, h) for name, h in replay['files'].items()}
    ledger = store['source_reviews.json']
    cases, outcomes = store['flat_cases.json'], store['skill_outcomes.json']
    eligible = origin['eligible'] if mode == 'reviewed' else set(previous)
    require(set(ledger) == set(cases) == set(outcomes) == origin['eligible'], 'Incomplete source-local rebuild')
    exp = {sid: original.bound(origin['ext'] / 'experiences' / (sid + '.json'), origin['evidence'])['value']
           for sid in eligible}
    require(all(vr.digest(e['semantic_audit']) == ledger[sid]['audit_sha256'] for sid, e in exp.items()),
            'Rule candidate review differs from flat inputs')
    registration = dict(source_mode=mode, adjudication_sha256=REVIEW_SHA, parent_sha256=PARENT_SHA,
                        source_replay_sha256=vr.sha(work / 'source_replay.json'))
    write = scale.immutable
    output = work / ('flat_' + mode)
    parent_index = vr.unseal(root / PARENT / 'release.json')
    write(output / 'selection.json', registration)
    groups = {}
    for name, draw, parent in registered_draws(root, origin):
        directory = output / name
        local = [sid for sid in draw if sid in eligible]
        ids = sorted(local)
        old = root / PARENT / name
        parent_receipt = original.checked(old / 'construction_acceptance.json',
                                          parent_index['libraries'][name]['acceptance_sha256'])
        old_sources = original.checked(old / 'sources.json', parent_receipt['sources_sha256'])['common_reviews']
        require({s: ledger[s]['audit_sha256'] for s in set(draw) & set(previous)} == old_sources,
                'Old same-review group does not reproduce')
        common = {sid: ledger[sid]['audit_sha256'] for sid in ids}
        flat = [cases[sid] for sid in ids]
        skills, counts = bounded.materialize_skills([dict(id=s) for s in ids], [outcomes[s] for s in ids])
        selected = set(ids)
        require(skills == project_skills(origin['first_review_skills'], selected),
                'Skill is not derived solely from this draw')
        previous_flat = [d for d in flat if d['source_ids'][0] in old_sources]
        previous_skills = project_skills(skills, old_sources.keys())
        for file, value in [('cases.json', previous_flat), ('skills.json', previous_skills)]:
            require(value == original.checked(old / 'materials' / file, parent_receipt['materials'][file]),
                    'Shared cases or skills changed on the previous sources')
        rules, provenance = scale.local_rules([exp[s] for s in local], selected)
        for rid, support in provenance.items():
            for rec in support:
                sid, e = rec['sample_id'], rec['source_error']
                require(e['id'] in ledger[sid]['candidate_errors'][rid]
                        and vr.digest(e) == ledger[sid]['errors'][e['id']], 'Rule candidate has a different error')
        bundle = dict(taxonomy=origin['taxonomy'], rules=rules, provenance=provenance)
        if name == 'n2000':
            old_bundle = origin['bundle']
        else:
            old_bundle = original.checked(parent / 'tree_bundle.json',
                                          vr.read(parent / 'plan.json')['configuration']['files']['tree_bundle.json'])
        identical = all(v == old_bundle[k] for k, v in bundle.items())
        write(directory / 'materials/cases.json', flat)
        write(directory / 'materials/skills.json', skills)
        write(directory / 'tree_bundle.json', bundle)
        write(directory / 'group_sources.json', dict(draw=draw, common_reviews=common, restored=sorted(selected & restored)))
        files = {n: vr.sha(directory / n) for n in
                 ('materials/cases.json', 'materials/skills.json', 'tree_bundle.json', 'group_sources.json')}
        entry = dict(name=name, drawn=len(draw), cases=len(flat), skills=len(skills), candidates=len(rules),
                     skill_failures=counts.get('content_failure', 0), skill_empty=counts.get('valid_empty_sources', 0),
                     skill_sources=len({s for d in skills for s in d['source_ids']}),
                     restored=len(selected & restored), common_reviews_sha256=vr.digest(common),
                     unchanged_original_tree_input=identical, parent_tree=str(parent),
                     parent_review_receipt_sha256=vr.sha(old / 'construction_acceptance.json'), files=files)
        write(directory / 'flat_acceptance.json', entry)
        groups[name] = dict(entry, acceptance_sha256=vr.sha(directory / 'flat_acceptance.json'))
    require(len(groups) == 1 + (1 + len(scale.RANDOM_SEEDS)) * len(scale.BUILD_SIZES), 'Missing registered draw')
    evidence = dict(**registration, groups=groups, new_model_calls=0,
                    released_materials=['cases.json', 'skills.json'], human_gold=False)
    write(output / 'flat_release.json', evidence)
    return {k: {f: v[f] for f in ('cases', 'skills', 'candidates', 'restored', 'unchanged_original_tree_input')}
            for k, v in groups.items()}


def project_skills(skills, selected):
    projected = []
    for skill in skills:
        sources = [s for s in skill['source_ids'] if s in selected]
        if sources:
            projected.append(dict(skill, source_ids=sources))
    return projected


def compare_tree_inputs(root):
    """Compare candidate and supporting-source identities with the organizer's actual inputs."""
    work = root / shared.REVIEWED_DIRECTORY / 'flat_reviewed'
    origin, _, _, _ = checked_inputs(root)
    differences = {}
    for name, _, parent in registered_draws(root, origin):
        directory = shared.reviewed_flat_library(root, name)
        old = origin['bundle'] if name == 'n2000' else vr.read(parent / 'tree_bundle.json')
        new = vr.read(directory / 'tree_bundle.json')
        require(new['taxonomy'] == old['taxonomy'], 'Taxonomy changed')
        before, after = ({r['rule_id'] for r in b['rules']} for b in (old, new))
        before_sources, after_sources = ({s for r in b['rules'] for s in r['sample_ids']} for b in (old, new))
        differences[name] = dict(
            original_bundle_sha256=vr.sha(parent / 'tree_bundle.json'),
            reviewed_bundle_sha256=vr.sha(directory / 'tree_bundle.json'),
            previous_candidates=len(before), reviewed_candidates=len(after),
            added_candidates=len(after-before), removed_candidates=len(before-after),
            added_sources=len(after_sources-before_sources), removed_sources=len(before_sources-after_sources))
    report = dict(flat_release_sha256=shared.REVIEWED_FLAT_SHA256, groups=differences, new_model_calls=0)
    scale.immutable(work / 'tree_input_comparison.json', report)
    return report


def verify_groups(root):
    """Check released material against each source without running the materializer."""
    work = root / shared.REVIEWED_DIRECTORY
    folder = work / 'flat_reviewed'
    replay = vr.unseal(work / 'source_replay.json')
    ledger = original.checked(work / 'source_reviews.json', replay['files']['source_reviews.json'])
    index = original.checked(folder / 'flat_release.json', shared.REVIEWED_FLAT_SHA256)
    old = shared.release_index(root / PARENT)
    require(index['adjudication_sha256'] == REVIEW_SHA and index['groups'].keys() == old['libraries'].keys(),
            'Different decisions or draws')
    counts = {}
    for name in index['groups']:
        directory = shared.reviewed_flat_library(root, name)
        group = index['groups'][name]
        sources = vr.read(directory / 'group_sources.json')
        ids = set(sources['common_reviews'])
        draw = sources['draw']
        parent = shared.library_path(root, name)
        prior = vr.read(parent / 'sources.json')
        require(draw == prior['draw'] and len(draw) == len(set(draw)) == group['drawn']
                and ids == set(ledger).intersection(draw)
                and sources['common_reviews'] == {s: ledger[s]['audit_sha256'] for s in ids}
                and sorted(ids - prior['common_reviews'].keys()) == sources['restored'], 'Source cohort changed')
        flat = vr.read(directory / 'materials/cases.json')
        skills = vr.read(directory / 'materials/skills.json')
        require([c['source_ids'] for c in flat] == [[s] for s in sorted(ids)]
                and all(c['id'] == ledger[s]['case_id'] and vr.digest(c) == ledger[s]['case_sha256']
                        for c, s in zip(flat, sorted(ids))), 'Wrong first review in RAG')
        skill_sources = {}
        for sid in sorted(ids):
            for key in ledger[sid]['skill_ids']:
                skill_sources.setdefault(key, set()).add(sid)
        require(len(skills) == len(skill_sources) == group['skills'] and len(flat) == group['cases']
                and {s['id']: set(s['source_ids']) for s in skills} == skill_sources
                and all(s['id'] == 'procedure_' + vr.digest({k: s[k] for k in ('title', 'text')})[:24]
                        for s in skills), 'Skill text or parent source changed')
        require(sum(ledger[s]['skill_status'] == 'content_failure' for s in ids) == group['skill_failures'],
                'Incorrect handling of original failures')
        counts[name] = dict(cases=len(flat), skills=len(skills), restored=len(sources['restored']),
                            original_failures=group['skill_failures'])
    report = dict(passed=True, flat_release_sha256=shared.REVIEWED_FLAT_SHA256, libraries=counts,
                  new_model_calls=0)
    scale.immutable(folder / 'verification.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('replay', 'groups', 'verify', 'compare-tree-inputs'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, help='Defaults to the reviewed library directory under the results root')
    parser.add_argument('--source-mode', choices=('reviewed', 'existing'), help='Explicit group source decision')
    args = parser.parse_args()
    root = args.root.resolve()
    work = args.work.resolve() if args.work else root / shared.REVIEWED_DIRECTORY
    work.mkdir(parents=True, exist_ok=True)
    with original.build.c.corpus.r.run_lock(work / 'active.lock'):
        if args.stage == 'replay':
            value = prepare_sources(root, work)
        elif args.stage in {'verify', 'compare-tree-inputs'}:
            require(work == root / shared.REVIEWED_DIRECTORY, 'Verify the published result directory')
            value = verify_groups(root) if args.stage == 'verify' else compare_tree_inputs(root)
        else:
            require(args.source_mode, 'Choose the source cohort before preparing groups')
            value = group_inputs(root, work, args.source_mode)
        print(json.dumps(value, ensure_ascii=False))
