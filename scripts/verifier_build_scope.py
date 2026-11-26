"""Bind the full construction roster, its held-out families and the first-review text."""
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from rule_framework import verified_resume as vr


PROJECT = Path(__file__).resolve().parents[1]
SCOPE_FILE = PROJECT / 'experiments/verifier_build_scope.json'
SCOPE_SHA = '45069193260a70295387de18c96017b3331f440eadde4c49a658a30d1f7e82ea'
HIPHO_SHA = '6ca95621b3b37490fbf4d4092d82e0d4dd471b4d6c9fe6f3bf3b957f7e802a59'
HIPHO_AUDIT_SHA = '7da3aef051672584a464a52a0edebb6be08da2f7e808bcdfc4e58da6f9c6bf2d'


def source_scope(results):
    from scripts import rerun_reviewed_libraries as prior
    from scripts import run_step_match_judge as judge
    from scripts import run_aligned_step_evaluation as step

    evidence = {}

    def bind(path, expected):
        if vr.sha(path) != expected:
            raise ValueError(f'Frozen construction or held-out file changed: {path}')
        evidence[str(path.resolve())] = expected
        return path

    reviewed = vr.read(bind(prior.REVIEW, prior.REVIEW_SHA))
    update = vr.read(bind(SCOPE_FILE, SCOPE_SHA))
    screen = results / 'corrected/source_screen'
    split = vr.read(bind(screen / 'split.json', reviewed['split_sha256']))
    families = vr.read(bind(screen / 'family_components.json', reviewed['families_sha256']))['components']
    build = [r['id'] for r in split['build']]
    reserve = {r['id'] for r in split['reserve']}
    if (len(build) != len(set(build)) or len(build) != 2000 or len(reserve) != 702 or set(build) & reserve
            or update['split_sha256'] != reviewed['split_sha256']):
        raise ValueError('Different source population')
    isolation = vr.read(bind(results / 'aligned_old99/evaluation/isolation.json', reviewed['initial_isolation_sha256']))
    later = vr.read(bind(results / 'aligned_old99/evaluation_revision/source_quarantine.json',
                         reviewed['later_source_quarantine_sha256']))
    benchmark = results / 'aligned_old99/evaluation'
    step_inputs = vr.read(bind(benchmark / 'inputs.json', judge.BENCH_FILES['inputs.json']))
    final = results / 'aligned_old99/final_revision_balanced'
    release = vr.read(bind(final / 'bench_release.json', 'ff89b8ef9f0fc9d894f06051057e634849db572e27e34f970dcc8273ec28af91'))
    final_inputs = vr.read(bind(final / 'inputs.json', release['inputs_sha256']))
    final_ids = {r['id'] for r in final_inputs}
    pilot = step.frozen_development_inputs(final / 'pilot_inputs.json', final_ids | {r['id'] for r in step_inputs}, evidence)
    if len(step_inputs) != 100 or len(final_ids) != 474 or not final_ids <= reserve:
        raise ValueError('Different registered benchmark')
    cohorts = {'build': split['build'], 'reserve': split['reserve'], 'step': step_inputs, 'pilot': pilot}

    def member(entry, cohort):
        position = entry['position']
        if not 1 <= position <= len(cohorts[cohort]) or cohorts[cohort][position - 1]['id'] != entry['id']:
            raise ValueError('Different question at registered position')
        return entry['id']

    related = set()
    comparison = []
    for family in update['families']:
        sources = [member(x, 'build') for x in family['build']]
        targets = [member(x, x['cohort']) for x in family['heldout']]
        if not sources or not targets or related.intersection(sources):
            raise ValueError('Empty family or overlapping source decisions')
        related.update(sources)
        comparison.append(dict(family=family['name'], build=sources, heldout=targets,
                               current_bench=any(x['cohort'] == 'step' or x['id'] in final_ids for x in family['heldout'])))
    boundaries = {member(x, 'build') for x in update['inference_holds']}
    views = {member(x, 'build'): x for x in update['retained_passages']}
    for entry in views.values():
        for ref in entry['shared_prefix_with']:
            member(ref, 'reserve')
    known = isolation['known_family_exclusion']
    if (not any(known['build_source'] in f['build'] and known['test_source'] in f['heldout'] for f in comparison)
            or related & boundaries or set(views) & (related | boundaries)
            or any((set(g['members']) & (set(build) - related)) and (set(g['members']) & reserve) for g in families)):
        raise ValueError('Conflicting family or input-scope decisions')
    quality_only = ((set(isolation['excluded_source_ids']) - {known['build_source']}
                     - {sid for sid, val in reviewed['parent_phrase_review'].items() if val['status'] == 'eligible'})
                    | set(later['excluded']))
    groups = dict(heldout_family=sorted(related), outside_model_processing=sorted(boundaries))
    audit_path = results / 'hipho_overlap_audit.json'
    audit = vr.read(bind(audit_path, HIPHO_AUDIT_SHA))
    if (audit['benchmark_sha256'] != HIPHO_SHA or
            audit['status'] != 'screening_and_assistant_review_complete_no_confirmed_overlap'):
        raise ValueError('Invalid external isolation audit')
    bind(PROJECT / 'data/heldout/hipho_frozen.jsonl', HIPHO_SHA)
    registry = [dict(id=kind + ':' + r['id'], question=r['question'], context=r.get('context') or '')
                for kind, rows in [('step', step_inputs), ('reserve', split['reserve']), ('pilot', pilot)] for r in rows]
    return dict(registered=build, excluded=groups, heldout=registry, family_decisions=comparison,
                retained_passages=views, prior_input_quality_flags=sorted(quality_only),
                hipho_audit=str(audit_path.resolve()), evidence=evidence)


def apply_scope(rows, failures, scope):
    complete = [r['id'] for r in rows]
    failed = {r['id'] for r in failures}
    if (len(failures) != len(failed) or set(complete) & failed or set(complete) | failed != set(scope['registered'])
            or complete != [sid for sid in scope['registered'] if sid not in failed]):
        raise ValueError('Original build order or response population changed')
    held = {sid for group in scope['excluded'].values() for sid in group}
    return [r for r in rows if r['id'] not in held]


def reviewed_input(row, scope):
    """Strip only a reviewed, unrelated pasted prefix; keep the complete original answer and label."""
    view = scope.get('retained_passages', {}).get(row['id'])
    if view is None:
        return row
    first = view['first_line'] - 1
    lines = row['question'].splitlines(keepends=True)
    kept = ''.join(lines[first:])
    def sha(text):
        return hashlib.sha256(text.encode()).hexdigest()
    if (not 0 < first < len(lines) or not kept.strip()
            or sha(row['question']) != view['question_sha256'] or sha(kept) != view['retained_sha256']
            or sha(row['prediction']) != view['prediction_sha256']
            or vr.digest(row['official_reference']['label']) != view['reference_digest']):
        raise ValueError('Independent subquestion, full 30B answer or official reference changed')
    return dict(row, question=kept)


def lexical_candidates(build, heldout):
    """Find short shared introductions and repeated prose across split boundaries for review."""
    prefixes, passages = defaultdict(set), defaultdict(set)

    def features(q):
        q = re.sub(r'\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]', ' formula ', q, flags=re.S)
        words = re.findall(r'[a-z]{2,}|[\u4e00-\u9fff]', unicodedata.normalize('NFKC', q).lower())
        phrases = {tuple(words[i:i+9]) for i in range(len(words)-8) if len(set(words[i:i+9]) - {'formula'}) >= 6}
        return tuple(words[:7]), phrases

    for row in heldout:
        prefix, phrases = features(row['question'])
        if len(prefix) == 7:
            prefixes[prefix].add(row['id'])
        for phrase in phrases:
            passages[phrase].add(row['id'])
    pairs = []
    for row in build:
        prefix, phrases = features(row['question'])
        counts = Counter(sid for phrase in phrases for sid in passages.get(phrase, ()))
        hits = prefixes.get(prefix, set()) | {sid for sid, n in counts.items() if n >= 3}
        for sid in sorted(hits):
            pairs.append(dict(build=row['id'], heldout=sid, same_introduction=sid in prefixes.get(prefix, ()),
                              shared_nine_word_phrases=counts[sid]))
    return pairs


def audit(results):
    """Reproduce the scope on the original records without preparing a formal release or calling a model."""
    from scripts import run_corrected_extraction as extraction
    import tempfile
    scope = source_scope(results)
    screen = results / 'corrected/source_screen'
    collection = results / 'corrected/collection'
    raw_build = vr.read(screen/'split.json')['build']
    external = vr.read(Path(scope['hipho_audit']))['question_references']
    external = [dict(x, id='hipho:' + x['id']) for x in external]
    candidates = lexical_candidates(raw_build, scope['heldout'] + external)
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory) / 'construction'
        extraction.prepare(folder, collection, PROJECT/'data/combined_language_only.json', results_root=results)
        prepared = extraction.verify(folder)['configuration']
        rows = {r['id']: r for r in vr.read(folder/'inputs.json')}
        failed = {r['id'] for r in vr.read(folder/'generation_failures.json')}
        generator = vr.read(collection/'plan.json')['configuration']['generator_config']
        records = []
        for i, source in enumerate(raw_build, 1):
            sid = source['id']
            decision = next((k for k, group in scope['excluded'].items() if sid in group), None)
            if sid in failed:
                decision = decision or 'incomplete_30b'
            else:
                generated = vr.read(collection/'generated'/(sid+'.json'))
                raw = vr.read(collection/generated['evidence']['file'])
                extraction.c.prior.validate_import(dict(id=sid, question=source['question']), generator, generated, raw)
            if decision is None:
                decision = 'independent_passage' if sid in scope['retained_passages'] else 'full_question'
                if sid not in rows:
                    raise ValueError('Eligible question missing from the actual first-review input')
            elif sid in rows:
                raise ValueError('Excluded question present in first-review input')
            records.append(dict(position=i, id=sid, disposition=decision))
        counts = dict(Counter(x['disposition'] for x in records))
        if len(records) != 2000 or counts.get('full_question', 0) + counts.get('independent_passage', 0) != len(rows):
            raise ValueError('Incomplete source accounting')
        return dict(registered=len(records), complete_30b=len(records)-len(failed), reviewable=len(rows),
                    generator=generator['generator'], heldout_benchmark_inputs=len(scope['heldout']),
                    external_heldout=len(external), source_manifest_sha256=SCOPE_SHA,
                    dispositions=counts, families=scope['family_decisions'], source_rows=records,
                    lexical_review_pairs=candidates, retained_passages=scope['retained_passages'],
                    retained_input_quality_flags=sorted(set(rows) & set(scope['prior_input_quality_flags'])),
                    evidence=scope['evidence'], inputs_digest=prepared['inputs_sha256'],
                    source_evidence_digest=prepared['evidence_sha256'], model_calls=0, formal_release=False)
