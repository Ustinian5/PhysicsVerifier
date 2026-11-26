"""Publish the current rule fields from the audited November 26 first-review responses."""
from collections import Counter
from pathlib import Path
import tempfile

from rule_framework import review_fields
from scripts import run_corrected_extraction as extract

read, write, sha, digest, require = extract.read, extract.write, extract.sha, extract.digest, extract.require
AUDIT = 'construction_symbolic_hint_audit.json'
RAW_REPLAY = 'construction_symbolic_hint_raw_replay.json'
AUDIT_SHA = '60f8beb98412f91c640884a265322e11555b9a1408d0f6a1b446d938f476b685'
REPLAY_SHA = '4d8118e4ad4c9d02d18319dd3a8920fdb8934826be291751eb3a449cd33340c4'
SOURCE_PLAN = '736ddd98948a0fe20ee4ab97d0c9678a225eaf8850d2e1c4c27877839884a458'
SHARED = ('inputs.json', 'taxonomy.json', 'build_scope.json', 'topic_skeleton.json',
          'source_evidence.json', 'generation_failures.json')
PRODUCTS = (*SHARED, 'semantic_experience.json', 'distilled.json', 'invalid_outputs.json', 'execution_status.json')


def select(row, cfg, taxonomy, source, audit):
    sid = row['id']
    record = read(source/'experiences'/(sid + '.json'))
    payload = dict(sample=row, taxonomy=taxonomy, max_rules_per_sample=cfg['max_rules_per_sample'])
    require(record['id'] == audit['id'] == sid and record['signature'] == digest([payload, cfg]) and
            1 <= len(record['evidence']) == len(audit['attempts']) <= 2, 'Wrong archived first review')
    evidence, selected, value, error = [], 'invalid', None, None
    for i, (call, proof) in enumerate(zip(record['evidence'], audit['attempts'])):
        stage = ('extract_experience', 'repair_experience')[i]
        raw = (source/call['file']).resolve(strict=True)
        decoded = source/'decoded'/stage/(digest(sid) + '.json')
        require(raw.is_relative_to(source/'raw'/stage) and proof['stage'] == stage and
                sha(raw) == call['sha256'] == proof['raw_sha256'] and sha(decoded) == proof['decoded_sha256'],
                'Archived response differs from the completed raw replay')
        draft = read(decoded)
        require(draft['evidence'] == call, 'Decoded draft points to a different response')
        if value is not None:
            continue  # An already recorded second attempt is evidence, but was not selected.
        evidence.append(dict(file=str(raw), sha256=call['sha256']))
        data = draft['value']
        error = data.get('_content_failure')
        if error is None:
            try:
                value = extract.old.extraction.normalize_120b_experience(data, **payload, legacy_symbolic=False)
            except ValueError as exc:
                error = str(exc)
            else:
                selected = ('first', 'existing_repair')[i]
        require(error == proof['canonical_error'], 'Physics, evidence or rule validation changed')
    require(selected == audit['canonical_branch'] and
            (value['semantic_audit']['status'] if value is not None else 'invalid_output') == audit['canonical_status'],
            'Different source selection after removing the retired field')
    if value is not None:
        # Compare the other fields against the already signed audit, in its original serialization only.
        previous = dict(value, experience_rules=[dict(r, symbolic_hint=dict(primitive='none', canonical='',
                             required_symbols=[])) for r in value['experience_rules']])
        require(digest(previous) == audit['canonical_value_sha256'], 'Other first-review fields changed')
        value = review_fields.without_rule_metadata(value)
    return selected, value, error, evidence


def verify_files(out, cfg):
    receipt = read(out/'field_release.json')
    original = Path(cfg['archived_review_plan'])
    source = original.parent.parent
    require(cfg['archived_configuration_sha256'] == digest(read(original)['configuration']) == SOURCE_PLAN and
            sha(source/AUDIT) == AUDIT_SHA and sha(source/RAW_REPLAY) == REPLAY_SHA, 'Archived evidence changed')
    counts = read(source/AUDIT)['summary']
    require(receipt['selection'] == counts['fixed_selection'] and receipt['counts'] == counts['fixed_final'] and
            receipt['rules'] == counts['canonical']['deduplicated_rules'] and receipt['new_model_calls'] == 0 and
            receipt['rule_schema'] == cfg['rule_schema'] == review_fields.SCHEMA and set(receipt['files']) == set(PRODUCTS) and
            receipt['original_plan'] == SOURCE_PLAN and receipt['audit_sha256'] == AUDIT_SHA and
            receipt['raw_replay_sha256'] == REPLAY_SHA and receipt['configuration_sha256'] == digest(cfg) and
            all(sha(out/name) == h for name,h in receipt['files'].items()) and
            receipt['experiences_digest'] == digest({p.stem:sha(p) for p in (out/'experiences').glob('*.json')}),
            'Published first-review projection changed')
    return receipt


def prepare(source, out):
    source, out = source.resolve(strict=True), out.absolute()
    upstream = source.parent
    require(out != source and sha(upstream/AUDIT) == AUDIT_SHA and sha(upstream/RAW_REPLAY) == REPLAY_SHA,
            'Different completed first-review audit')
    audit = read(upstream/AUDIT)
    original = read(source/'plan.json')
    old = original['configuration']
    require(original['configuration_sha256'] == digest(old) == SOURCE_PLAN and
            all(sha(source/n) == h for n,h in audit['source_hashes'].items()) and
            digest(read(source/'build_scope.json')) == old['build_scope_sha256'],
            'Original construction release changed')
    cfg = dict(old, rule_schema=review_fields.SCHEMA, archived_review_plan=str(source/'plan.json'),
               archived_configuration_sha256=SOURCE_PLAN, archived_prompt_digest=digest(old['audit_system_prompt']),
               audit_system_prompt=extract.AUDIT_PROMPT_FILE.read_text().rstrip('\n'),
               policy='First valid archived draft under the current rule fields, then its recorded repair if needed.')
    names = set(old['sources']) | {'rule_framework/review_fields.py', 'scripts/project_verifier_first_reviews.py'}
    cfg['sources'] = {n:sha(extract.c.ROOT/n) for n in sorted(names)}
    if out.exists():
        require(read(out/'plan.json')['configuration'] == cfg, 'Different first-review projection already exists')
        extract.verify(out)
        return read(out/'field_release.json')

    rows, taxonomy = read(source/'inputs.json'), read(source/'taxonomy.json')
    require(len(rows) == 1952 and [x['id'] for x in rows] == [x['id'] for x in audit['samples']] and
            digest(rows) == old['inputs_sha256'], 'Different frozen construction population')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as staging:
        release = Path(staging)/'construction'
        release.mkdir()
        for name in SHARED:
            (release/name).symlink_to(source/name)
        valid, invalid, statuses, branches, records = [], [], Counter(), Counter(), {}
        for row, detail in zip(rows,audit['samples']):
            sid = row['id']
            branch, value, error, evidence = select(row,old,taxonomy,source,detail)
            branches[branch] += 1
            statuses[detail['canonical_status']] += 1
            if value is None:
                invalid.append(dict(id=sid, error=error))
            else:
                valid.append(value)
            payload = dict(sample=row, taxonomy=taxonomy, max_rules_per_sample=cfg['max_rules_per_sample'])
            path = release/'experiences'/(sid+'.json')
            write(path, dict(id=sid, signature=digest([payload,cfg]), value=value, error=error,
                             evidence=evidence, selected=branch, human_gold=False))
            records[sid] = sha(path)
        distilled = extract.old.extraction._build_distilled_library(valid, min_count=1, legacy_symbolic=False, include_auxiliary=False)
        summary = audit['summary']
        require(dict(statuses) == summary['fixed_final'] and dict(branches) == summary['fixed_selection'] and
                len(distilled['rules']) == summary['canonical']['deduplicated_rules'], 'Different aggregate review release')
        write(release/'plan.json',dict(configuration=cfg,configuration_sha256=digest(cfg)))
        write(release/'semantic_experience.json',valid)
        write(release/'distilled.json',distilled)
        write(release/'invalid_outputs.json',invalid)
        write(release/'execution_status.json',dict(state='extraction_complete',processed=len(rows),counts=dict(statuses),
              rules=len(distilled['rules']),tree_complete=False,skills_complete=False))
        receipt = dict(configuration_sha256=digest(cfg),original_plan=SOURCE_PLAN,audit_sha256=AUDIT_SHA,
                       raw_replay_sha256=REPLAY_SHA,rule_schema=review_fields.SCHEMA,new_model_calls=0,
                       selection=dict(branches),counts=dict(statuses),rules=len(distilled['rules']),
                       experiences_digest=digest(records),files={name:sha(release/name) for name in PRODUCTS})
        write(release/'field_release.json',receipt)
        extract.verify(release)
        release.rename(out)
    return receipt
