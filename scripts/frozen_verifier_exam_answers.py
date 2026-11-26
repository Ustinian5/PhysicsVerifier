"""Bind published Combined Language answers to the exact frozen verifier inputs."""
from pathlib import Path

from rule_framework import verified_resume as vr, verifier_english as en
from rule_framework.flat_skill_rag import require, source_payload
from rule_framework.oss120b_construction import question_key
from scripts.combined_language_io import iter_rollout_batches
from scripts.combined_language_samples import extract_question
from scripts.run_final_answer_labels import CONTROLS

DATA_SHA = 'c1c2401fb8dd8a730dd3777f5d7103bcd4c1a2750f42b5b58911ec2b49da7b24'
PILOT_SHA = '618991c554aebc6d2875aab871eaa6a8713e6089008534b7718e586254385855'
RELEASES = {
    'step': ('29222b9f6a2ced51616d069bde2242b02fc61b8a913d92df17a68c0477455363', 'answers', 'label'),
    'final': ('af20cbb402c2ac4311b4ebc669403a5d3ffa53a669079e9c601c43e35b96fa01', 'rows', 'answer'),
}


def binding(row, answer):
    return dict(input_sha256=vr.digest(source_payload(row)), label=en.exam_answer(answer))


def official(root, task, rows, files):
    sha, records_field, answer_field = RELEASES[task]
    path = root / (task + '_exam_reference/source_answers.json')
    require(vr.sha(path) == sha, 'Different official-answer publication')
    release = vr.read(path)
    require(release['dataset_sha256'] == DATA_SHA, 'Different Combined Language data')
    records = release[records_field]
    require({r['id'] for r in rows} == set(records) and len(rows) == len(records), 'Wrong formal reference cohort')
    result = {}
    for row in rows:
        record = records[row['id']]
        value = binding(row, record[answer_field])
        require(value['input_sha256'] == record['input_sha256'], 'Exam answer belongs to a different 30B input')
        result[row['id']] = value
    files[str(path)] = sha
    return result


def development(root, data, files):
    path = root / 'aligned_old99/final_revision_balanced/pilot_inputs.json'
    require(vr.sha(path) == PILOT_SHA and vr.sha(data) == DATA_SHA, 'Different development set or raw data')
    rows = vr.read(path)
    synthetic = {r['id']: r for r in CONTROLS[:2]}
    pending, bound = {}, {}
    for row in rows:
        sid = row['id']
        if sid in synthetic:
            control = synthetic[sid]
            require(source_payload(control) == row, 'Synthetic control changed')
            bound[sid] = binding(row, control['reference'])
        else:
            tag, rollout, sample = sid.split('_')
            require(tag == 'cl', 'Unknown development source')
            pending.setdefault(int(rollout), {})[int(sample)] = row
    for batch in iter_rollout_batches(data):
        targets = pending.pop(batch['rollout_id'], {})
        for sample in batch['samples'] if targets else ():
            row = targets.pop(sample['index'], None)
            if row is not None:
                require(question_key(extract_question(sample)) == question_key(row['question']),
                        'Development question differs at original raw position')
                bound[row['id']] = binding(row, sample['label'])
        require(not targets, 'Missing raw development sample')
        if not pending:
            break
    require(len(rows) == len(bound) == 6 and not pending, 'Incomplete fixed development references')
    files.update({str(path): PILOT_SHA, str(data): DATA_SHA})
    return rows, bound
