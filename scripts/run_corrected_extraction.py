"""Extract fresh experiences from the corrected, isolated 2000-question cohort.

Freeze the per-answer audit prompt when preparing a new extraction plan.
The new first review uses the original Combined Language exam solution.
This stage exports raw experiences and deduplicated candidates, not a tree.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import threading
import time
import httpx

from scripts import run_corrected_collection as c
from scripts import run_experience_first_120b as old
from scripts.combined_language_io import iter_rollout_batches
from scripts.combined_language_samples import extract_question
from rule_framework.oss120b_construction import question_key
from rule_framework import review_fields, verifier_english as en
from scripts import verifier_build_scope as scope
from scripts.run_qwen30b_step_checks import validate_effective_input

read, write, sha, digest, require = c.read, c.write, c.sha, c.digest, c.require
AUDIT_PROMPT_FILE = c.ROOT / 'rule_framework/build_review_prompt.txt'


def build_rows(source):
    plan = c.verify(source)
    tasks = read(source/'tasks.json')
    require(read(source/'execution_status.json')['state'] == 'generation_finished', 'Generation not finished')
    rows, rejected, evidence = [], [], {}
    for task in tasks:
        if task['role'] != 'build':
            continue
        q = task['row']; path = source/'generated'/(q['id']+'.json'); value = read(path)
        require(value['signature'] == digest([q, plan['configuration']['generator_config']]), 'Generation drift')
        c.corpus.evidence_valid(source, value)
        evidence[str(path)] = sha(path)
        evidence[str(source/value['evidence']['file'])] = value['evidence']['sha256']
        if value['state'] != 'complete':
            rejected.append(dict(id=q['id'], state=value['state'], finish_reason=value['finish_reason']))
            continue
        rows.append(dict(id=q['id'], question=q['question'], prediction=value['prediction'], context='', answer=''))
    require(len(rows)+len(rejected) == 2000, 'Wrong build population')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate source')
    return rows, rejected, evidence


def build_references(rows, screen, inventory_path, dataset):
    """Bind each completed build answer to the raw record that supplied its frozen question."""
    plan = read(screen/'plan.json')
    cfg = plan['configuration']
    require(digest(cfg) == plan['configuration_sha256'], 'Source-screen plan drift')
    split = read(screen/'split.json')
    require(sha(screen/'plan.json') == split['source_plan_sha256'], 'Different build split')
    source_ids = {row['id'] for row in split['build']}
    require(all(row['id'] in source_ids for row in rows), 'Reference requested outside build split')
    inventory = read(inventory_path)
    index = read(screen/'references_private.json')
    dataset_hash = sha(dataset)
    require(sha(inventory_path) == cfg['source_inventory_sha256'] and digest(index) == cfg['source_refs_sha256']
            and dataset_hash == index['source_sha256'] == cfg['source_sha256'], 'Different Combined Language source')
    originals = {item['id']: item for item in inventory['questions']}
    variants = {item['id']: item['variants'] for item in index['references']}
    pending = defaultdict(list)
    for row in rows:
        sid = row['id']
        original = originals[sid]
        require(question_key(original['question']) == question_key(row['question']), 'Different frozen question')
        loc = original['first_source']
        pending[loc['rollout_id']].append((row, loc))
    found = {}
    for batch in iter_rollout_batches(dataset):
        for row, loc in pending.pop(batch['rollout_id'], []):
            sample = batch['samples'][loc['sample_position']]
            sid = row['id']
            require(sample['index'] == loc['sample_index'] and
                    question_key(extract_question(sample)) == question_key(row['question']),
                    'Raw Combined Language question mismatch')
            label = en.exam_answer(sample['label'])
            require(any(v['first_source'] == loc and v['value'] == label for v in variants[sid]),
                    'Raw exam answer differs from the frozen source index')
            found[sid] = dict(label=label, original_location=loc)
        if not pending:
            break
    require(not pending and len(found) == len(rows), 'Incomplete build reference binding')
    return found, {str(dataset): dataset_hash, str(inventory_path): cfg['source_inventory_sha256'],
                   str(screen/'references_private.json'): sha(screen/'references_private.json'),
                   str(screen/'plan.json'): split['source_plan_sha256'], str(screen/'split.json'): sha(screen/'split.json')}


def prepare(out, source, dataset=None, *, results_root=None):
    require(not (out/'plan.json').exists(), 'Already prepared')
    audit_prompt = AUDIT_PROMPT_FILE.read_text(encoding='utf-8').rstrip('\n')
    require(bool(audit_prompt.strip()), 'Empty first-review prompt')
    rows, rejected, evidence = build_rows(source)
    approved = scope.source_scope(results_root or source.parent.parent)
    rows = scope.apply_scope(rows, rejected, approved)
    evidence.update(approved['evidence'])
    screen = (source/read(source/'plan.json')['configuration']['split_path']).resolve().parent
    references, bound = build_references(rows, screen, screen.parent.parent/'source_question_inventory.json',
                                         dataset or c.ROOT/'data/combined_language_only.json')
    rows = [scope.reviewed_input(dict(row, official_reference=references[row['id']]), approved) for row in rows]
    evidence.update(bound)
    skeleton = old.flow.experience_topic_skeleton(read(c.ROOT/'catalogs/rules_catalog_top_down.json'))
    cfg = dict(model=c.corpus.r.EXPECTED_MODEL, auditor_api='http://127.0.0.1:18083/v1',
        max_tokens=16384, max_request_input_bytes=400000, reasoning_effort='medium', timeout_seconds=1200,
        technical_attempts=3, retry_cooldown_seconds=60, max_rules_per_sample=2,
        structural_repair_rounds=1, fixed_date='2026-11-07', pilot_size=12,
        prompt_profile=en.PROFILE, rule_schema=review_fields.SCHEMA, audit_system_prompt=audit_prompt, build_scope_sha256=digest(approved),
        source_plan_sha256=sha(source/'plan.json'), source=str(source),
        inputs_sha256=digest(rows), generation_failures_sha256=digest(rejected),
        skeleton_sha256=digest(skeleton), evidence_sha256=digest(evidence),
        sources={n:sha(c.ROOT/n) for n in sorted(set(list(old.SOURCES)+list(c.SOURCES)+
            ['scripts/run_corrected_extraction.py','scripts/run_qwen30b_step_checks.py','rule_framework/verifier_english.py',
             'rule_framework/review_fields.py',
             'scripts/combined_language_io.py','scripts/combined_language_samples.py','scripts/verifier_build_scope.py']))},
        model_files=old.b.existing.MODEL_FILES, human_gold=False,
        policy='Frozen build order; one continuous audit+extraction; content failures retained; no resampling')
    for name, value in [('inputs',rows),('generation_failures',rejected),('topic_skeleton',skeleton),
                        ('build_scope',approved),('source_evidence',evidence)]:
        write(out/(name+'.json'), value)
    write(out/'plan.json',dict(configuration=cfg,configuration_sha256=digest(cfg)))
    write(out/'execution_status.json',dict(state='prepared',complete_inputs=len(rows),generation_failures=len(rejected)))


def verify(out):
    plan=read(out/'plan.json');cfg=plan['configuration']
    require(digest(cfg)==plan['configuration_sha256'], 'Plan drift')
    require(all(sha(c.ROOT/n)==h for n,h in cfg['sources'].items()),'Code drift')
    for name,key in [('inputs','inputs_sha256'),('topic_skeleton','skeleton_sha256'),('source_evidence','evidence_sha256')]:
        require(digest(read(out/(name+'.json')))==cfg[key], 'Input drift: '+name)
    if 'build_scope_sha256' in cfg:
        approved=read(out/'build_scope.json')
        require(digest(approved)==cfg['build_scope_sha256'], 'Source exclusions changed')
        held={sid for group in approved['excluded'].values() for sid in group}
        failures=read(out/'generation_failures.json')
        require(digest(failures)==cfg['generation_failures_sha256'], 'Generation failures changed')
        failed={r['id'] for r in failures}
        require([r['id'] for r in read(out/'inputs.json')]==[s for s in approved['registered'] if s not in held|failed],
                'First review escaped the accepted source population')
    require(sha(Path(cfg['source'])/'plan.json')==cfg['source_plan_sha256'], 'Source plan drift')
    if 'archived_review_plan' in cfg:
        from scripts.project_verifier_first_reviews import verify_files
        verify_files(out, cfg)
    return plan


def confirm(out):
    p=verify(out);ready=read(out/'disconnect_ready.json')
    require(c.corpus.runtime.process_identity(ready['pid'])==ready['process_start'] and
        time.time()-ready['unix']>=5 and ready['configuration_sha256']==p['configuration_sha256'], 'Fresh reconnect required')
    write(out/'disconnect_acceptance.json',dict(ready,passed=True,confirmed_unix=time.time()))


def request_messages(stage, payload, config):
    if stage == 'repair_experience' and review_fields.current(config):
        draft = review_fields.without_symbolic_hint(payload['draft'])
        payload = dict(payload, draft=review_fields.without_rule_metadata(draft) if review_fields.compact(config) else draft)
    messenger = old.Journal.__new__(old.Journal)
    messages = messenger.messages(stage, payload)
    if en.active(config):
        if stage == 'taxonomy':
            messages[0]['content'] = en.TAXONOMY
        elif stage in {'extract_experience', 'repair_experience'}:
            source = payload['original'] if stage == 'repair_experience' else payload
            fields = json.loads(messages[1]['content'])
            fields[en.EXAM_REFERENCE] = en.exam_answer(source['sample']['official_reference']['label'])
            messages[1]['content'] = json.dumps(fields, ensure_ascii=False, sort_keys=True).replace('<|', '\\u003c|')
            if stage == 'repair_experience':
                messages[-1]['content'] = en.AUDIT_REPAIR.format(error=payload['validation_error']).replace('<|', '\\u003c|')
    # The November frozen plans use the original first review, including in a repair turn.
    if stage in {'extract_experience', 'repair_experience'} and (en.active(config) or 'audit_system_prompt' in config):
        messages[0] = {'role': 'system', 'content': config['audit_system_prompt']}
    return messages


def call(out, journal, tokenizer, stage, key, payload):
    cfg=journal.config
    messages=request_messages(stage, payload, cfg)
    body=c.corpus.r.request_body(cfg['model'],messages,cfg['max_tokens'],backend='vllm')
    body.update(stream=False,return_token_ids=True,reasoning_effort='medium');body.pop('stream_options',None)
    require(len(json.dumps(body,ensure_ascii=False).encode())<=cfg['max_request_input_bytes'],'Request byte budget exceeded')
    raw,evidence=journal.call(stage,key,body)
    effective=validate_effective_input(raw,tokenizer,cfg['fixed_date'])
    if en.active(cfg) and stage in {'extract_experience', 'repair_experience'}:
        en.verify_visible_users(tokenizer.decode(raw['prompt_token_ids'], skip_special_tokens=False), messages)
    try:
        text = c.corpus.harmony.decode_response(raw, model=cfg['model'], max_tokens=cfg['max_tokens'],
                                                native_tools=None)['content'].strip()
    except ValueError as exc:
        if not en.active(cfg) or stage == 'taxonomy' or str(exc) not in {
            'Incomplete model output', 'No authoritative final/tool termination', 'Incomplete Harmony message', 'Missing final message'
        }:
            raise
        value, controls, positions = {'_content_failure': str(exc)}, [], []
    else:
        if text.startswith('```json\n') and text.endswith('```'):
            text = text[8:-3].strip()
        try:
            escaped, controls = old.escape_json_controls(text)
            value, _, positions = old.b.parse_json_escapes(escaped, repair=True)
            require(isinstance(value, dict), 'Expected JSON object')
        except (ValueError, TypeError):
            value, controls, positions = {'_unparsed_response_text': text}, [], []
    write(out/'decoded'/stage/(digest(key)+'.json'),dict(value=value,evidence=evidence,
        effective_input=effective,literal_control_positions=controls,literal_backslash_positions=positions))
    return value,evidence


def review_one(out, row, taxonomy, cfg, journal, tokenizer, *, saved_only=False):
    path = out/'experiences'/(row['id']+'.json')
    payload = dict(sample=row, taxonomy=taxonomy, max_rules_per_sample=cfg['max_rules_per_sample'])
    signature = digest([payload, cfg])
    if path.exists():
        saved = read(path)
        require(saved['id'] == row['id'] and saved['signature'] == signature, 'Saved experience drift')
        for evidence in saved['evidence']:
            require(sha(out/evidence['file']) == evidence['sha256'], 'Response drift')
        return saved
    require(not saved_only and not (out/'STOP').exists(), 'Review missing or stopped')
    draft, first = call(out, journal, tokenizer, 'extract_experience', row['id'], payload)
    evidence = [first]
    value, error = None, None
    for attempt in range(2):
        if '_content_failure' in draft:
            error = draft['_content_failure']
            break
        try:
            value = old.extraction.normalize_120b_experience(draft, **payload, legacy_symbolic=not review_fields.current(cfg))
            if review_fields.compact(cfg):
                value = review_fields.without_rule_metadata(value)
            error = None
            break
        except ValueError as exc:
            error = str(exc)
        if attempt == 0:
            draft, repair = call(out, journal, tokenizer, 'repair_experience', row['id'],
                                 dict(original=payload, draft=draft, validation_error=error))
            evidence.append(repair)
    saved = dict(id=row['id'], signature=signature, value=value, error=error, evidence=evidence, human_gold=False)
    write(path, saved)
    return saved


def serial_reviews(out, rows, taxonomy, cfg, journal, tokenizer):
    for row in rows:
        yield review_one(out, row, taxonomy, cfg, journal, tokenizer)


def run(out, *, wait_detached=None, reviews=serial_reviews):
    require(Path('/proc').exists(),'Server only')
    with c.corpus.r.run_lock(out/'active.lock'),c.corpus.r.run_lock(out.parent/'organization_service_18083.lock'):
        plan=verify(out);cfg=plan['configuration'];code=1
        require('archived_review_plan' not in cfg, 'Archived reviews are published offline')
        (out/'runner.pid').write_text(str(os.getpid()))
        try:
            (wait_detached or c.corpus.runtime.wait_detached)(out,plan['configuration_sha256'])
            require(not en.active(cfg) or 'build_scope_sha256' in cfg, 'New first reviews need a frozen source scope')
            for n,h in cfg['model_files'].items():require(sha(Path(cfg['model'])/n)==h,'Model configuration drift')
            for n,h in read(out/'source_evidence.json').items():require(sha(Path(n))==h,'Source evidence changed')
            with httpx.Client(trust_env=False,timeout=20) as client:
                response=client.get(cfg['auditor_api']+'/models');response.raise_for_status()
                require(any(x['id']==cfg['model'] and x.get('max_model_len',0)>=131072 for x in response.json()['data']),
                    'Unexpected model/context service')
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
            journal=c.corpus.Journal(out,cfg,threading.Event());taxonomy={'domains':[]}
            for domain in read(out/'topic_skeleton.json')['domains']:
                write(out/'execution_status.json',dict(state='taxonomy',domain=domain['id'],updated_unix=time.time()))
                value,_=call(out,journal,tokenizer,'taxonomy',domain['id'],{'domains':[domain]})
                old.extraction._experience_topics(value)
                require(old.flow.experience_topic_skeleton(value)=={'domains':[domain]},'Ontology changed')
                taxonomy['domains'].extend(value['domains'])
            write(out/'taxonomy.json',taxonomy)
            rows=read(out/'inputs.json');valid=[];invalid=[];counts=Counter()
            for i, saved in enumerate(reviews(out, rows, taxonomy, cfg, journal, tokenizer)):
                row = rows[i]
                require(saved['id'] == row['id'] and not (out/'STOP').exists(), 'Wrong order or STOP requested')
                if saved['value'] is None:invalid.append(dict(id=row['id'],error=saved['error']));counts['invalid_output']+=1
                else:valid.append(saved['value']);counts[saved['value']['semantic_audit']['status']]+=1
                write(out/'execution_status.json',dict(state='extracting',processed=i+1,total=len(rows),counts=dict(counts),updated_unix=time.time()))
                if i+1==cfg['pilot_size']:
                    write(out/'pilot_acceptance.json',dict(passed=bool(valid),scope='engineering_only_not_physical_quality',
                        configuration_sha256=plan['configuration_sha256'],counts=dict(counts),
                        files={str((out/'experiences'/(x['id']+'.json')).relative_to(out)):sha(out/'experiences'/(x['id']+'.json')) for x in rows[:i+1]}))
                    require(bool(valid),'All pilot outputs failed structure validation; inspect before continuing')
            require(len(valid)+len(invalid)==len(rows), 'Incomplete first-review population')
            write(out/'semantic_experience.json',valid);write(out/'invalid_outputs.json',invalid)
            distilled=old.extraction._build_distilled_library(valid,min_count=1,legacy_symbolic=not review_fields.current(cfg),
                                                              include_auxiliary=not review_fields.compact(cfg))
            write(out/'distilled.json',distilled)
            write(out/'execution_status.json',dict(state='extraction_complete',processed=len(rows),counts=dict(counts),
                rules=len(distilled['rules']),tree_complete=False,skills_complete=False,updated_unix=time.time()))
            code=0
        except BaseException as exc:
            write(out/'execution_status.json',dict(state='stopped',error=str(exc),updated_unix=time.time()));raise
        finally:
            (out/'runner.exit_code').write_text(str(code));(out/'runner.pid').unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','verify','run','confirm-detached'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--dataset', type=Path, help='Original combined_language_only.json for prepare')
    p.add_argument('--results-root', type=Path, help='Frozen benchmark and source-isolation archive')
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='prepare':prepare(out,a.source.resolve(),a.dataset.resolve() if a.dataset else None,
                                 results_root=a.results_root.resolve() if a.results_root else None)
    elif a.mode=='verify':verify(out)
    elif a.mode=='confirm-detached':confirm(out)
    else:run(out)


if __name__=='__main__':main()
