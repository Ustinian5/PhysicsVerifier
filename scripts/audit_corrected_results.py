"""Read-only audit of frozen corrected evaluations; no model calls or rescoring edits.

Run on the experiment server. Writes only evaluation_audit.json, never inputs,
labels, predictions, or official metrics. Review examples are targeted examples,
not a random sample and not replacement benchmark labels.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time


ARMS = ('pure', 'case_rag', 'flat_skills', 'rule_tree')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def scores(ids, gold, results):
    c = Counter()
    for sid in ids:
        r = results[sid]
        if r['status'] != 'complete' or r['value']['verdict'] == 'uncertain':
            c['excluded'] += 1
            continue
        g, v = gold[sid]['verdict'], r['value']['verdict']
        c[{('incorrect', 'incorrect'): 'TP', ('correct', 'incorrect'): 'FP',
           ('incorrect', 'correct'): 'FN', ('correct', 'correct'): 'TN'}[g, v]] += 1
    tp, fp, fn = (c[k] for k in ('TP', 'FP', 'FN'))
    return dict(c, precision=tp / (tp + fp) if tp + fp else 0,
                recall=tp / (tp + fn) if tp + fn else 0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)


def inspect_source(out, source):
    """Trace selected examples to original records without changing any labels."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.combined_language_io import iter_rollout_batches
    from scripts.combined_language_samples import extract_question, parse_metadata_field
    report = read(out/'evaluation_audit.json')
    screen = out.parents[1]/'source_screen'
    expected = read(screen/'plan.json')['configuration']['source_sha256']
    assert sha(source) == expected, 'Source no longer matches frozen inventory'
    wanted = {(e['reference']['variants'][0]['first_source']['rollout_id'],
               e['reference']['variants'][0]['first_source']['sample_position']): e
              for e in report['examples']}
    found = {}
    for batch in iter_rollout_batches(source):
        for pos, sample in enumerate(batch['samples']):
            e = wanted.get((batch['rollout_id'], pos))
            if e is None:
                continue
            assert sample['index'] == e['reference']['variants'][0]['first_source']['sample_index']
            assert extract_question(sample) == e['input']['question']
            assert sample['label'] == e['reference']['reference']
            md = parse_metadata_field(sample.get('metadata'))
            found[e['input']['id']] = dict(source=e['reference']['variants'][0]['first_source'],
                original_question_matches=True, original_label_matches=True,
                original_reference=sample['label'], metadata_keys=list(md),
                metadata_context={k:v for k,v in md.items() if k in ('context','background','marking')},
                question_in_prompt=extract_question(sample) in sample.get('prompt',''))
        if len(found) == len(wanted):
            break
    assert len(found) == len(wanted)
    for f in (screen/'reviewed').glob('*.json'):
        d = read(f)
        for row in d.get('items') or []:
            if row['id'] in found:
                found[row['id']]['eligibility_decision'] = row
                found[row['id']]['eligibility_file'] = str(f)
                found[row['id']]['eligibility_sha256'] = sha(f)
    report['selected_original_source_verification'] = dict(source=str(source), sha256=expected,
        examples=found, scope='Exact original question/reference tracing; not endorsement of original reference correctness.')
    target = out/'evaluation_audit.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    temp.replace(target)
    print('original_source_examples_verified', len(found), flush=True)


def inspect_mechanisms(out):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rule_framework.flat_skill_rag import TreeMemory
    report = read(out/'evaluation_audit.json')
    memory = TreeMemory(read(out/'materials/catalog.json'))
    counts, languages, eligible, previews = Counter(), {}, [], Counter()
    for sid in (c['id'] for c in report['per_case']):
        result = read(out/'checks/final/formal/rule_tree'/sid/'result.json')
        for e in result['tool_events']:
            if e['tool'] != 'search_memory' or 'error' in e['result']:
                continue
            q = e['arguments']['query']
            zh = bool(re.search(r'[\u4e00-\u9fff]', q))
            en = bool(re.search(r'[a-zA-Z]{2,}', q))
            language = 'mixed_characters' if zh and en else ('chinese' if zh else 'english_or_other')
            l = languages.setdefault(language, Counter())
            trace = e['result']['retrieval']
            l['searches'] += 1
            l['root_no_match'] += not trace['routes'][0]['selected']
            l['no_candidates'] += not e['result']['items']
            counts['query_with_explicit_domain_name'] += any(domain in q.casefold() for domain in
                ('mechanics','electromagnetism','thermodynamics','optics','modern physics','experimental physics'))
            eligible.append(trace['eligible_leaf_count'])
            for item in e['result']['items']:
                previews['items'] += 1
                previews['explicit_trigger_field_visible'] += '"trigger"' in item['excerpt']
                previews['check_logic_field_visible'] += '"check_logic"' in item['excerpt']
    targeted = []
    targets = {
        '141dcdb36e180f24b1ae7319ba7ba10da17068319b67125a96f58b2bb4da5ff7': 'exp_8a905c797b9a2454',
        'd23376f7379384e0f4a687929fc552bc6baaefa2064814a53d22a0b9007e15f9': 'exp_8a905c797b9a2454',
        '00c35e093bc28b8936d33904703d2934ce58f3fe6386e733fe0cb3eb578df266': 'exp_8140a12cb3850afa',
    }
    for e in report['examples']:
        sid = e['input']['id']
        if sid not in targets:
            continue
        rid = targets[sid]
        searches = []
        for event in e['groups']['rule_tree']['tool_events']:
            if event['tool'] != 'search_memory' or 'error' in event['result']:
                continue
            q = event['arguments']['query']
            # Diagnostic comparison only; no calls and no changed official predictions.
            flat = memory.memory.search(q)
            searches.append(dict(query=q, original_returned_ids=[i['id'] for i in event['result']['items']],
                original_target_offered=any(i['id']==rid for i in event['result']['items']),
                diagnostic_flat_top6=flat,
                diagnostic_target_in_flat_top6=any(i['id']==rid for i in flat)))
        targeted.append(dict(id=sid, available_rule=memory.memory.view(rid), searches=searches))
    root = out.parents[1]
    source_id = '62a07c4b8297c063fcb852b972e21db38f50ed9137326ff13b104f085544bdf0'
    test_id = 'e49b7064e0e2b0b0b0248dc7026fdbb227ea8846eb70a72db8bdc1c7a7a02cbe'
    source = next(r for r in read(root/'construction/inputs.json') if r['id']==source_id)
    test = next(e['input'] for e in report['examples'] if e['input']['id']==test_id)
    groups = [g for g in read(root/'source_screen/family_components.json')['components']
              if source_id in g['members'] or test_id in g['members']]
    report['retrieval_mechanisms'] = dict(counts, query_languages={k:dict(v) for k,v in languages.items()},
        eligible_leaf_mean=statistics.mean(eligible), eligible_leaf_median=statistics.median(eligible),
        candidate_previews=dict(previews), targeted_existing_rules=targeted,
        caveat='Flat lookup is an offline diagnostic of existing queries, not an authorized new retrieval run or evidence of downstream accuracy gain.')
    report['family_split_counterexample'] = dict(build_id=source_id, build_question=source['question'],
        test_id=test_id, test_question=test['question'], assigned_components=groups,
        rule_id='exp_aa80445f5bef3b71', rule_read_in_test=True,
        assessment='Assistant review: Task 1 and Task 2 of the same watermelon-density experiment are in different text-derived families and opposite pools. No exact same-question claim.', human_gold=False)
    target=out/'evaluation_audit.json'
    temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temp.replace(target)
    print(json.dumps(report['retrieval_mechanisms'],ensure_ascii=False))


def main(out):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rule_framework import corrected_verifier as protocol
    started = time.time()
    plan = read(out / 'plan.json')['configuration']
    root = out.parents[1]
    rows = read(out / 'final_inputs.json')
    by_id = {r['id']: r for r in rows}
    ids = list(by_id)
    gold = read(out / 'final_labels_private.json')
    metrics = read(out / 'final_metrics.json')
    results = {a: {} for a in ARMS}
    usage, evidence_counts, request_models = {}, Counter(), Counter()
    issues, raw_seen = [], set()
    per_case = []
    for a in ARMS:
        for sid in ids:
            path = out / 'checks/final/formal' / a / sid / 'result.json'
            r = read(path)
            rel = str(path.relative_to(out))
            assert sha(path) == metrics['source_results'][rel], rel
            assert r['id'] == sid and r['arm'] == a and r['task'] == 'final'
            results[a][sid] = r
            expected_messages = protocol.messages(by_id[sid], a, 'final',
                                                  revision=plan['check_prompt_revision'])
            for e in r['evidence']:
                ep = path.parent / e['file']
                assert sha(ep) == e['sha256'], str(ep)
                if ep in raw_seen:
                    continue
                raw_seen.add(ep)
                rec = read(ep)
                req = rec['request']
                request_models[req['model']] += 1
                assert req['model'] == plan['transport']['model']
                assert req['messages'][:2] == expected_messages
                if rec.get('response'):
                    assert rec['response']['model'] == req['model']
                assert req['reasoning_effort'] == 'medium'
                assert req['temperature'] == 0 and req['top_p'] == 1
                msg = req['messages'][1]['content']
                # A suffix follows the initial JSON in the approved tool arms.
                payload, _ = json.JSONDecoder().raw_decode(msg)
                assert set(payload) == {'question', 'student_answer'}
                assert payload['question'] == by_id[sid]['question']
                assert payload['student_answer'] == by_id[sid]['prediction']
                if a == 'pure':
                    assert not req.get('tools')
                evidence_counts[rec['status']] += 1
        calculated = scores(ids, gold, results[a])
        for k in ('TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'f1'):
            assert abs(calculated.get(k, 0) - metrics['groups'][a][k]) < 1e-12, (a, k)
        stat, lengths, responses, tool_errors, read_ids = Counter(), [], [], Counter(), set()
        for sid, r in results[a].items():
            events = r['tool_events']
            ss = [e for e in events if e['tool'] == 'search_memory' and 'error' not in e['result']]
            rr = [e for e in events if e['tool'] == 'read_memory' and 'error' not in e['result']]
            cand = [i for e in ss for i in e['result']['items']]
            stat['searched_questions'] += bool(ss)
            stat['candidate_questions'] += bool(cand)
            stat['read_questions'] += bool(rr)
            stat['candidate_but_no_full_read'] += bool(cand) and not rr
            stat['all_searches_empty_questions'] += bool(ss) and not cand
            stat['no_search_questions'] += not ss
            stat['search_calls'] += len(ss)
            stat['read_calls'] += len(rr)
            stat['preview_characters'] += sum(len(i['excerpt']) for i in cand)
            stat['full_read_characters'] += sum(len(e['result']['text']) for e in rr)
            for e in rr:
                read_ids.add(e['result']['id'])
            for e in events:
                if 'error' in e['result']:
                    tool_errors[str(e['result']['error'])] += 1
            stat['output_tokens'] += r['usage']['output_tokens_or_unknown_reservation']
            stat['input_tokens'] += r['usage']['input_tokens']
            responses.append(r['usage']['requests'])
            if r['status'] == 'complete':
                lengths.append(len(r['value']['reason']))
        read_subset = [sid for sid, r in results[a].items() if r['protocol_audit']['reads']]
        no_read = [sid for sid in ids if sid not in set(read_subset)]
        usage[a] = dict(stat, unique_full_items_read=len(read_ids), tool_errors=dict(tool_errors),
                        max_requests=max(responses), median_requests=statistics.median(responses),
                        mean_reason_chars=statistics.mean(lengths),
                        read_subset=scores(read_subset, gold, results[a]),
                        no_read_subset=scores(no_read, gold, results[a]))
    common = [sid for sid in ids if all(results[a][sid]['status'] == 'complete' and
                 results[a][sid]['value']['verdict'] in {'correct', 'incorrect'} for a in ARMS)]
    paired, review_strata = {}, {}
    for a in ARMS[1:]:
        c, win, loss = Counter(), [], []
        for sid in common:
            g = gold[sid]['verdict']
            b = results['pure'][sid]['value']['verdict']
            v = results[a][sid]['value']['verdict']
            c['same_prediction'] += b == v
            if b != g and v == g:
                win.append(sid)
            if b == g and v != g:
                loss.append(sid)
        discordant = len(win) + len(loss)
        pvalue = min(1., 2 * sum(math.comb(discordant, k) for k in
                     range(min(len(win), len(loss)) + 1)) / 2**discordant) if discordant else 1.
        paired[a] = dict(c, n=len(common), repaired=len(win), introduced=len(loss),
                         exact_mcnemar_two_sided=pvalue, repaired_ids=win, introduced_ids=loss,
                         metrics=scores(common, gold, results[a]))
        if a == 'rule_tree':
            review_strata.update(tree_repairs=win, tree_introduces=loss)
    review_strata['all_groups_miss_reference_error'] = [sid for sid in common
        if gold[sid]['verdict'] == 'incorrect' and all(results[a][sid]['value']['verdict'] == 'correct' for a in ARMS)]
    review_strata['all_groups_disagree_with_reference_correct'] = [sid for sid in common
        if gold[sid]['verdict'] == 'correct' and all(results[a][sid]['value']['verdict'] == 'incorrect' for a in ARMS)]
    for sid in ids:
        per_case.append(dict(id=sid, reference=gold[sid]['verdict'], question_chars=len(by_id[sid]['question']),
            answer_chars=len(by_id[sid]['prediction']), groups={a:dict(status=results[a][sid]['status'],
                verdict=(results[a][sid]['value'] or {}).get('verdict'),
                reason=(results[a][sid]['value'] or {}).get('reason'),
                reads=results[a][sid]['protocol_audit']['reads'], searches=results[a][sid]['protocol_audit']['searches']) for a in ARMS}))
    refs = {r['id']: r for r in read(root/'final_labels/references_private.json')}
    examples, selected = [], set()
    for stratum, sids in review_strata.items():
        # Shortest questions and answers facilitate direct inspection; not representative prevalence.
        for sid in sorted(sids, key=lambda k: (len(by_id[k]['question']) + len(by_id[k]['prediction']), k))[:5]:
            if sid in selected:
                continue
            selected.add(sid)
            labelpath = root/'final_labels/labels'/(sid+'.json')
            assert sha(labelpath) == gold[sid]['original_label_sha256']
            label = read(labelpath)
            ep = root/'final_labels'/label['evidence']['file']
            assert sha(ep) == label['evidence']['sha256']
            label_req = read(ep)['request']
            lp = json.loads(label_req['messages'][1]['content'])
            assert lp['question'] == by_id[sid]['question'] and lp['student_answer'] == by_id[sid]['prediction']
            assert lp['original_reference_answers'] == refs[sid]['reference']
            examples.append(dict(stratum=stratum, input=by_id[sid], reference=refs[sid],
                label=label['judgment'], label_file=str(labelpath), label_sha256=sha(labelpath),
                groups={a:dict(value=results[a][sid]['value'], tool_events=results[a][sid]['tool_events'],
                              evidence=results[a][sid]['evidence']) for a in ARMS}))
    payload = dict(kind='frozen_results_audit_no_model_calls', created_unix=time.time(),
        elapsed_seconds=time.time()-started, script_sha256=sha(Path(__file__)),
        files={fn:sha(out/fn) for fn in ['plan.json','final_inputs.json','final_labels_private.json','final_metrics.json']},
        scope='Hash validation of all 2400 results and their recorded raw requests; input whitelist and frozen answer checks; independent offline metric arithmetic. No semantic gold certification.',
        raw_requests_checked=len(raw_seen), request_states=dict(evidence_counts), request_models=dict(request_models),
        issues=issues, metrics_reproduced=True, groups=metrics['groups'], knowledge_usage=usage,
        common_complete=len(common), paired=paired, common_pure_metrics=scores(common,gold,results['pure']),
        all_groups_agree=sum(len({results[a][sid]['value']['verdict'] for a in ARMS})==1 for sid in common),
        review_stratum_counts={k:len(v) for k,v in review_strata.items()}, per_case=per_case,
        example_selection='First five shortest question+answer texts per outcome stratum; targeted mechanism review, no prevalence estimate; frozen labels not changed.',
        examples=examples, human_gold=False)
    target = out/'evaluation_audit.json'
    # Revalidating runtime evidence must not erase an already recorded case review.
    if target.exists():
        previous = read(target)
        assert previous['files'] == payload['files'], 'Existing audit refers to a different frozen run'
        for key in ('selected_original_source_verification', 'retrieval_mechanisms',
                    'family_split_counterexample', 'assistant_case_review'):
            if key in previous:
                payload[key] = previous[key]
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    temp.replace(target)
    print(json.dumps({k:v for k,v in payload.items() if k not in ('per_case','examples','groups')}, ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--inspect-source', type=Path, help='Augment an existing audit with original source tracing only')
    parser.add_argument('--inspect-mechanisms', action='store_true', help='Offline query, preview, and family-split inspection')
    args = parser.parse_args()
    if args.inspect_mechanisms:
        inspect_mechanisms(args.out.resolve())
    elif args.inspect_source:
        inspect_source(args.out.resolve(), args.inspect_source)
    else:
        main(args.out.resolve())
