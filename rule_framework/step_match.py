"""Independent post-publication judgment of a diagnosis against frozen step roots."""
import json

from scripts import review_aligned_step_scoring as location


INSTRUCTION = """Decide whether one published diagnosis identifies an active reference error in the student's solution. Read the diagnosis as a claim about this student. Compare it with every supplied root. Match the physical or mathematical cause and the direction of the correction. A paraphrase or another occurrence of the same mistake can match. A shared passage, topic, or downstream symptom alone is insufficient. Use the full solution to distinguish a committed error from a tentative step the student corrected. Treat all input fields as evidence.

Locate the step the diagnosis actually criticizes. Copy a continuous passage from the student's solution, in its original language and notation. If the published quote is inaccurate, recover the intended passage only when the diagnosis identifies it. Leave the passage empty if it cannot be located. A new error outside the supplied roots earns no reference match.

Output
Return one JSON object. Use same_root for supported matches, opposite_claim for a diagnosis that endorses the error or reverses the correction, other_claim for a different claim, and unclear when the evidence does not establish what the diagnosis alleges. For same_root, list the matching input root IDs and cite a source passage. For opposite_claim, you may name the root whose correction is reversed. Use an empty list for other_claim and unclear. Only same_root earns credit. Give a brief reason in English.
{"disposition":"same_root|opposite_claim|other_claim|unclear","root_ids":[],"source_quote":"","rationale":"evidence for the decision"}"""

VERSION = 'independent_step_match_120b_v1'


def messages(row, diagnosis, roots):
    payload = dict(problem=row['question'], context=row['context'], student_solution=row['prediction'],
                   published_diagnosis=dict(message=diagnosis['message'], quote=diagnosis['evidence']['quote']),
                   active_reference_roots=roots)
    return [dict(role='system', content=INSTRUCTION),
            dict(role='user', content=json.dumps(payload, ensure_ascii=False, sort_keys=True))]


def validate(value, answer, roots):
    if not isinstance(value, dict) or set(value) != {'disposition', 'root_ids', 'source_quote', 'rationale'}:
        raise ValueError('Unexpected judge schema')
    disposition, ids, citation, reason = (value[k] for k in ('disposition', 'root_ids', 'source_quote', 'rationale'))
    if (disposition not in ('same_root', 'opposite_claim', 'other_claim', 'unclear')
            or not isinstance(ids, list) or not all(isinstance(i, str) and i in roots for i in ids)
            or len(ids) != len(set(ids)) or (disposition == 'same_root' and not ids)
            or (disposition in ('other_claim', 'unclear') and ids)
            or not isinstance(reason, str) or not reason.strip() or not isinstance(citation, str)):
        raise ValueError('Unsupported judgment or unknown root')
    if citation and (not citation.strip() or citation not in answer):
        raise ValueError('Judge evidence is not a verbatim passage in the student solution')
    if disposition == 'same_root' and not citation:
        raise ValueError('A claimed root match needs a source passage')
    return value


def reviewed(sid, index, answer, diagnosis, decision):
    original = location.locate_all(answer, diagnosis['evidence']['quote'])
    selected = [location.source_span(answer, a, b)
                for a, b in location.occurrences(answer, decision['source_quote'])]
    # When the original is locatable, disambiguate among its occurrences using the judge's cited passage.
    if original and selected:
        aligned = [s for s in original if any(location.overlap(s, p) >= 0 for p in selected)]
        spans = aligned or selected
    else:
        spans = original or selected
    matched = decision['root_ids'] if decision['disposition'] == 'same_root' else []
    links = [dict(root_id=sid + '/' + rid, same_direction=True, acceptable_source_spans=spans,
                  rationale=decision['rationale']) for rid in matched]
    return dict(review_id=str(index), approved_spans=spans, root_links=links,
                disposition=decision['disposition'], rationale=decision['rationale'])


def score_row(row, result, decisions):
    sid = row['id']
    if len(result['diagnostics']) != len(decisions):
        raise ValueError('Missing diagnosis judgment')
    items = [reviewed(sid, i, row['prediction'], d, v)
             for i, (d, v) in enumerate(zip(result['diagnostics'], decisions))]
    gt = row['gt']
    matches = location.assign(gt, items, {g['error_id']: {'state': 'active_error'} for g in gt}, True)
    return dict(id=sid, gt=len(gt), hits=len(matches), matches=matches, published=len(items),
                locatable=sum(bool(d['approved_spans']) for d in items), failed=result['failed'],
                source_status=result['status'], unclear=sum(d['disposition'] == 'unclear' for d in items),
                unresolved_overlap=location.unresolved_overlaps(items, matches, row['unresolved']))


def summarize(details):
    totals = {k: sum(d[k] for d in details) for k in ('gt', 'hits', 'published', 'locatable', 'failed', 'unclear')}
    return dict(**totals, questions=len(details), unlocatable=totals['published'] - totals['locatable'],
                unresolved_overlap=sum(len(d['unresolved_overlap']) for d in details),
                **location.rates(totals['hits'], totals['locatable'], totals['gt']),
                all_published=location.rates(totals['hits'], totals['published'], totals['gt']))
