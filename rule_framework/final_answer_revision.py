"""Final-answer benchmark revision and explicit v15-style checking contract.

The archived final-answer protocol remains immutable.  This module changes
the delivery mechanism, not the binary target: incorrect is the positive class.
"""
import json

from rule_framework import final_answer_protocol as original
from rule_framework import flat_skill_rag as memory
from rule_framework import verifier_english as en

VERSION = 'final_answer_review_and_aligned_delivery_v1'
POSITIVE_CLASS = 'incorrect'
CONFUSION = {
    ('incorrect', 'incorrect'): 'TP',
    ('incorrect', 'correct'): 'FN',
    ('correct', 'correct'): 'TN',
    ('correct', 'incorrect'): 'FP',
}

REVIEW_PROMPT = original.COMMON + '''
You are auditing a machine benchmark, independently of every tested verifier.
You receive only the original question, one 30B answer, and original references.
No existing correctness label, verifier output, or knowledge-library item is available.

Check whether the VISIBLE question supplies the conditions needed for its requested
answer. A figure or an earlier part is not missing merely because it is mentioned:
identify the specific indispensable information that cannot be recovered from the text.
Do not invent a diagram, topology, previous result, definition, or numerical datum.
Do not mistake multiple equivalent reference forms for separate required subquestions.

Check the reference against the visible question using a concise physical or numerical
check: units, limiting cases, signs, requested quantity, and stated approximations.
Reference agreement alone does not prove correctness. If the reference conflicts with
the question, report the conflict and verdict=uncertain; never silently fix the reference.
Compare all committed final outputs, including conflicting final statements in a proof.
Accept equivalent expressions and units. Numerical tolerance must follow the problem's
precision or a justified approximation; do not accept an arbitrary percentage discrepancy
as rounding. Ignore abandoned intermediate mistakes when the final answer is correct.

Return one JSON object only, with exactly these keys:
{"question_status":"complete|missing_conditions|ambiguous",
 "reference_status":"consistent|conflicting|insufficient|uncertain",
 "verdict":"correct|incorrect|uncertain",
 "student_final_answer":"the committed final answer",
 "condition_check":"brief evidence for completeness or the specific missing condition",
 "reference_check":"brief reproducible physical/numerical check or the precise conflict",
 "answer_check":"brief evidence comparing the requested final outputs"}.
If question_status is not complete or reference_status is not consistent, verdict must
be uncertain. This is a machine review, not certified human ground truth.
'''

STAGED_PROMPT = original.VERIFY_PROMPT.replace("through your group's tools", "as Reference material").replace(original.OUTPUT, '''
In PREPARE return only {"queries":["short bilingual lookup phrase"]}; do not publish a
verdict. In CHECK return the final-answer JSON specified below.
''') + '''
Any supplied Reference material is fallible historical data. Read the complete supplied
items, distinguish the old answer from its machine review, and check applicability to the
CURRENT question and answer in the first user message. Continue checking the complete
current answer, including aspects outside the supplied material. No independent checker
or separate review of another agent's diagnoses is used.
''' + original.OUTPUT

RAG_BINDING = '''Only judge the current student answer: the `prediction` field in the first
user message. Every item under Reference material, including its problem, original 30B
answer and machine check note, is historical data. Do not transfer a historical error or
off-topic statement to the current answer. Use the material only when its conditions and
objects apply. Return the existing final-answer verdict JSON for the current answer.'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def public(row):
    require(isinstance(row.get('id'), str) and row['id'], 'Missing sample ID')
    require(all(isinstance(row.get(k), str) and row[k].strip()
                for k in ('question', 'prediction')), 'Incomplete public input')
    # The frozen original final task had no extra context. Never expose private
    # references or source metadata through the v15 payload's context field.
    require(not row.get('context'), 'Unexpected additional context; review before use')
    return {k: row[k] for k in ('id', 'question', 'prediction')} | {'context': ''}


def review_messages(row, references):
    row = public(row)
    require(isinstance(references, list) and references and
            all(isinstance(v, str) and v.strip() for v in references), 'Missing reference')
    payload = dict(question=row['question'], student_answer=row['prediction'],
                   original_reference_answers=references)
    return [dict(role='system', content=REVIEW_PROMPT),
            dict(role='user', content=json.dumps(payload, ensure_ascii=False).replace('<|', '\\u003c|'))]


def validate_review(value):
    expected = {'question_status', 'reference_status', 'verdict', 'student_final_answer',
                'condition_check', 'reference_check', 'answer_check'}
    require(isinstance(value, dict) and set(value) == expected, 'Invalid review schema')
    require(value['question_status'] in {'complete', 'missing_conditions', 'ambiguous'},
            'Invalid question status')
    require(value['reference_status'] in {'consistent', 'conflicting', 'insufficient', 'uncertain'},
            'Invalid reference status')
    require(value['verdict'] in original.VERDICTS, 'Invalid reviewed verdict')
    require(all(isinstance(value[k], str) and value[k].strip()
                for k in ('condition_check', 'reference_check', 'answer_check')),
            'Review lacks explicit evidence')
    require(isinstance(value['student_final_answer'], str), 'Invalid final answer')
    if value['verdict'] == 'correct':
        require(value['student_final_answer'].strip(), 'Correct answer not identified')
    if value['question_status'] != 'complete' or value['reference_status'] != 'consistent':
        require(value['verdict'] == 'uncertain', 'Unsafe definite label for unresolved input')
    return value


def staged_prompt(english):
    return en.FINAL_CHECK if english else STAGED_PROMPT


def direct_messages(row):
    return en.direct_messages(public(row), en.FINAL_DIRECT)


def query_messages(row, *, english=False):
    if english:
        return memory.prepare_messages(public(row), en.FINAL_CHECK, en.FINAL_QUERY)
    result = memory.query_messages(public(row))
    result[0]['content'] = STAGED_PROMPT
    return result


def continuation_messages(row, queries, references, *, arm, english=False):
    require(memory.uses_preparation(arm, english=english), 'English pure checks must use direct_messages')
    require(arm != 'pure' or not references, 'Pure arm received reference material')
    memory.validate_queries(queries, english=english)
    result = query_messages(row, english=english)
    if english and arm == 'case_rag':
        references = [en.case_view(item) for item in references]
    result.append(dict(role='assistant', content=json.dumps(queries, ensure_ascii=False)))
    fields = {'stage': 'CHECK', 'Reference material': references}
    content = json.dumps(fields, ensure_ascii=False).replace('<|', '\\u003c|')
    if english:
        content += '\nJudge the committed final answer in the first user message.'
        if arm == 'case_rag':
            content += ' ' + en.CASE_BINDING
    else:
        content += '\nJudge the final-answer correctness of the original current answer. Submit final-answer verdict JSON.'
        if arm == 'case_rag':
            content += '\n\n' + RAG_BINDING
    result.append(dict(role='user', content=content))
    return result


def reference_packet(row, arm, queries, store, count_tokens, *, english=False):
    # Reuse the exact lexical ranking, Top2 tree routing, no-flat-fallback,
    # complete-item admission and 8192 material-token cap from v15.
    def continue_for_arm(r, q, refs):
        return continuation_messages(r, q, refs, arm=arm, english=english)
    return memory.reference_packet(public(row), arm, queries, store, count_tokens,
                                   english=english, render=continue_for_arm)


def decision(candidate, reviewed, override=None):
    """Do not silently adopt a new machine disagreement or convert unknown to wrong."""
    sid = candidate['id']
    if override:
        require(override['id'] == sid and override['human_gold'] is False and
                isinstance(override['basis'], str) and override['basis'].strip(), 'Invalid assistant adjudication')
        action = override['action']
        require(action in {'exclude', 'set_label'}, 'Invalid adjudication action')
        if action == 'set_label':
            require(not candidate.get('eligibility_failure'), 'Cannot label incomplete generation/reference')
            require(override.get('verdict') in {'correct', 'incorrect'}, 'Nonbinary adjudicated label')
            return dict(id=sid, state='eligible', verdict=override['verdict'],
                        authority='assistant_adjudication', basis=override['basis'], human_gold=False)
        return dict(id=sid, state='excluded', reason=override['basis'],
                    authority='assistant_adjudication', human_gold=False)
    if candidate.get('eligibility_failure'):
        return dict(id=sid, state='excluded', reason=candidate['eligibility_failure'], human_gold=False)
    if reviewed is None:
        return dict(id=sid, state='pending_review', human_gold=False)
    if reviewed['status'] != 'complete':
        return dict(id=sid, state='review_failure', reason=reviewed.get('reason'), human_gold=False)
    value = validate_review(reviewed['value'])
    if value['verdict'] == 'uncertain':
        return dict(id=sid, state='pending_adjudication', reason='review_uncertain',
                    review=value, human_gold=False)
    prior = candidate.get('old_verdict')
    if prior and prior != value['verdict']:
        return dict(id=sid, state='pending_adjudication', reason='independent_review_disagrees',
                    old_verdict=prior, review=value, human_gold=False)
    return dict(id=sid, state='eligible', verdict=value['verdict'],
                authority='reference_assisted_120b_review', human_gold=False)


def select_balanced(candidates, decisions, quota=300):
    require(type(quota) is int and quota > 0, 'Invalid quota')
    ids = [r['id'] for r in candidates]
    require(len(ids) == len(set(ids)) and set(ids) == set(decisions), 'Cohort/decision mismatch')
    require(not any(d['state'] in {'pending_review', 'pending_adjudication'}
                    for d in decisions.values()), 'Review/adjudication incomplete')
    selected, counts = [], {'correct': 0, 'incorrect': 0}
    for row in candidates:  # the original frozen source order, never a score order
        d = decisions[row['id']]
        if d['state'] == 'eligible' and counts[d['verdict']] < quota:
            selected.append(public(row)); counts[d['verdict']] += 1
    require(counts == dict(correct=quota, incorrect=quota),
            'Cannot fill frozen balanced quota; do not shrink or sample new sources automatically')
    return selected


def contract(*, english=False):
    return dict(protocol=VERSION, task='final_answer_correctness', positive_class=POSITIVE_CLASS,
        confusion=[dict(label=k[0], prediction=k[1], cell=v) for k, v in CONFUSION.items()],
        review_prompt=REVIEW_PROMPT, staged_prompt=staged_prompt(english), rag_binding=en.CASE_BINDING if english else RAG_BINDING,
        max_items=memory.MAX_ITEMS, reference_token_cap=memory.REFERENCE_TOKEN_CAP,
        query_max_tokens=memory.QUERY_MAX_TOKENS, tree_branches=memory.TREE_BRANCHES,
        tools=None, flat_fallback=False, independent_checker=False,
        failures='Retain content failures without resampling; authentication/identity/unknown transport stops.',
        labels='120B reference-assisted machine labels with explicit assistant adjudication; human_gold=false',
        **(dict(prompt_profile=en.PROFILE, pure_prompt=en.FINAL_DIRECT, prepare_instruction=en.FINAL_QUERY,
                prepare_arms=memory.prepared_arms(english=True)) if english else {}))
