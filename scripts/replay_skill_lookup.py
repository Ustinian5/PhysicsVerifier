"""Replay proposed SEARCH/SELECT actions on a frozen development answer without running a model."""
import argparse
import json
import unicodedata
from pathlib import Path

from rule_framework import final_answer_revision as protocol
from rule_framework import shared_experience
from rule_framework import verified_resume as vr

PROMPT = (
    'Choose historical skills for the current answer. Search using {"search":[{"text":[["alias","synonym"]]}]}. '
    'Within one clause, require one alternative per group; clauses are alternatives. '
    'Use exact instead of text for case-sensitive units or literal formulas. '
    'If the count is zero, switch from the problem object to a governing relation. If broad, add a condition. '
    'After at most two searches, return {"select":[IDs]} with at most six displayed IDs or an empty list. '
    'Do not issue a verdict until CHECK.'
)
DEMO_ID = 'cl_202_125689'
DEMO = [
    {'search': [{'text': [['fiber', 'fibre', '光纤', '光纖'], ['graded index', 'graded-index', 'graded‑index',
                    '梯度折射率', '渐变折射率', 'numerical aperture', '数值孔径', 'acceptance angle', '接收角']]}]},
    {'search': [{'text': [['snell', '斯涅尔', '斯耐尔', '折射定律', 'total internal reflection', '全反射',
                            'optical invariant', '光学不变量']]}]},
    {'select': ['S604']},
]


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).replace('<|', '\\u003c|')


def fold(text):
    return unicodedata.normalize('NFKC', text).casefold()


def search(docs, clauses):
    protocol.require(isinstance(clauses, list) and 1 <= len(clauses) <= 3, 'Expected 1–3 clauses')
    counts, fields = [], {}
    for clause in clauses:
        protocol.require(isinstance(clause, dict) and clause and set(clause) <= {'text', 'exact'}, 'Invalid clause')
        groups = []
        for mode, terms in clause.items():
            protocol.require(isinstance(terms, list) and terms, 'Expected groups of alternatives')
            for alternatives in terms:
                protocol.require(isinstance(alternatives, list) and alternatives and
                                 all(isinstance(t, str) and t.strip() for t in alternatives), 'Invalid alternatives')
                groups.append((mode, [fold(t) if mode == 'text' else t for t in alternatives]))
        count = 0
        for key, doc in docs.items():
            raw = {name: doc[name] for name in ('title', 'text')}
            folded = {name: fold(value) for name, value in raw.items()}
            matched = set()
            for mode, alternatives in groups:
                values = folded if mode == 'text' else raw
                hits = {name for name, value in values.items() if any(term in value for term in alternatives)}
                if not hits:
                    break
                matched.update(hits)
            else:
                count += 1
                fields.setdefault(key, set()).update(matched)
        counts.append(count)
    cards = [{'id': k, 'title': docs[k]['title'], 'fields': sorted(fields[k])} for k in docs if k in fields]
    return {'counts': counts, 'total': len(cards), 'cards': cards if len(cards) <= 24 else []}


def replay(row, queries, skills, actions, count, config):
    limits = config['limits']
    cap = config['contract']['reference_token_cap']
    max_items = config['contract']['max_items']
    context = limits['context_window'] - limits['output_tokens_per_sample']
    docs = {f'S{i}': r for i, r in enumerate(sorted(skills, key=lambda x: x['id']), 1)}
    protocol.require(isinstance(actions, list) and 1 <= len(actions) <= 3, 'Expected up to two searches then SELECT')
    initial = protocol.continuation_messages(row, queries, [], arm='flat_skills')
    messages = initial[:-1] + [dict(role='user', content=PROMPT)]
    displayed, trace, preview_tokens = set(), [], 0
    for action in actions[:-1]:
        protocol.require(isinstance(action, dict) and set(action) == {'search'}, 'Expected SEARCH')
        result = search(docs, action['search'])
        result['searches_left'] = 1 - len(trace)
        messages.append(dict(role='assistant', content=wire(action)))
        blank = dict(result, cards=[])
        before = count(messages + [dict(role='user', content=wire(blank))])
        after = count(messages + [dict(role='user', content=wire(result))])
        cost = max(0, after - before)
        if preview_tokens + cost > cap or after > context:
            result, cost = blank, 0
        messages.append(dict(role='user', content=wire(result)))
        protocol.require(count(messages) <= context, 'Search response exceeds context')
        preview_tokens += cost
        displayed.update(c['id'] for c in result['cards'])
        trace.append(dict(action=action, result=result))
    choose = actions[-1]
    protocol.require(isinstance(choose, dict) and set(choose) == {'select'}, 'Expected SELECT')
    ids = choose['select']
    protocol.require(isinstance(ids, list) and len(ids) <= max_items and
                     all(isinstance(k, str) and k in displayed for k in ids) and len(ids) == len(set(ids)),
                     'Invalid, duplicate or undisplayed ID')
    messages.append(dict(role='assistant', content=wire(choose)))
    def check(refs):
        continuation = protocol.continuation_messages(row, queries, refs, arm='flat_skills')
        return messages + [continuation[-1]]
    empty = count(check([]))
    protocol.require(empty <= context, 'Empty CHECK exceeds context')
    refs, skipped = [], []
    for k in ids:
        proposed = refs + [{f: docs[k][f] for f in ('id', 'title', 'text')}]
        size = count(check(proposed))
        if size <= context and preview_tokens + max(0, size - empty) <= cap:
            refs = proposed
        else:
            skipped.append(k)
    final = check(refs)
    return dict(development_id=row['id'], model_calls=0, actions_source='supplied offline', searches=trace,
                select=ids, provided_ids=[r['id'] for r in refs], budget_skipped=skipped,
                preview_tokens=preview_tokens, full_material_tokens=max(0, count(final)-empty),
                prompt_tokens=count(final), check_messages_sha256=vr.digest(final))


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='Frozen final_revision development root on node14')
    parser.add_argument('--id', default=DEMO_ID)
    parser.add_argument('--library', type=Path, help='Explicit library directory; defaults to the shared-review 1000-question library')
    parser.add_argument('--actions', type=Path, help='JSON array; default is the hand-written optical-fiber example')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = vr.read(root/'plan.json')
    config = plan['configuration']
    protocol.require(vr.digest(config) == plan['configuration_sha256'], 'Configuration changed')
    protocol.require(vr.sha(root/'pilot_inputs.json') == config['pilot_sha256'], 'Frozen pilot inputs changed')
    library = (args.library.resolve() if args.library else
               shared_experience.library_path(root.parent.parent, 'n1000_seed1247261125'))
    publication = vr.read(library/'construction_acceptance.json')
    skill_file = library/'materials/skills.json'
    skill_hash = vr.sha(skill_file)
    if publication.get('protocol') == shared_experience.PROTOCOL:
        if args.library:
            protocol.require(shared_experience.library_path(library.parent.parent, library.name) == library,
                             'Library outside the shared release')
    else:
        library_plan = vr.read(library/'plan.json')
        protocol.require(args.library and publication['passed'] and vr.digest(library_plan['configuration']) ==
                         library_plan['configuration_sha256'] == publication['configuration_sha256'] and
                         vr.sha(library/'material_isolation.json') == publication['material_isolation_sha256'] and
                         skill_hash == publication['materials']['skills.json'], 'Wrong historical library release')
    row, = (r for r in vr.read(root/'pilot_inputs.json') if r['id'] == args.id)
    acceptance = vr.read(root/'pilot_acceptance.json')
    packet = root/'pilot/flat_skills'/vr.digest(args.id)[:24]/'reference_packet.json'
    protocol.require(acceptance['passed'] and acceptance['configuration_sha256'] ==
                     vr.read(root/'execution_repair.json')['configuration_sha256'] and
                     vr.sha(packet) == acceptance['files'][str(packet)], 'Development packet changed')
    prefix = vr.read(packet)['messages'][:3]
    queries = json.loads(prefix[2]['content'])
    protocol.require(protocol.continuation_messages(row, queries, [], arm='flat_skills')[:3] == prefix,
                     'Frozen PREPARE prefix differs')
    if args.actions:
        actions = vr.read(args.actions)
    else:
        # These two procedures come from the same source problem in the two frozen libraries.
        demo_item = {'634efb38bf73fe745e4731d2d408701e2bb55bdef6bd03d9ebefd55e69f8ce83': 'S604',
                     '6e11ffe22058f74ee317e7342f84e01d31ffa30595c024a6eb37c1d81efee6d2': 'S118'}
        protocol.require(args.id == DEMO_ID and skill_hash in demo_item, 'Supply actions for this question or library')
        actions = [*DEMO[:2], {'select': [demo_item[skill_hash]]}]
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True, trust_remote_code=False)
    def count(messages):
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 reasoning_effort='medium'))
    result = replay(row, queries, vr.read(skill_file), actions, count, config)
    result.update(script_sha256=vr.sha(__file__), skills_sha256=skill_hash, pilot_packet_sha256=vr.sha(packet),
                  library_acceptance_sha256=vr.sha(library/'construction_acceptance.json'))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
