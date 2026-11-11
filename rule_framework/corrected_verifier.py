"""Continuous, reference-blind checking for the corrected 30B experiments.

The ranking functions are the existing BM25 implementation. This adapter only
adds stable ten-result pages and single complete-item reads. No shared PREPARE,
separate checker, flat tree fallback, or six-item material cap is introduced.
"""
import copy
import json

from rule_framework.flat_skill_rag import FlatMemory, TreeMemory, require
from rule_framework import final_answer_protocol as final
from scripts import continuous_context_v2 as step

VERSION = 'corrected_continuous_four_arm_v1'
CHECK_REVISION = 'required_lookup_suffix_20261111'
LANGUAGE_REVISION = 'chinese_tree_lookup_and_full_read_20261111'
BILINGUAL_REVISION = 'bilingual_tree_lookup_20261111'
REQUIRED_LOOKUP_SUFFIX = '''Before publishing your final JSON, actually call search_memory in this conversation. If it returns a potentially relevant candidate, call read_memory to read the COMPLETE item before continuing your check. Then continue checking the original answer and publish the final JSON in this same conversation. Even when you already have an independent judgment, do not omit the lookup. If no candidate is relevant, retain the actual lookup attempt and continue independently; do not force an unrelated item. Do not simulate tool results.'''
READ_SUFFIX = '''If a returned candidate concerns the physical relationship you are checking, read its COMPLETE item with read_memory before deciding whether its conditions and advice apply. The search excerpt is only a preview, not the full item. A different historical setup or lack of an exact match to this problem is not by itself a reason to skip a candidate about the same physical relationship. Judge applicability after reading; reject inapplicable advice and do not force genuinely unrelated material.'''
TREE_LANGUAGE_SUFFIX = '''The rule-tree directory is Chinese and routing is lexical BM25. Use short Chinese physics terms for the relationships and conditions being checked, translating the problem's concepts into Chinese for your query. If you already tried an English query and it returned no candidates, retry that issue once using Chinese physics terms before concluding that the tree has no relevant candidate. This remains within the original response and token budget; no extra allowance is granted.'''
TREE_BILINGUAL_SUFFIX = '''The rule tree mixes languages: Domain/Topic names are English with Chinese summaries, while many lower-level cluster summaries are English. BM25 routes through EVERY level, so a single-language query can stop before reaching a relevant rule. In each search_memory query, include the corresponding physical domain name and BOTH Chinese and English physics terms for the relationship and conditions being checked. Infer those terms from the current problem and answer. The existing domains are Mechanics (力学), Electromagnetism (电磁学), Thermodynamics & Statistical Physics (热力学与统计物理), Optics (光学), Modern Physics (近代物理), and Experimental Physics (实验物理). Use the appropriate domain, not all domains. If an earlier single-language lookup missed, retry that issue once with a bilingual query before concluding that the library lacks relevant candidates. Read a potentially relevant full item as instructed above. All searches, reads and final output remain within the original response and token budget.'''
ARMS = ('pure', 'case_rag', 'flat_skills', 'rule_tree')
LIMITS = dict(responses=12, output_tokens=16384, input_tokens=600000,
              context_tokens=131072, seconds=1800, request_seconds=1200,
              tool_chars=500000, page_size=10)
USE = '''Use this group's fallible knowledge aid within this SAME conversation. Understand the complete problem and answer, then choose your own lookup queries when checking concrete relationships or assumptions. Do not hand selected material to another checker. Attempt a search before your final judgment; when it returns potentially relevant candidates, read at least one relevant COMPLETE item. If no relevant candidates are found, leave the failed lookup recorded and continue independent checking. Do not force an unrelated item's applicability or suppress independent findings outside the library.
search_memory is lexical BM25, not semantic retrieval. Use short physical concepts and distinguishing conditions; the materials are predominantly Chinese, so include appropriate Chinese terms when useful. It returns at most ten summaries per page with a stable offset. Summaries and scores do not establish applicability. read_memory reads ONE actual returned ID in full. Never invent IDs or simulate tools. No fixed number of different items must be read. Respect the original conditions, objects and variable meanings; reject conflicting or inapplicable advice. A historical 30B answer may be wrong: distinguish it from its 120B review. Knowledge is not a reference answer to the present problem.
Continue checking the entire original answer after consulting knowledge. Do not publish a separate selection response. Preserve enough of the common budget for the final JSON. Tool results report remaining budget; oversized complete items are refused rather than silently truncated. For step diagnostics, rule_refs may cite IDs of relevant complete experiences, skills or rules actually read; independently justified diagnoses need no citation.'''
DESCRIPTIONS = {
    'case_rag': 'Your knowledge consists of original historical problems, 30B answers and their 120B error reviews, searched as flat documents.',
    'flat_skills': 'Your knowledge consists of flat independent checking methods distilled from those same error experiences; there is no directory or tree.',
    'rule_tree': 'Your knowledge is a physical rule tree. Each search routes by BM25 over child names/summaries, keeping at most two positive children per parent, then ranks leaves in the reached scope. There is no flat fallback; a routing miss is a real empty result.',
}


def tools():
    return [dict(type='function', function=dict(name='search_memory',
        description='Search the assigned knowledge store; stable pages of at most ten summaries.',
        parameters=dict(type='object', properties=dict(query=dict(type='string'),
            offset=dict(type='integer', minimum=0)), required=['query', 'offset'], additionalProperties=False))),
        dict(type='function', function=dict(name='read_memory',
        description='Read one complete item by an ID returned in this conversation.',
        parameters=dict(type='object', properties=dict(id=dict(type='string')),
                        required=['id'], additionalProperties=False)))]


def messages(row, arm, task, *, revision=None):
    require(arm in ARMS and task in {'step', 'final'}, 'Unknown experiment')
    require(revision in {None, CHECK_REVISION, LANGUAGE_REVISION, BILINGUAL_REVISION}, 'Unknown checking prompt revision')
    # Whitelist even when callers possess benchmark references or labels.
    public = {k: row.get(k, '') for k in ('question', 'prediction', 'context')}
    extra = USE + '\n' + DESCRIPTIONS[arm] if arm != 'pure' else ''
    if task == 'final':
        result = final.verification_messages(public, tool_instructions=extra +
            '\nAt most 12 responses and 16384 cumulative generated tokens including reasoning and tools.')
    else:
        result = step.messages(public, rules=False, tree_root=None)
        if extra:
            result[0]['content'] += '\n\n' + extra
    if revision in {CHECK_REVISION, LANGUAGE_REVISION, BILINGUAL_REVISION} and arm != 'pure':
        result[-1]['content'] += '\n\n' + REQUIRED_LOOKUP_SUFFIX
        if revision in {LANGUAGE_REVISION, BILINGUAL_REVISION}:
            result[-1]['content'] += '\n\n' + READ_SUFFIX
            if arm == 'rule_tree':result[-1]['content'] += '\n\n' + (
                TREE_BILINGUAL_SUFFIX if revision==BILINGUAL_REVISION else TREE_LANGUAGE_SUFFIX)
    return result


def all_ranked(memory, query, allowed=None):
    """Extend the frozen six-result helper without changing scores or tie order."""
    remaining = set(memory.ids if allowed is None else allowed)
    ranked = []
    while remaining:
        page = memory.search(query, limit=6, allowed_ids=remaining)
        if not page:
            break
        ranked.extend(page)
        remaining.difference_update(x['id'] for x in page)
    return ranked


class Knowledge:
    def __init__(self, arm, memory):
        require(arm in ARMS, 'Unknown arm')
        require((arm == 'pure') == (memory is None), 'Wrong pure memory')
        if arm == 'rule_tree':
            require(isinstance(memory, TreeMemory), 'Tree must use hierarchical BM25')
        elif arm != 'pure':
            kind = 'case' if arm == 'case_rag' else 'skill'
            require(isinstance(memory, FlatMemory) and
                    all(d['kind'] == kind for d in memory.docs.values()), 'Wrong material kind')
        self.arm, self.memory = arm, memory
        self.offered, self.read_ids, self.events = set(), set(), []
        self.returned_chars = 0

    def preview(self, name, args):
        require(self.arm != 'pure', 'Pure model has no tools')
        require(isinstance(args, dict), 'Tool arguments must be an object')
        if name == 'search_memory':
            require(set(args) == {'query', 'offset'} and type(args['offset']) is int
                    and args['offset'] >= 0, 'Invalid search/page fields')
            query = args['query']
            if self.arm == 'rule_tree':
                _, trace = self.memory.retrieve(query)
                allowed = {rid for route in trace['routes'] for rid in route.get('leaf_ids', [])}
                ranked = all_ranked(self.memory.memory, query, allowed)
            else:
                trace = dict(algorithm='flat_bm25')
                ranked = all_ranked(self.memory, query)
            page = ranked[args['offset']:args['offset'] + LIMITS['page_size']]
            return dict(items=page, total=len(ranked), offset=args['offset'],
                        next_offset=args['offset']+len(page) if args['offset']+len(page)<len(ranked) else None,
                        retrieval=trace)
        require(name == 'read_memory' and set(args) == {'id'} and isinstance(args['id'], str), 'Invalid read')
        require(args['id'] in self.offered, 'ID was not returned by this conversation')
        flat = self.memory.memory if self.arm == 'rule_tree' else self.memory
        return flat.view(args['id'])

    def commit(self, name, args, result):
        size = len(json.dumps(result, ensure_ascii=False))
        require(self.returned_chars + size <= LIMITS['tool_chars'], 'Tool-character bound exceeded')
        self.returned_chars += size
        self.events.append(dict(tool=name, arguments=copy.deepcopy(args), result=copy.deepcopy(result)))
        if 'error' not in result:
            if name == 'search_memory':
                self.offered.update(x['id'] for x in result['items'])
            else:
                self.read_ids.add(args['id'])

    def audit(self):
        searches = [e for e in self.events if e['tool'] == 'search_memory' and 'error' not in e['result']]
        return dict(searches=len(searches), reads=sorted(self.read_ids),
            routing_or_search_misses=sum(not e['result']['items'] for e in searches),
            protocol_noncompliance=self.arm != 'pure' and (not searches or bool(self.offered) and not self.read_ids),
            compliance_note='Candidate relevance is not automatically judged; non-use is reported, never rerolled or excluded.')


def assistant_tool_message(decoded):
    # The pinned GPT-OSS template renders content as analysis for tool calls;
    # reasoning_content as a separate OpenAI field would silently be dropped.
    return dict(role='assistant', content=decoded['reasoning_content'], tool_calls=decoded['tool_calls'])


def final_scores(rows, labels, predictions):
    counts = dict(TP=0, FP=0, FN=0, TN=0, uncertain=0, execution_failure=0)
    for row in rows:
        p = predictions[row['id']]
        if p['status'] != 'complete':
            counts['execution_failure'] += 1
            continue
        actual = p['value']['verdict']; gold = labels[row['id']]['verdict']
        require(gold in {'correct', 'incorrect'}, 'Unresolved gold in balanced benchmark')
        if actual == 'uncertain':
            counts['uncertain'] += 1
        else:
            counts[{('incorrect','incorrect'):'TP', ('correct','incorrect'):'FP',
                    ('incorrect','correct'):'FN', ('correct','correct'):'TN'}[gold,actual]] += 1
    tp, fp, fn, tn = (counts[k] for k in ('TP','FP','FN','TN'))
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    return dict(counts, precision=precision, recall=recall,
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0,
        false_positive_rate=fp/(fp+tn) if fp+tn else 0.0,
        total=len(rows), binary_coverage=(tp+fp+fn+tn)/max(1,len(rows)),
        complete_binary_evaluation=tp+fp+fn+tn == len(rows), human_gold=False,
        metric_scope='Reference-assisted machine labels; P/R/F1 over explicit binary verdicts; abstentions and failures separate')
