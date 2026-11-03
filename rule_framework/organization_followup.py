"""Bounded flat skills and a separately scored positive-case replication.

The completed 99-question protocol is immutable. Extraction uses original source answers and bounded complete methods.
A shared-prose isolation extension applies to all knowledge arms; online checking
prompts, retrieval and per-question budgets remain fixed. The target rubric is scoring-only and never sent to the checker.
"""
import math
import copy
from collections import defaultdict

from rule_framework import organization_bench99 as previous

VERSION = 'experience_organization_99_bounded_skills_v15'
REPEAT_VERSION = 'organization_positive_case_repeat_v1'
MAX_SKILLS = 2
PILOT_SIZE = 12
EXTRACTION_PASSES = 1
PASS_OUTPUT_TOKENS = 16384
PILOT_POLICY = "engineering and provenance only; model content errors and known format failures retained, no semantic zero-error gate"
REPEATS = 5
TARGET_ID = 'cl_202_125689'
load_bench = previous.load_bench
ISOLATION_POLICY = dict(version='shared_subproblem_span_v1', normalized_span_length=160,
    heldout_stride=40, classification='conservative shared prose exclusion, not semantic independence proof')


class FamilyScreen(previous.FamilyScreen):
    """Detect shared subproblem stems even when whole-question overlap is low.

    Index held-out spans once. Scan source visible fields without pairwise full
    string alignment; preserve matched text as review evidence. Never use labels.
    """
    def __init__(self, heldout):
        super().__init__(heldout)
        self.spans = defaultdict(set)
        width = ISOLATION_POLICY['normalized_span_length']
        for row in heldout:
            text = previous.isolation.question_key(row['question'])
            for i in range(0, len(text)-width+1, ISOLATION_POLICY['heldout_stride']):
                span = text[i:i+width]
                if sum(c.isalpha() for c in span) >= width * .75:
                    self.spans[span].add(row['id'])

    def matches(self, row):
        found = {x['heldout_id']: x for x in super().matches(row)}
        width = ISOLATION_POLICY['normalized_span_length']
        for field in ('question', 'context', 'prediction', 'answer'):
            text = previous.isolation.question_key(row.get(field, '') or '')
            for i in range(len(text)-width+1):
                span = text[i:i+width]
                for sid in self.spans.get(span, ()):
                    found.setdefault(sid, dict(heldout_id=sid, reason='shared_subproblem_prose',
                                              field=field, normalized_shared_span=span))
        return [found[sid] for sid in sorted(found)]

INPUT_POLICY = 'same frozen case memory as original 99-case skills: question, 30B answer, and 120B review; one extraction, no added teacher or manual correction'

SKILL_PROMPT = previous.SKILL_PROMPT.replace(
    '不要求套用统一分步格式，不给固定数量配额，也不要求两道题共同支持。条目之间不建立调用树。',
    '不要求套用统一分步格式。默认提炼1条完整方法，最多2条；没有合适方法可为空，不为凑数量生成。'
    '同一方法的准备、计算、比较和条件说明合并为一条，第二条应服务于不同的独立检查目标。'
    '不要求两道题共同支持。条目之间不建立调用树。')


def skill_messages(case):
    messages = previous.skill_messages(case)
    messages[0]['content'] = SKILL_PROMPT
    return messages


def validate_skills(value):
    previous.validate_skills(value)
    previous.flat.require(len(value['skills']) <= MAX_SKILLS,
        'More than two skills: retain as content failure, do not truncate or reroll')
    return value


def materialize_skills(rows, outcomes):
    for item in outcomes:
        if item['status'] == 'collected': validate_skills(item['value'])
    skills, counts = previous.materialize_skills(rows, outcomes)
    counts.update(max_skills_per_source=MAX_SKILLS, default_skills_per_source=1,
                  source_input_policy=INPUT_POLICY, target_count_is_not_a_quota=True,
                  physical_quality_passed=False,
                  count_comparable_is_not_information_or_quality_equivalent=True)
    return skills, counts


def target_rubric():
    value = math.sqrt((1.500**2 - 1.460**2) / (1.500**2 * (25e-6)**2))
    return dict(id=TARGET_ID, scope='one arithmetic error, not whole-answer correctness',
        target='The numerical evaluation of alpha, not the already correct symbolic formula.',
        formula='sqrt((n1^2-n2^2)/(n1^2*a^2))',
        parameters=dict(n1=1.500, n2=1.460, a_m=25e-6),
        original_wrong_value_per_m=1.14e6, correct_value_per_m=value,
        review_instructions=[
            'Read the actual diagnosis and evidence. A matching location or number alone is insufficient.',
            'correct_detection: explicitly identifies the alpha numerical evaluation as wrong, with a correct correction (about 9.2e3 per metre) or an independently valid equivalent explanation.',
            'incorrect_target_claim: addresses alpha but gives a wrong correction/reason, or contradicts a correct correction elsewhere in the same output.',
            'missed: no diagnosis of this specific arithmetic error; diagnosing later acceptance-angle errors alone is insufficient.',
            'unassessable: evidence is insufficient to judge; report separately, never as correct.',
            'Citations are not required for correctness and do not establish causality. Do not judge other errors as part of this target score.'],
        actor='Codex assistant', human_gold=False, source_selected_after_observing_results=True,
        independent_test=False, planned_repeats_per_arm=REPEATS,
        denominator_policy='all five scheduled repetitions, including technical failures',
        extrapolation='conditional single-case repeatability; no overall P/R/F1 or causal rule benefit')
