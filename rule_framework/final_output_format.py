"""Frozen English question-level output schemas and Harmony final-channel boundary."""
from rule_framework.rank_output import CHANNEL, END, FINAL, MESSAGE, START


def object_schema(fields):
    return dict(type='object', properties=fields, required=list(fields), additionalProperties=False)


QUERY = object_schema({'queries': dict(type='array', minItems=1, maxItems=3,
                                      items=dict(type='string', minLength=1, maxLength=300))})
VERDICT = object_schema({'verdict': dict(type='string', enum=['correct', 'incorrect', 'uncertain']),
                         'student_final_answer': dict(type='string'), 'reason': dict(type='string', minLength=1)})


def final_header_ended(ids):
    # vLLM also calls this on the entire prompt. An earlier PREPARE final must
    # never start the grammar in the following CHECK's analysis.
    if not ids or ids[-1] != MESSAGE:
        return False
    for i in range(len(ids)-2, max(-1, len(ids)-24), -1):
        if ids[i] in (CHANNEL, MESSAGE, START, END):
            return ids[i] == CHANNEL and ids[i+1] == FINAL
    return False
