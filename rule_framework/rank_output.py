"""Bound Harmony analysis and serialize model-selected IDs using the actual 120B token vocabulary."""
from dataclasses import dataclass
from functools import lru_cache

ANALYSIS_BUDGET = 1024
# GPT-OSS Harmony. The server and client both check these constants against the checkpoint.
CHANNEL, ANALYSIS, MESSAGE = 200005, 35644, 200008
END, START, ASSISTANT, FINAL, RETURN = 200007, 200006, 173781, 17196, 200002
FIRST_SPECIAL = 199998
AFTER_ANALYSIS = (START, ASSISTANT, CHANNEL, FINAL, MESSAGE)
JSON_OPEN = (10848, 3375, 16853)
COMMA, CLOSE = 11, 28000


@dataclass(frozen=True)
class RankShape:
    count: int
    limit: int
    parents: tuple = ()

    def __post_init__(self):
        maximum = 2*len(set(self.parents)) if self.parents else 24
        if not (type(self.count) is int and 1 <= self.count <= 999 and
                type(self.limit) is int and 1 <= self.limit <= maximum):
            raise ValueError('Invalid ranking dimensions')
        if self.parents and (len(self.parents) != self.count or any(type(p) is not int for p in self.parents)):
            raise ValueError('Each child requires a parent group')

    def __call__(self, value):
        ids = value.get('ids') if isinstance(value, dict) else None
        if not isinstance(ids, list) or len(ids) > self.limit or any(
                type(i) is not int or not 1 <= i <= self.count for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Malformed or out-of-pool ranking')
        if self.parents:
            groups = [self.parents[i-1] for i in ids]
            if any(groups.count(p) > 2 for p in set(groups)):
                raise ValueError('Semantic route exceeded two children under one parent')
        return {'ids':ids}

    def wire(self):
        data = dict(pv_rank_count=self.count, pv_rank_limit=self.limit)
        if self.parents:
            data['pv_rank_parents'] = list(self.parents)
        return data

    @classmethod
    def from_wire(cls, args):
        return cls(args['pv_rank_count'], args['pv_rank_limit'], tuple(args.get('pv_rank_parents', ())))


@lru_cache(maxsize=2)
def number_tokens(tokenizer):
    encode = lambda s: tuple(tokenizer.encode(s, add_special_tokens=False))
    tokens = {
        '<|channel|>analysis<|message|>':(CHANNEL, ANALYSIS, MESSAGE),
        '<|end|><|start|>assistant<|channel|>final<|message|>':(END, *AFTER_ANALYSIS),
        '<|return|>':(RETURN,), '{"ids":[':JSON_OPEN, ',':(COMMA,), ']}':(CLOSE,)}
    if any(encode(s) != seq for s,seq in tokens.items()):
        raise ValueError('Different Harmony or JSON tokens')
    numbers = tuple(encode(str(i)) for i in range(1,1000))
    if any(len(seq) != 1 for seq in numbers) or len(set(numbers)) != 999:
        raise ValueError('Expected a unique token for each candidate number')
    return tuple(seq[0] for seq in numbers)


class RankGrammar:
    def __init__(self, shape, token_ids, max_tokens):
        self.shape = shape
        self.numbers = tuple(token_ids[:shape.count])
        self.number_id = {token:i for i,token in enumerate(self.numbers,1)}
        # END + assistant final header + JSON prefix + IDs/comma + closing token + RETURN.
        tail = 1 + len(AFTER_ANALYSIS) + len(JSON_OPEN) + 2*min(shape.count,shape.limit) + 1
        self.analysis_end = min(ANALYSIS_BUDGET, max_tokens-tail)
        if self.analysis_end < 3:
            raise ValueError('Not enough output budget for a complete ranked list')

    def remaining(self, chosen):
        if len(chosen) >= self.shape.limit:
            return []
        parents = self.shape.parents
        full = set()
        if parents:
            counts = {}
            for i in chosen:
                parent = parents[i-1]
                counts[parent] = counts.get(parent, 0)+1
                if counts[parent] == 2:
                    full.add(parent)
        return [token for i,token in enumerate(self.numbers,1)
                if i not in chosen and (not parents or parents[i-1] not in full)]

    def allowed(self, output):
        """None permits ordinary analysis text plus Harmony END; otherwise return legal next tokens."""
        if len(output) < 3:
            return ((CHANNEL,), (ANALYSIS, FINAL), (MESSAGE,))[len(output)]
        start = 3
        if output[1] == ANALYSIS:
            try:
                end = output.index(END, start)
            except ValueError:
                return (END,) if len(output) >= self.analysis_end else None
            header = output[end+1:]
            if len(header) < len(AFTER_ANALYSIS):
                return (AFTER_ANALYSIS[len(header)],)
            start = end+1+len(AFTER_ANALYSIS)
        body = output[start:]
        if len(body) < len(JSON_OPEN):
            return (JSON_OPEN[len(body)],)
        body = body[len(JSON_OPEN):]
        if body and body[-1] == CLOSE:
            return (RETURN,)
        chosen = [self.number_id[token] for token in body[::2]]
        candidates = self.remaining(chosen)
        if not body:
            return [*candidates, CLOSE]
        if body[-1] == COMMA:
            return candidates
        return (COMMA, CLOSE) if candidates else (CLOSE,)

    def audit(self, tokens, value):
        """Bind every selected integer and both channel boundaries to the raw output tokens."""
        ids = self.shape(value)['ids']
        if tokens[:3] == [CHANNEL, ANALYSIS, MESSAGE]:
            end = tokens.index(END, 3)
            if end > self.analysis_end or any(i >= FIRST_SPECIAL for i in tokens[3:end]):
                raise ValueError('Unbounded or malformed analysis channel')
            prefix = [END, *AFTER_ANALYSIS]
            tail = tokens[end:]
        elif tokens[:3] == [CHANNEL, FINAL, MESSAGE]:
            end, prefix, tail = 0, [], tokens[3:]
        else:
            raise ValueError('Unexpected selector channel header')
        body = list(JSON_OPEN)
        for n,i in enumerate(ids):
            if n:
                body.append(COMMA)
            body.append(self.numbers[i-1])
        if tail != [*prefix,*body,CLOSE,RETURN]:
            raise ValueError('Decoded ranking differs from generated IDs or terminal')
        return dict(analysis_tokens=max(0,end-3), reached_analysis_limit=end==self.analysis_end,
                    final_items=len(ids))
