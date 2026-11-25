import pytest

from rule_framework import rank_output as rank

TOKENS = tuple(range(4000,4999))


def test_same_format_handles_empty_lists_and_distinct_choices():
    shape = rank.RankShape(4,3)
    grammar = rank.RankGrammar(shape,TOKENS,2048)
    prefix = [rank.CHANNEL,rank.FINAL,rank.MESSAGE,*rank.JSON_OPEN]
    assert set(grammar.allowed(prefix)) == {*TOKENS[:4],rank.CLOSE}
    assert grammar.audit([*prefix,rank.CLOSE,rank.RETURN],{'ids':[]})['final_items']==0
    chosen = [*prefix,TOKENS[2],rank.COMMA]
    assert set(grammar.allowed(chosen)) == {TOKENS[0],TOKENS[1],TOKENS[3]}
    full = [*chosen,TOKENS[0],rank.COMMA,TOKENS[1]]
    assert grammar.allowed(full)==(rank.CLOSE,)
    assert grammar.audit([*full,rank.CLOSE,rank.RETURN],{'ids':[3,1,2]})['final_items']==3
    with pytest.raises(ValueError):
        grammar.audit([*full,rank.CLOSE,rank.RETURN],{'ids':[3,1,4]})


def test_analysis_reserves_enough_tokens_to_emit_maximum_ranking():
    grammar = rank.RankGrammar(rank.RankShape(256,24), TOKENS, 120)
    output = [rank.CHANNEL,rank.ANALYSIS,rank.MESSAGE]+[36]*(grammar.analysis_end-3)
    assert grammar.allowed(output[:-1]) is None and grammar.allowed(output)==(rank.END,)
    expected = [rank.END,*rank.AFTER_ANALYSIS,*rank.JSON_OPEN]
    for i in range(24):
        if i:
            expected.append(rank.COMMA)
        expected.append(TOKENS[i])
    for token in [*expected,rank.CLOSE,rank.RETURN]:
        assert token in grammar.allowed(output)
        output.append(token)
    assert len(output)==120 and grammar.audit(output,{'ids':list(range(1,25))})['reached_analysis_limit']


def test_route_numbering_preserves_parent_quotas_and_natural_analysis_end():
    shape=rank.RankShape(4,4,(7,7,7,8))
    assert rank.RankShape.from_wire(shape.wire())==shape
    grammar=rank.RankGrammar(shape,TOKENS,2048)
    head=[rank.CHANNEL,rank.ANALYSIS,rank.MESSAGE,357,rank.END,*rank.AFTER_ANALYSIS,*rank.JSON_OPEN]
    body=[*head,TOKENS[2],rank.COMMA,TOKENS[0],rank.COMMA]
    assert grammar.allowed(body)==[TOKENS[3]]
    assert grammar.audit([*body,TOKENS[3],rank.CLOSE,rank.RETURN],{'ids':[3,1,4]})['analysis_tokens']==1
    with pytest.raises(ValueError):
        shape({'ids':[3,1,2]})
    with pytest.raises(ValueError):
        shape({'ids':[3,3]})
