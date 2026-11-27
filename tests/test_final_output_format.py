import json
from pathlib import Path
import pytest

from rule_framework import final_output_format as fmt
from rule_framework.rank_output import CHANNEL, END, FINAL, MESSAGE, START
from scripts import run_structured_final as runner


def test_grammar_starts_only_at_this_responses_final_body():
    final = [CHANNEL,FINAL,MESSAGE]
    history = final + [11, END, START, 173781]
    assert not fmt.final_header_ended(history)
    assert not fmt.final_header_ended(history + [CHANNEL, 35644, MESSAGE])
    assert fmt.final_header_ended(history + final)
    assert fmt.final_header_ended(history + [CHANNEL,FINAL,200003,937,MESSAGE])
    assert not fmt.final_header_ended(history + final + [11])
    assert not fmt.final_header_ended(final + [MESSAGE])


def test_query_cardinality_and_verdict_contract():
    queries = runner.schema_for(Path('pilot/queries/id'))
    verdict = runner.schema_for(Path('formal/pure/id/check'))
    terms = queries['properties']['queries']
    assert (terms['minItems'],terms['maxItems']) == (1,3)
    assert set(verdict['required']) == {'student_final_answer','verdict','reason'}
    assert not verdict['additionalProperties']
    with pytest.raises(ValueError):
        runner.schema_for(Path('formal/unknown'))


def test_compiled_server_grammar():
    xgr = pytest.importorskip('xgrammar')
    from xgrammar.testing import _is_grammar_accept_string
    query = xgr.Grammar.from_json_schema(json.dumps(fmt.QUERY))
    verdict = xgr.Grammar.from_json_schema(json.dumps(fmt.VERDICT))
    for size in (1,2,3):
        assert _is_grammar_accept_string(query,json.dumps({'queries':['force']*size}))
    for size in (0,4,6):
        assert not _is_grammar_accept_string(query,json.dumps({'queries':['force']*size}))
    for state in ('correct','incorrect','uncertain'):
        text = json.dumps(dict(verdict=state,student_final_answer='42',reason='evidence'))
        assert _is_grammar_accept_string(verdict,text)
        assert not _is_grammar_accept_string(verdict,text+'\n'+text)
