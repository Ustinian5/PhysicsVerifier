import copy
import pytest

from scripts.audit_aligned_step_evaluation import check_request, recount, base, vr


def test_request_contract_rejects_prompt_or_sampling_or_budget_drift():
    messages=[dict(role='user',content='frozen input')]
    value=dict(model=base.r.EXPECTED_MODEL,messages=messages,temperature=0,top_p=1,
               max_tokens=123,stream=False,return_token_ids=True,reasoning_effort='medium')
    check_request(dict(request=value,request_sha256=vr.digest(value)),messages,123)
    for key,changed in [('temperature',.5),('max_tokens',124),('reasoning_effort','high'),
                        ('messages',[dict(role='user',content='other input')])]:
        alternate=copy.deepcopy(value);alternate[key]=changed
        with pytest.raises(ValueError):
            check_request(dict(request=alternate,request_sha256=vr.digest(alternate)),messages,123)


def test_recount_retains_failed_question_ground_truth():
    rows=[dict(matched_error_count=1,matched_pred_locatable=1,pred_locatable_finding_count=2,gt_error_count=3),
          dict(matched_error_count=0,matched_pred_locatable=0,pred_locatable_finding_count=0,gt_error_count=7)]
    assert recount(rows)==dict(precision=.5,recall=.1,f1=1/6)
    rows[1]['matched_pred_locatable']=1
    with pytest.raises(ValueError,match='bijective'):recount(rows)


def test_recount_has_no_division_by_zero():
    assert recount([])==dict(precision=0.,recall=0.,f1=0.)
