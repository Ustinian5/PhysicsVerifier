"""Match the frozen BM25 control to the existing source hold on the 50 questions."""
import argparse
from pathlib import Path
import time
import httpx

from scripts import timed_retrieval_baseline as base
from scripts import timed_retrieval_runtime as runtime
from scripts import confirm_rag_tree_retrieval as previous

cohort,b,vr,require = base.cohort,base.b,base.vr,base.require


def signature(out):
    return vr.digest([cohort.IDS,vr.sha(__file__),vr.sha(runtime.__file__),vr.sha(base.__file__),
                      vr.sha(out/'bm25_baseline.json'),vr.sha(out/'bm25_timing.json')])


def run(folder, out, stamp):
    from contextlib import nullcontext
    from transformers import AutoTokenizer
    directory = out/'bm25_aligned'
    control = vr.read(out/'bm25_baseline.json')
    rows = vr.read(out/'inputs.json')
    expected = vr.read(out/'bm25_timing.json')['rows']
    changed = {(a,k) for a,group in expected.items() for k,x in group.items() if not x['same_delivery']}
    require(changed=={('rule_tree',rows[4]['id'])}, 'Unexpected source-hold adjustment')
    arm = 'rule_tree'
    k = rows[4]['id']
    dest = directory/'rows'/k/'check'
    old = b.final.execution_plan(b.FORMAL)
    original = control['rows'][arm][k]
    qrow = rows[4]
    _, prep = b.prefix_record(b.FORMAL,old,'formal',qrow)
    plan = runtime.checking_plan(arm,stamp)
    tok = AutoTokenizer.from_pretrained(b.host.HOT/'models/gpt-oss-120b',local_files_only=True)
    idx = base.load_index(b.RESULTS,arm)
    messages, ids, _ = base.bm25_lookup(qrow,arm,prep['value'],idx,lambda m:base.count_messages(tok,m),plan['configuration']['limits'])
    require(ids == expected[arm][k]['eligible_ids'] and ids != original['delivered'],'Unexpected corrected delivery')
    service = nullcontext() if (dest/'result.json').is_file() else runtime.large_service(folder,arm,stamp,directory/'STOP')
    with service, httpx.Client(trust_env=False) as client:
        budget = previous.Budget(directory,plan,tok,client)
        check = budget.call(dest,messages,plan['configuration']['limits']['max_tokens_per_response'],
                            b.final.p.original.validate_verdict,[prep],time.monotonic())
    predictions = dict(control['rows'][arm])
    predictions[k] = dict(status=check['status'], value=check['value'], delivered=ids,
                          file_sha256=vr.sha(dest/'result.json'),source='source_hold_aligned_check')
    gold = vr.read(b.FORMAL/'labels_private.json')
    truth = {q:gold[q]['verdict'] for q in predictions}
    scores = dict(control['groups'],rule_tree=b.final.measure(truth,predictions))
    b.final.immutable(directory/'metrics.json',dict(groups=scores,changed=[k],replacement=predictions[k],
                    parent_sha256=vr.sha(out/'bm25_baseline.json'),cohort=cohort.IDS,positive_class='incorrect'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','confirm'))
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args()
    out=b.RESULTS/cohort.RESULT
    stamp=signature(out)
    runtime.detached(out/'bm25_aligned',args.attempt,stamp,lambda folder:run(folder,out,stamp),args.action=='confirm')
