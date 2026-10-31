"""Retain audited model-output failures and continue without resampling.

The original runner/results remain immutable. A sidecar accounts for responses
whose JSON-format label failed before the original engine recorded their usage.
An explicit amendment permits known output failures in controls and formal
tasks. Failed controls remain failed; identity, transport and unknown errors stop.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import contextmanager

from scripts import run_organization_comparison as base

vr, r = base.vr, base.r
RECEIPT = 'native_format_recovery.json'
OUTPUT_AMENDMENT = 'output_failure_recovery.json'


def missing_final_usage(raw, request):
    """Validate a terminated analysis-only response; never extract an answer."""
    from vllm.entrypoints import harmony_utils as h
    from scripts.gpt_oss_harmony_transport import decode_response
    try:
        decode_response(raw, model=request['model'], max_tokens=request['max_tokens'], native_tools=None)
    except ValueError as exc:
        base.require(str(exc) == 'Missing final message', 'Unknown native output failure')
    else:
        raise ValueError('Valid final requires separate review, not discard')
    parsed = h.parse_output_into_messages(raw['choices'][0]['token_ids'])
    base.require(not parsed.current_content and bool(parsed.messages) and
                 all(m.channel == 'analysis' and not m.recipient for m in parsed.messages),
                 'Not an analysis-only terminated response')
    return raw['usage']


@contextmanager
def released_artifact_check(output):
    """Verify consumed release bytes; retain, but do not re-stat, old raw history."""
    expected=vr.read(output/'plan.json')['configuration']['rules']
    def release_plan(path, *, full=False):
        base.require(not full and Path(path).resolve()==Path(expected['path']).resolve(),
                     'Bounded check cannot replace an explicit full audit')
        plan=vr.read(Path(path)/'plan.json')
        base.require(vr.digest(plan['configuration'])==plan['configuration_sha256']==expected['configuration_sha256'],
                     'Rule construction plan changed')
        # rules_asset additionally checks release SHA, flags, model and counts;
        # base.verify compares the actual catalog SHA with the frozen input SHA.
        return plan
    with patch.object(base.old.navigation,'verify',release_plan):yield


def decode_final(raw, request):
    from vllm.entrypoints import harmony_utils as h
    from scripts.resume_continuous_context_v2 import ORIGINAL_DECODE
    original = h.parse_output_into_messages
    def normalize(ids):
        parsed = original(ids)
        messages = [SimpleNamespace(channel=m.channel, content=m.content,
            recipient=('<|constrain|>json' if m.channel == 'final' and
                       m.recipient == '<|constrain|>diagnostics' else m.recipient)) for m in parsed.messages]
        return SimpleNamespace(messages=messages, current_content=parsed.current_content)
    with patch.object(h, 'parse_output_into_messages', normalize):
        return ORIGINAL_DECODE(raw, model=request['model'], max_tokens=request['max_tokens'], native_tools=None)


def inspect_failure(path):
    value = vr.read(path)
    base.runtime.audit_result(value)
    reason = value.get('failure_reason', '')
    missing = reason == 'ValueError: Missing final message'
    base.require(value['status']=='failed' and value.get('diagnostics')==[] and
                 (reason.startswith('JSONDecodeError:') or missing), 'Not an audited output failure')
    state = vr.read(path.with_name('state.json')); pending = state['pending']
    request_path, raw_path, response_path = [Path(pending[k]) for k in ('request_path','raw_path','response_path')]
    saved, raw, record = [vr.read(p) for p in (request_path, raw_path, response_path)]
    request = saved['request']
    base.require(saved['request_sha256']==r.sha256_value(request) and
        request['model']==r.EXPECTED_MODEL and not request.get('tools') and
        request['max_tokens']==pending['max_tokens'] and 0 < request['max_tokens'] <= 16384,
        'Unexpected failed request')
    error_matches = (record.get('error') == reason if missing else
                     record.get('error','').startswith('JSONDecodeError:'))
    base.require(record.get('http_status')==200 and error_matches and
        record['raw_sse_sha256']==vr.sha(raw_path) and raw.get('model')==r.EXPECTED_MODEL and
        record.get('model_ids')==[r.EXPECTED_MODEL] and len(raw.get('choices',[]))==1 and
        raw['choices'][0]['finish_reason']=='stop' and raw.get('usage')==record.get('usage'),
        'Transport, identity or completion failure is not eligible')
    if missing:
        usage = missing_final_usage(raw, request)
    else:
        decoded = decode_final(raw, request)
        base.require(not decoded.get('tool_calls') and decoded['model_ids']==[r.EXPECTED_MODEL] and
                     decoded['finish_reasons']==['stop'], 'Not a complete plain final')
        try:
            # Match the frozen native format-label gate, not a permissive parser.
            json.loads(decoded['content'])
        except json.JSONDecodeError:
            pass
        else:
            raise ValueError('Valid final requires separate review, not discard')
        usage = decoded['usage']
    base.require(str(request_path) not in {e['request_path'] for e in value['trace']['response_evidence']},
                 'Response already accounted')
    base.require(0 <= usage['completion_tokens'] <= request['max_tokens'] and usage['prompt_tokens'] > 0,
                 'Invalid usage')
    return {'id':value['id'], 'result_sha256':vr.sha(path), 'request_path':str(request_path),
            'usage':usage, 'files':{str(p):vr.sha(p) for p in (path,path.with_name('state.json'),request_path,raw_path,response_path)},
            'status':'failed_content_preserved','failure_reason':reason,
            'new_model_calls':0,'content_modified':False}


def account(path):
    sidecar=path.with_name('native_format_accounting.json')
    if sidecar.exists():
        value=vr.read(sidecar)
        for name,sha in value['files'].items():base.require(vr.sha(Path(name))==sha,'Failed evidence changed')
        return value
    value=inspect_failure(path);vr.atomic(sidecar,value);return value


def prepare(output):
    with r.run_lock(output/'active.lock'):
        plan=base.verify(output)
        base.require(not (output/'runner.pid').exists() and not (output/RECEIPT).exists(), 'Existing run/recovery')
        affected=[]
        for path in (output/'formal').rglob('result.json'):
            value=vr.read(path)
            if not base.runtime.healthy_result(value) and not base.complete_content_failure(value):
                affected.append(account(path))
        base.require(affected,'No eligible decode failure')
        files=[p for part in ('formal','controls','queries') for p in (output/part).rglob('*')
               if p.is_file() and p.suffix in ('.json','.sse')]
        vr.atomic(output/RECEIPT,{'configuration_sha256':plan['configuration_sha256'],
            'plan_sha256':vr.sha(output/'plan.json'),'code_sha256':vr.sha(Path(__file__)),
            'base_sources':base.source_hashes(),'prefix_files':{str(p):vr.sha(p) for p in files},
            'affected':affected,'new_model_calls':0,'scientific_protocol_changed':False})


def verify(output):
    with r.run_lock(output.parent/'organization_verification.lock'),released_artifact_check(output):
        plan=base.verify(output)
    receipt=vr.read(output/RECEIPT)
    expected_code=receipt['code_sha256']; previous=output/RECEIPT
    prefix_files=dict(receipt['prefix_files'])
    for name in ('released_artifact_recovery.json', OUTPUT_AMENDMENT):
        path=output/name
        if not path.exists():continue
        amendment=vr.read(path)
        base.require(amendment['previous_receipt_sha256']==vr.sha(previous) and
            amendment['previous_code_sha256']==expected_code and
            amendment['configuration_sha256']==plan['configuration_sha256'], 'Unbound verification amendment')
        for filename,sha in amendment.get('prefix_files',{}).items():
            base.require(filename not in prefix_files or prefix_files[filename]==sha,'Conflicting prefix hashes')
            prefix_files[filename]=sha
        expected_code=amendment['new_code_sha256'];previous=path
    base.require(receipt['configuration_sha256']==plan['configuration_sha256'] and
        receipt['plan_sha256']==vr.sha(output/'plan.json') and expected_code==vr.sha(Path(__file__)) and
        receipt['base_sources']==base.source_hashes(),'Recovery binding changed')
    vr.EvidenceCache(output/'native_recovery_cache.json').check(prefix_files)
    return plan


def retained_failure(path, *, output_amended):
    """Return a failed record only after evidence classification, never a pass."""
    value=vr.read(path)
    base.runtime.audit_result(value)
    base.require(value.get('status')=='failed', 'Successful protocol violation must stop')
    if base.complete_content_failure(value):return value
    base.require(output_amended or value.get('failure_reason','').startswith('JSONDecodeError:'),
                 'Missing-final handling needs the explicit amendment')
    account(path)
    return value


def control_summary(output):
    groups={}
    for arm in base.p.ARMS:
        paths=list((output/'controls'/arm).rglob('result.json'))
        values=[vr.read(p) for p in paths]
        passed=sum(base.runtime.healthy_result(v) and v.get('protocol_audit',{}).get('passed',False)
                   for v in values)
        supplemental=[account(p) for p,v in zip(paths,values)
                      if not base.runtime.healthy_result(v) and not base.complete_content_failure(v)]
        request_paths={e['request_path'] for v in values for e in v['trace']['response_evidence']}
        request_paths.update(x['request_path'] for x in supplemental)
        groups[arm]={'recorded':len(values),'planned':len(base.runtime.CONTROLS),'passed':passed,
                     'failed':len(values)-passed,
                     'all_passed':len(values)==passed==len(base.runtime.CONTROLS),
                     'supplemental_failed_decode_usage':supplemental,
                     'distinct_actual_requests':len(request_paths)}
    return groups


def score(output, plan=None):
    report=base.score(output,plan)
    extras={}
    for arm in base.p.ARMS:
        extras[arm]=[account(p) for p in (output/'formal'/arm).rglob('result.json')
            if not base.runtime.healthy_result(vr.read(p)) and not base.complete_content_failure(vr.read(p))]
        report['usage'][arm]['distinct_actual_requests']+=len({x['request_path'] for x in extras[arm]})
    report['supplemental_failed_decode_usage']=extras
    report['shared_preparation_requests_counted_once_in_batch']+=len({x['request_path'] for v in extras.values() for x in v})
    if (output/OUTPUT_AMENDMENT).exists():
        report['output_failure_amendment_sha256']=vr.sha(output/OUTPUT_AMENDMENT)
        report['engineering_controls']=control_summary(output)
        report['all_engineering_controls_passed']=all(x['all_passed'] for x in report['engineering_controls'].values())
    vr.atomic(output/'metrics.json',report)
    return report


def run(output):
    plan=verify(output)
    output_amended=(output/OUTPUT_AMENDMENT).exists()
    original_condition, original_score=base.run_condition,base.score
    def condition(out,row,phase,arm,*args,**kwargs):
        try:return original_condition(out,row,phase,arm,*args,**kwargs)
        except ValueError:
            path=base.path_for(out,phase,arm,row)
            if phase not in ('formal','controls') or not path.exists():raise
            if phase=='controls' and not output_amended:raise
            value=retained_failure(path,output_amended=output_amended)
            base.require(value['configuration_sha256']==plan['configuration_sha256'], 'Failure binding changed')
            base.old.emit(out,phase,arm=arm,sample_id=row['id'],result_status='failed_content_preserved')
            return value
    def scoring(out,plan=None):
        with patch.object(base,'score',original_score):return score(out,plan)
    with patch.object(base,'run_condition',condition),patch.object(base,'score',scoring), \
         patch.object(base,'verify',lambda out:plan):base.run(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run','verify','score','confirm-detached'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if args.mode=='prepare':prepare(out)
    elif args.mode=='run':run(out)
    else:
        plan=verify(out)
        if args.mode=='score':score(out,plan)
        elif args.mode=='confirm-detached':
            with patch.object(base,'verify',lambda out:plan):base.confirm(out)


if __name__=='__main__':main()
