"""Continue complete malformed native finals without rerolling frozen samples.

The original runner/results remain immutable. A sidecar accounts for responses
whose JSON-format label failed before the original engine recorded their usage.
Unknown transport, identity, native channel, length and semantic errors stop.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_organization_comparison as base

vr, r = base.vr, base.r
RECEIPT = 'native_format_recovery.json'


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
    base.require(value['status']=='failed' and value.get('diagnostics')==[] and
                 value.get('failure_reason','').startswith('JSONDecodeError:'), 'Not an audited JSON failure')
    state = vr.read(path.with_name('state.json')); pending = state['pending']
    request_path, raw_path, response_path = [Path(pending[k]) for k in ('request_path','raw_path','response_path')]
    saved, raw, record = [vr.read(p) for p in (request_path, raw_path, response_path)]
    request = saved['request']
    base.require(saved['request_sha256']==r.sha256_value(request) and
        request['model']==r.EXPECTED_MODEL and not request.get('tools') and
        request['max_tokens']==pending['max_tokens'] and 0 < request['max_tokens'] <= 16384,
        'Unexpected failed request')
    base.require(record.get('http_status')==200 and record.get('error','').startswith('JSONDecodeError:') and
        record['raw_sse_sha256']==vr.sha(raw_path) and raw.get('model')==r.EXPECTED_MODEL and
        record.get('model_ids')==[r.EXPECTED_MODEL] and len(raw.get('choices',[]))==1 and
        raw['choices'][0]['finish_reason']=='stop' and raw.get('usage')==record.get('usage'),
        'Transport, identity or completion failure is not eligible')
    decoded = decode_final(raw, request)
    base.require(not decoded.get('tool_calls') and decoded['model_ids']==[r.EXPECTED_MODEL] and
                 decoded['finish_reasons']==['stop'], 'Not a complete plain final')
    try:
        # Match the native format-label gate exactly. Do not apply the later
        # permissive parser to rescue a response rejected by the frozen gate.
        json.loads(decoded['content'])
    except json.JSONDecodeError:
        pass
    else:
        raise ValueError('Valid final requires separate review, not discard')
    base.require(str(request_path) not in {e['request_path'] for e in value['trace']['response_evidence']},
                 'Response already accounted')
    usage = decoded['usage']
    base.require(0 <= usage['completion_tokens'] <= request['max_tokens'] and usage['prompt_tokens'] > 0,
                 'Invalid usage')
    return {'id':value['id'], 'result_sha256':vr.sha(path), 'request_path':str(request_path),
            'usage':usage, 'files':{str(p):vr.sha(p) for p in (path,path.with_name('state.json'),request_path,raw_path,response_path)},
            'status':'failed_content_preserved','new_model_calls':0,'content_modified':False}


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
    plan=base.verify(output);receipt=vr.read(output/RECEIPT)
    base.require(receipt['configuration_sha256']==plan['configuration_sha256'] and
        receipt['plan_sha256']==vr.sha(output/'plan.json') and receipt['code_sha256']==vr.sha(Path(__file__)) and
        receipt['base_sources']==base.source_hashes(),'Recovery binding changed')
    vr.EvidenceCache(output/'native_recovery_cache.json').check(receipt['prefix_files'])


def score(output, plan=None):
    report=base.score(output,plan)
    extras={}
    for arm in base.p.ARMS:
        extras[arm]=[account(p) for p in (output/'formal'/arm).rglob('result.json')
            if not base.runtime.healthy_result(vr.read(p)) and not base.complete_content_failure(vr.read(p))]
        report['usage'][arm]['distinct_actual_requests']+=len({x['request_path'] for x in extras[arm]})
    report['supplemental_failed_decode_usage']=extras
    report['shared_preparation_requests_counted_once_in_batch']+=len({x['request_path'] for v in extras.values() for x in v})
    vr.atomic(output/'metrics.json',report)
    return report


def run(output):
    verify(output)
    original_condition, original_score=base.run_condition,base.score
    def condition(out,row,phase,arm,*args,**kwargs):
        try:return original_condition(out,row,phase,arm,*args,**kwargs)
        except ValueError:
            path=base.path_for(out,phase,arm,row)
            if phase!='formal' or not path.exists():raise
            account(path)
            base.old.emit(out,phase,arm=arm,sample_id=row['id'],result_status='failed_content_preserved')
            return vr.read(path)
    def scoring(out,plan=None):
        with patch.object(base,'score',original_score):return score(out,plan)
    with patch.object(base,'run_condition',condition),patch.object(base,'score',scoring):base.run(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run','verify','score','confirm-detached'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if args.mode=='prepare':prepare(out)
    elif args.mode=='run':run(out)
    else:
        verify(out)
        if args.mode=='score':score(out)
        elif args.mode=='confirm-detached':base.confirm(out)


if __name__=='__main__':main()
