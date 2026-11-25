"""Resume the matched step trial after a verified CHECK output-token exhaustion."""
import argparse
from pathlib import Path
from unittest.mock import patch

from scripts import run_step_retrieval_comparison as trial

f, vr, require = trial.f, trial.vr, trial.require
native = f.step.old.a.native
prior_inspector = native.inspect_failure
TRUNCATED = 'ValueError: Incomplete model output'


def account_length(path, signature):
    row = vr.read(path)
    if row.get('failure_reason') != TRUNCATED:
        return prior_inspector(path)
    require(row.get('status') == 'failed' and row.get('diagnostics') == []
            and row['configuration_sha256'] == signature, 'Unexpected step failure')
    f.base.runtime.audit_result(row)
    state_path = path.with_name('state.json')
    state = vr.read(state_path)
    turn = path.parent/'turn_02'
    files = [turn/n for n in ('request.json','response.sse','response.json')]
    envelope, raw, response = map(vr.read, files)
    body = envelope['request']
    require(state['sample_id']==row['id'] and state['turn']==1 and state['configuration_sha256']==signature
            and [state['pending'][k] for k in ('request_path','raw_path','response_path')]==list(map(str,files))
            and envelope['request_sha256']==f.base.r.sha256_value(body)
            and body['messages']==vr.read(path.with_name('reference_packet.json'))['packet']['messages']
            and body['max_tokens']==state['pending']['max_tokens'] and 0<body['max_tokens']<=16384
            and body['temperature']==0 and body['top_p']==1 and body['reasoning_effort']=='medium'
            and 'tools' not in body and 'vllm_xargs' not in body, 'Unbound step CHECK')
    model=f.base.r.EXPECTED_MODEL
    usage=raw['usage']
    require(body['model']==raw['model']==model and response.get('model_ids')==[model]
            and response.get('http_status')==200 and response.get('error')==TRUNCATED
            and response['raw_sse_sha256']==vr.sha(files[1]) and len(raw['choices'])==1
            and raw['choices'][0]['finish_reason']=='length' and response['usage']==usage
            and usage['completion_tokens']==len(raw['choices'][0]['token_ids'])==body['max_tokens']
            and usage['prompt_tokens']==len(raw['prompt_token_ids'])
            and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
            and str(files[0]) not in {e['request_path'] for e in row['trace']['response_evidence']},
            'Missing raw proof of output-token exhaustion')
    try:
        f.step.old.a.historical_transport().decode_response(raw,model=model,max_tokens=body['max_tokens'],native_tools=None)
    except ValueError as exc:
        require('ValueError: '+str(exc)==TRUNCATED, 'Different frozen decoding failure')
    else:
        raise ValueError('Frozen decoder accepted the response')
    return dict(id=row['id'], result_sha256=vr.sha(path), request_path=str(files[0]), usage=usage,
                files={str(p):vr.sha(p) for p in [path,state_path,path.with_name('reference_packet.json'),*files]},
                status='failed_content_preserved',failure_reason=TRUNCATED,new_model_calls=0,content_modified=False)


def resume(folder,arm,signature):
    previous=folder.parent/str(int(folder.name)-1)/'exit.json'
    require(vr.read(previous)==dict(code=1,error='Not an audited output failure'), 'Unexpected prior exit')
    audited={}
    for mode in trial.MODES:
        for path in (trial.ROOT/mode/'formal'/arm).glob('samples/C0p/*/result.json'):
            if vr.read(path).get('failure_reason')==TRUNCATED:
                audited[str(path)]=account_length(path,signature)
    require(bool(audited), 'No verified truncated output to resume')
    trial.write(folder/'recovery.json',dict(design_sha256=vr.sha(trial.ROOT/'design.json'),
                recovery_code_sha256=vr.sha(__file__), prior_exit_sha256=vr.sha(previous),failures=audited))
    with patch.object(native,'inspect_failure',lambda p:account_length(p,signature)):
        trial.run(folder,arm,signature)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','confirm'))
    parser.add_argument('arm',choices=trial.ARMS)
    parser.add_argument('--attempt',type=int,required=True)
    args=parser.parse_args()
    trial.t.b.node.node()
    stamp=trial.design()
    trial.t.runtime.detached(trial.ROOT/args.arm,args.attempt,stamp,
                             lambda folder:resume(folder,args.arm,stamp),args.action=='confirm')
