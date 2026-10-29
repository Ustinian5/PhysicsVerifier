"""Offline recovery of the frozen skill probe's JSON-escaping false negative.

No inference, prompt changes, or modification of the frozen runner. The saved
tool payload must round-trip exactly through the actual tokenizer template.
"""
import argparse
import json
from pathlib import Path


def decode_tool_payload(rendered):
    segment = rendered.rsplit('<|start|>', 2)[-2]
    header, body = segment.split('<|message|>', 1)
    if 'read_rules' not in header or not body.endswith('<|end|>'):
        raise ValueError('Unexpected tool message framing')
    value = body[:-len('<|end|>')]
    for _ in range(3):
        if not isinstance(value, str):
            break
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError('Tool payload is not a decoded leaf list')
    return value


def repair(output):
    from scripts import run_rule_skill_comparison as x
    from transformers import AutoTokenizer
    folder = output / 'evaluation'
    f = folder / 'engineering/skills/native_io'
    with x.r.run_lock(output / 'active.lock'), x.r.run_lock(folder / 'active.lock'), x.configured_evaluation():
        plan = x.verify_eval(output)
        c = plan['configuration']
        receipt = f / 'acceptance.json'
        if receipt.exists():
            raise ValueError('Probe already accepted; do not overwrite')
        status = x.vr.read(folder / 'execution_status.json')
        x.require(status.get('state') == 'stopped' and status.get('error') ==
                  'Actual template lost instructions, rule text or exposed tool-call context', 'Unexpected stop')
        x.require(not (folder / 'formal').exists(), 'Formal evaluation already exists')
        request = x.vr.read(f / 'request.json')
        q, response = request['request'], x.vr.read(f / 'response.json')
        x.require(request['request_sha256'] == x.r.sha256_value(q), 'Request hash mismatch')
        x.require(response['model_ids'] == [x.r.EXPECTED_MODEL] and response['http_status'] == 200
                  and response['finish_reasons'] == ['stop'], 'Response identity/finish mismatch')
        x.require(response['raw_sse_sha256'] == x.vr.sha(f / 'raw.json'), 'Raw response hash mismatch')
        corpus = x.r.Corpus(x.vr.read(Path(c['assets']['skills']['path']) / 'catalog.json'), tree=True)
        rid = corpus.ids[0]
        messages = x.runtime.protocol.messages(x.runtime.CONTROLS[1], rules=True, tree_root=corpus.call(
            'read_tree', {'node_id': 'root', 'note': 'Initial tree context'}, max_chars=500000))
        messages.append({'role': 'user', 'content': f'Engineering IO probe only: call read_rules with rule_ids=["{rid}"]. This is not a physics evaluation.'})
        x.require(q['messages'] == messages and q['tools'] == x.runtime.native_tools(True), 'Original probe input mismatch')
        action = json.loads(response['content'])
        x.require(action['action'] == 'read_rules' and action['arguments']['rule_ids'] == [rid], 'Wrong forced call')
        returned = corpus.call('read_rules', action['arguments'], max_chars=500000)
        call = response['tool_calls'][0]
        messages.extend([{'role': 'assistant', 'content': response['reasoning_content'], 'tool_calls': [call]},
                         {'role': 'tool', 'tool_call_id': call['id'], 'name': 'read_rules', 'content': x.r.canonical(returned)}])
        tokenizer = AutoTokenizer.from_pretrained(x.r.EXPECTED_MODEL, local_files_only=True, trust_remote_code=False)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=q['tools'])
        x.require(x.runtime.protocol.RULES in rendered and rid in rendered and
                  (not response['reasoning_content'] or response['reasoning_content'] in rendered), 'Context actually missing')
        x.require(decode_tool_payload(rendered) == returned, 'Decoded full tool payload changed')
        recovery = {'kind': 'offline_exact_json_roundtrip', 'configuration_sha256': plan['configuration_sha256'],
                    'original_stop': status, 'repair_source_sha256': x.vr.sha(Path(__file__)),
                    'original_artifacts': {n: x.vr.sha(f / n) for n in ('request.json', 'raw.json', 'response.json')},
                    'decoded_full_payload_equal': True, 'new_model_calls': 0, 'human_gold': False}
        x.vr.atomic(f / 'serialization_recovery.json', recovery)
        x.r.write_json(f / 'history.json', {'messages': messages, 'rendered_sha256': x.r.sha256_value(rendered)})
        x.r.write_json(receipt, {'passed': True, 'configuration_sha256': x.r.sha256_value(c),
            'usage': response['usage'], 'physics_accuracy_not_gate': True,
            'artifacts': {n: x.vr.sha(f / n) for n in ('request.json', 'raw.json', 'response.json', 'history.json', 'serialization_recovery.json')}})
        print('Exact full tool-payload round-trip passed; zero new model calls.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    repair(parser.parse_args().output.resolve())
