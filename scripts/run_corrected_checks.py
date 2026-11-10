"""Durable bounded native-tool conversations, used by corrected_pipeline only."""
import json
from pathlib import Path
import threading
import time

from rule_framework import corrected_verifier as p
from rule_framework import verified_resume as vr
from scripts import run_corrected_collection as c
from scripts import run_corrected_extraction as extraction
from scripts.run_qwen30b_step_checks import validate_effective_input

require = c.require


def parsed(text):
    if text.startswith('```json\n') and text.endswith('```'):
        text = text[8:-3].strip()
    escaped, controls = extraction.old.escape_json_controls(text)
    value, _, positions = extraction.old.b.parse_json_escapes(escaped, repair=True)
    return value, dict(literal_control_positions=controls, literal_backslash_positions=positions)


def rendered(tokenizer, messages, native):
    value = tokenizer.apply_chat_template(messages, tools=native, tokenize=False,
        add_generation_prompt=True, reasoning_effort='medium')
    # Check the complete user input, tool returns and exposed pre-tool reasoning.
    # The pinned template may discard pre-final analysis during a quote-only patch;
    # the first diagnosis is already frozen, and no physics reconsideration is requested.
    for message in messages:
        if message['role'] in {'system','user','tool'} or 'tool_calls' in message:
            text = message.get('content') or ''
            if 'tool_calls' in message and any(m['role']=='assistant' and 'tool_calls' not in m
                                               for m in messages[messages.index(message)+1:]):
                continue
            require(not text or text in value, 'Template dropped visible context')
    return value


def token_count(tokenizer, messages, native):
    return len(tokenizer.encode(rendered(tokenizer, messages, native), add_special_tokens=False))


def run_case(folder, row, arm, task, memory, configuration, tokenizer):
    folder = Path(folder)
    signature = vr.digest([row, arm, task, configuration])
    result_file = folder/'result.json'
    if result_file.exists():
        saved = vr.read(result_file)
        require(saved['signature']==signature, 'Saved check input/configuration drift')
        for ev in saved['evidence']:
            require(vr.sha(folder/ev['file'])==ev['sha256'], 'Saved check evidence drift')
        return saved
    vr.atomic(folder/'input.json', dict(row=row, arm=arm, task=task, signature=signature))
    start_path = folder/'started.json'
    if not start_path.exists():
        vr.atomic(start_path, dict(unix=time.time(), signature=signature))
    started = vr.read(start_path)
    require(started['signature']==signature, 'Start binding drift')
    # A failed HTTP request may have consumed the whole advertised completion
    # allowance. One unknown-usage request exhausts the case conservatively;
    # it is not resampled under a fresh budget. Construction uses its own policy.
    cfg = dict(configuration['transport'], technical_attempts=1)
    history = p.messages(row, arm, task)
    knowledge = p.Knowledge(arm, memory)
    native = p.tools() if arm!='pure' else None
    used_in = used_out = 0
    evidence, input_audits, serializations = [], [], []
    first = value = None
    patch_pending = False
    quote_repair = None
    status, reason = 'budget_exhausted', 'No final output within common limits'
    turns = 0
    for turn in range(p.LIMITS['responses']):
        remaining = p.LIMITS['output_tokens'] - used_out
        active_tools = None if patch_pending else native
        estimated = token_count(tokenizer, history, active_tools)
        key = f'turn_{turn:02d}'
        prior = list((folder/'raw'/'check'/c.digest(key)[:24]).glob('request_*.json'))
        # Reconstruct the immutable successful prefix before enforcing elapsed
        # time for a new request. Never lose a final response after process death.
        if not prior and (remaining<=0 or used_in+estimated>p.LIMITS['input_tokens'] or
                estimated+remaining>p.LIMITS['context_tokens'] or
                time.time()-started['unix']>=p.LIMITS['seconds']):
            break
        require(remaining>0, 'Recorded response exceeds output budget')
        timeout_file=folder/'request_limits'/(key+'.json')
        if not timeout_file.exists():
            vr.atomic(timeout_file,dict(timeout_seconds=min(p.LIMITS['request_seconds'],
                max(1,int(p.LIMITS['seconds']-(time.time()-started['unix']))))))
        # A restored request uses its original transport limits/fingerprint.
        turn_cfg=dict(cfg,**vr.read(timeout_file))
        journal=c.corpus.Journal(folder,turn_cfg,threading.Event())
        body = c.corpus.r.request_body(cfg['model'], history, remaining, backend='vllm', native_tools=active_tools)
        body.update(stream=False, return_token_ids=True, reasoning_effort='medium')
        body.pop('stream_options',None)
        try:
            raw, ev = journal.call('check', key, body)
        except RuntimeError as exc:
            if str(exc) != 'Technical attempt cap reached; preserve unknown usage and failures':
                raise
            paths = sorted((folder/'raw'/'check'/c.digest(key)[:24]).glob('request_*.json'))
            evidence.extend(dict(file=str(x.relative_to(folder)),sha256=vr.sha(x)) for x in paths)
            used_out += remaining
            used_in += estimated
            status, reason = 'technical_failure', str(exc)
            turns += len(paths)
            break
        evidence.append(ev); turns += 1
        effective = validate_effective_input(raw,tokenizer,cfg['fixed_date'])
        actual = tokenizer.decode(raw['prompt_token_ids'], skip_special_tokens=False)
        # Compare all model-visible data in the authoritative deployed prompt.
        # Header date is validated separately; no local date is trusted.
        for m in history:
            if m['role'] in {'user','tool'}:
                require((m.get('content') or '') in actual, 'Service lost full user/tool context')
            if 'tool_calls' in m and not patch_pending:
                require((m.get('content') or '') in actual, 'Service lost prior tool-call analysis')
        require(history[0]['content'] in actual, 'Service lost task instructions')
        input_audits.append(effective)
        usage = raw.get('usage') or {}
        prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
        require(type(prompt) is int and type(completion) is int and 0<completion<=remaining,
                'Missing/invalid actual token usage')
        require(prompt==effective.get('token_count',prompt),'Actual input-token usage mismatch')
        require(prompt+remaining<=p.LIMITS['context_tokens'] and used_in+prompt<=p.LIMITS['input_tokens'],
                'Actual prompt exceeds common budget')
        used_in += prompt; used_out += completion
        try:
            decoded = c.corpus.harmony.decode_response(raw, model=cfg['model'],max_tokens=remaining,native_tools=active_tools)
        except ValueError as exc:
            if str(exc) not in {'Incomplete model output','No authoritative final/tool termination',
                                'Incomplete Harmony message','Missing final message'}:
                raise  # Unknown protocol/identity failures are global stops.
            status, reason = 'content_failure', str(exc)
            break
        if decoded.get('tool_calls'):
            call = decoded['tool_calls'][0]
            name = call['function']['name']; args = json.loads(call['function']['arguments'])
            assistant = p.assistant_tool_message(decoded)
            try:
                returned = knowledge.preview(name,args)
            except ValueError as exc:
                returned = dict(error=str(exc), full_item_not_returned=True)
            note = dict(responses_left=p.LIMITS['responses']-turns,
                        generated_tokens_left=p.LIMITS['output_tokens']-used_out)
            def tool_message(v):
                return dict(role='tool', tool_call_id=call['id'], name=name,
                            content=json.dumps(dict(result=v,budget=note),ensure_ascii=False).replace('<|','\\u003c|'))
            tentative = history+[assistant,tool_message(returned)]
            size = len(json.dumps(returned,ensure_ascii=False))
            next_input = token_count(tokenizer,tentative,native)
            if knowledge.returned_chars+size>p.LIMITS['tool_chars'] or next_input+(
                    p.LIMITS['output_tokens']-used_out)>p.LIMITS['context_tokens'] or used_in+next_input>p.LIMITS['input_tokens']:
                returned = dict(error='Complete item/page exceeds remaining material/context/input budget',
                                full_item_not_returned=True, item_chars=size)
            # A tiny remaining character budget cannot fit even an error response.
            if knowledge.returned_chars+len(json.dumps(returned,ensure_ascii=False))>p.LIMITS['tool_chars']:
                status,reason='budget_exhausted','Tool character budget exhausted'
                break
            knowledge.commit(name,args,returned)
            history.extend([assistant,tool_message(returned)])
            vr.atomic(folder/'conversation.json',history)
            continue
        history.append(dict(role='assistant',content=decoded['content']))
        try:
            draft, edits = parsed(decoded['content'].strip()); serializations.append(edits)
            if patch_pending:
                value,quote_repair = p.step.apply_quote_patches(first,draft,row)
                status,reason='complete',None
                break
            if task == 'final':
                value=p.final.validate_verdict(draft); status,reason='complete',None
                break
            require(isinstance(draft,dict) and set(draft)=={'diagnostics'},'Expected diagnostics object')
            c.corpus.r.validate_diagnostics(draft)
            require(all(isinstance(d.get('rule_refs',[]),list) and
                        all(isinstance(x,str) for x in d.get('rule_refs',[])) for d in draft['diagnostics']),
                    'Invalid citation IDs')
            first=value=draft
            vr.atomic(folder/'first_diagnosis.json',first)
            bad=p.step.invalid_quotes(value,row)
            if bad:
                patch_pending=True
                history.append(dict(role='user',content=p.step.repair_message(first,bad)))
                quote_repair=dict(status='pending_or_budget_exhausted',invalid_indices=bad)
                continue
            status,reason='complete',None
            break
        except (ValueError,TypeError,KeyError) as exc:
            status,reason='content_failure',str(exc)
            break
    if first is not None:
        # Quote repair failure never erases or regenerates published diagnoses.
        value = value or first
        if status!='complete':
            quote_repair=dict(status='failed_or_budget_exhausted_original_preserved',reason=reason)
            status,reason='complete',None
    audit = knowledge.audit()
    if task=='step' and value:
        audit['quote_invalid_indices']=p.step.invalid_quotes(value,row)
        audit['unread_citations']=[rid for d in value['diagnostics'] for rid in d.get('rule_refs',[]) if rid not in knowledge.read_ids]
        audit['protocol_noncompliance']=bool(audit['protocol_noncompliance'] or audit['unread_citations'])
    result=dict(id=row['id'], arm=arm, task=task, signature=signature, status=status, reason=reason,
        value=value, evidence=evidence, input_audits=input_audits, usage=dict(requests=turns,input_tokens=used_in,
        output_tokens_or_unknown_reservation=used_out), protocol_audit=audit, quote_repair=quote_repair,
        tool_events=knowledge.events, serialization_repairs=serializations, human_gold=False)
    vr.atomic(folder/'conversation.json',history)
    vr.atomic(result_file,result)
    return result
