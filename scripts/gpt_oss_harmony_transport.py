"""Metered vLLM transport using actual Harmony recipients and terminal tokens.

Consolidates the first batch's recipient and constrained-final recovery fixes.
No retry and no reconstruction of tool calls from reasoning text.
"""
from __future__ import annotations

import hashlib
import json
import time

import httpx

from scripts import run_gpt_oss_selfcheck as r


def decode_response(raw, *, model, max_tokens, native_tools):
    if raw.get("model") != model or len(raw.get("choices", [])) != 1:
        raise ValueError("Model identity or choice count mismatch")
    choice = raw["choices"][0]
    ids = choice.get("token_ids")
    used = (raw.get("usage") or {}).get("completion_tokens")
    prompt_used = (raw.get("usage") or {}).get("prompt_tokens")
    if (type(used) is not int or not 0 < used <= max_tokens or
            not isinstance(ids, list) or len(ids) != used or
            type(prompt_used) is not int or prompt_used < 0):
        raise ValueError("Missing or inconsistent token usage")
    if choice.get("finish_reason") not in {"stop", "tool_calls"}:
        raise ValueError("Incomplete model output")
    terminal = ids[-1]
    if terminal not in {200002, 200012}:
        raise ValueError("No authoritative final/tool termination")
    declared_stop = choice.get("stop_reason")
    if declared_stop is not None and declared_stop != terminal:
        raise ValueError("Contradictory stop metadata")
    from vllm.entrypoints.harmony_utils import parse_output_into_messages
    parsed = parse_output_into_messages(ids)
    if parsed.current_content:
        raise ValueError("Incomplete Harmony message")
    reasoning, finals, calls = [], [], []
    names = {t["function"]["name"] for t in native_tools or []}
    for message in parsed.messages:
        text = "".join(part.text for part in message.content)
        recipient = message.recipient
        if message.channel == "final" and recipient in {None, "", "<|constrain|>json"}:
            finals.append(text)
        elif recipient:
            if (message.channel not in {"analysis", "commentary"} or
                    not recipient.startswith("functions.") or recipient[10:] not in names):
                raise ValueError("Unregistered actual Harmony recipient")
            arguments = json.loads(text)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            calls.append({"id": "call_" + hashlib.sha256((recipient + text).encode()).hexdigest()[:24],
                          "type": "function", "function": {"name": recipient[10:], "arguments": text}})
        elif message.channel == "analysis":
            reasoning.append(text)
        else:
            raise ValueError("Unexpected unrouted Harmony message")
    if calls:
        if len(calls) != 1 or finals or terminal != 200012:
            raise ValueError("Ambiguous tool yield")
        content = r.canonical({"action": calls[0]["function"]["name"],
                               "arguments": json.loads(calls[0]["function"]["arguments"])})
    else:
        if len(finals) != 1 or terminal != 200002:
            raise ValueError("Missing final message")
        content = finals[0]
    result = {"content": content, "reasoning_content": "\n".join(reasoning),
              "model_ids": [model], "usage": raw["usage"],
              "finish_reasons": [choice["finish_reason"]], "terminal_token": terminal}
    if calls:
        result["tool_calls"] = calls
    return result


def chat_response(client, *, endpoint, model, messages, max_tokens, timeout,
                  raw_path, response_path, request_path, backend="vllm", native_tools=None,
                  force_tool=None):
    if backend != "vllm":
        raise ValueError("Harmony transport requires the pinned vLLM service")
    r.assert_no_control_tokens(messages)
    body = r.request_body(model, messages, max_tokens, backend, native_tools)
    body.update(stream=False, return_token_ids=True, reasoning_effort="medium")
    body.pop("stream_options", None)
    if force_tool is not None:
        if force_tool not in {t["function"]["name"] for t in native_tools or []}:
            raise ValueError("Unregistered engineering probe tool")
        body["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
    r.write_json(request_path, {"request": body, "request_sha256": r.sha256_value(body)})
    started = time.time()
    record = {"transport": "raw_harmony_v2", "model_ids": [], "usage": None}
    try:
        response = client.post(endpoint, json=body, timeout=httpx.Timeout(timeout, connect=20))
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(response.text, encoding="utf-8")
        record.update(http_status=response.status_code, raw_sse_sha256=r.sha256_file(raw_path))
        response.raise_for_status()
        raw = response.json()
        record.update(model_ids=[raw.get("model")], usage=raw.get("usage"))
        record.update(decode_response(raw, model=model, max_tokens=max_tokens, native_tools=native_tools))
        return record
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
        raise
    finally:
        record["elapsed_seconds"] = time.time() - started
        r.write_json(response_path, record)
