"""Persist Responses SSE evidence, including HTTP and in-stream failures."""
import json
import time

import httpx


def sse_payloads(lines):
    data = []
    for line in lines:
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


def read_response_stream(response, record, path, *, started, timeout, save):
    record.update(http_status=response.status_code,
                  request_id=response.headers.get("x-request-id"))
    if response.headers.get("retry-after"):
        record["retry_after"] = response.headers["retry-after"]
    if response.status_code >= 400:
        # raise_for_status() alone discards the only evidence distinguishing
        # overload, rate limits, invalid requests and exhausted credit.
        response.read()
        try:
            record["response"] = response.json()
        except ValueError:
            record["response_text"] = response.text[:10000]
        save(path, record)
        response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if content_type and "text/event-stream" not in content_type:
        # Some gateways buffer Responses into an ordinary JSON body. Preserve
        # and use a complete response instead of issuing another generation.
        response.read()
        try:
            data = response.json()
        except ValueError:
            record["response_text"] = response.text[:10000]
            save(path, record)
            raise ValueError("Non-SSE response has no valid JSON; raw body preserved")
        record["response"] = data
        save(path, record)
        if isinstance(data, dict) and data.get("status") in {"completed", "incomplete", "failed"}:
            return data
        raise ValueError("Non-SSE response is not a Responses result; raw body preserved")
    events_path = path.with_suffix(".events.jsonl")
    record["stream_events"] = events_path.name
    save(path, record)
    with events_path.open("w") as events:
        for raw in sse_payloads(response.iter_lines()):
            if time.monotonic() - started > timeout:
                raise httpx.ReadTimeout("stream total time limit exceeded")
            if raw.strip() == "[DONE]":
                break
            try:
                event = json.loads(raw)
                if not isinstance(event, dict):
                    raise ValueError("SSE event must be an object")
            except ValueError:
                events.write(json.dumps({"type": "client.sse_parse_error", "raw_data": raw}) + "\n")
                events.flush()
                raise ValueError("Malformed SSE event; raw data preserved")
            events.write(json.dumps(event, ensure_ascii=False) + "\n")
            events.flush()
            record["last_stream_event"] = event.get("type")
            record["last_stream_event_unix"] = time.time()
            record["stream_event_count"] = record.get("stream_event_count", 0) + 1
            data = event.get("response")
            if isinstance(data, dict) and data.get("id"):
                record["stream_response_id"] = data["id"]
            if event.get("type") == "error":
                record["stream_error"] = event.get("error", event)
                save(path, record)
                raise RuntimeError("Provider returned an in-stream error; raw event preserved")
            if event.get("type") in {"response.completed", "response.incomplete", "response.failed"}:
                if not isinstance(data, dict):
                    raise ValueError("Terminal SSE event lacks a response object")
                record["response"] = data
                save(path, record)
                return data
            # Progress survives a killed process, even before the final event.
            save(path, record)
    raise RuntimeError("stream ended without terminal response; no automatic retry")
