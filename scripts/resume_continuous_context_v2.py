"""Resume frozen v2 with JSON-format case compatibility; no generation retry.

Original inputs/code/results remain immutable. Complete stored responses are
decoded offline; successful outputs retain their original evidence references.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_continuous_context_v2 as runner
from scripts import gpt_oss_harmony_transport as transport

ORIGINAL_DECODE = transport.decode_response


def normalize_messages(parsed):
    messages = []
    for message in parsed.messages:
        recipient = message.recipient
        # Only the declared final serialization format is case insensitive.
        # Tool names/recipients, channels, model identity and terminal stay strict.
        if (message.channel == "final" and isinstance(recipient, str)
                and recipient.lower() == "<|constrain|>json"):
            recipient = "<|constrain|>json"
        messages.append(SimpleNamespace(channel=message.channel, recipient=recipient,
                                        content=message.content))
    return SimpleNamespace(messages=messages, current_content=parsed.current_content)


def decode_response(raw, **kwargs):
    from vllm.entrypoints import harmony_utils
    parse = harmony_utils.parse_output_into_messages
    with patch.object(harmony_utils, "parse_output_into_messages",
                      lambda ids: normalize_messages(parse(ids))):
        return ORIGINAL_DECODE(raw, **kwargs)


def prepare(parent, output):
    r = runner.r
    plan = runner.validate_plan(parent)
    if output.exists():
        raise ValueError("Recovery directory exists; inspect/reuse it, do not overwrite")
    with r.run_lock(parent / "active.lock"):
        if (parent / "runner.pid").exists():
            raise ValueError("Parent runner PID exists; verify it before recovery")
        stopped = json.loads((parent / "execution_status.json").read_text())
        if not stopped["state"].startswith("stopped_"):
            raise ValueError("Parent batch is not stopped")
        successes, failures, recovered = [], [], []
        # Inspect all failed tasks before creating any recovery directory.
        for phase in ("controls", "formal"):
            for path in sorted((parent / phase / "samples").glob("*/*/result.json")):
                item = json.loads(path.read_text())
                runner.audit_result(item)
                if runner.healthy_result(item):
                    successes.append((phase, path, item))
                    continue
                if item.get("failure_reason") != "ValueError: Unregistered actual Harmony recipient":
                    raise ValueError("Unreviewed failure; no automatic model retry")
                state = json.loads((path.parent / "state.json").read_text())
                pending = state["pending"]
                response_path = Path(pending["response_path"])
                record = json.loads(response_path.read_text())
                raw_path = Path(pending["raw_path"])
                request_path = Path(pending["request_path"])
                raw = json.loads(raw_path.read_text())
                request = json.loads(request_path.read_text())["request"]
                decoded = decode_response(raw, model=request["model"], max_tokens=request["max_tokens"],
                                          native_tools=request.get("tools"))
                if decoded.get("tool_calls"):
                    raise ValueError("This recovery accepts complete finals only")
                final = json.loads(decoded["content"])
                if set(final) != {"diagnostics"} or not isinstance(final["diagnostics"], list):
                    raise ValueError("Not a complete diagnostic final")
                failures.append((phase, path, state, pending, record, decoded))
        if not failures:
            raise ValueError("No reviewed parser failure to recover")
        output.mkdir(parents=True)
        shutil.copyfile(parent / "plan.json", output / "plan.json")
        shutil.copyfile(parent / "manifest.json", output / "manifest.json")
        shutil.copytree(parent / "native_io", output / "native_io")
        for phase, path, item in successes:
            target = output / phase / "samples" / path.parent.relative_to(parent / phase / "samples")
            r.write_json(target / "result.json", item)
        for phase, path, state, pending, record, decoded in failures:
            target = output / phase / "samples" / path.parent.relative_to(parent / phase / "samples")
            turn = target / f"turn_{state['turn'] + 1:02d}"
            turn.mkdir(parents=True)
            state = copy.deepcopy(state)
            for key, filename in (("request_path", "request.json"), ("raw_path", "response.sse")):
                shutil.copyfile(pending[key], turn / filename)
                state["pending"][key] = str(turn / filename)
            restored = dict(record, **decoded)
            restored.pop("error", None)
            r.write_json(turn / "response.json", restored)
            state["pending"]["response_path"] = str(turn / "response.json")
            elapsed = max(0, stopped["unix"] - state["started_unix"])
            state["started_unix"] = time.time() - elapsed
            state.pop("last_error", None)
            r.write_json(target / "state.json", state)
            recovered.append({"id": state["sample_id"], "original_failure": str(path),
                "original_failure_sha256": r.sha256_file(path), "elapsed_active_seconds": elapsed,
                "original_response_sha256": r.sha256_file(Path(pending["response_path"])),
                "restored_response_sha256": r.sha256_file(turn / "response.json"),
                "raw_sha256": r.sha256_file(turn / "response.sse"), "new_model_calls": 0})
        r.write_json(output / "recovery.json", {"parent": str(parent),
            "parent_configuration_sha256": plan["configuration_sha256"],
            "entry_sha256": r.sha256_file(Path(__file__)), "created_unix": time.time(),
            "fix": "case-insensitive JSON final serialization marker only",
            "unchanged_successes": [{"path": str(p), "sha256": r.sha256_file(p)} for _, p, _ in successes],
            "offline_restored": recovered, "offline_wait_excluded_from_active_wall_time": True,
            "prompts_model_budgets_inputs_unchanged": True})
    print(json.dumps({"reused_successes": len(successes), "offline_restored": len(failures),
                      "new_model_calls": 0, "output": str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "confirm-detached", "score"))
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("CONDA_PREFIX"):
        parser.error("Use the project conda environment")
    transport.decode_response = decode_response
    output = args.output.resolve()
    if args.mode == "prepare":
        if args.parent is None:
            parser.error("prepare requires --parent")
        prepare(args.parent.resolve(), output)
    else:
        evidence = json.loads((output / "recovery.json").read_text())
        if evidence["entry_sha256"] != runner.r.sha256_file(Path(__file__)):
            raise ValueError("Recovery implementation changed")
        {"run": runner.run_batch, "confirm-detached": runner.confirm_detached,
         "score": runner.score}[args.mode](output)


if __name__ == "__main__":
    main()
