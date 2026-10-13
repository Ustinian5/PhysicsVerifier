"""Recover a complete C4-batch final bearing the diagnostics format label.

Frozen generation code and original failures remain untouched. This accepts only
a final-channel, complete diagnostic JSON; no tools, partials or content changes.
"""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_continuous_context_v3 as execution
from scripts import resume_continuous_context_v2 as previous
from scripts import gpt_oss_harmony_transport as transport


def decode_response(raw, **kwargs):
    from vllm.entrypoints import harmony_utils
    parse = harmony_utils.parse_output_into_messages
    repairs = []

    def normalize(ids):
        parsed = previous.normalize_messages(parse(ids))
        messages = []
        for message in parsed.messages:
            recipient = message.recipient
            if message.channel == "final" and recipient in {"<|constrain|>diagnostics", "<|constrain|>quote_patches"}:
                content = "".join(part.text for part in message.content)
                value = json.loads(content)
                key = recipient.split(">")[-1]
                if (not isinstance(value, dict) or set(value) != {key}
                        or not isinstance(value[key], list)):
                    raise ValueError("Format label without corresponding complete JSON")
                recipient = "<|constrain|>json"
                repairs.append({"type": "complete_diagnostics_final_format_label",
                    "original_recipient": message.recipient,
                    "content_sha256": execution.runner.r.sha256_value(content),
                    "field_text_preserved": True})
            messages.append(SimpleNamespace(channel=message.channel, recipient=recipient,
                                            content=message.content))
        return SimpleNamespace(messages=messages, current_content=parsed.current_content)

    with patch.object(harmony_utils, "parse_output_into_messages", normalize):
        decoded = previous.ORIGINAL_DECODE(raw, **kwargs)
    if not decoded.get("tool_calls"):
        try:
            content, edits = previous.repair_quote_boundary(decoded["content"])
        except (json.JSONDecodeError, ValueError):
            edits = []
        else:
            decoded["content"] = content
        repairs.extend(edits)
    if repairs:
        decoded["serialization_repairs"] = repairs
    return decoded


def prepare(parent, output):
    """Reuse all healthy outputs and restore a stored quote-only response once."""
    runner, r = execution.runner, execution.runner.r
    runner.validate_plan(parent)
    if output.exists():
        raise ValueError("Recovery directory exists; do not overwrite")
    with r.run_lock(parent / "active.lock"):
        if (parent / "runner.pid").exists():
            raise ValueError("Verify parent PID before recovery")
        stopped = json.loads((parent / "execution_status.json").read_text())
        if not stopped["state"].startswith("stopped_"):
            raise ValueError("Parent batch is not stopped")
        successes, failures = [], []
        for phase in ("controls", "formal"):
            for path in sorted((parent / phase / "samples").glob("*/*/result.json")):
                item = json.loads(path.read_text())
                runner.audit_result(item)
                if runner.healthy_result(item):
                    successes.append((phase, path, item))
                    continue
                if (item.get("quote_repair", {}).get("status") != "technical_failure"
                        or item["quote_repair"].get("error") != "ValueError: Unregistered actual Harmony recipient"):
                    raise ValueError("Unreviewed failure; no automatic model retry")
                state = json.loads((path.parent / "state.json").read_text())
                pending = state["pending"]
                raw = json.loads(Path(pending["raw_path"]).read_text())
                request = json.loads(Path(pending["request_path"]).read_text())["request"]
                decoded = decode_response(raw, model=request["model"], max_tokens=request["max_tokens"],
                                          native_tools=request.get("tools"))
                patches = json.loads(decoded["content"])
                if (decoded.get("tool_calls") or set(patches) != {"quote_patches"}
                        or not isinstance(patches["quote_patches"], list) or request.get("tools")):
                    raise ValueError("Not a complete stored quote-only response")
                failures.append((phase, path, state, pending, decoded))
        if not failures:
            raise ValueError("No reviewed quote-response failure")
        output.mkdir(parents=True)
        for name in ("plan.json", "manifest.json"):
            shutil.copyfile(parent / name, output / name)
        shutil.copytree(parent / "native_io", output / "native_io")
        for phase, path, item in successes:
            target = output / phase / "samples" / path.parent.relative_to(parent / phase / "samples")
            r.write_json(target / "result.json", item)
        restored = []
        for phase, path, state, pending, decoded in failures:
            target = output / phase / "samples" / path.parent.relative_to(parent / phase / "samples")
            turn = target / f"turn_{state['turn'] + 1:02d}"
            turn.mkdir(parents=True)
            state = copy.deepcopy(state)
            for key, filename in (("request_path", "request.json"), ("raw_path", "response.sse")):
                shutil.copyfile(pending[key], turn / filename)
                state["pending"][key] = str(turn / filename)
            record = json.loads(Path(pending["response_path"]).read_text())
            record.update(decoded)
            record.pop("error", None)
            r.write_json(turn / "response.json", record)
            state["pending"]["response_path"] = str(turn / "response.json")
            # The failed decode had not consumed this response; engine meters its
            # original usage once on replay. No new generation or repair attempt.
            if any(e["turn"] == state["turn"] + 1 for e in state["response_evidence"]):
                raise ValueError("Unexpected already-metered failed decode")
            state["quote_repair"]["status"] = "pending"
            state["quote_repair"].pop("error", None)
            elapsed = max(0, stopped["unix"] - state["started_unix"])
            state["started_unix"] = time.time() - elapsed
            state.pop("last_error", None)
            shutil.copyfile(path.parent / "first_diagnosis.json", target / "first_diagnosis.json")
            r.write_json(target / "state.json", state)
            restored.append({"id": state["sample_id"], "original_failure": str(path),
                "original_failure_sha256": r.sha256_file(path),
                "original_response_sha256": r.sha256_file(Path(pending["response_path"])),
                "restored_response_sha256": r.sha256_file(turn / "response.json"),
                "raw_sha256": r.sha256_file(turn / "response.sse"),
                "elapsed_active_seconds": elapsed, "new_model_calls": 0})
        r.write_json(output / "recovery.json", {"parent": str(parent),
            "parent_configuration_sha256": runner.validate_plan(parent)["configuration_sha256"],
            "entry_sha256": r.sha256_file(Path(previous.__file__)),
            "recovery_entry_sha256": r.sha256_file(Path(__file__)),
            "unchanged_successes": [{"path": str(p), "sha256": r.sha256_file(p)} for _, p, _ in successes],
            "offline_restored": restored, "new_model_calls": 0,
            "prompts_model_budgets_inputs_unchanged": True,
            "offline_wait_excluded_from_active_wall_time": True})
    print(json.dumps({"reused_successes": len(successes), "offline_restored": len(failures),
                      "new_model_calls": 0, "output": str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "confirm-detached", "score"))
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    with execution.configured(), patch.object(previous, "decode_response", decode_response), \
            patch.object(transport, "decode_response", decode_response):
        runner = execution.runner
        if args.mode == "prepare":
            if args.parent is None:
                parser.error("prepare requires --parent")
            parent = args.parent.resolve()
            quote_failure = any(json.loads(p.read_text()).get("quote_repair", {}).get("status") == "technical_failure"
                for p in (parent / "formal/samples").glob("*/*/result.json"))
            (prepare if quote_failure else previous.prepare)(parent, output)
            evidence = json.loads((output / "recovery.json").read_text())
            evidence.update(recovery_entry_sha256=runner.r.sha256_file(Path(__file__)),
                fix="Complete final diagnostics format label plus prior deterministic serialization compatibility")
            runner.r.write_json(output / "recovery.json", evidence)
        else:
            evidence = json.loads((output / "recovery.json").read_text())
            if (evidence["recovery_entry_sha256"] != runner.r.sha256_file(Path(__file__))
                    or evidence["entry_sha256"] != runner.r.sha256_file(Path(previous.__file__))):
                raise ValueError("Recovery implementation changed")
            {"run": runner.run_batch, "confirm-detached": runner.confirm_detached,
             "score": runner.score}[args.mode](output)


if __name__ == "__main__":
    main()
