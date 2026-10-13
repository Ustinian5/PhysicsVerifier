"""Recover a complete C4-batch final bearing the diagnostics format label.

Frozen generation code and original failures remain untouched. This accepts only
a final-channel, complete diagnostic JSON; no tools, partials or content changes.
"""
import argparse
import json
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
            if message.channel == "final" and recipient == "<|constrain|>diagnostics":
                content = "".join(part.text for part in message.content)
                value = json.loads(content)
                if (not isinstance(value, dict) or set(value) != {"diagnostics"}
                        or not isinstance(value["diagnostics"], list)):
                    raise ValueError("Diagnostics format label without complete diagnostics JSON")
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
            previous.prepare(args.parent.resolve(), output)
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
