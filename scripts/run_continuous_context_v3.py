"""Run the C0p/C4 full-50 comparison using the existing continuous executor."""
import argparse
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from scripts import run_continuous_context_v2 as runner
from scripts import continuous_context_v3 as protocol
from scripts import gpt_oss_harmony_transport as transport
from scripts import resume_continuous_context_v2 as serialization


@contextmanager
def configured():
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner, "protocol", protocol))
        stack.enter_context(patch.object(runner, "LIMITS", {**runner.LIMITS, "review_protocol": protocol.PROTOCOL}))
        stack.enter_context(patch.object(runner, "SOURCE_FILES", [*runner.SOURCE_FILES,
            "scripts/continuous_context_v3.py", "scripts/run_continuous_context_v3.py",
            "scripts/resume_continuous_context_v2.py"]))
        # Previously validated deterministic decoding repairs; raw output and
        # token accounting remain intact, with repairs in response metadata.
        stack.enter_context(patch.object(transport, "decode_response", serialization.decode_response))
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "confirm-detached", "score"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with configured():
        {"prepare": runner.prepare, "run": runner.run_batch,
         "confirm-detached": runner.confirm_detached, "score": runner.score}[args.mode](args.output.resolve())


if __name__ == "__main__":
    main()
