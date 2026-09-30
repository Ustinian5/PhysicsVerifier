"""Start an isolated GPT-OSS Transformers service with corrected final filtering.

The shared Transformers installation is read-only. This launcher patches only
the imported method in this process's memory and exposes final-channel text
without exposing any preceding channel content.
"""
from __future__ import annotations

import hashlib
import ctypes
import inspect
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path


class FinalMessageBoundary:
    """Stop at the end of the first final message, never at analysis end."""

    def __init__(self, final_header_ids, message_end_id):
        self.header = list(final_header_ids)
        self.end_id = message_end_id
        self.tail = []
        self.in_final = False

    def observe(self, token):
        self.tail = (self.tail + [token])[-len(self.header):]
        if self.tail == self.header:
            self.in_final = True
        return self.in_final and token == self.end_id

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[0] != 1:
            raise ValueError("This isolated adapter supports one request at a time")
        return self.observe(input_ids[0, -1].item())


def anonymous_shard_fd() -> int:
    """Conda Python may lack os.memfd_create although Linux libc supports it."""
    libc = ctypes.CDLL(None, use_errno=True)
    create = libc.memfd_create
    create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    create.restype = ctypes.c_int
    fd = create(b"gptoss-weight-shard", 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    os.set_inheritable(fd, False)
    return fd


def build_patch() -> tuple[str, str]:
    from transformers.commands.serving import ServeCommand

    source = textwrap.dedent(inspect.getsource(ServeCommand.generate_chat_completion))
    old_marker = 'cot_trace_end = "<|channel|>final<|message|>"'
    new_marker = 'cot_trace_end = "<|channel|>final"'
    old_filter = '''if filter_cot:
                    if cot_trace_end in results:  # end of reasoning trace observed -> stop filtering
                        filter_cot = False
                        continue
                    else:
                        continue'''
    new_filter = '''if filter_cot:
                    if cot_trace_end not in results:
                        continue
                    final_start = results.find(cot_trace_end) + len(cot_trace_end)
                    message_start = results.find("<|message|>", final_start)
                    if message_start < 0:
                        continue
                    result = results[message_start + len("<|message|>"):]
                    results = result
                    filter_cot = False'''
    if source.count(old_marker) != 1:
        raise RuntimeError("Expected the installed GPT-OSS final-channel marker exactly once")
    if source.count(old_filter) != 1:
        raise RuntimeError("Expected the installed Chat Completions CoT filter block exactly once")
    patched = source.replace(old_marker, new_marker, 1).replace(old_filter, new_filter, 1)
    boundary_anchor = "    def stream_chat_completion(streamer, _request_id):"
    boundary_code = '''    if "gptoss" in model.config.architectures[0].lower():
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([
            FinalMessageBoundary(
                processor.encode("<|channel|>final<|message|>", add_special_tokens=False),
                processor.convert_tokens_to_ids("<|end|>"),
            )
        ])

'''
    if patched.count(boundary_anchor) != 1:
        raise RuntimeError("Expected exactly one Chat Completions stream function")
    patched = patched.replace(boundary_anchor, boundary_code + boundary_anchor, 1)
    terminal_filter = 'result = result.removesuffix("<|return|>")'
    if patched.count(terminal_filter) != 1:
        raise RuntimeError("Expected exactly one GPT-OSS terminal-token filter")
    patched = patched.replace(terminal_filter, terminal_filter + '.removesuffix("<|end|>")', 1)
    if "cot_trace_end = \"<|channel|>final\"" not in patched or old_filter in patched:
        raise RuntimeError("In-memory adapter patch validation failed")
    compile(patched, inspect.getsourcefile(ServeCommand.generate_chat_completion) or "serving.py", "exec")
    return patched, hashlib.sha256(patched.encode("utf-8")).hexdigest()


def main() -> None:
    patched, patch_sha = build_patch()
    if len(sys.argv) == 2 and sys.argv[1] == "--check-patch":
        print(json.dumps({"patched_method_sha256": patch_sha,
                          "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "mode": "check_only"}, sort_keys=True))
        return

    import transformers.commands.serving as serving
    import transformers.modeling_utils as modeling_utils

    # Read each NFS shard sequentially into anonymous process memory before
    # safetensors maps it. No persistent weight copy or shared-package edits.
    # Metadata-only opens stay lazy; only actual tensor-loading opens pass a
    # device argument. Transformers controls the number of concurrent shards.
    original_safe_open = modeling_utils.safe_open

    def local_safe_open(filename, *args, **kwargs):
        model_root = Path("/slow_share/jinjianhan/models/gpt-oss-120b").resolve()
        if not Path(filename).resolve().is_relative_to(model_root) or "device" not in kwargs:
            return original_safe_open(filename, *args, **kwargs)
        fd = anonymous_shard_fd()
        with os.fdopen(fd, "wb") as destination:
            with open(filename, "rb") as source:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
            destination.flush()
            kwargs["backend"] = "mmap"
            # safe_open owns its mapping/descriptor after this fd is closed.
            return original_safe_open(f"/proc/self/fd/{fd}", *args, **kwargs)

    modeling_utils.safe_open = local_safe_open

    namespace = serving.__dict__.copy()
    from transformers import StoppingCriteriaList
    namespace.update(StoppingCriteriaList=StoppingCriteriaList, FinalMessageBoundary=FinalMessageBoundary)
    exec(compile(patched, inspect.getsourcefile(serving.ServeCommand.generate_chat_completion) or "serving.py", "exec"), namespace)
    serving.ServeCommand.generate_chat_completion = namespace["generate_chat_completion"]

    print(json.dumps({"patched_method_sha256": patch_sha,
                      "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "model_storage_backend": "sequential NFS read to anonymous memfd, then mmap",
                      "parallel_loading_workers": os.environ.get("HF_PARALLEL_LOADING_WORKERS", "disabled"),
                      "mode": "service_starting"}, sort_keys=True), flush=True)
    from transformers.commands.transformers_cli import main as transformers_main
    transformers_main()


if __name__ == "__main__":
    main()
