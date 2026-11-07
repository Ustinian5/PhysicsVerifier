"""Pin Harmony's date in this personal server process, without editing vLLM.

Usage: PV_HARMONY_DATE=YYYY-MM-DD python -m scripts.serve_fixed_date_vllm <vLLM args>
"""
import datetime
import functools
import inspect
import os
import runpy


def pinned_system_message(original, date):
    datetime.date.fromisoformat(date)
    signature = inspect.signature(original)
    if 'start_date' not in signature.parameters:
        raise RuntimeError('Unsupported vLLM Harmony signature')
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        provided = bound.arguments.get('start_date')
        if provided is not None and provided != date:
            raise ValueError('Request date differs from frozen server date')
        bound.arguments['start_date'] = date
        return original(*bound.args, **bound.kwargs)
    return wrapped


def main():
    date = os.environ['PV_HARMONY_DATE']
    from vllm.entrypoints import harmony_utils
    harmony_utils.get_system_message = pinned_system_message(harmony_utils.get_system_message, date)
    # serving_chat imports the symbol directly; pin both references explicitly.
    from vllm.entrypoints.openai import serving_chat
    serving_chat.get_system_message = harmony_utils.get_system_message
    print('Personal Harmony fixed date:', date, flush=True)
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


if __name__ == '__main__':
    main()
