"""Personal process adapter: preserve assistant analysis before native tool calls.

The pinned vLLM Harmony chat parser otherwise ignores assistant.content whenever
tool_calls is present. No installed library, model, sampler or teacher is changed.
"""
import functools


def preserve_tool_analysis(original, make_analysis):
    @functools.wraps(original)
    def wrapped(chat_message):
        message = chat_message if isinstance(chat_message,dict) else chat_message.model_dump(exclude_none=True)
        result = original(chat_message)
        if message.get('role')=='assistant' and message.get('tool_calls'):
            content = message.get('content') or ''
            if not isinstance(content,str):
                raise ValueError('Assistant tool analysis must be text')
            if content:
                return [make_analysis(content)] + result
        return result
    return wrapped


def main():
    from openai_harmony import Message, Role
    from vllm.entrypoints import harmony_utils
    from vllm.entrypoints.openai import serving_chat
    parser = preserve_tool_analysis(harmony_utils.parse_input_to_harmony_message,
        lambda text:Message.from_role_and_content(Role.ASSISTANT,text).with_channel('analysis'))
    harmony_utils.parse_input_to_harmony_message=parser
    serving_chat.parse_input_to_harmony_message=parser
    print('Personal Harmony adapter: preserve assistant tool-call analysis',flush=True)
    from scripts.serve_fixed_date_vllm import main as serve
    serve()


if __name__=='__main__':main()
