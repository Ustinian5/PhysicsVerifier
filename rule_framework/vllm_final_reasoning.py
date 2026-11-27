"""vLLM 0.11.2 plugin to start JSON constraints only at the current final body."""
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser

from rule_framework.final_output_format import final_header_ended


@ReasoningParserManager.register_module('pv_final_harmony')
class VerifierFinalReasoning(GptOssReasoningParser):
    def is_reasoning_end(self, input_ids):
        return final_header_ended(input_ids)
