"""Request-scoped ranking grammar for the project's pinned vLLM V1 service."""
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from rule_framework.rank_output import END, FIRST_SPECIAL, RankGrammar, RankShape, number_tokens


class RankedIDsProcessor(AdapterLogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)
        self.numbers = number_tokens(cached_tokenizer_from_config(vllm_config.model_config))

    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, params):
        args = params.extra_args or {}
        if 'pv_rank_count' not in args:
            return None
        grammar = RankGrammar(RankShape.from_wire(args), self.numbers, params.max_tokens)

        def apply(output_ids, logits):
            allowed = grammar.allowed(output_ids)
            if allowed is None:
                logits[FIRST_SPECIAL:END].fill_(float('-inf'))
                logits[END+1:].fill_(float('-inf'))
            elif len(allowed) == 1:
                logits.fill_(float('-inf'))
                logits[allowed[0]] = 0
            else:
                ids = list(allowed)
                scores = logits[ids]
                logits.fill_(float('-inf'))
                logits[ids] = scores
            return logits

        return apply
