#!/usr/bin/env bash
# Error-level main pipeline: 30B + default catalog (no LLM metadata enhancement).
RUN_MAIN_30B_ONLY=1 exec bash "$(dirname "$0")/run_error_level_30b_no_sym_and_235b_e2e.sh"
