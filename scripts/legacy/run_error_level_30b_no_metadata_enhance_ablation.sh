#!/usr/bin/env bash
# Back-compat alias: formerly "no metadata enhance ablation"; now default main catalog.
TAG_MAIN_30B="${TAG_MAIN_30B:-e2e_no_metadata_enhance_30b_error}" RUN_MAIN_30B_ONLY=1 \
  exec bash "$(dirname "$0")/run_error_level_30b_no_sym_and_235b_e2e.sh"
