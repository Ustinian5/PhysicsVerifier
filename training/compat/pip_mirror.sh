#!/usr/bin/env bash
# Default PyPI mirror for slow/unreliable upstream connectivity.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORKSPACE_ROOT}/.cache/pip}"
export TMPDIR="${TMPDIR:-${WORKSPACE_ROOT}/tmp}"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

pip_install() {
  local pip_bin="${1:?pip binary required}"
  shift
  "${pip_bin}" install -i "${PIP_INDEX_URL}" "$@"
}
