#!/usr/bin/env bash
# Put this job's temp files on /slow_share/jinjianhan so a full /tmp cannot block training.
# Keep a short /tmp symlink for Ray AF_UNIX socket path length (<=107 bytes).
# Source this file; do not exec it.
SLOW_TMP_ROOT="${SLOW_TMP_ROOT:-/slow_share/jinjianhan/tmp}"
RAY_GCS_PORT="${RAY_GCS_PORT:-26379}"
RAY_TMP_REAL="${RAY_TMP_REAL:-${SLOW_TMP_ROOT}/orhf8b_ray_${RAY_GCS_PORT}}"
RAY_TMP_LINK="${RAY_TMP_LINK:-/tmp/orhf8b_ray_${RAY_GCS_PORT}}"
PY_TMP="${PY_TMP:-${SLOW_TMP_ROOT}/py}"
MIN_SLOW_TMP_GB="${MIN_SLOW_TMP_GB:-20}"

mkdir -p "${RAY_TMP_REAL}" "${PY_TMP}" || {
  echo "[error] cannot mkdir ${RAY_TMP_REAL} or ${PY_TMP}" >&2
  return 2 2>/dev/null || exit 2
}

if [[ -L "${RAY_TMP_LINK}" ]]; then
  :
elif [[ -e "${RAY_TMP_LINK}" ]]; then
  rm -rf "${RAY_TMP_LINK}"
fi
ln -sfn "${RAY_TMP_REAL}" "${RAY_TMP_LINK}"

export RAY_TMPDIR="${RAY_TMP_LINK}"
export RAY_TMPDIR_REAL="${RAY_TMP_REAL}"
export TMPDIR="${PY_TMP}"
export TEMP="${PY_TMP}"
export TMP="${PY_TMP}"

slow_avail_gb="$(df -Pk "${SLOW_TMP_ROOT}" | awk 'NR==2 {printf "%.1f", $4/1024/1024}')"
if ! python3 -c "import sys; sys.exit(0 if float('${slow_avail_gb}') >= float('${MIN_SLOW_TMP_GB}') else 1)"; then
  echo "[error] ${SLOW_TMP_ROOT} has ${slow_avail_gb}GB free; need >= ${MIN_SLOW_TMP_GB}GB" >&2
  return 2 2>/dev/null || exit 2
fi
echo "[tmp] RAY_TMPDIR=${RAY_TMPDIR} -> ${RAY_TMP_REAL}  TMPDIR=${TMPDIR}  slow_share_free=${slow_avail_gb}GB"