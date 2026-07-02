#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export LC_ALL=C

mkdir -p "${REPO_ROOT}/coverage"
set +e
REPORT_ROOT="${REPO_ROOT}/coverage" \
  COBERTURA_XML="${REPO_ROOT}/coverage/python-cobertura.xml" \
  SUMMARY_TXT="${REPO_ROOT}/coverage/python-coverage.txt" \
  HTML_DIR="${REPO_ROOT}/coverage/python-html" \
  "${REPO_ROOT}/tools/coverage/python_coverage.sh" -v --ignore=tests/builder/test_cli.py
cov_rc=$?
set -e

if [[ ! -f "${REPO_ROOT}/coverage/python-coverage.txt" || ! -f "${REPO_ROOT}/coverage/python-cobertura.xml" ]]; then
  if [[ "${cov_rc}" -ne 0 ]]; then
    exit "${cov_rc}"
  fi
  echo "ERROR: Python coverage artifacts are missing under ${REPO_ROOT}/coverage" >&2
  exit 2
fi

line_rate="$(
  sed -nE 's/.*line-rate=\"([0-9]+(\.[0-9]+)?)\".*/\1/p' "${REPO_ROOT}/coverage/python-cobertura.xml" | head -n1
)"
branch_rate="$(
  sed -nE 's/.*branch-rate=\"([0-9]+(\.[0-9]+)?)\".*/\1/p' "${REPO_ROOT}/coverage/python-cobertura.xml" | head -n1
)"

if [[ -z "${line_rate}" ]]; then
  echo "ERROR: Failed to parse Python line-rate from coverage/python-cobertura.xml" >&2
  exit 2
fi
line_pct="$(awk -v r="${line_rate}" 'BEGIN { printf "%.2f", r * 100.0 }')"
if [[ -n "${branch_rate}" ]]; then
  branch_pct="$(awk -v r="${branch_rate}" 'BEGIN { printf "%.2f", r * 100.0 }')"
fi

echo "PYTHON_COVERAGE_LINE=${line_pct}%"
if [[ -n "${branch_pct}" ]]; then
  echo "PYTHON_COVERAGE_BRANCH=${branch_pct}%"
fi
exit "${cov_rc}"
