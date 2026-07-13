#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
coverage_dir="${COVERAGE_DIR:-${REPO_ROOT}/coverage}"
export LC_ALL=C

mkdir -p "${coverage_dir}"

set +e
REPORT_ROOT="${coverage_dir}" \
  COBERTURA_XML="${coverage_dir}/cpp-cobertura.xml" \
  SUMMARY_TXT="${coverage_dir}/cpp-coverage-summary.txt" \
  HTML_REPORT="${coverage_dir}/cpp-coverage.html" \
  GATE_LOG="${coverage_dir}/cpp-gate.log" \
  "${REPO_ROOT}/tools/coverage/cpp_coverage.sh" "$@"
cov_rc=$?
set -e

if [[ ! -f "${coverage_dir}/cpp-coverage-summary.txt" ]]; then
  if [[ "${cov_rc}" -ne 0 ]]; then
    exit "${cov_rc}"
  fi
  echo "ERROR: C++ coverage summary is missing at ${coverage_dir}/cpp-coverage-summary.txt" >&2
  exit 2
fi

line_pct="$(sed -nE 's/^lines:[[:space:]]*([0-9]+(\.[0-9]+)?)%.*/\1/p' "${coverage_dir}/cpp-coverage-summary.txt")"
function_pct="$(sed -nE 's/^functions:[[:space:]]*([0-9]+(\.[0-9]+)?)%.*/\1/p' "${coverage_dir}/cpp-coverage-summary.txt")"
branch_pct="$(sed -nE 's/^branches:[[:space:]]*([0-9]+(\.[0-9]+)?)%.*/\1/p' "${coverage_dir}/cpp-coverage-summary.txt")"

if [[ -z "${line_pct}" || -z "${function_pct}" || -z "${branch_pct}" ]]; then
  echo "ERROR: Failed to parse gcovr summary percentages from ${coverage_dir}/cpp-coverage-summary.txt" >&2
  exit 2
fi

echo "CPP_COVERAGE_LINE=${line_pct}%"
echo "CPP_COVERAGE_FUNCTION=${function_pct}%"
echo "CPP_COVERAGE_BRANCH=${branch_pct}%"
exit "${cov_rc}"
