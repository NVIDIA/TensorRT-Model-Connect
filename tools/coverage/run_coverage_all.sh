#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/coverage/run_coverage_all.sh

Runs Python coverage gate first, then C++ coverage gates.

Environment:
  REPORT_ROOT      Parent report directory (default: artifacts/coverage)
  PYTHON_ARGS      Space-separated args forwarded to python_coverage.sh
  CPP_CTEST_ARGS   Space-separated args forwarded to cpp_coverage.sh (ctest args)

Examples:
  tools/coverage/run_coverage_all.sh
  PYTHON_ARGS="-k tokenizer -q" CPP_CTEST_ARGS="-R test_bundle_format" \
    tools/coverage/run_coverage_all.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPORT_ROOT="${REPORT_ROOT:-${REPO_ROOT}/artifacts/coverage}"
PYTHON_REPORT_ROOT="${PYTHON_REPORT_ROOT:-${REPORT_ROOT}/python}"
CPP_REPORT_ROOT="${CPP_REPORT_ROOT:-${REPORT_ROOT}/cpp}"

if [[ -n "${PYTHON_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  PYTHON_ARGS_ARR=(${PYTHON_ARGS})
else
  PYTHON_ARGS_ARR=()
fi

if [[ -n "${CPP_CTEST_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  CPP_CTEST_ARGS_ARR=(${CPP_CTEST_ARGS})
else
  CPP_CTEST_ARGS_ARR=()
fi

echo "[coverage-all] Running Python coverage gate..."
REPORT_ROOT="${PYTHON_REPORT_ROOT}" \
  "${SCRIPT_DIR}/python_coverage.sh" "${PYTHON_ARGS_ARR[@]}"

echo "[coverage-all] Running C++ coverage gates..."
REPORT_ROOT="${CPP_REPORT_ROOT}" \
  "${SCRIPT_DIR}/cpp_coverage.sh" "${CPP_CTEST_ARGS_ARR[@]}"

echo "[coverage-all] Completed. Combined report root: ${REPORT_ROOT}"
