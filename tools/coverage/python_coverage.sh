#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/coverage/python_coverage.sh [pytest args...]

Runs Python tests under coverage, writes Cobertura/XML + HTML + text reports,
and enforces Python coverage gates:
  - line coverage
  - branch coverage

Environment:
  PYTHON_BIN            Python executable (default: python3)
  REPORT_ROOT           Report directory (default: artifacts/coverage/python)
  PYTHON_TEST_TARGETS   Space-separated pytest targets
                        (default: "tests/builder tests/tools")
  COVERAGE_FILE         Coverage data file path
                        (default: <REPORT_ROOT>/.coverage)
  PYTHON_COVERAGE_MIN_LINE
                       Minimum required line coverage percent (default: 100)
  PYTHON_COVERAGE_MIN_BRANCH
                       Minimum required branch coverage percent (default: 100)

Outputs:
  <REPORT_ROOT>/cobertura-python.xml
  <REPORT_ROOT>/summary.txt
  <REPORT_ROOT>/html/index.html

Examples:
  tools/coverage/python_coverage.sh
  tools/coverage/python_coverage.sh -k tokenizer -q
  PYTHON_BIN=/opt/venv/bin/python tools/coverage/python_coverage.sh
  PYTHON_TEST_TARGETS="tests/tools" tools/coverage/python_coverage.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPORT_ROOT="${REPORT_ROOT:-${REPO_ROOT}/artifacts/coverage/python}"
COBERTURA_XML="${COBERTURA_XML:-${REPORT_ROOT}/cobertura-python.xml}"
SUMMARY_TXT="${SUMMARY_TXT:-${REPORT_ROOT}/summary.txt}"
HTML_DIR="${HTML_DIR:-${REPORT_ROOT}/html}"
PYTHON_COVERAGE_MIN_LINE="${PYTHON_COVERAGE_MIN_LINE:-100}"
PYTHON_COVERAGE_MIN_BRANCH="${PYTHON_COVERAGE_MIN_BRANCH:-100}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -m coverage --version >/dev/null 2>&1; then
  echo "ERROR: coverage.py is required. Install with '${PYTHON_BIN} -m pip install coverage'." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -m pytest --version >/dev/null 2>&1; then
  echo "ERROR: pytest is required. Install with '${PYTHON_BIN} -m pip install pytest'." >&2
  exit 1
fi

if [[ -n "${PYTHON_TEST_TARGETS:-}" ]]; then
  # shellcheck disable=SC2206
  TEST_TARGETS=(${PYTHON_TEST_TARGETS})
else
  TEST_TARGETS=(tests/builder tests/tools)
fi

mkdir -p "${REPORT_ROOT}" "${HTML_DIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export COVERAGE_FILE="${COVERAGE_FILE:-${REPORT_ROOT}/.coverage}"

echo "[python-coverage] Repo root: ${REPO_ROOT}"
echo "[python-coverage] Report root: ${REPORT_ROOT}"
echo "[python-coverage] Test targets: ${TEST_TARGETS[*]}"
echo "[python-coverage] Gate thresholds: line>=${PYTHON_COVERAGE_MIN_LINE}% branch>=${PYTHON_COVERAGE_MIN_BRANCH}%"

pushd "${REPO_ROOT}" >/dev/null
"${PYTHON_BIN}" -m coverage erase
"${PYTHON_BIN}" -m coverage run --branch -m pytest "${TEST_TARGETS[@]}" "$@"
"${PYTHON_BIN}" -m coverage report --show-missing | tee "${SUMMARY_TXT}"
"${PYTHON_BIN}" -m coverage xml -o "${COBERTURA_XML}"
"${PYTHON_BIN}" -m coverage html -d "${HTML_DIR}"
popd >/dev/null

"${PYTHON_BIN}" - "${COBERTURA_XML}" "${PYTHON_COVERAGE_MIN_LINE}" "${PYTHON_COVERAGE_MIN_BRANCH}" <<'PY'
import sys
import xml.etree.ElementTree as ET

xml_path = sys.argv[1]
line_min = float(sys.argv[2])
branch_min = float(sys.argv[3])
try:
    root = ET.parse(xml_path).getroot()
    line_rate = float(root.attrib.get("line-rate", "0"))
    branch_rate = float(root.attrib.get("branch-rate", "0"))
except Exception as exc:  # noqa: BLE001
    print(f"ERROR: Failed to parse Cobertura XML '{xml_path}': {exc}", file=sys.stderr)
    sys.exit(3)

line_pct = line_rate * 100.0
branch_pct = branch_rate * 100.0
failed = False
if line_pct + 1e-9 < line_min:
    print(
        "ERROR: Python line coverage gate failed: "
        f"required {line_min:.2f}%, actual {line_pct:.2f}%.",
        file=sys.stderr,
    )
    failed = True
if branch_pct + 1e-9 < branch_min:
    print(
        "ERROR: Python branch coverage gate failed: "
        f"required {branch_min:.2f}%, actual {branch_pct:.2f}%.",
        file=sys.stderr,
    )
    failed = True

print(f"PYTHON_COVERAGE_LINE={line_pct:.2f}%")
print(f"PYTHON_COVERAGE_BRANCH={branch_pct:.2f}%")

if failed:
    print(f"ERROR: Cobertura XML: {xml_path}", file=sys.stderr)
    sys.exit(2)

print(
    "PASS: Python coverage gates satisfied "
    f"(line={line_pct:.2f}% >= {line_min:.2f}%, "
    f"branch={branch_pct:.2f}% >= {branch_min:.2f}%)."
)
PY

echo "[python-coverage] Cobertura XML: ${COBERTURA_XML}"
echo "[python-coverage] HTML report : ${HTML_DIR}/index.html"
echo "[python-coverage] Text summary: ${SUMMARY_TXT}"
