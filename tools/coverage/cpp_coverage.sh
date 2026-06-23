#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/coverage/cpp_coverage.sh [ctest args...]

Configures/builds/tests C++ targets with coverage flags, writes Cobertura/XML +
HTML + text reports via gcovr, and enforces coverage gates for:
  - line coverage
  - function coverage
  - branch coverage

Environment:
  BUILD_DIR              Build directory (default: build-cov)
  REPORT_ROOT            Report directory (default: artifacts/coverage/cpp)
  CMAKE_GENERATOR        CMake generator (default: Ninja)
  CMAKE_BUILD_TYPE       Build type (default: Coverage)
  COVERAGE_COMPILE_FLAGS Compiler flags (default: "--coverage -O0 -g0")
  COVERAGE_LINK_FLAGS    Linker flags (default: "--coverage")
  BUILD_PARALLEL         Build parallelism passed to cmake --build --parallel
  CMAKE_EXTRA_ARGS       Extra CMake configure args (space-separated)
  TRTMC_ENABLE_LIBTORCH_MULTINOMIAL
                         Enable optional libtorch multinomial bridge (default: OFF)
  TRT_INC_DIR            TensorRT include root (optional, used for explicit cmake wiring)
  TRT_LIB_DIR            TensorRT library root (optional, used for explicit cmake wiring)
  CUDA_INC_DIR           CUDA include dir for explicit cmake wiring
                         (default: /usr/local/cuda/include)
  CUDART_LIBRARY         libcudart path for explicit cmake wiring
                         (default: /usr/local/cuda/lib64/libcudart.so)
  GCOVR_FILTERS          Space-separated gcovr --filter values
                         (default: "<repo>/src <repo>/include")
  GCOVR_EXCLUDES         Space-separated gcovr --exclude values
                         (default excludes tests, build outputs, compiler IDs,
                         and model-owned runtime plugins)
  CPP_COVERAGE_MIN_LINE
                         Minimum required line coverage percent (default: 100)
  CPP_COVERAGE_MIN_FUNCTION
                         Minimum required function coverage percent (default: 100)
  CPP_COVERAGE_MIN_BRANCH
                         Minimum required branch coverage percent (default: 100)

Outputs:
  <REPORT_ROOT>/cobertura-cpp.xml
  <REPORT_ROOT>/index.html
  <REPORT_ROOT>/summary.txt
  <REPORT_ROOT>/gate.log

Examples:
  tools/coverage/cpp_coverage.sh
  tools/coverage/cpp_coverage.sh -R test_bundle_format
  BUILD_DIR=build-cov-fast tools/coverage/cpp_coverage.sh
  TRT_INC_DIR=/usr/include tools/coverage/cpp_coverage.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-cov}"
REPORT_ROOT="${REPORT_ROOT:-${REPO_ROOT}/artifacts/coverage/cpp}"
COBERTURA_XML="${COBERTURA_XML:-${REPORT_ROOT}/cobertura-cpp.xml}"
HTML_REPORT="${HTML_REPORT:-${REPORT_ROOT}/index.html}"
SUMMARY_TXT="${SUMMARY_TXT:-${REPORT_ROOT}/summary.txt}"
GATE_LOG="${GATE_LOG:-${REPORT_ROOT}/gate.log}"

CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Coverage}"
COVERAGE_COMPILE_FLAGS="${COVERAGE_COMPILE_FLAGS:---coverage -O0 -g0}"
COVERAGE_LINK_FLAGS="${COVERAGE_LINK_FLAGS:---coverage}"
BUILD_PARALLEL="${BUILD_PARALLEL:-}"
TRTMC_ENABLE_LIBTORCH_MULTINOMIAL="${TRTMC_ENABLE_LIBTORCH_MULTINOMIAL:-OFF}"
CPP_COVERAGE_MIN_LINE="${CPP_COVERAGE_MIN_LINE:-100}"
CPP_COVERAGE_MIN_FUNCTION="${CPP_COVERAGE_MIN_FUNCTION:-100}"
CPP_COVERAGE_MIN_BRANCH="${CPP_COVERAGE_MIN_BRANCH:-100}"
TRT_INC_DIR="${TRT_INC_DIR:-}"
TRT_LIB_DIR="${TRT_LIB_DIR:-}"
CUDA_INC_DIR="${CUDA_INC_DIR:-/usr/local/cuda/include}"
CUDART_LIBRARY="${CUDART_LIBRARY:-/usr/local/cuda/lib64/libcudart.so}"

for tool in cmake ctest gcovr; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "ERROR: Required tool not found in PATH: ${tool}" >&2
    if [[ "${tool}" == "gcovr" ]]; then
      echo "ERROR: Install gcovr (for example: 'python3 -m pip install gcovr')." >&2
    fi
    exit 1
  fi
done

if ! gcovr --help 2>/dev/null | grep -q -- "--fail-under-function"; then
  echo "ERROR: Installed gcovr does not support '--fail-under-function'." >&2
  echo "ERROR: Please install a newer gcovr that supports line/function/branch gates." >&2
  exit 1
fi

if [[ -n "${GCOVR_FILTERS:-}" ]]; then
  # shellcheck disable=SC2206
  FILTERS=(${GCOVR_FILTERS})
else
  FILTERS=("${REPO_ROOT}/src" "${REPO_ROOT}/include")
fi

if [[ -n "${GCOVR_EXCLUDES:-}" ]]; then
  # shellcheck disable=SC2206
  EXCLUDES=(${GCOVR_EXCLUDES})
else
  EXCLUDES=(
    "${REPO_ROOT}/tests"
    "${REPO_ROOT}/build.*"
    "${REPO_ROOT}/src/runtime/models"
    ".*/CMakeFiles/.*/CompilerIdCXX/.*"
  )
fi

mkdir -p "${BUILD_DIR}" "${REPORT_ROOT}"

cmake_args=(
  -S "${REPO_ROOT}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  -DCMAKE_C_FLAGS="${COVERAGE_COMPILE_FLAGS}"
  -DCMAKE_CXX_FLAGS="${COVERAGE_COMPILE_FLAGS}"
  -DCMAKE_EXE_LINKER_FLAGS="${COVERAGE_LINK_FLAGS}"
  -DCMAKE_SHARED_LINKER_FLAGS="${COVERAGE_LINK_FLAGS}"
  -DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL="${TRTMC_ENABLE_LIBTORCH_MULTINOMIAL}"
)

if [[ -n "${CMAKE_GENERATOR}" ]]; then
  cmake_args=(-G "${CMAKE_GENERATOR}" "${cmake_args[@]}")
fi

if [[ -n "${TRT_INC_DIR}" && -n "${TRT_LIB_DIR}" ]]; then
  cmake_args+=("-DTRTMC_TRT_INCLUDE_DIR=${TRT_INC_DIR}")
  cmake_args+=("-DTRTMC_TRT_LIBRARY=${TRT_LIB_DIR}/libnvinfer.so")
  cmake_args+=("-DTRTMC_CUDA_INCLUDE_DIR=${CUDA_INC_DIR}")
  cmake_args+=("-DTRTMC_CUDART_LIBRARY=${CUDART_LIBRARY}")
fi

if [[ -n "${CMAKE_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_CMAKE_ARGS=(${CMAKE_EXTRA_ARGS})
  cmake_args+=("${EXTRA_CMAKE_ARGS[@]}")
fi

echo "[cpp-coverage] Repo root: ${REPO_ROOT}"
echo "[cpp-coverage] Build dir: ${BUILD_DIR}"
echo "[cpp-coverage] Report root: ${REPORT_ROOT}"
echo "[cpp-coverage] gcovr filters: ${FILTERS[*]}"
echo "[cpp-coverage] gcovr excludes: ${EXCLUDES[*]}"
echo "[cpp-coverage] Gate thresholds: line>=${CPP_COVERAGE_MIN_LINE}% function>=${CPP_COVERAGE_MIN_FUNCTION}% branch>=${CPP_COVERAGE_MIN_BRANCH}%"

cmake "${cmake_args[@]}"

build_parallel_args=(--parallel)
if [[ -n "${BUILD_PARALLEL}" ]]; then
  build_parallel_args=(--parallel "${BUILD_PARALLEL}")
fi

cmake --build "${BUILD_DIR}" "${build_parallel_args[@]}"
# C++ unit test executables are excluded from the default wheel build, so
# coverage must request the aggregate test target explicitly before ctest.
cmake --build "${BUILD_DIR}" --target trtmc_cpp_tests "${build_parallel_args[@]}"

# Remove stale runtime coverage data from previous runs.
find "${BUILD_DIR}" -name "*.gcda" -delete || true

ctest --test-dir "${BUILD_DIR}" --output-on-failure "$@"

gcovr_base=(
  --root "${REPO_ROOT}"
  --object-directory "${BUILD_DIR}"
  --gcov-ignore-errors source_not_found
  --gcov-ignore-errors no_working_dir_found
)
for filter in "${FILTERS[@]}"; do
  gcovr_base+=(--filter "${filter}")
done
for exclude in "${EXCLUDES[@]}"; do
  gcovr_base+=(--exclude "${exclude}")
done

gcovr "${gcovr_base[@]}" \
  --xml "${COBERTURA_XML}" \
  --xml-pretty \
  --html-details "${HTML_REPORT}"

gcovr "${gcovr_base[@]}" \
  --txt-summary | tee "${SUMMARY_TXT}"

set +e
gcovr "${gcovr_base[@]}" \
  --print-summary \
  --fail-under-line "${CPP_COVERAGE_MIN_LINE}" \
  --fail-under-function "${CPP_COVERAGE_MIN_FUNCTION}" \
  --fail-under-branch "${CPP_COVERAGE_MIN_BRANCH}" >"${GATE_LOG}" 2>&1
gate_rc=$?
set -e

if [[ ${gate_rc} -ne 0 ]]; then
  cat "${GATE_LOG}" >&2
  echo "ERROR: C++ coverage gate failed." >&2
  echo "ERROR: Required thresholds are line=${CPP_COVERAGE_MIN_LINE}%, function=${CPP_COVERAGE_MIN_FUNCTION}%, branch=${CPP_COVERAGE_MIN_BRANCH}%." >&2
  echo "ERROR: Reports: ${SUMMARY_TXT} and ${COBERTURA_XML}" >&2
  exit "${gate_rc}"
fi

cat "${GATE_LOG}"
echo "PASS: C++ coverage gates satisfied (line/function/branch >= configured thresholds)."
echo "[cpp-coverage] Cobertura XML: ${COBERTURA_XML}"
echo "[cpp-coverage] HTML report : ${HTML_REPORT}"
echo "[cpp-coverage] Text summary: ${SUMMARY_TXT}"
