#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

group() {
  echo "::group::$1"
}

endgroup() {
  echo "::endgroup::"
}

run_step() {
  local name="$1"
  shift
  group "$name"
  "$@"
  endgroup
}

run_with_timeout() {
  local limit="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout --kill-after=2m "$limit" "$@"
  else
    echo "WARNING: timeout command not found; running without ${limit} limit" >&2
    "$@"
  fi
}

generate_e2e_report() {
  if [ -d e2e_artifacts ]; then
    python scripts/generate_e2e_report.py \
      --artifacts-dir e2e_artifacts/artifacts \
      -o e2e_artifacts/e2e_report.html \
      --manifest-dir tests/e2e/models \
      --project-dir . \
      --title "GitHub Actions E2E Report - ${GITHUB_RUN_ID:-local}" \
      || true
  fi
}

trap generate_e2e_report EXIT

mkdir_if_set() {
  local path="${1:-}"
  if [ -n "$path" ]; then
    mkdir -p "$path"
  fi
}

prepare_shared_directories() {
  mkdir_if_set "${ENGINE_DIR:-}"
  mkdir_if_set "${HF_HOME:-}"
  mkdir_if_set "${HF_HUB_CACHE:-}"
  mkdir_if_set "${HUGGINGFACE_HUB_CACHE:-}"
  mkdir_if_set "${HF_MODULES_CACHE:-}"
}

prepare_shared_directories

ci_state_dir() {
  printf '%s\n' "${TRTMC_CI_STATE_DIR:-.ci}"
}

ensure_ci_state_dir() {
  mkdir -p "$(ci_state_dir)"
}

wheel_build_metadata_file() {
  printf '%s/trtmc-wheel-build.env\n' "$(ci_state_dir)"
}

wheel_install_metadata_file() {
  printf '%s/wheel-installed.env\n' "$(ci_state_dir)"
}

write_shell_var() {
  local name="$1"
  local value="$2"
  printf '%s=%q\n' "$name" "$value"
}

configure_e2e_timing_cache() {
  local cache_root="${TRTMC_STORAGE_ROOT:-${ENGINE_DIR:-.}}"
  local opt_level="${TRTMC_BUILDER_OPTIMIZATION_LEVEL:-default}"
  local cache_suffix="opt${opt_level}"
  if [ -n "${TRTMC_MAX_NUM_TACTICS:-}" ]; then
    cache_suffix="${cache_suffix}-tactics${TRTMC_MAX_NUM_TACTICS}"
  fi
  if [ -n "${TRTMC_AVG_TIMING_ITERATIONS:-}" ]; then
    cache_suffix="${cache_suffix}-avg${TRTMC_AVG_TIMING_ITERATIONS}"
  fi
  if [ -z "${TRTMC_TRT_TIMING_CACHE_PATH:-}" ]; then
    export TRTMC_TRT_TIMING_CACHE_PATH="${cache_root%/}/trt-timing-cache/tensorrt-${cache_suffix}.cache"
  fi
  mkdir -p "$(dirname "$TRTMC_TRT_TIMING_CACHE_PATH")"
  echo "TRTMC_TRT_TIMING_CACHE_PATH=${TRTMC_TRT_TIMING_CACHE_PATH}"
  if [ -n "${TRTMC_BUILDER_OPTIMIZATION_LEVEL:-}" ]; then
    echo "TRTMC_BUILDER_OPTIMIZATION_LEVEL=${TRTMC_BUILDER_OPTIMIZATION_LEVEL}"
  fi
  if [ -n "${TRTMC_MAX_NUM_TACTICS:-}" ]; then
    echo "TRTMC_MAX_NUM_TACTICS=${TRTMC_MAX_NUM_TACTICS}"
  fi
  if [ -n "${TRTMC_AVG_TIMING_ITERATIONS:-}" ]; then
    echo "TRTMC_AVG_TIMING_ITERATIONS=${TRTMC_AVG_TIMING_ITERATIONS}"
  fi
}

verify_environment() {
  git config --global --add safe.directory "${GITHUB_WORKSPACE:-$PWD}" || true
  git config --global --add safe.directory "*" || true
  echo "ENGINE_DIR=${ENGINE_DIR:-}"
  echo "HF_HOME=${HF_HOME:-}"
  echo "HF_HUB_CACHE=${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-}}"
  echo "HF_MODULES_CACHE=${HF_MODULES_CACHE:-}"
  python -c "import transformers, sys; print(f'python={sys.executable} transformers={transformers.__version__}'); assert transformers.__version__ == '5.2.0', transformers.__version__"
  chmod +x ./build/trtmc 2>/dev/null || true
}

install_built_wheel() {
  local install_wheel
  install_wheel="$(select_compatible_wheel dist)"
  python -m pip install --disable-pip-version-check --force-reinstall --no-deps "$install_wheel"
  python - "$install_wheel" <<'PY'
from __future__ import annotations

import importlib.resources as resources
import shutil
import sys
from pathlib import Path

import tensorrt_model_connect

wheel = Path(sys.argv[1])
repo = Path.cwd().resolve()
package_file = Path(tensorrt_model_connect.__file__).resolve()
try:
    package_file.relative_to(repo)
except ValueError:
    pass
else:
    raise SystemExit(
        f"tensorrt_model_connect imported from source tree after wheel install: {package_file}"
    )

installed_script = shutil.which("trtmc")
if installed_script is None:
    raise SystemExit("wheel did not install trtmc on PATH")
installed_script_path = Path(installed_script)
if installed_script_path.read_bytes()[:4] != b"\x7fELF":
    raise SystemExit(f"{installed_script_path} is not the native ELF trtmc executable")

native_dir = Path(resources.files("tensorrt_model_connect").joinpath("bin"))
native = native_dir / "trtmc"
backends = sorted(native_dir.glob("libtrtmc_backend_trt*.so*"))
if not native.is_file():
    raise SystemExit(f"packaged native trtmc executable is missing under {native_dir}")
if not backends:
    raise SystemExit(f"packaged TensorRT backend DSO is missing under {native_dir}")

print(f"installed_wheel={wheel}")
print(f"imported_package={package_file}")
print(f"installed_trtmc={installed_script_path}")
print(f"packaged_native_trtmc={native}")
for backend in backends:
    print(f"packaged_backend={backend}")
PY
}

install_built_wheel_once() {
  ensure_ci_state_dir
  local sentinel
  sentinel="$(wheel_install_metadata_file)"
  if [ -f "$sentinel" ]; then
    echo "Built wheel already installed in this CI container:"
    cat "$sentinel"
    return 0
  fi
  local install_wheel
  install_wheel="$(select_compatible_wheel dist)"
  install_built_wheel
  {
    write_shell_var TRTMC_INSTALLED_WHEEL "$install_wheel"
    write_shell_var TRTMC_WHEEL_INSTALLED_AT "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$sentinel"
}

verify_built_wheel_installed() {
  local sentinel
  sentinel="$(wheel_install_metadata_file)"
  if [ ! -f "$sentinel" ]; then
    echo "ERROR: built wheel has not been installed in this CI container; missing $sentinel" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$sentinel"
  python - "${TRTMC_INSTALLED_WHEEL:-}" <<'PY'
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import tensorrt_model_connect

wheel = Path(sys.argv[1])
repo = Path.cwd().resolve()
package_file = Path(tensorrt_model_connect.__file__).resolve()
try:
    package_file.relative_to(repo)
except ValueError:
    pass
else:
    raise SystemExit(
        f"tensorrt_model_connect imported from source tree after wheel install: {package_file}"
    )

installed_script = shutil.which("trtmc")
if installed_script is None:
    raise SystemExit("installed trtmc was not found on PATH")
installed_script_path = Path(installed_script)
if installed_script_path.read_bytes()[:4] != b"\x7fELF":
    raise SystemExit(f"{installed_script_path} is not the native ELF trtmc executable")

print(f"installed_wheel={wheel}")
print(f"imported_package={package_file}")
print(f"installed_trtmc={installed_script_path}")
PY
}

setup_source_check_environment() {
  verify_environment
}

setup_wheel_runtime_environment() {
  verify_environment
  verify_built_wheel_installed
}

setup_package_build_environment() {
  verify_environment
}

ensure_conan_cli() {
  if command -v conan >/dev/null 2>&1; then
    return 0
  fi
  python -m pip install --disable-pip-version-check --quiet "conan-py-build==0.4.3"
}

impact_analysis() {
  python3 tools/test_impact.py --validate
  python3 tools/coverage_map/fetch_latest.py \
    --output coverage_map.json \
    --local-fallback "${COVERAGE_MAP_PATH:-}" \
    || echo "No coverage map available -- using tier-level selection"

  local impact_args=(--base "$CI_BASE_REF")
  if [ -f coverage_map.json ]; then
    impact_args+=(--coverage-map coverage_map.json)
  fi
  python3 tools/test_impact.py "${impact_args[@]}" --json > impact.json

  echo "--- Impact Analysis ---"
  cat impact.json
  python3 - <<'PY'
import json

from tools.test_impact import ImpactResult, format_human

with open("impact.json", encoding="utf-8") as f:
    print(format_human(ImpactResult(**json.load(f))))
PY
}

load_wheel_build_metadata() {
  local metadata
  metadata="$(wheel_build_metadata_file)"
  if [ ! -f "$metadata" ]; then
    echo "ERROR: reusable wheel build metadata is missing: $metadata" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$metadata"
  : "${TRTMC_REUSE_CONAN_OUT_DIR:?TRTMC_REUSE_CONAN_OUT_DIR missing from $metadata}"
  : "${TRTMC_REUSE_CMAKE_BUILD_DIR:?TRTMC_REUSE_CMAKE_BUILD_DIR missing from $metadata}"
}

select_cpp_build_targets() {
  if [ "${FULL_E2E:-false}" = "true" ]; then
    printf '%s\n' "trtmc_cpp_tests"
    printf '%s\n' "trtmc_model_plugins"
    return 0
  fi

  python3 - <<'PY'
import json
from pathlib import Path

from tools import model_plugin_isolation

impact = Path("impact.json")
if not impact.exists():
    print("trtmc_cpp_tests")
    print("trtmc_model_plugins")
    raise SystemExit(0)

d = json.loads(impact.read_text())
targets = set()

if "cpp" in d.get("unit_tiers", []):
    tests = d.get("cpp_tests", [])
    fallback = d.get("fallback_tiers", [])
    if "cpp" in fallback or not tests:
        targets.add("trtmc_cpp_tests")
    else:
        targets.update(str(test) for test in tests)

selected_models = {str(model) for model in d.get("e2e_models", []) if str(model)}
for test_id in d.get("e2e_test_ids", []):
    match = model_plugin_isolation._NODE_ID_MODEL_RE.search(str(test_id))
    if match:
        selected_models.add(match.group(1))

if selected_models:
    manifests = model_plugin_isolation.discover_e2e_manifests(Path.cwd())
    runtime_plugins = model_plugin_isolation.discover_runtime_plugins(Path.cwd())
    targets.update(
        plugin.target
        for plugin in model_plugin_isolation.plugins_for_models(
            selected_models, manifests, runtime_plugins
        )
    )

for target in sorted(targets):
    print(target)
PY
}

build_cpp_test_executables() {
  local targets
  targets="$(select_cpp_build_targets)"
  if [ -z "$targets" ]; then
    echo "Skipping: no C++ test targets selected"
    return 0
  fi

  load_wheel_build_metadata
  ensure_conan_cli
  local conan_profile="${CONAN_PY_BUILD_PROFILE:-$PWD/conan-py-build.profile}"
  local conan_profile_args=()
  if [ -f "$conan_profile" ]; then
    conan_profile_args=(-pr:h "$conan_profile" -pr:b "$conan_profile")
  fi
  echo "Building C++ test target(s) via Conan: $targets"
  run_with_timeout "${BUILD_ALL_TIMEOUT:-15m}" env \
    TRTMC_CONAN_ENABLE_TEST_TARGETS=1 \
    TRTMC_CONAN_BUILD_TARGETS="$targets" \
    TRTMC_TRT_INCLUDE_DIR="${TRTMC_TRT_INCLUDE_DIR:-}" \
    TRTMC_TRT_LIBRARY="${TRTMC_TRT_LIBRARY:-}" \
    TRTMC_CUDA_INCLUDE_DIR="${TRTMC_CUDA_INCLUDE_DIR:-}" \
    TRTMC_CUDART_LIBRARY="${TRTMC_CUDART_LIBRARY:-}" \
    conan build . -of "$TRTMC_REUSE_CONAN_OUT_DIR" "${conan_profile_args[@]}"
}

prepare_model_plugin_dir() {
  local output_dir="$1"
  shift

  load_wheel_build_metadata
  rm -rf "$output_dir"
  mkdir -p "$output_dir"
  python3 tools/model_plugin_isolation.py prepare \
    --build-dir "$TRTMC_REUSE_CMAKE_BUILD_DIR" \
    --output-dir "$output_dir" \
    "$@"
}

check_family_coverage() {
  python scripts/check_family_coverage.py
}

check_cyclomatic_complexity() {
  lizard --version
  python tools/check_cyclomatic_complexity.py src --exclude src/cli --max-ccn "${CCM_MAX_CCN:-10}" --top 20
}

lint_changed_files() {
  if ! command -v ruff >/dev/null 2>&1 || ! command -v clang-format >/dev/null 2>&1; then
    python -m pip install --disable-pip-version-check --quiet ruff clang-format
  fi

  local base_ref="$CI_BASE_REF"
  if [ "${GITHUB_EVENT_NAME:-}" = "workflow_dispatch" ] || [ "${GITHUB_EVENT_NAME:-}" = "schedule" ]; then
    base_ref="${CI_BASE_REF:-origin/${GITHUB_REF_NAME:-main}}"
  fi

  local changed_py
  changed_py=$(git diff --diff-filter=d --name-only "$base_ref"...HEAD -- '*.py' || true)
  if [ -n "$changed_py" ]; then
    echo "Checking Python lint on changed files:"
    echo "$changed_py"
    echo "$changed_py" | xargs ruff check --config ruff.toml
  fi

  local changed_cpp
  changed_cpp=$(git diff --diff-filter=d --name-only "$base_ref"...HEAD -- '*.cpp' '*.h' || true)
  if [ -n "$changed_cpp" ]; then
    echo "Checking C++ formatting on changed files:"
    echo "$changed_cpp"
    echo "$changed_cpp" | xargs clang-format --dry-run --Werror
  fi
}

run_cpp_unit_tests() {
  if [ "${FULL_E2E:-false}" != "true" ]; then
    if ! python3 -c "import json; d=json.load(open('impact.json')); exit(0 if 'cpp' in d['unit_tiers'] else 1)"; then
      echo "Skipping: cpp tier not affected by this change"
      return 0
    fi
  fi

  local cpp_tests=""
  if [ "${FULL_E2E:-false}" != "true" ]; then
    cpp_tests=$(python3 -c "
import json, sys
d = json.load(open('impact.json'))
tests = d.get('cpp_tests', [])
fallback = d.get('fallback_tiers', [])
if 'cpp' in fallback or not tests:
    sys.exit(0)
print('|'.join(tests))
" 2>/dev/null) || true
  fi

  load_wheel_build_metadata
  if [ -n "$cpp_tests" ]; then
    echo "Selective C++ tests: $cpp_tests"
    run_with_timeout "${CPP_UNIT_TIMEOUT:-20m}" ctest --test-dir "$TRTMC_REUSE_CMAKE_BUILD_DIR" -R "$cpp_tests" --output-on-failure
  else
    echo "Running all C++ tests"
    run_with_timeout "${CPP_UNIT_TIMEOUT:-20m}" ctest --test-dir "$TRTMC_REUSE_CMAKE_BUILD_DIR" --output-on-failure
  fi
}

write_skipped_python_coverage() {
  local reason="${1:-Skipped}"
  echo '<?xml version="1.0" ?><coverage version="7.6" timestamp="0" lines-valid="0" lines-covered="0" line-rate="1.0" branches-valid="0" branches-covered="0" branch-rate="1.0" complexity="0"><packages/></coverage>' > coverage/python-cobertura.xml
  echo "PYTHON_COVERAGE_LINE=100.00%"
  echo "PYTHON_COVERAGE_BRANCH=100.00%"
  echo "$reason" > coverage/python-coverage.txt
}

write_python_package_gate_coverage_config() {
  local config_path="$1"
  cat > "$config_path" <<'EOF'
[run]
source =
    tensorrt_model_connect
branch = True
omit =
    */tests/*
    */__pycache__/*
    */tensorrt_model_connect/families/*

[report]
show_missing = True
precision = 1
exclude_lines =
    pragma: no cover
    if __name__ == .__main__.
    raise NotImplementedError
EOF
}

run_python_builder_tests() {
  python -m pip install --disable-pip-version-check --quiet "pytest-cov>=6.0"
  mkdir -p coverage

  local selected_tests_file="coverage/python-selected-tests.txt"
  if [ "${FULL_E2E:-false}" != "true" ]; then
    python3 -c "
import json
from pathlib import Path

d = json.load(open('impact.json'))
fallback = set(d.get('fallback_tiers', []))
builder_fallback = 'builder' in fallback
tools_fallback = 'tools' in fallback

selected = []

def add(paths):
    for path in paths:
        if path and path not in selected:
            selected.append(path)

if not (builder_fallback and tools_fallback):
    if builder_fallback:
        add(['tests/builder/'])
    else:
        add(d.get('builder_tests', []))

    if tools_fallback:
        add(['tests/tools/'])
        add(str(path) for path in sorted(Path('tests/e2e_harness').glob('test_*.py')))
    else:
        add(d.get('tools_tests', []))

    add([
        'tests/tools/test_github_actions_ci.py',
        'tests/tools/test_model_plugin_encapsulation_static.py',
        'tests/tools/test_schedule_e2e.py',
        'tests/tools/test_test_impact.py',
    ])

for test in selected:
    print(test)
" > "$selected_tests_file"
  fi

  local python_coverage_required="true"
  if [ "${FULL_E2E:-false}" != "true" ] && [ -s "$selected_tests_file" ]; then
    python_coverage_required="false"
    echo "Skipping Python package coverage gate: selected Python subset does not produce global package coverage"
  fi

  local cov_args=()
  if [ "$python_coverage_required" = "true" ]; then
    local python_cov_config="coverage/python-package-gate.coveragerc"
    write_python_package_gate_coverage_config "$python_cov_config"
    cov_args=(
      --cov=tensorrt_model_connect
      --cov-branch
      --cov-config="$python_cov_config"
      --cov-report=term-missing
      --cov-report=xml:coverage/python-cobertura.xml
    )
  fi
  if [ -s "$selected_tests_file" ]; then
    mapfile -t selected_python_tests < "$selected_tests_file"
    echo "Selective Python tests:"
    printf '  %s\n' "${selected_python_tests[@]}"
    run_with_timeout "${PYTHON_BUILDER_TIMEOUT:-40m}" python -m pytest "${selected_python_tests[@]}" -v \
      --ignore=tests/builder/test_cli.py \
      -n auto "${cov_args[@]}"
  else
    echo "Running all builder + tools tests"
    run_with_timeout "${PYTHON_BUILDER_TIMEOUT:-40m}" python -m pytest tests/builder/ tests/tools/ tests/e2e_harness/test_*.py -v \
      --ignore=tests/builder/test_cli.py \
      -n auto "${cov_args[@]}"
  fi

  if [ "$python_coverage_required" != "true" ]; then
    write_skipped_python_coverage "Skipped: selected Python subset does not produce global package coverage"
    return 0
  fi

  python -m coverage report --show-missing 2>/dev/null | tee coverage/python-coverage.txt || true
  python3 -c "
import xml.etree.ElementTree as ET, sys
root = ET.parse('coverage/python-cobertura.xml').getroot()
line_pct = float(root.attrib.get('line-rate', '0')) * 100
branch_pct = float(root.attrib.get('branch-rate', '0')) * 100
line_min = float('${PYTHON_COVERAGE_MIN_LINE}')
branch_min = float('${PYTHON_COVERAGE_MIN_BRANCH}')
print(f'PYTHON_COVERAGE_LINE={line_pct:.2f}%')
print(f'PYTHON_COVERAGE_BRANCH={branch_pct:.2f}%')
ok = True
if line_pct + 1e-9 < line_min:
    print(f'FAIL: Python line coverage {line_pct:.1f}% < {line_min}% gate', file=sys.stderr)
    ok = False
if branch_pct + 1e-9 < branch_min:
    print(f'FAIL: Python branch coverage {branch_pct:.1f}% < {branch_min}% gate', file=sys.stderr)
    ok = False
sys.exit(0 if ok else 1)
"
}

run_cpp_coverage() {
  case "${GITHUB_EVENT_NAME:-}" in
    schedule|workflow_dispatch)
      ;;
    pull_request)
      local changed_cpp
      changed_cpp=$(git diff --diff-filter=d --name-only "$CI_BASE_REF"...HEAD -- \
        'src/**/*.cpp' 'src/**/*.h' 'include/**/*.h' \
        'tests/cpp/**/*.cpp' 'tests/cpp/**/*.h' 'CMakeLists.txt' || true)
      if [ -z "$changed_cpp" ]; then
        echo "Skipping: no C++ source, C++ tests, or CMake changes in premerge diff"
        return 0
      fi
      echo "C++ coverage triggered by changed files:"
      echo "$changed_cpp"
      ;;
    *)
      echo "Skipping: C++ coverage only runs for nightly/manual pipelines and C++-affected premerge PRs"
      return 0
      ;;
  esac
  python -m pip install --disable-pip-version-check --quiet "gcovr==8.2"
  run_with_timeout "${CPP_COVERAGE_TIMEOUT:-40m}" bash tools/coverage_ci/run_cpp_coverage.sh
}

run_graph_op_tests() {
  nvidia-smi
  run_with_timeout "${GRAPH_OP_TIMEOUT:-20m}" python -m pytest tests/builder/test_graph_ops.py tests/builder/test_graph_ops_extended.py tests/builder/test_graph_blocks.py -v -n auto
}

cmake_cache_value() {
  local cache_file="$1"
  local variable="$2"
  local key value
  while IFS='=' read -r key value; do
    if [[ "$key" == "$variable:"* ]]; then
      printf '%s\n' "$value"
      return 0
    fi
  done < "$cache_file"
  return 1
}

configure_isolated_model_build() {
  local source_dir="$1"
  local build_dir="$2"
  local reusable_cache="$TRTMC_REUSE_CMAKE_BUILD_DIR/CMakeCache.txt"
  if [ ! -f "$reusable_cache" ]; then
    echo "ERROR: reusable CMake cache is missing: $reusable_cache" >&2
    return 1
  fi
  local cmake_args=(
    -S "$source_dir"
    -B "$build_dir"
    -DCMAKE_BUILD_TYPE=Release
    -DTRTMC_BUILD_TESTS=OFF
    -DTRTMC_BUILD_BENCHMARKS=OFF
    -DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF
  )
  local cache_value
  local variable

  if command -v ninja >/dev/null 2>&1; then
    cmake_args+=(-G Ninja)
  fi
  for variable in \
    TRTMC_TRT_BACKEND_ABI \
    TRTMC_TRT_INCLUDE_DIR \
    TRTMC_TRT_LIBRARY \
    TRTMC_CUDA_INCLUDE_DIR \
    TRTMC_CUDART_LIBRARY \
    CMAKE_CUDA_ARCHITECTURES; do
    cache_value="$(cmake_cache_value "$reusable_cache" "$variable" || true)"
    if [ -n "$cache_value" ]; then
      cmake_args+=("-D${variable}=${cache_value}")
    fi
  done

  local nlohmann_source="$TRTMC_REUSE_CMAKE_BUILD_DIR/_deps/nlohmann_json-src"
  if [ -d "$nlohmann_source" ]; then
    cmake_args+=("-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=$nlohmann_source")
  fi

  local conan_toolchain=""
  local search_root
  for search_root in \
    "${TRTMC_REUSE_CONAN_OUT_DIR:-}" \
    "$TRTMC_REUSE_CMAKE_BUILD_DIR"; do
    if [ -n "$search_root" ] && [ -d "$search_root" ]; then
      conan_toolchain="$(find "$search_root" -name conan_toolchain.cmake -print -quit)"
      if [ -n "$conan_toolchain" ]; then
        cmake_args+=("-DCMAKE_TOOLCHAIN_FILE=$conan_toolchain")
        break
      fi
    fi
  done
  run_with_timeout "${SELECTIVE_E2E_CONFIGURE_TIMEOUT:-10m}" cmake "${cmake_args[@]}"
}

discover_isolation_gpu_ids() {
  local -a gpu_lines=()
  mapfile -t gpu_lines < <(nvidia-smi -L 2>/dev/null)
  local gpu_count="${#gpu_lines[@]}"
  local hf_python="${HF_PYTHON:-/opt/venv/bin/python}"
  local -a healthy_gpu_ids=()
  local gpu_id
  for ((gpu_id = 0; gpu_id < gpu_count; gpu_id++)); do
    if CUDA_VISIBLE_DEVICES="$gpu_id" "$hf_python" - <<'PY' >/dev/null 2>&1
import tensorrt as trt

logger = trt.Logger(trt.Logger.ERROR)
if trt.Builder(logger) is None:
    raise SystemExit(1)
PY
    then
      healthy_gpu_ids+=("$gpu_id")
    else
      echo "WARN: GPU $gpu_id failed TensorRT builder health check" >&2
    fi
  done
  if [ "${#healthy_gpu_ids[@]}" -eq 0 ]; then
    echo "ERROR: no GPU passed the TensorRT builder health check" >&2
    return 1
  fi

  local exclude_gpu0="${TRTMC_E2E_EXCLUDE_GPU0:-}"
  if [ -z "$exclude_gpu0" ]; then
    if [ -n "${GITHUB_RUN_ID:-}" ]; then
      exclude_gpu0=1
    else
      exclude_gpu0=0
    fi
  fi
  if [ "$exclude_gpu0" != "0" ] && [ "${#healthy_gpu_ids[@]}" -gt 1 ]; then
    local -a filtered_gpu_ids=()
    for gpu_id in "${healthy_gpu_ids[@]}"; do
      if [ "$gpu_id" != "0" ]; then
        filtered_gpu_ids+=("$gpu_id")
      fi
    done
    healthy_gpu_ids=("${filtered_gpu_ids[@]}")
  fi
  printf '%s\n' "${healthy_gpu_ids[@]}"
}

run_isolated_e2e_group() {
  local group_manifest="$1"
  local result_dir="$2"
  local gpu_id="$3"
  local group_dir
  group_dir="$(dirname "$group_manifest")"
  local group_id model_target family
  local -a group_config
  mapfile -t group_config < <(python3 - "$group_manifest" <<'PY'
import json
import sys
from pathlib import Path

group = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(group["id"])
print(group["runtime_plugin"]["target"])
print(group["family"])
PY
)
  group_id="${group_config[0]}"
  model_target="${group_config[1]}"
  family="${group_config[2]}"

  local models_file="$group_dir/models.txt"
  local source_dir="$group_dir/source"
  local build_dir="$group_dir/build"
  local engine_dir="$group_dir/engines"
  local model_plugin_dir="$group_dir/model_plugins"
  local audit_dir="$result_dir/model_isolation/$group_id"

  echo "=== Isolated E2E group: $group_id on GPU $gpu_id ==="
  if ! python3 tools/model_plugin_isolation.py stage-source \
    --models-file "$models_file" \
    --output-dir "$source_dir" \
    --clean; then
    return 1
  fi
  mkdir -p "$audit_dir"
  cp "$group_manifest" "$audit_dir/group.json"
  cp "$source_dir/.trtmc-isolation.json" "$audit_dir/source-projection.json"

  if ! configure_isolated_model_build "$source_dir" "$build_dir"; then
    return 1
  fi
  if ! run_with_timeout "${SELECTIVE_E2E_BUILD_TIMEOUT:-30m}" \
    cmake --build "$build_dir" \
      --parallel "${TRTMC_ISOLATION_BUILD_JOBS:-16}" \
      --target trtmc trtmc_backend_trt "$model_target"; then
    return 1
  fi

  local -a built_model_dsos
  mapfile -t built_model_dsos < <(
    find "$build_dir/models" -type f -name 'libtrtmc_model_*.so' -print
  )
  if [ "${#built_model_dsos[@]}" -ne 1 ]; then
    echo "ERROR: isolated build $group_id produced ${#built_model_dsos[@]} model DSOs; expected exactly 1" >&2
    printf '  %s\n' "${built_model_dsos[@]}" >&2
    return 1
  fi

  if ! python3 "$source_dir/tools/model_plugin_isolation.py" prepare \
    --repo-root "$source_dir" \
    --models-file "$models_file" \
    --build-dir "$build_dir" \
    --output-dir "$model_plugin_dir"; then
    return 1
  fi

  local isolated_library_path="$build_dir"
  local library
  for library in "${TRTMC_TRT_LIBRARY:-}" "${TRTMC_CUDART_LIBRARY:-}"; do
    if [ -n "$library" ]; then
      isolated_library_path="${isolated_library_path}:$(dirname "$library")"
    fi
  done

  mkdir -p "$engine_dir" "$result_dir/artifacts"
  local -a group_test_files
  mapfile -t group_test_files < <(
    find "$source_dir/tests/e2e/models/$family" \
      -maxdepth 1 -type f -name 'test_*_e2e.py' | sort
  )
  if [ "${#group_test_files[@]}" -ne 1 ]; then
    echo "ERROR: $group_id has ${#group_test_files[@]} canonical E2E files; expected 1" >&2
    printf '  %s\n' "${group_test_files[@]}" >&2
    return 1
  fi
  local -a model_filter_args=()
  local model
  while IFS= read -r model; do
    if [ -n "$model" ]; then
      model_filter_args+=(--e2e-model "$model")
    fi
  done < "$models_file"

  local e2e_rc=0
  pushd "$source_dir" >/dev/null
  if run_with_timeout "${SELECTIVE_E2E_GROUP_TIMEOUT:-90m}" env \
      CUDA_VISIBLE_DEVICES="$gpu_id" \
      HF_HUB_OFFLINE=1 \
      PYTHONNOUSERSITE=1 \
      PYTHONPATH="$source_dir/python:$source_dir" \
      LD_LIBRARY_PATH="$isolated_library_path" \
      "${HF_PYTHON:-/opt/venv/bin/python}" -m pytest \
        "${group_test_files[@]}" -v \
        --rootdir "$source_dir" \
        -c "$source_dir/pyproject.toml" \
        --engine-dir "$engine_dir" \
        --trtmc-binary "$build_dir/trtmc" \
        --hf-python "${HF_PYTHON:-/opt/venv/bin/python}" \
        --e2e-artifacts-dir "$result_dir/artifacts" \
        --model-plugin-dir "$model_plugin_dir" \
        --e2e-models-file "$models_file" \
        --e2e-exclude-ci-tier nightly_only \
        "${model_filter_args[@]}" \
        --rebuild-engines \
        --junitxml="$audit_dir/junit.xml"; then
    e2e_rc=0
  else
    e2e_rc=$?
  fi
  popd >/dev/null

  local verification_rc=0
  if python3 "$source_dir/tools/model_plugin_isolation.py" verify-results \
      --repo-root "$source_dir" \
      --models-file "$models_file" \
      --artifacts-dir "$result_dir/artifacts" \
      --report "$audit_dir/verification.json"; then
    verification_rc=0
  else
    verification_rc=$?
  fi

  if [ "$e2e_rc" -ne 0 ]; then
    return "$e2e_rc"
  fi
  return "$verification_rc"
}

run_isolated_e2e_group_logged() {
  local group_manifest="$1"
  local result_dir="$2"
  local gpu_id="$3"
  local group_dir
  group_dir="$(dirname "$group_manifest")"
  local group_id
  group_id="$(basename "$group_dir")"
  local audit_dir="$result_dir/model_isolation/$group_id"
  mkdir -p "$audit_dir"
  local started_at
  started_at="$(date +%s)"
  local group_rc=0
  if run_isolated_e2e_group "$group_manifest" "$result_dir" "$gpu_id" \
      > "$audit_dir/console.log" 2>&1; then
    group_rc=0
  else
    group_rc=$?
  fi
  local elapsed=$(( $(date +%s) - started_at ))
  if [ "$group_rc" -eq 0 ]; then
    echo "PASS isolated group=$group_id gpu=$gpu_id elapsed=${elapsed}s"
    if [ "${TRTMC_ISOLATION_KEEP_WORKTREES:-0}" = "0" ]; then
      rm -rf \
        "$group_dir/source" \
        "$group_dir/build" \
        "$group_dir/engines" \
        "$group_dir/model_plugins"
    fi
  else
    echo "FAIL isolated group=$group_id gpu=$gpu_id rc=$group_rc elapsed=${elapsed}s" >&2
    tail -n 120 "$audit_dir/console.log" >&2 || true
  fi
  return "$group_rc"
}

run_isolated_gpu_queue() {
  local gpu_id="$1"
  local queue_file="$2"
  local result_dir="$3"
  local queue_rc=0
  local group_manifest
  while IFS= read -r group_manifest; do
    if [ -z "$group_manifest" ]; then
      continue
    fi
    if ! run_isolated_e2e_group_logged "$group_manifest" "$result_dir" "$gpu_id"; then
      queue_rc=1
    fi
  done < "$queue_file"
  return "$queue_rc"
}

run_model_owned_isolation_e2e() {
  local models_file="$1"
  local result_dir="$2"
  local isolation_root="$PWD/$(ci_state_dir)/model-isolation"
  rm -rf "$isolation_root" "$result_dir/model_isolation"
  mkdir -p "$result_dir/model_isolation"
  python3 tools/model_plugin_isolation.py plan \
    --models-file "$models_file" \
    --output-dir "$isolation_root" \
    --clean

  local -a gpu_ids
  mapfile -t gpu_ids < <(discover_isolation_gpu_ids)
  if [ "${#gpu_ids[@]}" -eq 0 ]; then
    return 1
  fi
  local max_parallel="${TRTMC_ISOLATION_MAX_PARALLEL_GROUPS:-${#gpu_ids[@]}}"
  if ! [[ "$max_parallel" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TRTMC_ISOLATION_MAX_PARALLEL_GROUPS must be a positive integer" >&2
    return 1
  fi
  if [ "$max_parallel" -lt "${#gpu_ids[@]}" ]; then
    gpu_ids=("${gpu_ids[@]:0:max_parallel}")
  fi

  if [ -z "${TRTMC_ISOLATION_BUILD_JOBS:-}" ]; then
    local total_build_jobs="${TRTMC_ISOLATION_TOTAL_BUILD_JOBS:-16}"
    if ! [[ "$total_build_jobs" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: TRTMC_ISOLATION_TOTAL_BUILD_JOBS must be a positive integer" >&2
      return 1
    fi
    local build_jobs=$(( total_build_jobs / ${#gpu_ids[@]} ))
    if [ "$build_jobs" -lt 2 ]; then
      build_jobs=2
    fi
    export TRTMC_ISOLATION_BUILD_JOBS="$build_jobs"
  elif ! [[ "$TRTMC_ISOLATION_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TRTMC_ISOLATION_BUILD_JOBS must be a positive integer" >&2
    return 1
  fi
  echo "Isolation workers: GPUs=${gpu_ids[*]} build_jobs_per_worker=$TRTMC_ISOLATION_BUILD_JOBS"

  local schedule_dir="$isolation_root/schedule"
  local schedule_args=(
    schedule
    --plan "$isolation_root/plan.json"
    --output-dir "$schedule_dir"
    --timing-estimates tests/e2e/timing_estimates.json
    --default-estimate-seconds "${TRTMC_ISOLATION_DEFAULT_ESTIMATE_S:-600}"
    --build-overhead-seconds "${TRTMC_ISOLATION_BUILD_ESTIMATE_S:-60}"
    --clean
  )
  local gpu_id
  for gpu_id in "${gpu_ids[@]}"; do
    schedule_args+=(--gpu-id "$gpu_id")
  done
  python3 tools/model_plugin_isolation.py "${schedule_args[@]}"
  cp "$schedule_dir/schedule.json" "$result_dir/model_isolation/schedule.json"

  local isolation_rc=0
  local -a queue_pids=()
  local -a queue_gpu_ids=()
  for gpu_id in "${gpu_ids[@]}"; do
    (
      trap - EXIT
      run_isolated_gpu_queue \
        "$gpu_id" \
        "$schedule_dir/gpu-$gpu_id.txt" \
        "$result_dir"
    ) &
    queue_pids+=("$!")
    queue_gpu_ids+=("$gpu_id")
  done
  local index
  for index in "${!queue_pids[@]}"; do
    if ! wait "${queue_pids[$index]}"; then
      isolation_rc=1
      echo "ERROR: isolated GPU queue ${queue_gpu_ids[$index]} failed" >&2
    fi
  done
  return "$isolation_rc"
}

run_selective_e2e() {
  if [ "${GITHUB_EVENT_NAME:-}" != "pull_request" ] || [ "${FULL_E2E:-false}" = "true" ]; then
    echo "Skipping: selective E2E only runs for pull_request events without full_e2e"
    return 0
  fi

  python3 -c "
import json
import re
d = json.load(open('impact.json'))
models = set(d.get('e2e_models', []))
test_ids = d.get('e2e_test_ids', [])
for test_id in test_ids:
    match = re.search(r'::test_model_e2e\[([^]]+)\]', test_id)
    if match:
        models.add(match.group(1))
models = sorted(models)
with open('e2e_models.txt', 'w') as f:
    for m in models:
        f.write(m + '\n')
with open('e2e_test_ids.txt', 'w') as f:
    for test_id in test_ids:
        f.write(test_id + '\n')
print(f'Selective E2E: {len(models)} models')
for m in models[:10]:
    print(f'  {m}')
if len(models) > 10:
    print(f'  ... and {len(models) - 10} more')
if test_ids:
    print(f'Selective E2E node IDs: {len(test_ids)}')
"
  python3 tools/model_plugin_isolation.py impact-models \
    --impact-json impact.json \
    --exclude-ci-tier multi_device \
    --exclude-ci-tier nightly_only \
    > e2e_isolation_models.txt
  echo "Model-owned isolation E2E: $(wc -l < e2e_isolation_models.txt) models"
  sed 's/^/  isolated: /' e2e_isolation_models.txt
  local model_count
  model_count=$(wc -l < e2e_models.txt)
  if [ "$model_count" -eq 0 ]; then
    echo "No E2E models affected by this change -- skipping E2E tests"
    mkdir -p e2e_artifacts/artifacts
    return 0
  fi

  export TRTMC_BUILDER_OPTIMIZATION_LEVEL="${TRTMC_BUILDER_OPTIMIZATION_LEVEL:-1}"
  configure_e2e_timing_cache
  echo "=== Phase 1: warming HF cache (online, sequential) ==="
  env -u HF_HUB_OFFLINE python scripts/warm_hf_cache.py \
    --models-file e2e_models.txt \
    --exclude-ci-tier multi_device \
    --exclude-ci-tier nightly_only

  echo "=== Phase 2: standard selective E2E for the full conservative impact set ==="
  load_wheel_build_metadata
  local result_dir="$PWD/e2e_artifacts"
  rm -rf "$result_dir"
  mkdir -p "$result_dir/artifacts"
  local full_model_plugin_dir="$result_dir/model_plugins"
  prepare_model_plugin_dir "$full_model_plugin_dir" --models-file e2e_models.txt
  local standard_args=(
    --engine-dir "$ENGINE_DIR"
    --result-dir "$result_dir"
    --trtmc-binary "$(command -v trtmc)"
    --workers-per-gpu 4
    --models-file e2e_models.txt
    --exclude-ci-tier nightly_only
    --model-plugin-dir "$full_model_plugin_dir"
  )
  if [ -s e2e_test_ids.txt ]; then
    standard_args+=(--tests-file e2e_test_ids.txt)
  fi
  if [ "${REBUILD_ENGINES:-true}" = "true" ]; then
    standard_args+=(--rebuild-engines)
  fi
  local standard_rc=0
  if run_with_timeout \
      "${SELECTIVE_E2E_STANDARD_TIMEOUT:-${SELECTIVE_E2E_TIMEOUT:-4h}}" \
      env HF_HUB_OFFLINE=1 ./scripts/run_e2e_parallel.sh "${standard_args[@]}"; then
    standard_rc=0
  else
    standard_rc=$?
    echo "ERROR: standard selective E2E failed with code $standard_rc" >&2
  fi
  if [ "$standard_rc" -ne 0 ]; then
    echo "Skipping strict model-owned isolation because standard selective E2E failed" >&2
    return "$standard_rc"
  fi

  local isolation_count
  isolation_count=$(wc -l < e2e_isolation_models.txt)
  local isolation_rc=0
  if [ "$isolation_count" -gt 0 ]; then
    echo "=== Phase 3: strict model-owned isolation E2E ==="
    if ! run_model_owned_isolation_e2e \
      e2e_isolation_models.txt \
      "$result_dir"; then
      isolation_rc=1
    fi
  else
    echo "No model-owned E2E cases changed -- strict isolation rerun not required"
  fi

  local vlm_rc=0
  if run_diffusion_vlm_assessment; then
    vlm_rc=0
  else
    vlm_rc=$?
  fi
  if [ "$standard_rc" -ne 0 ]; then
    return "$standard_rc"
  fi
  if [ "$isolation_rc" -ne 0 ]; then
    return "$isolation_rc"
  fi
  return "$vlm_rc"
}

run_full_e2e() {
  if [ "${FULL_E2E:-false}" != "true" ]; then
    echo "Skipping: full E2E was not requested"
    return 0
  fi

  echo "=== Nightly Full E2E: all models ==="
  nvidia-smi
  configure_e2e_timing_cache
  local model_plugin_dir="$PWD/e2e_artifacts/model_plugins"
  echo "=== Preparing isolated runtime model plugins ==="
  prepare_model_plugin_dir "$model_plugin_dir" --all
  echo "=== Phase 1: warming HF cache (online, sequential) ==="
  # TODO: Remove the multi_device exclusions once nightly CI has a runner pool
  # that can reserve all GPUs for tensor-parallel E2E cases.
  python scripts/warm_hf_cache.py \
    --exclude-ci-tier l0_only \
    --exclude-ci-tier multi_device
  echo "=== Phase 2: parallel rebuild (offline, local cache) ==="
  local args=(
    --engine-dir "$ENGINE_DIR"
    --result-dir e2e_artifacts
    --trtmc-binary "$(command -v trtmc)"
    --workers-per-gpu 4
    --exclude-ci-tier l0_only
    --exclude-ci-tier multi_device
    --model-plugin-dir "$model_plugin_dir"
  )
  if [ "${REBUILD_ENGINES:-true}" = "true" ]; then
    args+=(--rebuild-engines)
  fi
  run_e2e_with_diffusion_vlm "${FULL_E2E_TIMEOUT:-6h}" "${args[@]}"
}

run_e2e_with_diffusion_vlm() {
  local timeout_limit="$1"
  shift
  local e2e_rc=0
  local vlm_rc=0

  set +e
  run_with_timeout "$timeout_limit" env HF_HUB_OFFLINE=1 ./scripts/run_e2e_parallel.sh "$@"
  e2e_rc=$?
  set -e

  if [ "$e2e_rc" -ne 0 ]; then
    echo "E2E exited with code ${e2e_rc}; still attempting diffusion VLM assessment before returning that status."
  fi

  set +e
  run_diffusion_vlm_assessment
  vlm_rc=$?
  set -e

  if [ "$e2e_rc" -ne 0 ]; then
    return "$e2e_rc"
  fi
  return "$vlm_rc"
}

select_diffusion_vlm_config() {
  python - "${DIFFUSION_VLM_CONFIG:-}" <<'PY'
import json
import sys
from pathlib import Path

requested = sys.argv[1]
if requested:
    path = Path(requested)
    if not path.is_file():
        raise SystemExit(f"DIFFUSION_VLM_CONFIG does not exist: {path}")
    print(path)
    raise SystemExit(0)

configs = sorted(Path("tests/e2e/models").glob("*/diffusion_vlm_assessment.json"))
defaults = []
for path in configs:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Invalid diffusion VLM assessment config {path}: {exc}") from exc
    if data.get("default") is True:
        defaults.append(path)

if len(defaults) != 1:
    listed = ", ".join(str(path) for path in defaults) or "none"
    raise SystemExit(
        "Expected exactly one default diffusion VLM assessment config under "
        f"tests/e2e/models/*/diffusion_vlm_assessment.json; found {listed}"
    )
print(defaults[0])
PY
}

load_diffusion_vlm_config() {
  local config_path="$1"
  python - "$config_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
required = ("model_id", "max_side", "max_new_tokens", "timeout")
missing = [key for key in required if not data.get(key)]
if missing:
    raise SystemExit(f"{path} missing required diffusion VLM fields: {missing}")

for key in required:
    print(f"{key}\t{data[key]}")
PY
}

run_diffusion_vlm_assessment() {
  if [ "${DIFFUSION_VLM_ASSESSMENT:-true}" != "true" ]; then
    echo "Skipping: diffusion VLM assessment disabled"
    return 0
  fi
  if [ ! -d e2e_artifacts/artifacts ]; then
    echo "Skipping: no E2E artifacts directory for diffusion VLM assessment"
    return 0
  fi

  local pair_count
  pair_count=$(python3 tools/count_diffusion_frame_pairs.py e2e_artifacts/artifacts)
  if [ "${pair_count:-0}" -eq 0 ]; then
    echo "Skipping: no TRT/HF diffusion frame pairs for VLM assessment"
    return 0
  fi

  local vlm_config
  vlm_config="$(select_diffusion_vlm_config)"
  local config_model_id=""
  local config_max_side=""
  local config_max_new_tokens=""
  local config_timeout=""
  local key value
  while IFS=$'\t' read -r key value; do
    case "$key" in
      model_id) config_model_id="$value" ;;
      max_side) config_max_side="$value" ;;
      max_new_tokens) config_max_new_tokens="$value" ;;
      timeout) config_timeout="$value" ;;
    esac
  done < <(load_diffusion_vlm_config "$vlm_config")

  local vlm_model_id="${DIFFUSION_VLM_MODEL_ID:-$config_model_id}"
  local vlm_max_side="${DIFFUSION_VLM_MAX_SIDE:-$config_max_side}"
  local vlm_max_new_tokens="${DIFFUSION_VLM_MAX_NEW_TOKENS:-$config_max_new_tokens}"
  local vlm_timeout="${DIFFUSION_VLM_TIMEOUT:-$config_timeout}"

  echo "=== Phase 3: diffusion VLM semantic assessment (${pair_count} pairs) ==="
  echo "Using diffusion VLM assessment config ${vlm_config}"
  run_with_timeout "$vlm_timeout" env -u HF_HUB_OFFLINE \
    python tools/evaluate_diffusion_vlm_similarity.py \
      --artifacts-dir e2e_artifacts/artifacts \
      --output e2e_artifacts/diffusion_vlm_assessment.json \
      --config "$vlm_config" \
      --model-id "$vlm_model_id" \
      --max-side "$vlm_max_side" \
      --max-new-tokens "$vlm_max_new_tokens"
}

generate_coverage_map() {
  if [ "${RUN_COVERAGE_MAP:-false}" != "true" ]; then
    echo "Skipping: coverage map generation was not requested"
    return 0
  fi

  python -m pip install --disable-pip-version-check --quiet "coverage[toml]==7.6.10" "pytest-cov>=6.0" "gcovr==8.2"
  local cpp_coverage_build_dir="${CPP_COVERAGE_BUILD_DIR:-$PWD/build-cov}"
  run_with_timeout "${COVERAGE_MAP_TIMEOUT:-90m}" python -m tools.coverage_map.generate --output coverage_map.json --python-bin python --build-dir "$cpp_coverage_build_dir"
  python -m tools.coverage_map.generate --validate coverage_map.json
  python -c "import json; d=json.load(open('coverage_map.json')); m=d['meta']; print('Python tests: %s, C++ tests: %s, Source files: %d' % (m['python_tests'], m['cpp_tests'], len(d['source_to_tests'])))"
}

current_python_wheel_tag() {
  python - <<'PY'
import sys
print(f"py{sys.version_info.major}{sys.version_info.minor}")
PY
}

select_compatible_wheel() {
  local wheel_dir="${1:-dist}"
  local py_tag
  py_tag="$(current_python_wheel_tag)"
  local wheel_arch="${TRTMC_PACKAGE_WHEEL_ARCH:-manylinux_2_39_aarch64}"
  mapfile -t candidates < <(
    find "$wheel_dir" -maxdepth 1 -type f \( \
      -name "*-${py_tag}-none-${wheel_arch}.whl" -o \
      -name "*-py3-none-${wheel_arch}.whl" -o \
      -name "*-${py_tag}-none-linux_aarch64.whl" -o \
      -name "*-py3-none-linux_aarch64.whl" \
    \) | sort
  )
  if [ "${#candidates[@]}" -ne 1 ]; then
    printf 'ERROR: expected exactly one %s-compatible Linux aarch64 wheel under %s, found %d\n' \
      "$py_tag" "$wheel_dir" "${#candidates[@]}" >&2
    printf '  %s\n' "${candidates[@]:-}" >&2
    exit 1
  fi
  printf '%s\n' "${candidates[0]}"
}

select_wheel_by_tag() {
  local py_tag="$1"
  local wheel_dir="${2:-dist}"
  local wheel_arch="${TRTMC_PACKAGE_WHEEL_ARCH:-manylinux_2_39_aarch64}"
  mapfile -t candidates < <(
    find "$wheel_dir" -maxdepth 1 -type f \( \
      -name "*-${py_tag}-none-${wheel_arch}.whl" -o \
      -name "*-${py_tag}-none-linux_aarch64.whl" \
    \) | sort
  )
  if [ "${#candidates[@]}" -ne 1 ]; then
    printf 'ERROR: expected exactly one %s Linux aarch64 wheel under %s, found %d\n' \
      "$py_tag" "$wheel_dir" "${#candidates[@]}" >&2
    printf '  %s\n' "${candidates[@]:-}" >&2
    exit 1
  fi
  printf '%s\n' "${candidates[0]}"
}

validate_manylinux_build_environment() {
  local wheel_arch="$1"
  if [[ "$wheel_arch" =~ ^manylinux_2_([0-9]+)_aarch64$ ]]; then
    local max_glibc_minor="${BASH_REMATCH[1]}"
    local glibc_version
    glibc_version="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
    local glibc_major="${glibc_version%%.*}"
    local glibc_minor="${glibc_version#*.}"
    glibc_minor="${glibc_minor%%.*}"

    if ! [[ "$glibc_major" =~ ^[0-9]+$ && "$glibc_minor" =~ ^[0-9]+$ ]]; then
      echo "ERROR: could not parse build image glibc version: ${glibc_version}" >&2
      exit 1
    fi
    if [ "$glibc_major" -gt 2 ] || \
      { [ "$glibc_major" -eq 2 ] && [ "$glibc_minor" -gt "$max_glibc_minor" ]; }; then
      echo "ERROR: ${wheel_arch} requires building on glibc 2.${max_glibc_minor} or older; this image has glibc ${glibc_version}." >&2
      echo "Use TRTMC_CI_IMAGE built from the repository Dockerfile, or another image whose glibc is no newer than the requested wheel tag." >&2
      exit 1
    fi
    echo "manylinux build target=${wheel_arch} build_glibc=${glibc_version}"
  fi

  if ! command -v patchelf >/dev/null 2>&1; then
    echo "ERROR: patchelf is required in the release wheel build image" >&2
    exit 1
  fi
}

locate_conan_cmake_build_dir() {
  local conan_out="$1"
  mapfile -t cmake_caches < <(
    find "$conan_out/build" -mindepth 2 -maxdepth 2 -name CMakeCache.txt -type f | sort
  )
  if [ "${#cmake_caches[@]}" -ne 1 ]; then
    printf 'ERROR: expected exactly one reusable CMakeCache.txt under %s, found %d\n' \
      "$conan_out" "${#cmake_caches[@]}" >&2
    printf '  %s\n' "${cmake_caches[@]:-}" >&2
    exit 1
  fi
  dirname "${cmake_caches[0]}"
}

build_pip_package() {
  local trt_include="${TRTMC_TRT_INCLUDE_DIR:-${TRT_INC_DIR:-}}"
  local trt_library="${TRTMC_TRT_LIBRARY:-}"
  if [ -z "$trt_library" ] && [ -n "${TRT_LIB_DIR:-}" ]; then
    trt_library="${TRT_LIB_DIR%/}/libnvinfer.so"
  fi
  if [ -z "$trt_library" ]; then
    for candidate in \
      /opt/venv/lib/python*/site-packages/tensorrt_libs/libnvinfer.so \
      /usr/lib/aarch64-linux-gnu/libnvinfer.so \
      /usr/lib/x86_64-linux-gnu/libnvinfer.so \
      /usr/local/tensorrt/lib/libnvinfer.so; do
      if [ -f "$candidate" ]; then
        trt_library="$candidate"
        break
      fi
    done
  fi
  if [ -z "$trt_include" ]; then
    for candidate in /usr/local/tensorrt/include /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include; do
      if [ -f "$candidate/NvInfer.h" ]; then
        trt_include="$candidate"
        break
      fi
    done
  fi

  local cuda_include="${TRTMC_CUDA_INCLUDE_DIR:-/usr/local/cuda/include}"
  local cudart_library="${TRTMC_CUDART_LIBRARY:-/usr/local/cuda/lib64/libcudart.so}"

  : "${trt_include:?TensorRT include directory was not found}"
  : "${trt_library:?TensorRT libnvinfer.so was not found}"
  : "${cuda_include:?CUDA include directory was not found}"
  : "${cudart_library:?CUDA runtime library was not found}"

  python -m pip install --disable-pip-version-check --quiet "auditwheel>=6.2" "build>=1.2"
  ensure_ci_state_dir
  local package_build_root="${TRTMC_PACKAGE_BUILD_ROOT:-$PWD/.ci/conan-py-wheel-${GITHUB_RUN_ID:-local}}"
  rm -rf dist "$package_build_root" "$(wheel_build_metadata_file)" "$(wheel_install_metadata_file)" \
    python/tensorrt_model_connect/build python/tensorrt_model_connect/*.egg-info
  find python/tensorrt_model_connect -type d -name __pycache__ -prune -exec rm -rf {} +
  mkdir -p dist

  local python_tags="${TRTMC_PACKAGE_PYTHON_TAGS:-py310 py312}"
  local wheel_arch="${TRTMC_PACKAGE_WHEEL_ARCH:-manylinux_2_39_aarch64}"
  validate_manylinux_build_environment "$wheel_arch"
  local current_tag
  current_tag="$(current_python_wheel_tag)"
  local reuse_tag=""
  local reuse_conan_out=""
  local reuse_cmake_build_dir=""
  local expected_wheels=0
  local tag
  for tag in $python_tags; do
    expected_wheels=$((expected_wheels + 1))
    rm -rf "$package_build_root/$tag" python/tensorrt_model_connect/build python/tensorrt_model_connect/*.egg-info
    env \
      CONAN_PY_BUILD_PROFILE_AUTODETECT=1 \
      TRTMC_TRT_INCLUDE_DIR="$trt_include" \
      TRTMC_TRT_LIBRARY="$trt_library" \
      TRTMC_CUDA_INCLUDE_DIR="$cuda_include" \
      TRTMC_CUDART_LIBRARY="$cudart_library" \
      TRTMC_CONAN_ENABLE_TEST_TARGETS=1 \
      WHEEL_PYVER="$tag" \
      WHEEL_ABI=none \
      WHEEL_ARCH="$wheel_arch" \
      python -m build --wheel --outdir "$PWD/dist" \
        -C "build-dir=$package_build_root/$tag" \
        .
    local tag_conan_out="$package_build_root/$tag/conan_out"
    local tag_cmake_build_dir
    tag_cmake_build_dir="$(locate_conan_cmake_build_dir "$tag_conan_out")"
    if [ -z "$reuse_tag" ] || [ "$tag" = "$current_tag" ]; then
      reuse_tag="$tag"
      reuse_conan_out="$tag_conan_out"
      reuse_cmake_build_dir="$tag_cmake_build_dir"
    fi
  done

  mapfile -t wheels < <(find dist -maxdepth 1 -type f -name '*.whl' | sort)
  if [ "${#wheels[@]}" -ne "$expected_wheels" ]; then
    printf 'ERROR: expected %d wheels, found %d\n' "$expected_wheels" "${#wheels[@]}" >&2
    printf '  %s\n' "${wheels[@]:-}" >&2
    exit 1
  fi

  TRTMC_PACKAGE_WHEEL_ARCH="$wheel_arch" python - "${wheels[@]}" <<'PY'
import os
import re
import subprocess
import sys
import zipfile

EXPECTED_PLATFORM = os.environ.get("TRTMC_PACKAGE_WHEEL_ARCH", "manylinux_2_39_aarch64")
platform_match = re.fullmatch(r"manylinux_2_([0-9]+)_aarch64", EXPECTED_PLATFORM)
if not platform_match:
    raise SystemExit(f"expected a manylinux aarch64 platform tag, got {EXPECTED_PLATFORM}")
MAX_GLIBC_MINOR = int(platform_match.group(1))

for wheel in sys.argv[1:]:
    if not wheel.endswith(f"-{EXPECTED_PLATFORM}.whl"):
        raise SystemExit(f"{wheel}: expected platform tag {EXPECTED_PLATFORM}")
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        if any(name.startswith(".data/purelib/") for name in names):
            raise SystemExit(f"{wheel}: native wheel must not contain .data/purelib entries")
        if any(".data/purelib/" in name for name in names):
            raise SystemExit(f"{wheel}: native wheel must not install package files via purelib")
        bin_entries = [name for name in names if name.endswith("/bin/trtmc")]
        script_entries = [name for name in names if name.endswith(".data/scripts/trtmc")]
        backend_entries = [
            name for name in names if "/bin/libtrtmc_backend" in name and name.endswith(".so")
        ]
        metadata = zf.read(
            next(name for name in names if name.endswith(".dist-info/METADATA"))
        ).decode()
        wheel_metadata = zf.read(
            next(name for name in names if name.endswith(".dist-info/WHEEL"))
        ).decode()
    if len(bin_entries) != 1:
        raise SystemExit(f"{wheel}: expected one packaged trtmc executable")
    if len(script_entries) != 1:
        raise SystemExit(f"{wheel}: expected one native trtmc script executable")
    if any(name.endswith(".dist-info/entry_points.txt") for name in names):
        raise SystemExit(f"{wheel}: native trtmc must be installed directly, not via console_scripts")
    if not backend_entries:
        raise SystemExit(f"{wheel}: packaged native TensorRT backend DSO is missing")
    trt_dependency = "Requires-Dist: tensorrt==11.2.0.113"
    if trt_dependency not in metadata:
        raise SystemExit(
            f"{wheel}: pinned TensorRT 11.2.0.113 dependency metadata is missing"
        )
    if f"-{EXPECTED_PLATFORM}" not in wheel_metadata:
        raise SystemExit(f"{wheel}: WHEEL metadata is missing {EXPECTED_PLATFORM}")
    audit = subprocess.run(
        [sys.executable, "-m", "auditwheel", "show", wheel],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout
    print(audit, end="")
    manylinux_minors = [
        int(match)
        for line in audit.splitlines()
        if "platform tag" in line
        for match in re.findall(r"manylinux_2_([0-9]+)_aarch64", line)
    ]
    if not manylinux_minors or max(manylinux_minors) > MAX_GLIBC_MINOR:
        raise SystemExit(
            f"{wheel}: auditwheel did not confirm compatibility with "
            f"manylinux_2_{MAX_GLIBC_MINOR}_aarch64 or older"
        )
    print(f"validated wheel={wheel}")
    for entry in sorted([*bin_entries, *script_entries, *backend_entries]):
        print(f"  {entry}")
PY

  {
    write_shell_var TRTMC_REUSE_WHEEL_TAG "$reuse_tag"
    write_shell_var TRTMC_REUSE_CONAN_OUT_DIR "$reuse_conan_out"
    write_shell_var TRTMC_REUSE_CMAKE_BUILD_DIR "$reuse_cmake_build_dir"
    write_shell_var TRTMC_TRT_INCLUDE_DIR "$trt_include"
    write_shell_var TRTMC_TRT_LIBRARY "$trt_library"
    write_shell_var TRTMC_CUDA_INCLUDE_DIR "$cuda_include"
    write_shell_var TRTMC_CUDART_LIBRARY "$cudart_library"
  } > "$(wheel_build_metadata_file)"
  echo "Reusable wheel build metadata:"
  cat "$(wheel_build_metadata_file)"

  local install_wheel
  install_wheel="$(select_compatible_wheel dist)"
  local smoke_venv="/tmp/trtmc-wheel-smoke-${GITHUB_RUN_ID:-local}"
  rm -rf "$smoke_venv"
  python -m venv "$smoke_venv"
  "$smoke_venv/bin/python" -m pip install --disable-pip-version-check --upgrade pip
  "$smoke_venv/bin/python" -m pip install --disable-pip-version-check "$install_wheel"
  "$smoke_venv/bin/python" - "$smoke_venv/bin/trtmc" <<'PY'
from pathlib import Path
import sys

trtmc = Path(sys.argv[1])
if trtmc.read_bytes()[:4] != b"\x7fELF":
    raise SystemExit(f"{trtmc} is not the native ELF trtmc executable")
PY
  "$smoke_venv/bin/trtmc" version
  "$smoke_venv/bin/trtmc" --help >/tmp/trtmc-help.txt
  "$smoke_venv/bin/trtmc" build --help >/tmp/trtmc-build-help.txt
  "$smoke_venv/bin/python" - <<'PY'
import importlib.metadata as metadata
import importlib.resources as resources
from pathlib import Path

dist = metadata.distribution("tensorrt-model-connect")
native_dir = Path(resources.files("tensorrt_model_connect").joinpath("bin"))
native = native_dir / "trtmc"
backends = sorted(native_dir.glob("libtrtmc_backend*.so*"))
print(f"wheel={dist.metadata['Name']} {dist.version}")
print(f"native_trtmc={native}")
if not native.is_file():
    raise SystemExit("packaged native trtmc executable is missing")
if not backends:
    raise SystemExit("packaged native TensorRT backend DSO is missing")
for backend in backends:
    print(f"native_backend={backend}")
PY
}

select_wheel_smoke_config() {
  python - "${TRTMC_WHEEL_SMOKE_CONFIG:-}" <<'PY'
import json
import sys
from pathlib import Path

requested = sys.argv[1]
if requested:
    path = Path(requested)
    if not path.is_file():
        raise SystemExit(f"TRTMC_WHEEL_SMOKE_CONFIG does not exist: {path}")
    print(path)
    raise SystemExit(0)

configs = sorted(Path("tests/e2e/models").glob("*/package_smoke.json"))
defaults = []
for path in configs:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Invalid package smoke config {path}: {exc}") from exc
    if data.get("default") is True:
        defaults.append(path)

if len(defaults) != 1:
    listed = ", ".join(str(path) for path in defaults) or "none"
    raise SystemExit(
        "Expected exactly one default package smoke config under "
        f"tests/e2e/models/*/package_smoke.json; found {listed}"
    )
print(defaults[0])
PY
}

load_wheel_smoke_config() {
  local config_path="$1"
  python - "$config_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))

required = ("name", "model_id", "bundle", "timing_cache", "prompt", "precision")
missing = [key for key in required if not data.get(key)]
if missing:
    raise SystemExit(f"{path} missing required package smoke fields: {missing}")

for key in (
    "name",
    "model_id",
    "bundle",
    "timing_cache",
    "max_cache",
    "max_new_tokens",
    "optimization_level",
    "build_timeout",
    "run_timeout",
    "precision",
    "prompt",
):
    value = data.get(key, "")
    print(f"{key}\t{value}")

run_args = data.get("run_args", [])
if not isinstance(run_args, list) or not all(isinstance(item, str) for item in run_args):
    raise SystemExit(f"{path} field run_args must be a list of strings")
for value in run_args:
    print(f"run_arg\t{value}")
PY
}

run_wheel_model_smoke() {
  local wheel
  wheel="$(select_wheel_by_tag py312 dist)"

  python - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        "Python 3.12 is required for the py312 wheel model smoke test; "
        f"got {sys.version.split()[0]} from {sys.executable}"
    )
PY

  local smoke_config
  smoke_config="$(select_wheel_smoke_config)"

  local smoke_name=""
  local config_model_id=""
  local config_bundle=""
  local config_timing_cache=""
  local config_max_cache=""
  local config_max_new_tokens=""
  local config_optimization_level=""
  local config_build_timeout=""
  local config_run_timeout=""
  local config_precision=""
  local config_prompt=""
  local -a config_run_args=()
  local key value
  while IFS=$'\t' read -r key value; do
    case "$key" in
      name) smoke_name="$value" ;;
      model_id) config_model_id="$value" ;;
      bundle) config_bundle="$value" ;;
      timing_cache) config_timing_cache="$value" ;;
      max_cache) config_max_cache="$value" ;;
      max_new_tokens) config_max_new_tokens="$value" ;;
      optimization_level) config_optimization_level="$value" ;;
      build_timeout) config_build_timeout="$value" ;;
      run_timeout) config_run_timeout="$value" ;;
      precision) config_precision="$value" ;;
      prompt) config_prompt="$value" ;;
      run_arg) config_run_args+=("$value") ;;
    esac
  done < <(load_wheel_smoke_config "$smoke_config")

  local smoke_root="/tmp/trtmc-wheel-model-smoke-${GITHUB_RUN_ID:-local}"
  local smoke_venv="${smoke_root}/venv"
  local bundle="${smoke_root}/${config_bundle}"
  local timing_cache="${smoke_root}/${config_timing_cache}"
  local model_id="${TRTMC_WHEEL_SMOKE_MODEL_ID:-$config_model_id}"
  local max_cache="${TRTMC_WHEEL_SMOKE_MAX_CACHE:-$config_max_cache}"
  local max_new_tokens="${TRTMC_WHEEL_SMOKE_MAX_NEW_TOKENS:-$config_max_new_tokens}"
  local optimization_level="${TRTMC_WHEEL_SMOKE_OPTIMIZATION_LEVEL:-$config_optimization_level}"
  local build_timeout="${TRTMC_WHEEL_SMOKE_BUILD_TIMEOUT:-$config_build_timeout}"
  local run_timeout="${TRTMC_WHEEL_SMOKE_RUN_TIMEOUT:-$config_run_timeout}"

  echo "Running wheel model smoke '${smoke_name}' from ${smoke_config}"

  rm -rf "$smoke_root"
  mkdir -p "$smoke_root"
  python -m venv "$smoke_venv"
  "$smoke_venv/bin/python" -m pip install --disable-pip-version-check --upgrade pip
  "$smoke_venv/bin/python" -m pip install --disable-pip-version-check "$wheel"
  "$smoke_venv/bin/python" -m pip check
  "$smoke_venv/bin/python" - "$smoke_venv/bin/trtmc" <<'PY'
from pathlib import Path
import sys

trtmc = Path(sys.argv[1])
if trtmc.read_bytes()[:4] != b"\x7fELF":
    raise SystemExit(f"{trtmc} is not the native ELF trtmc executable")
PY
  env -u VIRTUAL_ENV -u CONDA_PREFIX -u TRTMC_TRT_LIBRARY_DIR -u LD_LIBRARY_PATH \
    "$smoke_venv/bin/trtmc" version

  run_with_timeout "$build_timeout" \
    env -u VIRTUAL_ENV -u CONDA_PREFIX -u TRTMC_TRT_LIBRARY_DIR -u LD_LIBRARY_PATH \
      TRTMC_TRT_TIMING_CACHE_PATH="$timing_cache" \
      TRTMC_BUILDER_OPTIMIZATION_LEVEL="$optimization_level" \
      "$smoke_venv/bin/trtmc" build "$model_id" \
        -o "$bundle" \
        --max-cache-length "$max_cache" \
        --precision "$config_precision"

  env -u VIRTUAL_ENV -u CONDA_PREFIX -u TRTMC_TRT_LIBRARY_DIR -u LD_LIBRARY_PATH \
    "$smoke_venv/bin/trtmc" inspect --list-engines "$bundle"

  run_with_timeout "$run_timeout" \
    env -u VIRTUAL_ENV -u CONDA_PREFIX -u TRTMC_TRT_LIBRARY_DIR -u LD_LIBRARY_PATH \
      "$smoke_venv/bin/trtmc" run "$bundle" \
        --prompt "$config_prompt" \
        --max-new-tokens "$max_new_tokens" \
        "${config_run_args[@]}"
}

run_stage() {
  local stage="$1"
  case "$stage" in
    setup)
      run_step "Install trtmc pip package" install_built_wheel_once
      ;;
    impact)
      run_step "Setup TensorRT-Model-Connect source checks" setup_source_check_environment
      run_step "Impact analysis" impact_analysis
      ;;
    build)
      run_step "Setup TensorRT-Model-Connect source checks" setup_source_check_environment
      run_step "Build C++ test executables" build_cpp_test_executables
      ;;
    family-coverage)
      run_step "Setup TensorRT-Model-Connect source checks" setup_source_check_environment
      run_step "Check family coverage" check_family_coverage
      ;;
    complexity)
      run_step "Setup TensorRT-Model-Connect source checks" setup_source_check_environment
      run_step "Check cyclomatic complexity" check_cyclomatic_complexity
      ;;
    lint)
      run_step "Setup TensorRT-Model-Connect source checks" setup_source_check_environment
      run_step "Lint changed files" lint_changed_files
      ;;
    cpp-unit)
      run_step "Setup TensorRT-Model-Connect source checks" setup_source_check_environment
      run_step "C++ unit tests" run_cpp_unit_tests
      ;;
    python-builder)
      run_step "Setup TensorRT-Model-Connect wheel runtime" setup_wheel_runtime_environment
      run_step "Python builder and tools tests" run_python_builder_tests
      ;;
    cpp-coverage)
      run_step "Setup TensorRT-Model-Connect source checks" setup_source_check_environment
      run_step "C++ coverage" run_cpp_coverage
      ;;
    graph-ops)
      run_step "Setup TensorRT-Model-Connect wheel runtime" setup_wheel_runtime_environment
      run_step "Graph-op GPU tests" run_graph_op_tests
      ;;
    selective-e2e)
      run_step "Setup TensorRT-Model-Connect wheel runtime" setup_wheel_runtime_environment
      run_step "Selective E2E tests" run_selective_e2e
      ;;
    full-e2e)
      run_step "Setup TensorRT-Model-Connect wheel runtime" setup_wheel_runtime_environment
      run_step "Full E2E tests" run_full_e2e
      ;;
    coverage-map)
      run_step "Setup TensorRT-Model-Connect wheel runtime" setup_wheel_runtime_environment
      run_step "Generate coverage map" generate_coverage_map
      ;;
    package)
      run_step "Setup TensorRT-Model-Connect package build environment" setup_package_build_environment
      run_step "Build trtmc pip package" build_pip_package
      run_step "Install trtmc pip package" install_built_wheel_once
      ;;
    wheel-model-smoke)
      run_step "Setup TensorRT-Model-Connect source checks" setup_source_check_environment
      run_step "Model smoke test from trtmc pip package" run_wheel_model_smoke
      ;;
    *)
      echo "ERROR: Unknown CI stage: $stage" >&2
      exit 2
      ;;
  esac
}

if [ "$#" -gt 0 ]; then
  run_stage "$1"
  exit 0
fi

run_step "Setup TensorRT-Model-Connect package build environment" setup_package_build_environment
run_step "Build trtmc pip package" build_pip_package
run_step "Install trtmc pip package" install_built_wheel_once
run_step "Impact analysis" impact_analysis
run_step "Build C++ test executables" build_cpp_test_executables
run_step "Check family coverage" check_family_coverage
run_step "Check cyclomatic complexity" check_cyclomatic_complexity
run_step "Lint changed files" lint_changed_files
run_step "C++ unit tests" run_cpp_unit_tests
run_step "Python builder and tools tests" run_python_builder_tests
run_step "C++ coverage" run_cpp_coverage
run_step "Graph-op GPU tests" run_graph_op_tests
run_step "Selective E2E tests" run_selective_e2e
run_step "Full E2E tests" run_full_e2e
run_step "Generate coverage map" generate_coverage_map
run_step "Model smoke test from trtmc pip package" run_wheel_model_smoke
