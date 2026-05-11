#!/usr/bin/env bash
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

setup_environment() {
  git config --global --add safe.directory "${GITHUB_WORKSPACE:-$PWD}" || true
  git config --global --add safe.directory "*" || true
  echo "ENGINE_DIR=${ENGINE_DIR:-}"
  echo "HF_HOME=${HF_HOME:-}"
  echo "HF_HUB_CACHE=${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-}}"
  echo "HF_MODULES_CACHE=${HF_MODULES_CACHE:-}"
  python -c "import transformers, sys; print(f'python={sys.executable} transformers={transformers.__version__}'); assert transformers.__version__ == '5.2.0', transformers.__version__"
  python -m pip install --disable-pip-version-check --no-deps -e tensorrt_model_connect/
  chmod +x ./build/trtmc 2>/dev/null || true
}

impact_analysis() {
  python3 tools/test_impact.py --validate
  python3 tools/coverage_map/fetch_latest.py \
    --output coverage_map.json \
    --local-fallback "${COVERAGE_MAP_PATH:-}" \
    || echo "No coverage map available -- using tier-level selection"

  if [ -f coverage_map.json ]; then
    python3 tools/test_impact.py --base "$CI_BASE_REF" --coverage-map coverage_map.json --json > impact.json
  else
    python3 tools/test_impact.py --base "$CI_BASE_REF" --json > impact.json
  fi

  echo "--- Impact Analysis ---"
  cat impact.json
  python3 tools/test_impact.py --base "$CI_BASE_REF" --verbose
}

build_all() {
  local trt_include="${TRTMC_TRT_INCLUDE_DIR:-${TRT_INC_DIR:-}}"
  local trt_library="${TRTMC_TRT_LIBRARY:-}"
  if [ -z "$trt_library" ] && [ -n "${TRT_LIB_DIR:-}" ]; then
    trt_library="${TRT_LIB_DIR%/}/libnvinfer.so"
  fi
  if [ -z "$trt_library" ]; then
    for candidate in \
      /opt/venv/lib/python*/site-packages/tensorrt_libs/libnvinfer.so \
      /usr/lib/x86_64-linux-gnu/libnvinfer.so \
      /usr/local/tensorrt/lib/libnvinfer.so; do
      if [ -f "$candidate" ]; then
        trt_library="$candidate"
        break
      fi
    done
  fi
  if [ -z "$trt_include" ]; then
    for candidate in /usr/local/tensorrt/include /usr/include/x86_64-linux-gnu /usr/include; do
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

  run_with_timeout "${BUILD_ALL_TIMEOUT:-15m}" cmake -S . -B build -G Ninja \
    -DTRTMC_TRT_INCLUDE_DIR="$trt_include" \
    -DTRTMC_TRT_LIBRARY="$trt_library" \
    -DTRTMC_CUDA_INCLUDE_DIR="$cuda_include" \
    -DTRTMC_CUDART_LIBRARY="$cudart_library"
  run_with_timeout "${BUILD_ALL_TIMEOUT:-15m}" cmake --build build -j
}

check_family_coverage() {
  python scripts/check_family_coverage.py
}

check_cyclomatic_complexity() {
  lizard --version
  python tools/check_cyclomatic_complexity.py src --max-ccn "${CCM_MAX_CCN:-10}" --top 20
}

lint_changed_files() {
  if ! command -v ruff >/dev/null 2>&1 || ! command -v clang-format >/dev/null 2>&1; then
    python -m pip install --disable-pip-version-check --quiet ruff clang-format
  fi

  local base_ref="$CI_BASE_REF"
  if [ "${GITHUB_EVENT_NAME:-}" = "workflow_dispatch" ] || [ "${GITHUB_EVENT_NAME:-}" = "schedule" ]; then
    base_ref="origin/${GITHUB_REF_NAME:-main}"
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

  if [ -n "$cpp_tests" ]; then
    echo "Selective C++ tests: $cpp_tests"
    run_with_timeout "${CPP_UNIT_TIMEOUT:-20m}" ctest --test-dir build -R "$cpp_tests" --output-on-failure
  else
    echo "Running all C++ tests"
    run_with_timeout "${CPP_UNIT_TIMEOUT:-20m}" ctest --test-dir build --output-on-failure
  fi
}

run_python_builder_tests() {
  python -m pip install --disable-pip-version-check --quiet "pytest-cov>=6.0"
  mkdir -p coverage

  if [ "${FULL_E2E:-false}" != "true" ]; then
    if ! python3 -c "import json; d=json.load(open('impact.json')); tiers=d['unit_tiers']; exit(0 if 'builder' in tiers or 'tools' in tiers else 1)"; then
      echo "Skipping: neither builder nor tools tier affected by this change"
      echo '<?xml version="1.0" ?><coverage version="7.6" timestamp="0" lines-valid="0" lines-covered="0" line-rate="1.0" branches-valid="0" branches-covered="0" branch-rate="1.0" complexity="0"><packages/></coverage>' > coverage/python-cobertura.xml
      echo "PYTHON_COVERAGE_LINE=100.00%"
      echo "PYTHON_COVERAGE_BRANCH=100.00%"
      echo "Skipped" > coverage/python-coverage.txt
      return 0
    fi
  fi

  local builder_tests=""
  if [ "${FULL_E2E:-false}" != "true" ]; then
    builder_tests=$(python3 -c "
import json, sys
d = json.load(open('impact.json'))
tests = d.get('builder_tests', [])
fallback = d.get('fallback_tiers', [])
if 'builder' in fallback or not tests:
    sys.exit(0)
print(' or '.join(tests))
" 2>/dev/null) || true
  fi

  local cov_args="--cov=tensorrt_model_connect/tensorrt_model_connect --cov-branch --cov-report=term-missing --cov-report=xml:coverage/python-cobertura.xml"
  if [ -n "$builder_tests" ]; then
    echo "Selective builder tests: $builder_tests"
    run_with_timeout "${PYTHON_BUILDER_TIMEOUT:-40m}" python -m pytest tests/builder/ tests/tools/ tests/engine_defs/torch_trt/ -v \
      --ignore=tests/builder/test_cli.py \
      --ignore=tests/engine_defs/torch_trt/test_pixart_vs_hf.py \
      -k "$builder_tests" -n auto $cov_args
  else
    echo "Running all builder + tools tests"
    run_with_timeout "${PYTHON_BUILDER_TIMEOUT:-40m}" python -m pytest tests/builder/ tests/tools/ tests/engine_defs/torch_trt/ -v \
      --ignore=tests/builder/test_cli.py \
      --ignore=tests/engine_defs/torch_trt/test_pixart_vs_hf.py \
      -n auto $cov_args
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

run_selective_e2e() {
  if [ "${GITHUB_EVENT_NAME:-}" != "pull_request" ] || [ "${FULL_E2E:-false}" = "true" ]; then
    echo "Skipping: selective E2E only runs for pull_request events without full_e2e"
    return 0
  fi

  python3 -c "
import json
d = json.load(open('impact.json'))
models = d.get('e2e_models', [])
with open('e2e_models.txt', 'w') as f:
    for m in models:
        f.write(m + '\n')
print(f'Selective E2E: {len(models)} models')
for m in models[:10]:
    print(f'  {m}')
if len(models) > 10:
    print(f'  ... and {len(models) - 10} more')
"
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
  env -u HF_HUB_OFFLINE python scripts/warm_hf_cache.py --models-file e2e_models.txt
  echo "=== Phase 2: parallel rebuild (offline, local cache) ==="
  local args=(
    --engine-dir "$ENGINE_DIR"
    --result-dir e2e_artifacts
    --trtmc-binary ./build/trtmc
    --workers-per-gpu 4
    --models-file e2e_models.txt
  )
  if [ "${REBUILD_ENGINES:-true}" = "true" ]; then
    args+=(--rebuild-engines)
  fi
  run_with_timeout "${SELECTIVE_E2E_TIMEOUT:-4h}" env HF_HUB_OFFLINE=1 ./scripts/run_e2e_parallel.sh "${args[@]}"
  run_diffusion_vlm_assessment
}

run_full_e2e() {
  if [ "${FULL_E2E:-false}" != "true" ]; then
    echo "Skipping: full E2E was not requested"
    return 0
  fi

  echo "=== Nightly Full E2E: all models ==="
  nvidia-smi
  configure_e2e_timing_cache
  echo "=== Phase 1: warming HF cache (online, sequential) ==="
  python scripts/warm_hf_cache.py --exclude-ci-tier l0_only
  echo "=== Phase 2: parallel rebuild (offline, local cache) ==="
  local args=(
    --engine-dir "$ENGINE_DIR"
    --result-dir e2e_artifacts
    --trtmc-binary ./build/trtmc
    --workers-per-gpu 4
    --exclude-ci-tier l0_only
  )
  if [ "${REBUILD_ENGINES:-true}" = "true" ]; then
    args+=(--rebuild-engines)
  fi
  run_with_timeout "${FULL_E2E_TIMEOUT:-6h}" env HF_HUB_OFFLINE=1 ./scripts/run_e2e_parallel.sh "${args[@]}"
  run_diffusion_vlm_assessment
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
  pair_count=$(python3 -c '
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
count = 0
for result_path in root.glob("*/result.json"):
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    case = result.get("case_config", {})
    if case.get("task_strategy") != "diffusion_media_generation":
        continue
    model_dir = result_path.parent
    has_trt = (model_dir / "frames").is_dir()
    has_ref = (model_dir / "hf_frames").is_dir() or (model_dir / "ref_frames").is_dir()
    if has_trt and has_ref:
        count += 1
print(count)
' e2e_artifacts/artifacts)
  if [ "${pair_count:-0}" -eq 0 ]; then
    echo "Skipping: no TRT/HF diffusion frame pairs for VLM assessment"
    return 0
  fi

  echo "=== Phase 3: diffusion VLM semantic assessment (${pair_count} pairs) ==="
  run_with_timeout "${DIFFUSION_VLM_TIMEOUT:-45m}" env -u HF_HUB_OFFLINE \
    python tools/evaluate_diffusion_vlm_similarity.py \
      --artifacts-dir e2e_artifacts/artifacts \
      --output e2e_artifacts/diffusion_vlm_assessment.json \
      --model-id "${DIFFUSION_VLM_MODEL_ID:-Qwen/Qwen2.5-VL-3B-Instruct}" \
      --max-side "${DIFFUSION_VLM_MAX_SIDE:-512}" \
      --max-new-tokens "${DIFFUSION_VLM_MAX_NEW_TOKENS:-384}" \
      --waives tests/e2e/waives.txt
}

generate_coverage_map() {
  if [ "${RUN_COVERAGE_MAP:-false}" != "true" ]; then
    echo "Skipping: coverage map generation was not requested"
    return 0
  fi

  python -m pip install --disable-pip-version-check --quiet "coverage[toml]==7.6.10" "pytest-cov>=6.0" "gcovr==8.2"
  run_with_timeout "${COVERAGE_MAP_TIMEOUT:-90m}" python -m tools.coverage_map.generate --output coverage_map.json --python-bin python --build-dir build
  python -m tools.coverage_map.generate --validate coverage_map.json
  python -c "import json; d=json.load(open('coverage_map.json')); m=d['meta']; print('Python tests: %s, C++ tests: %s, Source files: %d' % (m['python_tests'], m['cpp_tests'], len(d['source_to_tests'])))"
}

run_stage() {
  local stage="$1"
  case "$stage" in
    setup)
      run_step "Setup TensorRT-Model-Connect" setup_environment
      ;;
    impact)
      run_step "Setup TensorRT-Model-Connect" setup_environment
      run_step "Impact analysis" impact_analysis
      ;;
    build)
      run_step "Setup TensorRT-Model-Connect" setup_environment
      run_step "Build all" build_all
      ;;
    family-coverage)
      run_step "Setup TensorRT-Model-Connect" setup_environment
      run_step "Check family coverage" check_family_coverage
      ;;
    complexity)
      run_step "Setup TensorRT-Model-Connect" setup_environment
      run_step "Check cyclomatic complexity" check_cyclomatic_complexity
      ;;
    lint)
      run_step "Setup TensorRT-Model-Connect" setup_environment
      run_step "Lint changed files" lint_changed_files
      ;;
    cpp-unit)
      run_step "Setup TensorRT-Model-Connect" setup_environment
      run_step "C++ unit tests" run_cpp_unit_tests
      ;;
    python-builder)
      run_step "Setup TensorRT-Model-Connect" setup_environment
      run_step "Python builder and tools tests" run_python_builder_tests
      ;;
    cpp-coverage)
      run_step "Setup TensorRT-Model-Connect" setup_environment
      run_step "C++ coverage" run_cpp_coverage
      ;;
    graph-ops)
      run_step "Setup TensorRT-Model-Connect" setup_environment
      run_step "Graph-op GPU tests" run_graph_op_tests
      ;;
    selective-e2e)
      run_step "Setup TensorRT-Model-Connect" setup_environment
      run_step "Selective E2E tests" run_selective_e2e
      ;;
    full-e2e)
      run_step "Setup TensorRT-Model-Connect" setup_environment
      run_step "Full E2E tests" run_full_e2e
      ;;
    coverage-map)
      run_step "Setup TensorRT-Model-Connect" setup_environment
      run_step "Generate coverage map" generate_coverage_map
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

run_step "Setup TensorRT-Model-Connect" setup_environment
run_step "Impact analysis" impact_analysis
run_step "Build all" build_all
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
