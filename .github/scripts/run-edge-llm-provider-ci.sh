#!/usr/bin/env bash
set -euo pipefail

artifact_dir="${TRTMC_EDGE_LLM_ARTIFACT_DIR:-edge_llm_provider_artifacts}"
mkdir -p "$artifact_dir"

log_path="$artifact_dir/edge_llm_provider_ci.log"
exec > >(tee "$log_path") 2>&1

edge_src="third_party/tensorrt-edge-llm"
edge_build="${TRTMC_EDGE_LLM_BUILD_DIR:-$PWD/build/edge-llm}"
provider_build="${TRTMC_EDGE_LLM_PROVIDER_BUILD_DIR:-$PWD/build-edge-llm-provider}"
bundle_path="$artifact_dir/edge_llm_delegated.trtfb"
inspect_path="$artifact_dir/inspect.txt"
run_path="$artifact_dir/run.txt"
correctness_path="$artifact_dir/correctness.txt"
benchmark_path="$artifact_dir/benchmark.txt"
metadata_path="$artifact_dir/metadata.txt"
target="${TRTMC_EDGE_LLM_TARGET:-gb300}"
model_dir="${TRTMC_EDGE_LLM_MODEL_DIR:-Qwen/Qwen3-0.6B}"
engine_dir="${TRTMC_EDGE_LLM_ENGINE_DIR:-}"
max_cache_length="${TRTMC_EDGE_LLM_MAX_CACHE_LENGTH:-256}"
precision="${TRTMC_EDGE_LLM_PRECISION:-fp16}"
build_jobs="${TRTMC_EDGE_LLM_BUILD_JOBS:-$(nproc)}"

detect_cuda_arch() {
  python - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print(f"{major}{minor}")
except Exception:
    pass
PY
}

edge_cmake_args=()
if [ -n "${TRTMC_EDGE_LLM_CMAKE_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  edge_cmake_args=(${TRTMC_EDGE_LLM_CMAKE_ARGS})
fi
provider_cmake_args=()
if [ -n "${TRTMC_PROVIDER_CMAKE_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  provider_cmake_args=(${TRTMC_PROVIDER_CMAKE_ARGS})
fi
if [[ " ${edge_cmake_args[*]} " != *"CMAKE_CUDA_ARCHITECTURES"* ]]; then
  detected_cuda_arch="$(detect_cuda_arch)"
  if [ -n "$detected_cuda_arch" ]; then
    edge_cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=$detected_cuda_arch")
    provider_cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=$detected_cuda_arch")
  fi
fi

detect_trt_package_dir() {
  if [ -n "${TRT_PACKAGE_DIR:-}" ]; then
    echo "$TRT_PACKAGE_DIR"
    return 0
  fi
  if [ -n "${TRTMC_TRT_PACKAGE_DIR:-}" ]; then
    echo "$TRTMC_TRT_PACKAGE_DIR"
    return 0
  fi
  for candidate in /usr/local/tensorrt /usr /usr/local; do
    if [ -f "$candidate/include/NvInfer.h" ] || [ -f "$candidate/include/x86_64-linux-gnu/NvInfer.h" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

trt_package_dir="$(detect_trt_package_dir)" || {
  echo "ERROR: TensorRT package root not found. Set TRT_PACKAGE_DIR or TRTMC_TRT_PACKAGE_DIR." >&2
  exit 1
}

echo "Initializing Edge-LLM submodule"
git submodule update --init --recursive "$edge_src"

edge_commit="$(git -C "$edge_src" rev-parse HEAD)"
model_connect_commit="$(git rev-parse HEAD)"
cuda_version="$(nvcc --version 2>/dev/null | tail -n 1 || true)"
trt_version="$(python - <<'PY'
try:
    import tensorrt as trt
    print(trt.__version__)
except Exception:
    print("unknown")
PY
)"

{
  echo "model_connect_commit=$model_connect_commit"
  echo "edge_llm_commit=$edge_commit"
  echo "trt_package_dir=$trt_package_dir"
  echo "tensorrt_version=$trt_version"
  echo "cuda_version=$cuda_version"
  echo "target=$target"
  echo "model_dir=$model_dir"
  echo "engine_dir=$engine_dir"
  echo "edge_cmake_args=${edge_cmake_args[*]}"
  echo "provider_cmake_args=${provider_cmake_args[*]}"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
} > "$metadata_path"

python -m pip install --disable-pip-version-check --no-deps -e tensorrt_model_connect/
python -m pip install --disable-pip-version-check --no-deps -e "$edge_src"

echo "Building TensorRT Edge-LLM libraries"
cmake -S "$edge_src" -B "$edge_build" -G Ninja \
  -DTRT_PACKAGE_DIR="$trt_package_dir" \
  "${edge_cmake_args[@]}"
cmake --build "$edge_build" --target edgellmCore edgellmTokenizer NvInfer_edgellm_plugin llm_build -j "$build_jobs"

echo "Building Model-Connect Edge-LLM provider DSO through submodule root resolution"
cmake -S . -B "$provider_build" -G Ninja \
  -DTRTMC_ENABLE_EDGE_LLM_PROVIDER=ON \
  -DTRTMC_EDGE_LLM_BUILD_DIR="$edge_build" \
  "${provider_cmake_args[@]}"
cmake --build "$provider_build" --target trtmc trtmc_provider_edgellm -j "$build_jobs"

export EDGELLM_PLUGIN_PATH="$edge_build/libNvInfer_edgellm_plugin.so"
export TRTMC_EDGE_LLM_PROVIDER_LIBRARY="$provider_build/libtrtmc_provider_edgellm.so"
export LD_LIBRARY_PATH="$provider_build:$edge_build:${LD_LIBRARY_PATH:-}"

build_sets=(
  --set "deployment.provider=tensorrt-edge-llm"
  --set "deployment.target=$target"
)
if [ -n "$engine_dir" ]; then
  build_sets+=(--set "deployment.edge_llm_engine_dir=$engine_dir")
else
  export TRTMC_EDGE_LLM_BUILD_TOOL="$edge_build/examples/llm/llm_build"
  export TRTMC_EDGE_LLM_EXPORT_TOOL="${TRTMC_EDGE_LLM_EXPORT_TOOL:-tensorrt-edgellm-export-llm}"
  build_sets+=(--set "deployment.edge_llm_workspace=$artifact_dir/edge_llm_workspace")
  build_sets+=(--set "deployment.edge_llm_build_tool=$TRTMC_EDGE_LLM_BUILD_TOOL")
  build_sets+=(--set "deployment.edge_llm_export_tool=$TRTMC_EDGE_LLM_EXPORT_TOOL")
fi

echo "Packaging Edge-LLM delegated bundle"
trtmc-build build "$model_dir" \
  -o "$bundle_path" \
  --max-cache-length "$max_cache_length" \
  --precision "$precision" \
  "${build_sets[@]}" \
  --verbose

echo "Inspecting deployment manifest"
"$provider_build/trtmc" inspect "$bundle_path" --deployment | tee "$inspect_path"

echo "Running Edge-LLM delegated inference"
"$provider_build/trtmc" run "$bundle_path" \
  --prompt "${TRTMC_EDGE_LLM_PROMPT:-The capital of France is}" \
  --max-new-tokens "${TRTMC_EDGE_LLM_MAX_NEW_TOKENS:-20}" \
  --runtime-cache "$artifact_dir/runtime_cache" \
  --benchmark "${TRTMC_EDGE_LLM_BENCHMARK_ITERS:-3}" \
  --warmup "${TRTMC_EDGE_LLM_WARMUP_ITERS:-1}" | tee "$run_path"

if [ ! -s "$run_path" ]; then
  echo "ERROR: Edge-LLM delegated inference produced empty output" >&2
  exit 1
fi
python - "$run_path" "$correctness_path" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

run_path = Path(sys.argv[1])
correctness_path = Path(sys.argv[2])
lines = run_path.read_text(encoding="utf-8", errors="replace").splitlines()
text_lines = [
    line.strip()
    for line in lines
    if line.strip() and not line.lstrip().startswith("[")
]
if not text_lines:
    raise SystemExit("ERROR: Edge-LLM delegated inference produced no text output")
correctness_path.write_text(
    "comparator=non_empty_text_output\n"
    "result=pass\n"
    f"sample={text_lines[-1]}\n",
    encoding="utf-8",
)
PY
if ! grep "\\[trtmc.benchmark\\].*tokens_per_sec=.*sampled_peak_gpu" "$log_path" > "$benchmark_path"; then
  echo "ERROR: Edge-LLM delegated inference did not record benchmark metrics" >&2
  exit 1
fi

echo "Edge-LLM provider CI completed"
