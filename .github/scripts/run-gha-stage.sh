#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: run-gha-stage.sh <stage>}"

docker image inspect "$TRTMC_CI_IMAGE" >/dev/null || {
  echo "::error::Docker image '$TRTMC_CI_IMAGE' is not present on the self-hosted runner. Set repository variable TRTMC_CI_IMAGE if the runner uses a different local image tag."
  exit 1
}

extra_mounts=()
if [ -d /workspace/users/yifeif ]; then
  extra_mounts+=(-v /workspace/users/yifeif:/workspace/users/yifeif)
fi

mkdir_if_set() {
  local path="${1:-}"
  if [ -n "$path" ]; then
    mkdir -p "$path" 2>/dev/null || {
      echo "::warning::Could not create '$path' on the host; the CI container will try through mounted storage."
    }
  fi
}

mkdir_if_set "${TRTMC_STORAGE_ROOT:-}"
mkdir_if_set "${ENGINE_DIR:-}"
mkdir_if_set "${HF_HOME:-}"
mkdir_if_set "${HF_HUB_CACHE:-}"
mkdir_if_set "${HUGGINGFACE_HUB_CACHE:-}"
mkdir_if_set "${HF_MODULES_CACHE:-}"

if [ -n "${GITHUB_WORKSPACE:-}" ] && [ -d "$GITHUB_WORKSPACE" ]; then
  echo "Using workspace mount without host chmod: ${GITHUB_WORKSPACE}"
fi

container_options=()
configured_container_options=()
if [ -n "${TRTMC_CONTAINER_OPTIONS:-}" ]; then
  # shellcheck disable=SC2206
  configured_container_options=(${TRTMC_CONTAINER_OPTIONS})
fi

append_non_gpu_container_options() {
  local skip_next=0
  local opt
  for opt in "${configured_container_options[@]}"; do
    if [ "$skip_next" -eq 1 ]; then
      skip_next=0
      continue
    fi
    case "$opt" in
      --gpus)
        skip_next=1
        ;;
      --gpus=*)
        ;;
      *)
        container_options+=("$opt")
        ;;
    esac
  done
}

case "$stage" in
  cpp-unit|cpp-coverage)
    export TRTMC_CPP_CPU_ONLY=1
    echo "Skipping GPU container options for CPU-safe C++ stage '${stage}'"
    ;;
  graph-ops|selective-e2e|full-e2e)
    if [ -n "${TRTMC_GPU_DEVICES:-}" ] && [ "${TRTMC_GPU_DEVICES}" != "all" ]; then
      container_options+=(--gpus "\"device=${TRTMC_GPU_DEVICES}\"")
      append_non_gpu_container_options
      echo "Using GPU devices for stage '${stage}': ${TRTMC_GPU_DEVICES}"
    else
      container_options=("${configured_container_options[@]}")
      echo "Using GPU container options for stage '${stage}': ${TRTMC_CONTAINER_OPTIONS:-<none>}"
    fi
    ;;
  *)
    echo "Skipping GPU container options for non-GPU stage '${stage}'"
    ;;
esac

echo "Starting TensorRT-Model-Connect CI container for stage '${stage}'"
docker run --rm \
  "${container_options[@]}" \
  "${extra_mounts[@]}" \
  -v "$GITHUB_WORKSPACE:$GITHUB_WORKSPACE" \
  -w "$GITHUB_WORKSPACE" \
  -e CI_BASE_REF \
  -e ENGINE_DIR \
  -e TRTMC_STORAGE_ROOT \
  -e TRTMC_GPU_DEVICES \
  -e HF_HOME \
  -e HF_HUB_CACHE \
  -e HUGGINGFACE_HUB_CACHE \
  -e HF_MODULES_CACHE \
  -e FULL_E2E \
  -e RUN_COVERAGE_MAP \
  -e REBUILD_ENGINES \
  -e GITHUB_EVENT_NAME \
  -e GITHUB_REF_NAME \
  -e GITHUB_RUN_ID \
  -e TRTMC_E2E_EXCLUDE_GPU0 \
  -e TRTMC_E2E_DEPRIORITIZE_GPU0 \
  -e TRTMC_TRT_TIMING_CACHE_PATH \
  -e TRTMC_TRT_TIMING_CACHE_DIR \
  -e TRTMC_BUILDER_OPTIMIZATION_LEVEL \
  -e TRTMC_MAX_NUM_TACTICS \
  -e TRTMC_AVG_TIMING_ITERATIONS \
  -e PYTHONHASHSEED \
  -e PYTHON_COVERAGE_MIN_LINE \
  -e PYTHON_COVERAGE_MIN_BRANCH \
  -e CPP_COVERAGE_MIN_LINE \
  -e CPP_COVERAGE_MIN_FUNCTION \
  -e CPP_COVERAGE_MIN_BRANCH \
  -e CPP_CPU_ONLY_COVERAGE_MIN_LINE \
  -e CPP_CPU_ONLY_COVERAGE_MIN_FUNCTION \
  -e CPP_CPU_ONLY_COVERAGE_MIN_BRANCH \
  -e TRTMC_CPP_CPU_ONLY \
  -e BUILD_ALL_TIMEOUT \
  -e CPP_UNIT_TIMEOUT \
  -e PYTHON_BUILDER_TIMEOUT \
  -e CPP_COVERAGE_TIMEOUT \
  -e GRAPH_OP_TIMEOUT \
  -e SELECTIVE_E2E_TIMEOUT \
  -e FULL_E2E_TIMEOUT \
  -e COVERAGE_MAP_TIMEOUT \
  -e DIFFUSION_VLM_ASSESSMENT \
  -e DIFFUSION_VLM_MODEL_ID \
  -e DIFFUSION_VLM_MAX_SIDE \
  -e DIFFUSION_VLM_MAX_NEW_TOKENS \
  -e DIFFUSION_VLM_TIMEOUT \
  -e HF_TOKEN \
  -e HUGGING_FACE_HUB_TOKEN \
  "$TRTMC_CI_IMAGE" \
  bash .github/scripts/run-trtmc-ci.sh "$stage"
