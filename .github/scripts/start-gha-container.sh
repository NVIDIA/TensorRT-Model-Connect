#!/usr/bin/env bash
set -euo pipefail

container_name="${TRTMC_CI_CONTAINER_NAME:-trtmc-ci-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}}"

if [ -n "${GITHUB_ENV:-}" ]; then
  echo "TRTMC_CI_CONTAINER_NAME=${container_name}" >> "$GITHUB_ENV"
fi

docker image inspect "$TRTMC_CI_IMAGE" >/dev/null || {
  echo "::error::Docker image '$TRTMC_CI_IMAGE' is not present on the self-hosted runner. Set repository variable TRTMC_MANYLINUX_CI_IMAGE if the runner uses a different local manylinux image tag."
  exit 1
}

docker rm -f "$container_name" >/dev/null 2>&1 || true

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
  chmod -R a+rwX "$GITHUB_WORKSPACE" 2>/dev/null || {
    echo "::warning::Could not normalize workspace permissions before entering the CI container."
  }
fi

env_args=()
add_env() {
  local name="$1"
  env_args+=(-e "${name}=${!name-}")
}

for name in \
  CI_BASE_REF \
  ENGINE_DIR \
  TRTMC_STORAGE_ROOT \
  HF_HOME \
  HF_HUB_CACHE \
  HUGGINGFACE_HUB_CACHE \
  HF_MODULES_CACHE \
  FULL_E2E \
  RUN_COVERAGE_MAP \
  REBUILD_ENGINES \
  GITHUB_EVENT_NAME \
  GITHUB_REF_NAME \
  GITHUB_RUN_ID \
  GITHUB_RUN_ATTEMPT \
  TRTMC_CI_STATE_DIR \
  TRTMC_E2E_EXCLUDE_GPU0 \
  TRTMC_E2E_DEPRIORITIZE_GPU0 \
  TRTMC_TRT_TIMING_CACHE_PATH \
  TRTMC_TRT_TIMING_CACHE_DIR \
  TRTMC_BUILDER_OPTIMIZATION_LEVEL \
  TRTMC_MAX_NUM_TACTICS \
  TRTMC_AVG_TIMING_ITERATIONS \
  TRTMC_ENABLE_LIBTORCH_MULTINOMIAL \
  PYTHONHASHSEED \
  PYTHON_COVERAGE_MIN_LINE \
  PYTHON_COVERAGE_MIN_BRANCH \
  CPP_COVERAGE_MIN_LINE \
  CPP_COVERAGE_MIN_FUNCTION \
  CPP_COVERAGE_MIN_BRANCH \
  BUILD_ALL_TIMEOUT \
  CPP_UNIT_TIMEOUT \
  PYTHON_BUILDER_TIMEOUT \
  CPP_COVERAGE_TIMEOUT \
  CPP_COVERAGE_BUILD_DIR \
  GRAPH_OP_TIMEOUT \
  SELECTIVE_E2E_TIMEOUT \
  FULL_E2E_TIMEOUT \
  COVERAGE_MAP_TIMEOUT \
  TRTMC_PACKAGE_PYTHON_TAGS \
  TRTMC_PACKAGE_WHEEL_ARCH \
  TRTMC_PACKAGE_BUILD_ROOT \
  TRTMC_WHEEL_SMOKE_CONFIG \
  TRTMC_WHEEL_SMOKE_MODEL_ID \
  TRTMC_WHEEL_SMOKE_MAX_CACHE \
  TRTMC_WHEEL_SMOKE_MAX_NEW_TOKENS \
  TRTMC_WHEEL_SMOKE_OPTIMIZATION_LEVEL \
  TRTMC_WHEEL_SMOKE_BUILD_TIMEOUT \
  TRTMC_WHEEL_SMOKE_RUN_TIMEOUT \
  DIFFUSION_VLM_ASSESSMENT \
  DIFFUSION_VLM_CONFIG \
  DIFFUSION_VLM_MODEL_ID \
  DIFFUSION_VLM_MAX_SIDE \
  DIFFUSION_VLM_MAX_NEW_TOKENS \
  DIFFUSION_VLM_TIMEOUT \
  HF_TOKEN \
  HUGGING_FACE_HUB_TOKEN; do
  add_env "$name"
done

# shellcheck disable=SC2086
docker run -d \
  --name "$container_name" \
  $TRTMC_CONTAINER_OPTIONS \
  "${extra_mounts[@]}" \
  -v "$GITHUB_WORKSPACE:$GITHUB_WORKSPACE" \
  -w "$GITHUB_WORKSPACE" \
  "${env_args[@]}" \
  "$TRTMC_CI_IMAGE" \
  bash -lc 'trap "exit 0" TERM INT; sleep infinity & wait'

echo "Started CI container: $container_name"
