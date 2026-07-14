#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

stage="${1:?usage: run-gha-stage.sh <stage>}"

container_name="${TRTMC_CI_CONTAINER_NAME:-trtmc-ci-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}}"
workspace="${TRTMC_CI_WORKSPACE:-${GITHUB_WORKSPACE:-}}"
if [ -z "$workspace" ] || [ ! -d "$workspace" ]; then
  echo "::error::CI workspace does not exist: ${workspace:-unset}"
  exit 1
fi

stage_pid=""
cleanup_cancelled_stage() {
  local rc="$1"
  # GitHub first signals the step shell and may escalate shortly afterward.
  # Ignore repeated signals long enough to kill the exact run-owned container;
  # a later `if: always()` step is not guaranteed to start after cancellation.
  trap '' INT TERM
  docker rm -f "$container_name" >/dev/null 2>&1 || true
  if [ -n "$stage_pid" ] && kill -0 "$stage_pid" >/dev/null 2>&1; then
    kill -KILL "$stage_pid" >/dev/null 2>&1 || true
    wait "$stage_pid" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}

trap 'cleanup_cancelled_stage 130' INT
trap 'cleanup_cancelled_stage 143' TERM

if [ "$(docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null || true)" != "true" ]; then
  echo "::error::CI container '$container_name' is not running. Start it with .github/scripts/start-gha-container.sh before running stages."
  exit 1
fi

run_attached_stage() {
  local rc
  "$@" &
  stage_pid=$!
  # The wait builtin is interrupted immediately by INT/TERM, unlike waiting
  # for a foreground external command. This lets the traps above remove the
  # container even when docker exec or its attached process ignores signals.
  set +e
  wait "$stage_pid"
  rc=$?
  set -e
  stage_pid=""
  return "$rc"
}

if [ "${TRTMC_CI_HARDENED:-false}" = "true" ]; then
  run_attached_stage docker exec \
    -w "$workspace" \
    -e CI_BASE_REF \
    -e GITHUB_EVENT_NAME \
    -e GITHUB_REF_NAME \
    -e GITHUB_RUN_ID \
    -e GITHUB_RUN_ATTEMPT \
    -e PYTHONHASHSEED \
    -e BUILD_ALL_TIMEOUT \
    -e CPP_UNIT_TIMEOUT \
    -e PYTHON_BUILDER_TIMEOUT \
    -e TRTMC_UNIT_BUILD_JOBS \
    -e TRTMC_UNIT_TEST_JOBS \
    -e TRTMC_PREMERGE_UNIT_SCOPE \
    "$container_name" \
    bash .github/scripts/run-trtmc-ci.sh "$stage"
  exit $?
fi

run_attached_stage docker exec \
  -w "$workspace" \
  -e CI_BASE_REF \
  -e ENGINE_DIR \
  -e TRTMC_STORAGE_ROOT \
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
  -e GITHUB_RUN_ATTEMPT \
  -e TRTMC_CI_STATE_DIR \
  -e TRTMC_E2E_EXCLUDE_GPU0 \
  -e TRTMC_E2E_DEPRIORITIZE_GPU0 \
  -e TRTMC_TRT_TIMING_CACHE_PATH \
  -e TRTMC_TRT_TIMING_CACHE_DIR \
  -e TRTMC_BUILDER_OPTIMIZATION_LEVEL \
  -e TRTMC_MAX_NUM_TACTICS \
  -e TRTMC_AVG_TIMING_ITERATIONS \
  -e TRTMC_ENABLE_LIBTORCH_MULTINOMIAL \
  -e PYTHONHASHSEED \
  -e PYTHON_COVERAGE_MIN_LINE \
  -e PYTHON_COVERAGE_MIN_BRANCH \
  -e CPP_COVERAGE_MIN_LINE \
  -e CPP_COVERAGE_MIN_FUNCTION \
  -e CPP_COVERAGE_MIN_BRANCH \
  -e CPP_COVERAGE_SCOPE \
  -e BUILD_ALL_TIMEOUT \
  -e CPP_UNIT_TIMEOUT \
  -e PYTHON_BUILDER_TIMEOUT \
  -e TRTMC_UNIT_BUILD_JOBS \
  -e TRTMC_UNIT_TEST_JOBS \
  -e TRTMC_PREMERGE_UNIT_SCOPE \
  -e CPP_COVERAGE_TIMEOUT \
  -e CPP_COVERAGE_BUILD_DIR \
  -e GRAPH_OP_TIMEOUT \
  -e SELECTIVE_E2E_TIMEOUT \
  -e FULL_E2E_TIMEOUT \
  -e COVERAGE_MAP_TIMEOUT \
  -e TRTMC_PACKAGE_PYTHON_TAGS \
  -e TRTMC_PACKAGE_WHEEL_ARCH \
  -e TRTMC_PACKAGE_BUILD_ROOT \
  -e TRTMC_WHEEL_SMOKE_CONFIG \
  -e TRTMC_WHEEL_SMOKE_MODEL_ID \
  -e TRTMC_WHEEL_SMOKE_MAX_CACHE \
  -e TRTMC_WHEEL_SMOKE_MAX_NEW_TOKENS \
  -e TRTMC_WHEEL_SMOKE_OPTIMIZATION_LEVEL \
  -e TRTMC_WHEEL_SMOKE_BUILD_TIMEOUT \
  -e TRTMC_WHEEL_SMOKE_RUN_TIMEOUT \
  -e DIFFUSION_VLM_ASSESSMENT \
  -e DIFFUSION_VLM_CONFIG \
  -e DIFFUSION_VLM_MODEL_ID \
  -e DIFFUSION_VLM_MAX_SIDE \
  -e DIFFUSION_VLM_MAX_NEW_TOKENS \
  -e DIFFUSION_VLM_TIMEOUT \
  -e HF_TOKEN \
  -e HUGGING_FACE_HUB_TOKEN \
  "$container_name" \
  bash .github/scripts/run-trtmc-ci.sh "$stage"
