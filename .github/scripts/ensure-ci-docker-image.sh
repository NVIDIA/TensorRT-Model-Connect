#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

image="${TRTMC_CI_IMAGE:-trtmc-dev-gb300:manylinux_2_39}"
dockerfile="${TRTMC_CI_DOCKERFILE:-Dockerfile}"

read_docker_arg() {
  local name="$1"
  awk -F= -v key="$name" '
    $1 == "ARG " key {
      print $2
      exit
    }
  ' "$dockerfile"
}

expected_trt="$(read_docker_arg TENSORRT_VERSION)"
expected_modelopt="$(read_docker_arg MODELOPT_VERSION)"

if [ -z "$expected_trt" ]; then
  echo "ERROR: Could not find ARG TENSORRT_VERSION in $dockerfile" >&2
  exit 1
fi

if [ -z "$expected_modelopt" ]; then
  echo "ERROR: Could not find ARG MODELOPT_VERSION in $dockerfile" >&2
  exit 1
fi

summary() {
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    printf '%s\n' "$*" >> "$GITHUB_STEP_SUMMARY"
  fi
}

query_image_versions() {
  docker run --rm --entrypoint /bin/bash "$image" -lc '
python3 - <<'"'"'PY'"'"'
import importlib.metadata as metadata
from pathlib import Path

import tensorrt

print(f"TENSORRT_VERSION={tensorrt.__version__}")
print("MODELOPT_VERSION=" + metadata.version("nvidia-modelopt"))
print(
    "NLOHMANN_JSON_HEADER="
    + ("present" if Path("/usr/include/nlohmann/json.hpp").is_file() else "missing")
)
PY
'
}

collect_docker_input_changes() {
  local base="${CI_BASE_REF:-}"
  if [ -z "$base" ] || ! git cat-file -e "$base^{commit}" 2>/dev/null; then
    return 0
  fi

  git diff --name-only "$base"...HEAD -- \
    Dockerfile \
    scripts/docker_build_gb300.sh \
    .github/scripts/start-gha-container.sh \
    .github/scripts/ensure-ci-docker-image.sh \
    .github/workflows/trtmc-ci.yml \
    .github/workflows/nightly.yml
}

rebuild_reasons=()
mapfile -t changed_paths < <(collect_docker_input_changes)
if [ "${#changed_paths[@]}" -gt 0 ]; then
  rebuild_reasons+=("CI Docker image inputs changed")
fi

version_file="$(mktemp)"
trap 'rm -f "$version_file"' EXIT

if ! docker image inspect "$image" >/dev/null 2>&1; then
  rebuild_reasons+=("CI Docker image '$image' is missing")
else
  if ! query_image_versions > "$version_file"; then
    rebuild_reasons+=("CI Docker image '$image' could not report dependency versions")
  else
    current_trt="$(awk -F= '$1 == "TENSORRT_VERSION" { print $2 }' "$version_file")"
    current_modelopt="$(awk -F= '$1 == "MODELOPT_VERSION" { print $2 }' "$version_file")"
    current_nlohmann="$(awk -F= '$1 == "NLOHMANN_JSON_HEADER" { print $2 }' "$version_file")"

    if [ "$current_trt" != "$expected_trt" ]; then
      rebuild_reasons+=("TensorRT version mismatch: image has '${current_trt:-unknown}', Dockerfile expects '$expected_trt'")
    fi

    if [ "$current_modelopt" != "$expected_modelopt" ]; then
      rebuild_reasons+=("modelopt version mismatch: image has '${current_modelopt:-unknown}', Dockerfile expects '$expected_modelopt'")
    fi

    if [ "$current_nlohmann" != "present" ]; then
      rebuild_reasons+=("nlohmann/json development headers are missing")
    fi
  fi
fi

if [ "${#rebuild_reasons[@]}" -gt 0 ]; then
  echo "Rebuilding CI Docker image '$image' from $dockerfile"
  printf '  reason: %s\n' "${rebuild_reasons[@]}"
  summary "Rebuilding CI Docker image \`$image\` from \`$dockerfile\`."
  for reason in "${rebuild_reasons[@]}"; do
    summary "- $reason"
  done
  if [ "${#changed_paths[@]}" -gt 0 ]; then
    summary ""
    summary "Changed CI Docker image inputs:"
    for path in "${changed_paths[@]}"; do
      summary "- \`$path\`"
    done
  fi

  empty_context="${RUNNER_TEMP:-/tmp}/trtmc-empty-docker-context"
  mkdir -p "$empty_context"
  docker build \
    -t "$image" \
    -f "$dockerfile" \
    "$empty_context"
else
  echo "CI Docker image '$image' already matches $dockerfile"
fi

query_image_versions > "$version_file"
current_trt="$(awk -F= '$1 == "TENSORRT_VERSION" { print $2 }' "$version_file")"
current_modelopt="$(awk -F= '$1 == "MODELOPT_VERSION" { print $2 }' "$version_file")"
current_nlohmann="$(awk -F= '$1 == "NLOHMANN_JSON_HEADER" { print $2 }' "$version_file")"

if [ "$current_trt" != "$expected_trt" ]; then
  echo "ERROR: CI Docker image '$image' has TensorRT '$current_trt'; expected '$expected_trt' from $dockerfile" >&2
  exit 1
fi

if [ "$current_modelopt" != "$expected_modelopt" ]; then
  echo "ERROR: CI Docker image '$image' has modelopt '$current_modelopt'; expected '$expected_modelopt' from $dockerfile" >&2
  exit 1
fi

if [ "$current_nlohmann" != "present" ]; then
  echo "ERROR: CI Docker image '$image' lacks /usr/include/nlohmann/json.hpp" >&2
  exit 1
fi

echo "CI Docker image '$image' verified: TensorRT $current_trt, modelopt $current_modelopt, nlohmann/json headers present"
