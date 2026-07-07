#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

base_image="${TRTMC_CI_IMAGE:-trtmc-dev-gb300:manylinux_2_39}"
dockerfile="${TRTMC_CI_DOCKERFILE:-Dockerfile}"
fingerprint_label="org.nvidia.trtmc.ci-input-fingerprint"

docker_input_paths=(
  "$dockerfile"
  scripts/docker_build_gb300.sh
  .github/scripts/start-gha-container.sh
  .github/scripts/ensure-ci-docker-image.sh
  .github/workflows/trtmc-ci.yml
  .github/workflows/nightly.yml
)

compute_docker_input_fingerprint() {
  local path
  {
    for path in "${docker_input_paths[@]}"; do
      printf '%s\0' "$path"
      if [ -f "$path" ]; then
        sha256sum "$path" | awk '{ print $1 }'
      else
        printf 'missing\n'
      fi
    done
  } | sha256sum | awk '{ print $1 }'
}

expected_fingerprint="$(compute_docker_input_fingerprint)"
image="${base_image}-${expected_fingerprint:0:12}"

if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'TRTMC_CI_IMAGE=%s\n' "$image" >> "$GITHUB_ENV"
fi

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

import tensorrt
from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
    EncDecRNNTBPEModelWithPrompt,
)

print(f"TENSORRT_VERSION={tensorrt.__version__}")
print("MODELOPT_VERSION=" + metadata.version("nvidia-modelopt"))
print("NEMO_PROMPT_RNNT=available")
PY
'
}

query_image_fingerprint() {
  docker image inspect \
    --format "{{ index .Config.Labels \"$fingerprint_label\" }}" \
    "$image"
}

collect_docker_input_changes() {
  local base="${CI_BASE_REF:-}"
  if [ -z "$base" ] || ! git cat-file -e "$base^{commit}" 2>/dev/null; then
    return 0
  fi

  git diff --name-only "$base"...HEAD -- "${docker_input_paths[@]}"
}

rebuild_reasons=()
mapfile -t changed_paths < <(collect_docker_input_changes)

version_file="$(mktemp)"
trap 'rm -f "$version_file"' EXIT

if ! docker image inspect "$image" >/dev/null 2>&1; then
  rebuild_reasons+=("CI Docker image '$image' is missing")
else
  current_fingerprint="$(query_image_fingerprint 2>/dev/null || true)"
  if [ "$current_fingerprint" != "$expected_fingerprint" ]; then
    rebuild_reasons+=("Docker input fingerprint mismatch: image has '${current_fingerprint:-missing}', source expects '$expected_fingerprint'")
  fi
  if ! query_image_versions > "$version_file"; then
    rebuild_reasons+=("CI Docker image '$image' could not report dependency versions")
  else
    current_trt="$(awk -F= '$1 == "TENSORRT_VERSION" { print $2 }' "$version_file")"
    current_modelopt="$(awk -F= '$1 == "MODELOPT_VERSION" { print $2 }' "$version_file")"
    current_nemo_prompt_rnnt="$(awk -F= '$1 == "NEMO_PROMPT_RNNT" { print $2 }' "$version_file")"

    if [ "$current_trt" != "$expected_trt" ]; then
      rebuild_reasons+=("TensorRT version mismatch: image has '${current_trt:-unknown}', Dockerfile expects '$expected_trt'")
    fi

    if [ "$current_modelopt" != "$expected_modelopt" ]; then
      rebuild_reasons+=("modelopt version mismatch: image has '${current_modelopt:-unknown}', Dockerfile expects '$expected_modelopt'")
    fi

    if [ "$current_nemo_prompt_rnnt" != "available" ]; then
      rebuild_reasons+=("required NeMo prompt RNN-T capability is missing")
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
    --label "$fingerprint_label=$expected_fingerprint" \
    -t "$image" \
    -f "$dockerfile" \
    "$empty_context"
else
  echo "CI Docker image '$image' already matches $dockerfile"
fi

current_fingerprint="$(query_image_fingerprint 2>/dev/null || true)"
if [ "$current_fingerprint" != "$expected_fingerprint" ]; then
  echo "ERROR: CI Docker image '$image' has input fingerprint '${current_fingerprint:-missing}'; expected '$expected_fingerprint'" >&2
  exit 1
fi

query_image_versions > "$version_file"
current_trt="$(awk -F= '$1 == "TENSORRT_VERSION" { print $2 }' "$version_file")"
current_modelopt="$(awk -F= '$1 == "MODELOPT_VERSION" { print $2 }' "$version_file")"
current_nemo_prompt_rnnt="$(awk -F= '$1 == "NEMO_PROMPT_RNNT" { print $2 }' "$version_file")"

if [ "$current_trt" != "$expected_trt" ]; then
  echo "ERROR: CI Docker image '$image' has TensorRT '$current_trt'; expected '$expected_trt' from $dockerfile" >&2
  exit 1
fi

if [ "$current_modelopt" != "$expected_modelopt" ]; then
  echo "ERROR: CI Docker image '$image' has modelopt '$current_modelopt'; expected '$expected_modelopt' from $dockerfile" >&2
  exit 1
fi

if [ "$current_nemo_prompt_rnnt" != "available" ]; then
  echo "ERROR: CI Docker image '$image' is missing the required NeMo prompt RNN-T capability" >&2
  exit 1
fi

echo "CI Docker image '$image' verified: TensorRT $current_trt, modelopt $current_modelopt, NeMo prompt RNN-T available"
