#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# TEMPORARY: Remove this script and Dockerfile.tensorrt-sdk when TensorRT 11.2
# is publicly released and CI can install the official packages directly.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK_DOCKERFILE="$REPO_ROOT/Dockerfile.tensorrt-sdk"

read_docker_arg() {
  local dockerfile="$1"
  local name="$2"
  awk -F= -v key="$name" '
    $1 == "ARG " key {
      sub(/^[^=]*=/, "")
      print
      exit
    }
  ' "$dockerfile"
}

load_artifactory_credentials() {
  if [ -n "${TRTMC_ARTIFACTORY_USERNAME:-}" ] &&
     [ -n "${TRTMC_ARTIFACTORY_PASSWORD:-}" ]; then
    export TRTMC_ARTIFACTORY_USERNAME TRTMC_ARTIFACTORY_PASSWORD
    return 0
  fi

  local credential_file="${TRTMC_ARTIFACTORY_CREDENTIAL_FILE:-}"
  if [ -z "$credential_file" ]; then
    echo "ERROR: Set TRTMC_ARTIFACTORY_USERNAME and TRTMC_ARTIFACTORY_PASSWORD," >&2
    echo "       or set TRTMC_ARTIFACTORY_CREDENTIAL_FILE to a two-line credential file." >&2
    return 1
  fi
  if [ ! -r "$credential_file" ]; then
    echo "ERROR: Artifactory credential file is not readable: $credential_file" >&2
    return 1
  fi

  local -a credentials=()
  mapfile -t credentials < "$credential_file"
  if [ "${#credentials[@]}" -lt 2 ] ||
     [ -z "${credentials[0]}" ] || [ -z "${credentials[1]}" ]; then
    echo "ERROR: Artifactory credential file must contain username and password on separate lines." >&2
    return 1
  fi

  TRTMC_ARTIFACTORY_USERNAME="${credentials[0]}"
  TRTMC_ARTIFACTORY_PASSWORD="${credentials[1]}"
  export TRTMC_ARTIFACTORY_USERNAME TRTMC_ARTIFACTORY_PASSWORD
}

stage_tensorrt_sdk() {
  local dockerfile="$1"
  local destination="$2"
  local sdk_url sdk_range sdk_sha cache_dir cache_file temporary_file

  sdk_url="$(read_docker_arg "$dockerfile" TENSORRT_SDK_URL)"
  sdk_range="$(read_docker_arg "$dockerfile" TENSORRT_SDK_RANGE)"
  sdk_sha="$(read_docker_arg "$dockerfile" TENSORRT_SDK_SHA256)"
  if [ -z "$sdk_url" ] || [ -z "$sdk_range" ] || [ -z "$sdk_sha" ]; then
    echo "ERROR: Missing pinned TensorRT SDK metadata in $dockerfile" >&2
    return 1
  fi

  cache_dir="${TRTMC_TENSORRT_SDK_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/trtmc}"
  cache_file="$cache_dir/tensorrt-sdk-$sdk_sha.tar.zst"
  mkdir -p "$cache_dir" "$(dirname "$destination")"

  if [ ! -f "$cache_file" ] ||
     ! echo "$sdk_sha  $cache_file" | sha256sum --check --status; then
    load_artifactory_credentials
    temporary_file="$cache_file.tmp.$$"
    rm -f "$temporary_file"
    trap 'rm -f "$temporary_file"' RETURN
    curl --fail --location --silent --show-error \
      --user "${TRTMC_ARTIFACTORY_USERNAME}:${TRTMC_ARTIFACTORY_PASSWORD}" \
      --range "$sdk_range" \
      --output "$temporary_file" \
      "$sdk_url"
    echo "$sdk_sha  $temporary_file" | sha256sum --check
    mv "$temporary_file" "$cache_file"
    trap - RETURN
  fi

  rm -f "$destination"
  cp --reflink=auto "$cache_file" "$destination"
  echo "$sdk_sha  $destination" | sha256sum --check --status
}

version="$(read_docker_arg "$SDK_DOCKERFILE" TENSORRT_VERSION)"
image="$(read_docker_arg "$SDK_DOCKERFILE" TENSORRT_SDK_IMAGE)"
if [ -z "$version" ] || [ -z "$image" ]; then
  echo "ERROR: Missing TensorRT version or GHCR image in $SDK_DOCKERFILE" >&2
  exit 1
fi
if [ "$image" != "ghcr.io/nvidia/tensorrt-model-connect/tensorrt-sdk:$version" ]; then
  echo "ERROR: GHCR image tag '$image' does not match TensorRT version '$version'" >&2
  exit 1
fi

context="$(mktemp -d "${TMPDIR:-/tmp}/trtmc-tensorrt-sdk-context.XXXXXX")"
docker_config="$(mktemp -d "${TMPDIR:-/tmp}/trtmc-ghcr-auth.XXXXXX")"
cleanup() {
  rm -rf "$context" "$docker_config"
}
trap cleanup EXIT

stage_tensorrt_sdk "$SDK_DOCKERFILE" "$context/tensorrt-sdk.tar.zst"
docker build \
  --build-arg "TENSORRT_VERSION=$version" \
  --tag "$image" \
  --file "$SDK_DOCKERFILE" \
  "$context"

ghcr_username="${GHCR_USERNAME:-}"
ghcr_token="${GHCR_TOKEN:-}"
if [ -z "$ghcr_username" ]; then
  ghcr_username="$(gh api user --jq .login)"
fi
if [ -z "$ghcr_token" ]; then
  ghcr_token="$(gh auth token)"
fi

export DOCKER_CONFIG="$docker_config"
printf '%s' "$ghcr_token" | docker login ghcr.io \
  --username "$ghcr_username" --password-stdin
unset ghcr_token
docker push "$image"

echo "Published $image"
docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$image"
