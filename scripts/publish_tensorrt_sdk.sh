#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK_DOCKERFILE="$REPO_ROOT/Dockerfile.tensorrt-sdk"
source "$REPO_ROOT/scripts/load_artifactory_credentials.sh"
source "$REPO_ROOT/scripts/fetch_tensorrt_sdk.sh"

version="$(trtmc_read_docker_arg "$SDK_DOCKERFILE" TENSORRT_VERSION)"
image="$(trtmc_read_docker_arg "$SDK_DOCKERFILE" TENSORRT_SDK_IMAGE)"
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

trtmc_stage_tensorrt_sdk "$SDK_DOCKERFILE" "$context/tensorrt-sdk.tar.zst"
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
