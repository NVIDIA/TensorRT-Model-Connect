#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${TRTMC_DOCKER_IMAGE:-trtmc-dev-a100x86-trt11}"
BASE_IMAGE="${TRTMC_DOCKER_BASE_IMAGE:-${IMAGE}-base}"

# Dockerfile.gb300 does not COPY from the repository.
# Use an empty context to avoid uploading large local artifacts.
EMPTY_CONTEXT="${TMPDIR:-/tmp}/trtmc-empty-docker-context"
mkdir -p "$EMPTY_CONTEXT"

docker build \
  -t "$BASE_IMAGE" \
  -f "$REPO_ROOT/Dockerfile.gb300" \
  --build-arg TRT_INC_DIR=/usr/include/x86_64-linux-gnu \
  --build-arg TRTMC_CUBLAS_PRELOAD= \
  "$EMPTY_CONTEXT"

docker build \
  -t "$IMAGE" \
  -f "$REPO_ROOT/Dockerfile.a100x86-trt11" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  "$EMPTY_CONTEXT"
