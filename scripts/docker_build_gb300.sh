#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# The repository Dockerfile does not COPY from the repository.
# Use an empty context to avoid uploading large local artifacts.
EMPTY_CONTEXT="${TMPDIR:-/tmp}/trtmc-empty-docker-context"
mkdir -p "$EMPTY_CONTEXT"

docker build \
  -t trtmc-dev-gb300:latest \
  -t trtmc-dev-gb300:manylinux_2_39 \
  -f "$REPO_ROOT/Dockerfile" \
  "$EMPTY_CONTEXT"
