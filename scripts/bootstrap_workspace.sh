#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Create a canonical host checkout and start its one-to-one dev container.
# Existing checkouts and containers are reused but never removed by this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANAGER="$SCRIPT_DIR/manage_gpu_workspace.sh"
HOST_CONFIG="${TRTMC_HOST_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/trtmc/host.env}"

if [ -f "$HOST_CONFIG" ]; then
    # shellcheck source=/dev/null
    source "$HOST_CONFIG"
fi

GIT_REMOTE="${TRTMC_GIT_REMOTE:-https://github.com/NVIDIA/TensorRT-Model-Connect.git}"
DOCKER_IMAGE="${TRTMC_DOCKER_IMAGE:-trtmc-dev-gb300:latest}"
WORKSPACE_ID=""
BRANCH="main"
DETACH=false
BUILD=true

usage() {
    cat <<EOF
Usage: $0 --id ID [--branch BRANCH] [--detach] [--no-build] [--image IMAGE]

Options:
  --id ID          Stable identifier shared by the worktree, run directory,
                   state manifest, and container
  --branch BRANCH  Branch to clone when the canonical checkout is absent
                   (default: main)
  --detach         Return after the container is ready instead of opening a shell
  --no-build       Skip editable install and C++ build
  --image IMAGE    Container image (default: $DOCKER_IMAGE)

The checkout is created at:
  TRTMC_HOST_ROOT/workspaces/ID/repo
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --id) WORKSPACE_ID="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --detach) DETACH=true; shift ;;
        --no-build) BUILD=false; shift ;;
        --image) DOCKER_IMAGE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

[ -n "$WORKSPACE_ID" ] || {
    echo "ERROR: --id is required so the remote workspace has a stable owner." >&2
    exit 1
}
[[ "$WORKSPACE_ID" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || {
    echo "ERROR: workspace ID must match [a-z0-9][a-z0-9._-]*" >&2
    exit 1
}
[ "${#WORKSPACE_ID}" -le 48 ] || {
    echo "ERROR: workspace ID must be at most 48 characters" >&2
    exit 1
}

: "${TRTMC_HOST_ROOT:?Set TRTMC_HOST_ROOT or create $HOST_CONFIG}"
HOST_ROOT="${TRTMC_HOST_ROOT%/}"
REPO_DIR="$HOST_ROOT/workspaces/$WORKSPACE_ID/repo"
CONTAINER_PREFIX="${TRTMC_CONTAINER_PREFIX:-trtmc-dev-gb300}"
CONTAINER_NAME="$CONTAINER_PREFIX-$WORKSPACE_ID"

echo "=== Workspace: $WORKSPACE_ID ==="
echo "  Repo:      $REPO_DIR"
echo "  Container: $CONTAINER_NAME"
echo "  Branch:    $BRANCH"

if [ -e "$REPO_DIR" ]; then
    [ -d "$REPO_DIR" ] || {
        echo "ERROR: canonical repo path exists but is not a directory: $REPO_DIR" >&2
        exit 1
    }
    echo "Using the existing deployed checkout; no fetch or checkout was performed."
else
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone --branch "$BRANCH" --single-branch "$GIT_REMOTE" "$REPO_DIR"
fi

TRTMC_DOCKER_IMAGE="$DOCKER_IMAGE" "$MANAGER" start "$WORKSPACE_ID"

if [ "$BUILD" = true ]; then
    echo "Installing tensorrt_model_connect and building the C++ runtime..."
    "$MANAGER" exec "$WORKSPACE_ID" -- \
        /opt/venv/bin/python3 -m pip install --no-deps -e . -C py-only=true
    "$MANAGER" exec "$WORKSPACE_ID" -- bash -lc '
        cmake -S . -B build -G Ninja \
            -DTRTMC_TRT_INCLUDE_DIR=$TRT_INC_DIR \
            -DTRTMC_TRT_LIBRARY=$TRT_LIB_DIR/libnvinfer.so \
            -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
            -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so &&
        cmake --build build -j'
fi

echo
echo "=== Workspace ready ==="
echo "  Source:    $REPO_DIR"
echo "  Artifacts: $HOST_ROOT/runs/$WORKSPACE_ID"
echo "  State:     $HOST_ROOT/state/$WORKSPACE_ID/workspace.env"
echo "  Container: $CONTAINER_NAME"
echo
echo "Inspect without changing anything:"
echo "  $MANAGER inspect $WORKSPACE_ID"
echo "  $MANAGER audit"
echo
echo "Stop without deleting:"
echo "  $MANAGER stop $WORKSPACE_ID"

if [ "$DETACH" = false ]; then
    "$MANAGER" shell "$WORKSPACE_ID"
fi
