#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Bootstrap an isolated workspace: clone repo + start a uniquely-named container.
#
# Each workspace gets:
#   - Its own repo clone under TRTMC_WORKSPACE_ROOT
#   - Its own Docker container named trtmc-dev-gb300-<id>
#   - Shared HF cache and engine storage (read-mostly, safe to share)
#   - Isolated build artifacts and git state
#
# Usage:
#   ./scripts/bootstrap_workspace.sh                    # auto-generate ID
#   ./scripts/bootstrap_workspace.sh --id my-feature    # custom ID
#   ./scripts/bootstrap_workspace.sh --branch feat/foo  # checkout a branch
#   ./scripts/bootstrap_workspace.sh --id agent-1 --branch feat/foo --detach
#
# The script prints the container name and repo path for use by agent teams.

set -euo pipefail

# --- Defaults ----------------------------------------------------------------

WORKSPACE_ROOT="${TRTMC_WORKSPACE_ROOT:-${HOME}/trtmc-workspaces}"
GIT_REMOTE="${TRTMC_GIT_REMOTE:-https://github.com/NVIDIA/TensorRT-Model-Connect.git}"
DOCKER_IMAGE="${TRTMC_DEV_IMAGE:-trtmc-dev-gb300:latest}"
STORAGE_ROOT="${TRTMC_STORAGE_ROOT:-${HOME}/.cache/trtmc}"
HF_CACHE="${TRTMC_HF_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}"

WORKSPACE_ID=""
BRANCH="main"
DETACH=false
BUILD=true

# --- Parse args --------------------------------------------------------------

while [ $# -gt 0 ]; do
    case "$1" in
        --id)          WORKSPACE_ID="$2"; shift 2 ;;
        --branch)      BRANCH="$2"; shift 2 ;;
        --detach)      DETACH=true; shift ;;
        --no-build)    BUILD=false; shift ;;
        --image)       DOCKER_IMAGE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--id NAME] [--branch BRANCH] [--detach] [--no-build] [--image IMAGE]"
            echo ""
            echo "Options:"
            echo "  --id NAME       Workspace identifier (default: auto-generated UUID)"
            echo "  --branch BRANCH Git branch to checkout (default: main)"
            echo "  --detach        Run container in background (default: interactive)"
            echo "  --no-build      Skip C++ build step"
            echo "  --image IMAGE   Docker image to use (default: trtmc-dev-gb300:latest)"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Auto-generate ID if not provided
if [ -z "$WORKSPACE_ID" ]; then
    WORKSPACE_ID="ws-$(head -c 4 /dev/urandom | xxd -p)"
fi

REPO_DIR="${WORKSPACE_ROOT}/${WORKSPACE_ID}/tensorrt-model-connect"
CONTAINER_NAME="trtmc-dev-gb300-${WORKSPACE_ID}"
ENGINE_DIR="${STORAGE_ROOT}/engines"

# --- Validate ----------------------------------------------------------------

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "ERROR: Container ${CONTAINER_NAME} already exists." >&2
    echo "  Stop it:   docker rm -f ${CONTAINER_NAME}" >&2
    echo "  Or pick a different --id" >&2
    exit 1
fi

# --- Clone -------------------------------------------------------------------

echo "=== Workspace: ${WORKSPACE_ID} ==="
echo "  Repo:      ${REPO_DIR}"
echo "  Container: ${CONTAINER_NAME}"
echo "  Branch:    ${BRANCH}"
echo ""

if [ -d "${REPO_DIR}/.git" ]; then
    echo "Repo already exists, fetching latest..."
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$BRANCH" 2>/dev/null \
        || git -C "$REPO_DIR" checkout -b "$BRANCH" "origin/${BRANCH}"
else
    echo "Cloning repo..."
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone "$GIT_REMOTE" "$REPO_DIR"
    if [ "$BRANCH" != "main" ]; then
        git -C "$REPO_DIR" checkout -b "$BRANCH"
    fi
fi

# --- Write workspace ID so agents can self-discover -------------------------

echo "$WORKSPACE_ID" > "${REPO_DIR}/.workspace_id"

# --- Create shared dirs -----------------------------------------------------

mkdir -p "$ENGINE_DIR" 2>/dev/null || true
mkdir -p "$HF_CACHE" 2>/dev/null || true

# --- Start container ---------------------------------------------------------

echo ""
echo "Starting container ${CONTAINER_NAME}..."

DOCKER_ARGS=(
    --gpus all
    -v "${REPO_DIR}":/workspace/tensorrt-model-connect
    -v "${STORAGE_ROOT}:${STORAGE_ROOT}"
    -v "${HF_CACHE}":/root/.cache/huggingface/hub
    -w /workspace/tensorrt-model-connect
    --name "$CONTAINER_NAME"
    "$DOCKER_IMAGE"
)

if [ "$DETACH" = true ]; then
    docker run -d "${DOCKER_ARGS[@]}" sleep infinity
    echo "Container started in background."

    # Build inside detached container
    if [ "$BUILD" = true ]; then
        echo ""
        echo "Installing tensorrt_model_connect and building C++ runtime..."
        docker exec "$CONTAINER_NAME" pip install --no-deps -e . -C py-only=true
        docker exec "$CONTAINER_NAME" bash -c '
            cmake -S . -B build -G Ninja \
                -DTRTMC_TRT_INCLUDE_DIR=$TRT_INC_DIR \
                -DTRTMC_TRT_LIBRARY=$TRT_LIB_DIR/libnvinfer.so \
                -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
                -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so &&
            cmake --build build -j'
        echo "Build complete."
    fi
else
    echo "(Interactive mode — run build commands manually inside the container)"
    docker run --rm -it "${DOCKER_ARGS[@]}" bash
fi

# --- Summary -----------------------------------------------------------------

echo ""
echo "=== Workspace ready ==="
echo "  ID:        ${WORKSPACE_ID}"
echo "  Repo:      ${REPO_DIR}"
echo "  Container: ${CONTAINER_NAME}"
echo "  Branch:    ${BRANCH}"
echo ""
echo "To use:"
echo "  docker exec -it ${CONTAINER_NAME} bash"
echo ""
echo "To run tests:"
echo "  docker exec ${CONTAINER_NAME} ctest --test-dir build --output-on-failure"
echo "  docker exec ${CONTAINER_NAME} python -m pytest tests/builder/ -v -n auto"
echo ""
echo "To tear down:"
echo "  docker rm -f ${CONTAINER_NAME}"
echo "  rm -rf ${REPO_DIR%/*}"
