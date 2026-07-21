#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

HOST_CONFIG="${TRTMC_HOST_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/trtmc/host.env}"
if [ -f "$HOST_CONFIG" ]; then
  # shellcheck source=/dev/null
  source "$HOST_CONFIG"
fi

# Shared GPU hosts must use the stable one-worktree/one-container mapping.
if [ -n "${TRTMC_HOST_ROOT:-}" ]; then
  WORKSPACE_ID="${TRTMC_WORKSPACE_ID:-}"
  if [ -z "$WORKSPACE_ID" ] && [ -f .workspace_id ]; then
    WORKSPACE_ID="$(<.workspace_id)"
  fi
  if [ -z "$WORKSPACE_ID" ] && [ "$(basename "$PWD")" = repo ]; then
    WORKSPACE_ID="$(basename "$(dirname "$PWD")")"
  fi
  [ -n "$WORKSPACE_ID" ] || {
    echo "ERROR: Set TRTMC_WORKSPACE_ID or run from a canonical workspaces/ID/repo checkout." >&2
    exit 1
  }
  ./scripts/manage_gpu_workspace.sh start "$WORKSPACE_ID"
  exec ./scripts/manage_gpu_workspace.sh shell "$WORKSPACE_ID"
fi

# Unmanaged local developer fallback. Shared GPU hosts never take this path.
STORAGE_ROOT="${TRTMC_STORAGE_ROOT:-$HOME/trtmc-storage}"
HF_CACHE="${TRTMC_HF_CACHE:-$HOME/.cache/huggingface/hub}"
ENGINE_DIR="${STORAGE_ROOT}/engines"

# Create dirs if needed
mkdir -p "$HF_CACHE" "$ENGINE_DIR" 2>/dev/null || true

docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace/tensorrt-model-connect \
  -v "${STORAGE_ROOT}:${STORAGE_ROOT}" \
  -v "${HF_CACHE}":/root/.cache/huggingface/hub \
  -e ENGINE_DIR="${ENGINE_DIR}" \
  -w /workspace/tensorrt-model-connect \
  --name trtmc-dev-gb300 \
  trtmc-dev-gb300 bash
