#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

# Persistent storage paths (host) — adjust these to your local environment
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
