#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker build \
  -t trtmc-dev-gb300:latest \
  -t trtmc-dev-gb300:manylinux_2_39 \
  -f "$REPO_ROOT/Dockerfile" \
  "$REPO_ROOT"
