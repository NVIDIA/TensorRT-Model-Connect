# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate CMake-built test helpers independently of the packaged runtime."""

import os
from pathlib import Path


def native_tool(name):
    native = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    root = native if native is not None else os.environ.get("TRTMC_RUNTIME_ROOT")
    root = Path(root) if root else Path(__file__).resolve().parents[3] / "build"
    binary = root / name
    if native is not None:
        assert native and binary.is_file() and os.access(binary, os.X_OK), (
            f"Qwen3-VL MoE native helper is not executable: {binary}"
        )
    return binary
