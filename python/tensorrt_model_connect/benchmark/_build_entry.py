# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start the standard TRTMC builder with a benchmark-selected TensorRT binding."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def main() -> None:
    root = os.environ.get("TRTMC_BENCH_TRT_PYTHON_ROOT", "").strip()
    if root:
        resolved = Path(root).expanduser().resolve()
        sys.path.insert(0, str(resolved))
    if os.environ.get("TRTMC_BENCH_BLOCK_TRT_LIBS_WHEEL") == "1":
        # Debian TensorRT bindings use the matching system DSO.  A different
        # standalone wheel in the parent venv must not preload its own DSO.
        sys.modules["tensorrt_libs"] = None

    from tensorrt_model_connect.build_cli import main as build_main

    build_main()


if __name__ == "__main__":
    main()
