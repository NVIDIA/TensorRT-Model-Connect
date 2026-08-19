# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned TensorRT availability marker."""

from __future__ import annotations

import pytest


def _trt_available() -> bool:
    try:
        import tensorrt as trt  # noqa: F401

        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart  # type: ignore[no-redef]

        status, count = cudart.cudaGetDeviceCount()
        return int(status) == 0 and int(count) > 0
    except (ImportError, RuntimeError):
        return False


def _gpu_trt_skipif(condition: bool, reason: str):
    def decorator(obj):
        obj = pytest.mark.skipif(condition, reason=reason)(obj)
        obj = pytest.mark.gpu(obj)
        obj = pytest.mark.trt(obj)
        return obj

    return decorator


requires_trt = _gpu_trt_skipif(
    not _trt_available(),
    "TensorRT + CUDA not available",
)
