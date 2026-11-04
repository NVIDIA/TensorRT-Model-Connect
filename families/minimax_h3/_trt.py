# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT Python backend selection for the MiniMax-H3 builders.

The family builds with either full TensorRT (``tensorrt``) or TensorRT-RTX
(``tensorrt_rtx``). ``MINIMAX_H3_TRT_BACKEND`` forces one of ``tensorrt``,
``tensorrt_rtx`` or ``tensorrt_rtx_bindings``; by default the first importable
one wins, preferring RTX because that is the multi-device target.
"""

from __future__ import annotations

import importlib
import os

_CANDIDATES = ("tensorrt_rtx", "tensorrt_rtx_bindings", "tensorrt")


def _load():
    forced = os.environ.get("MINIMAX_H3_TRT_BACKEND")
    names = (forced,) if forced else _CANDIDATES
    errors = []
    for name in names:
        try:
            return name, importlib.import_module(name)
        except ImportError as error:
            errors.append(f"{name}: {error}")
    raise ImportError("no TensorRT Python backend importable: " + "; ".join(errors))


BACKEND, trt = _load()
IS_RTX = BACKEND.startswith("tensorrt_rtx")
