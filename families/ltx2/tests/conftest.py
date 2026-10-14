# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 test backend selection.

The builders import ``tensorrt``. Hosts that only have TensorRT-RTX installed (the
Windows RTX demo host) bind it through the same alias the build CLI uses
(``select_backend("trt_rtx")``). ``TRTMC_LTX2_TEST_BACKEND=trt|trt_rtx`` forces a choice.
"""

from __future__ import annotations

import importlib.util
import os


def _bind_backend() -> None:
    choice = os.environ.get("TRTMC_LTX2_TEST_BACKEND", "").strip()
    if not choice:
        if importlib.util.find_spec("tensorrt") is not None:
            return
        if importlib.util.find_spec("tensorrt_rtx") is None:
            return
        choice = "trt_rtx"
    from tensorrt_model_connect.build import select_backend

    select_backend(choice)


_bind_backend()
