# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic helpers for TensorRT engine builders."""

from __future__ import annotations


import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops


trt = trt_compat.get_trt()


def const_in_work_dtype(
    network: trt.INetworkDefinition,
    shape: tuple,
    values: np.ndarray,
    work_np_dtype: np.dtype,
    work_trt_dtype: trt.DataType,
) -> trt.ITensor:
    """Create a constant in storage dtype and cast it to runtime dtype."""
    const = graph_ops.add_constant(network, shape, values, dtype=work_np_dtype)
    if const.dtype != work_trt_dtype:
        const = network.add_cast(const, work_trt_dtype).get_output(0)
    return const
