# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal normalization helper for the native DeepSeek-OCR builders."""

from __future__ import annotations

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops


trt = trt_compat.get_trt()


def _apply_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Apply the RMSNorm used by the native text and vision graphs."""
    if norm_type != "rmsnorm" or beta is not None:
        raise ValueError("DeepSeek-OCR native builders support only RMSNorm")
    return graph_ops.add_rms_norm(
        network, inp, hidden_size, gamma, eps_tensor, dtype=dtype)


__all__ = ["_apply_norm"]
