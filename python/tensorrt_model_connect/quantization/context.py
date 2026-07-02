# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantization context — threading object for graph construction.

QuantContext is passed through graph_blocks as an optional parameter.
When present, matmul operations are routed through the quantization
profile's format strategy. When None, all matmuls are plain FP16/FP32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()

from .profile import QuantProfile
from .scales import LayerScales


@dataclass
class QuantContext:
    """Optional quantization context threaded through graph construction."""

    profile: QuantProfile
    graph_ops: Any | None = None

    def maybe_quantized_matmul(
        self,
        network: trt.INetworkDefinition,
        lhs: trt.ITensor,
        lhs_width: int,
        rhs_width: int,
        rhs_weights: np.ndarray,
        weight_name: str,
        dtype: np.dtype = np.float32,
    ) -> trt.ITensor:
        """Dispatch to quantized or plain matmul based on profile.

        If the weight_name is quantizable (has scales and is not excluded),
        delegates to the format's wrap_matmul. Otherwise falls back to a
        plain add_matmul_rhs_constant.
        """
        graph_ops = self.graph_ops
        if graph_ops is None:
            raise RuntimeError(
                "QuantContext requires the owning family graph_ops module")

        if self.profile.should_quantize(weight_name):
            scales = self.profile.scale_map.get(weight_name)
            if scales is None:
                # Dynamic quantization: create default scales
                scales = LayerScales()
            return self.profile.format.wrap_matmul(
                network, lhs, rhs_weights, scales,
                lhs_width=lhs_width, rhs_width=rhs_width, dtype=dtype,
                graph_ops=graph_ops)

        # Fallback: plain matmul (layer not quantized)
        return graph_ops.add_matmul_rhs_constant(
            network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype)
