# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composable architectural building blocks for TRT engine construction.

Layer 2 in the three-layer builder stack:

    graph_ops.py        Layer 1: Atomic TRT operations (tensor-in/tensor-out)
        |
    graph_blocks.py     Layer 2: Composable blocks (weight-aware)  <- THIS FILE
        |
    builders / plugins  Layer 3: Full engine assembly

Each block composes multiple graph_ops into a reusable sub-structure
(full attention block, SwiGLU MLP, GELU MLP, norm dispatch). Functions
accept a ``weights`` dict + ``prefix`` string to resolve weight names.

Blocks do NOT apply residual connections. Callers compose the residual
pattern, which is what varies across architectures (sequential vs parallel
residual, DeepStack injection, MoE routing, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


# ---------------------------------------------------------------------------
# Precision boundary helpers (used by standard_decoder_builder, not inside
# blocks themselves).
# ---------------------------------------------------------------------------


def make_matmul_fn(network, dtype):
    """Create the projection matmul callable with a constant right-hand side."""

    def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
        del weight_name
        return graph_ops.add_matmul_rhs_constant(
            network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype
        )

    return matmul


_make_matmul_fn = make_matmul_fn


def add_swiglu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
    layer_prefix: str = "",
) -> trt.ITensor:
    """Gate/up/down SwiGLU MLP. Returns output tensor."""
    matmul = _make_matmul_fn(network, dtype)
    _lp = layer_prefix or prefix

    gate = matmul(inp, hidden_size, mlp_size, weights[f"{prefix}.w_gate"], f"{_lp}.w_gate")
    up = matmul(inp, hidden_size, mlp_size, weights[f"{prefix}.w_up"], f"{_lp}.w_up")

    # silu(gate) * up and the down projection in FP32, rounded once: in FP16 the product, or the projection of a row
    # that mixes large and small products, can overflow or underflow where the result is representable.
    work = gate.dtype
    if up.dtype != trt.float32:
        up = network.add_cast(up, trt.float32).get_output(0)
    product = network.add_elementwise(
        graph_ops.add_sigmoid_product(network, gate, gate, trt.float32), up, trt.ElementWiseOperation.PROD
    ).get_output(0)
    mlp_out = graph_ops.add_matmul_rhs_constant(
        network, product, mlp_size, hidden_size, weights[f"{prefix}.w_down"], dtype=np.float32
    )
    if work != trt.float32:
        mlp_out = network.add_cast(mlp_out, work).get_output(0)
    return mlp_out
