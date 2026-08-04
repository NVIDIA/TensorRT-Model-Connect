# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared weight, cache-geometry, and normalization helpers for InternVL."""

from __future__ import annotations

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops

trt = trt_compat.get_trt()


def make_matmul_fn(network, dtype, quant_ctx):
    """Create a matmul callable that routes through quant_ctx if present."""
    if quant_ctx is None:
        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype)
        return matmul

    def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
        return quant_ctx.maybe_quantized_matmul(
            network, lhs, lhs_w, rhs_w, rhs_weights, weight_name,
            dtype=dtype)
    return matmul


_make_matmul_fn = make_matmul_fn


def infer_kv_attention_size(
    weights: dict,
    *,
    prefix: str = "layer.0",
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """Validate and return the compact K/V row width."""
    expected = int(num_kv_heads * head_dim)
    explicit = weights.get("_kv_attention_size")
    if explicit is not None and int(explicit) != expected:
        raise ValueError(
            f"Compact K/V cache width must be num_kv_heads * head_dim "
            f"({expected}), got _kv_attention_size={int(explicit)}")
    w_k = weights.get(f"{prefix}.w_k")
    if isinstance(w_k, np.ndarray) and w_k.ndim == 2:
        actual = int(w_k.shape[1])
        if actual != expected:
            raise ValueError(
                f"{prefix}.w_k must use compact K/V width {expected}, "
                f"got {actual}")
    return expected


def apply_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
    dtype: np.dtype = np.float32,
    eps: float | None = None,
) -> trt.ITensor:
    """Dispatch to RMSNorm or LayerNorm based on norm_type."""
    if norm_type == "layernorm":
        if beta is None:
            beta = np.zeros(hidden_size, dtype=np.float32)
        if eps is not None:
            return graph_ops.add_layer_norm_native(
                network, inp, hidden_size, gamma, beta, eps, dtype=dtype)
        return graph_ops.add_layer_norm(
            network, inp, hidden_size, gamma, beta, eps_tensor, dtype=dtype)
    return graph_ops.add_rms_norm(
        network, inp, hidden_size, gamma, eps_tensor, dtype=dtype)
