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

import numpy as np
import tensorrt as trt

from . import graph_ops


# ---------------------------------------------------------------------------
# Precision boundary helpers (used by standard_decoder_builder, not inside
# blocks themselves).
# ---------------------------------------------------------------------------


def make_matmul_fn(network, dtype):
    """Create the Qwen3.5-MoE projection matmul callable."""

    def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
        del weight_name
        return graph_ops.add_matmul_rhs_constant(
            network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype
        )

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
            f"({expected}), got _kv_attention_size={int(explicit)}"
        )
    w_k = weights.get(f"{prefix}.w_k")
    if isinstance(w_k, np.ndarray) and w_k.ndim == 2:
        actual = int(w_k.shape[1])
        if actual != expected:
            raise ValueError(f"{prefix}.w_k must use compact K/V width {expected}, got {actual}")
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
                network, inp, hidden_size, gamma, beta, eps, dtype=dtype
            )
        # An epsilon tensor requires the explicit LayerNorm graph.
        return graph_ops.add_layer_norm(
            network, inp, hidden_size, gamma, beta, eps_tensor, dtype=dtype
        )
    else:
        return graph_ops.add_rms_norm(network, inp, hidden_size, gamma, eps_tensor, dtype=dtype)


def add_sparse_mlp(network, inp, *, weights, prefix, hidden_size, mlp_size, dtype=np.float32):
    """Apply this family's sparse/shared MLP; normalization/residual stay in the decoder."""
    from .moe import add_sparse_moe

    del hidden_size, mlp_size, dtype
    return add_sparse_moe(network, inp, weights[f"{prefix}.moe"])
