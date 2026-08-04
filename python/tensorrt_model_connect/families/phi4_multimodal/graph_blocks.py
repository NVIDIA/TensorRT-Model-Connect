# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small weight-aware helpers for the Phi-4 Multimodal native decoder."""

from __future__ import annotations

import numpy as np

from . import graph_ops


def make_matmul_fn(network, dtype):
    """Create the family decoder's constant-weight matmul callable."""

    def matmul(lhs, lhs_width, rhs_width, rhs_weights, weight_name):
        del weight_name
        return graph_ops.add_matmul_rhs_constant(
            network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype)

    return matmul


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
            "Compact K/V cache width must be num_kv_heads * head_dim "
            f"({expected}), got _kv_attention_size={int(explicit)}")
    w_k = weights.get(f"{prefix}.w_k")
    if isinstance(w_k, np.ndarray) and w_k.ndim == 2:
        actual = int(w_k.shape[1])
        if actual != expected:
            raise ValueError(
                f"{prefix}.w_k must use compact K/V width {expected}, "
                f"got {actual}")
    return expected
