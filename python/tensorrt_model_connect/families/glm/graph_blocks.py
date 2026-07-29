# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small graph helpers used by GLM's native-KV decoder builder."""

from __future__ import annotations

from . import graph_ops


def make_matmul_fn(network, dtype):
    """Return the one weight-matmul form supported by native GLM."""

    def matmul(lhs, lhs_width, rhs_width, rhs_weights):
        return graph_ops.add_matmul_rhs_constant(
            network,
            lhs,
            lhs_width,
            rhs_width,
            rhs_weights,
            dtype=dtype,
        )

    return matmul
