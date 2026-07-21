# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TimesFM graph-operation compatibility surface."""

from .model.model import (
    add_attention_core,
    add_constant,
    add_layer_norm_native,
    add_rms_norm_last_dim,
    add_silu,
)


__all__ = [
    "add_attention_core",
    "add_constant",
    "add_layer_norm_native",
    "add_rms_norm_last_dim",
    "add_silu",
]
