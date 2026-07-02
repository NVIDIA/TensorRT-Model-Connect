# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for the shared default dual-profile decoder builder."""

from .default_dual_profile_decoder import (
    _const_in_work_dtype,
    _make_matmul_fn,
    _norm_multi,
    build_dual_profile_decoder_engine,
)

__all__ = [
    "_const_in_work_dtype",
    "_make_matmul_fn",
    "_norm_multi",
    "build_dual_profile_decoder_engine",
]
