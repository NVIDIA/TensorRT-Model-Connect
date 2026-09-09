# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for the shared default decoder builder."""

from .default_decoder import (
    _apply_norm,
    _mark_debug_output,
    build_standard_decoder_engine,
)

__all__ = [
    "_apply_norm",
    "_mark_debug_output",
    "build_standard_decoder_engine",
]
