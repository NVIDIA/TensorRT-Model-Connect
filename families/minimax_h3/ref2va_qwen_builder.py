# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the existing shared Qwen plans with Ref2VA-capable profiles.

These wrappers deliberately delegate to ``multimodal_*_builder``.  They do not
define a second graph or weight partition and still serialize the canonical
``vision_encoder.plan`` and ``text_encoder.plan`` bundle sections.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ref2va_qwen_contract import (
    REF2VA_SHARED_TEXT_PROFILE,
    REF2VA_SHARED_VISION_PROFILE,
)


def build_ref2va_shared_vision_encoder_engine(
    weights: dict[str, np.ndarray],
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build the one shared Qwen vision plan with the 65,536-row superset."""

    # Lazy import keeps the pure contract importable without loading TensorRT.
    from .multimodal_vision_builder import build_multimodal_vision_encoder_engine

    return build_multimodal_vision_encoder_engine(
        weights,
        REF2VA_SHARED_VISION_PROFILE,
        verbose=verbose,
        consume_weights=consume_weights,
        workspace_bytes=workspace_bytes,
        weight_streaming=weight_streaming,
        output_path=output_path,
    )


def build_ref2va_shared_text_encoder_engine(
    weights: dict[str, np.ndarray],
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build the one shared Qwen language plan with the 262,144-row superset."""

    # Lazy import keeps the pure contract importable without loading TensorRT.
    from .multimodal_text_encoder_builder import build_multimodal_text_encoder_engine

    return build_multimodal_text_encoder_engine(
        weights,
        REF2VA_SHARED_TEXT_PROFILE,
        verbose=verbose,
        consume_weights=consume_weights,
        workspace_bytes=workspace_bytes,
        weight_streaming=weight_streaming,
        output_path=output_path,
    )
