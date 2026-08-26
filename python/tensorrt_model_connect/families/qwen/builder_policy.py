# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT builder policy shared by Qwen decoder layouts."""

from __future__ import annotations

import os
from typing import Any


_EXPLICIT_NATIVE_KV_MASK_SMS = frozenset({(8, 6), (12, 1)})
_EXPLICIT_NATIVE_KV_MASK_TRT_RELEASES = frozenset({(11, 1), (11, 2)})
_EXPLICIT_NATIVE_KV_MASK_MAX_CACHE_LENGTH = 16384


def _trt_major_minor(trt_version: str) -> tuple[int, int] | None:
    parts = trt_version.split(".", maxsplit=2)
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def requires_explicit_native_kv_mask(
    precision: str,
    trt_version: str,
    compute_capability: tuple[int, int],
) -> bool:
    """Select the validated workaround for broken native active-length masking.

    On affected targets, BF16 uses the explicit-mask fused kernel directly.
    FP16 uses the same kernel behind a narrow BF16 attention/KV boundary,
    because the target's FP16 fused kernel rejects every fourth live input.
    """
    return (
        precision.lower() in {"fp16", "bf16"}
        and _trt_major_minor(trt_version)
        in _EXPLICIT_NATIVE_KV_MASK_TRT_RELEASES
        and tuple(compute_capability) in _EXPLICIT_NATIVE_KV_MASK_SMS
    )


def validate_explicit_native_kv_cache_length(max_cache_length: int) -> None:
    """Fail before TRT for compatibility-path capacities not yet qualified."""
    if max_cache_length <= _EXPLICIT_NATIVE_KV_MASK_MAX_CACHE_LENGTH:
        return
    raise ValueError(
        "Qwen native-KV attention on this GPU/TensorRT combination currently "
        "supports max_cache_length <= 16384; got "
        f"{max_cache_length}. Rebuild with --max-cache-length 16384"
    )


def configure_qwen_builder(
    trt_config: Any,
    quant_ctx: Any | None,
    trt_version: str,
    num_hidden_layers: int,
) -> None:
    """Apply the accuracy-stable Qwen FP8 policy for the TensorRT release."""
    quant_format = getattr(getattr(quant_ctx, "profile", None), "format", None)
    if getattr(quant_format, "name", None) != "fp8":
        return

    version_parts = trt_version.split(".", maxsplit=2)
    if len(version_parts) < 2:
        return
    try:
        major_minor = tuple(int(part) for part in version_parts[:2])
    except ValueError:
        return

    fp16_tail_length: int | None = None
    if major_minor == (11, 0):
        fp16_tail_length = 8
    elif (
        major_minor[0] == 11
        and major_minor[1] >= 2
    ):
        if not os.environ.get("TRTMC_BUILDER_OPTIMIZATION_LEVEL", "").strip():
            # TRT 11.2+ can otherwise select numerically unstable FP8 tactics.
            trt_config.builder_optimization_level = 0
        fp16_tail_length = 22

    if fp16_tail_length is not None:
        # Choice logits are unstable when every up projection is quantized.
        # Keep a version-specific tail in FP16; earlier layers remain FP8 and
        # preserve the model's quantized execution path.
        exclude_patterns = quant_ctx.profile.exclude_patterns
        tail_start = max(0, num_hidden_layers - fp16_tail_length)
        for layer_index in range(tail_start, num_hidden_layers):
            pattern = f"layer.{layer_index}.w_up"
            if pattern not in exclude_patterns:
                exclude_patterns.append(pattern)
