# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantization framework — extensible low-precision support.

Public API:
    build_quant_context() — construct a QuantContext from CLI args
    get_format() — look up a registered format by name
    list_formats() — list available format names
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .adapters import resolve_calibration_adapter
from .context import QuantContext
from .formats import QuantFormat
from .plan import QuantPlan, canonicalize_quant_format
from .profile import QuantProfile
from .registry import get_format, list_formats, register_format
from .scales import LayerScales, QuantScaleMap
from .scale_providers import (  # noqa: F401
    DynamicQuantizationProvider,
    ModelOptCalibrationProvider,
    PrecomputedJsonProvider,
    PreQuantizedCheckpointProvider,
)

if TYPE_CHECKING:
    from ..config import ModelConfig

logger = logging.getLogger(__name__)

__all__ = [
    "QuantContext",
    "QuantFormat",
    "QuantPlan",
    "QuantProfile",
    "QuantScaleMap",
    "LayerScales",
    "build_quant_context",
    "canonicalize_quant_format",
    "get_format",
    "list_formats",
    "register_format",
]


def build_quant_context(
    format_name: str | None,
    model_dir: str,
    config: ModelConfig,
    exclude_patterns: list[str] | None = None,
    *,
    scales_json: str | None = None,
    num_calibration_samples: int = 512,
    calibration_prompts: list[str] | None = None,
    plugin: object | None = None,
    quant_plan: QuantPlan | None = None,
    graph_ops: Any | None = None,
) -> QuantContext:
    """Construct a QuantContext from high-level parameters.

    This is the main entry point called by engine_builder.py.

    Args:
        format_name: Quantization format ('fp8', 'int8_sq', 'int4_awq', etc.)
        model_dir: Path to HuggingFace model directory.
        config: Parsed ModelConfig.
        exclude_patterns: Weight name patterns to skip (norms, embeddings).
        scales_json: Path to pre-computed scales JSON. If provided, skips
            calibration entirely.
        num_calibration_samples: Number of calibration samples for PTQ.
        calibration_prompts: Custom calibration prompts. None = default.
        graph_ops: Family-owned graph helper module.

    Returns:
        QuantContext ready to thread through graph_blocks.
    """
    plan = quant_plan or QuantPlan.from_build_args(
        precision="fp32",
        quantize=format_name,
        quant_scales=scales_json,
        quant_calibration_samples=num_calibration_samples,
    )
    if not plan.enabled or plan.quant_format is None:
        raise ValueError("build_quant_context requires a quantized plan")

    fmt = get_format(plan.quant_format)

    if exclude_patterns is None:
        exclude_patterns = _default_exclude_patterns()
    if calibration_prompts is None and plugin is not None:
        calibration_data = getattr(plugin, "calibration_data", None)
        if callable(calibration_data):
            calibration_prompts = calibration_data(plan.quant_format)

    adapter = resolve_calibration_adapter(plugin, plan.quant_format)

    # Select scale provider
    if plan.scale_source == "precomputed" and plan.scale_artifact:
        provider = PrecomputedJsonProvider(plan.scale_artifact)
    elif plan.scale_source == "dynamic":
        # NVFP4 uses dynamic quantization (runtime scales)
        provider = DynamicQuantizationProvider()
    elif plan.scale_source == "prequantized" or str(config.raw.get(
            "quantization_config", {}).get(
            "quant_method", "")).lower() in {
                "awq", "gptq", "compressed-tensors", "compressed_tensors"
            }:
        provider = PreQuantizedCheckpointProvider()
    else:
        # Auto-calibrate with ModelOpt
        provider = ModelOptCalibrationProvider(
            num_samples=plan.calibration_samples,
            calibration_prompts=calibration_prompts,
        )

    scale_map = provider.acquire_scales(
        model_dir, config, fmt, exclude_patterns, adapter=adapter)

    profile = QuantProfile(
        format=fmt,
        scale_map=scale_map,
        exclude_patterns=exclude_patterns,
    )

    logger.info(
        "Built quantization context: format=%s, %d layers quantized, "
        "%d excluded patterns",
        plan.quant_format, len(scale_map.scales), len(exclude_patterns))

    return QuantContext(profile=profile, graph_ops=graph_ops)


def _default_exclude_patterns() -> list[str]:
    """Default weight name patterns to exclude from quantization.

    Norms, embeddings, and output heads are kept in full precision.
    """
    return [
        "embedding",
        "final_norm",
        "w_out",
        "lm_head",
        "*.input_norm",
        "*.post_attn_norm",
        "*_norm*",
    ]
