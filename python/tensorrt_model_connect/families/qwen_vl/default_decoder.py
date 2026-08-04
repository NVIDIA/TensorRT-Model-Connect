# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entrypoint for Qwen-VL's native-KV decoder builder.

Qwen-VL no longer exposes its legacy cache-concatenation graph. Callers of
the historical standard-decoder entrypoint are routed to an explicit split
prefill or decode engine and unsupported modes fail closed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_blocks
from .build_routing import native_kv_build_capability
from .default_dual_profile_decoder import build_dual_profile_decoder_engine
from .lora import DynamicLoraConfig
from .native_kv_contract import validate_native_kv_weights

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig
    from ...quantization.context import QuantContext


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    """Preserve the historical helper for family-owned debug tooling."""
    cast = network.add_cast(tensor, trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


def _apply_norm(
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
    """Preserve the historical family-owned norm helper."""
    return graph_blocks.apply_norm(
        network, inp, hidden_size, gamma, beta, eps_tensor, norm_type,
        dtype=dtype, eps=eps)


def build_standard_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "bf16",
    quant_ctx: "QuantContext | None" = None,
    norm_type: str = "rmsnorm",
    mlp_type: str = "swiglu",
    position_type: str = "rope",
    activation: str = "silu",
    partial_rotary_factor: float = 1.0,
    interleaved_rope: bool = False,
    parallel_residual: bool = False,
    scale_attn_weights: bool = True,
    embed_input: bool = False,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    hidden_state_output: bool = False,
) -> bytes:
    """Build a split Qwen-VL native-KV engine through the legacy entrypoint."""
    if debug_layer_outputs or hidden_state_output:
        raise ValueError(
            "Qwen-VL native KV does not support debug or hidden-state outputs")

    unsupported = []
    if norm_type != "rmsnorm":
        unsupported.append("norm_type must be 'rmsnorm'")
    if mlp_type != "swiglu":
        unsupported.append("mlp_type must be 'swiglu'")
    if position_type != "rope":
        unsupported.append("position_type must be 'rope'")
    if activation != "silu":
        unsupported.append("activation must be 'silu'")
    if partial_rotary_factor != 1.0:
        unsupported.append("partial_rotary_factor must be 1.0")
    if interleaved_rope:
        unsupported.append("mRoPE interleaving is derived from the model config")
    if parallel_residual:
        unsupported.append("parallel_residual must be false")
    if not scale_attn_weights:
        unsupported.append("scale_attn_weights must be true")
    if not embed_input:
        unsupported.append("embed_input must be true")
    if unsupported:
        raise ValueError(
            "Qwen-VL native KV standard entrypoint rejected: "
            + "; ".join(unsupported))

    lora_config = DynamicLoraConfig.from_model_config(config)
    capability = native_kv_build_capability(
        config,
        precision=precision,
        max_cache_length=max_cache_length,
        tp_size=1,
        dynamic_kv_cache=bool(config.raw.get("dynamic_kv_cache", False)),
        quantized=quant_ctx is not None,
        debug_layer_outputs=False,
        lora_enabled=lora_config.enabled,
    )
    if not capability.eligible:
        raise ValueError(
            "Qwen-VL has no legacy KV fallback; native build rejected: "
            + capability.reason)

    role = str(config.raw.get("_decoder_engine_role", ""))
    if role not in ("prefill", "decode"):
        raise ValueError(
            "Qwen-VL native KV requires an explicit split prefill/decode role")
    validate_native_kv_weights(config, weights)
    return build_dual_profile_decoder_engine(
        config,
        weights,
        max_cache_length,
        precision=precision,
        quant_ctx=quant_ctx,
        norm_type=norm_type,
        mlp_type=mlp_type,
        position_type=position_type,
        activation=activation,
        partial_rotary_factor=partial_rotary_factor,
        parallel_residual=parallel_residual,
        scale_attn_weights=scale_attn_weights,
        embed_input=embed_input,
        verbose=verbose,
        profile_mode=role,
    )
