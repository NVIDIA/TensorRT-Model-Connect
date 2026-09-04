# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the two explicit Qwen-VL decoder roles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tensorrt as trt

from .config import ModelConfig
from .default_dual_profile_decoder import build_dual_profile_decoder_engine


if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from typing import Any as QuantContext


def _decoder_build_options(config: ModelConfig) -> dict:
    family_options = config.raw.get("_family_build_options", {})
    if not isinstance(family_options, dict):
        return {}
    decoder_options = family_options.get("qwen_vl_decoder", {})
    if not isinstance(decoder_options, dict):
        raise ValueError("qwen_vl_decoder build options must be an object")
    return decoder_options


def _decode_attention_backend(config: ModelConfig) -> str:
    backend = str(_decoder_build_options(config).get("decode_attention", "native"))
    if backend not in {"native", "decomposed"}:
        raise ValueError(
            f"qwen_vl_decoder.decode_attention must be 'native' or 'decomposed', got {backend!r}"
        )
    return backend


def _decoder_profile_options(config: ModelConfig) -> tuple[int, int | None, int | None]:
    options = _decoder_build_options(config)
    max_prefill_length = int(options.get("max_prefill_length", 0))
    opt_prefill_length = int(options.get("opt_prefill_length", 64))
    builder_workspace_gib = int(options.get("builder_workspace_gib", 0))
    if max_prefill_length < 0:
        raise ValueError("qwen_vl_decoder.max_prefill_length must be >= 0")
    if opt_prefill_length <= 0:
        raise ValueError("qwen_vl_decoder.opt_prefill_length must be > 0")
    if builder_workspace_gib < 0:
        raise ValueError("qwen_vl_decoder.builder_workspace_gib must be >= 0")
    return (
        opt_prefill_length,
        max_prefill_length or None,
        builder_workspace_gib << 30 if builder_workspace_gib else None,
    )


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    cast = network.add_cast(tensor, trt.float32)
    output = cast.get_output(0)
    output.name = name
    network.mark_output(output)


def build_standard_decoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx: QuantContext | None = None,
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
    """Build exactly one prefill or decode profile for the active split."""
    if not config.raw.get("_active_split_decoder_build"):
        raise ValueError("Qwen-VL decoder requires the family split build")
    role = config.raw.get("_decoder_engine_role")
    if role not in {"prefill", "decode"}:
        raise ValueError("Qwen-VL decoder role must be prefill or decode")
    if not embed_input:
        raise ValueError("Qwen-VL decoder requires explicit vision-language embeddings")
    if debug_layer_outputs or hidden_state_output:
        raise NotImplementedError("Qwen-VL split decoder does not expose debug hidden states")

    decode_attention = _decode_attention_backend(config)
    opt_prefill_length, max_prefill_length, workspace = _decoder_profile_options(config)
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
        interleaved_rope=interleaved_rope,
        parallel_residual=parallel_residual,
        scale_attn_weights=scale_attn_weights,
        embed_input=True,
        verbose=verbose,
        opt_prefill_length=opt_prefill_length,
        max_prefill_length=max_prefill_length,
        builder_workspace_bytes=workspace,
        force_decomposed_attention=role == "decode" and decode_attention == "decomposed",
        profile_mode=role,
    )
