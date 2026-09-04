# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic helpers for TensorRT engine builders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorrt as trt

from . import graph_ops




def resolve_rope_parameters(config) -> dict:
    """Return the RoPE scaling dict from the raw HF config.json.

    Configs serialized by transformers < 5.x store the scaling dict under
    ``rope_scaling``; transformers 5.x standardizes on ``rope_parameters``.
    GPT-OSS checkpoints on the Hub still ship ``rope_scaling``, so accept
    both (``rope_parameters`` wins when both are present).
    """
    raw = getattr(config, "raw", None) or {}
    for key in ("rope_parameters", "rope_scaling"):
        params = raw.get(key)
        if isinstance(params, dict):
            return params
    return {}


def make_rope_half_tables(
    config,
    attention_window: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (cos, sin) half-dim RoPE tables honoring YaRN when configured."""
    rope_params = resolve_rope_parameters(config)
    rope_type = rope_params.get(
        "rope_type", rope_params.get("type", "default"))
    if rope_type == "yarn":
        attention_factor = rope_params.get("attention_factor")
        yarn_kwargs = dict(
            scaling_factor=float(rope_params.get("factor", 1.0)),
            original_max_position_embeddings=int(rope_params.get(
                "original_max_position_embeddings", 4096)),
            beta_fast=float(rope_params.get("beta_fast", 32.0)),
            beta_slow=float(rope_params.get("beta_slow", 1.0)),
            truncate=bool(rope_params.get("truncate", True)),
            attention_factor=(
                None if attention_factor is None else float(attention_factor)),
        )
        return (
            graph_ops.make_yarn_rope_table_half_dim(
                attention_window, head_dim, config.rope_theta, True,
                **yarn_kwargs),
            graph_ops.make_yarn_rope_table_half_dim(
                attention_window, head_dim, config.rope_theta, False,
                **yarn_kwargs),
        )
    return (
        graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, True),
        graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, False),
    )


@dataclass(frozen=True)
class BuilderContext:
    """TensorRT objects shared by engine builders."""

    logger: trt.Logger
    builder: trt.Builder
    network: trt.INetworkDefinition
    config: trt.IBuilderConfig


def create_builder_context(
    *,
    verbose: bool,
    workspace_bytes: int | None = None,
    strongly_typed: bool = True,
    disable_tf32: bool = False,
) -> BuilderContext:
    """Create a TensorRT builder, network, and config with common defaults."""
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    config = builder.create_builder_config()
    if workspace_bytes is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if disable_tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    return BuilderContext(
        logger=logger,
        builder=builder,
        network=network,
        config=config,
    )


def const_in_work_dtype(
    network: trt.INetworkDefinition,
    shape: tuple,
    values: np.ndarray,
    work_np_dtype: np.dtype,
    work_trt_dtype: trt.DataType,
) -> trt.ITensor:
    """Create a constant in storage dtype and cast it to runtime dtype."""
    const = graph_ops.add_constant(network, shape, values, dtype=work_np_dtype)
    if const.dtype != work_trt_dtype:
        const = network.add_cast(const, work_trt_dtype).get_output(0)
    return const


def norm_multi(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
    dtype: np.dtype,
) -> trt.ITensor:
    """Apply LayerNorm or RMSNorm from the same call site."""
    if norm_type == "layernorm":
        if beta is None:
            beta = np.zeros(hidden, dtype=np.float32)
        return graph_ops.add_layer_norm(
            network, inp, hidden, gamma, beta, eps_tensor, dtype=dtype)
    return graph_ops.add_rms_norm(
        network, inp, hidden, gamma, eps_tensor, dtype=dtype)
