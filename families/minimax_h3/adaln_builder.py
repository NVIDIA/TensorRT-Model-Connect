# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT AdaLN precompute engine for MiniMax-H3.

This implements the lossless Sol-Engine AdaLN optimization as a component
boundary: compute every block's modulation table for all scheduler timesteps,
then unload this weight-heavy plan before loading the recurrent DiT plan.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import tensorrt as trt
from . import trt_compat

from . import graph_ops as op
from .turbo_checkpoint import TurboLoraWeight
from .config import (
    ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
    MiniMaxH3Config,
    SOL_ENGINE_1344X768_124F,
)


def _block_range(profile, block_start, block_end, include_final):
    end = profile.num_layers if block_end is None else block_end
    if (
        isinstance(block_start, bool)
        or not isinstance(block_start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or not 0 <= block_start < end <= profile.num_layers
        or not isinstance(include_final, bool)
    ):
        raise ValueError("MiniMax-H3 AdaLN block range or final selector is invalid")
    if not profile.turbo and (block_start != 0 or end != profile.num_layers or not include_final):
        raise ValueError("MiniMax-H3 partial AdaLN plans require the Turbo profile")
    return block_start, end


def checkpoint_keys(
    profile: MiniMaxH3Config = SOL_ENGINE_1344X768_124F,
    *,
    block_start: int = 0,
    block_end: int | None = None,
    include_final: bool = True,
) -> tuple[str, ...]:
    """Checkpoint tensors used exclusively by the AdaLN precompute plan."""

    block_start, block_end = _block_range(profile, block_start, block_end, include_final)
    names = [
        "time_embedder.linear_1.weight",
        "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight",
        "time_embedder.linear_2.bias",
    ]
    for index in range(block_start, block_end):
        prefix = f"transformer_blocks.{index}.adaln_proj.linear"
        names.extend((f"{prefix}.weight", f"{prefix}.bias"))
    if include_final:
        names.extend(("norm_out.linear.weight", "norm_out.linear.bias"))
    return tuple(names)


@op.cleanup_failed_build
def build_adaln_precompute_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
    block_start: int = 0,
    block_end: int | None = None,
    include_final: bool = True,
) -> bytes | dict[str, int | str]:
    profile.validate()
    block_start, block_end = _block_range(profile, block_start, block_end, include_final)
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    op.configure_builder(config, weight_streaming=weight_streaming)
    # Let TensorRT-RTX use its native maximum pools by default. An explicit
    # override remains useful for constrained build hosts and unit tests.
    if workspace_bytes is not None:
        op.configure_workspace(
            config,
            workspace_bytes,
            default_bytes=ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
        )
    # Match PyTorch's default FP32 matmul policy used by the reference.
    config.clear_flag(trt.BuilderFlag.TF32)

    features = network.add_input(
        "timestep_features",
        trt.float32,
        (profile.max_timestep_count, profile.timestep_input_dim),
    )
    temb = op.linear(
        network,
        features,
        weights["time_embedder.linear_1.weight"],
        weights["time_embedder.linear_1.bias"],
        bf16=False,
    )
    temb = op.silu(network, temb)
    temb = op.linear(
        network,
        temb,
        weights["time_embedder.linear_2.weight"],
        weights["time_embedder.linear_2.bias"],
        bf16=False,
    )
    activated = op.silu(network, temb)

    for index in range(block_start, block_end):
        prefix = f"transformer_blocks.{index}.adaln_proj.linear"
        modulation = op.linear(
            network,
            activated,
            weights[f"{prefix}.weight"],
            weights[f"{prefix}.bias"],
        )
        reshape = network.add_shuffle(modulation)
        reshape.reshape_dims = (
            profile.adaln_table_rows,
            6,
            profile.hidden_size,
        )
        output = reshape.get_output(0)
        output.name = f"block_modulation_{index}"
        network.mark_output(output)

    if include_final:
        # The Turbo reference casts the time embedding before final-modulation
        # SiLU; its block AdaLN branches instead cast after SiLU.
        final_activated = (
            op.silu(network, op.cast(network, temb, trt.bfloat16))
            if isinstance(weights["norm_out.linear.weight"], TurboLoraWeight)
            else activated
        )
        final_modulation = op.linear(
            network,
            final_activated,
            weights["norm_out.linear.weight"],
            weights["norm_out.linear.bias"],
        )
        final_reshape = network.add_shuffle(final_modulation)
        final_reshape.reshape_dims = (
            profile.max_timestep_count,
            2,
            profile.hidden_size,
        )
        final_output = final_reshape.get_output(0)
        final_output.name = "final_modulation"
        network.mark_output(final_output)
    op.validate_native_network(network, expected_attentions=0, label="AdaLN precompute")

    print(
        f"[minimax-h3] building AdaLN precompute: blocks=[{block_start},{block_end}), "
        f"final={include_final}, timesteps={profile.max_timestep_count}",
        file=sys.stderr,
    )
    plan = None
    record = None
    try:
        if output_path is None:
            plan = builder.build_serialized_network(network, config)
        else:
            record = trt_compat.build_serialized_network_to_file(
                builder, network, config, output_path
            )
    finally:
        op.release_weight_buffers(network)
        if consume_weights:
            weights.clear()
    if output_path is None and plan is None:
        raise RuntimeError("TensorRT failed to build MiniMax-H3 AdaLN precompute engine")
    del network, config, builder
    gc.collect()
    return record if record is not None else bytes(plan)
