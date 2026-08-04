# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT AdaLN precompute engine for MiniMax-H3.

This implements the lossless Sol-Engine AdaLN optimization as a component
boundary: compute every block's modulation table for all scheduler timesteps,
then unload this weight-heavy plan before loading the recurrent DiT plan.
"""

from __future__ import annotations

import sys

from tensorrt_model_connect import trt_compat

from . import graph_ops as op
from .config import MiniMaxH3Config


trt = trt_compat.get_trt()


def build_adaln_precompute_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    verbose: bool = False,
) -> bytes:
    profile.validate()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    op.configure_builder(config)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)
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

    for index in range(profile.num_layers):
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

    final_modulation = op.linear(
        network,
        activated,
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
        f"[minimax-h3] building AdaLN precompute: layers={profile.num_layers}, "
        f"timesteps={profile.max_timestep_count}",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build MiniMax-H3 AdaLN precompute engine")
    return bytes(plan)
