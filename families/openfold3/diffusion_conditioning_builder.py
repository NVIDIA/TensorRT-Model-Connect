# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT implementation of OpenFold3 diffusion conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .builder_utils import build_plan, create_network, write_plan
from .checkpoint import load_weight_prefixes
from .graph_ops import Graph, diffusion_compute_dtype
from .pairformer_builder import add_transition


@dataclass(frozen=True)
class DiffusionConditioningBuildResult:
    engine_path: str
    engine_size_bytes: int
    build_seconds: float
    token_count: int
    precision: str


def define_diffusion_conditioning_network(
    network,
    trt,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    precision: str = "fp16",
):
    """Define AF3 Algorithms 21–22 for a static token shape."""

    lowp = diffusion_compute_dtype(trt, precision)
    graph = Graph(network, trt, weights, precision=precision)
    t = network.add_input("noise_level", trt.float32, (1,))
    s_input = network.add_input("s_input", trt.float32, (1, token_count, 449))
    s_trunk = network.add_input("s_trunk", trt.float32, (1, token_count, 384))
    z_trunk = network.add_input("z_trunk", trt.float32, (1, token_count, token_count, 128))
    relpos = network.add_input("relpos", trt.float32, (1, token_count, token_count, 139))
    token_mask = network.add_input("token_mask", trt.float32, (1, token_count))
    prefix = "diffusion_module.diffusion_conditioning"

    z = graph.concatenate((z_trunk, relpos), 3)
    z = graph.layer_norm(z, f"{prefix}.layer_norm_z")
    z = graph.linear(graph.cast(z, lowp), f"{prefix}.linear_z")
    pair_mask = graph.mul(
        graph.reshape(token_mask, (1, token_count, 1)),
        graph.reshape(token_mask, (1, 1, token_count)),
    )
    pair_mask = graph.reshape(pair_mask, (1, token_count, token_count, 1))
    for index in range(2):
        update = add_transition(graph, z, f"{prefix}.transition_z.{index}")
        update = graph.mul(update, graph.cast(pair_mask, update.dtype))
        z = graph.add(z, graph.cast(update, z.dtype))

    s = graph.concatenate((s_trunk, s_input), 2)
    s = graph.layer_norm(s, f"{prefix}.layer_norm_s")
    s = graph.linear(graph.cast(s, lowp), f"{prefix}.linear_s")
    normalized_noise = graph.mul(
        graph.unary(graph.div(t, graph.scalar_like(16.0, t)), trt.UnaryOperation.LOG),
        graph.scalar_like(0.25, t),
    )
    w = graph.constant(graph.weight(f"{prefix}.fourier_emb.w"), (1, 256))
    b = graph.constant(graph.weight(f"{prefix}.fourier_emb.b"), (1, 256))
    fourier = graph.add(graph.mul(graph.reshape(normalized_noise, (1, 1)), w), b)
    fourier = graph.mul(fourier, graph.scalar_like(2.0 * np.pi, fourier))
    fourier = graph.unary(fourier, trt.UnaryOperation.COS)
    fourier = graph.layer_norm(fourier, f"{prefix}.layer_norm_n")
    noise_embedding = graph.linear(graph.cast(fourier, lowp), f"{prefix}.linear_n")
    s = graph.add(
        s,
        graph.cast(graph.reshape(noise_embedding, (1, 1, 384)), s.dtype),
    )
    single_mask = graph.reshape(token_mask, (1, token_count, 1))
    for index in range(2):
        update = add_transition(graph, s, f"{prefix}.transition_s.{index}")
        update = graph.mul(update, graph.cast(single_mask, update.dtype))
        s = graph.add(s, graph.cast(update, s.dtype))
    s = graph.cast(s, trt.float32)
    z = graph.cast(z, trt.float32)
    s.name = "s_conditioned"
    z.name = "z_conditioned"
    network.mark_output(s)
    network.mark_output(z)
    return s, z


def build_diffusion_conditioning_engine(
    checkpoint_path: Path,
    engine_path: Path,
    *,
    token_count: int,
    workspace_bytes: int = 8 << 30,
    verbose: bool = False,
    verify_checkpoint: bool = True,
    precision: str = "fp16",
) -> DiffusionConditioningBuildResult:
    weights = load_weight_prefixes(
        checkpoint_path,
        ("diffusion_module.diffusion_conditioning.",),
        verify=verify_checkpoint,
    )
    trt, builder, network = create_network(verbose=verbose)
    define_diffusion_conditioning_network(
        network, trt, weights, token_count=token_count, precision=precision
    )
    plan, build_seconds = build_plan(builder, network, workspace_bytes=workspace_bytes)
    engine_size_bytes = write_plan(engine_path, plan)
    return DiffusionConditioningBuildResult(
        engine_path=str(engine_path),
        engine_size_bytes=engine_size_bytes,
        build_seconds=build_seconds,
        token_count=token_count,
        precision=f"{precision}-mixed",
    )
