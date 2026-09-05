# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Segmented FP16 TensorRT builder for OpenFold3's 24 diffusion blocks."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat

from .atom_attention_builder import _adaln, _conditioned_transition
from .checkpoint import load_weight_prefixes
from .graph_ops import Graph, diffusion_compute_dtype


TOKEN_CHANNELS = 768
TOKEN_HEADS = 16
TOKEN_HEAD_WIDTH = 48
TOKEN_LAYERS = 24


@dataclass(frozen=True)
class DiffusionTokenBuildResult:
    engine_path: str
    engine_sha256: str
    engine_size_bytes: int
    build_seconds: float
    first_layer: int
    layer_count: int
    token_count: int
    precision: str


def _token_attention(
    graph: Graph,
    a,
    condition,
    z,
    token_mask,
    prefix: str,
    *,
    token_count: int,
    lowp,
):
    adapted = _adaln(graph, a, condition, f"{prefix}.layer_norm_a", lowp)
    compute = graph.cast(adapted, lowp)

    def projected(name: str):
        value = graph.linear(compute, f"{prefix}.mha.linear_{name}")
        value = graph.reshape(value, (1, token_count, TOKEN_HEADS, TOKEN_HEAD_WIDTH))
        return graph.transpose(value, (0, 2, 1, 3))

    query = projected("q")
    key = projected("k")
    value = projected("v")
    scores = graph.network.add_matrix_multiply(
        graph.cast(query, graph.stable_attention_dtype),
        graph.trt.MatrixOperation.NONE,
        graph.cast(key, graph.stable_attention_dtype),
        graph.trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    scores = graph.mul(scores, graph.scalar_like(1.0 / np.sqrt(TOKEN_HEAD_WIDTH), scores))
    pair_bias = graph.linear(graph.cast(z, lowp), f"{prefix}.linear_z")
    pair_bias = graph.transpose(pair_bias, (0, 3, 1, 2))
    scores = graph.add(scores, graph.cast(pair_bias, scores.dtype))
    key_mask = graph.reshape(token_mask, (1, 1, 1, token_count))
    mask_bias = graph.mul(
        graph.sub(key_mask, graph.scalar_like(1.0, key_mask)),
        graph.scalar_like(1.0e9, key_mask),
    )
    probabilities = graph.softmax_last(graph.add(scores, graph.cast(mask_bias, scores.dtype)))
    attended = graph.network.add_matrix_multiply(
        probabilities,
        graph.trt.MatrixOperation.NONE,
        graph.cast(value, probabilities.dtype),
        graph.trt.MatrixOperation.NONE,
    ).get_output(0)
    attended = graph.transpose(attended, (0, 2, 1, 3))
    attended = graph.cast(graph.reshape(attended, (1, token_count, TOKEN_CHANNELS)), lowp)
    gate = graph.sigmoid(graph.linear(compute, f"{prefix}.mha.linear_g"))
    attended = graph.linear(graph.mul(attended, gate), f"{prefix}.mha.linear_o")
    output_gate = graph.sigmoid(
        graph.linear(graph.cast(condition, lowp), f"{prefix}.linear_ada_out")
    )
    return graph.mul(attended, output_gate)


def define_diffusion_token_network(
    network,
    trt,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    first_layer: int,
    layer_count: int,
    precision: str = "fp16",
):
    if first_layer < 0 or layer_count <= 0 or first_layer + layer_count > TOKEN_LAYERS:
        raise ValueError("OpenFold3 diffusion block range must lie within [0, 24)")
    lowp = diffusion_compute_dtype(trt, precision)
    graph = Graph(network, trt, weights, precision=precision)
    a = network.add_input("a", trt.float32, (1, token_count, TOKEN_CHANNELS))
    condition = network.add_input("single_condition", trt.float32, (1, token_count, 384))
    z = network.add_input("pair_condition", trt.float32, (1, token_count, token_count, 128))
    token_mask = network.add_input("token_mask", trt.float32, (1, token_count))
    z = graph.layer_norm(z, "diffusion_module.diffusion_transformer.layer_norm_z")
    for layer in range(first_layer, first_layer + layer_count):
        prefix = f"diffusion_module.diffusion_transformer.blocks.{layer}.attention_pair_bias"
        update = _token_attention(
            graph,
            a,
            condition,
            z,
            token_mask,
            prefix,
            token_count=token_count,
            lowp=lowp,
        )
        a = graph.add(a, graph.cast(update, a.dtype))
        transition_prefix = (
            f"diffusion_module.diffusion_transformer.blocks.{layer}.conditioned_transition"
        )
        update = _conditioned_transition(graph, a, condition, token_mask, transition_prefix, lowp)
        a = graph.add(a, graph.cast(update, a.dtype))
    a.name = "a_out"
    network.mark_output(a)
    return a


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_diffusion_token_engine(
    checkpoint_path: Path,
    engine_path: Path,
    *,
    first_layer: int,
    layer_count: int = 6,
    token_count: int,
    workspace_bytes: int = 16 << 30,
    verbose: bool = False,
    verify_checkpoint: bool = True,
    precision: str = "fp16",
) -> DiffusionTokenBuildResult:
    prefixes = ("diffusion_module.diffusion_transformer.layer_norm_z.",) + tuple(
        f"diffusion_module.diffusion_transformer.blocks.{index}."
        for index in range(first_layer, first_layer + layer_count)
    )
    weights = load_weight_prefixes(checkpoint_path, prefixes, verify=verify_checkpoint)
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    )
    define_diffusion_token_network(
        network,
        trt,
        weights,
        token_count=token_count,
        first_layer=first_layer,
        layer_count=layer_count,
        precision=precision,
    )
    config = builder.create_builder_config()
    config.avg_timing_iterations = 8
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT failed to build an OpenFold3 diffusion segment")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return DiffusionTokenBuildResult(
        engine_path=str(engine_path),
        engine_sha256=_sha256(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        first_layer=first_layer,
        layer_count=layer_count,
        token_count=token_count,
        precision=f"{precision}-mixed",
    )
