# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct FP32 TensorRT segments for the Boltz-2 diffusion token transformer."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorrt_model_connect import trt_compat

from .checkpoint import load_weight_prefixes
from .graph_ops import Graph
from .input_embedder_builder import _adaln, _conditioned_transition


TOKEN_COUNT = 117
TOKEN_CHANNELS = 768
TOKEN_HEADS = 16
TOKEN_HEAD_WIDTH = TOKEN_CHANNELS // TOKEN_HEADS
TOKEN_LAYERS = 24
TOKEN_SEGMENT_SIZE = 6
TOKEN_BIAS_CHANNELS = TOKEN_LAYERS * TOKEN_HEADS


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
    a: Any,
    condition: Any,
    layer_bias: Any,
    token_mask: Any,
    token_count: int,
    prefix: str,
):
    adapted = _adaln(graph, a, condition, f"{prefix}.adaln", graph.trt.float32)

    def projected(name: str):
        value = graph.linear(adapted, f"{prefix}.pair_bias_attn.proj_{name}")
        value = graph.reshape(value, (1, token_count, TOKEN_HEADS, TOKEN_HEAD_WIDTH))
        return graph.transpose(value, (0, 2, 1, 3))

    query = projected("q")
    key = projected("k")
    value = projected("v")
    scores = graph.network.add_matrix_multiply(
        query,
        graph.trt.MatrixOperation.NONE,
        key,
        graph.trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    scores = graph.div(scores, graph.scalar_like(np.sqrt(TOKEN_HEAD_WIDTH), scores))
    scores = graph.add(scores, graph.transpose(layer_bias, (0, 3, 1, 2)))
    key_mask = graph.reshape(token_mask, (1, 1, 1, token_count))
    mask_bias = graph.mul(
        graph.sub(graph.scalar_like(1.0, key_mask), key_mask),
        graph.scalar_like(-1.0e6, key_mask),
    )
    probabilities = graph.softmax_last(graph.add(scores, mask_bias))
    output = graph.network.add_matrix_multiply(
        probabilities,
        graph.trt.MatrixOperation.NONE,
        value,
        graph.trt.MatrixOperation.NONE,
    ).get_output(0)
    output = graph.transpose(output, (0, 2, 1, 3))
    output = graph.reshape(output, (1, token_count, TOKEN_CHANNELS))
    gate = graph.sigmoid(graph.linear(adapted, f"{prefix}.pair_bias_attn.proj_g"))
    output = graph.linear(graph.mul(gate, output), f"{prefix}.pair_bias_attn.proj_o")
    projection = graph.sigmoid(graph.linear(condition, f"{prefix}.output_projection.0"))
    return graph.mul(projection, output)


def define_diffusion_token_network(
    network: Any,
    trt: Any,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    first_layer: int,
    layer_count: int,
):
    """Define one contiguous score token-transformer segment."""

    if token_count <= 0:
        raise ValueError("Boltz-2 diffusion token count must be positive")
    if first_layer < 0 or layer_count <= 0 or first_layer + layer_count > TOKEN_LAYERS:
        raise ValueError("Boltz-2 diffusion token layer range must be within [0, 24)")
    graph = Graph(network, trt, weights)
    a = network.add_input("a", trt.float32, (1, token_count, TOKEN_CHANNELS))
    condition = network.add_input("single_condition", trt.float32, (1, token_count, TOKEN_CHANNELS))
    bias = network.add_input(
        "token_trans_bias",
        trt.float32,
        (1, token_count, token_count, TOKEN_BIAS_CHANNELS),
    )
    token_mask = network.add_input("token_mask", trt.float32, (1, token_count))
    bias = graph.reshape(
        bias,
        (1, token_count, token_count, TOKEN_LAYERS, TOKEN_HEADS),
    )
    for layer in range(first_layer, first_layer + layer_count):
        prefix = f"structure_module.score_model.token_transformer.layers.{layer}"
        layer_bias = graph.slice(
            bias,
            (0, 0, 0, layer, 0),
            (1, token_count, token_count, 1, TOKEN_HEADS),
        )
        layer_bias = graph.reshape(
            layer_bias,
            (1, token_count, token_count, TOKEN_HEADS),
        )
        a = graph.add(
            a,
            _token_attention(
                graph,
                a,
                condition,
                layer_bias,
                token_mask,
                token_count,
                prefix,
            ),
        )
        update = _conditioned_transition(
            graph,
            a,
            condition,
            f"{prefix}.transition",
            trt.float32,
        )
        a = graph.add(a, update)
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
    layer_count: int = TOKEN_SEGMENT_SIZE,
    token_count: int = TOKEN_COUNT,
    workspace_bytes: int = 16 << 30,
    verbose: bool = False,
    verify_checkpoint: bool = True,
) -> DiffusionTokenBuildResult:
    """Build one direct score token-transformer segment."""

    _, weights = load_weight_prefixes(
        checkpoint_path,
        ("structure_module.score_model.token_transformer.",),
        verify=verify_checkpoint,
    )
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
    )
    config = builder.create_builder_config()
    config.avg_timing_iterations = 8
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT failed to build Boltz-2 diffusion token segment")
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
        precision="fp32-upstream-exact",
    )
