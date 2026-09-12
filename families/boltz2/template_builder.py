# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct strongly typed TensorRT builder for Boltz-2 template conditioning."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import trt_compat
from .checkpoint import PINNED_PAIRFORMER, load_weight_prefixes
from .graph_ops import Graph
from .pairformer_builder import add_pairformer_no_seq_block


TEMPLATE_CHANNELS = 64
TEMPLATE_BLOCKS = 2
TEMPLATE_BINS = 38
TEMPLATE_ALPHABET = 33


@dataclass(frozen=True)
class TemplateBuildResult:
    engine_path: str
    engine_size_bytes: int
    build_seconds: float
    token_count: int
    template_count: int
    precision: str


def _broadcast(graph: Graph, tensor: Any, shape: tuple[int, ...]):
    zeros = graph.cast(graph.constant(np.zeros(shape, dtype=np.float32)), tensor.dtype)
    return graph.add(tensor, zeros)


def _pair_distances(graph: Graph, coordinates: Any, token_count: int):
    first = graph.reshape(coordinates, (1, int(coordinates.shape[1]), token_count, 1, 3))
    second = graph.reshape(coordinates, (1, int(coordinates.shape[1]), 1, token_count, 3))
    delta = graph.sub(first, second)
    squared = graph.mul(delta, delta)
    squared = graph.reduce_sum(squared, 4, keep_dims=False)
    return graph.unary(squared, graph.trt.UnaryOperation.SQRT)


def _unit_vectors(
    graph: Graph,
    frame_rot: Any,
    frame_t: Any,
    ca_coords: Any,
    token_count: int,
):
    template_count = int(frame_rot.shape[1])
    rotation = graph.reshape(frame_rot, (1, template_count, 1, token_count, 3, 3))
    rotation = graph.transpose(rotation, (0, 1, 2, 3, 5, 4))
    origin = graph.reshape(frame_t, (1, template_count, 1, token_count, 3, 1))
    position = graph.reshape(ca_coords, (1, template_count, token_count, 1, 3, 1))
    delta = graph.sub(position, origin)
    vector = graph.network.add_matrix_multiply(
        rotation,
        graph.trt.MatrixOperation.NONE,
        delta,
        graph.trt.MatrixOperation.NONE,
    ).get_output(0)
    squared = graph.mul(vector, vector)
    norm = graph.reduce_sum(squared, 4, keep_dims=True)
    norm = graph.unary(norm, graph.trt.UnaryOperation.SQRT)
    nonzero = graph.elementwise(
        norm,
        graph.scalar_like(0.0, norm),
        graph.trt.ElementWiseOperation.GREATER,
    )
    normalized = graph.div(vector, graph.maximum(norm, graph.scalar_like(1.0e-12, norm)))
    vector = graph.select(nonzero, normalized, graph.mul(vector, graph.scalar_like(0.0, vector)))
    return graph.reshape(vector, (1, template_count, token_count, token_count, 3))


def define_template_network(
    network: Any,
    trt: Any,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    template_count: int,
):
    """Define TemplateV2 and return its residual update applied to ``z``."""

    if token_count <= 0 or template_count <= 0:
        raise ValueError("Boltz-2 template profile dimensions must be positive")
    if getattr(trt, "bfloat16", None) is None:
        raise RuntimeError("Boltz-2 requires TensorRT with strongly typed BF16 support")
    graph = Graph(network, trt, weights)
    z_input = network.add_input(
        "z", trt.float32, (1, token_count, token_count, PINNED_PAIRFORMER.token_z)
    )
    restype = network.add_input(
        "template_restype", trt.int32, (1, template_count, token_count, TEMPLATE_ALPHABET)
    )
    frame_rot = network.add_input(
        "template_frame_rot", trt.float32, (1, template_count, token_count, 3, 3)
    )
    frame_t = network.add_input(
        "template_frame_t", trt.float32, (1, template_count, token_count, 3)
    )
    cb_coords = network.add_input(
        "template_cb", trt.float32, (1, template_count, token_count, 3)
    )
    ca_coords = network.add_input(
        "template_ca", trt.float32, (1, template_count, token_count, 3)
    )
    cb_mask = network.add_input(
        "template_mask_cb", trt.float32, (1, template_count, token_count)
    )
    frame_mask = network.add_input(
        "template_mask_frame", trt.float32, (1, template_count, token_count)
    )
    template_mask = network.add_input(
        "template_mask", trt.float32, (1, template_count, token_count)
    )
    visibility = network.add_input(
        "visibility_ids", trt.float32, (1, template_count, token_count)
    )
    token_mask = network.add_input("token_mask", trt.float32, (1, token_count))

    distance = _pair_distances(graph, cb_coords, token_count)
    boundaries = graph.constant(
        np.linspace(3.25, 50.75, TEMPLATE_BINS - 1, dtype=np.float32),
        (1, 1, 1, 1, TEMPLATE_BINS - 1),
    )
    distance = graph.reshape(distance, (1, template_count, token_count, token_count, 1))
    bins = graph.elementwise(distance, boundaries, trt.ElementWiseOperation.GREATER)
    bins = graph.reduce_sum(graph.cast(bins, trt.int32), 4, keep_dims=False)
    distogram = graph.one_hot(bins, TEMPLATE_BINS, trt.float32)

    cb_first = graph.reshape(cb_mask, (1, template_count, token_count, 1, 1))
    cb_second = graph.reshape(cb_mask, (1, template_count, 1, token_count, 1))
    frame_first = graph.reshape(frame_mask, (1, template_count, token_count, 1, 1))
    frame_second = graph.reshape(frame_mask, (1, template_count, 1, token_count, 1))
    pair_features = graph.concatenate(
        (
            distogram,
            graph.mul(cb_first, cb_second),
            _unit_vectors(graph, frame_rot, frame_t, ca_coords, token_count),
            graph.mul(frame_first, frame_second),
        ),
        4,
    )
    visible_first = graph.reshape(visibility, (1, template_count, token_count, 1))
    visible_second = graph.reshape(visibility, (1, template_count, 1, token_count))
    visible = graph.cast(graph.equal(visible_first, visible_second), pair_features.dtype)
    pair_features = graph.mul(
        pair_features,
        graph.reshape(visible, (1, template_count, token_count, token_count, 1)),
    )

    restype = graph.cast(restype, trt.float32)
    restype_i = graph.reshape(restype, (1, template_count, token_count, 1, TEMPLATE_ALPHABET))
    restype_i = _broadcast(graph, restype_i, (1, 1, 1, token_count, 1))
    restype_j = graph.reshape(restype, (1, template_count, 1, token_count, TEMPLATE_ALPHABET))
    restype_j = _broadcast(graph, restype_j, (1, 1, token_count, 1, 1))
    pair_features = graph.concatenate((pair_features, restype_i, restype_j), 4)
    pair_features = graph.linear(pair_features, "template_module.a_proj")

    normalized_z = graph.layer_norm(z_input, "template_module.z_norm")
    projected_z = graph.linear(graph.cast(normalized_z, trt.bfloat16), "template_module.z_proj")
    projected_z = graph.cast(projected_z, pair_features.dtype)
    projected_z = graph.reshape(projected_z, (1, 1, token_count, token_count, TEMPLATE_CHANNELS))
    v = graph.add(projected_z, pair_features)
    v = graph.reshape(v, (template_count, token_count, token_count, TEMPLATE_CHANNELS))
    rows = graph.reshape(token_mask, (1, token_count, 1))
    columns = graph.reshape(token_mask, (1, 1, token_count))
    pair_mask = graph.mul(rows, columns)
    pair_mask = _broadcast(
        graph,
        graph.reshape(pair_mask, (1, token_count, token_count)),
        (template_count, 1, 1),
    )
    updated = v
    for block in range(TEMPLATE_BLOCKS):
        updated = add_pairformer_no_seq_block(
            graph,
            updated,
            pair_mask,
            f"template_module.pairformer.layers.{block}",
            pairwise_num_heads=4,
            pairwise_head_width=32,
        )
    v = graph.add(v, updated)
    v = graph.layer_norm(v, "template_module.v_norm")
    v = graph.reshape(v, (1, template_count, token_count, token_count, TEMPLATE_CHANNELS))

    active = graph.reduce_sum(template_mask, 2, keep_dims=False)
    active = graph.cast(
        graph.elementwise(active, graph.scalar_like(0.0, active), trt.ElementWiseOperation.GREATER),
        v.dtype,
    )
    weighted = graph.mul(v, graph.reshape(active, (1, template_count, 1, 1, 1)))
    aggregated = graph.reduce_sum(weighted, 1, keep_dims=False)
    count = graph.reduce_sum(active, 1, keep_dims=False)
    count = graph.maximum(count, graph.scalar_like(1.0, count))
    aggregated = graph.div(aggregated, graph.reshape(count, (1, 1, 1, 1)))
    update = graph.linear(
        graph.cast(graph.relu(aggregated), trt.bfloat16), "template_module.u_proj"
    )
    output = graph.add(z_input, graph.cast(update, z_input.dtype))
    output.name = "z_out"
    network.mark_output(output)
    return output


def build_template_engine(
    checkpoint_path: Path,
    engine_path: Path,
    *,
    token_count: int = 117,
    template_count: int = 4,
    workspace_bytes: int = 16 << 30,
    avg_timing_iterations: int = 8,
    verbose: bool = False,
    verify_checkpoint: bool = True,
) -> TemplateBuildResult:
    """Build the direct static-profile Boltz-2 TemplateV2 engine."""

    _, weights = load_weight_prefixes(
        checkpoint_path, ("template_module.",), verify=verify_checkpoint
    )
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    )
    define_template_network(
        network,
        trt,
        weights,
        token_count=token_count,
        template_count=template_count,
    )
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    config.avg_timing_iterations = avg_timing_iterations
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT failed to build the Boltz-2 template engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return TemplateBuildResult(
        engine_path=str(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        token_count=token_count,
        template_count=template_count,
        precision="bf16-mixed",
    )
