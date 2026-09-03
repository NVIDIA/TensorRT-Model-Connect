# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT graph for Boltz-2 token initialization and recycling."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorrt_model_connect import trt_compat

from .checkpoint import PINNED_PAIRFORMER, load_weight_prefixes
from .graph_ops import Graph


TRUNK_INIT_WEIGHT_PREFIXES = (
    "s_init.",
    "z_init_1.",
    "z_init_2.",
    "rel_pos.",
    "token_bonds.",
    "token_bonds_type.",
    "contact_conditioning.",
    "s_norm.",
    "z_norm.",
    "s_recycle.",
    "z_recycle.",
)


@dataclass(frozen=True)
class TrunkInitBuildResult:
    engine_path: str
    engine_sha256: str
    engine_size_bytes: int
    build_seconds: float
    token_count: int
    precision: str


def _relative_position_encoding(
    graph: Graph,
    features: dict[str, Any],
    dtype: Any,
    *,
    prefix: str = "rel_pos",
):
    r_max = 32
    s_max = 2
    tokens = int(features["asym_id"].shape[1])

    def rows(tensor):
        return graph.reshape(tensor, (1, tokens, 1))

    def columns(tensor):
        return graph.reshape(tensor, (1, 1, tokens))

    same_chain = graph.equal(rows(features["asym_id"]), columns(features["asym_id"]))
    same_residue = graph.equal(
        rows(features["residue_index"]),
        columns(features["residue_index"]),
    )
    same_entity = graph.equal(
        rows(features["entity_id"]),
        columns(features["entity_id"]),
    )

    def clipped_difference(lhs, rhs, offset: int, maximum: int):
        difference = graph.sub(rows(lhs), columns(rhs))
        difference = graph.add(
            difference,
            graph.integer_scalar_like(offset, difference),
        )
        difference = graph.maximum(
            difference,
            graph.integer_scalar_like(0, difference),
        )
        return graph.minimum(
            difference,
            graph.integer_scalar_like(maximum, difference),
        )

    residue = clipped_difference(
        features["residue_index"],
        features["residue_index"],
        r_max,
        2 * r_max,
    )
    residue = graph.select(
        same_chain,
        residue,
        graph.integer_scalar_like(2 * r_max + 1, residue),
    )
    residue = graph.one_hot(residue, 2 * r_max + 2, dtype)

    token = clipped_difference(
        features["token_index"],
        features["token_index"],
        r_max,
        2 * r_max,
    )
    same_chain_and_residue = graph.elementwise(
        same_chain,
        same_residue,
        graph.trt.ElementWiseOperation.AND,
    )
    token = graph.select(
        same_chain_and_residue,
        token,
        graph.integer_scalar_like(2 * r_max + 1, token),
    )
    token = graph.one_hot(token, 2 * r_max + 2, dtype)

    chain = clipped_difference(
        features["sym_id"],
        features["sym_id"],
        s_max,
        2 * s_max,
    )
    # The pinned checkpoint sets fix_sym_check=True. It reserves the overflow
    # class only for different entities; same-entity chains retain sym_id deltas.
    different_entity = graph.unary(same_entity, graph.trt.UnaryOperation.NOT)
    chain = graph.select(
        different_entity,
        graph.integer_scalar_like(2 * s_max + 1, chain),
        chain,
    )
    chain = graph.one_hot(chain, 2 * s_max + 2, dtype)
    relative = graph.concatenate(
        (
            residue,
            token,
            graph.cast(graph.reshape(same_entity, (1, tokens, tokens, 1)), dtype),
            chain,
        ),
        3,
    )
    return graph.linear(relative, f"{prefix}.linear_layer")


def _contact_conditioning(
    graph: Graph,
    conditioning: Any,
    threshold: Any,
    dtype: Any,
    *,
    prefix: str = "contact_conditioning",
):
    tokens = int(conditioning.shape[1])
    normalized = graph.sub(threshold, graph.scalar_like(4.0, threshold))
    normalized = graph.elementwise(
        normalized,
        graph.scalar_like(16.0, normalized),
        graph.trt.ElementWiseOperation.DIV,
    )
    flattened = graph.reshape(normalized, (tokens * tokens, 1))
    fourier = graph.linear(flattened, f"{prefix}.fourier_embedding.proj")
    fourier = graph.mul(fourier, graph.scalar_like(2.0 * np.pi, fourier))
    fourier = graph.unary(fourier, graph.trt.UnaryOperation.COS)
    fourier = graph.reshape(fourier, (1, tokens, tokens, PINNED_PAIRFORMER.token_z))

    selected_features = graph.slice(conditioning, (0, 0, 0, 2), (1, tokens, tokens, 3))
    selected_features = graph.cast(selected_features, graph.trt.float32)
    normalized_feature = graph.reshape(normalized, (1, tokens, tokens, 1))
    encoded_input = graph.concatenate(
        (selected_features, normalized_feature, graph.cast(fourier, graph.trt.float32)),
        3,
    )
    encoded = graph.linear(graph.cast(encoded_input, dtype), f"{prefix}.encoder")

    status = graph.slice(conditioning, (0, 0, 0, 0), (1, tokens, tokens, 2))
    status_sum = graph.reduce_sum(status, 3, keep_dims=True)
    selected = graph.sub(
        graph.scalar_like(1.0, graph.cast(status_sum, dtype)),
        graph.cast(status_sum, dtype),
    )
    unspecified = graph.slice(conditioning, (0, 0, 0, 0), (1, tokens, tokens, 1))
    unselected = graph.slice(conditioning, (0, 0, 0, 1), (1, tokens, tokens, 1))
    parameter_shape = (1, 1, 1, PINNED_PAIRFORMER.token_z)
    unspecified_encoding = graph.constant(
        graph.weight(f"{prefix}.encoding_unspecified"), parameter_shape
    )
    unselected_encoding = graph.constant(
        graph.weight(f"{prefix}.encoding_unselected"), parameter_shape
    )
    output = graph.mul(encoded, selected)
    output = graph.add(
        graph.cast(output, graph.trt.float32),
        graph.mul(unspecified_encoding, graph.cast(unspecified, graph.trt.float32)),
    )
    return graph.add(
        output,
        graph.mul(unselected_encoding, graph.cast(unselected, graph.trt.float32)),
    )


def define_trunk_init_network(
    network: Any,
    trt: Any,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
):
    """Define token initialization, pair features, and one recycling update."""

    bf16 = getattr(trt, "bfloat16", None)
    if bf16 is None:
        raise RuntimeError("Boltz-2 requires TensorRT with strongly typed BF16 support")
    graph = Graph(network, trt, weights)
    s_inputs = network.add_input(
        "s_inputs",
        trt.float32,
        (1, token_count, PINNED_PAIRFORMER.token_s),
    )
    recycle_s = network.add_input(
        "recycle_s",
        trt.float32,
        (1, token_count, PINNED_PAIRFORMER.token_s),
    )
    recycle_z = network.add_input(
        "recycle_z",
        trt.float32,
        (1, token_count, token_count, PINNED_PAIRFORMER.token_z),
    )
    features = {
        name: network.add_input(name, trt.int32, (1, token_count))
        for name in ("asym_id", "residue_index", "entity_id", "token_index", "sym_id")
    }
    token_bonds = network.add_input(
        "token_bonds",
        trt.float32,
        (1, token_count, token_count, 1),
    )
    type_bonds = network.add_input(
        "type_bonds",
        trt.int32,
        (1, token_count, token_count),
    )
    contact_conditioning = network.add_input(
        "contact_conditioning",
        trt.int32,
        (1, token_count, token_count, 5),
    )
    contact_threshold = network.add_input(
        "contact_threshold",
        trt.float32,
        (1, token_count, token_count),
    )

    lowp_inputs = graph.cast(s_inputs, bf16)
    s_init = graph.linear(lowp_inputs, "s_init")
    z_first = graph.linear(lowp_inputs, "z_init_1")
    z_second = graph.linear(lowp_inputs, "z_init_2")
    z_init = graph.add(
        graph.reshape(z_first, (1, token_count, 1, PINNED_PAIRFORMER.token_z)),
        graph.reshape(z_second, (1, 1, token_count, PINNED_PAIRFORMER.token_z)),
    )
    relative = _relative_position_encoding(graph, features, bf16)
    z_init = graph.add(z_init, relative)
    z_init = graph.add(
        z_init,
        graph.linear(graph.cast(token_bonds, bf16), "token_bonds"),
    )
    z_init = graph.add(
        z_init,
        graph.embedding(type_bonds, "token_bonds_type", bf16),
    )
    z_init = graph.add(
        graph.cast(z_init, trt.float32),
        _contact_conditioning(
            graph,
            contact_conditioning,
            contact_threshold,
            bf16,
        ),
    )

    normalized_s = graph.layer_norm(recycle_s, "s_norm")
    recycled_s = graph.linear(graph.cast(normalized_s, bf16), "s_recycle")
    s = graph.add(s_init, recycled_s)
    normalized_z = graph.layer_norm(recycle_z, "z_norm")
    recycled_z = graph.linear(graph.cast(normalized_z, bf16), "z_recycle")
    z = graph.add(z_init, graph.cast(recycled_z, trt.float32))
    # Pairformer explicitly promotes sequence attention to FP32, and the
    # caller's contact/recycling residual keeps the pair track in FP32 too.
    s = graph.cast(s, trt.float32)
    for name, tensor in (("s", s), ("z", z), ("relative_position_encoding", relative)):
        tensor.name = name
        network.mark_output(tensor)
    return s, z, relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_trunk_init_engine(
    checkpoint_path: Path,
    engine_path: Path,
    *,
    token_count: int = 117,
    workspace_bytes: int = 16 << 30,
    avg_timing_iterations: int = 8,
    verbose: bool = False,
    verify_checkpoint: bool = True,
) -> TrunkInitBuildResult:
    """Build the direct TensorRT trunk-initialization engine."""

    if token_count <= 0:
        raise ValueError("Boltz-2 trunk token_count must be positive")
    _, weights = load_weight_prefixes(
        checkpoint_path,
        TRUNK_INIT_WEIGHT_PREFIXES,
        verify=verify_checkpoint,
    )
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    )
    define_trunk_init_network(
        network,
        trt,
        weights,
        token_count=token_count,
    )
    config = builder.create_builder_config()
    config.avg_timing_iterations = avg_timing_iterations
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT failed to build the Boltz-2 trunk initializer")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return TrunkInitBuildResult(
        engine_path=str(engine_path),
        engine_sha256=_sha256(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        token_count=token_count,
        precision="bf16-mixed",
    )
