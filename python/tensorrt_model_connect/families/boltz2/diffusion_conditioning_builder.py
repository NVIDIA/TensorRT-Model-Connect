# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT builder for static Boltz-2 diffusion conditioning."""

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
from .input_embedder_builder import (
    ATOM_CHANNELS,
    ATOM_COUNT,
    ATOM_HEADS,
    ATOM_LAYERS,
    ATOM_WINDOW_KEYS,
    ATOM_WINDOW_QUERIES,
    _to_keys,
    atom_windows,
)
from .pairformer_builder import add_transition


TOKEN_TRANSFORMER_LAYERS = 24
TOKEN_TRANSFORMER_HEADS = 16
TOKEN_CHANNELS = 384
PAIR_CHANNELS = 128


@dataclass(frozen=True)
class DiffusionConditioningBuildResult:
    engine_path: str
    engine_sha256: str
    engine_size_bytes: int
    build_seconds: float
    token_count: int
    atom_count: int
    precision: str


def _atom_reference_encoder(
    graph: Graph,
    features: dict[str, Any],
    s_trunk: Any,
    z: Any,
):
    ref_pos = features["ref_pos"]
    atom_count = int(ref_pos.shape[1])
    windows = atom_windows(atom_count)
    ref_space_uid = features["ref_space_uid"]
    atom_pad_mask = features["atom_pad_mask"]
    atom_to_token = features["atom_to_token"]
    atom_features = graph.concatenate(
        (
            ref_pos,
            graph.reshape(features["ref_charge"], (1, atom_count, 1)),
            graph.cast(features["ref_element"], graph.trt.float32),
            graph.cast(
                graph.reshape(features["ref_atom_name_chars"], (1, atom_count, 256)),
                graph.trt.float32,
            ),
        ),
        2,
    )
    c = graph.linear(
        atom_features,
        "diffusion_conditioning.atom_encoder.embed_atom_features",
    )
    q = c
    ref_queries = graph.reshape(ref_pos, (1, windows, ATOM_WINDOW_QUERIES, 1, 3))
    ref_keys = graph.reshape(_to_keys(graph, ref_pos), (1, windows, 1, ATOM_WINDOW_KEYS, 3))
    displacement = graph.sub(ref_keys, ref_queries)
    squared = graph.reduce_sum(graph.mul(displacement, displacement), 4, keep_dims=True)
    reciprocal_distance = graph.div(
        graph.scalar_like(1.0, squared),
        graph.add(graph.scalar_like(1.0, squared), squared),
    )
    mask_queries = graph.reshape(
        graph.cast(atom_pad_mask, graph.trt.bool),
        (1, windows, ATOM_WINDOW_QUERIES, 1),
    )
    mask_keys = graph.reshape(
        _to_keys(graph, graph.reshape(atom_pad_mask, (1, atom_count, 1))),
        (1, windows, 1, ATOM_WINDOW_KEYS),
    )
    uid_queries = graph.reshape(ref_space_uid, (1, windows, ATOM_WINDOW_QUERIES, 1))
    uid_keys = graph.reshape(
        _to_keys(
            graph,
            graph.cast(graph.reshape(ref_space_uid, (1, atom_count, 1)), graph.trt.float32),
        ),
        (1, windows, 1, ATOM_WINDOW_KEYS),
    )
    valid = graph.elementwise(
        mask_queries,
        graph.cast(mask_keys, graph.trt.bool),
        graph.trt.ElementWiseOperation.AND,
    )
    valid = graph.elementwise(
        valid,
        graph.equal(uid_queries, graph.cast(uid_keys, uid_queries.dtype)),
        graph.trt.ElementWiseOperation.AND,
    )
    valid = graph.cast(
        graph.reshape(
            valid,
            (1, windows, ATOM_WINDOW_QUERIES, ATOM_WINDOW_KEYS, 1),
        ),
        graph.trt.float32,
    )

    def masked_linear(tensor: Any, prefix: str):
        return graph.mul(graph.linear(tensor, prefix), valid)

    prefix = "diffusion_conditioning.atom_encoder"
    p = masked_linear(displacement, f"{prefix}.embed_atompair_ref_pos")
    p = graph.add(
        p,
        masked_linear(reciprocal_distance, f"{prefix}.embed_atompair_ref_dist"),
    )
    p = graph.add(p, masked_linear(valid, f"{prefix}.embed_atompair_mask"))
    base_pair = p

    # Structure conditioning is explicitly FP32 in upstream Boltz.
    s_projection = graph.layer_norm(s_trunk, f"{prefix}.s_to_c_trans.0")
    s_projection = graph.linear(s_projection, f"{prefix}.s_to_c_trans.1")
    token_map = graph.cast(atom_to_token, graph.trt.float32)
    s_projection = graph.network.add_matrix_multiply(
        token_map,
        graph.trt.MatrixOperation.NONE,
        s_projection,
        graph.trt.MatrixOperation.NONE,
    ).get_output(0)
    c = graph.add(c, s_projection)

    z_projection = graph.layer_norm(z, f"{prefix}.z_to_p_trans.0")
    z_projection = graph.linear(z_projection, f"{prefix}.z_to_p_trans.1")
    query_map = graph.reshape(
        token_map,
        (1, windows, ATOM_WINDOW_QUERIES, int(token_map.shape[-1])),
    )
    key_map = _to_keys(graph, token_map)
    # Express the two one-hot token-to-atom contractions as explicit matrix
    # multiplies. TensorRT 11.2 accepts the equivalent three-input Einsum but
    # produces an incorrect result for this rank-five equation.
    z_flat = graph.reshape(
        z_projection,
        (1, 1, int(z_projection.shape[1]), int(z_projection.shape[2]) * 16),
    )
    z_projection = graph.network.add_matrix_multiply(
        query_map,
        graph.trt.MatrixOperation.NONE,
        z_flat,
        graph.trt.MatrixOperation.NONE,
    ).get_output(0)
    z_projection = graph.reshape(
        z_projection,
        (1, windows, ATOM_WINDOW_QUERIES, int(token_map.shape[-1]), 16),
    )
    z_projection = graph.transpose(z_projection, (0, 1, 2, 4, 3))
    key_map = graph.transpose(key_map, (0, 1, 3, 2))
    key_map = graph.reshape(
        key_map,
        (1, windows, 1, int(token_map.shape[-1]), ATOM_WINDOW_KEYS),
    )
    z_projection = graph.network.add_matrix_multiply(
        z_projection,
        graph.trt.MatrixOperation.NONE,
        key_map,
        graph.trt.MatrixOperation.NONE,
    ).get_output(0)
    z_projection = graph.transpose(z_projection, (0, 1, 2, 4, 3))
    p = graph.add(p, z_projection)

    c_queries = graph.reshape(c, (1, windows, ATOM_WINDOW_QUERIES, 1, ATOM_CHANNELS))
    c_keys = graph.reshape(
        _to_keys(graph, c), (1, windows, 1, ATOM_WINDOW_KEYS, ATOM_CHANNELS)
    )
    query_pair = graph.linear(graph.relu(c_queries), f"{prefix}.c_to_p_trans_q.1")
    key_pair = graph.linear(graph.relu(c_keys), f"{prefix}.c_to_p_trans_k.1")
    p = graph.add(p, query_pair)
    p = graph.add(p, key_pair)
    p_update = graph.linear(graph.relu(p), f"{prefix}.p_mlp.1")
    p_update = graph.linear(graph.relu(p_update), f"{prefix}.p_mlp.3")
    p_update = graph.linear(graph.relu(p_update), f"{prefix}.p_mlp.5")
    return (
        q,
        c,
        graph.add(p, p_update),
        (
            base_pair,
            z_projection,
            query_pair,
            key_pair,
            p_update,
        ),
    )


def define_diffusion_conditioning_network(
    network: Any,
    trt: Any,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    atom_count: int,
    debug_pair_output: bool = False,
):
    """Define conditioning computed once before the 200 denoising steps."""

    atom_windows(atom_count)
    bf16 = getattr(trt, "bfloat16", None)
    if bf16 is None:
        raise RuntimeError("Boltz-2 requires TensorRT with strongly typed BF16 support")
    graph = Graph(network, trt, weights)
    s_trunk = network.add_input("s_trunk", trt.float32, (1, token_count, TOKEN_CHANNELS))
    z_trunk = network.add_input(
        "z_trunk", trt.float32, (1, token_count, token_count, PAIR_CHANNELS)
    )
    relative = network.add_input(
        "relative_position_encoding",
        bf16,
        (1, token_count, token_count, PAIR_CHANNELS),
    )
    features = {
        "ref_pos": network.add_input("ref_pos", trt.float32, (1, atom_count, 3)),
        "ref_space_uid": network.add_input("ref_space_uid", trt.int32, (1, atom_count)),
        "ref_charge": network.add_input("ref_charge", trt.float32, (1, atom_count)),
        "ref_element": network.add_input("ref_element", trt.int32, (1, atom_count, 128)),
        "ref_atom_name_chars": network.add_input(
            "ref_atom_name_chars", trt.int32, (1, atom_count, 4, 64)
        ),
        "atom_to_token": network.add_input(
            "atom_to_token", trt.int32, (1, atom_count, token_count)
        ),
        "atom_pad_mask": network.add_input("atom_pad_mask", trt.float32, (1, atom_count)),
    }

    z = graph.concatenate((z_trunk, graph.cast(relative, z_trunk.dtype)), 3)
    z = graph.layer_norm(z, "diffusion_conditioning.pairwise_conditioner.dim_pairwise_init_proj.0")
    z = graph.linear(
        graph.cast(z, bf16),
        "diffusion_conditioning.pairwise_conditioner.dim_pairwise_init_proj.1",
    )
    for layer in range(2):
        update = add_transition(
            graph,
            z,
            f"diffusion_conditioning.pairwise_conditioner.transitions.{layer}",
        )
        z = graph.add(z, update)

    q, c, p, pair_parts = _atom_reference_encoder(
        graph, features, s_trunk, graph.cast(z, trt.float32)
    )

    def projected_biases(prefix: str, layers: int, heads: int, source: Any):
        outputs = []
        for layer in range(layers):
            normalized = graph.layer_norm(source, f"{prefix}.{layer}.0")
            outputs.append(graph.linear(graph.cast(normalized, bf16), f"{prefix}.{layer}.1"))
        return graph.concatenate(outputs, len(source.shape) - 1)

    atom_enc_bias = projected_biases(
        "diffusion_conditioning.atom_enc_proj_z", ATOM_LAYERS, ATOM_HEADS, p
    )
    atom_dec_bias = projected_biases(
        "diffusion_conditioning.atom_dec_proj_z", ATOM_LAYERS, ATOM_HEADS, p
    )
    token_trans_bias = projected_biases(
        "diffusion_conditioning.token_trans_proj_z",
        TOKEN_TRANSFORMER_LAYERS,
        TOKEN_TRANSFORMER_HEADS,
        z,
    )
    outputs = (
        ("q", q),
        ("c", c),
        ("atom_enc_bias", graph.cast(atom_enc_bias, trt.float32)),
        ("atom_dec_bias", graph.cast(atom_dec_bias, trt.float32)),
        ("token_trans_bias", graph.cast(token_trans_bias, trt.float32)),
    )
    for name, tensor in outputs:
        tensor.name = name
        network.mark_output(tensor)
    if debug_pair_output:
        p.name = "atom_pair_conditioning"
        network.mark_output(p)
        for name, tensor in zip(
            ("pair_base", "pair_z", "pair_query", "pair_key", "pair_mlp"),
            pair_parts,
            strict=True,
        ):
            tensor.name = name
            network.mark_output(tensor)
    return tuple(tensor for _, tensor in outputs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_diffusion_conditioning_engine(
    checkpoint_path: Path,
    engine_path: Path,
    *,
    token_count: int = 117,
    atom_count: int = ATOM_COUNT,
    workspace_bytes: int = 16 << 30,
    verbose: bool = False,
    verify_checkpoint: bool = True,
    avg_timing_iterations: int = 8,
) -> DiffusionConditioningBuildResult:
    """Build one static-profile diffusion-conditioning engine."""

    _, weights = load_weight_prefixes(
        checkpoint_path,
        ("diffusion_conditioning.",),
        verify=verify_checkpoint,
    )
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    )
    define_diffusion_conditioning_network(
        network,
        trt,
        weights,
        token_count=token_count,
        atom_count=atom_count,
    )
    config = builder.create_builder_config()
    config.avg_timing_iterations = avg_timing_iterations
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT failed to build Boltz-2 diffusion conditioning")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return DiffusionConditioningBuildResult(
        engine_path=str(engine_path),
        engine_sha256=_sha256(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        token_count=token_count,
        atom_count=atom_count,
        precision="bf16-mixed",
    )
