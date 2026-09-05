# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenFold3 diffusion atom encoder and token-input TensorRT builder."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat

from .atom_attention_builder import (
    _key_blocks,
    _query_blocks,
    atom_pair_mask,
    atom_transformer,
    finalize_atom_pair,
    padded_atom_count,
    reference_atom_features,
)
from .checkpoint import load_weight_prefixes
from .graph_ops import Graph, diffusion_compute_dtype


@dataclass(frozen=True)
class DiffusionScoreInputBuildResult:
    engine_path: str
    engine_sha256: str
    engine_size_bytes: int
    build_seconds: float
    token_count: int
    atom_count: int
    precision: str


def define_diffusion_score_input_network(
    network,
    trt,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    atom_count: int,
    precision: str = "fp16",
):
    """Define Algorithm 20 through the input of its 24 token blocks."""

    padded_atoms = padded_atom_count(atom_count)
    windows = padded_atoms // 32
    compute_dtype = diffusion_compute_dtype(trt, precision)
    graph = Graph(network, trt, weights, precision=precision)
    features = {
        "ref_pos": network.add_input("ref_pos", trt.float32, (1, padded_atoms, 3)),
        "ref_mask": network.add_input("ref_mask", trt.float32, (1, padded_atoms)),
        "ref_element": network.add_input("ref_element", trt.float32, (1, padded_atoms, 119)),
        "ref_charge": network.add_input("ref_charge", trt.float32, (1, padded_atoms)),
        "ref_atom_name_chars": network.add_input(
            "ref_atom_name_chars", trt.float32, (1, padded_atoms, 4, 64)
        ),
        "ref_space_uid": network.add_input("ref_space_uid", trt.int32, (1, padded_atoms)),
        "atom_mask": network.add_input("atom_mask", trt.float32, (1, padded_atoms)),
    }
    atom_to_token = network.add_input("atom_to_token_index", trt.int32, (1, padded_atoms))
    noisy = network.add_input("noisy_positions", trt.float32, (1, padded_atoms, 3))
    noise_level = network.add_input("noise_level", trt.float32, (1,))
    s_conditioned = network.add_input("s_conditioned", trt.float32, (1, token_count, 384))
    s_trunk = network.add_input("s_trunk", trt.float32, (1, token_count, 384))
    z_conditioned = network.add_input(
        "z_conditioned", trt.float32, (1, token_count, token_count, 128)
    )
    prefix = "diffusion_module.atom_attn_enc"
    cl, plm = reference_atom_features(
        graph,
        features,
        prefix,
        atom_count=atom_count,
        padded_atoms=padded_atoms,
        finalize_pair=False,
    )
    token_to_atom = graph.one_hot(atom_to_token, token_count, trt.float32)
    token_to_atom = graph.mul(
        token_to_atom,
        graph.reshape(features["atom_mask"], (1, padded_atoms, 1)),
    )
    s_embed = graph.layer_norm(s_trunk, f"{prefix}.noisy_position_embedder.layer_norm_s")
    s_embed = graph.linear(
        graph.cast(s_embed, compute_dtype),
        f"{prefix}.noisy_position_embedder.linear_s",
    )
    atom_s = graph.network.add_matrix_multiply(
        token_to_atom,
        trt.MatrixOperation.NONE,
        graph.cast(s_embed, trt.float32),
        trt.MatrixOperation.NONE,
    ).get_output(0)
    cl = graph.add(cl, graph.cast(atom_s, cl.dtype))

    z_embed = graph.layer_norm(z_conditioned, f"{prefix}.noisy_position_embedder.layer_norm_z")
    z_embed = graph.linear(
        graph.cast(z_embed, compute_dtype),
        f"{prefix}.noisy_position_embedder.linear_z",
    )
    query_map = _query_blocks(graph, token_to_atom, padded_atoms)
    key_map = _key_blocks(
        graph,
        token_to_atom,
        atom_count=atom_count,
        padded_atoms=padded_atoms,
    )
    atom_pair_z = graph.einsum(
        (query_map, graph.cast(z_embed, query_map.dtype)),
        "bwqi,bijc->bwqjc",
    )
    atom_pair_z = graph.einsum((atom_pair_z, key_map), "bwqjc,bwkj->bwqkc")
    pair_mask = atom_pair_mask(
        graph,
        features["atom_mask"],
        atom_count=atom_count,
        padded_atoms=padded_atoms,
    )
    atom_pair_z = graph.mul(atom_pair_z, graph.reshape(pair_mask, (1, windows, 32, 128, 1)))
    plm = graph.add(plm, graph.cast(atom_pair_z, plm.dtype))

    t_squared = graph.mul(noise_level, noise_level)
    scale = graph.unary(
        graph.add(t_squared, graph.scalar_like(256.0, t_squared)),
        trt.UnaryOperation.SQRT,
    )
    rl = graph.div(noisy, graph.reshape(scale, (1, 1, 1)))
    q = graph.add(
        cl,
        graph.cast(
            graph.linear(
                graph.cast(rl, compute_dtype),
                f"{prefix}.noisy_position_embedder.linear_r",
            ),
            cl.dtype,
        ),
    )

    plm = finalize_atom_pair(
        graph,
        cl,
        plm,
        pair_mask,
        prefix,
        atom_count=atom_count,
        padded_atoms=padded_atoms,
    )
    q = atom_transformer(
        graph,
        q,
        cl,
        plm,
        features["atom_mask"],
        f"{prefix}.atom_transformer",
        atom_count=atom_count,
        padded_atoms=padded_atoms,
        compute_dtype=compute_dtype,
    )
    token_atom = graph.relu(
        graph.linear(graph.cast(q, compute_dtype), f"{prefix}.linear_q.0")
    )
    weighted_map = graph.mul(
        token_to_atom,
        graph.reshape(features["atom_mask"], (1, padded_atoms, 1)),
    )
    counts = graph.reduce_sum(weighted_map, 1, keep_dims=True)
    weighted_map = graph.div(
        weighted_map,
        graph.add(counts, graph.scalar_like(1.0e-9, counts)),
    )
    ai = graph.network.add_matrix_multiply(
        graph.transpose(weighted_map, (0, 2, 1)),
        trt.MatrixOperation.NONE,
        graph.cast(token_atom, trt.float32),
        trt.MatrixOperation.NONE,
    ).get_output(0)
    s_for_ai = graph.layer_norm(s_conditioned, "diffusion_module.layer_norm_s")
    s_for_ai = graph.linear(
        graph.cast(s_for_ai, compute_dtype),
        "diffusion_module.linear_s",
    )
    ai = graph.add(ai, graph.cast(s_for_ai, ai.dtype))
    outputs = (
        (graph.cast(ai, trt.float32), "token_representation"),
        (graph.cast(q, trt.float32), "atom_representation"),
        (graph.cast(cl, trt.float32), "atom_conditioning"),
        (graph.cast(plm, trt.float32), "atom_pair_conditioning"),
    )
    for tensor, name in outputs:
        tensor.name = name
        network.mark_output(tensor)
    return tuple(tensor for tensor, _ in outputs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_diffusion_score_input_engine(
    checkpoint_path: Path,
    engine_path: Path,
    *,
    token_count: int,
    atom_count: int,
    workspace_bytes: int = 16 << 30,
    verbose: bool = False,
    verify_checkpoint: bool = True,
    precision: str = "fp16",
) -> DiffusionScoreInputBuildResult:
    weights = load_weight_prefixes(
        checkpoint_path,
        (
            "diffusion_module.atom_attn_enc.",
            "diffusion_module.layer_norm_s.",
            "diffusion_module.linear_s.",
        ),
        verify=verify_checkpoint,
    )
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    )
    define_diffusion_score_input_network(
        network,
        trt,
        weights,
        token_count=token_count,
        atom_count=atom_count,
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
        raise RuntimeError("TensorRT failed to build OpenFold3 diffusion atom input")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return DiffusionScoreInputBuildResult(
        engine_path=str(engine_path),
        engine_sha256=_sha256(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        token_count=token_count,
        atom_count=atom_count,
        precision=f"{precision}-mixed",
    )
