# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT builder for OpenFold3 input and atom embeddings."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat

from .atom_attention_builder import atom_transformer, padded_atom_count, reference_atom_features
from .checkpoint import load_weight_prefixes
from .graph_ops import Graph, low_precision_dtype


@dataclass(frozen=True)
class InputEmbedderBuildResult:
    engine_path: str
    engine_sha256: str
    engine_size_bytes: int
    build_seconds: float
    token_count: int
    atom_count: int
    padded_atom_count: int
    precision: str


def _input(network, name: str, dtype, shape: tuple[int, ...]):
    return network.add_input(name, dtype, shape)


def define_input_embedder_network(
    network,
    trt,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    atom_count: int,
    precision: str = "fp16",
):
    """Define Algorithms 2–5 for one exact, preprocessed request shape."""

    padded_atoms = padded_atom_count(atom_count)
    lowp = low_precision_dtype(trt, precision)
    graph = Graph(network, trt, weights, precision=precision)
    features = {
        "ref_pos": _input(network, "ref_pos", trt.float32, (1, padded_atoms, 3)),
        "ref_mask": _input(network, "ref_mask", trt.float32, (1, padded_atoms)),
        "ref_element": _input(network, "ref_element", trt.float32, (1, padded_atoms, 119)),
        "ref_charge": _input(network, "ref_charge", trt.float32, (1, padded_atoms)),
        "ref_atom_name_chars": _input(
            network, "ref_atom_name_chars", trt.float32, (1, padded_atoms, 4, 64)
        ),
        "ref_space_uid": _input(network, "ref_space_uid", trt.int32, (1, padded_atoms)),
        "atom_mask": _input(network, "atom_mask", trt.float32, (1, padded_atoms)),
    }
    atom_to_token_index = _input(network, "atom_to_token_index", trt.int32, (1, padded_atoms))
    restype = _input(network, "restype", trt.float32, (1, token_count, 32))
    profile = _input(network, "profile", trt.float32, (1, token_count, 32))
    deletion_mean = _input(network, "deletion_mean", trt.float32, (1, token_count))
    relpos = _input(network, "relpos", trt.float32, (1, token_count, token_count, 139))
    token_bonds = _input(network, "token_bonds", trt.float32, (1, token_count, token_count))

    prefix = "input_embedder.atom_attn_enc"
    condition, pair = reference_atom_features(
        graph,
        features,
        prefix,
        atom_count=atom_count,
        padded_atoms=padded_atoms,
    )
    atom_representation = atom_transformer(
        graph,
        condition,
        condition,
        pair,
        features["atom_mask"],
        f"{prefix}.atom_transformer",
        atom_count=atom_count,
        padded_atoms=padded_atoms,
        compute_dtype=lowp,
    )
    atom_representation = graph.relu(
        graph.linear(
            graph.cast(atom_representation, lowp),
            f"{prefix}.linear_q.0",
        )
    )
    atom_map = graph.one_hot(atom_to_token_index, token_count, trt.float32)
    atom_map = graph.mul(
        atom_map,
        graph.reshape(features["atom_mask"], (1, padded_atoms, 1)),
    )
    counts = graph.reduce_sum(atom_map, 1, keep_dims=True)
    atom_map = graph.div(
        atom_map,
        graph.add(counts, graph.scalar_like(1.0e-9, counts)),
    )
    pooled = graph.network.add_matrix_multiply(
        graph.transpose(atom_map, (0, 2, 1)),
        trt.MatrixOperation.NONE,
        graph.cast(atom_representation, trt.float32),
        trt.MatrixOperation.NONE,
    ).get_output(0)
    s_input = graph.concatenate(
        (
            pooled,
            restype,
            profile,
            graph.reshape(deletion_mean, (1, token_count, 1)),
        ),
        2,
    )
    s_init = graph.linear(graph.cast(s_input, lowp), "input_embedder.linear_s")
    z_i = graph.linear(graph.cast(s_input, lowp), "input_embedder.linear_z_i")
    z_j = graph.linear(graph.cast(s_input, lowp), "input_embedder.linear_z_j")
    z_init = graph.add(
        graph.reshape(z_i, (1, token_count, 1, 128)),
        graph.reshape(z_j, (1, 1, token_count, 128)),
    )
    z_init = graph.add(
        z_init,
        graph.linear(graph.cast(relpos, lowp), "input_embedder.linear_relpos"),
    )
    bond_embedding = graph.linear(
        graph.cast(graph.reshape(token_bonds, (1, token_count, token_count, 1)), lowp),
        "input_embedder.linear_token_bonds",
    )
    z_init = graph.add(z_init, bond_embedding)
    outputs = (
        (graph.cast(s_input, trt.float32), "s_input"),
        (graph.cast(s_init, trt.float32), "s_init"),
        (graph.cast(z_init, trt.float32), "z_init"),
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


def build_input_embedder_engine(
    checkpoint_path: Path,
    engine_path: Path,
    *,
    token_count: int,
    atom_count: int,
    workspace_bytes: int = 16 << 30,
    verbose: bool = False,
    verify_checkpoint: bool = True,
    precision: str = "fp16",
) -> InputEmbedderBuildResult:
    """Build the pinned OpenFold3 input graph directly with TensorRT."""

    weights = load_weight_prefixes(checkpoint_path, ("input_embedder.",), verify=verify_checkpoint)
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    )
    define_input_embedder_network(
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
        raise RuntimeError("TensorRT failed to build the OpenFold3 input engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return InputEmbedderBuildResult(
        engine_path=str(engine_path),
        engine_sha256=_sha256(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        token_count=token_count,
        atom_count=atom_count,
        padded_atom_count=padded_atom_count(atom_count),
        precision=f"{precision}-mixed",
    )
