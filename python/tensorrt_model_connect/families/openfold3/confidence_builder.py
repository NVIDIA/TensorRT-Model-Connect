# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenFold3 confidence Pairformer and all structure prediction heads."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat

from .checkpoint import PairformerConfig, load_weight_prefixes
from .graph_ops import Graph, low_precision_dtype
from .pairformer_builder import add_pairformer_block


CONFIDENCE_BLOCKS = 4
MAX_ATOMS_PER_TOKEN = 23
CONFIDENCE_CONFIG = PairformerConfig(num_blocks=CONFIDENCE_BLOCKS)


@dataclass(frozen=True)
class ConfidenceBuildResult:
    engine_path: str
    engine_sha256: str
    engine_size_bytes: int
    build_seconds: float
    token_count: int
    atom_count: int
    precision: str


def _distance_embedding(graph: Graph, positions, token_count: int):
    row = graph.reshape(positions, (1, token_count, 1, 3))
    column = graph.reshape(positions, (1, 1, token_count, 3))
    delta = graph.sub(row, column)
    squared_distance = graph.reduce_sum(graph.mul(delta, delta), 3, keep_dims=True)
    squared_bins = np.linspace(3.25, 50.75, 39, dtype=np.float32) ** 2
    upper = np.concatenate((squared_bins[1:], np.asarray([1.0e8], np.float32)))
    shape = (1, 1, 1, 39)
    lower_test = graph.elementwise(
        squared_distance,
        graph.constant(squared_bins, shape),
        graph.trt.ElementWiseOperation.GREATER,
    )
    upper_test = graph.elementwise(
        squared_distance,
        graph.constant(upper, shape),
        graph.trt.ElementWiseOperation.LESS,
    )
    bins = graph.mul(
        graph.cast(lower_test, graph.trt.float32),
        graph.cast(upper_test, graph.trt.float32),
    )
    return graph.linear(bins, "aux_heads.pairformer_embedding.linear_distance")


def define_confidence_network(
    network,
    trt,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    atom_count: int,
    precision: str = "fp16",
):
    """Define AF3 Algorithm 31 and the symmetric distogram head."""

    if token_count <= 0 or atom_count <= 0:
        raise ValueError("OpenFold3 confidence shapes must be positive")
    lowp = low_precision_dtype(trt, precision)
    graph = Graph(network, trt, weights, precision=precision)
    s_input = network.add_input("s_input", trt.float32, (1, token_count, 449))
    s = network.add_input("s", trt.float32, (1, token_count, 384))
    z = network.add_input("z", trt.float32, (1, token_count, token_count, 128))
    positions = network.add_input("positions", trt.float32, (1, atom_count, 3))
    representative_map = network.add_input(
        "representative_atom_map", trt.float32, (1, token_count, atom_count)
    )
    atom_head_index = network.add_input("atom_head_index", trt.int32, (atom_count,))
    token_mask = network.add_input("token_mask", trt.float32, (1, token_count))

    distogram = graph.linear(graph.cast(z, lowp), "aux_heads.distogram.linear")
    distogram = graph.add(distogram, graph.transpose(distogram, (0, 2, 1, 3)))

    representative = graph.network.add_matrix_multiply(
        representative_map,
        trt.MatrixOperation.NONE,
        positions,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    pair_prefix = "aux_heads.pairformer_embedding"
    z_confidence = z
    linear_i = graph.linear(graph.cast(s_input, lowp), f"{pair_prefix}.linear_i")
    linear_j = graph.linear(graph.cast(s_input, lowp), f"{pair_prefix}.linear_j")
    z_confidence = graph.add(
        z_confidence,
        graph.cast(
            graph.reshape(linear_i, (1, token_count, 1, 128)),
            z_confidence.dtype,
        ),
    )
    z_confidence = graph.add(
        z_confidence,
        graph.cast(
            graph.reshape(linear_j, (1, 1, token_count, 128)),
            z_confidence.dtype,
        ),
    )
    z_confidence = graph.add(
        z_confidence,
        graph.cast(_distance_embedding(graph, representative, token_count), z.dtype),
    )
    pair_mask = graph.mul(
        graph.reshape(token_mask, (1, token_count, 1)),
        graph.reshape(token_mask, (1, 1, token_count)),
    )
    for block in range(CONFIDENCE_BLOCKS):
        s, z_confidence = add_pairformer_block(
            graph,
            s,
            z_confidence,
            token_mask,
            pair_mask,
            block,
            CONFIDENCE_CONFIG,
            module_prefix=f"{pair_prefix}.pairformer_stack.blocks",
        )

    pae = graph.linear(
        graph.cast(graph.layer_norm(z_confidence, "aux_heads.pae.layer_norm"), lowp),
        "aux_heads.pae.linear",
    )
    pde_input = graph.layer_norm(z_confidence, "aux_heads.pde.layer_norm")
    pde = graph.linear(graph.cast(pde_input, lowp), "aux_heads.pde.linear")
    pde = graph.add(pde, graph.transpose(pde, (0, 2, 1, 3)))
    normalized_s = graph.layer_norm(s, "aux_heads.plddt.layer_norm")
    plddt_padded = graph.linear(graph.cast(normalized_s, lowp), "aux_heads.plddt.linear")
    plddt_padded = graph.reshape(plddt_padded, (1, token_count * MAX_ATOMS_PER_TOKEN, 50))
    plddt = graph.gather(plddt_padded, atom_head_index, 1)
    resolved_s = graph.layer_norm(s, "aux_heads.experimentally_resolved.layer_norm")
    resolved_padded = graph.linear(
        graph.cast(resolved_s, lowp), "aux_heads.experimentally_resolved.linear"
    )
    resolved_padded = graph.reshape(resolved_padded, (1, token_count * MAX_ATOMS_PER_TOKEN, 2))
    resolved = graph.gather(resolved_padded, atom_head_index, 1)

    outputs = (
        ("pae_logits", pae),
        ("pde_logits", pde),
        ("plddt_logits", plddt),
        ("experimentally_resolved_logits", resolved),
        ("distogram_logits", distogram),
    )
    for name, tensor in outputs:
        tensor.name = name
        network.mark_output(tensor)
    return tuple(tensor for _, tensor in outputs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_confidence_engine(
    checkpoint_path: Path,
    engine_path: Path,
    *,
    token_count: int,
    atom_count: int,
    workspace_bytes: int = 16 << 30,
    verbose: bool = False,
    verify_checkpoint: bool = True,
    precision: str = "fp16",
) -> ConfidenceBuildResult:
    weights = load_weight_prefixes(checkpoint_path, ("aux_heads.",), verify=verify_checkpoint)
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    )
    define_confidence_network(
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
        raise RuntimeError("TensorRT failed to build the OpenFold3 confidence engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return ConfidenceBuildResult(
        engine_path=str(engine_path),
        engine_sha256=_sha256(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        token_count=token_count,
        atom_count=atom_count,
        precision=f"{precision}-mixed",
    )
