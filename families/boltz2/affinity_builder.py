# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT builders for the two Boltz-2 affinity ensemble heads."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import trt_compat
from .checkpoint import (
    PINNED_PAIRFORMER,
    load_weight_prefixes,
    validate_affinity_checkpoint,
)
from .graph_ops import Graph
from .pairformer_builder import add_pairformer_no_seq_block, add_transition


AFFINITY_BLOCKS = {1: 8, 2: 4}


@dataclass(frozen=True)
class AffinityBuildResult:
    engine_path: str
    engine_size_bytes: int
    build_seconds: float
    ensemble_member: int
    block_count: int
    token_count: int
    atom_count: int
    precision: str


def _validate_configuration(hparams: Mapping[str, Any], ensemble_member: int) -> int:
    if hparams.get("affinity_prediction") is not True:
        raise ValueError("Boltz-2 affinity checkpoint does not enable affinity prediction")
    if hparams.get("affinity_ensemble") is not True:
        raise ValueError("Boltz-2 affinity checkpoint does not contain the pinned ensemble")
    expected_blocks = AFFINITY_BLOCKS.get(ensemble_member)
    if expected_blocks is None:
        raise ValueError("Boltz-2 affinity ensemble_member must be 1 or 2")
    raw = hparams.get(f"affinity_model_args{ensemble_member}")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("pairformer_args"), Mapping):
        raise ValueError("Boltz-2 affinity checkpoint is missing model arguments")
    blocks = raw["pairformer_args"].get("num_blocks")
    if blocks != expected_blocks:
        raise ValueError(
            "Boltz-2 affinity Pairformer topology differs from the pinned topology: "
            f"{blocks!r} != {expected_blocks}"
        )
    return expected_blocks


def _pair_distances(graph: Graph, coordinates: Any, token_count: int):
    first = graph.reshape(coordinates, (1, token_count, 1, 3))
    second = graph.reshape(coordinates, (1, 1, token_count, 3))
    delta = graph.sub(first, second)
    squared = graph.reduce_sum(graph.mul(delta, delta), 3, keep_dims=False)
    return graph.unary(squared, graph.trt.UnaryOperation.SQRT)


def _cross_pair_mask(
    graph: Graph,
    mol_type: Any,
    affinity_token_mask: Any,
    token_mask: Any,
    token_count: int,
):
    receptor = graph.cast(
        graph.equal(mol_type, graph.integer_scalar_like(0, mol_type)),
        graph.trt.float32,
    )
    ligand = graph.cast(
        graph.equal(
            affinity_token_mask,
            graph.integer_scalar_like(1, affinity_token_mask),
        ),
        graph.trt.float32,
    )
    receptor = graph.mul(receptor, token_mask)
    ligand = graph.mul(ligand, token_mask)
    ligand_rows = graph.reshape(ligand, (1, token_count, 1))
    ligand_columns = graph.reshape(ligand, (1, 1, token_count))
    receptor_rows = graph.reshape(receptor, (1, token_count, 1))
    receptor_columns = graph.reshape(receptor, (1, 1, token_count))
    return graph.add(
        graph.add(
            graph.mul(ligand_rows, receptor_columns),
            graph.mul(receptor_rows, ligand_columns),
        ),
        graph.mul(ligand_rows, ligand_columns),
    )


def _pairwise_conditioning(graph: Graph, z: Any, distogram: Any, prefix: str):
    conditioned = graph.concatenate((z, distogram), 3)
    conditioned = graph.layer_norm(
        conditioned,
        f"{prefix}.pairwise_conditioner.dim_pairwise_init_proj.0",
    )
    conditioned = graph.linear(
        conditioned,
        f"{prefix}.pairwise_conditioner.dim_pairwise_init_proj.1",
    )
    for index in range(2):
        update = add_transition(
            graph,
            conditioned,
            f"{prefix}.pairwise_conditioner.transitions.{index}",
            low_precision=False,
        )
        conditioned = graph.add(conditioned, update)
    return conditioned


def _affinity_mlp(graph: Graph, value: Any, prefix: str):
    value = graph.relu(graph.linear(value, f"{prefix}.0"))
    value = graph.relu(graph.linear(value, f"{prefix}.2"))
    return graph.linear(value, f"{prefix}.4")


def define_affinity_network(
    network: Any,
    trt: Any,
    weights: dict[str, np.ndarray],
    *,
    ensemble_member: int,
    block_count: int,
    token_count: int,
    atom_count: int,
):
    """Define one exact inference-mode member of the affinity ensemble."""

    if token_count <= 0 or atom_count <= 0:
        raise ValueError("Boltz-2 affinity profile dimensions must be positive")
    prefix = f"affinity_module{ensemble_member}"
    graph = Graph(network, trt, weights)
    s_inputs = network.add_input(
        "s_inputs_affinity", trt.float32, (1, token_count, PINNED_PAIRFORMER.token_s)
    )
    z = network.add_input(
        "z", trt.float32, (1, token_count, token_count, PINNED_PAIRFORMER.token_z)
    )
    x_pred = network.add_input("x_pred", trt.float32, (1, atom_count, 3))
    token_to_rep_atom = network.add_input(
        "token_to_rep_atom", trt.int32, (1, token_count, atom_count)
    )
    mol_type = network.add_input("mol_type", trt.int32, (1, token_count))
    affinity_token_mask = network.add_input("affinity_token_mask", trt.int32, (1, token_count))
    token_mask = network.add_input("token_mask", trt.float32, (1, token_count))

    pair_mask = _cross_pair_mask(
        graph,
        mol_type,
        affinity_token_mask,
        token_mask,
        token_count,
    )
    z = graph.mul(z, graph.reshape(pair_mask, (1, token_count, token_count, 1)))
    z = graph.linear(graph.layer_norm(z, f"{prefix}.z_norm"), f"{prefix}.z_linear")
    first = graph.reshape(
        graph.linear(s_inputs, f"{prefix}.s_to_z_prod_in1"),
        (1, token_count, 1, PINNED_PAIRFORMER.token_z),
    )
    second = graph.reshape(
        graph.linear(s_inputs, f"{prefix}.s_to_z_prod_in2"),
        (1, 1, token_count, PINNED_PAIRFORMER.token_z),
    )
    z = graph.add(graph.add(z, first), second)

    representative = network.add_matrix_multiply(
        graph.cast(token_to_rep_atom, trt.float32),
        trt.MatrixOperation.NONE,
        x_pred,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    distances = _pair_distances(graph, representative, token_count)
    boundaries = graph.constant(
        graph.weight(f"{prefix}.boundaries"),
        (1, 1, 1, 63),
    )
    bins = graph.elementwise(
        graph.reshape(distances, (1, token_count, token_count, 1)),
        boundaries,
        trt.ElementWiseOperation.GREATER,
    )
    bins = graph.reduce_sum(graph.cast(bins, trt.int32), 3, keep_dims=False)
    distogram = graph.embedding(bins, f"{prefix}.dist_bin_pairwise_embed", trt.float32)
    z = graph.add(z, _pairwise_conditioning(graph, z, distogram, prefix))

    for block in range(block_count):
        z = add_pairformer_no_seq_block(
            graph,
            z,
            pair_mask,
            f"{prefix}.pairformer_stack.layers.{block}",
            pairwise_num_heads=PINNED_PAIRFORMER.pairwise_num_heads,
            pairwise_head_width=PINNED_PAIRFORMER.pairwise_head_width,
            low_precision=False,
        )

    diagonal = graph.constant(
        np.eye(token_count, dtype=np.float32),
        (1, token_count, token_count),
    )
    head_mask = graph.mul(
        pair_mask,
        graph.sub(graph.scalar_like(1.0, diagonal), diagonal),
    )
    pooled = graph.mul(z, graph.reshape(head_mask, (1, token_count, token_count, 1)))
    pooled = graph.reduce_sum(pooled, 1, keep_dims=False)
    pooled = graph.reduce_sum(pooled, 1, keep_dims=False)
    count = graph.reduce_sum(head_mask, 1, keep_dims=False)
    count = graph.reduce_sum(count, 1, keep_dims=False)
    count = graph.add(count, graph.scalar_like(1.0e-7, count))
    pooled = graph.div(pooled, graph.reshape(count, (1, 1)))

    heads = f"{prefix}.affinity_heads"
    pooled = graph.relu(graph.linear(pooled, f"{heads}.affinity_out_mlp.0"))
    pooled = graph.relu(graph.linear(pooled, f"{heads}.affinity_out_mlp.2"))
    affinity_value = _affinity_mlp(
        graph,
        pooled,
        f"{heads}.to_affinity_pred_value",
    )
    affinity_score = _affinity_mlp(
        graph,
        pooled,
        f"{heads}.to_affinity_pred_score",
    )
    affinity_probability = graph.sigmoid(
        graph.linear(affinity_score, f"{heads}.to_affinity_logits_binary")
    )
    affinity_value.name = "affinity_pred_value"
    affinity_probability.name = "affinity_probability_binary"
    network.mark_output(affinity_value)
    network.mark_output(affinity_probability)
    return affinity_value, affinity_probability


def build_affinity_engine(
    checkpoint_path: Path,
    engine_path: Path,
    *,
    ensemble_member: int,
    token_count: int = 117,
    atom_count: int = 928,
    workspace_bytes: int = 16 << 30,
    avg_timing_iterations: int = 8,
    verbose: bool = False,
    verify_checkpoint: bool = True,
) -> AffinityBuildResult:
    """Build one direct static-profile Boltz-2 affinity ensemble engine."""

    if verify_checkpoint:
        validate_affinity_checkpoint(checkpoint_path)
    prefix = f"affinity_module{ensemble_member}."
    hparams, weights = load_weight_prefixes(
        checkpoint_path,
        (prefix,),
        verify=False,
    )
    block_count = _validate_configuration(hparams, ensemble_member)
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    )
    define_affinity_network(
        network,
        trt,
        weights,
        ensemble_member=ensemble_member,
        block_count=block_count,
        token_count=token_count,
        atom_count=atom_count,
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
        raise RuntimeError("TensorRT failed to build the Boltz-2 affinity engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return AffinityBuildResult(
        engine_path=str(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        ensemble_member=ensemble_member,
        block_count=block_count,
        token_count=token_count,
        atom_count=atom_count,
        precision="fp32",
    )
