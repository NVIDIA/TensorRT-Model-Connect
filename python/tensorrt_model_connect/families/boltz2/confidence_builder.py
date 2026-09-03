# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT confidence stack and prediction heads for Boltz-2."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorrt_model_connect import trt_compat

from .checkpoint import PairformerConfig, load_weight_prefixes
from .graph_ops import Graph
from .pairformer_builder import add_pairformer_block
from .trunk_init_builder import _contact_conditioning, _relative_position_encoding


TOKEN_COUNT = 117
ATOM_COUNT = 928
TOKEN_CHANNELS = 384
PAIR_CHANNELS = 128
CONFIDENCE_BLOCKS = 8
CONFIDENCE_CONFIG = PairformerConfig(
    token_s=TOKEN_CHANNELS,
    token_z=PAIR_CHANNELS,
    num_blocks=CONFIDENCE_BLOCKS,
    num_heads=16,
    pairwise_head_width=32,
    pairwise_num_heads=4,
    post_layer_norm=False,
    v2=True,
)


@dataclass(frozen=True)
class ConfidenceBuildResult:
    engine_path: str
    engine_sha256: str
    engine_size_bytes: int
    build_seconds: float
    token_count: int
    atom_count: int
    precision: str


def define_confidence_network(
    network: Any,
    trt: Any,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    atom_count: int,
):
    """Define confidence preparation, eight Pairformer blocks, and logits."""

    if token_count <= 0 or atom_count <= 0:
        raise ValueError("Boltz-2 confidence token and atom counts must be positive")
    bf16 = getattr(trt, "bfloat16", None)
    if bf16 is None:
        raise RuntimeError("Boltz-2 requires TensorRT with strongly typed BF16 support")
    graph = Graph(network, trt, weights)
    s_inputs_raw = network.add_input("s_inputs", trt.float32, (1, token_count, TOKEN_CHANNELS))
    s_raw = network.add_input("s", trt.float32, (1, token_count, TOKEN_CHANNELS))
    z_raw = network.add_input("z", trt.float32, (1, token_count, token_count, PAIR_CHANNELS))
    x_pred = network.add_input("x_pred", trt.float32, (1, atom_count, 3))
    token_to_rep_atom = network.add_input(
        "token_to_rep_atom", trt.int32, (1, token_count, atom_count)
    )
    features = {
        name: network.add_input(name, trt.int32, (1, token_count))
        for name in ("asym_id", "residue_index", "entity_id", "token_index", "sym_id")
    }
    token_bonds = network.add_input("token_bonds", trt.float32, (1, token_count, token_count, 1))
    type_bonds = network.add_input("type_bonds", trt.int32, (1, token_count, token_count))
    contact_conditioning = network.add_input(
        "contact_conditioning", trt.int32, (1, token_count, token_count, 5)
    )
    contact_threshold = network.add_input(
        "contact_threshold", trt.float32, (1, token_count, token_count)
    )
    token_mask = network.add_input("token_mask", trt.float32, (1, token_count))

    prefix = "confidence_module"
    s_inputs = graph.layer_norm(s_inputs_raw, f"{prefix}.s_inputs_norm")
    s = graph.layer_norm(s_raw, f"{prefix}.s_norm")
    s_update = graph.linear(graph.cast(s_inputs, bf16), f"{prefix}.s_input_to_s")
    s = graph.add(s, graph.cast(s_update, s.dtype))
    z = graph.layer_norm(z_raw, f"{prefix}.z_norm")
    relative = _relative_position_encoding(
        graph,
        features,
        bf16,
        prefix=f"{prefix}.rel_pos",
    )
    z = graph.add(z, graph.cast(relative, z.dtype))
    z = graph.add(
        z,
        graph.cast(
            graph.linear(graph.cast(token_bonds, bf16), f"{prefix}.token_bonds"),
            z.dtype,
        ),
    )
    z = graph.add(
        z,
        graph.cast(graph.embedding(type_bonds, f"{prefix}.token_bonds_type", bf16), z.dtype),
    )
    contact = _contact_conditioning(
        graph,
        contact_conditioning,
        contact_threshold,
        bf16,
        prefix=f"{prefix}.contact_conditioning",
    )
    z = graph.add(z, graph.cast(contact, z.dtype))

    first_s = graph.linear(graph.cast(s_inputs, bf16), f"{prefix}.s_to_z")
    second_s = graph.linear(graph.cast(s_inputs, bf16), f"{prefix}.s_to_z_transpose")
    z = graph.add(
        z,
        graph.cast(graph.reshape(first_s, (1, token_count, 1, PAIR_CHANNELS)), z.dtype),
    )
    z = graph.add(
        z,
        graph.cast(graph.reshape(second_s, (1, 1, token_count, PAIR_CHANNELS)), z.dtype),
    )
    product_first = graph.linear(graph.cast(s_inputs, bf16), f"{prefix}.s_to_z_prod_in1")
    product_second = graph.linear(graph.cast(s_inputs, bf16), f"{prefix}.s_to_z_prod_in2")
    product = graph.mul(
        graph.reshape(product_first, (1, token_count, 1, PAIR_CHANNELS)),
        graph.reshape(product_second, (1, 1, token_count, PAIR_CHANNELS)),
    )
    product = graph.linear(product, f"{prefix}.s_to_z_prod_out")
    z = graph.add(z, graph.cast(product, z.dtype))

    representative = graph.network.add_matrix_multiply(
        graph.cast(token_to_rep_atom, trt.float32),
        trt.MatrixOperation.NONE,
        x_pred,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    rows = graph.reshape(representative, (1, token_count, 1, 3))
    columns = graph.reshape(representative, (1, 1, token_count, 3))
    delta = graph.sub(rows, columns)
    distance = graph.reduce_sum(graph.mul(delta, delta), 3, keep_dims=False)
    distance = graph.unary(distance, trt.UnaryOperation.SQRT)
    boundaries = graph.constant(
        graph.weight(f"{prefix}.boundaries"),
        (1, 1, 1, 63),
    )
    expanded_distance = graph.reshape(distance, (1, token_count, token_count, 1))
    bins = graph.elementwise(expanded_distance, boundaries, trt.ElementWiseOperation.GREATER)
    bins = graph.reduce_sum(graph.cast(bins, trt.int32), 3, keep_dims=False)
    distance_embedding = graph.embedding(bins, f"{prefix}.dist_bin_pairwise_embed", trt.float32)
    z = graph.add(z, distance_embedding)

    rows_mask = graph.reshape(token_mask, (1, token_count, 1))
    columns_mask = graph.reshape(token_mask, (1, 1, token_count))
    pair_mask = graph.mul(rows_mask, columns_mask)
    for block in range(CONFIDENCE_BLOCKS):
        s, z = add_pairformer_block(
            graph,
            s,
            z,
            token_mask,
            pair_mask,
            block,
            CONFIDENCE_CONFIG,
            module_prefix=f"{prefix}.pairformer_stack.layers",
        )

    same_chain = graph.equal(
        graph.reshape(features["asym_id"], (1, token_count, 1)),
        graph.reshape(features["asym_id"], (1, 1, token_count)),
    )
    different_chain = graph.unary(same_chain, trt.UnaryOperation.NOT)
    head = f"{prefix}.confidence_heads"
    pae_intra = graph.linear(graph.cast(z, bf16), f"{head}.to_pae_intra_logits")
    pae_inter = graph.linear(graph.cast(z, bf16), f"{head}.to_pae_inter_logits")
    head_shape = (1, token_count, token_count, 1)
    pae = graph.add(
        graph.mul(pae_intra, graph.cast(graph.reshape(same_chain, head_shape), pae_intra.dtype)),
        graph.mul(
            pae_inter,
            graph.cast(graph.reshape(different_chain, head_shape), pae_inter.dtype),
        ),
    )
    symmetric_z = graph.add(z, graph.transpose(z, (0, 2, 1, 3)))
    pde_intra = graph.linear(graph.cast(symmetric_z, bf16), f"{head}.to_pde_intra_logits")
    pde_inter = graph.linear(graph.cast(symmetric_z, bf16), f"{head}.to_pde_inter_logits")
    pde = graph.add(
        graph.mul(pde_intra, graph.cast(graph.reshape(same_chain, head_shape), pde_intra.dtype)),
        graph.mul(
            pde_inter,
            graph.cast(graph.reshape(different_chain, head_shape), pde_inter.dtype),
        ),
    )
    plddt = graph.linear(graph.cast(s, bf16), f"{head}.to_plddt_logits")
    resolved = graph.linear(graph.cast(s, bf16), f"{head}.to_resolved_logits")

    # These trunk heads share this engine so the native runtime does not need
    # another launch or access to checkpoint weights.
    distogram_input = graph.add(z_raw, graph.transpose(z_raw, (0, 2, 1, 3)))
    pdistogram = graph.linear(graph.cast(distogram_input, bf16), "distogram_module.distogram")
    pdistogram = graph.reshape(pdistogram, (1, token_count, token_count, 1, 64))
    pbfactor = graph.linear(graph.cast(s_raw, bf16), "bfactor_module.bfactor")

    outputs = (
        ("pae_logits", pae),
        ("pde_logits", pde),
        ("plddt_logits", plddt),
        ("resolved_logits", resolved),
        ("representative_distance", distance),
        ("pdistogram", pdistogram),
        ("pbfactor", pbfactor),
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
    token_count: int = TOKEN_COUNT,
    atom_count: int = ATOM_COUNT,
    workspace_bytes: int = 16 << 30,
    verbose: bool = False,
    verify_checkpoint: bool = True,
    avg_timing_iterations: int = 8,
) -> ConfidenceBuildResult:
    """Build the complete static-profile confidence and output-head engine."""

    _, weights = load_weight_prefixes(
        checkpoint_path,
        ("confidence_module.", "distogram_module.", "bfactor_module."),
        verify=verify_checkpoint,
    )
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
    )
    config = builder.create_builder_config()
    config.avg_timing_iterations = avg_timing_iterations
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT failed to build Boltz-2 confidence engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return ConfidenceBuildResult(
        engine_path=str(engine_path),
        engine_sha256=_sha256(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        token_count=token_count,
        atom_count=atom_count,
        precision="bf16-mixed",
    )
