# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct strongly typed TensorRT builder for the Boltz-2 MSA stack."""

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
from .pairformer_builder import add_pairformer_no_seq_block, add_transition


MSA_CHANNELS = 64
MSA_BLOCKS = 4
MSA_HEADS = 8
MSA_HEAD_WIDTH = 32
MSA_ALPHABET = 33


@dataclass(frozen=True)
class MsaBuildResult:
    engine_path: str
    engine_sha256: str
    engine_size_bytes: int
    build_seconds: float
    token_count: int
    msa_depth: int
    precision: str


def _pair_weighted_averaging(
    graph: Graph,
    m: Any,
    z: Any,
    pair_mask: Any,
    prefix: str,
):
    batch, msa_depth, tokens, _ = (int(dim) for dim in m.shape)
    normalized_m = graph.layer_norm(m, f"{prefix}.norm_m")
    normalized_z = graph.layer_norm(z, f"{prefix}.norm_z")
    value = graph.linear(normalized_m, f"{prefix}.proj_m")
    value = graph.reshape(
        value,
        (batch, msa_depth, tokens, MSA_HEADS, MSA_HEAD_WIDTH),
    )
    value = graph.transpose(value, (0, 3, 1, 2, 4))

    bias = graph.linear(
        graph.cast(normalized_z, graph.trt.bfloat16),
        f"{prefix}.proj_z",
    )
    bias = graph.transpose(bias, (0, 3, 1, 2))
    mask_bias = graph.reshape(pair_mask, (batch, 1, tokens, tokens))
    mask_bias = graph.sub(graph.scalar_like(1.0, mask_bias), mask_bias)
    mask_bias = graph.mul(mask_bias, graph.scalar_like(-1.0e6, mask_bias))
    weights = graph.softmax_last(graph.add(bias, graph.cast(mask_bias, bias.dtype)))

    output = graph.einsum((weights, value), "bhij,bhsjd->bhsid")
    output = graph.transpose(output, (0, 2, 3, 1, 4))
    output = graph.reshape(
        output,
        (batch, msa_depth, tokens, MSA_HEADS * MSA_HEAD_WIDTH),
    )
    gate = graph.sigmoid(graph.linear(normalized_m, f"{prefix}.proj_g"))
    return graph.linear(graph.mul(gate, output), f"{prefix}.proj_o")


def _outer_product_mean(
    graph: Graph,
    m: Any,
    msa_mask: Any,
    prefix: str,
):
    batch, _, tokens, _ = (int(dim) for dim in m.shape)
    normalized = graph.layer_norm(m, f"{prefix}.norm")
    mask = graph.reshape(msa_mask, (batch, int(msa_mask.shape[1]), tokens, 1))
    mask = graph.cast(mask, normalized.dtype)
    first = graph.mul(graph.linear(normalized, f"{prefix}.proj_a"), mask)
    second = graph.mul(graph.linear(normalized, f"{prefix}.proj_b"), mask)
    outer = graph.einsum(
        (graph.cast(first, graph.trt.float32), graph.cast(second, graph.trt.float32)),
        "bsic,bsjd->bijcd",
    )
    outer = graph.reshape(outer, (batch, tokens, tokens, MSA_HEAD_WIDTH**2))
    count = graph.einsum((mask, mask), "bsic,bsjc->bij")
    count = graph.maximum(count, graph.scalar_like(1.0, count))
    count = graph.reshape(count, (batch, tokens, tokens, 1))
    outer = graph.elementwise(
        outer,
        graph.cast(count, outer.dtype),
        graph.trt.ElementWiseOperation.DIV,
    )
    return graph.linear(graph.cast(outer, m.dtype), f"{prefix}.proj_o")


def define_msa_network(
    network: Any,
    trt: Any,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    msa_depth: int,
):
    """Define the four-block, depth-one qualified MSA graph."""

    if msa_depth != 1:
        raise ValueError("the initial Boltz-2 TensorRT MSA profile requires depth 1")
    bf16 = getattr(trt, "bfloat16", None)
    if bf16 is None:
        raise RuntimeError("Boltz-2 requires TensorRT with strongly typed BF16 support")
    graph = Graph(network, trt, weights)
    z_input = network.add_input(
        "z",
        trt.float32,
        (1, token_count, token_count, PINNED_PAIRFORMER.token_z),
    )
    s_inputs = network.add_input(
        "s_inputs",
        trt.float32,
        (1, token_count, PINNED_PAIRFORMER.token_s),
    )
    msa = network.add_input("msa", trt.int32, (1, msa_depth, token_count))
    has_deletion = network.add_input(
        "has_deletion",
        trt.int32,
        (1, msa_depth, token_count),
    )
    deletion_value = network.add_input(
        "deletion_value",
        trt.float32,
        (1, msa_depth, token_count),
    )
    msa_paired = network.add_input(
        "msa_paired",
        trt.float32,
        (1, msa_depth, token_count),
    )
    msa_mask = network.add_input(
        "msa_mask",
        trt.int32,
        (1, msa_depth, token_count),
    )
    token_mask = network.add_input("token_mask", trt.float32, (1, token_count))
    rows = graph.reshape(token_mask, (1, token_count, 1))
    columns = graph.reshape(token_mask, (1, 1, token_count))
    pair_mask = graph.mul(rows, columns)

    msa_features = graph.one_hot(msa, MSA_ALPHABET, bf16)
    extras = tuple(
        graph.reshape(
            graph.cast(feature, bf16),
            (1, msa_depth, token_count, 1),
        )
        for feature in (has_deletion, deletion_value, msa_paired)
    )
    m = graph.linear(
        graph.concatenate((msa_features, *extras), 3),
        "msa_module.msa_proj",
    )
    projected_s = graph.linear(graph.cast(s_inputs, bf16), "msa_module.s_proj")
    m = graph.add(m, graph.reshape(projected_s, (1, 1, token_count, MSA_CHANNELS)))

    z = z_input
    for block in range(MSA_BLOCKS):
        prefix = f"msa_module.layers.{block}"
        m = graph.add(
            m,
            _pair_weighted_averaging(
                graph,
                m,
                z,
                pair_mask,
                f"{prefix}.pair_weighted_averaging",
            ),
        )
        m = graph.add(m, add_transition(graph, m, f"{prefix}.msa_transition"))
        outer_update = _outer_product_mean(
                graph,
                m,
                msa_mask,
                f"{prefix}.outer_product_mean",
        )
        z = graph.add(z, graph.cast(outer_update, z.dtype))
        z = add_pairformer_no_seq_block(
            graph,
            z,
            pair_mask,
            f"{prefix}.pairformer_layer",
            pairwise_num_heads=4,
            pairwise_head_width=32,
        )

    # Boltz2.forward adds the MSA module's returned z to its input z.
    z = graph.add(z_input, z)
    z.name = "z_out"
    network.mark_output(z)
    return z


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_msa_engine(
    checkpoint_path: Path,
    engine_path: Path,
    *,
    token_count: int = 117,
    msa_depth: int = 1,
    workspace_bytes: int = 16 << 30,
    avg_timing_iterations: int = 8,
    verbose: bool = False,
    verify_checkpoint: bool = True,
) -> MsaBuildResult:
    """Build the direct static-profile Boltz-2 MSA engine."""

    _, weights = load_weight_prefixes(
        checkpoint_path,
        ("msa_module.",),
        verify=verify_checkpoint,
    )
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    )
    define_msa_network(
        network,
        trt,
        weights,
        token_count=token_count,
        msa_depth=msa_depth,
    )
    config = builder.create_builder_config()
    config.avg_timing_iterations = avg_timing_iterations
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT failed to build the Boltz-2 MSA engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return MsaBuildResult(
        engine_path=str(engine_path),
        engine_sha256=_sha256(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        token_count=token_count,
        msa_depth=msa_depth,
        precision="bf16-mixed",
    )
