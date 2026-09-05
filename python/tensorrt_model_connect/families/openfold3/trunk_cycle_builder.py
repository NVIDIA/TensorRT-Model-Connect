# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenFold3 recycling and four-block MSA trunk-cycle TensorRT builder."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat

from .checkpoint import load_weight_prefixes
from .graph_ops import Graph, low_precision_dtype
from .pairformer_builder import add_pairformer_no_seq_block, add_transition


@dataclass(frozen=True)
class TrunkCycleBuildResult:
    engine_path: str
    engine_sha256: str
    engine_size_bytes: int
    build_seconds: float
    token_count: int
    msa_depth: int
    precision: str


def _pair_weighted_average(
    graph: Graph,
    m,
    z,
    pair_mask,
    prefix: str,
    *,
    msa_depth: int,
    tokens: int,
    lowp,
):
    normalized_m = graph.layer_norm(m, f"{prefix}.layer_norm_m")
    normalized_z = graph.layer_norm(z, f"{prefix}.layer_norm_z")
    pair_logits = graph.linear(graph.cast(normalized_z, lowp), f"{prefix}.linear_z")
    pair_logits = graph.transpose(pair_logits, (0, 3, 1, 2))
    pair_logits = graph.reshape(pair_logits, (1, 1, 8, tokens, tokens))
    mask_bias = graph.mul(
        graph.sub(pair_mask, graph.scalar_like(1.0, pair_mask)),
        graph.scalar_like(1.0e9, pair_mask),
    )
    mask_bias = graph.reshape(mask_bias, (1, 1, 1, tokens, tokens))
    pair_logits = graph.add(
        graph.cast(pair_logits, graph.accumulation_dtype),
        graph.cast(mask_bias, graph.accumulation_dtype),
    )
    probabilities = graph.softmax_last(pair_logits)
    value = graph.linear(graph.cast(normalized_m, lowp), f"{prefix}.linear_v")
    value = graph.reshape(value, (1, msa_depth, tokens, 8, 8))
    value = graph.transpose(value, (0, 1, 3, 2, 4))
    attended = graph.einsum(
        (probabilities, graph.cast(value, probabilities.dtype)),
        "bmhij,bmhjd->bmihd",
    )
    gate = graph.sigmoid(graph.linear(graph.cast(normalized_m, lowp), f"{prefix}.linear_g"))
    gate = graph.reshape(gate, (1, msa_depth, tokens, 8, 8))
    attended = graph.mul(graph.cast(attended, gate.dtype), gate)
    attended = graph.reshape(attended, (1, msa_depth, tokens, 64))
    return graph.linear(attended, f"{prefix}.linear_o")


def _outer_product_mean(
    graph: Graph,
    m,
    msa_mask,
    prefix: str,
    *,
    tokens: int,
    lowp,
):
    normalized = graph.layer_norm(m, f"{prefix}.layer_norm")
    mask = graph.reshape(msa_mask, (1, int(msa_mask.shape[1]), tokens, 1))
    a = graph.linear(graph.cast(normalized, lowp), f"{prefix}.linear_1")
    b = graph.linear(graph.cast(normalized, lowp), f"{prefix}.linear_2")
    a = graph.mul(a, graph.cast(mask, a.dtype))
    b = graph.mul(b, graph.cast(mask, b.dtype))
    outer = graph.einsum(
        (graph.cast(a, graph.accumulation_dtype), graph.cast(b, graph.accumulation_dtype)),
        "bmic,bmjd->bijcd",
    )
    outer = graph.reshape(outer, (1, tokens, tokens, 1024))
    outer = graph.linear(graph.cast(outer, lowp), f"{prefix}.linear_out")
    norm = graph.einsum((mask, mask), "bmic,bmjc->bijc")
    norm = graph.add(norm, graph.scalar_like(1.0e-3, norm))
    return graph.div(outer, graph.cast(norm, outer.dtype))


def _no_template_embedding(graph: Graph, z, pair_mask, *, lowp):
    """Apply the upstream four-identical-dummy-template path once.

    OpenFold3 v0.5.0 represents disabled template search as four identical
    placeholder templates.  Their geometric features are zero and every
    residue type is the final (unknown) class.  Template processing is
    independent per template before averaging, so evaluating one placeholder
    is mathematically identical while retaining the learned z-dependent path.
    """

    prefix = "template_embedder"
    pair_prefix = f"{prefix}.template_pair_embedder"
    unknown = np.zeros((1, 1, 1, 32), dtype=np.float32)
    unknown[..., 31] = 1.0
    placeholder = graph.cast(graph.constant(unknown), lowp)
    template = graph.linear(
        graph.cast(graph.layer_norm(z, f"{pair_prefix}.layer_norm_z"), lowp),
        f"{pair_prefix}.linear_z",
    )
    template = graph.add(
        template,
        graph.linear(placeholder, f"{pair_prefix}.aatype_linear_1"),
    )
    template = graph.add(
        template,
        graph.linear(placeholder, f"{pair_prefix}.aatype_linear_2"),
    )
    for block in range(2):
        template = add_pairformer_no_seq_block(
            graph,
            template,
            pair_mask,
            f"{prefix}.template_pair_stack.blocks.{block}",
            pairwise_num_heads=4,
            pairwise_head_width=16,
            pair_stack_component="",
        )
    template = graph.layer_norm(template, f"{prefix}.template_pair_stack.layer_norm")
    template = graph.relu(template)
    return graph.linear(graph.cast(template, lowp), f"{prefix}.linear_t")


def define_trunk_cycle_network(
    network,
    trt,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    msa_depth: int,
    precision: str = "fp16",
):
    """Define one recycle, MSA module, and pre-Pairformer single update."""

    lowp = low_precision_dtype(trt, precision)
    graph = Graph(network, trt, weights, precision=precision)
    s_input = network.add_input("s_input", trt.float32, (1, token_count, 449))
    s_init = network.add_input("s_init", trt.float32, (1, token_count, 384))
    z_init = network.add_input("z_init", trt.float32, (1, token_count, token_count, 128))
    s_previous = network.add_input("s_previous", trt.float32, (1, token_count, 384))
    z_previous = network.add_input("z_previous", trt.float32, (1, token_count, token_count, 128))
    token_mask = network.add_input("token_mask", trt.float32, (1, token_count))
    msa = network.add_input("msa", trt.float32, (1, msa_depth, token_count, 32))
    has_deletion = network.add_input("has_deletion", trt.float32, (1, msa_depth, token_count))
    deletion_value = network.add_input("deletion_value", trt.float32, (1, msa_depth, token_count))
    msa_mask = network.add_input("msa_mask", trt.float32, (1, msa_depth, token_count))
    pair_mask = graph.mul(
        graph.reshape(token_mask, (1, token_count, 1)),
        graph.reshape(token_mask, (1, 1, token_count)),
    )

    recycled_z = graph.layer_norm(z_previous, "layer_norm_z")
    recycled_z = graph.linear(graph.cast(recycled_z, lowp), "linear_z")
    z = graph.add(z_init, graph.cast(recycled_z, z_init.dtype))
    z = graph.add(z, graph.cast(_no_template_embedding(graph, z, pair_mask, lowp=lowp), z.dtype))

    msa_features = graph.concatenate(
        (
            msa,
            graph.reshape(has_deletion, (1, msa_depth, token_count, 1)),
            graph.reshape(deletion_value, (1, msa_depth, token_count, 1)),
        ),
        3,
    )
    m = graph.linear(graph.cast(msa_features, lowp), "msa_module_embedder.linear_m")
    s_to_m = graph.linear(graph.cast(s_input, lowp), "msa_module_embedder.linear_s_input")
    m = graph.add(
        m,
        graph.reshape(s_to_m, (1, 1, token_count, 64)),
    )
    for block in range(4):
        prefix = f"msa_module.blocks.{block}"
        opm = _outer_product_mean(
            graph,
            m,
            msa_mask,
            f"{prefix}.outer_product_mean",
            tokens=token_count,
            lowp=lowp,
        )
        z = graph.add(z, graph.cast(opm, z.dtype))
        if block != 3:
            update = _pair_weighted_average(
                graph,
                m,
                z,
                pair_mask,
                f"{prefix}.msa_att_row",
                msa_depth=msa_depth,
                tokens=token_count,
                lowp=lowp,
            )
            m = graph.add(m, graph.cast(update, m.dtype))
            update = add_transition(graph, m, f"{prefix}.msa_transition")
            update = graph.mul(
                update,
                graph.cast(
                    graph.reshape(msa_mask, (1, msa_depth, token_count, 1)),
                    update.dtype,
                ),
            )
            m = graph.add(m, graph.cast(update, m.dtype))
        z = add_pairformer_no_seq_block(
            graph,
            z,
            pair_mask,
            prefix,
            pairwise_num_heads=4,
            pairwise_head_width=32,
        )

    recycled_s = graph.layer_norm(s_previous, "layer_norm_s")
    recycled_s = graph.linear(graph.cast(recycled_s, lowp), "linear_s")
    s = graph.add(s_init, graph.cast(recycled_s, s_init.dtype))
    s.name = "s"
    z.name = "z"
    network.mark_output(s)
    network.mark_output(z)
    return s, z


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_trunk_cycle_engine(
    checkpoint_path: Path,
    engine_path: Path,
    *,
    token_count: int,
    msa_depth: int = 1,
    workspace_bytes: int = 16 << 30,
    verbose: bool = False,
    verify_checkpoint: bool = True,
    precision: str = "fp16",
) -> TrunkCycleBuildResult:
    """Build one reusable OpenFold3 recycle/MSA graph."""

    weights = load_weight_prefixes(
        checkpoint_path,
        (
            "layer_norm_z.",
            "linear_z.",
            "layer_norm_s.",
            "linear_s.",
            "msa_module_embedder.",
            "msa_module.",
            "template_embedder.",
        ),
        verify=verify_checkpoint,
    )
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    )
    define_trunk_cycle_network(
        network,
        trt,
        weights,
        token_count=token_count,
        msa_depth=msa_depth,
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
        raise RuntimeError("TensorRT failed to build the OpenFold3 trunk-cycle engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return TrunkCycleBuildResult(
        engine_path=str(engine_path),
        engine_sha256=_sha256(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        token_count=token_count,
        msa_depth=msa_depth,
        precision=f"{precision}-mixed",
    )
