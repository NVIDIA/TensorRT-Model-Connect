# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct strongly typed TensorRT builder for the OpenFold3 Pairformer trunk."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorrt_model_connect import trt_compat

from .checkpoint import PairformerConfig, load_pairformer_weights
from .graph_ops import Graph


@dataclass(frozen=True)
class PairformerBuildResult:
    engine_path: str
    engine_sha256: str
    engine_size_bytes: int
    build_seconds: float
    first_block: int
    block_count: int
    token_count: int
    precision: str
    topology: dict[str, Any]


def _pair_mask(graph: Graph, token_mask: Any, tokens: int):
    rows = graph.reshape(token_mask, (1, tokens, 1))
    columns = graph.reshape(token_mask, (1, 1, tokens))
    return graph.mul(rows, columns)


def _transition(
    graph: Graph,
    tensor: Any,
    prefix: str,
    *,
    low_precision: bool = True,
):
    normalized = graph.layer_norm(tensor, f"{prefix}.layer_norm")
    if low_precision:
        normalized = graph.cast(normalized, graph.low_precision_dtype)
    first = graph.silu(graph.linear(normalized, f"{prefix}.swiglu.linear_a"))
    second = graph.linear(normalized, f"{prefix}.swiglu.linear_b")
    return graph.linear(graph.mul(first, second), f"{prefix}.linear_out")


def add_transition(graph: Graph, tensor: Any, prefix: str):
    """Add the transition topology reused by OpenFold3 trunk submodules."""

    return _transition(graph, tensor, prefix)


def _triangle_multiplication(
    graph: Graph,
    z: Any,
    pair_mask: Any,
    prefix: str,
    *,
    outgoing: bool,
):
    normalized = graph.layer_norm(z, f"{prefix}.layer_norm_in")
    normalized_lowp = graph.cast(normalized, graph.low_precision_dtype)
    first = graph.mul(
        graph.linear(normalized_lowp, f"{prefix}.linear_a_p"),
        graph.sigmoid(graph.linear(normalized_lowp, f"{prefix}.linear_a_g")),
    )
    second = graph.mul(
        graph.linear(normalized_lowp, f"{prefix}.linear_b_p"),
        graph.sigmoid(graph.linear(normalized_lowp, f"{prefix}.linear_b_g")),
    )

    batch, tokens, _, channels = (int(dim) for dim in first.shape)
    mask = graph.reshape(pair_mask, (batch, tokens, tokens, 1))
    first = graph.mul(graph.cast(first, mask.dtype), mask)
    second = graph.mul(graph.cast(second, mask.dtype), mask)
    first = graph.cast(first, graph.accumulation_dtype)
    second = graph.cast(second, graph.accumulation_dtype)
    equation = "bikd,bjkd->bijd" if outgoing else "bkid,bkjd->bijd"
    contracted = graph.einsum((first, second), equation)

    contracted = graph.layer_norm(contracted, f"{prefix}.layer_norm_out")
    contracted = graph.cast(contracted, graph.low_precision_dtype)
    projected_out = graph.linear(contracted, f"{prefix}.linear_z")
    output_gate = graph.sigmoid(graph.linear(normalized_lowp, f"{prefix}.linear_g"))
    return graph.mul(projected_out, output_gate)


def _triangle_attention(
    graph: Graph,
    z: Any,
    pair_mask: Any,
    prefix: str,
    *,
    starting: bool,
    heads: int,
    head_width: int,
    transpose_bias: bool = False,
):
    if not starting:
        z = graph.transpose(z, (0, 2, 1, 3))
        pair_mask = graph.transpose(pair_mask, (0, 2, 1))

    batch, tokens, _, channels = (int(dim) for dim in z.shape)
    normalized = graph.layer_norm(z, f"{prefix}.layer_norm")
    normalized_lowp = graph.cast(normalized, graph.low_precision_dtype)
    triangle_bias = graph.linear(normalized_lowp, f"{prefix}.linear_z")
    triangle_bias = graph.transpose(
        triangle_bias,
        (0, 3, 2, 1) if transpose_bias else (0, 3, 1, 2),
    )
    triangle_bias = graph.reshape(triangle_bias, (batch, 1, heads, tokens, tokens))

    def projected(name: str):
        value = graph.linear(normalized_lowp, f"{prefix}.mha.linear_{name}")
        value = graph.reshape(value, (batch, tokens, tokens, heads, head_width))
        return graph.transpose(value, (0, 1, 3, 2, 4))

    query = projected("q")
    key = projected("k")
    value = projected("v")
    query = graph.mul(query, graph.scalar_like(1.0 / np.sqrt(head_width), query))
    scores = graph.network.add_matrix_multiply(
        graph.cast(query, graph.triangle_attention_dtype),
        graph.trt.MatrixOperation.NONE,
        graph.cast(key, graph.triangle_attention_dtype),
        graph.trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)

    mask_bias = graph.reshape(pair_mask, (batch, tokens, 1, 1, tokens))
    mask_bias = graph.sub(mask_bias, graph.scalar_like(1.0, mask_bias))
    mask_bias = graph.mul(mask_bias, graph.scalar_like(1.0e9, mask_bias))
    mask_bias = graph.cast(mask_bias, scores.dtype)
    scores = graph.add(scores, mask_bias)
    scores = graph.add(scores, graph.cast(triangle_bias, scores.dtype))
    probabilities = graph.softmax_last(scores)
    output = graph.network.add_matrix_multiply(
        probabilities,
        graph.trt.MatrixOperation.NONE,
        graph.cast(value, probabilities.dtype),
        graph.trt.MatrixOperation.NONE,
    ).get_output(0)
    output = graph.transpose(output, (0, 1, 3, 2, 4))

    gate = graph.sigmoid(graph.linear(normalized_lowp, f"{prefix}.mha.linear_g"))
    gate = graph.reshape(gate, (batch, tokens, tokens, heads, head_width))
    output = graph.mul(graph.cast(output, gate.dtype), gate)
    output = graph.reshape(output, (batch, tokens, tokens, channels))
    output = graph.linear(output, f"{prefix}.mha.linear_o")
    if not starting:
        output = graph.transpose(output, (0, 2, 1, 3))
    return output


def _sequence_attention(
    graph: Graph,
    s: Any,
    z: Any,
    token_mask: Any,
    prefix: str,
    *,
    heads: int,
):
    batch, tokens, channels = (int(dim) for dim in s.shape)
    head_width = channels // heads
    s = graph.cast(s, graph.trt.float32)
    normalized = graph.layer_norm(s, f"{prefix}.layer_norm_a")
    normalized_lowp = graph.cast(normalized, graph.low_precision_dtype)

    attention_prefix = prefix
    pair_bias = graph.cast(z, graph.trt.float32)
    pair_bias = graph.layer_norm(pair_bias, f"{attention_prefix}.layer_norm_z")
    pair_bias = graph.linear(
        graph.cast(pair_bias, graph.low_precision_dtype),
        f"{attention_prefix}.linear_z",
    )
    pair_bias = graph.cast(pair_bias, graph.stable_attention_dtype)
    pair_bias = graph.transpose(pair_bias, (0, 3, 1, 2))

    def projected(name: str):
        value = graph.linear(normalized_lowp, f"{attention_prefix}.mha.linear_{name}")
        value = graph.reshape(value, (batch, tokens, heads, head_width))
        return graph.transpose(value, (0, 2, 1, 3))

    query = projected("q")
    key = projected("k")
    value = projected("v")
    scores = graph.network.add_matrix_multiply(
        graph.cast(query, graph.stable_attention_dtype),
        graph.trt.MatrixOperation.NONE,
        graph.cast(key, graph.stable_attention_dtype),
        graph.trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    scores = graph.mul(scores, graph.scalar_like(1.0 / np.sqrt(head_width), scores))
    scores = graph.add(scores, pair_bias)

    mask = graph.reshape(token_mask, (batch, 1, 1, tokens))
    mask = graph.sub(graph.scalar_like(1.0, mask), mask)
    mask = graph.mul(mask, graph.scalar_like(-1.0e6, mask))
    scores = graph.add(scores, graph.cast(mask, scores.dtype))
    probabilities = graph.softmax_last(scores)
    output = graph.network.add_matrix_multiply(
        probabilities,
        graph.trt.MatrixOperation.NONE,
        graph.cast(value, graph.stable_attention_dtype),
        graph.trt.MatrixOperation.NONE,
    ).get_output(0)
    output = graph.transpose(output, (0, 2, 1, 3))
    output = graph.reshape(output, (batch, tokens, channels))
    output = graph.cast(output, graph.low_precision_dtype)
    gate = graph.sigmoid(graph.linear(normalized_lowp, f"{attention_prefix}.mha.linear_g"))
    output = graph.linear(graph.mul(gate, output), f"{attention_prefix}.mha.linear_o")
    return graph.add(s, graph.cast(output, s.dtype))


def add_pairformer_block(
    graph: Graph,
    s: Any,
    z: Any,
    token_mask: Any,
    pair_mask: Any,
    block: int,
    config: PairformerConfig,
    *,
    module_prefix: str = "pairformer_stack.blocks",
):
    """Add one exact inference-mode OpenFold3 Pairformer block."""

    prefix = f"{module_prefix}.{block}"
    z = add_pairformer_no_seq_block(
        graph,
        z,
        pair_mask,
        prefix,
        pairwise_num_heads=config.pairwise_num_heads,
        pairwise_head_width=config.pairwise_head_width,
    )

    s = _sequence_attention(
        graph,
        s,
        z,
        token_mask,
        f"{prefix}.attn_pair_bias",
        heads=config.num_heads,
    )
    s = graph.add(
        s,
        _transition(
            graph,
            s,
            f"{prefix}.single_transition",
            low_precision=False,
        ),
    )
    return s, z


def add_pairformer_no_seq_block(
    graph: Graph,
    z: Any,
    pair_mask: Any,
    prefix: str,
    *,
    pairwise_num_heads: int,
    pairwise_head_width: int,
    pair_stack_component: str = "pair_stack",
):
    """Add the family-owned pair-only stack shared by OpenFold3 modules."""

    stack_prefix = f"{prefix}.{pair_stack_component}" if pair_stack_component else prefix

    update = _triangle_multiplication(
        graph,
        z,
        pair_mask,
        f"{stack_prefix}.tri_mul_out",
        outgoing=True,
    )
    z = graph.add(z, graph.cast(update, z.dtype))
    update = _triangle_multiplication(
        graph,
        z,
        pair_mask,
        f"{stack_prefix}.tri_mul_in",
        outgoing=False,
    )
    z = graph.add(z, graph.cast(update, z.dtype))
    update = _triangle_attention(
        graph,
        z,
        pair_mask,
        f"{stack_prefix}.tri_att_start",
        starting=True,
        heads=pairwise_num_heads,
        head_width=pairwise_head_width,
    )
    z = graph.add(z, graph.cast(update, z.dtype))
    update = _triangle_attention(
        graph,
        z,
        pair_mask,
        f"{stack_prefix}.tri_att_end",
        starting=False,
        heads=pairwise_num_heads,
        head_width=pairwise_head_width,
        transpose_bias=True,
    )
    z = graph.add(z, graph.cast(update, z.dtype))
    transition = _transition(graph, z, f"{stack_prefix}.pair_transition")
    z = graph.add(z, graph.cast(transition, z.dtype))
    return z


def define_pairformer_network(
    network: Any,
    trt: Any,
    weights: dict[str, np.ndarray],
    config: PairformerConfig,
    *,
    first_block: int,
    block_count: int,
    token_count: int,
    precision: str = "fp16",
):
    """Define a contiguous direct-TensorRT Pairformer engine network."""

    s = network.add_input("s", trt.float32, (1, token_count, config.token_s))
    z = network.add_input(
        "z",
        trt.float32,
        (1, token_count, token_count, config.token_z),
    )
    token_mask = network.add_input("token_mask", trt.float32, (1, token_count))
    graph = Graph(network, trt, weights, precision=precision)
    pair_mask = _pair_mask(graph, token_mask, token_count)
    for block in range(first_block, first_block + block_count):
        s, z = add_pairformer_block(
            graph,
            s,
            z,
            token_mask,
            pair_mask,
            block,
            config,
        )
    s.name = "s_out"
    z.name = "z_out"
    network.mark_output(s)
    network.mark_output(z)
    return s, z


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pairformer_engine(
    checkpoint_path: Path,
    engine_path: Path,
    *,
    first_block: int = 0,
    block_count: int = 1,
    token_count: int = 117,
    workspace_bytes: int = 16 << 30,
    verbose: bool = False,
    verify_checkpoint: bool = True,
    precision: str = "fp16",
) -> PairformerBuildResult:
    """Build contiguous Pairformer blocks directly with TensorRT."""

    if token_count <= 0:
        raise ValueError("OpenFold3 Pairformer token_count must be positive")
    config = PairformerConfig()
    weights = load_pairformer_weights(
        checkpoint_path,
        first_block=first_block,
        block_count=block_count,
        verify=verify_checkpoint,
    )
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    )
    define_pairformer_network(
        network,
        trt,
        weights,
        config,
        first_block=first_block,
        block_count=block_count,
        token_count=token_count,
        precision=precision,
    )
    build_config = builder.create_builder_config()
    build_config.avg_timing_iterations = 8
    build_config.max_aux_streams = 0
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, build_config)
    build_seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT failed to build the OpenFold3 Pairformer engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return PairformerBuildResult(
        engine_path=str(engine_path),
        engine_sha256=_sha256(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        first_block=first_block,
        block_count=block_count,
        token_count=token_count,
        precision=f"{precision}-mixed",
        topology=asdict(config),
    )
