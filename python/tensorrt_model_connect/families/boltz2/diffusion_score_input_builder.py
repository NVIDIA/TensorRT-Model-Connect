# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct FP32 TensorRT graph for Boltz-2 score-model input encoding."""

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
    _atom_transformer,
    atom_windows,
)


TOKEN_COUNT = 117
TOKEN_CHANNELS = 384
SCORE_TOKEN_CHANNELS = 2 * TOKEN_CHANNELS
FOURIER_CHANNELS = 256


@dataclass(frozen=True)
class DiffusionScoreInputBuildResult:
    engine_path: str
    engine_sha256: str
    engine_size_bytes: int
    build_seconds: float
    token_count: int
    atom_count: int
    precision: str


def _transition(graph: Graph, tensor: Any, prefix: str):
    normalized = graph.layer_norm(tensor, f"{prefix}.norm")
    first = graph.silu(graph.linear(normalized, f"{prefix}.fc1"))
    second = graph.linear(normalized, f"{prefix}.fc2")
    return graph.linear(graph.mul(first, second), f"{prefix}.fc3")


def define_diffusion_score_input_network(
    network: Any,
    trt: Any,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    atom_count: int,
    debug_atom_outputs: bool = False,
    scheduler_fence_atom_attention: bool = True,
):
    """Define time/single conditioning plus the score atom encoder."""

    if token_count <= 0:
        raise ValueError("Boltz-2 score input token count must be positive")
    windows = atom_windows(atom_count)
    graph = Graph(network, trt, weights)
    s_inputs = network.add_input("s_inputs", trt.float32, (1, token_count, TOKEN_CHANNELS))
    s_trunk = network.add_input("s_trunk", trt.float32, (1, token_count, TOKEN_CHANNELS))
    q_static = network.add_input("q_static", trt.float32, (1, atom_count, ATOM_CHANNELS))
    c_static = network.add_input("c_static", trt.float32, (1, atom_count, ATOM_CHANNELS))
    atom_enc_bias = network.add_input(
        "atom_enc_bias",
        trt.float32,
        (1, windows, ATOM_WINDOW_QUERIES, ATOM_WINDOW_KEYS, ATOM_LAYERS * ATOM_HEADS),
    )
    r_noisy = network.add_input("r_noisy", trt.float32, (1, atom_count, 3))
    time = network.add_input("time", trt.float32, (1,))
    atom_to_token = network.add_input("atom_to_token", trt.int32, (1, atom_count, token_count))
    atom_pad_mask = network.add_input("atom_pad_mask", trt.float32, (1, atom_count))

    prefix = "structure_module.score_model.single_conditioner"
    single = graph.concatenate((s_trunk, s_inputs), 2)
    single = graph.layer_norm(single, f"{prefix}.norm_single")
    single = graph.linear(single, f"{prefix}.single_embed")
    fourier = graph.reshape(time, (1, 1))
    fourier = graph.linear(fourier, f"{prefix}.fourier_embed.proj")
    fourier = graph.mul(fourier, graph.scalar_like(2.0 * np.pi, fourier))
    fourier = graph.unary(fourier, trt.UnaryOperation.COS)
    fourier = graph.layer_norm(fourier, f"{prefix}.norm_fourier")
    time_embedding = graph.linear(fourier, f"{prefix}.fourier_to_single")
    single = graph.add(single, graph.reshape(time_embedding, (1, 1, SCORE_TOKEN_CHANNELS)))
    for layer in range(2):
        single = graph.add(single, _transition(graph, single, f"{prefix}.transitions.{layer}"))

    score_prefix = "structure_module.score_model.atom_attention_encoder"
    q = graph.add(q_static, graph.linear(r_noisy, f"{score_prefix}.r_to_q_trans"))
    q_initial = q
    atom_layers: list[Any] = []
    atom_attention_layers: list[Any] = []
    atom_adapted_layers: list[Any] = []
    atom_attention_details: list[dict[str, Any]] = []
    q = _atom_transformer(
        graph,
        q,
        c_static,
        atom_enc_bias,
        atom_pad_mask,
        layer_prefix=f"{score_prefix}.atom_encoder.diffusion_transformer.layers",
        compute_dtype=trt.float32,
        debug_outputs=atom_layers if debug_atom_outputs else None,
        debug_attention_outputs=atom_attention_layers if debug_atom_outputs else None,
        debug_adapted_outputs=atom_adapted_layers if debug_atom_outputs else None,
        debug_attention_detail_outputs=(
            atom_attention_details
            if debug_atom_outputs or scheduler_fence_atom_attention
            else None
        ),
    )
    q_skip = q
    token_projection = graph.relu(graph.linear(q, f"{score_prefix}.atom_to_token_trans.0"))
    token_map = graph.cast(atom_to_token, trt.float32)
    token_counts = graph.reduce_sum(token_map, 1, keep_dims=True)
    token_map = graph.div(
        token_map,
        graph.add(token_counts, graph.scalar_like(1.0e-6, token_counts)),
    )
    pooled = graph.network.add_matrix_multiply(
        graph.transpose(token_map, (0, 2, 1)),
        trt.MatrixOperation.NONE,
        token_projection,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    conditioned = graph.layer_norm(single, "structure_module.score_model.s_to_a_linear.0")
    conditioned = graph.linear(conditioned, "structure_module.score_model.s_to_a_linear.1")
    a = graph.add(pooled, conditioned)
    outputs = (
        ("a", a),
        ("single_condition", single),
        ("q_skip", q_skip),
        ("c_skip", graph.add(c_static, graph.scalar_like(0.0, c_static))),
    )
    for name, tensor in outputs:
        tensor.name = name
        network.mark_output(tensor)
    if debug_atom_outputs:
        q_initial.name = "q_initial"
        network.mark_output(q_initial)
        for layer, tensor in enumerate(atom_layers):
            tensor.name = f"q_layer_{layer}"
            network.mark_output(tensor)
        for layer, tensor in enumerate(atom_attention_layers):
            tensor.name = f"q_attention_{layer}"
            network.mark_output(tensor)
        for layer, tensor in enumerate(atom_adapted_layers):
            tensor.name = f"q_adapted_{layer}"
            network.mark_output(tensor)
        for layer, details in enumerate(atom_attention_details):
            for name, tensor in details.items():
                tensor.name = f"attention_{layer}_{name}"
                network.mark_output(tensor)
    elif scheduler_fence_atom_attention:
        for layer, details in enumerate(atom_attention_details):
            for name, tensor in details.items():
                tensor.name = f"encoder_fence_{layer}_{name}"
                network.mark_output(tensor)
    return tuple(tensor for _, tensor in outputs)


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
    token_count: int = TOKEN_COUNT,
    atom_count: int = ATOM_COUNT,
    workspace_bytes: int = 16 << 30,
    verbose: bool = False,
    verify_checkpoint: bool = True,
    debug_atom_outputs: bool = False,
    avg_timing_iterations: int = 8,
    allow_tf32: bool = False,
    scheduler_fence_atom_attention: bool = True,
    builder_optimization_level: int = 3,
) -> DiffusionScoreInputBuildResult:
    """Build one static-profile score-model input engine."""

    _, weights = load_weight_prefixes(
        checkpoint_path,
        ("structure_module.score_model.",),
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
        debug_atom_outputs=debug_atom_outputs,
        scheduler_fence_atom_attention=scheduler_fence_atom_attention,
    )
    config = builder.create_builder_config()
    config.avg_timing_iterations = avg_timing_iterations
    config.builder_optimization_level = builder_optimization_level
    config.max_aux_streams = 0
    if not allow_tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT failed to build Boltz-2 diffusion score input")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return DiffusionScoreInputBuildResult(
        engine_path=str(engine_path),
        engine_sha256=_sha256(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        token_count=token_count,
        atom_count=atom_count,
        precision="fp32-upstream-exact",
    )
