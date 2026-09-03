# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct FP32 TensorRT graph for the Boltz-2 score-model atom decoder."""

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
    atom_attention_detail_shapes,
    atom_windows,
)


TOKEN_COUNT = 117
SCORE_TOKEN_CHANNELS = 768


@dataclass(frozen=True)
class DiffusionScoreOutputBuildResult:
    engine_path: str
    engine_sha256: str
    engine_size_bytes: int
    build_seconds: float
    token_count: int
    atom_count: int
    precision: str


def define_diffusion_score_output_network(
    network: Any,
    trt: Any,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    atom_count: int,
    scheduler_fence_atom_attention: bool = True,
):
    """Define score normalization, atom broadcast, decoder, and coordinate head."""

    if token_count <= 0:
        raise ValueError("Boltz-2 score output token count must be positive")
    windows = atom_windows(atom_count)
    attention_shapes = atom_attention_detail_shapes(atom_count)
    graph = Graph(network, trt, weights)
    a = network.add_input("a", trt.float32, (1, token_count, SCORE_TOKEN_CHANNELS))
    q_skip = network.add_input("q_skip", trt.float32, (1, atom_count, ATOM_CHANNELS))
    c_skip = network.add_input("c_skip", trt.float32, (1, atom_count, ATOM_CHANNELS))
    atom_dec_bias = network.add_input(
        "atom_dec_bias",
        trt.float32,
        (1, windows, ATOM_WINDOW_QUERIES, ATOM_WINDOW_KEYS, ATOM_LAYERS * ATOM_HEADS),
    )
    atom_to_token = network.add_input("atom_to_token", trt.int32, (1, atom_count, token_count))
    atom_pad_mask = network.add_input("atom_pad_mask", trt.float32, (1, atom_count))

    a = graph.layer_norm(a, "structure_module.score_model.a_norm")
    decoder = "structure_module.score_model.atom_attention_decoder"
    token_update = graph.linear(a, f"{decoder}.a_to_q_trans")
    token_map = graph.cast(atom_to_token, trt.float32)
    atom_update = graph.network.add_matrix_multiply(
        token_map,
        trt.MatrixOperation.NONE,
        token_update,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    q = graph.add(q_skip, atom_update)
    attention_details: list[dict[str, Any]] = []
    q = _atom_transformer(
        graph,
        q,
        c_skip,
        atom_dec_bias,
        atom_pad_mask,
        layer_prefix=f"{decoder}.atom_decoder.diffusion_transformer.layers",
        compute_dtype=trt.float32,
        debug_attention_detail_outputs=(
            attention_details if scheduler_fence_atom_attention else None
        ),
    )
    q = graph.layer_norm(q, f"{decoder}.atom_feat_to_atom_pos_update.0")
    r_update = graph.linear(q, f"{decoder}.atom_feat_to_atom_pos_update.1")
    r_update.name = "r_update"
    network.mark_output(r_update)
    if scheduler_fence_atom_attention:
        for layer, details in enumerate(attention_details):
            for name, tensor in details.items():
                if tuple(int(dim) for dim in tensor.shape) != attention_shapes[name]:
                    raise RuntimeError(f"unexpected decoder attention fence shape: {name}")
                tensor.name = f"decoder_fence_{layer}_{name}"
                network.mark_output(tensor)
    return r_update


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_diffusion_score_output_engine(
    checkpoint_path: Path,
    engine_path: Path,
    *,
    token_count: int = TOKEN_COUNT,
    atom_count: int = ATOM_COUNT,
    workspace_bytes: int = 16 << 30,
    verbose: bool = False,
    verify_checkpoint: bool = True,
    scheduler_fence_atom_attention: bool = True,
    avg_timing_iterations: int = 8,
) -> DiffusionScoreOutputBuildResult:
    """Build the direct score-model output engine."""

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
    define_diffusion_score_output_network(
        network,
        trt,
        weights,
        token_count=token_count,
        atom_count=atom_count,
        scheduler_fence_atom_attention=scheduler_fence_atom_attention,
    )
    config = builder.create_builder_config()
    config.avg_timing_iterations = avg_timing_iterations
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT failed to build Boltz-2 diffusion score output")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return DiffusionScoreOutputBuildResult(
        engine_path=str(engine_path),
        engine_sha256=_sha256(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        token_count=token_count,
        atom_count=atom_count,
        precision="fp32-upstream-exact",
    )
