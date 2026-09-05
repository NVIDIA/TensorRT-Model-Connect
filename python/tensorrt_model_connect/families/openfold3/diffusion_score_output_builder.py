# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixed-FP16 TensorRT graph for the OpenFold3 diffusion atom decoder."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat

from .atom_attention_builder import atom_transformer, padded_atom_count
from .checkpoint import load_weight_prefixes
from .graph_ops import Graph, diffusion_compute_dtype


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
    network,
    trt,
    weights: dict[str, np.ndarray],
    *,
    token_count: int,
    atom_count: int,
    precision: str = "fp16",
):
    """Define Algorithm 20 lines 14-17, including denoising coefficients."""

    padded_atoms = padded_atom_count(atom_count)
    lowp = diffusion_compute_dtype(trt, precision)
    graph = Graph(network, trt, weights, precision=precision)
    a = network.add_input("a", trt.float32, (1, token_count, 768))
    q = network.add_input("atom_representation", trt.float32, (1, padded_atoms, 128))
    condition = network.add_input("atom_conditioning", trt.float32, (1, padded_atoms, 128))
    pair = network.add_input(
        "atom_pair_conditioning",
        trt.float32,
        (1, padded_atoms // 32, 32, 128, 16),
    )
    atom_to_token = network.add_input("atom_to_token_index", trt.int32, (1, padded_atoms))
    atom_mask = network.add_input("atom_mask", trt.float32, (1, padded_atoms))
    noisy = network.add_input("noisy_positions", trt.float32, (1, padded_atoms, 3))
    noise_level = network.add_input("noise_level", trt.float32, (1,))

    a = graph.layer_norm(a, "diffusion_module.layer_norm_a")
    token_update = graph.linear(graph.cast(a, lowp), "diffusion_module.atom_attn_dec.linear_q_in")
    atom_map = graph.one_hot(atom_to_token, token_count, trt.float32)
    atom_map = graph.mul(
        atom_map,
        graph.reshape(atom_mask, (1, padded_atoms, 1)),
    )
    atom_update = graph.network.add_matrix_multiply(
        atom_map,
        trt.MatrixOperation.NONE,
        graph.cast(token_update, trt.float32),
        trt.MatrixOperation.NONE,
    ).get_output(0)
    q = graph.add(q, graph.cast(atom_update, q.dtype))
    q = atom_transformer(
        graph,
        q,
        condition,
        pair,
        atom_mask,
        "diffusion_module.atom_attn_dec.atom_transformer",
        atom_count=atom_count,
        padded_atoms=padded_atoms,
        compute_dtype=lowp,
    )
    q = graph.layer_norm(q, "diffusion_module.atom_attn_dec.layer_norm")
    update = graph.linear(graph.cast(q, lowp), "diffusion_module.atom_attn_dec.linear_q_out")

    t_squared = graph.mul(noise_level, noise_level)
    denominator = graph.add(t_squared, graph.scalar_like(256.0, t_squared))
    skip_scale = graph.div(graph.scalar_like(256.0, denominator), denominator)
    output_scale = graph.div(
        graph.mul(graph.scalar_like(16.0, noise_level), noise_level),
        graph.unary(denominator, trt.UnaryOperation.SQRT),
    )
    denoised = graph.add(
        graph.mul(noisy, graph.reshape(skip_scale, (1, 1, 1))),
        graph.mul(
            graph.cast(update, noisy.dtype),
            graph.reshape(output_scale, (1, 1, 1)),
        ),
    )
    denoised = graph.mul(denoised, graph.reshape(atom_mask, (1, padded_atoms, 1)))
    denoised.name = "denoised_positions"
    network.mark_output(denoised)
    return denoised


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
    token_count: int,
    atom_count: int,
    workspace_bytes: int = 16 << 30,
    verbose: bool = False,
    verify_checkpoint: bool = True,
    precision: str = "fp16",
) -> DiffusionScoreOutputBuildResult:
    weights = load_weight_prefixes(
        checkpoint_path,
        ("diffusion_module.layer_norm_a.", "diffusion_module.atom_attn_dec."),
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
        raise RuntimeError("TensorRT failed to build OpenFold3 diffusion atom output")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return DiffusionScoreOutputBuildResult(
        engine_path=str(engine_path),
        engine_sha256=_sha256(engine_path),
        engine_size_bytes=engine_path.stat().st_size,
        build_seconds=build_seconds,
        token_count=token_count,
        atom_count=atom_count,
        precision=f"{precision}-mixed",
    )
