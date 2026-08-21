# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FNet encoder builder — TRT engine with 2D DFT replacing self-attention.

FNet replaces self-attention with a 2D Discrete Fourier Transform:
  - DFT2D(X) = DFT_seq @ X @ DFT_hidden (taking real part only)
  - Implemented via pre-computed cosine/sine matrices as constants
  - POST-norm: residual + LayerNorm after DFT and FFN

Tensor names for the C++ runtime:
  Inputs:  input_ids [seq_len], attention_mask [seq_len]
  Outputs: hidden_states [seq_len, hidden_size]
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from .config import ModelConfig

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def _compute_dft_matrices(n: int):
    """Pre-compute real and imaginary DFT matrices for dimension n.

    Returns (cos_mat, sin_mat) each of shape [n, n].
    DFT definition: X_k = sum_j x_j * exp(-2pi*i*j*k/n)
    real part: cos(2*pi*j*k/n), imag part: -sin(2*pi*j*k/n)
    """
    j = np.arange(n, dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    jk = np.outer(k, j)  # [n, n]
    angle = 2.0 * np.pi * jk / n
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


def _add_seq_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    dtype: np.dtype,
) -> trt.ITensor:
    """LayerNorm over [seq_len, hidden] using TRT native normalization."""
    return graph_ops.add_layer_norm_native(
        network, inp, hidden_size, gamma, beta, eps, dtype=dtype)


def build_fnet_encoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_seq_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build a TRT engine plan for FNet encoder with 2D DFT."""
    hidden = config.hidden_size
    num_layers = config.num_hidden_layers
    intermediate = config.intermediate_size
    eps = config.rms_norm_eps
    type_vocab_size = config.raw.get("type_vocab_size", 4)
    hidden_act = config.hidden_act or config.raw.get("activation", "") or "gelu_new"

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    S = max_seq_length
    H = hidden
    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"FNet supports fp32 or fp16 precision, got {precision!r}")
    work_np_dtype = np.float16 if precision == "fp16" else np.float32

    # Inputs
    input_ids = network.add_input("input_ids", trt.int32, (S,))
    network.add_input("attention_mask", trt.int32, (S,))

    # token_type_ids: constant zeros
    tt_zeros = network.add_constant(
        (S,), trt.Weights(np.zeros(S, dtype=np.int32)))
    token_type_ids = tt_zeros.get_output(0)

    # Embedding tables (may use embedding_size != hidden for factorized embeddings)
    embedding_size = weights["embedding"].shape[1]
    embedding_table = graph_ops.add_constant(
        network, weights["embedding"].shape, weights["embedding"],
        dtype=work_np_dtype)
    position_embed_table = graph_ops.add_constant(
        network, weights["position_embedding"].shape, weights["position_embedding"],
        dtype=work_np_dtype)
    token_type_table = graph_ops.add_constant(
        network, (type_vocab_size, embedding_size), weights["token_type_embedding"],
        dtype=work_np_dtype)

    # Position indices
    position_indices = graph_ops.add_constant(
        network, (S,), np.arange(S, dtype=np.int32).astype(np.float32))
    pos_int = network.add_cast(position_indices, trt.int32)

    # Embedding: word + position + token_type
    word_embed = network.add_gather(embedding_table, input_ids, 0)
    pos_embed = network.add_gather(position_embed_table, pos_int.get_output(0), 0)
    tt_embed = network.add_gather(token_type_table, token_type_ids, 0)

    embed_sum1 = network.add_elementwise(
        word_embed.get_output(0), pos_embed.get_output(0),
        trt.ElementWiseOperation.SUM)
    embed_sum2 = network.add_elementwise(
        embed_sum1.get_output(0), tt_embed.get_output(0),
        trt.ElementWiseOperation.SUM)

    # Embedding LayerNorm (over embedding_size)
    hidden_state = _add_seq_layer_norm(
        network, embed_sum2.get_output(0), embedding_size,
        weights["embed_norm"], weights["embed_norm_beta"], eps,
        work_np_dtype)

    # Optional embedding projection: embedding_size -> hidden_size
    if "embed_projection" in weights:
        hidden_state = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, embedding_size, hidden,
            weights["embed_projection"], dtype=work_np_dtype)
        if "embed_projection_bias" in weights:
            hidden_state = graph_ops.add_bias_sum(
                network, hidden_state, hidden,
                weights["embed_projection_bias"], dtype=work_np_dtype)

    # Pre-compute 2D DFT matrices as constants
    # real(DFT2D(X)) = cos_S @ X @ cos_H - sin_S @ X @ sin_H
    cos_s, sin_s = _compute_dft_matrices(S)
    cos_h, sin_h = _compute_dft_matrices(H)

    cos_s_const = graph_ops.add_constant(
        network, (S, S), cos_s, dtype=work_np_dtype)
    sin_s_const = graph_ops.add_constant(
        network, (S, S), sin_s, dtype=work_np_dtype)
    cos_h_const = graph_ops.add_constant(
        network, (H, H), cos_h, dtype=work_np_dtype)
    sin_h_const = graph_ops.add_constant(
        network, (H, H), sin_h, dtype=work_np_dtype)

    # Encoder layers
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        # --- 2D DFT (replaces self-attention) ---
        # real(DFT2D(X)) = cos_S @ X @ cos_H - sin_S @ X @ sin_H
        # Term 1: cos_S @ X @ cos_H
        cx = network.add_matrix_multiply(
            cos_s_const, trt.MatrixOperation.NONE,
            hidden_state, trt.MatrixOperation.NONE)
        cxch = network.add_matrix_multiply(
            cx.get_output(0), trt.MatrixOperation.NONE,
            cos_h_const, trt.MatrixOperation.NONE)

        # Term 2: sin_S @ X @ sin_H
        sx = network.add_matrix_multiply(
            sin_s_const, trt.MatrixOperation.NONE,
            hidden_state, trt.MatrixOperation.NONE)
        sxsh = network.add_matrix_multiply(
            sx.get_output(0), trt.MatrixOperation.NONE,
            sin_h_const, trt.MatrixOperation.NONE)

        # real = term1 - term2
        dft_out = network.add_elementwise(
            cxch.get_output(0), sxsh.get_output(0),
            trt.ElementWiseOperation.SUB).get_output(0)

        # POST-norm: residual + LayerNorm after DFT
        residual1 = network.add_elementwise(
            hidden_state, dft_out, trt.ElementWiseOperation.SUM)
        normed1 = _add_seq_layer_norm(
            network, residual1.get_output(0), hidden,
            weights[f"{prefix}.post_attn_norm"],
            weights[f"{prefix}.post_attn_norm_beta"], eps, work_np_dtype)

        # --- FFN ---
        fc1 = graph_ops.add_matmul_rhs_constant(
            network, normed1, hidden, intermediate,
            weights[f"{prefix}.w_fc1"], dtype=work_np_dtype)
        fc1 = graph_ops.add_bias_sum(
            network, fc1, intermediate, weights[f"{prefix}.fc1_bias"],
            dtype=work_np_dtype)
        activated = graph_ops.add_activation(
            network, fc1, hidden_act, dtype=work_np_dtype)
        fc2 = graph_ops.add_matmul_rhs_constant(
            network, activated, intermediate, hidden,
            weights[f"{prefix}.w_fc2"], dtype=work_np_dtype)
        fc2 = graph_ops.add_bias_sum(
            network, fc2, hidden, weights[f"{prefix}.fc2_bias"],
            dtype=work_np_dtype)

        # POST-norm: residual + LayerNorm after FFN
        residual2 = network.add_elementwise(
            normed1, fc2, trt.ElementWiseOperation.SUM)
        hidden_state = _add_seq_layer_norm(
            network, residual2.get_output(0), hidden,
            weights[f"{prefix}.output_norm"],
            weights[f"{prefix}.output_norm_beta"], eps, work_np_dtype)

    # Output
    if precision == "fp16":
        hidden_state = network.add_cast(hidden_state, trt.float32).get_output(0)
    hidden_state.name = "hidden_states"
    network.mark_output(hidden_state)

    if verbose:
        print(f"[trtmc build] Building FNet encoder TRT engine "
              f"({num_layers} layers, hidden={hidden}, "
              f"seq_len={S}) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)
