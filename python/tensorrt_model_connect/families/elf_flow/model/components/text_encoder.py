# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ELF-flow-owned T5-small encoder engine builder.

Builds the original non-gated T5 encoder variant used by the official GitHub
ELF JAX checkpoint.

Engine I/O:
    Input:  input_ids [1, max_seq_len] int32
    Output: text_embeddings [1, max_seq_len, d_model] float32

Single forward pass (no cache), like vision encoder.
"""

from __future__ import annotations

import pickle
import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from .. import model as graph_ops

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...weights import WeightDict


def build_t5_encoder_engine(
    weights: WeightDict,
    *,
    d_model: int = 4096,
    num_heads: int = 64,
    d_kv: int = 64,
    d_ff: int = 10240,
    num_layers: int = 24,
    vocab_size: int = 250112,
    max_seq_len: int = 512,
    relative_attention_num_buckets: int = 32,
    relative_attention_max_distance: int = 128,
    eps: float = 1e-6,
    verbose: bool = False,
) -> bytes:
    """Build the GitHub ELF T5 encoder TRT engine plan.

    Args:
        weights: Weight dict with T5 encoder weights. Expected keys:
            - shared.weight: [vocab_size, d_model] embedding
            - encoder.block.{i}.layer.0.SelfAttention.q/k/v/o.weight
            - encoder.block.{i}.layer.0.layer_norm.weight
            - encoder.block.{i}.layer.1.DenseReluDense.wi/wo.weight
            - encoder.block.{i}.layer.1.layer_norm.weight
            - encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight
            - encoder.final_layer_norm.weight
        d_model: Hidden dimension.
        num_heads: Number of attention heads.
        d_kv: Key/value dimension per head.
        d_ff: Feed-forward inner dimension.
        num_layers: Number of encoder layers.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length.
        relative_attention_num_buckets: T5 relative position bias buckets.
        relative_attention_max_distance: Max distance for relative position.
        eps: RMSNorm epsilon.
        verbose: Enable TRT builder verbose logging.

    Returns:
        Serialized TRT engine plan bytes.
    """
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)

    # --- Inputs ---
    input_ids = network.add_input(
        "input_ids", trt.int32, (1, max_seq_len))
    # Attention mask: [1, max_seq_len] float32, 0.0 for valid, -1e9 for padding
    attention_mask_input = network.add_input(
        "attention_mask", trt.float32, (1, max_seq_len))

    # --- Constants ---
    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([eps], dtype=np.float32))

    # Embedding table [vocab_size, d_model]
    embed_table = graph_ops.add_constant(
        network, (vocab_size, d_model), weights["shared.weight"])

    # --- Embedding lookup ---
    # Flatten input_ids to [max_seq_len] for gather
    flatten_ids = network.add_shuffle(input_ids)
    flatten_ids.reshape_dims = (max_seq_len,)

    gather = network.add_gather(embed_table, flatten_ids.get_output(0), 0)
    hidden = gather.get_output(0)  # [max_seq_len, d_model]

    # --- Relative position bias ---
    # Precompute bucket indices [max_seq_len, max_seq_len]
    bucket_indices = graph_ops.make_t5_relative_position_bias(
        num_heads, max_seq_len,
        num_buckets=relative_attention_num_buckets,
        max_distance=relative_attention_max_distance,
    )

    # Reshape attention mask: [1, max_seq_len] -> [1, 1, max_seq_len]
    attn_mask_3d = network.add_shuffle(attention_mask_input)
    attn_mask_3d.reshape_dims = (1, 1, max_seq_len)

    # UMT5: each layer has its own relative_attention_bias (unlike T5 which shares layer 0's)
    # Precompute per-layer bias tables as constants
    per_layer_bias = []
    for layer_idx in range(num_layers):
        bias_key = f"encoder.block.{layer_idx}.layer.0.SelfAttention.relative_attention_bias.weight"
        if bias_key in weights:
            rel_bias_weight = weights[bias_key]
        else:
            # Fallback: use layer 0's bias (vanilla T5 behavior)
            rel_bias_weight = weights[
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]

        bias_table = np.zeros(
            (num_heads, max_seq_len, max_seq_len), dtype=np.float32)
        for q_pos in range(max_seq_len):
            for k_pos in range(max_seq_len):
                bucket = bucket_indices[q_pos, k_pos]
                for h in range(num_heads):
                    bias_table[h, q_pos, k_pos] = rel_bias_weight[bucket, h]

        rel_bias_const = graph_ops.add_constant(
            network, (num_heads, max_seq_len, max_seq_len), bias_table)

        # Combine position bias + attention mask
        position_bias_masked = network.add_elementwise(
            rel_bias_const, attn_mask_3d.get_output(0),
            trt.ElementWiseOperation.SUM)
        per_layer_bias.append(position_bias_masked.get_output(0))

    # --- Encoder layers ---
    attention_size = num_heads * d_kv

    for layer_idx in range(num_layers):
        prefix = f"encoder.block.{layer_idx}"

        # === Self-attention sub-layer ===
        # Pre-norm (RMSNorm)
        norm1_gamma = weights[f"{prefix}.layer.0.layer_norm.weight"]
        normed = graph_ops.add_rms_norm(
            network, hidden, d_model, norm1_gamma, eps_t)

        # Q, K, V projections
        # T5 uses no bias on Q/K/V/O projections
        w_q = weights[f"{prefix}.layer.0.SelfAttention.q.weight"]
        w_k = weights[f"{prefix}.layer.0.SelfAttention.k.weight"]
        w_v = weights[f"{prefix}.layer.0.SelfAttention.v.weight"]
        w_o = weights[f"{prefix}.layer.0.SelfAttention.o.weight"]

        # Self-attention with relative position bias
        q = graph_ops.add_matmul_rhs_constant(
            network, normed, d_model, attention_size, w_q)
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, d_model, attention_size, w_k)
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, d_model, attention_size, w_v)

        # T5 attention is intentionally unscaled; relative position bias and
        # padding mask are folded into the native IAttention additive mask.
        mask_4d = network.add_shuffle(per_layer_bias[layer_idx])
        mask_4d.reshape_dims = (1, num_heads, max_seq_len, max_seq_len)
        context_flat = graph_ops.add_attention_from_rows(
            network, q, k, v,
            num_heads=num_heads, head_dim=d_kv,
            q_seq=max_seq_len, kv_seq=max_seq_len,
            mask=mask_4d.get_output(0),
            scale=1.0)

        # Output projection
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, context_flat,
            attention_size, d_model, w_o)

        # Residual
        hidden = network.add_elementwise(
            hidden, attn_out,
            trt.ElementWiseOperation.SUM).get_output(0)

        # === FFN sub-layer ===
        # Pre-norm (RMSNorm)
        norm2_gamma = weights[f"{prefix}.layer.1.layer_norm.weight"]
        ffn_normed = graph_ops.add_rms_norm(
            network, hidden, d_model, norm2_gamma, eps_t)

        # Original T5 FFN: relu(wi(x)), then wo. ELF's published t5-small
        # JAX encoder checkpoint uses this non-gated variant.
        w_fc1 = weights[f"{prefix}.layer.1.DenseReluDense.wi.weight"]
        w_fc2 = weights[f"{prefix}.layer.1.DenseReluDense.wo.weight"]
        fc1 = graph_ops.add_matmul_rhs_constant(
            network, ffn_normed, d_model, d_ff, w_fc1)
        ffn_input = graph_ops.add_activation(network, fc1, "relu")

        ffn_out = graph_ops.add_matmul_rhs_constant(
            network, ffn_input, d_ff, d_model, w_fc2)

        # Residual
        hidden = network.add_elementwise(
            hidden, ffn_out,
            trt.ElementWiseOperation.SUM).get_output(0)

    # --- Final norm ---
    final_norm_gamma = weights["encoder.final_layer_norm.weight"]
    hidden = graph_ops.add_rms_norm(
        network, hidden, d_model, final_norm_gamma, eps_t)

    # --- Output ---
    # Reshape to [1, max_seq_len, d_model]
    out_reshape = network.add_shuffle(hidden)
    out_reshape.reshape_dims = (1, max_seq_len, d_model)
    out_tensor = out_reshape.get_output(0)
    cast_out = network.add_cast(out_tensor, trt.float32)
    out_final = cast_out.get_output(0)
    out_final.name = "text_embeddings"
    network.mark_output(out_final)

    # --- Build ---
    print(f"[elf-t5-encoder] Building TRT engine "
          f"(d_model={d_model}, layers={num_layers}, seq={max_seq_len}) ...",
          file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for T5 encoder")
    return bytes(plan)


def load_jax_t5_encoder_weights(
    checkpoint_path: str,
    *,
    precision: str = "fp32",
    num_layers: int = 6,
) -> WeightDict:
    """Load the official GitHub ELF JAX T5-small encoder checkpoint.

    The checkpoint published at
    ``embedded-language-flows/t5_small_encoder_jax/t5_small_encoder_jax.pkl``
    stores Flax Dense kernels already in ``[in, out]`` layout, which is the
    layout consumed by ``graph_ops.add_matmul_rhs_constant``.
    """
    from ...weights import WeightDict, _target_np_dtype

    with open(checkpoint_path, "rb") as f:
        payload = pickle.load(f)
    params = payload.get("params", payload)
    target_dtype = _target_np_dtype(precision)
    weights = WeightDict()

    weights["shared.weight"] = np.ascontiguousarray(
        params["shared"]["embedding"], dtype=target_dtype)
    enc = params["encoder"]
    for i in range(num_layers):
        src = enc[f"block_{i}"]
        prefix = f"encoder.block.{i}"
        attn = src["layer_0"]["SelfAttention"]
        for proj in ("q", "k", "v", "o"):
            weights[f"{prefix}.layer.0.SelfAttention.{proj}.weight"] = np.ascontiguousarray(
                attn[proj]["kernel"], dtype=target_dtype)
        if i == 0:
            weights[
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            ] = np.ascontiguousarray(
                attn["relative_attention_bias"]["rel_embedding"], dtype=np.float32)
        weights[f"{prefix}.layer.0.layer_norm.weight"] = np.ascontiguousarray(
            src["layer_0"]["layer_norm"]["weight"], dtype=np.float32)
        dense = src["layer_1"]["DenseReluDense"]
        weights[f"{prefix}.layer.1.DenseReluDense.wi.weight"] = np.ascontiguousarray(
            dense["wi"]["kernel"], dtype=target_dtype)
        weights[f"{prefix}.layer.1.DenseReluDense.wo.weight"] = np.ascontiguousarray(
            dense["wo"]["kernel"], dtype=target_dtype)
        weights[f"{prefix}.layer.1.layer_norm.weight"] = np.ascontiguousarray(
            src["layer_1"]["layer_norm"]["weight"], dtype=np.float32)

    weights["encoder.final_layer_norm.weight"] = np.ascontiguousarray(
        enc["final_layer_norm"]["weight"], dtype=np.float32)
    return weights
