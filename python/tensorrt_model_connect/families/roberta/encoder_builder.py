# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder-only engine builder (BERT-style).

Builds a TensorRT engine for an encoder-only transformer. Unlike the standard
decoder builder, this has:
  - No KV cache (processes entire sequence at once)
  - No causal attention mask (full bidirectional attention)
  - POST-norm architecture (residual + LayerNorm after attention/FFN)
  - Token type embeddings (segment A/B)
  - Output: hidden_states [seq_len, hidden_size]

Tensor names for the C++ runtime:
  Inputs:  input_ids [seq_len], attention_mask [seq_len]
  Outputs: hidden_states [seq_len, hidden_size]

token_type_ids is baked as constant zeros (single-segment inference).
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


def build_encoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_seq_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build a TRT engine plan for a BERT-style encoder.

    Args:
        config: Model architecture from config.json.
        weights: Loaded weight dict from BERT plugin.
        max_seq_length: Maximum sequence length the engine is compiled for.
        precision: TensorRT graph precision.
        verbose: Print TRT builder logs.

    Returns:
        Serialized engine plan bytes.
    """
    hidden = config.hidden_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = hidden // num_heads
    intermediate = config.intermediate_size
    eps = config.rms_norm_eps  # Actually layer_norm_eps for BERT
    type_vocab_size = config.raw.get("type_vocab_size", 2)
    requested_fp32_layers = frozenset(
        int(layer) for layer in config.raw.get("_fp32_layers", ()))
    invalid_fp32_layers = sorted(
        layer for layer in requested_fp32_layers
        if layer < 0 or layer >= num_layers)
    if invalid_fp32_layers:
        raise ValueError(
            "fp32_layers contains out-of-range indices: "
            f"{invalid_fp32_layers}")
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "bf16":
        work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
        requested_fp32_layers = frozenset()
    else:
        raise ValueError(f"Unsupported BERT precision: {precision}")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    # Disable TF32 to ensure full FP32 precision. TF32 uses 10-bit mantissa
    # which causes significant accuracy loss across 12+ encoder layers.
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    # Resolve GELU variant from config
    hidden_act = config.hidden_act or config.raw.get("activation", "") or "gelu"

    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------
    input_ids = network.add_input("input_ids", trt.int32, (max_seq_length,))
    attention_mask_input = network.add_input("attention_mask", trt.int32, (max_seq_length,))

    # token_type_ids: constant zeros (all segment-0) — the C++ encoder
    # pipeline doesn't provide this input, and inference is single-segment.
    tt_zeros = network.add_constant(
        (max_seq_length,), trt.Weights(np.zeros(max_seq_length, dtype=np.int32)))
    token_type_ids = tt_zeros.get_output(0)

    # -------------------------------------------------------------------
    # Shared constants (support embedding factorization: embedding_size != hidden)
    # -------------------------------------------------------------------
    embedding_size = weights["embedding"].shape[1]  # may differ from hidden (e.g. ALBERT 128 vs 768)
    embedding_table = graph_ops.add_constant(
        network, weights["embedding"].shape, weights["embedding"],
        dtype=work_np_dtype)
    position_embed_table = graph_ops.add_constant(
        network, weights["position_embedding"].shape,
        weights["position_embedding"], dtype=work_np_dtype)
    token_type_table = graph_ops.add_constant(
        network, (type_vocab_size, embedding_size),
        weights["token_type_embedding"], dtype=work_np_dtype)

    graph_ops.add_constant(
        network, (1, 1), np.array([eps], dtype=work_np_dtype),
        dtype=work_np_dtype)
    attn_scale_tensor = graph_ops.add_constant(
        network, (1, 1, 1),
        np.array([1.0 / np.sqrt(max(head_dim, 1))], dtype=work_np_dtype),
        dtype=work_np_dtype)

    # Build additive attention mask from attention_mask input:
    # attention_mask is [seq_len] with 1=real, 0=padding.
    # Convert to [1, 1, seq_len] additive mask: 0.0 for real, -1e10 for padding.
    mask_float = network.add_cast(attention_mask_input, work_trt_dtype)
    ones_mask = graph_ops.add_constant(
        network, (1,), np.array([1.0], dtype=work_np_dtype),
        dtype=work_np_dtype)
    mask_penalty = -1e4 if precision in {"fp16", "bf16"} else -1e10
    neg_large = graph_ops.add_constant(
        network, (1,), np.array([mask_penalty], dtype=work_np_dtype),
        dtype=work_np_dtype)
    inv_mask = network.add_elementwise(
        ones_mask, mask_float.get_output(0),
        trt.ElementWiseOperation.SUB)  # 0 for real, 1 for pad
    pad_penalty = network.add_elementwise(
        inv_mask.get_output(0), neg_large,
        trt.ElementWiseOperation.PROD)  # 0.0 for real, -1e10 for pad
    # Reshape to [1, 1, seq_len] for broadcasting: [num_heads, seq_len, seq_len]
    pad_mask_reshape = network.add_shuffle(pad_penalty.get_output(0))
    pad_mask_reshape.reshape_dims = (1, 1, max_seq_length)

    # Position indices: [0, 1, 2, ..., max_seq_length-1]
    position_indices = network.add_constant(
        (max_seq_length,),
        trt.Weights(np.arange(max_seq_length, dtype=np.int32)))

    # -------------------------------------------------------------------
    # Embedding: word + position + token_type + LayerNorm
    # -------------------------------------------------------------------
    word_embed = network.add_gather(embedding_table, input_ids, 0)
    pos_embed = network.add_gather(
        position_embed_table, position_indices.get_output(0), 0)
    tt_embed = network.add_gather(token_type_table, token_type_ids, 0)

    # Sum all three embedding types
    embed_sum1 = network.add_elementwise(
        word_embed.get_output(0), pos_embed.get_output(0),
        trt.ElementWiseOperation.SUM)
    embed_sum2 = network.add_elementwise(
        embed_sum1.get_output(0), tt_embed.get_output(0),
        trt.ElementWiseOperation.SUM)

    # Embedding LayerNorm (over embedding_size, which may differ from hidden)
    hidden_state = _add_seq_layer_norm(
        network, embed_sum2.get_output(0), embedding_size, max_seq_length,
        weights["embed_norm"], weights["embed_norm_beta"], eps,
        dtype=work_np_dtype)

    # Optional embedding projection: embedding_size -> hidden_size (ALBERT, FNet)
    if "embed_projection" in weights:
        hidden_state = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, embedding_size, hidden,
            weights["embed_projection"], dtype=work_np_dtype)
        if "embed_projection_bias" in weights:
            hidden_state = graph_ops.add_bias_sum(
                network, hidden_state, hidden,
                weights["embed_projection_bias"], dtype=work_np_dtype)

    # -------------------------------------------------------------------
    # Optional relative position bias (MPNet, T5-style)
    # -------------------------------------------------------------------
    rel_pos_bias_tensor = None
    if "relative_position_bias" in weights:
        # Pre-computed [num_heads, seq_len, seq_len] bias matrix
        rpb = weights["relative_position_bias"]
        rel_pos_bias_tensor = graph_ops.add_constant(
            network, rpb.shape, rpb, dtype=work_np_dtype)

    # -------------------------------------------------------------------
    # Encoder layers
    # -------------------------------------------------------------------
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        layer_is_fp32 = layer_idx in requested_fp32_layers
        layer_np_dtype = np.float32 if layer_is_fp32 else work_np_dtype
        layer_trt_dtype = trt.float32 if layer_is_fp32 else work_trt_dtype

        def _cast_layer_dtype(tensor: trt.ITensor | None) -> trt.ITensor | None:
            if tensor is None or tensor.dtype == layer_trt_dtype:
                return tensor
            return network.add_cast(tensor, layer_trt_dtype).get_output(0)

        hidden_state = _add_encoder_layer(
            network=network,
            hidden=_cast_layer_dtype(hidden_state),
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_length=max_seq_length,
            attn_scale_tensor=_cast_layer_dtype(attn_scale_tensor),
            attn_mask=_cast_layer_dtype(pad_mask_reshape.get_output(0)),
            rel_pos_bias=_cast_layer_dtype(rel_pos_bias_tensor),
            hidden_act=hidden_act,
            eps=eps,
            dtype=layer_np_dtype,
        )
        if hidden_state.dtype != work_trt_dtype:
            hidden_state = network.add_cast(
                hidden_state, work_trt_dtype).get_output(0)

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------
    # EncoderPipeline copies this public boundary into std::vector<float>.
    # Keep the external tensor FP32 even when the internal graph is reduced
    # precision so the runtime does not reinterpret FP16 bytes as floats.
    if hidden_state.dtype != trt.float32:
        hidden_state = network.add_cast(
            hidden_state, trt.float32).get_output(0)
    hidden_state.name = "hidden_states"
    network.mark_output(hidden_state)

    # -------------------------------------------------------------------
    # Build engine
    # -------------------------------------------------------------------
    if verbose:
        print(f"[trtmc build] Building BERT encoder TRT engine "
              f"({num_layers} layers, hidden={hidden}, "
              f"seq_len={max_seq_length}, precision={precision}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


def _add_seq_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    seq_length: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LayerNorm over sequence of vectors: [seq_len, hidden] -> [seq_len, hidden].

    Uses the shared TRT native normalization helper.
    """
    return graph_ops.add_layer_norm_native(
        network, inp, hidden_size, gamma, beta, eps, dtype=dtype)


def _add_encoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    head_dim: int,
    seq_length: int,
    attn_scale_tensor: trt.ITensor,
    attn_mask: trt.ITensor,
    rel_pos_bias: trt.ITensor | None = None,
    hidden_act: str = "gelu",
    eps: float,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add one BERT encoder layer with POST-norm.

    BERT architecture per layer:
        attn_out = MultiHeadSelfAttention(hidden)
        hidden = LayerNorm(hidden + attn_out)  # post-norm
        ffn_out = FFN(hidden)
        hidden = LayerNorm(hidden + ffn_out)   # post-norm
    """
    attention_size = num_heads * head_dim

    # --- Self-attention (no causal mask, bidirectional) ---
    # QKV projections
    q = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size,
        weights[f"{prefix}.w_q"], dtype=dtype)
    k = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size,
        weights[f"{prefix}.w_k"], dtype=dtype)
    v = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size,
        weights[f"{prefix}.w_v"], dtype=dtype)

    # QKV biases
    q = graph_ops.add_bias_sum(
        network, q, attention_size, weights[f"{prefix}.q_bias"], dtype=dtype)
    k = graph_ops.add_bias_sum(
        network, k, attention_size, weights[f"{prefix}.k_bias"], dtype=dtype)
    v = graph_ops.add_bias_sum(
        network, v, attention_size, weights[f"{prefix}.v_bias"], dtype=dtype)

    # Add relative position bias if present (MPNet/T5-style). IAttention
    # applies scaling internally through graph_ops.add_attention_from_rows, so
    # the mask contains only additive bias/masking terms.
    additive_mask = attn_mask
    if rel_pos_bias is not None:
        additive_mask = network.add_elementwise(
            rel_pos_bias, attn_mask, trt.ElementWiseOperation.SUM).get_output(0)

    mask_4d = network.add_shuffle(additive_mask)
    mask_4d.reshape_dims = (
        (1, num_heads, seq_length, seq_length)
        if rel_pos_bias is not None
        else (1, 1, 1, seq_length)
    )

    context_flat = graph_ops.add_attention_from_rows(
        network, q, k, v,
        num_heads=num_heads, head_dim=head_dim,
        q_seq=seq_length, kv_seq=seq_length,
        mask=mask_4d.get_output(0),
        tag=prefix + ".attn")

    # Output projection
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, context_flat,
        attention_size, hidden_size,
        weights[f"{prefix}.w_o"], dtype=dtype)
    attn_out = graph_ops.add_bias_sum(
        network, attn_out, hidden_size, weights[f"{prefix}.o_bias"],
        dtype=dtype)

    # POST-norm: LayerNorm(hidden + attn_out)
    residual1 = network.add_elementwise(
        hidden, attn_out, trt.ElementWiseOperation.SUM)
    normed1 = _add_seq_layer_norm(
        network, residual1.get_output(0), hidden_size, seq_length,
        weights[f"{prefix}.post_attn_norm"],
        weights[f"{prefix}.post_attn_norm_beta"], eps, dtype=dtype)

    # --- FFN: fc1 -> GELU -> fc2 ---
    fc1 = graph_ops.add_matmul_rhs_constant(
        network, normed1, hidden_size, intermediate_size,
        weights[f"{prefix}.w_fc1"], dtype=dtype)
    fc1 = graph_ops.add_bias_sum(network, fc1, intermediate_size,
                                  weights[f"{prefix}.fc1_bias"], dtype=dtype)
    activated = graph_ops.add_activation(
        network, fc1, hidden_act, dtype=dtype)
    fc2 = graph_ops.add_matmul_rhs_constant(
        network, activated, intermediate_size, hidden_size,
        weights[f"{prefix}.w_fc2"], dtype=dtype)
    fc2 = graph_ops.add_bias_sum(network, fc2, hidden_size,
                                  weights[f"{prefix}.fc2_bias"], dtype=dtype)

    # POST-norm: LayerNorm(normed1 + ffn_out)
    residual2 = network.add_elementwise(
        normed1, fc2, trt.ElementWiseOperation.SUM)
    normed2 = _add_seq_layer_norm(
        network, residual2.get_output(0), hidden_size, seq_length,
        weights[f"{prefix}.output_norm"],
        weights[f"{prefix}.output_norm_beta"], eps, dtype=dtype)

    return normed2
