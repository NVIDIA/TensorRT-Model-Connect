# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ConvBERT encoder builder — custom TRT engine builder for ConvBERT.

ConvBERT has a hybrid architecture where each layer uses BOTH:
  1. Standard multi-head self-attention (on half the head dimensions)
  2. Span-based dynamic convolution (on the other half)

The two outputs are concatenated and projected back to hidden_size.

Key implementation details:
  - SeparableConv1D is implemented as depthwise conv1d + pointwise conv1d (1x1)
  - Unfold (im2col) for sliding windows is implemented via slice+concat on 4D tensors
  - Dynamic conv kernels are generated per-position, softmaxed, then applied
  - All operations use static shapes (no dynamic axes)
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


def build_convbert_encoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_seq_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build a TRT engine plan for ConvBERT encoder."""
    hidden = config.hidden_size
    embedding_size = config.raw.get("embedding_size", hidden)
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    eps = config.rms_norm_eps  # layer_norm_eps
    type_vocab_size = config.raw.get("type_vocab_size", 2)
    hidden_act = config.hidden_act or "gelu"
    intermediate = config.intermediate_size
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Unsupported ConvBERT precision: {precision}")

    # ConvBERT specific
    new_num_heads = int(weights["_convbert_new_num_heads"][0])
    head_size = int(weights["_convbert_head_size"][0])
    all_head_size = int(weights["_convbert_all_head_size"][0])
    conv_kernel_size = int(weights["_convbert_conv_kernel_size"][0])

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    S = max_seq_length  # alias for brevity

    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------
    input_ids = network.add_input("input_ids", trt.int32, (S,))
    attention_mask_input = network.add_input("attention_mask", trt.int32, (S,))

    # token_type_ids: constant zeros (all segment-0) — the C++ encoder
    # pipeline doesn't provide this input, and inference is single-segment.
    tt_zeros = network.add_constant(
        (S,), trt.Weights(np.zeros(S, dtype=np.int32)))
    token_type_ids = tt_zeros.get_output(0)

    # -------------------------------------------------------------------
    # Shared constants
    # -------------------------------------------------------------------
    embedding_table = graph_ops.add_constant(
        network, (vocab, embedding_size), weights["embedding"],
        dtype=work_np_dtype)
    position_embed_table = graph_ops.add_constant(
        network, weights["position_embedding"].shape,
        weights["position_embedding"], dtype=work_np_dtype)
    token_type_table = graph_ops.add_constant(
        network, (type_vocab_size, embedding_size),
        weights["token_type_embedding"], dtype=work_np_dtype)

    # Additive attention mask: [1, 1, S]
    mask_float = network.add_cast(attention_mask_input, work_trt_dtype)
    ones_mask = graph_ops.add_constant(
        network, (1,), np.array([1.0], dtype=work_np_dtype),
        dtype=work_np_dtype)
    mask_penalty = -1e4 if precision == "fp16" else -1e10
    neg_large = graph_ops.add_constant(
        network, (1,), np.array([mask_penalty], dtype=work_np_dtype),
        dtype=work_np_dtype)
    inv_mask = network.add_elementwise(
        ones_mask, mask_float.get_output(0), trt.ElementWiseOperation.SUB)
    pad_penalty = network.add_elementwise(
        inv_mask.get_output(0), neg_large, trt.ElementWiseOperation.PROD)
    pad_mask_reshape = network.add_shuffle(pad_penalty.get_output(0))
    pad_mask_reshape.reshape_dims = (1, 1, S)
    attn_mask = pad_mask_reshape.get_output(0)

    # Position indices
    position_indices = graph_ops.add_constant(
        network, (S,), np.arange(S, dtype=np.int32).astype(work_np_dtype),
        dtype=work_np_dtype)
    pos_int = network.add_cast(position_indices, trt.int32)

    # -------------------------------------------------------------------
    # Embedding: word + position + token_type + LayerNorm
    # -------------------------------------------------------------------
    word_embed = network.add_gather(embedding_table, input_ids, 0)
    pos_embed = network.add_gather(position_embed_table, pos_int.get_output(0), 0)
    tt_embed = network.add_gather(token_type_table, token_type_ids, 0)

    embed_sum1 = network.add_elementwise(
        word_embed.get_output(0), pos_embed.get_output(0), trt.ElementWiseOperation.SUM)
    embed_sum2 = network.add_elementwise(
        embed_sum1.get_output(0), tt_embed.get_output(0), trt.ElementWiseOperation.SUM)

    hidden_state = _add_seq_layer_norm(
        network, embed_sum2.get_output(0), embedding_size, S,
        weights["embed_norm"], weights["embed_norm_beta"], eps,
        dtype=work_np_dtype)

    # -------------------------------------------------------------------
    # Encoder layers
    # -------------------------------------------------------------------
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        hidden_state = _add_convbert_layer(
            network=network,
            hidden=hidden_state,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            intermediate_size=intermediate,
            new_num_heads=new_num_heads,
            head_size=head_size,
            all_head_size=all_head_size,
            conv_kernel_size=conv_kernel_size,
            seq_length=S,
            attn_mask=attn_mask,
            hidden_act=hidden_act,
            eps=eps,
            dtype=work_np_dtype,
        )

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------
    public_output = hidden_state
    if public_output.dtype != trt.float32:
        public_output = network.add_cast(
            public_output, trt.float32).get_output(0)
    public_output.name = "hidden_states"
    network.mark_output(public_output)

    # -------------------------------------------------------------------
    # Build engine
    # -------------------------------------------------------------------
    if verbose:
        print(f"[trtmc build] Building ConvBERT encoder TRT engine "
              f"({num_layers} layers, hidden={hidden}, "
              f"seq_len={S}, conv_kernel={conv_kernel_size}, "
              f"precision={precision}) ...", file=sys.stderr)

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
    """LayerNorm over [seq_len, hidden] using TRT native normalization."""
    return graph_ops.add_layer_norm_native(
        network, inp, hidden_size, gamma, beta, eps, dtype=dtype)


def _add_separable_conv1d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    all_head_size: int,
    conv_kernel_size: int,
    seq_length: int,
    dw_weight: np.ndarray,
    pw_weight: np.ndarray,
    bias: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """SeparableConv1D: depthwise conv1d + pointwise conv1d + bias.

    Input: [seq_len, hidden_size] (our 2D layout)
    Output: [seq_len, all_head_size]
    """
    pad = conv_kernel_size // 2

    # Reshape: [seq, hidden] -> [1, hidden, seq, 1] for TRT Conv2d
    shuf_in = network.add_shuffle(inp)
    shuf_in.first_transpose = trt.Permutation([1, 0])  # [hidden, seq]
    shuf_in.reshape_dims = (1, hidden_size, seq_length, 1)

    # Depthwise convolution: kernel [hidden, 1, kernel_size, 1]
    dw_w_4d = dw_weight.reshape(hidden_size, 1, conv_kernel_size, 1)
    dw_trt = trt.Weights(np.ascontiguousarray(dw_w_4d, dtype=dtype))
    dw_conv = network.add_convolution_nd(
        shuf_in.get_output(0),
        num_output_maps=hidden_size,
        kernel_shape=(conv_kernel_size, 1),
        kernel=dw_trt,
    )
    dw_conv.padding_nd = (pad, 0)
    dw_conv.num_groups = hidden_size

    # Pointwise conv: kernel [all_head_size, hidden, 1, 1]
    pw_w_4d = pw_weight.reshape(all_head_size, hidden_size, 1, 1)
    pw_trt = trt.Weights(np.ascontiguousarray(pw_w_4d, dtype=dtype))
    pw_conv = network.add_convolution_nd(
        dw_conv.get_output(0),
        num_output_maps=all_head_size,
        kernel_shape=(1, 1),
        kernel=pw_trt,
    )

    # Add bias: [1, all_head_size, 1, 1]
    bias_4d = graph_ops.add_constant(
        network, (1, all_head_size, 1, 1),
        bias.reshape(1, all_head_size, 1, 1), dtype=dtype)
    biased = network.add_elementwise(
        pw_conv.get_output(0), bias_4d, trt.ElementWiseOperation.SUM)

    # Reshape: [1, all_head_size, seq, 1] -> [seq, all_head_size]
    squeeze = network.add_shuffle(biased.get_output(0))
    squeeze.reshape_dims = (all_head_size, seq_length)
    squeeze.second_transpose = trt.Permutation([1, 0])

    return squeeze.get_output(0)


def _add_unfold(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    channels: int,
    seq_length: int,
    kernel_size: int,
) -> trt.ITensor:
    """Unfold (im2col) for 1D signal.

    Input: [channels, seq_length] (channel-first)
    Output: [channels * kernel_size, seq_length]

    For each position p, gathers values at positions [p-pad, ..., p+pad]
    for each channel, with zero-padding at boundaries.

    Uses TRT slice layers with zero-fill mode for out-of-bounds positions.
    """
    pad = kernel_size // 2
    shifts = []

    # Expand to 4D: [1, channels, seq_length, 1] for TRT padding (requires 4D)
    expand = network.add_shuffle(inp)
    expand.reshape_dims = (1, channels, seq_length, 1)
    inp_4d = expand.get_output(0)

    for k in range(kernel_size):
        offset = k - pad  # shift amount: -pad to +pad

        if offset == 0:
            # No shift needed
            identity = network.add_shuffle(inp_4d)
            identity.reshape_dims = (channels, seq_length)
            shifts.append(identity.get_output(0))
        elif offset < 0:
            # Pad left (prepend zeros along H=seq dim), slice from start
            abs_off = -offset
            # For 4D [N,C,H,W], padding is 2D: (H_pad, W_pad)
            pad_layer = network.add_padding_nd(
                inp_4d,
                pre_padding=(abs_off, 0),
                post_padding=(0, 0),
            )
            # Slice from start: [1, channels, seq_length, 1]
            sl = network.add_slice(
                pad_layer.get_output(0),
                start=(0, 0, 0, 0),
                shape=(1, channels, seq_length, 1),
                stride=(1, 1, 1, 1),
            )
            reshape = network.add_shuffle(sl.get_output(0))
            reshape.reshape_dims = (channels, seq_length)
            shifts.append(reshape.get_output(0))
        else:
            # Pad right (append zeros along H=seq dim), slice from offset
            pad_layer = network.add_padding_nd(
                inp_4d,
                pre_padding=(0, 0),
                post_padding=(offset, 0),
            )
            # Slice from offset: [1, channels, seq_length, 1]
            sl = network.add_slice(
                pad_layer.get_output(0),
                start=(0, 0, offset, 0),
                shape=(1, channels, seq_length, 1),
                stride=(1, 1, 1, 1),
            )
            reshape = network.add_shuffle(sl.get_output(0))
            reshape.reshape_dims = (channels, seq_length)
            shifts.append(reshape.get_output(0))

    # Concatenate along channel dim: [channels * kernel_size, seq_length]
    if len(shifts) == 1:
        return shifts[0]
    cat = network.add_concatenation(shifts)
    cat.axis = 0  # channel dim
    return cat.get_output(0)


def _add_convbert_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    intermediate_size: int,
    new_num_heads: int,
    head_size: int,
    all_head_size: int,
    conv_kernel_size: int,
    seq_length: int,
    attn_mask: trt.ITensor,
    hidden_act: str,
    eps: float,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add one ConvBERT encoder layer with mixed attention + dynamic convolution."""
    S = seq_length

    # === Branch 1: Standard multi-head self-attention ===
    q = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, all_head_size,
        weights[f"{prefix}.w_q"], dtype=dtype)
    k = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, all_head_size,
        weights[f"{prefix}.w_k"], dtype=dtype)
    v = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, all_head_size,
        weights[f"{prefix}.w_v"], dtype=dtype)

    q = graph_ops.add_bias_sum(
        network, q, all_head_size, weights[f"{prefix}.q_bias"], dtype=dtype)
    k = graph_ops.add_bias_sum(
        network, k, all_head_size, weights[f"{prefix}.k_bias"], dtype=dtype)
    v = graph_ops.add_bias_sum(
        network, v, all_head_size, weights[f"{prefix}.v_bias"], dtype=dtype)

    mixed_query = q  # save for conv branch

    # Key padding mask broadcasts across every query row.
    mask_row = network.add_shuffle(attn_mask)
    mask_row.reshape_dims = (1, S)
    zero_col = graph_ops.add_constant(
        network, (S, 1), np.zeros((S, 1), dtype=dtype), dtype=dtype)
    mask_2d = network.add_elementwise(
        zero_col, mask_row.get_output(0), trt.ElementWiseOperation.SUM)
    mask_4d = graph_ops.add_2d_mask_to_4d(network, mask_2d.get_output(0))

    context = graph_ops.add_attention_from_rows(
        network, q, k, v,
        num_heads=new_num_heads, head_dim=head_size,
        q_seq=S, kv_seq=S, mask=mask_4d)

    context_perm = network.add_shuffle(context)
    context_perm.reshape_dims = (S, new_num_heads, head_size)

    # === Branch 2: Span-based dynamic convolution ===

    # SeparableConv1D on hidden_states
    key_conv_attn = _add_separable_conv1d(
        network, hidden,
        hidden_size, all_head_size, conv_kernel_size, S,
        weights[f"{prefix}.sep_conv_dw"],
        weights[f"{prefix}.sep_conv_pw"],
        weights[f"{prefix}.sep_conv_bias"],
        dtype=dtype,
    )

    # conv_attn = key_conv_attn * query (element-wise)
    conv_attn = network.add_elementwise(
        key_conv_attn, mixed_query, trt.ElementWiseOperation.PROD)

    # conv_kernel = linear(conv_attn) -> [seq, num_heads * kernel_size]
    conv_kernel = graph_ops.add_matmul_rhs_constant(
        network, conv_attn.get_output(0), all_head_size,
        new_num_heads * conv_kernel_size,
        weights[f"{prefix}.conv_kernel_w"], dtype=dtype)
    conv_kernel = graph_ops.add_bias_sum(
        network, conv_kernel, new_num_heads * conv_kernel_size,
        weights[f"{prefix}.conv_kernel_bias"], dtype=dtype)

    # Reshape to [seq * num_heads, kernel_size, 1], softmax on kernel_size dim
    ck_reshape = network.add_shuffle(conv_kernel)
    ck_reshape.reshape_dims = (S * new_num_heads, conv_kernel_size, 1)
    ck_softmax = network.add_softmax(ck_reshape.get_output(0))
    ck_softmax.axes = 1 << 1

    # conv_out = linear(hidden) -> [seq, all_head_size]
    conv_out = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, all_head_size,
        weights[f"{prefix}.conv_out_w"], dtype=dtype)
    conv_out = graph_ops.add_bias_sum(
        network, conv_out, all_head_size,
        weights[f"{prefix}.conv_out_bias"], dtype=dtype)

    # Transpose to [all_head_size, seq] for unfold
    conv_out_t = network.add_shuffle(conv_out)
    conv_out_t.first_transpose = trt.Permutation([1, 0])

    # Unfold: [kernel_size * all_head_size, seq] (kernel-major ordering)
    unfolded = _add_unfold(
        network, conv_out_t.get_output(0),
        all_head_size, S, conv_kernel_size)

    # Rearrange from kernel-major [K*C, seq] to channel-major [C*K, seq]
    # Reshape to [K, C, seq], permute to [C, K, seq], reshape to [C*K, seq]
    unf_reorder = network.add_shuffle(unfolded)
    unf_reorder.reshape_dims = (conv_kernel_size, all_head_size, S)
    unf_reorder.second_transpose = trt.Permutation([1, 0, 2])
    # Now [all_head_size, kernel_size, seq]

    # Transpose to [seq, all_head_size, kernel_size]
    unf_to_seq_first = network.add_shuffle(unf_reorder.get_output(0))
    unf_to_seq_first.first_transpose = trt.Permutation([2, 0, 1])
    # [seq, all_head_size, kernel_size]

    # Reshape to [seq * num_heads, head_size, kernel_size]
    unf_reshape = network.add_shuffle(unf_to_seq_first.get_output(0))
    unf_reshape.reshape_dims = (S * new_num_heads, head_size, conv_kernel_size)

    # Matmul: [S*H, head_size, K] @ [S*H, K, 1] -> [S*H, head_size, 1]
    conv_result = network.add_matrix_multiply(
        unf_reshape.get_output(0), trt.MatrixOperation.NONE,
        ck_softmax.get_output(0), trt.MatrixOperation.NONE)

    # Reshape to [S, new_num_heads, head_size]
    conv_reshaped = network.add_shuffle(conv_result.get_output(0))
    conv_reshaped.reshape_dims = (S, new_num_heads, head_size)

    # === Concatenate: [S, 2*num_heads, head_size] ===
    cat = network.add_concatenation([context_perm.get_output(0), conv_reshaped.get_output(0)])
    cat.axis = 1

    # Flatten: [S, hidden_size]
    cat_flat = network.add_shuffle(cat.get_output(0))
    cat_flat.reshape_dims = (S, 2 * new_num_heads * head_size)

    # === Output projection ===
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, cat_flat.get_output(0), hidden_size, hidden_size,
        weights[f"{prefix}.w_o"], dtype=dtype)
    attn_out = graph_ops.add_bias_sum(
        network, attn_out, hidden_size, weights[f"{prefix}.o_bias"],
        dtype=dtype)

    # POST-norm
    residual1 = network.add_elementwise(
        hidden, attn_out, trt.ElementWiseOperation.SUM)
    normed1 = _add_seq_layer_norm(
        network, residual1.get_output(0), hidden_size, S,
        weights[f"{prefix}.post_attn_norm"],
        weights[f"{prefix}.post_attn_norm_beta"], eps, dtype=dtype)

    # === FFN ===
    fc1 = graph_ops.add_matmul_rhs_constant(
        network, normed1, hidden_size, intermediate_size,
        weights[f"{prefix}.w_fc1"], dtype=dtype)
    fc1 = graph_ops.add_bias_sum(
        network, fc1, intermediate_size, weights[f"{prefix}.fc1_bias"],
        dtype=dtype)
    activated = graph_ops.add_activation(network, fc1, hidden_act)
    fc2 = graph_ops.add_matmul_rhs_constant(
        network, activated, intermediate_size, hidden_size,
        weights[f"{prefix}.w_fc2"], dtype=dtype)
    fc2 = graph_ops.add_bias_sum(
        network, fc2, hidden_size, weights[f"{prefix}.fc2_bias"],
        dtype=dtype)

    # POST-norm
    residual2 = network.add_elementwise(
        normed1, fc2, trt.ElementWiseOperation.SUM)
    normed2 = _add_seq_layer_norm(
        network, residual2.get_output(0), hidden_size, S,
        weights[f"{prefix}.output_norm"],
        weights[f"{prefix}.output_norm_beta"], eps, dtype=dtype)

    return normed2
