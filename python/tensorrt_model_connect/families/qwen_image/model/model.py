"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations


import numpy as np
from tensorrt_model_connect import trt_compat
from typing import TYPE_CHECKING
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Union
# Graph Ops


trt = trt_compat.get_trt()


def _cast_back_to_trt_dtype(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target_dtype: trt.DataType,
) -> trt.ITensor:
    """Cast a tensor back to the original TRT runtime dtype after FP32 compute."""
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add a constant tensor in the given *dtype* (default float32)."""
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    layer = network.add_constant(shape, weights)
    return layer.get_output(0)


def add_matmul_rhs_constant(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    rhs_weights: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Matrix multiply: lhs @ rhs_constant.  rhs is [lhs_width, rhs_width]."""
    rank = len(tuple(lhs.shape))
    rhs_shape = (lhs_width, rhs_width) if rank <= 2 else (1,) * (rank - 2) + (lhs_width, rhs_width)
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(rhs_weights).reshape(rhs_shape),
        dtype=dtype,
    )
    rhs = _cast_back_to_trt_dtype(network, rhs, lhs.dtype)
    mm = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    )
    return _cast_back_to_trt_dtype(network, mm.get_output(0), lhs.dtype)


def add_bias_sum(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    bias: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Element-wise add a bias broadcast over all non-feature axes."""
    rank = len(tuple(inp.shape))
    bias_shape = (width,) if rank <= 1 else (1,) * (rank - 1) + (width,)
    bias_t = add_constant(network, bias_shape, np.asarray(bias).reshape(bias_shape), dtype=dtype)
    bias_t = _cast_back_to_trt_dtype(network, bias_t, inp.dtype)
    s = network.add_elementwise(inp, bias_t, trt.ElementWiseOperation.SUM)
    return _cast_back_to_trt_dtype(network, s.get_output(0), inp.dtype)


def add_rms_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """RMSNorm: gamma * (x / sqrt(mean(x^2) + eps)).

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.

    TRT's native normalization API implements mean-centered LayerNorm, not
    RMSNorm, so this remains a manual shared implementation.
    """
    need_cast = dtype != np.float32
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(mean.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(inp, recip.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
    result = scaled.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def add_rms_norm_per_head_batched(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
    sequence_length: int | None = None,
) -> trt.ITensor:
    """Per-head RMSNorm for ``[B, S, num_heads * head_dim]`` tensors.

    Batched companion to :func:`add_rms_norm_per_head` used by diffusion
    builders whose leading dim is a dynamic batch (``-1``). ``gamma`` may be
    ``[num_heads * head_dim]`` or ``[head_dim]`` broadcast to heads.
    """
    seq_dim = -1 if sequence_length is None else sequence_length
    need_cast = dtype != np.float32
    output_dtype = inp.dtype

    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (-1, seq_dim, num_heads, head_dim)
    reshaped = reshape_in.get_output(0)
    if need_cast:
        reshaped = network.add_cast(reshaped, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)

    eps_4d = network.add_shuffle(eps_tensor)
    eps_4d.reshape_dims = (1, 1, 1, 1)
    sq = network.add_elementwise(reshaped, reshaped, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 3, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_4d.get_output(0), trt.ElementWiseOperation.SUM
    )
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        reshaped, recip.get_output(0), trt.ElementWiseOperation.PROD
    )

    gamma_arr = np.asarray(gamma, dtype=np.float32)
    if gamma_arr.size == head_dim:
        gamma_shape = (1, 1, 1, head_dim)
        gamma_arr = gamma_arr.reshape(gamma_shape)
    else:
        gamma_shape = (1, 1, num_heads, head_dim)
        gamma_arr = gamma_arr.reshape(gamma_shape)
    gamma_t = add_constant(network, gamma_shape, gamma_arr, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )

    result = scaled.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    reshape_out = network.add_shuffle(result)
    reshape_out.reshape_dims = (-1, seq_dim, num_heads * head_dim)
    return reshape_out.get_output(0)


def add_gelu_new(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """GELU (tanh approximation): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).

    Constants are cast to ``inp.dtype`` so the elementwise ops are valid in
    a STRONGLY_TYPED network when ``inp`` is bf16 (storage np_dtype is
    fp16, runtime trt_dtype is bfloat16) or any other non-matching combo.
    """
    target_dtype = inp.dtype
    const_shape = (1,) * max(1, len(tuple(inp.shape)))

    def _const(name, value):
        c = add_constant(network, const_shape, np.array([value], dtype=np.float32), dtype=dtype)
        return _cast_back_to_trt_dtype(network, c, target_dtype)

    # x^3
    x_sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    x_cu = network.add_elementwise(x_sq.get_output(0), inp, trt.ElementWiseOperation.PROD)
    # 0.044715 * x^3
    coeff = _const("coeff", 0.044715)
    scaled_cube = network.add_elementwise(x_cu.get_output(0), coeff, trt.ElementWiseOperation.PROD)
    # x + 0.044715 * x^3
    inner_sum = network.add_elementwise(
        inp, scaled_cube.get_output(0), trt.ElementWiseOperation.SUM
    )
    # sqrt(2/pi) * (x + 0.044715 * x^3)
    sqrt_2_over_pi = _const("sqrt_2_over_pi", np.sqrt(2.0 / np.pi))
    tanh_arg = network.add_elementwise(
        sqrt_2_over_pi, inner_sum.get_output(0), trt.ElementWiseOperation.PROD
    )
    # tanh(...)
    tanh_l = network.add_activation(tanh_arg.get_output(0), trt.ActivationType.TANH)
    # 1 + tanh(...)
    one = _const("one", 1.0)
    one_plus_tanh = network.add_elementwise(one, tanh_l.get_output(0), trt.ElementWiseOperation.SUM)
    # 0.5 * x
    half = _const("half", 0.5)
    half_x = network.add_elementwise(half, inp, trt.ElementWiseOperation.PROD)
    # 0.5 * x * (1 + tanh(...))
    result = network.add_elementwise(
        half_x.get_output(0), one_plus_tanh.get_output(0), trt.ElementWiseOperation.PROD
    )
    return result.get_output(0)


def add_self_attention_block_with_rope(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    hidden_size: int,
    num_heads: int,
    seq_length: int,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    q_bias: np.ndarray | None = None,
    k_bias: np.ndarray | None = None,
    v_bias: np.ndarray | None = None,
    o_bias: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Full self-attention with precomputed RoPE (for vision encoders with 3D RoPE).

    Unlike the KV-cache decoder attention, this processes all positions at once
    and applies RoPE via precomputed per-position cos/sin tables.

    Input hidden: [seq_length, hidden_size]
    cos_table/sin_table: [seq_length, hidden_size] precomputed constants
    Output: [seq_length, hidden_size]
    """
    head_dim = hidden_size // num_heads

    # Q, K, V projections: [seq, hidden] @ [hidden, hidden] = [seq, hidden]
    q = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_q, dtype=dtype)
    k = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_k, dtype=dtype)
    v = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_v, dtype=dtype)

    if q_bias is not None:
        q = add_bias_sum(network, q, hidden_size, q_bias, dtype=dtype)
    if k_bias is not None:
        k = add_bias_sum(network, k, hidden_size, k_bias, dtype=dtype)
    if v_bias is not None:
        v = add_bias_sum(network, v, hidden_size, v_bias, dtype=dtype)

    rope_dim = head_dim
    cos_half = cos_table[:, : rope_dim // 2]
    sin_half = sin_table[:, : rope_dim // 2]
    cos_const = add_constant(
        network, (1, seq_length, rope_dim // 2), cos_half.reshape(1, seq_length, -1), dtype=dtype
    )
    sin_const = add_constant(
        network, (1, seq_length, rope_dim // 2), sin_half.reshape(1, seq_length, -1), dtype=dtype
    )

    q = add_apply_rope_native_sequence(
        network,
        q,
        num_heads,
        head_dim,
        cos_const,
        sin_const,
        rotary_embedding_dim=rope_dim,
        sequence_length=seq_length,
    )
    k = add_apply_rope_native_sequence(
        network,
        k,
        num_heads,
        head_dim,
        cos_const,
        sin_const,
        rotary_embedding_dim=rope_dim,
        sequence_length=seq_length,
    )

    context_flat = add_attention_from_rows(
        network,
        q,
        k,
        v,
        num_heads=num_heads,
        head_dim=head_dim,
        q_seq=seq_length,
        kv_seq=seq_length,
    )

    # Output projection
    out = add_matmul_rhs_constant(network, context_flat, hidden_size, hidden_size, w_o, dtype=dtype)
    if o_bias is not None:
        out = add_bias_sum(network, out, hidden_size, o_bias, dtype=dtype)

    return out


def add_windowed_self_attention_with_rope(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    hidden_size: int,
    num_heads: int,
    seq_length: int,
    num_windows: int,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    window_patch_counts: np.ndarray | None = None,
    q_bias: np.ndarray | None = None,
    k_bias: np.ndarray | None = None,
    v_bias: np.ndarray | None = None,
    o_bias: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Windowed self-attention with precomputed RoPE.

    Splits the already window-ordered sequence into windows. Most Qwen-VL
    builds have equal-sized windows and use one batched attention op; HF
    smart-resized images can produce partial edge windows, which are handled
    by static per-window slices when ``window_patch_counts`` is provided.

    Input hidden: [seq_length, hidden_size]
    cos_table/sin_table: [seq_length, hidden_size]
    Output: [seq_length, hidden_size]
    """
    head_dim = hidden_size // num_heads
    counts = None
    if window_patch_counts is not None:
        counts = [
            int(v) for v in np.asarray(window_patch_counts).reshape(-1).tolist() if int(v) > 0
        ]
        if not counts or sum(counts) != seq_length:
            raise ValueError(
                "window_patch_counts must be positive and sum to seq_length: "
                f"sum={sum(counts) if counts else 0}, seq_length={seq_length}"
            )
        if all(c == counts[0] for c in counts):
            num_windows = len(counts)
            counts = None
    win_seq = seq_length // num_windows  # patches per window
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

    # Q, K, V projections: [seq, hidden] @ [hidden, hidden]
    q = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_q, dtype=dtype)
    k = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_k, dtype=dtype)
    v = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_v, dtype=dtype)

    if q_bias is not None:
        q = add_bias_sum(network, q, hidden_size, q_bias, dtype=dtype)
    if k_bias is not None:
        k = add_bias_sum(network, k, hidden_size, k_bias, dtype=dtype)
    if v_bias is not None:
        v = add_bias_sum(network, v, hidden_size, v_bias, dtype=dtype)

    rope_dim = head_dim
    cos_half = cos_table[:, : rope_dim // 2]
    sin_half = sin_table[:, : rope_dim // 2]
    cos_const = add_constant(
        network, (1, seq_length, rope_dim // 2), cos_half.reshape(1, seq_length, -1), dtype=dtype
    )
    sin_const = add_constant(
        network, (1, seq_length, rope_dim // 2), sin_half.reshape(1, seq_length, -1), dtype=dtype
    )

    q = add_apply_rope_native_sequence(
        network,
        q,
        num_heads,
        head_dim,
        cos_const,
        sin_const,
        rotary_embedding_dim=rope_dim,
        sequence_length=seq_length,
    )
    k = add_apply_rope_native_sequence(
        network,
        k,
        num_heads,
        head_dim,
        cos_const,
        sin_const,
        rotary_embedding_dim=rope_dim,
        sequence_length=seq_length,
    )

    if counts is None:
        q_win = network.add_shuffle(q)
        q_win.reshape_dims = (num_windows, win_seq, num_heads, head_dim)
        q_win.second_transpose = trt.Permutation([0, 2, 1, 3])

        k_win = network.add_shuffle(k)
        k_win.reshape_dims = (num_windows, win_seq, num_heads, head_dim)
        k_win.second_transpose = trt.Permutation([0, 2, 1, 3])

        v_win = network.add_shuffle(v)
        v_win.reshape_dims = (num_windows, win_seq, num_heads, head_dim)
        v_win.second_transpose = trt.Permutation([0, 2, 1, 3])

        context = add_attention_core(
            network,
            q_win.get_output(0),
            k_win.get_output(0),
            v_win.get_output(0),
            scale=attn_scale,
        )
        ctx_flat = network.add_shuffle(context)
        ctx_flat.first_transpose = trt.Permutation([0, 2, 1, 3])
        ctx_flat.reshape_dims = (seq_length, hidden_size)
        context_flat = ctx_flat.get_output(0)
    else:
        window_outputs = []
        offset = 0
        for window_len in counts:
            q_slice = network.add_slice(
                q, start=(offset, 0), shape=(window_len, hidden_size), stride=(1, 1)
            )
            k_slice = network.add_slice(
                k, start=(offset, 0), shape=(window_len, hidden_size), stride=(1, 1)
            )
            v_slice = network.add_slice(
                v, start=(offset, 0), shape=(window_len, hidden_size), stride=(1, 1)
            )
            window_outputs.append(
                add_attention_from_rows(
                    network,
                    q_slice.get_output(0),
                    k_slice.get_output(0),
                    v_slice.get_output(0),
                    num_heads=num_heads,
                    head_dim=head_dim,
                    q_seq=window_len,
                    kv_seq=window_len,
                    scale=attn_scale,
                )
            )
            offset += window_len
        concat = network.add_concatenation(window_outputs)
        concat.axis = 0
        context_flat = concat.get_output(0)

    out = add_matmul_rhs_constant(network, context_flat, hidden_size, hidden_size, w_o, dtype=dtype)
    if o_bias is not None:
        out = add_bias_sum(network, out, hidden_size, o_bias, dtype=dtype)

    return out


def add_patch_embed_3d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    in_channels: int,
    embed_dim: int,
    temporal_patch_size: int,
    patch_size: int,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """3D patch embedding via convolution.

    Input: [T*C, H, W] (already flattened temporal*channels) or [T, C, H, W]
    Output: [num_patches, embed_dim]

    The 3D convolution is implemented as a 2D convolution over the flattened
    temporal*channel dimension, matching HuggingFace's PatchEmbed3D.
    """
    # Input may be [T*C, H, W] (3D) or [T, C, H, W] (4D).
    # We need [1, T*C, H, W] for conv2d.
    inp_ndims = len(inp.shape)
    reshape_in = network.add_shuffle(inp)
    if inp_ndims == 3:
        # [T*C, H, W] -> [1, T*C, H, W]
        tc = inp.shape[0]
        h = inp.shape[1]
        w = inp.shape[2]
        reshape_in.reshape_dims = (1, tc, h, w)
    else:
        # [T, C, H, W] -> [1, T*C, H, W]
        reshape_in.reshape_dims = (1, temporal_patch_size * in_channels, -1, 0)

    # Conv2D with kernel [embed_dim, T*C, patch_size, patch_size]
    # weight shape from HF: [embed_dim, T*C, patch_size, patch_size]
    conv_w = trt.Weights(np.ascontiguousarray(weight, dtype=dtype))
    conv_b = trt.Weights()
    if bias is not None:
        conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=dtype))

    conv = network.add_convolution_nd(
        reshape_in.get_output(0),
        num_output_maps=embed_dim,
        kernel_shape=(patch_size, patch_size),
        kernel=conv_w,
        bias=conv_b,
    )
    conv.stride_nd = (patch_size, patch_size)

    # Output shape: [1, embed_dim, H', W'] -> flatten to [num_patches, embed_dim]
    reshape_out = network.add_shuffle(conv.get_output(0))
    reshape_out.first_transpose = trt.Permutation([0, 2, 3, 1])
    reshape_out.reshape_dims = (-1, embed_dim)

    return reshape_out.get_output(0)


def add_silu(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
) -> trt.ITensor:
    """SiLU (Swish): x * sigmoid(x)."""
    sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID)
    return network.add_elementwise(
        inp, sigmoid.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)


# Alias: add_gelu_tanh is the same as add_gelu_new (tanh approximation)
add_gelu_tanh = add_gelu_new


def validate_native_rope_dim(
    rotary_embedding_dim: int,
    *,
    field_name: str = "rotary_embedding_dim",
) -> int:
    """Validate the dimension contract required by TRT native RoPE."""
    rotary_embedding_dim = int(rotary_embedding_dim)
    if rotary_embedding_dim < 2 or rotary_embedding_dim % 2 != 0:
        raise ValueError(
            f"TRT native RoPE requires {field_name} to be an even value >= 2; "
            f"got {rotary_embedding_dim}"
        )
    return rotary_embedding_dim


def make_rope_table_half_dim(
    max_cache_length: int,
    head_dim: int,
    rope_theta: float,
    cosine: bool,
    partial_rotary_factor: float = 1.0,
    interleaved: bool = False,
) -> np.ndarray:
    """Build a RoPE cos/sin table of shape [max_cache_length, rotary_ndims // 2].

    IRotaryEmbeddingLayer expects the cos/sin cache with only the *half*
    rotary dimension (it internally handles both halves).  This is different
    from make_rope_table which produces [max_cache_length, hidden_size] by
    repeating the per-head values across all heads.

    Args:
        max_cache_length: Number of positions (rows in the table).
        head_dim:         Full head dimension (D).
        rope_theta:       Base frequency for inverse-frequency computation.
        cosine:           True → cos table, False → sin table.
        partial_rotary_factor: Fraction of head dims that rotate (default 1.0).
        interleaved:      If True, adjacent-pair frequencies (CodeGen/GPT-J).
                          If False, half-split frequencies (LLaMA/Qwen).

    Returns:
        Float32 array [max_cache_length, rotary_ndims // 2].
    """
    rotary_ndims = int(head_dim * partial_rotary_factor)
    rotary_ndims = validate_native_rope_dim(rotary_ndims)
    half = rotary_ndims // 2
    default = 1.0 if cosine else 0.0
    if max_cache_length <= 0 or rope_theta <= 0.0:
        return np.full((max(max_cache_length, 1), max(half, 1)), default, dtype=np.float32)
    table = np.full((max_cache_length, half), default, dtype=np.float32)
    for pos in range(max_cache_length):
        for d in range(half):
            # For both interleaved and rotate-half the frequency index is d
            # (the distinction only affects which input pair is rotated; the
            # freq assignment per half-dim is the same).
            exponent = (2.0 * d) / rotary_ndims
            inv_freq = rope_theta ** (-exponent)
            angle = pos * inv_freq
            table[pos, d] = np.cos(angle) if cosine else np.sin(angle)
    return table


def reshape_rows_to_heads_4d(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    num_heads: int,
    head_dim: int,
    sequence_length: int | None = None,
    tag: str | None = None,
) -> trt.ITensor:
    """Reshape [S, H * D] rows into [1, H, S, D].

    The transpose is required for S > 1 because each input row contains all
    heads for one token. ``sequence_length=None`` means runtime-dynamic S.
    """
    seq_dim = -1 if sequence_length is None else sequence_length
    r1 = network.add_shuffle(x)
    if tag:
        r1.name = tag + "_s_h_d"
    r1.reshape_dims = (seq_dim, num_heads, head_dim)
    r1.second_transpose = trt.Permutation([1, 0, 2])

    r2 = network.add_shuffle(r1.get_output(0))
    if tag:
        r2.name = tag + "_1_h_s_d"
    r2.reshape_dims = (1, num_heads, seq_dim, head_dim)
    return r2.get_output(0)


def reshape_heads_4d_to_rows(
    network: trt.INetworkDefinition,
    x_4d: trt.ITensor,
    attention_size: int,
    sequence_length: int | None = None,
    tag: str | None = None,
) -> trt.ITensor:
    """Reshape [1, H, S, D] back to [S, H * D]."""
    seq_dim = -1 if sequence_length is None else sequence_length
    out = network.add_shuffle(x_4d)
    if tag:
        out.name = tag + "_s_h_d"
    out.first_transpose = trt.Permutation([0, 2, 1, 3])
    out.reshape_dims = (seq_dim, attention_size)
    return out.get_output(0)


def add_apply_rope_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache_2d: trt.ITensor,
    sin_cache_2d: trt.ITensor,
    position_id: trt.ITensor,
    rotary_embedding_dim: int,
    interleaved: bool = False,
    sequence_length: int | None = 1,
) -> trt.ITensor:
    """Apply RoPE via TRT native IRotaryEmbeddingLayer.

    Handles both single-token decoder steps and dynamic-Sq prefill/decode
    graphs without a manual rotate-half matmul chain.

    Shape contract (IRotaryEmbeddingLayer with position_ids):
      input:           [1, num_heads, Sq, head_dim]  (reshaped internally)
      cos_cache_2d:    [max_S, rotary_embedding_dim // 2]  (2-D constant)
      sin_cache_2d:    [max_S, rotary_embedding_dim // 2]  (2-D constant)
      position_id:     [Sq] int32, reshaped to [1, Sq] internally
      interleaved:     False → rotate-half (LLaMA/Qwen)
                       True  → adjacent-pair (CodeGen/GPT-J)

    Args:
        inp:                  [Sq, num_heads * head_dim].
        num_heads:            Number of attention heads.
        head_dim:             Per-head dimension.
        cos_cache_2d:         Pre-built 2-D cos table constant.
        sin_cache_2d:         Pre-built 2-D sin table constant.
        position_id:          Runtime position indices, shape [Sq] int32.
        rotary_embedding_dim: Number of head dims that participate in RoPE.
        interleaved:          Frequency layout (see above).
        sequence_length:      Static Sq, or None for runtime-dynamic Sq.

    Returns:
        [Sq, num_heads * head_dim] with RoPE applied.
    """
    rotary_embedding_dim = validate_native_rope_dim(rotary_embedding_dim)
    attention_size = num_heads * head_dim

    inp_4d = reshape_rows_to_heads_4d(network, inp, num_heads, head_dim, sequence_length)

    # Reshape position_id [Sq] -> [1, Sq] (batch=1).
    seq_dim = -1 if sequence_length is None else sequence_length
    pos_2d = network.add_shuffle(position_id)
    pos_2d.reshape_dims = (1, seq_dim)

    rope = network.add_rotary_embedding(
        inp_4d,
        cos_cache_2d,
        sin_cache_2d,
        interleaved,
        rotary_embedding_dim,
    )
    rope.set_input(3, pos_2d.get_output(0))

    return reshape_heads_4d_to_rows(network, rope.get_output(0), attention_size, sequence_length)


def add_apply_rope_native_sequence(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache_3d: trt.ITensor,
    sin_cache_3d: trt.ITensor,
    rotary_embedding_dim: int,
    interleaved: bool = False,
    sequence_length: int | None = None,
) -> trt.ITensor:
    """Apply native RoPE with per-position caches [1, Sq, rotary_dim / 2]."""
    rotary_embedding_dim = validate_native_rope_dim(rotary_embedding_dim)
    attention_size = num_heads * head_dim
    inp_4d = reshape_rows_to_heads_4d(network, inp, num_heads, head_dim, sequence_length)
    rope = network.add_rotary_embedding(
        inp_4d,
        cos_cache_3d,
        sin_cache_3d,
        interleaved,
        rotary_embedding_dim,
    )
    return reshape_heads_4d_to_rows(network, rope.get_output(0), attention_size, sequence_length)


def add_attention_core(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    causal: bool = False,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
    fp32_accumulation: bool = False,
) -> trt.ITensor:
    """Scaled dot-product attention via TRT native IAttention layer.

    Replaces the manual Q@K^T → scale → softmax → @V chain.  TRT 10 fuses
    this into a single kernel when a compatible implementation is available;
    decomposable=True ensures a correct fallback to primitives otherwise.

    NOTE: TRT IAttention computes raw BMM1 = Q @ K^T without any built-in
    1/sqrt(D) scaling.  We pre-scale Q by 1/sqrt(D) so that the fused kernel
    computes the standard scaled dot-product attention formula.

    Args:
        q_4d:    Query  [B, H, q_seq, D].
        k_4d:    Key    [B, H, kv_seq, D].
        v_4d:    Value  [B, H, kv_seq, D].
        causal:  Apply causal (autoregressive) mask.  Mutually exclusive
                 with ``mask``.
        mask:    Optional additive float mask [B, H, q_seq, kv_seq] added
                 to scaled logits before softmax.  Cannot be used with
                 causal=True.
        scale:   Optional Q pre-scale factor.  Defaults to 1/sqrt(D).
        fp32_accumulation:
                 Cast Q/K/V to FP32 before IAttention, then cast the context
                 back to the original Q dtype.  TRT may still select a
                 Half-input fused MHA tactic after optimizing the casts, while
                 keeping the IAttention accumulation/output boundary in FP32.

    Returns:
        Context tensor [B, H, q_seq, D].
    """
    output_dtype = q_4d.dtype
    if fp32_accumulation and output_dtype != trt.float32:
        q_4d = network.add_cast(q_4d, trt.float32).get_output(0)
        k_4d = network.add_cast(k_4d, trt.float32).get_output(0)
        v_4d = network.add_cast(v_4d, trt.float32).get_output(0)
        if mask is not None and mask.dtype != trt.float32:
            mask = network.add_cast(mask, trt.float32).get_output(0)

    # Pre-scale Q: TRT IAttention does not apply score scaling itself.
    # Match the scale constant's dtype to Q's dtype: in strongly-typed networks
    # a FP32 constant mixed with a FP16/BF16 Q causes add_elementwise to emit
    # a type-mismatch error and produce a tensor with corrupted dimensions,
    # which makes add_attention return None.
    if scale is None:
        head_dim = q_4d.shape[-1]
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    # Use FP16 weights directly for FP16; BF16 has no numpy native type so
    # create as FP32 and cast; FP32 falls through to the default.
    scale_np_dtype = np.float16 if q_4d.dtype == trt.float16 else np.float32
    scale_t = add_constant(network, (1, 1, 1, 1), np.array([[[[scale]]]]), dtype=scale_np_dtype)
    if q_4d.dtype == trt.bfloat16:
        scale_t = network.add_cast(scale_t, trt.bfloat16).get_output(0)
    q_scaled = network.add_elementwise(q_4d, scale_t, trt.ElementWiseOperation.PROD)

    attn = network.add_attention(
        q_scaled.get_output(0),
        k_4d,
        v_4d,
        trt.AttentionNormalizationOp.SOFTMAX,
        causal,
    )
    # Allow TRT to decompose into primitive ops when no fused kernel is
    # available (e.g. unsupported head-dim or dtype).  This guarantees
    # correctness on any configuration at the cost of potential performance.
    attn.decomposable = True
    if mask is not None and not causal:
        attn.mask = mask
    return _cast_back_to_trt_dtype(network, attn.get_output(0), output_dtype)


def _scalar_constant_for_trt_dtype(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    value: float,
    dtype: trt.DataType,
) -> trt.ITensor:
    np_dtype = np.float16 if dtype == trt.float16 else np.float32
    const = add_constant(network, shape, np.full(shape, value, dtype=np_dtype), dtype=np_dtype)
    if dtype == trt.bfloat16:
        const = network.add_cast(const, trt.bfloat16).get_output(0)
    return const


def add_tanh_softcap(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    cap: float,
    *,
    scalar_shape: tuple[int, ...],
) -> trt.ITensor:
    """Apply ``tanh(tensor / cap) * cap`` using scalar broadcasting."""
    cap_t = _scalar_constant_for_trt_dtype(network, scalar_shape, float(cap), tensor.dtype)
    scaled = network.add_elementwise(tensor, cap_t, trt.ElementWiseOperation.DIV).get_output(0)
    capped = network.add_activation(scaled, trt.ActivationType.TANH).get_output(0)
    return network.add_elementwise(capped, cap_t, trt.ElementWiseOperation.PROD).get_output(0)


def _repeat_kv_heads_4d(
    network: trt.INetworkDefinition,
    x_4d: trt.ITensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> trt.ITensor:
    if num_kv_heads == num_heads:
        return x_4d
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")

    repeat = num_heads // num_kv_heads
    if num_kv_heads == 1:
        concat = network.add_concatenation([x_4d] * repeat)
        concat.axis = 1
        return concat.get_output(0)

    x_shape = network.add_shape(x_4d).get_output(0)
    one = add_constant(network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    seq = network.add_slice(x_shape, start=(2,), shape=(1,), stride=(1,))
    dim = add_constant(network, (1,), np.array([head_dim], dtype=np.int64), dtype=np.int64)
    slice_shape = network.add_concatenation([one, one, seq.get_output(0), dim])
    slice_shape.axis = 0

    repeated = []
    for head_idx in range(num_kv_heads):
        head_slice = network.add_slice(
            x_4d, start=(0, head_idx, 0, 0), shape=(1, 1, 1, head_dim), stride=(1, 1, 1, 1)
        )
        head_slice.set_input(2, slice_shape.get_output(0))
        repeated.extend([head_slice.get_output(0)] * repeat)

    concat = network.add_concatenation(repeated)
    concat.axis = 1
    return concat.get_output(0)


def _add_attention_core_with_logit_softcap(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mask: trt.ITensor | None,
    scale: float,
    logit_softcap: float,
) -> trt.ITensor:
    output_dtype = q_4d.dtype
    k_4d = _repeat_kv_heads_4d(
        network, k_4d, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    v_4d = _repeat_kv_heads_4d(
        network, v_4d, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )

    score_q = q_4d
    score_k = k_4d
    score_mask = mask
    if output_dtype != trt.float32:
        score_q = network.add_cast(score_q, trt.float32).get_output(0)
        score_k = network.add_cast(score_k, trt.float32).get_output(0)
        if score_mask is not None and score_mask.dtype != trt.float32:
            score_mask = network.add_cast(score_mask, trt.float32).get_output(0)

    scale_t = _scalar_constant_for_trt_dtype(network, (1, 1, 1, 1), scale, score_q.dtype)
    scores = network.add_matrix_multiply(
        score_q, trt.MatrixOperation.NONE, score_k, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scores = network.add_elementwise(scores, scale_t, trt.ElementWiseOperation.PROD).get_output(0)

    scores = add_tanh_softcap(network, scores, logit_softcap, scalar_shape=(1, 1, 1, 1))

    if score_mask is not None:
        scores = network.add_elementwise(
            scores, score_mask, trt.ElementWiseOperation.SUM
        ).get_output(0)

    probs = network.add_softmax(scores)
    probs.axes = 1 << 3
    probs_t = probs.get_output(0)
    if probs_t.dtype != output_dtype:
        probs_t = network.add_cast(probs_t, output_dtype).get_output(0)

    context = network.add_matrix_multiply(
        probs_t, trt.MatrixOperation.NONE, v_4d, trt.MatrixOperation.NONE
    ).get_output(0)
    return _cast_back_to_trt_dtype(network, context, output_dtype)


def add_attention_from_rows(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int | None = None,
    q_seq: int | None,
    kv_seq: int | None,
    causal: bool = False,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
    logit_softcap: float | None = None,
    fp32_accumulation: bool = False,
    tag: str | None = None,
) -> trt.ITensor:
    """Native IAttention for row-major [S, H * D] Q/K/V tensors.

    ``num_kv_heads`` can be smaller than ``num_heads`` for GQA/MQA. TRT
    native IAttention supports this directly, so callers should not expand K/V
    heads unless the model semantics require per-query-head K/V values.
    """
    attention_size = num_heads * head_dim
    kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    q_4d = reshape_rows_to_heads_4d(
        network,
        q,
        num_heads,
        head_dim,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".q",
    )
    k_4d = reshape_rows_to_heads_4d(
        network,
        k,
        kv_heads,
        head_dim,
        sequence_length=kv_seq,
        tag=None if tag is None else tag + ".k",
    )
    v_4d = reshape_rows_to_heads_4d(
        network,
        v,
        kv_heads,
        head_dim,
        sequence_length=kv_seq,
        tag=None if tag is None else tag + ".v",
    )
    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    if logit_softcap is not None and float(logit_softcap) > 0.0:
        if causal:
            raise NotImplementedError("logit_softcap attention requires an explicit additive mask")
        ctx_4d = _add_attention_core_with_logit_softcap(
            network,
            q_4d,
            k_4d,
            v_4d,
            num_heads=num_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            mask=mask,
            scale=scale,
            logit_softcap=float(logit_softcap),
        )
    else:
        ctx_4d = add_attention_core(
            network,
            q_4d,
            k_4d,
            v_4d,
            causal=causal,
            mask=mask,
            scale=scale,
            fp32_accumulation=fp32_accumulation,
        )
    return reshape_heads_4d_to_rows(
        network,
        ctx_4d,
        attention_size,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".ctx",
    )


# Backward-compatible name used by existing tests and call sites.
_add_attention_core = add_attention_core


# Graph Blocks


trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..weights import WeightDict
    from ....quantization.context import QuantContext


# ---------------------------------------------------------------------------
# Precision boundary helpers (used by standard_decoder_builder, not inside
# blocks themselves).
# ---------------------------------------------------------------------------


def make_matmul_fn(network, dtype, quant_ctx):
    """Create a matmul callable that routes through quant_ctx if present.

    Returns a function: (lhs, lhs_w, rhs_w, rhs_weights, weight_name) -> ITensor
    """
    if quant_ctx is None:

        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return add_matmul_rhs_constant(network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype)

        return matmul
    else:

        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return quant_ctx.maybe_quantized_matmul(
                network, lhs, lhs_w, rhs_w, rhs_weights, weight_name, dtype=dtype
            )

        return matmul


_make_matmul_fn = make_matmul_fn


def cast_to_dtype(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target_dtype: trt.DataType,
) -> trt.ITensor:
    """Cast tensor to target dtype (no-op if already matching)."""
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)


def add_swiglu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
) -> trt.ITensor:
    """Gate/up/down SwiGLU MLP. Returns output tensor."""
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    _lp = layer_prefix or prefix

    gate = matmul(inp, hidden_size, mlp_size, weights[f"{prefix}.w_gate"], f"{_lp}.w_gate")
    up = matmul(inp, hidden_size, mlp_size, weights[f"{prefix}.w_up"], f"{_lp}.w_up")

    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(swish.get_output(0), up, trt.ElementWiseOperation.PROD)

    mlp_out = matmul(
        gated.get_output(0), mlp_size, hidden_size, weights[f"{prefix}.w_down"], f"{_lp}.w_down"
    )
    return mlp_out


# Utils


trt = trt_compat.get_trt()


@dataclass(frozen=True)
class BuilderContext:
    """TensorRT objects shared by engine builders."""

    logger: trt.Logger
    builder: trt.Builder
    network: trt.INetworkDefinition
    config: trt.IBuilderConfig


def create_builder_context(
    *,
    verbose: bool,
    workspace_bytes: int,
    strongly_typed: bool = True,
    disable_tf32: bool = False,
) -> BuilderContext:
    """Create a TensorRT builder, network, and config with common defaults."""
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if disable_tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    return BuilderContext(
        logger=logger,
        builder=builder,
        network=network,
        config=config,
    )


# Qwen Image Dit Builder


trt = trt_compat.get_trt()

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# BF16 internal-compute helpers.
#
# Strategy: STRONGLY_TYPED network with bf16 weights/activations for the heavy
# compute (matmul, attention, AdaLN, GELU FFN, RoPE complex muls). The
# network's IO stays fp32 -- inputs are cast to bf16 at the boundary and
# outputs are cast back to fp32 before mark_output, so the C++ runtime
# (qwen_image_pipeline.cpp) can keep
# binding fp32 host buffers without changes. Mirrors the pattern used by
# flux2_dit_builder.py.
#
# RoPE cos/sin tables and the eps constants stay fp32 at construction time
# and are cast to bf16 at point-of-use (the numerical sensitivity of the
# tables is preserved that way; flux2 does the same).
# ---------------------------------------------------------------------------

_CAST_DTYPE = trt.bfloat16

# Anchors for bf16 numpy arrays so Python GC doesn't free them before
# builder.build_serialized_network() returns. Without this, TRT silently
# reads garbage (or segfaults).
_weight_refs: list[np.ndarray] = []


def _to_compute_dtype(network, tensor):
    """Cast a tensor to the reduced compute dtype if not already."""
    if tensor.dtype == _CAST_DTYPE:
        return tensor
    return network.add_cast(tensor, _CAST_DTYPE).get_output(0)


def _to_fp32(network, tensor):
    """Cast a tensor back to fp32 if not already."""
    if tensor.dtype == trt.float32:
        return tensor
    return network.add_cast(tensor, trt.float32).get_output(0)


def _make_reduced_weights(data_fp32: np.ndarray):
    """Build (trt.Weights, anchored_ndarray) in the reduced compute dtype."""
    import ml_dtypes

    if (
        isinstance(data_fp32, np.ndarray)
        and data_fp32.dtype == ml_dtypes.bfloat16
        and data_fp32.flags.c_contiguous
    ):
        bf16_arr = data_fp32
    else:
        bf16_arr = np.ascontiguousarray(np.asarray(data_fp32).astype(ml_dtypes.bfloat16))
    w = trt.Weights(trt.bfloat16, bf16_arr.ctypes.data, bf16_arr.size)
    return w, bf16_arr


def _add_constant_reduced(network, shape, values_fp32) -> "trt.ITensor":
    """Add a constant in the reduced compute dtype (anchors the bf16 array)."""
    w, arr_ref = _make_reduced_weights(values_fp32)
    _weight_refs.append(arr_ref)
    return network.add_constant(shape, w).get_output(0)


def _reset_weight_refs() -> None:
    """Clear the bf16 array anchor list before a fresh engine build."""
    global _weight_refs
    _weight_refs = []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_builder(verbose: bool):
    """Create a TRT builder + STRONGLY_TYPED network with sane defaults.

    Also resets the bf16 weight-anchor list so a fresh build doesn't keep
    references to arrays from a prior build (would only matter for tests
    that build many engines in one process, but it's cheap insurance).
    """
    _reset_weight_refs()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    # Shape ops are tiny; 256 MiB workspace is plenty.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    return builder, config, network


def _serialize_and_write(builder, network, config, out_path: PathLike, label: str) -> Path:
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"{label} TRT engine build failed")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(bytes(plan))
    return out_path


def _add_get_timestep_embedding(
    network,
    timestep,
    *,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1000.0,
    max_period: float = 10000.0,
):
    """In-network port of ``diffusers.models.embeddings.get_timestep_embedding``.

    Input  ``timestep``: ITensor of shape ``[N]`` (fp32).
    Output ITensor of shape ``[N, embedding_dim]`` (fp32).
    """
    assert embedding_dim % 2 == 0, "embedding_dim must be even"
    half_dim = embedding_dim // 2

    # exponent = -log(max_period) * arange(half_dim) / (half_dim - downscale_freq_shift)
    # emb_base = exp(exponent)
    base = -math.log(max_period) * np.arange(half_dim, dtype=np.float32)
    base = base / (half_dim - downscale_freq_shift)
    base = np.exp(base).reshape(1, half_dim) * float(scale)

    base_const = network.add_constant((1, half_dim), trt.Weights(base)).get_output(0)

    # ts: [N] -> [N, 1]
    ts_reshape = network.add_shuffle(timestep)
    ts_reshape.reshape_dims = (-1, 1)
    ts_2d = ts_reshape.get_output(0)

    # emb = ts * base                              [N, half_dim]
    prod = network.add_elementwise(ts_2d, base_const, trt.ElementWiseOperation.PROD)
    arg = prod.get_output(0)

    sin_p = network.add_unary(arg, trt.UnaryOperation.SIN).get_output(0)
    cos_p = network.add_unary(arg, trt.UnaryOperation.COS).get_output(0)

    if flip_sin_to_cos:
        # diffusers: emb = cat([sin, cos]); if flip_sin_to_cos: emb = cat([cos, sin])
        order = [cos_p, sin_p]
    else:
        order = [sin_p, cos_p]
    cat = network.add_concatenation(order)
    cat.axis = 1
    return cat.get_output(0)


# ---------------------------------------------------------------------------
# Dynamic-batch slice helpers.
#
# When ``max_batch_size > 1`` the leading dim of the engine inputs is dynamic
# (``-1``). TRT's static-shape ``add_slice`` requires every component of the
# ``shape`` argument to be a baked-in integer, so we build a runtime shape
# tensor that copies dim 0 from a reference tensor (whichever input
# transitively carries the dynamic batch) and concatenates the static trailing
# dims. Callers with a static-batch graph (``batch_size == 1``) keep the
# original baked-shape slice — no perf or layer-count regression on the
# default path.
# ---------------------------------------------------------------------------


def _dynamic_batch_shape(network, reference, tail: tuple[int, ...]):
    """Build a shape tensor ``[B, *tail]`` reading dim 0 from ``reference``."""
    ref_shape = network.add_shape(reference).get_output(0)
    batch = network.add_slice(ref_shape, start=(0,), shape=(1,), stride=(1,))
    tail_t = add_constant(network, (len(tail),), np.asarray(tail, dtype=np.int64), dtype=np.int64)
    target = network.add_concatenation([batch.get_output(0), tail_t])
    target.axis = 0
    return target.get_output(0)


def _slice_batch_vector(network, x, start_width: int, width: int, batch_size: int):
    """Slice ``[B, D]`` along D, preserving dynamic ``B`` when present."""
    if batch_size == 1:
        return network.add_slice(
            x, start=(0, start_width), shape=(1, width), stride=(1, 1)
        ).get_output(0)
    s = network.add_slice(x, start=(0, start_width), shape=(0, 0), stride=(1, 1))
    s.set_input(2, _dynamic_batch_shape(network, x, (width,)))
    return s.get_output(0)


def _slice_batch_sequence(network, x, start_seq: int, length: int, width: int, batch_size: int):
    """Slice ``[B, S, D]`` along S, preserving dynamic ``B`` when present."""
    if batch_size == 1:
        return network.add_slice(
            x, start=(0, start_seq, 0), shape=(1, length, width), stride=(1, 1, 1)
        ).get_output(0)
    s = network.add_slice(x, start=(0, start_seq, 0), shape=(0, 0, 0), stride=(1, 1, 1))
    s.set_input(2, _dynamic_batch_shape(network, x, (length, width)))
    return s.get_output(0)


def _slice_batch_complex_part(
    network, x, seq_len: int, num_heads: int, half: int, complex_index: int, batch_size: int
):
    """Slice real/imag part from ``[B, S, H, D/2, 2]`` -> ``[B, S, H, D/2]``."""
    if batch_size == 1:
        s = network.add_slice(
            x,
            start=(0, 0, 0, 0, complex_index),
            shape=(1, seq_len, num_heads, half, 1),
            stride=(1, 1, 1, 1, 1),
        )
    else:
        s = network.add_slice(
            x, start=(0, 0, 0, 0, complex_index), shape=(0, 0, 0, 0, 0), stride=(1, 1, 1, 1, 1)
        )
        s.set_input(2, _dynamic_batch_shape(network, x, (seq_len, num_heads, half, 1)))
    r = network.add_shuffle(s.get_output(0))
    r.reshape_dims = (-1, seq_len, num_heads, half)
    return r.get_output(0)


def _add_linear(network, x, weight_np: np.ndarray, bias_np: np.ndarray):
    """Linear(in -> out): y = x @ W^T + b in reduced compute dtype.

    weight_np shape: [out_dim, in_dim]; bias_np shape: [out_dim].
    Input x shape:  [N, in_dim]; output: [N, out_dim].
    Caller-supplied weights are fp32; this helper internally converts them
    to bf16 (anchored against GC) and emits a bf16 matmul + bias-add. ``x``
    is cast to bf16 at the boundary if needed; output stays in bf16.
    """
    out_dim, in_dim = weight_np.shape
    x_c = _to_compute_dtype(network, x)
    w_const = _add_constant_reduced(
        network, (out_dim, in_dim), np.ascontiguousarray(weight_np, dtype=np.float32)
    )
    # x [N, in_dim] @ W^T [in_dim, out_dim] = [N, out_dim]
    matmul = network.add_matrix_multiply(
        x_c, trt.MatrixOperation.NONE, w_const, trt.MatrixOperation.TRANSPOSE
    )
    y = matmul.get_output(0)
    if bias_np is not None:
        b_const = _add_constant_reduced(
            network, (1, out_dim), np.ascontiguousarray(bias_np, dtype=np.float32)
        )
        add = network.add_elementwise(y, b_const, trt.ElementWiseOperation.SUM)
        y = add.get_output(0)
    return y


# ---------------------------------------------------------------------------
# 3-axis RoPE (frame, height, width) for the Qwen-Image MMDiT denoiser.
#
# Matches diffusers ``QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56],
# scale_rope=True)`` (transformer_qwenimage.py:199-321):
#
#   * For each axis k the complex freqs at position i are
#         freqs[i, j] = exp(i * theta**(-2j/axes_dim[k]))    j in [0, axes_dim[k]/2)
#     (rope_params: torch.polar(ones_like(angles), angles)).
#   * Image (T=1) positions for token (h, w) are
#         frame_idx = 0
#         h_idx in {-(H - H//2), ..., -1, 0, 1, ..., H//2 - 1}  (scale_rope=True)
#         w_idx analogous, length W.
#     Concatenated along the last (head_dim/2) axis.
#   * Text positions: pos_freqs rows in
#         [max_vid_index, max_vid_index + n_text), where
#         max_vid_index = max(H//2, W//2).
#     Same row used for all three axes (the pos_freqs table is built once
#     from the same pos_index across axes).
#
# Output layout: concatenated as [img_freqs | txt_freqs] along the seq axis,
# giving a single (cos, sin) pair of shape [seq_len, head_dim] suitable for
# the attention application
#     out = x * cos + rotate_half(x) * sin
# (the cos/sin tables are "expanded" from the complex form by interleaved
# duplication along the last axis: cos[..., 2j] = cos[..., 2j+1] = real;
# sin[..., 2j] = sin[..., 2j+1] = imag).
# ---------------------------------------------------------------------------

_ROPE_PRECOMPUTE_LEN = 4096  # diffusers uses arange(4096) for pos/neg indices


def _rope_params_complex(index: np.ndarray, dim: int, theta: float) -> np.ndarray:
    """Return complex freqs of shape ``[len(index), dim // 2]`` (cos + i sin)."""
    assert dim % 2 == 0
    inv_freq = 1.0 / np.power(theta, np.arange(0, dim, 2, dtype=np.float64) / float(dim))
    angles = np.outer(index.astype(np.float64), inv_freq)
    return np.cos(angles) + 1j * np.sin(angles)


def _qwen_rope_axis_freqs(
    axes_dim: list[int],
    theta: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return positive/negative complex RoPE tables split by Qwen image axis."""
    pos_index = np.arange(_ROPE_PRECOMPUTE_LEN)
    neg_index = pos_index[::-1] * -1 - 1  # [-1, -2, ..., -_ROPE_PRECOMPUTE_LEN]

    pos_axis_freqs = [_rope_params_complex(pos_index, axes_dim[k], theta) for k in range(3)]
    neg_axis_freqs = [_rope_params_complex(neg_index, axes_dim[k], theta) for k in range(3)]
    return pos_axis_freqs, neg_axis_freqs


def _precompute_qwen_image_freqs(
    *,
    axes_dim: list[int],
    pos_axis_freqs: list[np.ndarray],
    neg_axis_freqs: list[np.ndarray],
    h_lat: int,
    w_lat: int,
    frame_index: int,
) -> np.ndarray:
    """Pre-compute one image/grid chunk of Qwen 3-axis RoPE complex freqs.

    ``frame_index`` mirrors ``QwenEmbedRope._compute_video_freqs(..., idx)``
    and is what distinguishes multiple Edit condition-image grids from the
    generated-image grid.
    """
    head_dim = sum(axes_dim)
    splits = [a // 2 for a in axes_dim]

    frame = 1
    H, W = h_lat, w_lat

    # Frame axis (pos rows [0:1], shared across all (h, w)).
    freqs_frame = np.broadcast_to(
        pos_axis_freqs[0][frame_index : frame_index + frame].reshape(frame, 1, 1, splits[0]),
        (frame, H, W, splits[0]),
    )

    # Height: scale_rope=True split [-(H - H//2):] from neg, then [:H//2] from pos.
    h_neg = neg_axis_freqs[1][-(H - H // 2) :]
    h_pos = pos_axis_freqs[1][: H // 2]
    h_combined = np.concatenate([h_neg, h_pos], axis=0)  # [H, splits[1]]
    freqs_height = np.broadcast_to(h_combined.reshape(1, H, 1, splits[1]), (frame, H, W, splits[1]))

    w_neg = neg_axis_freqs[2][-(W - W // 2) :]
    w_pos = pos_axis_freqs[2][: W // 2]
    w_combined = np.concatenate([w_neg, w_pos], axis=0)
    freqs_width = np.broadcast_to(w_combined.reshape(1, 1, W, splits[2]), (frame, H, W, splits[2]))

    return np.concatenate([freqs_frame, freqs_height, freqs_width], axis=-1).reshape(
        frame * H * W, head_dim // 2
    )


def _expand_qwen_rope_complex(combined: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand complex Qwen RoPE freqs to duplicated real-valued cos/sin rows."""
    # Expand to [seq, head_dim] via interleaved duplication: each complex
    # entry contributes (cos, sin) repeated twice -- one for each of the
    # two real elements in a rotation pair.
    real = np.repeat(combined.real, 2, axis=-1).astype(np.float32)
    imag = np.repeat(combined.imag, 2, axis=-1).astype(np.float32)
    return np.ascontiguousarray(real), np.ascontiguousarray(imag)


def _precompute_qwen_rope_tables_for_shapes(
    axes_dim: list[int],
    image_shapes: list[tuple[int, int]],
    n_text: int,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute Qwen image/text RoPE tables for one or more image grids.

    ``image_shapes`` contains packed-token grids ``(h, w)`` in the order they
    are concatenated into the denoiser's image stream. T2I passes one shape.
    Edit passes the generated-image shape first, followed by VAE-condition
    image shapes. This mirrors diffusers ``QwenEmbedRope.forward`` where
    ``img_shapes`` is a list like ``[(gen_f, gen_h, gen_w),
    (cond_f, cond_h, cond_w)]``.
    """
    if len(axes_dim) != 3:
        raise ValueError(f"axes_dim must have 3 entries, got {axes_dim!r}")
    if any(a % 2 != 0 for a in axes_dim):
        raise ValueError(f"each axis dim must be even, got {axes_dim!r}")
    if not image_shapes:
        raise ValueError("image_shapes must contain at least one image grid")
    if n_text < 0:
        raise ValueError(f"n_text must be non-negative, got {n_text}")

    pos_axis_freqs, neg_axis_freqs = _qwen_rope_axis_freqs(axes_dim, theta)
    vid_freqs: list[np.ndarray] = []
    max_vid_index = 0
    for idx, (h_lat, w_lat) in enumerate(image_shapes):
        if h_lat <= 0 or w_lat <= 0:
            raise ValueError(f"image_shapes[{idx}] must be positive, got {(h_lat, w_lat)!r}")
        max_vid_index = max(max_vid_index, h_lat // 2, w_lat // 2)
        vid_freqs.append(
            _precompute_qwen_image_freqs(
                axes_dim=axes_dim,
                pos_axis_freqs=pos_axis_freqs,
                neg_axis_freqs=neg_axis_freqs,
                h_lat=h_lat,
                w_lat=w_lat,
                frame_index=idx,
            )
        )

    if max_vid_index + n_text > _ROPE_PRECOMPUTE_LEN:
        raise ValueError(
            "n_text + max image grid half-extent exceeds the 4096-row Qwen RoPE pre-compute budget"
        )

    # Text freqs: rows [max_vid_index, max_vid_index + n_text) from the
    # concatenated positive freqs across all axes.
    pos_freqs_all = np.concatenate(pos_axis_freqs, axis=1)  # [4096, head_dim/2]
    txt_freqs = pos_freqs_all[max_vid_index : max_vid_index + n_text]

    combined = np.concatenate([*vid_freqs, txt_freqs], axis=0)  # [seq, head_dim/2]
    return _expand_qwen_rope_complex(combined)


# ---------------------------------------------------------------------------
# MMDiT joint-stream block.
#
# This implements one ``QwenImageTransformerBlock`` from diffusers
# ``transformer_qwenimage.py``. The math (verified against
# diffusers/src/diffusers/models/transformers/transformer_qwenimage.py
# at lines 473-730):
#
#   img_mod_params = Linear(SiLU(temb))                        # [B, 6*dim]
#   txt_mod_params = Linear(SiLU(temb))                        # [B, 6*dim]
#   img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)       # each [B, 3*dim]
#   txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)
#   (shift_msa, scale_msa, gate_msa) = mod1.chunk(3, dim=-1)
#   (shift_mlp, scale_mlp, gate_mlp) = mod2.chunk(3, dim=-1)
#
#   img_normed = LayerNorm(img_tokens)  *  (1 + scale_msa_img) + shift_msa_img
#   txt_normed = LayerNorm(txt_tokens)  *  (1 + scale_msa_txt) + shift_msa_txt
#
#   img_q = to_q(img_normed); img_k = to_k(img_normed); img_v = to_v(img_normed)
#   txt_q = add_q_proj(txt_normed); txt_k = add_k_proj(txt_normed); txt_v = add_v_proj(txt_normed)
#
#   # qk_norm="rms_norm" → RMSNorm over head_dim, per-head, elementwise_affine
#   img_q = norm_q(img_q.unflatten(-1, H, D));     img_k = norm_k(...)
#   txt_q = norm_added_q(txt_q.unflatten(-1, H, D)); txt_k = norm_added_k(...)
#
#   # RoPE (apply_rotary_emb_qwen, use_real=False) — pair-based rotation
#   #   x[..., 2j], x[..., 2j+1] = (x[..., 2j]*c - x[..., 2j+1]*s,
#   #                                x[..., 2j]*s + x[..., 2j+1]*c)
#   # where c, s come from `rope_cos`/`rope_sin` (interleaved-duplicated;
#   # cos[..., 2j] = cos[..., 2j+1] = real_part).
#   img_q/img_k <- RoPE(img_freqs)     txt_q/txt_k <- RoPE(txt_freqs)
#
#   # Joint attention with concat order [txt, img] along seq dim.
#   joint_q = cat([txt_q, img_q], dim=seq)
#   joint_k = cat([txt_k, img_k], dim=seq)
#   joint_v = cat([txt_v, img_v], dim=seq)
#   joint_out = SDPA(joint_q, joint_k, joint_v)
#   txt_attn_out, img_attn_out = split(joint_out, [N_txt, N_img], dim=seq)
#   img_attn_out = to_out.0(img_attn_out)
#   txt_attn_out = to_add_out(txt_attn_out)
#
#   # Gated residual.
#   hs_img  = img_tokens + gate_msa_img.unsqueeze(1) * img_attn_out
#   hs_txt  = txt_tokens + gate_msa_txt.unsqueeze(1) * txt_attn_out
#
#   # MLP step: LayerNorm + modulate(mod2) + FF (GELU-approximate "tanh").
#   img_n2   = LayerNorm(hs_img)   * (1 + scale_mlp_img) + shift_mlp_img
#   img_mlp  = Linear(GELU_tanh(Linear(img_n2)))
#   hs_img  += gate_mlp_img.unsqueeze(1) * img_mlp
#   # ... same for text via ff_context/txt_mlp ...
#
#   return (txt_out, img_out)
#
# Norms in the block are plain ``nn.LayerNorm(dim, elementwise_affine=False)``
# i.e. no learnable scale/bias on the norm itself. The modulation parameters
# come from ``img_mod.1`` / ``txt_mod.1`` (nn.Linear → 6*dim).
# ---------------------------------------------------------------------------


@dataclass
class JointBlockConfig:
    """Static configuration for one Qwen-Image MMDiT joint block.

    Defaults match ``Qwen/Qwen-Image-2512`` (hidden_size=3072, 24 heads,
    head_dim=128, MLP mult=4 → intermediate=12288).
    """

    hidden_size: int = 3072
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    intermediate_size: int = 12288  # FeedForward(dim, mult=4) → 4 * 3072.
    rms_norm_eps: float = 1e-6
    layer_norm_eps: float = 1e-6


# ---------------------------------------------------------------------------
# Internal TRT helpers (joint block only).
#
# TODO: consider promoting _add_modulate, _add_gate_residual, and
# _add_layernorm_no_affine_3d to graph_blocks now that the shape contract
# has been confirmed to hold for full-stack composition. Defer until
# end-to-end image generation works (avoid disrupting the working pipeline
# mid-build-out).
# ---------------------------------------------------------------------------


def _add_2d_constant(network, arr: np.ndarray) -> "trt.ITensor":
    """Add a small fp32 constant (e.g. eps, ones/zeros for LayerNorm, RoPE).

    Kept fp32 by design -- numerically sensitive values that we don't want
    to quantize at construction time. Callers cast to the compute dtype at
    point-of-use when needed.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return network.add_constant(arr.shape, trt.Weights(arr)).get_output(0)


def _add_linear_2d(
    network, x, in_dim: int, out_dim: int, w_hf: np.ndarray, b_hf: "np.ndarray | None"
):
    """Linear projection for [N, in_dim] input. ``w_hf`` is HF-order [out, in].

    Returns [N, out_dim]. Internally transposes weight to [in, out] and emits
    a bf16 matmul + optional bias-add. ``x`` is cast to bf16 at the boundary.
    Weights are converted fp32 -> bf16 once at build time and anchored.
    """
    x_c = _to_compute_dtype(network, x)
    w_t = np.ascontiguousarray(w_hf.T, dtype=np.float32)  # [in, out]
    w_const = _add_constant_reduced(network, (in_dim, out_dim), w_t)
    mm = network.add_matrix_multiply(
        x_c,
        trt.MatrixOperation.NONE,
        w_const,
        trt.MatrixOperation.NONE,
    )
    y = mm.get_output(0)
    if b_hf is not None:
        b_const = _add_constant_reduced(network, (1, out_dim), np.asarray(b_hf, dtype=np.float32))
        y = network.add_elementwise(y, b_const, trt.ElementWiseOperation.SUM).get_output(0)
    return y


def _add_layernorm_no_affine_3d(network, x, hidden_size: int, eps: float):
    """LayerNorm over last dim for [B, S, D] tensors, no learnable params.

    Input is cast to the bf16 compute dtype on the way in (so the layer's
    in/out tensors match the surrounding bf16 graph under STRONGLY_TYPED);
    the reduction itself runs in fp32 (``compute_precision=trt.float32``)
    so the LayerNorm mean/var don't lose precision -- matches what
    ``add_layer_norm_native`` does for flux2.
    """
    x_c = _to_compute_dtype(network, x)
    ones = _add_constant_reduced(
        network,
        (1, 1, hidden_size),
        np.ones((1, 1, hidden_size), dtype=np.float32),
    )
    zeros = _add_constant_reduced(
        network,
        (1, 1, hidden_size),
        np.zeros((1, 1, hidden_size), dtype=np.float32),
    )
    norm = network.add_normalization(x_c, ones, zeros, axesMask=1 << 2)
    norm.epsilon = float(eps)
    # TensorRT 11 removed the Python INormalizationLayer.compute_precision
    # setter; TRT 10 emitted a "setComputePrecision ignored for strongly
    # typed network" warning when set on a strongly-typed network. Guard
    # both behaviours so this builder works across 10/11. (Matches the
    # hasattr guard in add_layer_norm_native.)
    if hasattr(norm, "compute_precision"):
        norm.compute_precision = trt.float32
    return norm.get_output(0)


def _add_modulate(network, x_3d, shift_2d, scale_2d, hidden_size: int):
    """x_3d * (1 + scale_2d[:, None, :]) + shift_2d[:, None, :], in bf16.

    x_3d:    [B, S, D]  (already bf16; otherwise cast at boundary)
    shift_2d, scale_2d: [B, D]  (bf16 from the modulation Linear)
    """
    # Reshape to [B, 1, D] for broadcasting over S.
    shift_3d = network.add_shuffle(shift_2d)
    shift_3d.reshape_dims = (-1, 1, hidden_size)
    scale_3d = network.add_shuffle(scale_2d)
    scale_3d.reshape_dims = (-1, 1, hidden_size)

    one_const = _add_constant_reduced(
        network,
        (1, 1, 1),
        np.ones((1, 1, 1), dtype=np.float32),
    )
    one_plus_scale = network.add_elementwise(
        scale_3d.get_output(0), one_const, trt.ElementWiseOperation.SUM
    ).get_output(0)

    x_c = _to_compute_dtype(network, x_3d)
    scaled = network.add_elementwise(x_c, one_plus_scale, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    shifted = network.add_elementwise(
        scaled, shift_3d.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)
    return shifted


def _add_gate_residual(network, residual_3d, gate_2d, branch_3d, hidden_size: int):
    """residual + gate.unsqueeze(1) * branch, in bf16.

    All inputs may originate from mixed dtypes; cast each operand to the
    compute dtype so the elementwise adds remain valid in a STRONGLY_TYPED
    network.
    """
    gate_3d = network.add_shuffle(gate_2d)
    gate_3d.reshape_dims = (-1, 1, hidden_size)
    branch_c = _to_compute_dtype(network, branch_3d)
    residual_c = _to_compute_dtype(network, residual_3d)
    gated = network.add_elementwise(
        gate_3d.get_output(0), branch_c, trt.ElementWiseOperation.PROD
    ).get_output(0)
    out = network.add_elementwise(residual_c, gated, trt.ElementWiseOperation.SUM).get_output(0)
    return out


def _add_rms_norm_per_head(
    network, x_3d, num_heads: int, head_dim: int, gamma: np.ndarray, eps: float, seq_len: int
):
    """RMSNorm over head_dim for [B, S, H*D] with dynamic leading B.

    Delegates to :func:`add_rms_norm_per_head_batched`, which takes
    a [B, S, H*D] tensor with B free, reshapes to [B, S, H, D], reduces over
    head_dim only, and reshapes back -- so the leading batch dim flows
    through without baked-in shape constants.

    The bf16 storage path is selected (``dtype=np.float16``) so the in/out
    tensor dtype stays bf16 while the reduction internally runs in fp32 --
    matches the QK-norm precision pattern of flux2/Qwen-VL.
    """
    eps_t = _add_2d_constant(network, np.array([eps], dtype=np.float32))
    eps_scalar = network.add_shuffle(eps_t)
    eps_scalar.reshape_dims = ()
    return add_rms_norm_per_head_batched(
        network,
        x_3d,
        num_heads,
        head_dim,
        gamma,
        eps_scalar.get_output(0),
        sequence_length=seq_len,
        dtype=np.float16,  # signal "non-fp32 in/out, fp32 compute internally"
    )


def _add_rope_pair(
    network, x_3d, cos_2d, sin_2d, num_heads: int, head_dim: int, seq_len: int, batch_size: int = 1
):
    """Apply Qwen pair-based RoPE under a dynamic leading batch dim.

    x_3d:    [B, S, H*D]
    cos_2d:  [S, D]  (interleaved duplicated: cos[..., 2j] = cos[..., 2j+1])
    sin_2d:  [S, D]  (similar)

    Pair-based rotation:
        x'[..., 2j]   = x[..., 2j]   * cos[..., 2j]   - x[..., 2j+1] * sin[..., 2j]
        x'[..., 2j+1] = x[..., 2j+1] * cos[..., 2j+1] + x[..., 2j]   * sin[..., 2j+1]

    Returns [B, S, H*D].
    """
    # [B, S, H*D] -> [B, S, H, D/2, 2]
    rb = network.add_shuffle(x_3d)
    rb.reshape_dims = (-1, seq_len, num_heads, head_dim // 2, 2)
    x_pairs = rb.get_output(0)

    # x_real = x[..., 0], x_imag = x[..., 1]  (along last axis), each
    # [B, S, H, D/2].
    half = head_dim // 2
    x_real = _slice_batch_complex_part(network, x_pairs, seq_len, num_heads, half, 0, batch_size)
    x_imag = _slice_batch_complex_part(network, x_pairs, seq_len, num_heads, half, 1, batch_size)

    # cos_pair / sin_pair: take every-other element from cos/sin (they are
    # interleaved-duplicated, so cos_pair[..., j] = cos[..., 2j]).
    # RoPE tables are stored as fp32 for numerical sensitivity; cast to the
    # bf16 compute dtype at point-of-use so the complex muls below stay in
    # bf16 (matches flux2_dit_builder.py).
    cos_pair_sl = network.add_slice(
        cos_2d,
        start=(0, 0),
        shape=(seq_len, head_dim // 2),
        stride=(1, 2),
    )
    sin_pair_sl = network.add_slice(
        sin_2d,
        start=(0, 0),
        shape=(seq_len, head_dim // 2),
        stride=(1, 2),
    )
    cos_pair_c = _to_compute_dtype(network, cos_pair_sl.get_output(0))
    sin_pair_c = _to_compute_dtype(network, sin_pair_sl.get_output(0))
    # Reshape cos/sin to [1, S, 1, D/2] for broadcast across batch and heads.
    cos_4d = network.add_shuffle(cos_pair_c)
    cos_4d.reshape_dims = (1, seq_len, 1, head_dim // 2)
    sin_4d = network.add_shuffle(sin_pair_c)
    sin_4d.reshape_dims = (1, seq_len, 1, head_dim // 2)

    # new_real = x_real * cos - x_imag * sin
    r_cos = network.add_elementwise(
        x_real,
        cos_4d.get_output(0),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    i_sin = network.add_elementwise(
        x_imag,
        sin_4d.get_output(0),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    new_real = network.add_elementwise(
        r_cos,
        i_sin,
        trt.ElementWiseOperation.SUB,
    ).get_output(0)

    # new_imag = x_real * sin + x_imag * cos
    r_sin = network.add_elementwise(
        x_real,
        sin_4d.get_output(0),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    i_cos = network.add_elementwise(
        x_imag,
        cos_4d.get_output(0),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    new_imag = network.add_elementwise(
        r_sin,
        i_cos,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)

    # Stack along the last (complex) axis: each operand is [B, S, H, D/2],
    # bring back the singleton axis and concat -> [B, S, H, D/2, 2].
    # axis=4 is the last (complex) dim of the 5D layout — correct under B>1.
    nr_5d = network.add_shuffle(new_real)
    nr_5d.reshape_dims = (-1, seq_len, num_heads, head_dim // 2, 1)
    ni_5d = network.add_shuffle(new_imag)
    ni_5d.reshape_dims = (-1, seq_len, num_heads, head_dim // 2, 1)
    cat = network.add_concatenation([nr_5d.get_output(0), ni_5d.get_output(0)])
    cat.axis = 4

    # [B, S, H, D/2, 2] -> [B, S, H*D]
    flat = network.add_shuffle(cat.get_output(0))
    flat.reshape_dims = (-1, seq_len, num_heads * head_dim)
    return flat.get_output(0)


def _add_joint_attention(
    network,
    q_img_3d,
    k_img_3d,
    v_img_3d,
    q_txt_3d,
    k_txt_3d,
    v_txt_3d,
    num_heads: int,
    head_dim: int,
    n_img: int,
    n_txt: int,
    batch_size: int = 1,
):
    """Joint attention with concat order [txt, img].

    Inputs are [B, S, H*D] for each stream. Returns
    (attn_txt [B, N_txt, H*D], attn_img [B, N_img, H*D]).
    """
    seq_total = n_txt + n_img
    # Concat along the sequence axis (axis=1 of [B, S, D]): correct under
    # any batch size — joining along the token axis, not batch.
    q = network.add_concatenation([q_txt_3d, q_img_3d])
    q.axis = 1
    k = network.add_concatenation([k_txt_3d, k_img_3d])
    k.axis = 1
    v = network.add_concatenation([v_txt_3d, v_img_3d])
    v.axis = 1

    # Reshape [B, S, H*D] -> [B, S, H, D] -> [B, H, S, D]. The leading
    # ``-1`` carries the dynamic batch dim through unchanged.
    def to_4d(x_3d):
        r1 = network.add_shuffle(x_3d)
        r1.reshape_dims = (-1, seq_total, num_heads, head_dim)
        r1.second_transpose = trt.Permutation([0, 2, 1, 3])
        return r1.get_output(0)

    q_4d = to_4d(q.get_output(0))
    k_4d = to_4d(k.get_output(0))
    v_4d = to_4d(v.get_output(0))

    ctx_4d = add_attention_core(
        network,
        q_4d,
        k_4d,
        v_4d,
        causal=False,
        mask=None,
        scale=float(1.0 / math.sqrt(head_dim)),
    )

    # [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
    out_shuffle = network.add_shuffle(ctx_4d)
    out_shuffle.first_transpose = trt.Permutation([0, 2, 1, 3])
    out_shuffle.reshape_dims = (-1, seq_total, num_heads * head_dim)
    out_3d = out_shuffle.get_output(0)

    # Split back into [B, N_txt, H*D] and [B, N_img, H*D] via the
    # dynamic-batch-aware slice helper.
    txt = _slice_batch_sequence(network, out_3d, 0, n_txt, num_heads * head_dim, batch_size)
    img = _slice_batch_sequence(network, out_3d, n_txt, n_img, num_heads * head_dim, batch_size)
    return txt, img


def _add_mlp_block(
    network,
    x_3d,
    hidden_size: int,
    intermediate_size: int,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    seq_len: int,
    batch_size: int = 1,
):
    """FeedForward block: Linear -> GELU(tanh) -> Linear.

    Diffusers FeedForward (gelu-approximate, mult=4, bias=True):
        net.0 = GELU(approximate="tanh"): Linear(dim, inner) followed by GELU
                in forward — but ``self.net[0]`` is the ``GELU`` module so the
                weight key is ``net.0.proj.weight``.
        net.1 = Dropout (skipped at inference)
        net.2 = Linear(inner, dim)

    Uses :func:`_add_linear_3d` so the leading batch dim flows through
    dynamically — no flatten that would bake a static ``[B*S, in]`` shape.
    """
    w1 = np.asarray(weights[f"{prefix}.net.0.proj.weight"], dtype=np.float32)  # [inner, hidden]
    b1 = np.asarray(weights[f"{prefix}.net.0.proj.bias"], dtype=np.float32)
    w2 = np.asarray(weights[f"{prefix}.net.2.weight"], dtype=np.float32)  # [hidden, inner]
    b2 = np.asarray(weights[f"{prefix}.net.2.bias"], dtype=np.float32)

    h = _add_linear_3d(
        network,
        x_3d,
        hidden_size,
        intermediate_size,
        w1,
        b1,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    h = add_gelu_tanh(network, h)
    return _add_linear_3d(
        network,
        h,
        intermediate_size,
        hidden_size,
        w2,
        b2,
        seq_len=seq_len,
        batch_size=batch_size,
    )


def _add_qkv_with_norm_and_rope(
    network,
    x_3d,
    weights: Mapping[str, np.ndarray],
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    q_key: str,
    k_key: str,
    v_key: str,
    norm_q_key: str | None,
    norm_k_key: str | None,
    rms_eps: float,
    cos_2d,
    sin_2d,
    batch_size: int = 1,
):
    """Compute QKV for one stream with q-norm/k-norm + RoPE.

    Inputs are [B, S, D]. Returns (q_3d, k_3d, v_3d) each [B, S, H*D].
    Q and K get q-norm/k-norm + RoPE applied. V is passed through.

    Uses :func:`_add_linear_3d` so the leading batch dim is dynamic.
    """

    def _proj(weight_key: str):
        w_hf = np.asarray(weights[f"{weight_key}.weight"], dtype=np.float32)
        b_hf = weights.get(f"{weight_key}.bias")
        b = np.asarray(b_hf, dtype=np.float32) if b_hf is not None else None
        return _add_linear_3d(
            network,
            x_3d,
            hidden_size,
            hidden_size,
            w_hf,
            b,
            seq_len=seq_len,
            batch_size=batch_size,
        )

    q_3d = _proj(q_key)
    k_3d = _proj(k_key)
    v_3d = _proj(v_key)

    # qk-norm: RMSNorm over head_dim with weight [head_dim].
    if norm_q_key is not None and f"{norm_q_key}.weight" in weights:
        gamma = np.asarray(weights[f"{norm_q_key}.weight"], dtype=np.float32)
        q_3d = _add_rms_norm_per_head(
            network,
            q_3d,
            num_heads,
            head_dim,
            gamma,
            rms_eps,
            seq_len,
        )
    if norm_k_key is not None and f"{norm_k_key}.weight" in weights:
        gamma = np.asarray(weights[f"{norm_k_key}.weight"], dtype=np.float32)
        k_3d = _add_rms_norm_per_head(
            network,
            k_3d,
            num_heads,
            head_dim,
            gamma,
            rms_eps,
            seq_len,
        )

    # RoPE on Q and K (V untouched).
    q_3d = _add_rope_pair(network, q_3d, cos_2d, sin_2d, num_heads, head_dim, seq_len, batch_size)
    k_3d = _add_rope_pair(network, k_3d, cos_2d, sin_2d, num_heads, head_dim, seq_len, batch_size)
    return q_3d, k_3d, v_3d


# ---------------------------------------------------------------------------
# Reusable joint-block graph helper.
#
# This is the inlined math for one ``QwenImageTransformerBlock``. Both
# ``build_joint_block_engine`` (single-block engine, used by the HF parity
# test) and ``build_qwen_image_dit_engine`` (full denoiser) call this.
# Single source of truth keeps the per-block math from drifting between
# the two builders.
#
# Inputs are pre-built ITensors (the caller is responsible for adding the
# network inputs / constants); the helper returns ``(img_out_3d, txt_out_3d)``.
# ---------------------------------------------------------------------------


def _add_joint_block_graph(
    network,
    *,
    img_3d,  # [B, n_img, hidden]
    txt_3d,  # [B, n_text, hidden]
    temb_2d,  # [B, hidden]
    cos_img,  # [n_img, head_dim]
    sin_img,  # [n_img, head_dim]
    cos_txt,  # [n_text, head_dim]
    sin_txt,  # [n_text, head_dim]
    weights: Mapping[str, "np.ndarray"],
    weights_prefix: str = "",
    cfg: "JointBlockConfig",
    n_img: int,
    n_text: int,
    batch_size: int = 1,
):
    """Build the math of one Qwen-Image MMDiT joint block.

    ``weights_prefix`` is prepended to every state-dict key lookup, so the
    full-denoiser builder can pass ``"transformer_blocks.0."`` and get the
    same block-local keys (``img_mod.1.weight``, etc.).

    Returns ``(img_out_3d, txt_out_3d)``, both [B, S_*, hidden].

    ``batch_size`` is the static batch (``1``) or sentinel for dynamic
    (``>1``): the dim is treated as dynamic when ``batch_size > 1``, in
    which case slice helpers build a runtime shape tensor from input dim 0.
    """
    H = cfg.num_attention_heads
    D = cfg.attention_head_dim
    dim = cfg.hidden_size

    def _w(key: str) -> np.ndarray:
        return np.asarray(weights[f"{weights_prefix}{key}"], dtype=np.float32)

    def _w_opt(key: str):
        v = weights.get(f"{weights_prefix}{key}")
        return None if v is None else np.asarray(v, dtype=np.float32)

    # Wrap weights so qkv-helper can use ``weights[key]`` directly with the
    # prefix already baked in.
    class _PrefixedWeights:
        def __init__(self, base: Mapping[str, np.ndarray], prefix: str):
            self._base = base
            self._prefix = prefix

        def __getitem__(self, key: str):
            return self._base[f"{self._prefix}{key}"]

        def get(self, key: str, default=None):
            return self._base.get(f"{self._prefix}{key}", default)

        def __contains__(self, key: str) -> bool:
            return f"{self._prefix}{key}" in self._base

    prefixed = _PrefixedWeights(weights, weights_prefix)

    # ----- AdaLN modulation: SiLU(temb) -> Linear -> 6*dim, chunk 2 then 3.
    img_mod_w = _w("img_mod.1.weight")
    img_mod_b = _w("img_mod.1.bias")
    txt_mod_w = _w("txt_mod.1.weight")
    txt_mod_b = _w("txt_mod.1.bias")

    temb_silu = add_silu(network, temb_2d)
    img_mod_params = _add_linear_2d(
        network,
        temb_silu,
        dim,
        6 * dim,
        img_mod_w,
        img_mod_b,
    )
    txt_mod_params = _add_linear_2d(
        network,
        temb_silu,
        dim,
        6 * dim,
        txt_mod_w,
        txt_mod_b,
    )

    def _six_chunks(mod_params):
        # ``mod_params`` is [B, 6*dim]; slice 6 disjoint [B, dim] chunks via
        # the dynamic-batch-aware helper (no baked leading dim).
        chunks = []
        for i in range(6):
            chunks.append(_slice_batch_vector(network, mod_params, i * dim, dim, batch_size))
        return chunks

    img_shift_msa, img_scale_msa, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = (
        _six_chunks(img_mod_params)
    )
    txt_shift_msa, txt_scale_msa, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = (
        _six_chunks(txt_mod_params)
    )

    # ----- norm1 + modulate per stream.
    img_normed = _add_layernorm_no_affine_3d(network, img_3d, dim, cfg.layer_norm_eps)
    img_modulated = _add_modulate(network, img_normed, img_shift_msa, img_scale_msa, dim)
    txt_normed = _add_layernorm_no_affine_3d(network, txt_3d, dim, cfg.layer_norm_eps)
    txt_modulated = _add_modulate(network, txt_normed, txt_shift_msa, txt_scale_msa, dim)

    # ----- QKV + qk-norm + RoPE.
    img_q, img_k, img_v = _add_qkv_with_norm_and_rope(
        network,
        img_modulated,
        prefixed,
        hidden_size=dim,
        num_heads=H,
        head_dim=D,
        seq_len=n_img,
        q_key="attn.to_q",
        k_key="attn.to_k",
        v_key="attn.to_v",
        norm_q_key="attn.norm_q",
        norm_k_key="attn.norm_k",
        rms_eps=cfg.rms_norm_eps,
        cos_2d=cos_img,
        sin_2d=sin_img,
        batch_size=batch_size,
    )
    txt_q, txt_k, txt_v = _add_qkv_with_norm_and_rope(
        network,
        txt_modulated,
        prefixed,
        hidden_size=dim,
        num_heads=H,
        head_dim=D,
        seq_len=n_text,
        q_key="attn.add_q_proj",
        k_key="attn.add_k_proj",
        v_key="attn.add_v_proj",
        norm_q_key="attn.norm_added_q",
        norm_k_key="attn.norm_added_k",
        rms_eps=cfg.rms_norm_eps,
        cos_2d=cos_txt,
        sin_2d=sin_txt,
        batch_size=batch_size,
    )

    # ----- joint attention; concat order [txt, img].
    attn_txt_3d, attn_img_3d = _add_joint_attention(
        network,
        img_q,
        img_k,
        img_v,
        txt_q,
        txt_k,
        txt_v,
        num_heads=H,
        head_dim=D,
        n_img=n_img,
        n_txt=n_text,
        batch_size=batch_size,
    )

    # ----- output projections (separate for each stream). Use the rank-3
    # matmul helper so the leading batch dim flows through dynamically.
    def _out_proj(x_3d, key_suffix: str, seq_len: int):
        w = _w(f"{key_suffix}.weight")
        b = _w_opt(f"{key_suffix}.bias")
        return _add_linear_3d(
            network,
            x_3d,
            dim,
            dim,
            w,
            b,
            seq_len=seq_len,
            batch_size=batch_size,
        )

    img_attn_out = _out_proj(attn_img_3d, "attn.to_out.0", n_img)
    txt_attn_out = _out_proj(attn_txt_3d, "attn.to_add_out", n_text)

    # ----- gated residual (post-attention).
    hs_img = _add_gate_residual(network, img_3d, img_gate_msa, img_attn_out, dim)
    hs_txt = _add_gate_residual(network, txt_3d, txt_gate_msa, txt_attn_out, dim)

    # ----- norm2 + modulate(mod2) + MLP + gated residual.
    img_n2 = _add_layernorm_no_affine_3d(network, hs_img, dim, cfg.layer_norm_eps)
    img_mod2_out = _add_modulate(network, img_n2, img_shift_mlp, img_scale_mlp, dim)
    img_mlp_out = _add_mlp_block(
        network,
        img_mod2_out,
        dim,
        cfg.intermediate_size,
        prefixed,
        "img_mlp",
        n_img,
        batch_size=batch_size,
    )
    img_out = _add_gate_residual(network, hs_img, img_gate_mlp, img_mlp_out, dim)

    txt_n2 = _add_layernorm_no_affine_3d(network, hs_txt, dim, cfg.layer_norm_eps)
    txt_mod2_out = _add_modulate(network, txt_n2, txt_shift_mlp, txt_scale_mlp, dim)
    txt_mlp_out = _add_mlp_block(
        network,
        txt_mod2_out,
        dim,
        cfg.intermediate_size,
        prefixed,
        "txt_mlp",
        n_text,
        batch_size=batch_size,
    )
    txt_out = _add_gate_residual(network, hs_txt, txt_gate_mlp, txt_mlp_out, dim)
    return img_out, txt_out


# ---------------------------------------------------------------------------
# Full MMDiT denoiser stack.
#
# Stacks 60 joint blocks plus the surrounding scaffolding from
# ``diffusers.models.transformers.transformer_qwenimage.QwenImageTransformer2DModel.forward``
# (lines 879-957 in the diffusers checkout) into one TRT engine:
#
#     hidden_states     = self.img_in(hidden_states)         # Linear [in_ch -> hidden]
#     encoder_hs        = self.txt_norm(encoder_hidden_states)  # RMSNorm over text_embed_dim
#     encoder_hs        = self.txt_in(encoder_hs)            # Linear [text_d -> hidden]
#     temb              = self.time_text_embed(timestep, ...)  # sinusoidal + MLP
#     image_rotary_emb  = self.pos_embed(img_shapes, max_txt_seq_len, device)
#     for block in self.transformer_blocks:
#         encoder_hs, hidden_states = block(hidden_states, encoder_hs, ...)
#     hidden_states     = self.norm_out(hidden_states, temb)  # AdaLayerNormContinuous
#     output            = self.proj_out(hidden_states)        # Linear [hidden -> p*p*out_ch]
#
# AdaLayerNormContinuous (diffusers/models/normalization.py:307-351):
#     emb         = self.linear(self.silu(temb))             # [B, 2*hidden]
#     scale, shift = torch.chunk(emb, 2, dim=1)              # scale FIRST, then shift
#     x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
# elementwise_affine=False -> the inner LayerNorm has no learnable params.
#
# Trace IDs: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01.
# ---------------------------------------------------------------------------


@dataclass
class QwenImageDiTConfig:
    """Configuration for the full Qwen-Image MMDiT denoiser.

    Defaults match ``Qwen/Qwen-Image-2512`` (60 joint blocks, hidden=3072,
    24 heads, head_dim=128, rope_axes_dim=[16, 56, 56], rope_theta=10000).
    """

    in_channels: int = 64
    out_channels: int = 16
    patch_size: int = 2
    hidden_size: int = 3072
    num_joint_blocks: int = 60
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    intermediate_size: int = 12288  # FeedForward(dim, mult=4) -> 4 * hidden.
    text_embed_dim: int = 3584  # joint_attention_dim (Qwen2.5-VL hidden).
    rope_axes_dim: list[int] = field(default_factory=lambda: [16, 56, 56])
    rope_theta: float = 10000.0
    timestep_embed_dim: int = 256  # sinusoidal embed width pre-MLP.
    rms_norm_eps: float = 1e-6
    layer_norm_eps: float = 1e-6
    max_image_tokens: int = 8192
    max_text_tokens: int = 1024
    guidance_embeds: bool = False  # Qwen-Image-2512: False.


def _joint_block_cfg_from(cfg: QwenImageDiTConfig) -> JointBlockConfig:
    """Adapter so the joint-block helper can be called with its own config."""
    return JointBlockConfig(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        attention_head_dim=cfg.attention_head_dim,
        intermediate_size=cfg.intermediate_size,
        rms_norm_eps=cfg.rms_norm_eps,
        layer_norm_eps=cfg.layer_norm_eps,
    )


def _add_linear_3d(
    network,
    x_3d,
    in_dim: int,
    out_dim: int,
    w_hf: np.ndarray,
    b_hf: "np.ndarray | None",
    seq_len: int,
    batch_size: int = 1,
):
    """Linear on a [B, S, in_dim] tensor -> [B, S, out_dim].

    Uses a rank-3 matmul so the leading batch dimension can be dynamic
    (B = -1). The weight is broadcast as ``[1, in_dim, out_dim]`` and TRT's
    matmul handles the implicit broadcast along axis 0. ``seq_len`` and
    ``batch_size`` are kept for API parity with the older flatten-based
    helper; neither participates in the lowered graph anymore.
    """
    # Suppress unused-arg lint warnings — kept for API parity.
    _ = seq_len
    _ = batch_size
    x_c = _to_compute_dtype(network, x_3d)
    w_t = np.ascontiguousarray(w_hf.T, dtype=np.float32)  # [in, out]
    w_const = _add_constant_reduced(network, (1, in_dim, out_dim), w_t.reshape(1, in_dim, out_dim))
    mm = network.add_matrix_multiply(
        x_c,
        trt.MatrixOperation.NONE,
        w_const,
        trt.MatrixOperation.NONE,
    )
    y = mm.get_output(0)
    if b_hf is not None:
        b_const = _add_constant_reduced(
            network,
            (1, 1, out_dim),
            np.asarray(b_hf, dtype=np.float32).reshape(1, 1, out_dim),
        )
        y = network.add_elementwise(y, b_const, trt.ElementWiseOperation.SUM).get_output(0)
    return y


def _add_rms_norm_last_dim_3d(network, x_3d, hidden_size: int, gamma: np.ndarray, eps: float):
    """RMSNorm over the last axis of a [B, S, D] tensor.

    Implements ``gamma * x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps)``,
    matching ``diffusers.models.normalization.RMSNorm`` with
    ``elementwise_affine=True`` and no bias. ``gamma`` shape: [hidden_size].

    We don't reuse :func:`add_rms_norm` because that helper is
    hard-coded for 2D [N, D] inputs (reduces over axis 1).
    """
    sq = network.add_elementwise(x_3d, x_3d, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0),
        trt.ReduceOperation.AVG,
        1 << 2,
        keep_dims=True,
    )
    eps_const = _add_2d_constant(
        network,
        np.array([[[eps]]], dtype=np.float32),
    )
    denom_in = network.add_elementwise(
        mean.get_output(0),
        eps_const,
        trt.ElementWiseOperation.SUM,
    )
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        x_3d,
        recip.get_output(0),
        trt.ElementWiseOperation.PROD,
    )
    gamma_const = _add_2d_constant(
        network,
        np.asarray(gamma, dtype=np.float32).reshape(1, 1, hidden_size),
    )
    scaled = network.add_elementwise(
        normalized.get_output(0),
        gamma_const,
        trt.ElementWiseOperation.PROD,
    )
    return scaled.get_output(0)


def _add_time_text_embed(
    network,
    timestep,
    *,
    weights: Mapping[str, np.ndarray],
    in_dim: int,
    hidden_size: int,
):
    """In-network port of ``QwenTimestepProjEmbeddings`` (the no-guidance path).

    timestep [B] -> sinusoidal [B, in_dim] -> linear_1 [B, hidden]
                 -> SiLU -> linear_2 [B, hidden]
    """
    sample = _add_get_timestep_embedding(
        network,
        timestep,
        embedding_dim=in_dim,
        flip_sin_to_cos=True,
        downscale_freq_shift=0.0,
        scale=1000.0,
        max_period=10000.0,
    )
    w1 = np.asarray(
        weights["time_text_embed.timestep_embedder.linear_1.weight"],
        dtype=np.float32,
    )
    b1 = np.asarray(
        weights["time_text_embed.timestep_embedder.linear_1.bias"],
        dtype=np.float32,
    )
    w2 = np.asarray(
        weights["time_text_embed.timestep_embedder.linear_2.weight"],
        dtype=np.float32,
    )
    b2 = np.asarray(
        weights["time_text_embed.timestep_embedder.linear_2.bias"],
        dtype=np.float32,
    )
    h = _add_linear(network, sample, w1, b1)
    h = add_silu(network, h)
    h = _add_linear(network, h, w2, b2)
    return h  # [B, hidden]


def _add_norm_out_3d(
    network,
    x_3d,
    temb_2d,
    *,
    weights: Mapping[str, np.ndarray],
    hidden_size: int,
    eps: float,
    batch_size: int,
):
    """Apply ``AdaLayerNormContinuous(elementwise_affine=False)`` for [B, S, D].

    Matches diffusers.models.normalization.AdaLayerNormContinuous.forward
    (normalization.py:346-351):
        emb         = self.linear(self.silu(temb))         # [B, 2*D]
        scale, shift = torch.chunk(emb, 2, dim=1)          # scale FIRST.
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
    where ``self.norm`` is ``LayerNorm(D, eps, elementwise_affine=False)``.
    """
    w = np.asarray(weights["norm_out.linear.weight"], dtype=np.float32)  # [2*D, D]
    b = np.asarray(weights["norm_out.linear.bias"], dtype=np.float32)  # [2*D]
    temb_silu = add_silu(network, temb_2d)
    emb = _add_linear_2d(network, temb_silu, hidden_size, 2 * hidden_size, w, b)
    # chunk(2, dim=1) -> (scale, shift).  diffusers convention is scale-first.
    # Use the dynamic-batch-aware slice helper so a B=-1 ``emb`` survives.
    scale = _slice_batch_vector(network, emb, 0, hidden_size, batch_size)
    shift = _slice_batch_vector(network, emb, hidden_size, hidden_size, batch_size)

    # LayerNorm(x), no affine.
    x_normed = _add_layernorm_no_affine_3d(network, x_3d, hidden_size, eps)
    # (1 + scale)[:, None, :] and shift[:, None, :]
    scale_3d = network.add_shuffle(scale)
    scale_3d.reshape_dims = (-1, 1, hidden_size)
    shift_3d = network.add_shuffle(shift)
    shift_3d.reshape_dims = (-1, 1, hidden_size)
    # bf16 constant so it matches scale_3d's dtype under STRONGLY_TYPED.
    one_const = _add_constant_reduced(
        network,
        (1, 1, 1),
        np.ones((1, 1, 1), dtype=np.float32),
    )
    one_plus_scale = network.add_elementwise(
        scale_3d.get_output(0),
        one_const,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    scaled = network.add_elementwise(
        x_normed,
        one_plus_scale,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    shifted = network.add_elementwise(
        scaled,
        shift_3d.get_output(0),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    return shifted


def _validate_full_weights(
    cfg: QwenImageDiTConfig,
    weights: Mapping[str, np.ndarray],
) -> None:
    """Lightweight schema check on the weight dict.

    Only checks presence (and shapes for the top-level params). Per-block
    keys are checked by the joint-block helper at build time.
    """
    H = cfg.hidden_size
    in_ch = cfg.in_channels
    out_ch = cfg.out_channels
    p = cfg.patch_size
    txt_d = cfg.text_embed_dim

    required = {
        "img_in.weight": (H, in_ch),
        "img_in.bias": (H,),
        "txt_norm.weight": (txt_d,),
        "txt_in.weight": (H, txt_d),
        "txt_in.bias": (H,),
        "time_text_embed.timestep_embedder.linear_1.weight": (H, cfg.timestep_embed_dim),
        "time_text_embed.timestep_embedder.linear_1.bias": (H,),
        "time_text_embed.timestep_embedder.linear_2.weight": (H, H),
        "time_text_embed.timestep_embedder.linear_2.bias": (H,),
        "norm_out.linear.weight": (2 * H, H),
        "norm_out.linear.bias": (2 * H,),
        "proj_out.weight": (out_ch * p * p, H),
        "proj_out.bias": (out_ch * p * p,),
    }
    for key, want in required.items():
        if key not in weights:
            raise KeyError(f"build_qwen_image_dit_engine: missing weight {key!r}")
        arr = np.asarray(weights[key])
        if tuple(arr.shape) != tuple(want):
            raise ValueError(f"build_qwen_image_dit_engine: {key!r} shape {arr.shape} != {want}")


def build_qwen_image_dit_engine(
    cfg: QwenImageDiTConfig,
    weights: Mapping[str, "np.ndarray"],
    out_path: PathLike,
    *,
    h_lat: int,
    w_lat: int,
    n_text: int,
    image_token_shapes: list[tuple[int, int]] | None = None,
    batch_size: int = 1,
    max_batch_size: int = 1,
    opt_batch_size: int | None = None,
    verbose: bool = False,
) -> Path:
    """Build the full Qwen-Image MMDiT denoiser as a single TRT engine.

    The denoiser stacks ``cfg.num_joint_blocks`` ``QwenImageTransformerBlock``
    instances plus the surrounding scaffolding (``img_in``, ``txt_norm``,
    ``txt_in``, ``time_text_embed``, final ``AdaLayerNormContinuous``,
    ``proj_out``).

    Engine inputs (fp32):
      ``img_patched`` [B, N_img, in_channels]
          packed-patch latents (already produced by the patchify engine /
          diffusers pack_latents). For Qwen-Image: in_channels = 64.
          T2I uses ``N_img = h_lat * w_lat``. Edit may pass
          ``image_token_shapes=[(gen_h, gen_w), (cond_h, cond_w), ...]``;
          then ``N_img`` is the sum of those packed-token grids, matching
          diffusers' ``torch.cat([latents, image_latents], dim=1)`` input.
      ``txt_hidden`` [B, N_txt = n_text, text_embed_dim]
          text encoder hidden states (Qwen2.5-VL hidden = 3584).
      ``timestep``  [B]
          diffusion timestep (fp32; will be cast/multiplied by ``scale=1000``
          inside the sinusoidal embedding, matching diffusers).

    When ``max_batch_size == 1`` every input has a statically baked leading
    dim of 1 (byte-for-byte identical to the pre-batch build). When
    ``max_batch_size > 1`` the leading dim is replaced with ``-1`` and a
    TensorRT optimization profile attaches with
    ``kMIN=1, kOPT=opt_batch_size, kMAX=max_batch_size`` per design Decisions
    A and C (2026-05-19).

    Engine output (fp32):
      ``noise_patched`` [B, N_img, out_channels * patch_size**2]
          packed-patch noise prediction. The runtime unpatchifies this
          before handing it to the VAE decoder.

    ``h_lat`` and ``w_lat`` describe the packed-token grid (= latent-grid
    divided by ``cfg.patch_size``). They bake into the engine via the RoPE
    tables, so callers wanting a different grid must build a fresh engine.
    For Edit, the first ``image_token_shapes`` entry must match
    ``(h_lat, w_lat)`` and represents the generated-image tokens; subsequent
    entries represent condition-image VAE latents.

    Real Qwen-Image-2512 weights load via the diffusers state-dict keys
    (no remapping); see ``_validate_full_weights`` for the full key list.

    Args:
        batch_size: Legacy static-batch knob retained for backwards
            compatibility with existing callers that drove a fixed B>1
            engine. Prefer ``max_batch_size`` for the dynamic-profile path.
            When both default to 1 the build is byte-for-byte identical to
            today's behavior.
        max_batch_size: Maximum DiT batch the engine should accept. Drives
            the dynamic optimization profile when ``>1`` (Decisions A and C).
            Per Decision C the Qwen-Image DiT cap is 4.
        opt_batch_size: ``kOPT`` for the dynamic batch dim. Ignored when
            ``max_batch_size==1``. Defaults to ``min(max_batch_size, 4)``.

    Returns the path to the written serialised plan.
    """
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1 (got {max_batch_size})")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1 (got {batch_size})")
    if cfg.num_attention_heads * cfg.attention_head_dim != cfg.hidden_size:
        raise ValueError(
            f"hidden_size ({cfg.hidden_size}) != num_heads ({cfg.num_attention_heads}) "
            f"* head_dim ({cfg.attention_head_dim})"
        )
    if sum(cfg.rope_axes_dim) != cfg.attention_head_dim:
        raise ValueError(
            f"sum(rope_axes_dim) ({sum(cfg.rope_axes_dim)}) != head_dim ({cfg.attention_head_dim})"
        )
    if cfg.guidance_embeds:
        raise NotImplementedError(
            "build_qwen_image_dit_engine does not support guidance_embeds=True"
        )

    _validate_full_weights(cfg, weights)

    if image_token_shapes is None:
        image_token_shapes = [(h_lat, w_lat)]
    else:
        image_token_shapes = [(int(h), int(w)) for h, w in image_token_shapes]
        if not image_token_shapes:
            raise ValueError("image_token_shapes must not be empty")
        if image_token_shapes[0] != (h_lat, w_lat):
            raise ValueError(
                "image_token_shapes[0] must match (h_lat, w_lat); got "
                f"{image_token_shapes[0]!r} vs {(h_lat, w_lat)!r}"
            )
    n_img = sum(h * w for h, w in image_token_shapes)
    H_dim = cfg.hidden_size
    head_dim = cfg.attention_head_dim
    in_ch = cfg.in_channels
    out_ch = cfg.out_channels
    p = cfg.patch_size
    txt_d = cfg.text_embed_dim

    # Pre-compute RoPE tables (baked as constants).
    cos_table_np, sin_table_np = _precompute_qwen_rope_tables_for_shapes(
        list(cfg.rope_axes_dim),
        image_token_shapes,
        n_text,
        cfg.rope_theta,
    )
    seq_total = n_img + n_text
    assert cos_table_np.shape == (seq_total, head_dim), (
        f"rope cos shape {cos_table_np.shape} != ({seq_total}, {head_dim})"
    )

    builder, config, network = _make_builder(verbose)

    # When ``max_batch_size > 1`` we expose a dynamic leading batch dim via a
    # TensorRT optimization profile (design Decision A: one wide profile per
    # component). Otherwise the engine remains statically batched at
    # ``batch_size`` (today's behavior). The internal compute graph still
    # uses the static ``batch_size`` everywhere it bakes a shape — full
    # dynamic-batch correctness for the inner blocks is a Phase 1.5 follow-up
    # (see RFC 2026-05-11). The profile call below is the source of truth
    # for the engine's batch envelope.
    use_dynamic_batch = max_batch_size > 1
    input_batch = -1 if use_dynamic_batch else batch_size

    img_patched = network.add_input(
        "img_patched",
        trt.float32,
        (input_batch, n_img, in_ch),
    )
    txt_hidden = network.add_input(
        "txt_hidden",
        trt.float32,
        (input_batch, n_text, txt_d),
    )
    timestep = network.add_input("timestep", trt.float32, (input_batch,))

    if use_dynamic_batch:
        from ....engine_builder import add_dynamic_batch_profile

        opt_batch = min(max_batch_size, 4) if opt_batch_size is None else opt_batch_size
        add_dynamic_batch_profile(
            builder,
            config,
            network,
            input_names=["img_patched", "txt_hidden", "timestep"],
            max_batch=max_batch_size,
            opt_batch=opt_batch,
            static_shape={
                "img_patched": (n_img, in_ch),
                "txt_hidden": (n_text, txt_d),
                "timestep": (),
            },
        )

    # Sentinel for the inner-block slice helpers: when ``use_dynamic_batch``
    # is true the leading dim is ``-1`` at runtime, so we pass
    # ``max_batch_size`` (which is >1 by definition) so they take the
    # dynamic-slice path; otherwise leave at the legacy ``batch_size``.
    inner_batch_size = max_batch_size if use_dynamic_batch else batch_size

    # ----- img_in: Linear(in_ch -> hidden) over [B, N_img, in_ch].
    img_in_w = np.asarray(weights["img_in.weight"], dtype=np.float32)
    img_in_b = np.asarray(weights["img_in.bias"], dtype=np.float32)
    img_tokens = _add_linear_3d(
        network,
        img_patched,
        in_ch,
        H_dim,
        img_in_w,
        img_in_b,
        seq_len=n_img,
        batch_size=inner_batch_size,
    )

    # ----- txt_norm: RMSNorm over text_embed_dim, then txt_in: Linear(text_d -> hidden).
    txt_norm_gamma = np.asarray(weights["txt_norm.weight"], dtype=np.float32)
    txt_normed = _add_rms_norm_last_dim_3d(
        network,
        txt_hidden,
        txt_d,
        txt_norm_gamma,
        cfg.rms_norm_eps,
    )
    txt_in_w = np.asarray(weights["txt_in.weight"], dtype=np.float32)
    txt_in_b = np.asarray(weights["txt_in.bias"], dtype=np.float32)
    txt_tokens = _add_linear_3d(
        network,
        txt_normed,
        txt_d,
        H_dim,
        txt_in_w,
        txt_in_b,
        seq_len=n_text,
        batch_size=inner_batch_size,
    )

    # ----- time_text_embed: sinusoidal + MLP -> [B, hidden].
    temb = _add_time_text_embed(
        network,
        timestep,
        weights=weights,
        in_dim=cfg.timestep_embed_dim,
        hidden_size=H_dim,
    )

    # ----- RoPE cos/sin constants split into img and txt sub-tables.
    # _precompute_qwen_rope_tables returns [vid_freqs (n_img rows) | txt_freqs (n_text rows)].
    cos_const = network.add_constant(
        (seq_total, head_dim),
        trt.Weights(cos_table_np),
    ).get_output(0)
    sin_const = network.add_constant(
        (seq_total, head_dim),
        trt.Weights(sin_table_np),
    ).get_output(0)
    cos_img_sl = network.add_slice(cos_const, start=(0, 0), shape=(n_img, head_dim), stride=(1, 1))
    sin_img_sl = network.add_slice(sin_const, start=(0, 0), shape=(n_img, head_dim), stride=(1, 1))
    cos_txt_sl = network.add_slice(
        cos_const, start=(n_img, 0), shape=(n_text, head_dim), stride=(1, 1)
    )
    sin_txt_sl = network.add_slice(
        sin_const, start=(n_img, 0), shape=(n_text, head_dim), stride=(1, 1)
    )

    # ----- Joint blocks loop.
    jb_cfg = _joint_block_cfg_from(cfg)
    cur_img = img_tokens
    cur_txt = txt_tokens
    for i in range(cfg.num_joint_blocks):
        prefix = f"transformer_blocks.{i}."
        cur_img, cur_txt = _add_joint_block_graph(
            network,
            img_3d=cur_img,
            txt_3d=cur_txt,
            temb_2d=temb,
            cos_img=cos_img_sl.get_output(0),
            sin_img=sin_img_sl.get_output(0),
            cos_txt=cos_txt_sl.get_output(0),
            sin_txt=sin_txt_sl.get_output(0),
            weights=weights,
            weights_prefix=prefix,
            cfg=jb_cfg,
            n_img=n_img,
            n_text=n_text,
            batch_size=inner_batch_size,
        )

    # ----- AdaLayerNormContinuous(elementwise_affine=False) -> proj_out.
    normed = _add_norm_out_3d(
        network,
        cur_img,
        temb,
        weights=weights,
        hidden_size=H_dim,
        eps=cfg.layer_norm_eps,
        batch_size=inner_batch_size,
    )
    proj_w = np.asarray(weights["proj_out.weight"], dtype=np.float32)
    proj_b = np.asarray(weights["proj_out.bias"], dtype=np.float32)
    proj_out_dim = out_ch * p * p
    noise = _add_linear_3d(
        network,
        normed,
        H_dim,
        proj_out_dim,
        proj_w,
        proj_b,
        seq_len=n_img,
        batch_size=inner_batch_size,
    )
    # Engine output is fp32 by contract (the C++ runtime + Python debug
    # runner bind fp32 host buffers); internal compute stays bf16.
    noise = _to_fp32(network, noise)
    noise.name = "noise_patched"
    network.mark_output(noise)

    b_label = f"1..{max_batch_size}" if use_dynamic_batch else str(batch_size)
    print(
        f"[qwen-image-dit] Building full denoiser engine "
        f"(B={b_label}, n_img={n_img}, n_text={n_text}, "
        f"image_token_shapes={image_token_shapes}, "
        f"blocks={cfg.num_joint_blocks}, hidden={H_dim}, "
        f"heads={cfg.num_attention_heads}, head_dim={head_dim}, "
        f"text_d={txt_d}, in_ch={in_ch}, out_ch={out_ch}, p={p}) "
        f"-> [{b_label}, {n_img}, {proj_out_dim}]",
        file=sys.stderr,
    )
    return _serialize_and_write(builder, network, config, out_path, "qwen_image_dit")


# ---------------------------------------------------------------------------
# Full-denoiser HF weight loader.
#
# Loads every parametric tensor from a diffusers Qwen-Image-2512 transformer/
# directory and packages it together with a QwenImageDiTConfig built from the
# component config.json. The returned dict's keys are the diffusers
# state-dict keys exactly (no renaming) so it can be passed straight into
# build_qwen_image_dit_engine -- which uses _validate_full_weights to assert
# the schema matches the diffusers convention.
# ---------------------------------------------------------------------------


def load_qwen_image_dit_weights(
    transformer_dir: PathLike,
) -> tuple[QwenImageDiTConfig, dict[str, np.ndarray]]:
    """Load full ``QwenImageTransformer2DModel`` weights from a HF ``transformer/`` dir.

    Reads ``transformer/config.json`` to populate :class:`QwenImageDiTConfig`,
    then scans all ``*.safetensors`` shards in the directory and returns
    every tensor as an fp32 ``np.ndarray``. The returned ``cfg`` and
    ``weights`` dict are suitable for direct use with
    :func:`build_qwen_image_dit_engine` — no key remapping is needed
    because the engine builder uses the diffusers state-dict naming
    convention throughout (see :func:`_validate_full_weights`).

    The full transformer dict for Qwen/Qwen-Image-2512 has 1933 parametric
    tensors (verified by listing the safetensors index).
    Returns ALL of them — the full denoiser uses everything (img_in,
    txt_norm, txt_in, time_text_embed, transformer_blocks.*, norm_out,
    proj_out). No filtering is applied because the transformer directory
    only contains the denoiser; sibling components (text encoder, VAE) live
    in their own per-component directories under the diffusers snapshot.

    The Qwen-Image-2512 ``config.json`` is minimal (no ``rope_theta``,
    ``rms_norm_eps``, etc. fields exist); this loader uses the same
    hardcoded defaults that the diffusers source bakes in:
      - ``rope_theta = 10000.0`` (transformer_qwenimage.py:200 default).
      - ``rms_norm_eps = 1e-6`` (QwenImageTransformerBlock default).
      - ``timestep_embed_dim = 256`` (QwenTimestepProjEmbeddings default).
      - ``max_text_tokens = 1024`` (Qwen-Image pipeline default).
      - ``max_image_tokens = 8192`` (Qwen-Image pipeline ``max_image_seq_len``).

    Args:
        transformer_dir: Path to a directory containing ``config.json`` and
            one or more ``*.safetensors`` shards in the HF/diffusers layout.
            This is typically ``<repo>/transformer/`` under a diffusers
            snapshot of Qwen/Qwen-Image-2512.

    Returns:
        Tuple of ``(config, weights)`` where the keys of ``weights`` are the
        diffusers state-dict keys (e.g. ``img_in.weight``,
        ``transformer_blocks.0.attn.to_q.weight``), each cast to fp32
        ``np.ndarray``.

    Raises:
        FileNotFoundError: If ``config.json`` or any ``*.safetensors`` shard
            is missing from ``transformer_dir``.
    """
    import json

    from safetensors import safe_open

    try:
        # Required when safetensors emits a bf16 numpy view that needs
        # ml_dtypes registered before .astype() can decode it. Qwen-Image's
        # transformer shards are bf16-stored, so without this the
        # safe_open(..., framework="numpy") call below cannot return the
        # tensors (numpy has no native bf16 dtype).
        import ml_dtypes  # noqa: F401
    except ImportError:  # pragma: no cover -- best-effort
        pass

    text_dir = Path(transformer_dir)
    config_path = text_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {text_dir}")
    cfg_json = json.loads(config_path.read_text())

    num_attention_heads = int(cfg_json.get("num_attention_heads", 24))
    attention_head_dim = int(cfg_json.get("attention_head_dim", 128))
    hidden_size = num_attention_heads * attention_head_dim

    cfg = QwenImageDiTConfig(
        in_channels=int(cfg_json.get("in_channels", 64)),
        out_channels=int(cfg_json.get("out_channels", 16)),
        patch_size=int(cfg_json.get("patch_size", 2)),
        hidden_size=hidden_size,
        num_joint_blocks=int(cfg_json.get("num_layers", 60)),
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        intermediate_size=hidden_size * 4,  # FeedForward(dim, mult=4) -> 4*hidden.
        text_embed_dim=int(cfg_json.get("joint_attention_dim", 3584)),
        rope_axes_dim=list(cfg_json.get("axes_dims_rope", [16, 56, 56])),
        rope_theta=10000.0,  # hardcoded in diffusers source
        timestep_embed_dim=256,  # hardcoded in QwenTimestepProjEmbeddings
        rms_norm_eps=1e-6,
        layer_norm_eps=1e-6,
        max_image_tokens=8192,
        max_text_tokens=1024,
        guidance_embeds=bool(cfg_json.get("guidance_embeds", False)),
    )

    safetensor_files = sorted(text_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No *.safetensors in {text_dir}")

    weights: dict[str, np.ndarray] = {}
    for sf in safetensor_files:
        with safe_open(str(sf), framework="numpy") as f:
            for key in f.keys():
                arr = f.get_tensor(key)
                # Promote any low-precision dtype (bf16/fp16) to fp32. The
                # builder accepts fp32 only via _as_numpy().
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                weights[key] = arr

    return cfg, weights
