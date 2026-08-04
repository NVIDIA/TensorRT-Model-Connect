# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT graph operations for Python engine builds.

Tensor names and shapes must stay compatible with the C++ bundle runtime.
"""

from __future__ import annotations

from contextlib import nullcontext
import numpy as np
from tensorrt_model_connect import trt_compat


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


def _add_matrix_multiply_with_fp32_accumulation(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_op: trt.MatrixOperation,
    rhs: trt.ITensor,
    rhs_op: trt.MatrixOperation,
) -> trt.ITensor:
    """Request TensorRT's fused FP16 GEMM with FP32 accumulation."""
    output_dtype = lhs.dtype
    if lhs.dtype == trt.float16 and rhs.dtype == trt.float16:
        lhs = network.add_cast(lhs, trt.float32).get_output(0)
        rhs = network.add_cast(rhs, trt.float32).get_output(0)
    output = network.add_matrix_multiply(lhs, lhs_op, rhs, rhs_op).get_output(0)
    return _cast_back_to_trt_dtype(network, output, output_dtype)

def layer_tensor_name(stem: str, layer: int) -> str:
    return f"{stem}_{layer}"


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
    fp32_accumulation: bool = True,
) -> trt.ITensor:
    """Matrix multiply: lhs @ rhs_constant.  rhs is [lhs_width, rhs_width]."""
    rank = len(tuple(lhs.shape))
    rhs_shape = (
        (lhs_width, rhs_width)
        if rank <= 2
        else (1,) * (rank - 2) + (lhs_width, rhs_width)
    )
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(rhs_weights).reshape(rhs_shape),
        dtype=dtype,
    )
    rhs = _cast_back_to_trt_dtype(network, rhs, lhs.dtype)
    if fp32_accumulation:
        return _add_matrix_multiply_with_fp32_accumulation(
            network,
            lhs, trt.MatrixOperation.NONE,
            rhs, trt.MatrixOperation.NONE,
        )
    mm = network.add_matrix_multiply(
        lhs, trt.MatrixOperation.NONE,
        rhs, trt.MatrixOperation.NONE,
    )
    return _cast_back_to_trt_dtype(network, mm.get_output(0), lhs.dtype)


def add_matmul_rhs_tensor(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    rhs: trt.ITensor,
    *,
    fp32_accumulation: bool = True,
) -> trt.ITensor:
    """Matrix multiply ``lhs @ rhs`` when both operands are runtime tensors."""
    if fp32_accumulation:
        return _add_matrix_multiply_with_fp32_accumulation(
            network,
            lhs, trt.MatrixOperation.NONE,
            rhs, trt.MatrixOperation.NONE,
        )
    mm = network.add_matrix_multiply(
        lhs, trt.MatrixOperation.NONE,
        rhs, trt.MatrixOperation.NONE,
    )
    return _cast_back_to_trt_dtype(network, mm.get_output(0), lhs.dtype)


def add_lora_delta(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lora_a: trt.ITensor,
    lora_b: trt.ITensor,
) -> trt.ITensor:
    """Compute ``(lhs @ A) @ B`` for dynamically bound LoRA tensors.

    ``A`` is stored as ``[in_features, max_rank]`` and ``B`` as
    ``[max_rank, out_features]``.  The PEFT scale ``alpha / rank`` must be
    folded into B by the adapter loader before binding.
    """
    low_rank = add_matmul_rhs_tensor(network, lhs, lora_a)
    return add_matmul_rhs_tensor(network, low_rank, lora_b)


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
    bias_t = add_constant(
        network, bias_shape, np.asarray(bias).reshape(bias_shape), dtype=dtype)
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
    need_cast = (dtype != np.float32)
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        inp, recip.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)
    result = scaled.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def add_rms_norm_per_head(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
    sequence_length: int | None = 1,
) -> trt.ITensor:
    """Per-head RMSNorm for [Sq, num_heads * head_dim] tensors.

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    ``sequence_length=None`` means runtime-dynamic Sq.
    ``gamma`` may be [num_heads * head_dim] or [head_dim] broadcast to heads.
    """
    need_cast = (dtype != np.float32)
    output_dtype = inp.dtype
    seq_dim = -1 if sequence_length is None else sequence_length
    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (seq_dim, num_heads, head_dim)

    reshaped = reshape_in.get_output(0)
    if need_cast:
        reshaped = network.add_cast(reshaped, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    eps_3d = network.add_shuffle(eps_tensor)
    eps_3d.reshape_dims = (1, 1, 1)
    sq = network.add_elementwise(reshaped, reshaped, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 2, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_3d.get_output(0), trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        reshaped, recip.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_arr = np.asarray(gamma, dtype=np.float32)
    if gamma_arr.size == head_dim:
        gamma_t = add_constant(
            network, (1, 1, head_dim), gamma_arr.reshape(1, 1, head_dim),
            dtype=np.float32)
    else:
        gamma_t = add_constant(
            network, (1, num_heads, head_dim),
            gamma_arr.reshape(num_heads, head_dim), dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)

    result = scaled.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    reshape_out = network.add_shuffle(result)
    reshape_out.reshape_dims = (seq_dim, num_heads * head_dim)
    return reshape_out.get_output(0)


def add_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LayerNorm: gamma * ((x - mean) / sqrt(var + eps)) + beta.

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    """
    need_cast = (dtype != np.float32)
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    # mean = reduce_mean(x)
    mean = network.add_reduce(
        inp, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # x - mean
    centered = network.add_elementwise(
        inp, mean.get_output(0), trt.ElementWiseOperation.SUB)
    # variance = mean((x - mean)^2)
    sq = network.add_elementwise(
        centered.get_output(0), centered.get_output(0),
        trt.ElementWiseOperation.PROD)
    var = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # sqrt(var + eps)
    denom_in = network.add_elementwise(
        var.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    # normalized = (x - mean) / sqrt(var + eps)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0),
        trt.ElementWiseOperation.PROD)
    # gamma * normalized + beta
    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)
    beta_t = add_constant(network, (1, hidden_size), beta, dtype=np.float32)
    result = network.add_elementwise(
        scaled.get_output(0), beta_t, trt.ElementWiseOperation.SUM)
    result = result.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


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
        c = add_constant(
            network, const_shape, np.array([value], dtype=np.float32), dtype=dtype)
        return _cast_back_to_trt_dtype(network, c, target_dtype)

    # x^3
    x_sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    x_cu = network.add_elementwise(
        x_sq.get_output(0), inp, trt.ElementWiseOperation.PROD)
    # 0.044715 * x^3
    coeff = _const("coeff", 0.044715)
    scaled_cube = network.add_elementwise(
        x_cu.get_output(0), coeff, trt.ElementWiseOperation.PROD)
    # x + 0.044715 * x^3
    inner_sum = network.add_elementwise(
        inp, scaled_cube.get_output(0), trt.ElementWiseOperation.SUM)
    # sqrt(2/pi) * (x + 0.044715 * x^3)
    sqrt_2_over_pi = _const("sqrt_2_over_pi", np.sqrt(2.0 / np.pi))
    tanh_arg = network.add_elementwise(
        sqrt_2_over_pi, inner_sum.get_output(0),
        trt.ElementWiseOperation.PROD)
    # tanh(...)
    tanh_l = network.add_activation(
        tanh_arg.get_output(0), trt.ActivationType.TANH)
    # 1 + tanh(...)
    one = _const("one", 1.0)
    one_plus_tanh = network.add_elementwise(
        one, tanh_l.get_output(0), trt.ElementWiseOperation.SUM)
    # 0.5 * x
    half = _const("half", 0.5)
    half_x = network.add_elementwise(
        half, inp, trt.ElementWiseOperation.PROD)
    # 0.5 * x * (1 + tanh(...))
    result = network.add_elementwise(
        half_x.get_output(0), one_plus_tanh.get_output(0),
        trt.ElementWiseOperation.PROD)
    return result.get_output(0)


def add_activation(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    activation_type: str,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Dispatch activation by name: 'silu', 'gelu_new', 'gelu', 'relu', 'relu2'/'squared_relu'."""
    if activation_type in ("gelu_new", "gelu"):
        return add_gelu_new(network, inp, dtype=dtype)
    elif activation_type == "relu":
        act = network.add_activation(inp, trt.ActivationType.RELU)
        return act.get_output(0)
    elif activation_type in ("relu2", "squared_relu"):
        relu = network.add_activation(inp, trt.ActivationType.RELU)
        sq = network.add_elementwise(
            relu.get_output(0), relu.get_output(0),
            trt.ElementWiseOperation.PROD)
        return sq.get_output(0)
    elif activation_type == "silu":
        sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID)
        swish = network.add_elementwise(
            inp, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
        return swish.get_output(0)
    else:
        raise ValueError(f"Unsupported activation: {activation_type}")


def compute_alibi_slopes(num_heads: int) -> np.ndarray:
    """Compute ALiBi slopes for each attention head (from the ALiBi paper).

    For power-of-2 num_heads: geometric sequence 2^(-8/n * i), i in 1..n.
    For non-power-of-2: interleave two geometric sequences.

    Returns: [num_heads] float32 array.
    """
    def _get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(np.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if num_heads > 0 and (num_heads & (num_heads - 1)) == 0:
        # Power of 2
        return np.array(_get_slopes_power_of_2(num_heads), dtype=np.float32)
    else:
        closest_power_of_2 = 2 ** int(np.floor(np.log2(num_heads)))
        slopes_a = _get_slopes_power_of_2(closest_power_of_2)
        slopes_b = _get_slopes_power_of_2(2 * closest_power_of_2)
        slopes_b = slopes_b[0::2][: num_heads - closest_power_of_2]
        return np.array(slopes_a + slopes_b, dtype=np.float32)


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
        network, (1, seq_length, rope_dim // 2), cos_half.reshape(1, seq_length, -1), dtype=dtype)
    sin_const = add_constant(
        network, (1, seq_length, rope_dim // 2), sin_half.reshape(1, seq_length, -1), dtype=dtype)

    q = add_apply_rope_native_sequence(
        network, q, num_heads, head_dim, cos_const, sin_const,
        rotary_embedding_dim=rope_dim, sequence_length=seq_length)
    k = add_apply_rope_native_sequence(
        network, k, num_heads, head_dim, cos_const, sin_const,
        rotary_embedding_dim=rope_dim, sequence_length=seq_length)

    context_flat = add_attention_from_rows(
        network, q, k, v,
        num_heads=num_heads, head_dim=head_dim,
        q_seq=seq_length, kv_seq=seq_length)

    # Output projection
    out = add_matmul_rhs_constant(
        network, context_flat, hidden_size, hidden_size, w_o, dtype=dtype)
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
        network, (1, seq_length, rope_dim // 2), cos_half.reshape(1, seq_length, -1), dtype=dtype)
    sin_const = add_constant(
        network, (1, seq_length, rope_dim // 2), sin_half.reshape(1, seq_length, -1), dtype=dtype)

    q = add_apply_rope_native_sequence(
        network, q, num_heads, head_dim, cos_const, sin_const,
        rotary_embedding_dim=rope_dim, sequence_length=seq_length)
    k = add_apply_rope_native_sequence(
        network, k, num_heads, head_dim, cos_const, sin_const,
        rotary_embedding_dim=rope_dim, sequence_length=seq_length)

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
            network, q_win.get_output(0), k_win.get_output(0), v_win.get_output(0),
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
                q, start=(offset, 0), shape=(window_len, hidden_size), stride=(1, 1))
            k_slice = network.add_slice(
                k, start=(offset, 0), shape=(window_len, hidden_size), stride=(1, 1))
            v_slice = network.add_slice(
                v, start=(offset, 0), shape=(window_len, hidden_size), stride=(1, 1))
            window_outputs.append(add_attention_from_rows(
                network,
                q_slice.get_output(0),
                k_slice.get_output(0),
                v_slice.get_output(0),
                num_heads=num_heads,
                head_dim=head_dim,
                q_seq=window_len,
                kv_seq=window_len,
                scale=attn_scale,
            ))
            offset += window_len
        concat = network.add_concatenation(window_outputs)
        concat.axis = 0
        context_flat = concat.get_output(0)

    out = add_matmul_rhs_constant(
        network, context_flat, hidden_size, hidden_size, w_o, dtype=dtype)
    if o_bias is not None:
        out = add_bias_sum(network, out, hidden_size, o_bias, dtype=dtype)

    return out


def _runtime_rope_cache_3d(
    network: trt.INetworkDefinition,
    cache: trt.ITensor,
    half_head_dim: int,
    target_dtype: trt.DataType,
) -> trt.ITensor:
    """Normalize a runtime [N, D/2] vision RoPE cache to [1, N, D/2]."""
    if cache.dtype != target_dtype:
        cache = network.add_cast(cache, target_dtype).get_output(0)
    shaped = network.add_shuffle(cache)
    shaped.reshape_dims = (1, -1, half_head_dim)
    return shaped.get_output(0)


def add_dynamic_self_attention_with_rope(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    hidden_size: int,
    num_heads: int,
    cos_half: trt.ITensor,
    sin_half: trt.ITensor,
    q_bias: np.ndarray | None = None,
    k_bias: np.ndarray | None = None,
    v_bias: np.ndarray | None = None,
    o_bias: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Full vision attention for a runtime-variable number of patch rows."""
    head_dim = hidden_size // num_heads
    q = add_matmul_rhs_constant(
        network, hidden, hidden_size, hidden_size, w_q, dtype=dtype)
    k = add_matmul_rhs_constant(
        network, hidden, hidden_size, hidden_size, w_k, dtype=dtype)
    v = add_matmul_rhs_constant(
        network, hidden, hidden_size, hidden_size, w_v, dtype=dtype)
    if q_bias is not None:
        q = add_bias_sum(network, q, hidden_size, q_bias, dtype=dtype)
    if k_bias is not None:
        k = add_bias_sum(network, k, hidden_size, k_bias, dtype=dtype)
    if v_bias is not None:
        v = add_bias_sum(network, v, hidden_size, v_bias, dtype=dtype)

    cos_3d = _runtime_rope_cache_3d(
        network, cos_half, head_dim // 2, q.dtype)
    sin_3d = _runtime_rope_cache_3d(
        network, sin_half, head_dim // 2, q.dtype)
    q = add_apply_rope_native_sequence(
        network, q, num_heads, head_dim, cos_3d, sin_3d,
        rotary_embedding_dim=head_dim, sequence_length=None)
    k = add_apply_rope_native_sequence(
        network, k, num_heads, head_dim, cos_3d, sin_3d,
        rotary_embedding_dim=head_dim, sequence_length=None)
    context = add_attention_from_rows(
        network, q, k, v, num_heads=num_heads, head_dim=head_dim,
        q_seq=None, kv_seq=None)
    out = add_matmul_rhs_constant(
        network, context, hidden_size, hidden_size, w_o, dtype=dtype)
    if o_bias is not None:
        out = add_bias_sum(network, out, hidden_size, o_bias, dtype=dtype)
    return out


def add_dynamic_windowed_self_attention_with_rope(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    hidden_size: int,
    num_heads: int,
    cos_half: trt.ITensor,
    sin_half: trt.ITensor,
    padded_window_indices: trt.ITensor,
    compact_window_indices: trt.ITensor,
    window_mask: trt.ITensor,
    window_patch_size: int,
    q_bias: np.ndarray | None = None,
    k_bias: np.ndarray | None = None,
    v_bias: np.ndarray | None = None,
    o_bias: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Windowed vision attention for runtime-variable smart-resize grids."""
    if window_patch_size <= 0:
        raise ValueError("window_patch_size must be positive")
    head_dim = hidden_size // num_heads
    q = add_matmul_rhs_constant(
        network, hidden, hidden_size, hidden_size, w_q, dtype=dtype)
    k = add_matmul_rhs_constant(
        network, hidden, hidden_size, hidden_size, w_k, dtype=dtype)
    v = add_matmul_rhs_constant(
        network, hidden, hidden_size, hidden_size, w_v, dtype=dtype)
    if q_bias is not None:
        q = add_bias_sum(network, q, hidden_size, q_bias, dtype=dtype)
    if k_bias is not None:
        k = add_bias_sum(network, k, hidden_size, k_bias, dtype=dtype)
    if v_bias is not None:
        v = add_bias_sum(network, v, hidden_size, v_bias, dtype=dtype)

    cos_3d = _runtime_rope_cache_3d(
        network, cos_half, head_dim // 2, q.dtype)
    sin_3d = _runtime_rope_cache_3d(
        network, sin_half, head_dim // 2, q.dtype)
    q = add_apply_rope_native_sequence(
        network, q, num_heads, head_dim, cos_3d, sin_3d,
        rotary_embedding_dim=head_dim, sequence_length=None)
    k = add_apply_rope_native_sequence(
        network, k, num_heads, head_dim, cos_3d, sin_3d,
        rotary_embedding_dim=head_dim, sequence_length=None)

    windowed_qkv = []
    for tensor in (q, k, v):
        padded = network.add_gather(
            tensor, padded_window_indices, 0).get_output(0)
        shaped = network.add_shuffle(padded)
        shaped.reshape_dims = (-1, window_patch_size, num_heads, head_dim)
        shaped.second_transpose = trt.Permutation([0, 2, 1, 3])
        windowed_qkv.append(shaped.get_output(0))

    if window_mask.dtype != windowed_qkv[0].dtype:
        window_mask = network.add_cast(
            window_mask, windowed_qkv[0].dtype).get_output(0)
    context = add_attention_core(
        network, *windowed_qkv, mask=window_mask,
        scale=1.0 / np.sqrt(max(head_dim, 1)))
    flattened = network.add_shuffle(context)
    flattened.first_transpose = trt.Permutation([0, 2, 1, 3])
    flattened.reshape_dims = (-1, hidden_size)
    compact = network.add_gather(
        flattened.get_output(0), compact_window_indices, 0).get_output(0)

    out = add_matmul_rhs_constant(
        network, compact, hidden_size, hidden_size, w_o, dtype=dtype)
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


def add_spatial_merge(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    w_fc1: np.ndarray,
    w_fc2: np.ndarray,
    b_fc1: np.ndarray | None,
    b_fc2: np.ndarray | None,
    norm_gamma: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    eps_tensor: trt.ITensor,
    seq_length: int,
    merge_size: int = 2,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Spatial merge: 2x2 merge MLP that reduces spatial resolution.

    Reshapes [seq, dim] -> merge adjacent 2x2 patches, then MLP.
    Input: [seq_length, input_dim]
    Output: [seq_length // (merge_size^2), output_dim]

    Note: This is a simplified version. For Qwen2.5-VL, the merge
    concatenates merge_size^2 adjacent patches, then applies layernorm + MLP.
    """
    # LayerNorm on the merged representation
    norm = add_layer_norm(
        network, inp, input_dim,
        norm_gamma, np.zeros(input_dim, dtype=np.float32), eps_tensor,
        dtype=dtype)

    # For simplicity in the TRT graph, we use a 2-layer MLP directly
    # on the already-flattened input. The spatial rearrangement is handled
    # during preprocessing.
    fc1 = add_matmul_rhs_constant(network, norm, input_dim, hidden_dim, w_fc1, dtype=dtype)
    if b_fc1 is not None:
        fc1 = add_bias_sum(network, fc1, hidden_dim, b_fc1, dtype=dtype)

    # GELU activation
    activated = add_gelu_new(network, fc1, dtype=dtype)

    fc2 = add_matmul_rhs_constant(network, activated, hidden_dim, output_dim, w_fc2, dtype=dtype)
    if b_fc2 is not None:
        fc2 = add_bias_sum(network, fc2, output_dim, b_fc2, dtype=dtype)

    return fc2


# Alias: add_gelu_tanh is the same as add_gelu_new (tanh approximation)
add_gelu_tanh = add_gelu_new


# ---------------------------------------------------------------------------
# TRT 10 native attention APIs (TRT 10.x)
#
# Three primitives replace manual primitive chains:
#   add_layer_norm_native  → INormalizationLayer  (replaces add_layer_norm)
#   add_apply_rope_native  → IRotaryEmbeddingLayer
#   add_attention_core     → IAttention           (replaces score+softmax+V)
# ---------------------------------------------------------------------------

def add_layer_norm_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LayerNorm via TRT native INormalizationLayer (add_normalization_v2).

    Replaces the manual reduce/elementwise chain in add_layer_norm with a
    single fused layer that TRT can optimize end-to-end. In strongly typed
    networks, input/scale/bias must have identical tensor types; compute
    precision is set to FP32 for numerical stability when the TensorRT Python
    layer exposes that control.

    Note: INormalizationLayer computes (x - mean) / sqrt(var + eps) * gamma + beta.
    This is LayerNorm, NOT RMSNorm.  Use add_rms_norm for RMSNorm models.

    Args:
        inp:         Input tensor [*, hidden_size].
        hidden_size: Size of the normalized dimension (last axis).
        gamma:       Scale weights [hidden_size].
        beta:        Bias weights [hidden_size].
        eps:         Numerical stability epsilon (scalar, not a tensor).
        dtype:       Storage dtype for gamma/beta constants before TRT cast.
    """
    inp_shape = getattr(inp, "shape", None)
    rank = len(tuple(inp_shape)) if inp_shape is not None else 2
    param_shape = (
        (hidden_size,) if rank <= 1 else (1,) * (rank - 1) + (hidden_size,)
    )
    gamma_t = add_constant(
        network, param_shape, np.asarray(gamma).reshape(param_shape), dtype=dtype)
    beta_t = add_constant(
        network, param_shape, np.asarray(beta).reshape(param_shape), dtype=dtype)
    gamma_t = _cast_back_to_trt_dtype(network, gamma_t, inp.dtype)
    beta_t = _cast_back_to_trt_dtype(network, beta_t, inp.dtype)
    # axesMask bit i selects axis i as a reduction axis. The normalized
    # hidden dimension is always the last axis for [*, hidden_size] tensors.
    norm = network.add_normalization_v2(inp, gamma_t, beta_t, 1 << (rank - 1))
    norm.epsilon = eps
    # TensorRT 11 removed the Python INormalizationLayer.compute_precision
    # attribute. Keep the TRT 10 hint, and let TRT 11 infer the precision.
    if hasattr(norm, "compute_precision"):
        norm.compute_precision = trt.float32
    return norm.get_output(0)


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
            f"got {rotary_embedding_dim}")
    return rotary_embedding_dim


def make_native_active_rope_inv_freq(
    head_dim: int,
    rope_theta: float,
    partial_rotary_factor: float = 1.0,
) -> np.ndarray:
    """Reproduce HF's CPU-FP32 inverse frequencies without a context table."""

    rotary_ndims = validate_native_rope_dim(int(head_dim * partial_rotary_factor))
    rope_theta = float(rope_theta)
    if not np.isfinite(rope_theta) or rope_theta <= 0.0:
        raise ValueError("active RoPE requires finite positive rope_theta")
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Qwen-VL native active mRoPE requires PyTorch at build time"
        ) from exc
    with torch.no_grad():
        dims = torch.arange(
            0, rotary_ndims, 2, dtype=torch.int64, device="cpu"
        ).to(dtype=torch.float32)
        inv_freq = 1.0 / (rope_theta ** (dims / rotary_ndims))
    return np.asarray(inv_freq.detach().cpu().numpy(), dtype=np.float32).copy()


def add_active_rope_cache(
    network: trt.INetworkDefinition,
    position_id: trt.ITensor,
    inv_freq: np.ndarray,
    output_dtype: trt.DataType,
) -> tuple[trt.ITensor, trt.ITensor]:
    """Build rank-3 cos/sin tensors only for runtime-active positions."""

    inv_freq = np.asarray(inv_freq, dtype=np.float32)
    if inv_freq.ndim != 1 or inv_freq.size == 0:
        raise ValueError("active RoPE inverse frequencies must be rank-1")
    pos_float = network.add_cast(position_id, trt.float32).get_output(0)
    pos_col = network.add_shuffle(pos_float)
    pos_col.reshape_dims = (-1, 1)
    inv_freq_tensor = add_constant(
        network, (1, int(inv_freq.size)), inv_freq.reshape(1, -1),
        dtype=np.float32)
    angles = network.add_elementwise(
        pos_col.get_output(0), inv_freq_tensor, trt.ElementWiseOperation.PROD
    ).get_output(0)
    cos_2d = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    sin_2d = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    cos_3d = network.add_shuffle(cos_2d)
    cos_3d.reshape_dims = (1, -1, int(inv_freq.size))
    sin_3d = network.add_shuffle(sin_2d)
    sin_3d.reshape_dims = (1, -1, int(inv_freq.size))
    cos_cache = cos_3d.get_output(0)
    sin_cache = sin_3d.get_output(0)
    if cos_cache.dtype != output_dtype:
        cos_cache = network.add_cast(cos_cache, output_dtype).get_output(0)
        sin_cache = network.add_cast(sin_cache, output_dtype).get_output(0)
    return cos_cache, sin_cache


def add_active_mrope_cache(
    network: trt.INetworkDefinition,
    position_ids: trt.ITensor,
    inv_freq: np.ndarray,
    mrope_section: tuple[int, int, int],
    output_dtype: trt.DataType,
    *,
    mrope_interleaved: bool = False,
) -> tuple[trt.ITensor, trt.ITensor]:
    """Build HF-equivalent Qwen-VL 3-axis cos/sin rows for active tokens."""

    inv_freq = np.asarray(inv_freq, dtype=np.float32)
    sections = tuple(int(value) for value in mrope_section)
    if len(sections) != 3 or any(value <= 0 for value in sections):
        raise ValueError("mrope_section must contain three positive integers")
    if inv_freq.ndim != 1 or sum(sections) != int(inv_freq.size):
        raise ValueError("mrope_section must cover every inverse frequency")

    axis_angles: list[trt.ITensor] = []
    inv_tensor = add_constant(
        network, (1, int(inv_freq.size)), inv_freq.reshape(1, -1),
        dtype=np.float32)
    for axis in range(3):
        axis_index = add_constant(
            network, (1,), np.array([axis], dtype=np.int32), dtype=np.int32)
        axis_positions = network.add_gather(
            position_ids, axis_index, 0).get_output(0)
        axis_positions = network.add_cast(
            axis_positions, trt.float32).get_output(0)
        axis_column = network.add_shuffle(axis_positions)
        axis_column.reshape_dims = (-1, 1)
        axis_angles.append(network.add_elementwise(
            axis_column.get_output(0), inv_tensor,
            trt.ElementWiseOperation.PROD).get_output(0))

    parts: list[trt.ITensor] = []
    if mrope_interleaved:
        # HF Qwen3-VL starts from temporal frequencies, then overwrites
        # 1::3 with height and 2::3 with width within each section limit.
        for column in range(int(inv_freq.size)):
            axis = 0
            if column % 3 == 1 and column < sections[1] * 3:
                axis = 1
            elif column % 3 == 2 and column < sections[2] * 3:
                axis = 2
            column_index = add_constant(
                network, (1,), np.array([column], dtype=np.int32),
                dtype=np.int32)
            parts.append(network.add_gather(
                axis_angles[axis], column_index, 1).get_output(0))
    else:
        offset = 0
        for axis, width in enumerate(sections):
            column_indices = add_constant(
                network, (width,),
                np.arange(offset, offset + width, dtype=np.int32),
                dtype=np.int32)
            parts.append(network.add_gather(
                axis_angles[axis], column_indices, 1).get_output(0))
            offset += width

    joined = network.add_concatenation(parts)
    joined.axis = 1
    angles = joined.get_output(0)
    cos_2d = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    sin_2d = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    cos_3d = network.add_shuffle(cos_2d)
    cos_3d.reshape_dims = (1, -1, int(inv_freq.size))
    sin_3d = network.add_shuffle(sin_2d)
    sin_3d.reshape_dims = (1, -1, int(inv_freq.size))
    cos_cache = cos_3d.get_output(0)
    sin_cache = sin_3d.get_output(0)
    if cos_cache.dtype != output_dtype:
        cos_cache = network.add_cast(cos_cache, output_dtype).get_output(0)
        sin_cache = network.add_cast(sin_cache, output_dtype).get_output(0)
    return cos_cache, sin_cache


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
        return np.full((max(max_cache_length, 1), max(half, 1)),
                       default, dtype=np.float32)
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


def add_2d_mask_to_4d(
    network: trt.INetworkDefinition,
    mask_2d: trt.ITensor,
) -> trt.ITensor:
    """Reshape additive attention mask [Sq, K] to [1, 1, Sq, K]."""
    mask_shape = network.add_shape(mask_2d).get_output(0)
    ones = add_constant(
        network, (2,), np.array([1, 1], dtype=np.int64), dtype=np.int64)
    target = network.add_concatenation([ones, mask_shape])
    target.axis = 0
    mask_4d = network.add_shuffle(mask_2d)
    mask_4d.set_input(1, target.get_output(0))
    return mask_4d.get_output(0)


def add_alibi_mask_4d(
    network: trt.INetworkDefinition,
    mask_2d: trt.ITensor,
    position_id: trt.ITensor,
    alibi_slopes_tensor: trt.ITensor,
    cache_position_indices: trt.ITensor,
    num_heads: int,
    target_dtype: trt.DataType | None = None,
) -> trt.ITensor:
    """Build a per-head ALiBi additive mask for native IAttention.

    Args:
        mask_2d: [Sq, K] additive mask.
        position_id: [Sq] query positions.
        alibi_slopes_tensor: [H, 1, 1] per-head slopes.
        cache_position_indices: [cache_rows] key positions for cached rows.
        target_dtype: Optional dtype for the returned mask. Defaults to
            ``mask_2d.dtype``.

    Returns:
        [1, H, Sq, K] additive mask containing both ``mask_2d`` and
        ``slope[h] * (key_pos[k] - query_pos[q])``.
    """
    pos_float = network.add_cast(position_id, trt.float32).get_output(0)
    cache_positions = cache_position_indices
    if cache_positions.dtype != trt.float32:
        cache_positions = network.add_cast(cache_positions, trt.float32).get_output(0)

    key_pos = network.add_concatenation([cache_positions, pos_float])
    key_pos.axis = 0

    mask_shape = network.add_shape(mask_2d).get_output(0)
    one_const = add_constant(
        network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    sq_size = network.add_slice(mask_shape, start=(0,), shape=(1,), stride=(1,))
    k_size = network.add_slice(mask_shape, start=(1,), shape=(1,), stride=(1,))
    sq_size_t = sq_size.get_output(0)
    k_size_t = k_size.get_output(0)

    key_pos_shape = network.add_concatenation([one_const, k_size_t])
    key_pos_shape.axis = 0
    key_pos_2d = network.add_shuffle(key_pos.get_output(0))
    key_pos_2d.set_input(1, key_pos_shape.get_output(0))

    query_pos_shape = network.add_concatenation([sq_size_t, one_const])
    query_pos_shape.axis = 0
    query_pos_2d = network.add_shuffle(pos_float)
    query_pos_2d.set_input(1, query_pos_shape.get_output(0))

    rel_pos = network.add_elementwise(
        key_pos_2d.get_output(0), query_pos_2d.get_output(0),
        trt.ElementWiseOperation.SUB)

    one_const2 = add_constant(
        network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    rel_shape = network.add_concatenation([one_const, one_const2, sq_size_t, k_size_t])
    rel_shape.axis = 0
    rel_4d = network.add_shuffle(rel_pos.get_output(0))
    rel_4d.set_input(1, rel_shape.get_output(0))

    slopes = alibi_slopes_tensor
    if slopes.dtype != trt.float32:
        slopes = network.add_cast(slopes, trt.float32).get_output(0)
    slopes_4d = network.add_shuffle(slopes)
    slopes_4d.reshape_dims = (1, num_heads, 1, 1)

    alibi_bias = network.add_elementwise(
        slopes_4d.get_output(0), rel_4d.get_output(0),
        trt.ElementWiseOperation.PROD)
    alibi_bias_t = alibi_bias.get_output(0)

    mask_4d = add_2d_mask_to_4d(network, mask_2d)
    out_dtype = target_dtype or mask_4d.dtype
    if alibi_bias_t.dtype != out_dtype:
        alibi_bias_t = network.add_cast(alibi_bias_t, out_dtype).get_output(0)

    combined = network.add_elementwise(
        mask_4d, alibi_bias_t, trt.ElementWiseOperation.SUM)
    return combined.get_output(0)


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

    inp_4d = reshape_rows_to_heads_4d(
        network, inp, num_heads, head_dim, sequence_length)

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

    return reshape_heads_4d_to_rows(
        network, rope.get_output(0), attention_size, sequence_length)


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
    inp_4d = reshape_rows_to_heads_4d(
        network, inp, num_heads, head_dim, sequence_length)
    rope = network.add_rotary_embedding(
        inp_4d,
        cos_cache_3d,
        sin_cache_3d,
        interleaved,
        rotary_embedding_dim,
    )
    return reshape_heads_4d_to_rows(
        network, rope.get_output(0), attention_size, sequence_length)


def add_apply_mrope_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache_2d: trt.ITensor,
    sin_cache_2d: trt.ITensor,
    position_ids: trt.ITensor,
    mrope_section: tuple[int, int, int],
    rotary_embedding_dim: int,
    interleaved: bool = False,
) -> trt.ITensor:
    """Apply Qwen2.5-VL temporal/height/width RoPE to one token."""
    rotary_embedding_dim = validate_native_rope_dim(rotary_embedding_dim)
    sections = tuple(int(value) for value in mrope_section)
    if len(sections) != 3 or any(value <= 0 for value in sections):
        raise ValueError("mrope_section must contain three positive integers")
    if sum(sections) != rotary_embedding_dim // 2:
        raise ValueError(
            "mrope_section must sum to half the rotary embedding dimension; "
            f"got {sections} for rotary dimension {rotary_embedding_dim}")

    def build_cache(cache: trt.ITensor) -> trt.ITensor:
        selected = network.add_gather(cache, position_ids, 0).get_output(0)
        offset = 0
        parts = []
        for axis, width in enumerate(sections):
            part = network.add_slice(
                selected, start=(axis, offset), shape=(1, width), stride=(1, 1))
            parts.append(part.get_output(0))
            offset += width
        joined = network.add_concatenation(parts)
        joined.axis = 1
        shaped = network.add_shuffle(joined.get_output(0))
        shaped.reshape_dims = (1, 1, rotary_embedding_dim // 2)
        return shaped.get_output(0)

    return add_apply_rope_native_sequence(
        network, inp, num_heads, head_dim,
        build_cache(cos_cache_2d), build_cache(sin_cache_2d),
        rotary_embedding_dim, interleaved, sequence_length=1)


def add_apply_mrope_native_sequence(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache_2d: trt.ITensor,
    sin_cache_2d: trt.ITensor,
    position_ids: trt.ITensor,
    mrope_section: tuple[int, int, int],
    rotary_embedding_dim: int,
    interleaved: bool = False,
) -> trt.ITensor:
    """Apply Qwen2.5-VL mRoPE to a runtime-dynamic token sequence."""
    rotary_embedding_dim = validate_native_rope_dim(rotary_embedding_dim)
    sections = tuple(int(value) for value in mrope_section)
    if len(sections) != 3 or any(value <= 0 for value in sections):
        raise ValueError("mrope_section must contain three positive integers")
    half_dim = rotary_embedding_dim // 2
    if sum(sections) != half_dim:
        raise ValueError(
            "mrope_section must sum to half the rotary embedding dimension; "
            f"got {sections} for rotary dimension {rotary_embedding_dim}")

    def build_cache(cache: trt.ITensor) -> trt.ITensor:
        selected = network.add_gather(cache, position_ids, 0).get_output(0)
        offset = 0
        parts = []
        for axis, width in enumerate(sections):
            axis_index = add_constant(
                network, (1,), np.array([axis], dtype=np.int32), dtype=np.int32)
            axis_values = network.add_gather(selected, axis_index, 0).get_output(0)
            column_indices = add_constant(
                network, (width,), np.arange(offset, offset + width, dtype=np.int32),
                dtype=np.int32)
            part = network.add_gather(axis_values, column_indices, 2)
            parts.append(part.get_output(0))
            offset += width
        joined = network.add_concatenation(parts)
        joined.axis = 2
        return joined.get_output(0)

    return add_apply_rope_native_sequence(
        network, inp, num_heads, head_dim,
        build_cache(cos_cache_2d), build_cache(sin_cache_2d),
        rotary_embedding_dim, interleaved, sequence_length=None)


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
        q_scaled.get_output(0), k_4d, v_4d,
        trt.AttentionNormalizationOp.SOFTMAX,
        causal,
    )
    # Allow TRT to decompose into primitive ops when no fused kernel is
    # available (e.g. unsupported head-dim or dtype).  This guarantees
    # correctness on any configuration at the cost of potential performance.
    # Keep the FP32-accumulation path opaque: TRT 11.2's Myelin compiler can
    # otherwise merge every decoder layer into one graph and fail SSA
    # validation during a full Qwen-VL engine build.
    attn.decomposable = not fp32_accumulation
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
    const = add_constant(
        network, shape, np.full(shape, value, dtype=np_dtype),
        dtype=np_dtype)
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
    cap_t = _scalar_constant_for_trt_dtype(
        network, scalar_shape, float(cap), tensor.dtype)
    scaled = network.add_elementwise(
        tensor, cap_t, trt.ElementWiseOperation.DIV).get_output(0)
    capped = network.add_activation(
        scaled, trt.ActivationType.TANH).get_output(0)
    return network.add_elementwise(
        capped, cap_t, trt.ElementWiseOperation.PROD).get_output(0)


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
        raise ValueError(
            f"num_heads={num_heads} must be divisible by "
            f"num_kv_heads={num_kv_heads}")

    repeat = num_heads // num_kv_heads
    if num_kv_heads == 1:
        concat = network.add_concatenation([x_4d] * repeat)
        concat.axis = 1
        return concat.get_output(0)

    x_shape = network.add_shape(x_4d).get_output(0)
    one = add_constant(
        network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    seq = network.add_slice(x_shape, start=(2,), shape=(1,), stride=(1,))
    dim = add_constant(
        network, (1,), np.array([head_dim], dtype=np.int64), dtype=np.int64)
    slice_shape = network.add_concatenation([one, one, seq.get_output(0), dim])
    slice_shape.axis = 0

    repeated = []
    for head_idx in range(num_kv_heads):
        head_slice = network.add_slice(
            x_4d, start=(0, head_idx, 0, 0),
            shape=(1, 1, 1, head_dim), stride=(1, 1, 1, 1))
        head_slice.set_input(2, slice_shape.get_output(0))
        repeated.extend([head_slice.get_output(0)] * repeat)

    concat = network.add_concatenation(repeated)
    concat.axis = 1
    return concat.get_output(0)


def _add_decomposed_attention_core(
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
    logit_softcap: float | None = None,
) -> trt.ITensor:
    output_dtype = q_4d.dtype
    k_4d = _repeat_kv_heads_4d(
        network, k_4d, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim)
    v_4d = _repeat_kv_heads_4d(
        network, v_4d, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim)

    score_q = q_4d
    score_k = k_4d
    score_mask = mask
    if output_dtype != trt.float32:
        score_q = network.add_cast(score_q, trt.float32).get_output(0)
        score_k = network.add_cast(score_k, trt.float32).get_output(0)
        if score_mask is not None and score_mask.dtype != trt.float32:
            score_mask = network.add_cast(score_mask, trt.float32).get_output(0)

    scale_t = _scalar_constant_for_trt_dtype(
        network, (1, 1, 1, 1), scale, score_q.dtype)
    scores = network.add_matrix_multiply(
        score_q, trt.MatrixOperation.NONE,
        score_k, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = network.add_elementwise(
        scores, scale_t, trt.ElementWiseOperation.PROD).get_output(0)

    if logit_softcap is not None and float(logit_softcap) > 0.0:
        scores = add_tanh_softcap(
            network, scores, float(logit_softcap),
            scalar_shape=(1, 1, 1, 1))

    if score_mask is not None:
        scores = network.add_elementwise(
            scores, score_mask, trt.ElementWiseOperation.SUM).get_output(0)

    probs = network.add_softmax(scores)
    probs.axes = 1 << 3
    probs_t = probs.get_output(0)
    if probs_t.dtype != output_dtype:
        probs_t = network.add_cast(probs_t, output_dtype).get_output(0)

    context = network.add_matrix_multiply(
        probs_t, trt.MatrixOperation.NONE,
        v_4d, trt.MatrixOperation.NONE).get_output(0)
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
    force_decomposed_attention: bool = False,
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
        network, q, num_heads, head_dim, sequence_length=q_seq,
        tag=None if tag is None else tag + ".q")
    k_4d = reshape_rows_to_heads_4d(
        network, k, kv_heads, head_dim, sequence_length=kv_seq,
        tag=None if tag is None else tag + ".k")
    v_4d = reshape_rows_to_heads_4d(
        network, v, kv_heads, head_dim, sequence_length=kv_seq,
        tag=None if tag is None else tag + ".v")
    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    use_decomposed_attention = (
        force_decomposed_attention
        or (logit_softcap is not None and float(logit_softcap) > 0.0))
    if use_decomposed_attention:
        if causal:
            raise NotImplementedError(
                "decomposed attention requires an explicit additive mask")
        ctx_4d = _add_decomposed_attention_core(
            network, q_4d, k_4d, v_4d,
            num_heads=num_heads, num_kv_heads=kv_heads, head_dim=head_dim,
            mask=mask, scale=scale, logit_softcap=logit_softcap)
    else:
        ctx_4d = add_attention_core(
            network, q_4d, k_4d, v_4d, causal=causal, mask=mask, scale=scale,
            fp32_accumulation=fp32_accumulation)
    return reshape_heads_4d_to_rows(
        network, ctx_4d, attention_size, sequence_length=q_seq,
        tag=None if tag is None else tag + ".ctx")


def add_native_kv_cache_attention_from_rows(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k_update: trt.ITensor,
    v_update: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    cache_write_indices: trt.ITensor,
    key_value_lengths: trt.ITensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_seq: int | None,
    scale: float | None = None,
    tag: str | None = None,
    recipe_instance: str | None = None,
) -> dict[str, trt.ITensor]:
    """Update a full-capacity user buffer and attend only its valid prefix."""

    if not hasattr(network, "add_kv_cache_update") or not hasattr(
        network, "add_attention_v2"
    ):
        raise RuntimeError(
            "Qwen-VL native KV requires TensorRT add_kv_cache_update and "
            "add_attention_v2 support"
        )
    k_update_4d = reshape_rows_to_heads_4d(
        network, k_update, num_kv_heads, head_dim, sequence_length=q_seq,
        tag=None if tag is None else tag + ".k_update")
    v_update_4d = reshape_rows_to_heads_4d(
        network, v_update, num_kv_heads, head_dim, sequence_length=q_seq,
        tag=None if tag is None else tag + ".v_update")
    update_k = network.add_kv_cache_update(
        cache_k, k_update_4d, cache_write_indices, trt.KVCacheMode.LINEAR)
    update_v = network.add_kv_cache_update(
        cache_v, v_update_4d, cache_write_indices, trt.KVCacheMode.LINEAR)
    if update_k is None or update_v is None:
        raise RuntimeError("TensorRT failed to create Qwen-VL KV-cache update layers")
    if tag:
        update_k.name = tag + ".cache_k_update"
        update_v.name = tag + ".cache_v_update"
    updated_k = update_k.get_output(0)
    updated_v = update_v.get_output(0)

    q_4d = reshape_rows_to_heads_4d(
        network, q, num_heads, head_dim, sequence_length=q_seq,
        tag=None if tag is None else tag + ".q")
    if q_4d.dtype != trt.bfloat16:
        raise ValueError("Qwen-VL native KV attention requires BF16 queries")
    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    q_fp32 = network.add_cast(q_4d, trt.float32).get_output(0)
    scale_t = add_constant(
        network, (1, 1, 1, 1), np.array([[[[scale]]]], dtype=np.float32),
        dtype=np.float32)
    q_scaled = network.add_elementwise(
        q_fp32, scale_t, trt.ElementWiseOperation.PROD).get_output(0)
    q_scaled = network.add_cast(q_scaled, trt.bfloat16).get_output(0)

    recipe = nullcontext()
    if recipe_instance is not None:
        from ...tvm_ffi.graph_build import graph_recipe_region

        recipe = graph_recipe_region(
            network, "qwen.decode_attention_region@2", recipe_instance,
            output_shape_input=0)
    with recipe:
        attention = network.add_attention_v2(
            q_scaled,
            updated_k,
            updated_v,
            trt.AttentionNormalizationOp.SOFTMAX,
            trt.CausalMaskKind.LOWER_RIGHT,
        )
        if attention is None:
            raise RuntimeError("TensorRT failed to create Qwen-VL native attention")
        attention.decomposable = False
        attention.key_value_lengths = key_value_lengths
        if tag:
            attention.name = tag

    context = reshape_heads_4d_to_rows(
        network, attention.get_output(0), num_heads * head_dim,
        sequence_length=q_seq, tag=None if tag is None else tag + ".ctx")
    return {
        "context": context,
        "present_k": updated_k,
        "present_v": updated_v,
    }


# Backward-compatible name used by existing tests and call sites.
_add_attention_core = add_attention_core


def add_decoder_attention_ffi(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    all_k: trt.ITensor,
    all_v: trt.ITensor,
    *,
    kernel_name: str,
    num_heads: int,
    head_dim: int,
    attention_window: int,
) -> trt.ITensor:
    """Decoder attention via TVM-FFI kernel (FlashInfer, CuTe, etc).

    The kernel must be registered as a TVM-FFI global before engine build.

    Inputs:
        q:              [1, attention_size]
        all_k, all_v:   [attention_window, attention_size]
    Returns:
        context:        [1, attention_size]
    """
    attention_size = num_heads * head_dim

    q_2d = network.add_shuffle(q)
    q_2d.reshape_dims = (num_heads, head_dim)
    k_3d = network.add_shuffle(all_k)
    k_3d.reshape_dims = (attention_window, num_heads, head_dim)
    v_3d = network.add_shuffle(all_v)
    v_3d.reshape_dims = (attention_window, num_heads, head_dim)

    scale_val = 1.0 / (head_dim ** 0.5)
    ffi_outputs = add_tvm_ffi_kernel(
        network,
        kernel_name=kernel_name,
        inputs=[q_2d.get_output(0), k_3d.get_output(0),
                v_3d.get_output(0)],
        output_specs=[{"dims": [num_heads, head_dim], "dtype": "float16"}],
        workspace_bytes=32 * 1024 * 1024,  # 32MB for FlashInfer tmp
        extra_args=[
            {"type": "none"},              # maybe_lse
            {"type": "int", "value": 0},    # kv_layout_code (NHD)
            {"type": "int", "value": -1},   # window_left
            {"type": "none"},              # alibi_slopes
            {"type": "float", "value": 0.0},     # logits_soft_cap
            {"type": "float", "value": scale_val}, # sm_scale
            {"type": "float", "value": 1.0},      # rope_rcp_scale
            {"type": "float", "value": 0.0001},   # rope_rcp_theta
        ],
    )
    context_flat = network.add_shuffle(ffi_outputs[0])
    context_flat.reshape_dims = (1, attention_size)
    return context_flat.get_output(0)


# ---------------------------------------------------------------------------
# TVM-FFI kernel bridge
# ---------------------------------------------------------------------------

def add_tvm_ffi_kernel(
    network: trt.INetworkDefinition,
    kernel_name: str,
    inputs: list[trt.ITensor],
    output_specs: list[dict],
    workspace_bytes: int = 0,
    extra_args: list[dict] | None = None,
) -> list[trt.ITensor]:
    """Add a TVM-FFI kernel call as a TRT plugin layer.

    Args:
        network: TRT network being built.
        kernel_name: TVM-FFI global function name (e.g. "my_ns.my_kernel").
        inputs: List of input ITensor objects.
        output_specs: List of dicts, one per output. Each dict has:
            - "dims": "same_as_input_N" or list of ints for fixed shape
            - "dtype": "float32" or "float16" (default "float32")
        workspace_bytes: Extra workspace bytes for the kernel (default 0).
        extra_args: Optional list of extra scalar/pointer args to pass after
            tensors. Each dict has "type" ("none"|"int"|"float"|"ptr") and
            optional "value".

    Returns:
        List of output ITensor objects.
    """
    import json

    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("TvmFfiKernel", "1", "")
    if creator is None:
        raise RuntimeError(
            "TvmFfiKernel plugin not found in TRT registry. "
            "Ensure the C++ plugin is compiled with TRTMC_HAS_TVM_FFI=1."
        )

    spec_dict = {
        "num_inputs": len(inputs),
        "num_outputs": len(output_specs),
        "outputs": output_specs,
        "workspace_bytes": workspace_bytes,
    }
    if extra_args:
        spec_dict["extra_args"] = extra_args
    shape_spec = json.dumps(spec_dict)

    fields = [
        trt.PluginField("kernel_name", kernel_name.encode("utf-8"),
                         trt.PluginFieldType.CHAR),
        trt.PluginField("shape_spec", shape_spec.encode("utf-8"),
                         trt.PluginFieldType.CHAR),
    ]
    fc = trt.PluginFieldCollection(fields)
    plugin = creator.create_plugin("tvm_ffi_kernel", fc)

    layer = network.add_plugin_v2(inputs, plugin)
    return [layer.get_output(i) for i in range(layer.num_outputs)]
