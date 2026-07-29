# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT graph operations for Python engine builds.

Tensor names and shapes must stay compatible with the C++ bundle runtime.
"""

from __future__ import annotations

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
    mm = network.add_matrix_multiply(
        lhs, trt.MatrixOperation.NONE,
        rhs, trt.MatrixOperation.NONE,
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


def add_gelu_exact(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
) -> trt.ITensor:
    """PyTorch/HF exact GELU: ``0.5*x*(1+erf(x/sqrt(2)))``."""

    output_dtype = inp.dtype
    value = inp
    if value.dtype != trt.float32:
        value = network.add_cast(value, trt.float32).get_output(0)
    const_shape = (1,) * max(1, len(tuple(value.shape)))
    inv_sqrt_two = add_constant(
        network,
        const_shape,
        np.array([1.0 / np.sqrt(2.0)], dtype=np.float32),
        dtype=np.float32,
    )
    erf_input = network.add_elementwise(
        value,
        inv_sqrt_two,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    erf = network.add_unary(
        erf_input,
        trt.UnaryOperation.ERF,
    ).get_output(0)
    one = add_constant(
        network,
        const_shape,
        np.array([1.0], dtype=np.float32),
        dtype=np.float32,
    )
    one_plus_erf = network.add_elementwise(
        one,
        erf,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    half = add_constant(
        network,
        const_shape,
        np.array([0.5], dtype=np.float32),
        dtype=np.float32,
    )
    half_x = network.add_elementwise(
        half,
        value,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    result = network.add_elementwise(
        half_x,
        one_plus_erf,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    return _cast_back_to_trt_dtype(network, result, output_dtype)


def add_activation(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    activation_type: str,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Dispatch activation by name: 'silu', 'gelu_new', 'gelu', 'relu', 'relu2'/'squared_relu'."""
    if activation_type == "gelu_new":
        return add_gelu_new(network, inp, dtype=dtype)
    elif activation_type == "gelu":
        return add_gelu_exact(network, inp)
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
    *,
    rotary_dim: int | None = None,
) -> np.ndarray:
    """Return HF/Torch-exact frequencies for active-position RoPE."""

    if rotary_dim is None:
        rotary_dim = head_dim
    rotary_dim = validate_native_rope_dim(rotary_dim)
    if rotary_dim > int(head_dim):
        raise ValueError("rotary_dim cannot exceed head_dim")
    rope_theta = float(rope_theta)
    if not np.isfinite(rope_theta) or rope_theta <= 0.0:
        raise ValueError(
            "TRT native GPT-NeoX RoPE requires rope_theta to be finite "
            f"and positive; got {rope_theta}"
        )
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "TensorRT native GPT-NeoX KV build requires PyTorch in the "
            "Model Connect build environment to generate Hugging "
            "Face-exact RoPE frequencies"
        ) from exc

    with torch.no_grad():
        exponents = (
            torch.arange(
                0,
                rotary_dim,
                2,
                dtype=torch.int64,
                device="cpu",
            ).to(dtype=torch.float32)
            / rotary_dim
        )
        inv_freq = 1.0 / (rope_theta**exponents)
    return np.asarray(
        inv_freq.detach().cpu().contiguous().numpy(),
        dtype=np.float32,
    ).copy()


def add_active_rope_cache(
    network: trt.INetworkDefinition,
    position_id: trt.ITensor,
    inv_freq: np.ndarray,
    output_dtype: trt.DataType,
) -> tuple[trt.ITensor, trt.ITensor]:
    """Build ``[1, Sq, rotary_dim/2]`` cos/sin for active positions."""

    inv_freq = np.asarray(inv_freq, dtype=np.float32)
    if inv_freq.ndim != 1 or inv_freq.size == 0:
        raise ValueError(
            "active RoPE inverse frequencies must be a non-empty rank-1 array"
        )

    pos_float = network.add_cast(position_id, trt.float32).get_output(0)
    pos_col = network.add_shuffle(pos_float)
    pos_col.reshape_dims = (-1, 1)
    inv_freq_tensor = add_constant(
        network,
        (1, int(inv_freq.size)),
        inv_freq.reshape(1, -1),
        dtype=np.float32,
    )
    angles = network.add_elementwise(
        pos_col.get_output(0),
        inv_freq_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    cos_2d = network.add_unary(
        angles,
        trt.UnaryOperation.COS,
    ).get_output(0)
    sin_2d = network.add_unary(
        angles,
        trt.UnaryOperation.SIN,
    ).get_output(0)

    cos_3d = network.add_shuffle(cos_2d)
    cos_3d.reshape_dims = (1, -1, int(inv_freq.size))
    sin_3d = network.add_shuffle(sin_2d)
    sin_3d.reshape_dims = (1, -1, int(inv_freq.size))
    cos_cache = cos_3d.get_output(0)
    sin_cache = sin_3d.get_output(0)
    if cos_cache.dtype != output_dtype:
        cos_cache = network.add_cast(
            cos_cache,
            output_dtype,
        ).get_output(0)
        sin_cache = network.add_cast(
            sin_cache,
            output_dtype,
        ).get_output(0)
    return cos_cache, sin_cache


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
    position_id: trt.ITensor | None,
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

    rope = network.add_rotary_embedding(
        inp_4d,
        cos_cache_2d,
        sin_cache_2d,
        interleaved,
        rotary_embedding_dim,
    )
    if position_id is not None:
        # Reshape position_id [Sq] -> [1, Sq] (batch=1).
        seq_dim = -1 if sequence_length is None else sequence_length
        pos_2d = network.add_shuffle(position_id)
        pos_2d.reshape_dims = (1, seq_dim)
        rope.set_input(3, pos_2d.get_output(0))

    return reshape_heads_4d_to_rows(
        network, rope.get_output(0), attention_size, sequence_length)


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
) -> dict[str, trt.ITensor]:
    """Update a user-owned cache and attend over its active prefix."""

    if not hasattr(network, "add_kv_cache_update") or not hasattr(
        network,
        "add_attention_v2",
    ):
        raise RuntimeError(
            "GPT-NeoX native KV cache requires TensorRT "
            "add_kv_cache_update and add_attention_v2 support"
        )

    k_update_4d = reshape_rows_to_heads_4d(
        network,
        k_update,
        num_kv_heads,
        head_dim,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".k_update",
    )
    v_update_4d = reshape_rows_to_heads_4d(
        network,
        v_update,
        num_kv_heads,
        head_dim,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".v_update",
    )

    update_k = network.add_kv_cache_update(
        cache_k,
        k_update_4d,
        cache_write_indices,
        trt.KVCacheMode.LINEAR,
    )
    update_v = network.add_kv_cache_update(
        cache_v,
        v_update_4d,
        cache_write_indices,
        trt.KVCacheMode.LINEAR,
    )
    if update_k is None or update_v is None:
        raise RuntimeError(
            "TensorRT failed to create GPT-NeoX KV-cache update layers"
        )
    if tag:
        update_k.name = tag + ".cache_k_update"
        update_v.name = tag + ".cache_v_update"
    updated_k = update_k.get_output(0)
    updated_v = update_v.get_output(0)

    q_4d = reshape_rows_to_heads_4d(
        network,
        q,
        num_heads,
        head_dim,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".q",
    )
    if scale is None:
        scale = (
            float(1.0 / np.sqrt(head_dim))
            if head_dim > 0
            else 1.0
        )
    if q_4d.dtype != trt.float16:
        raise ValueError(
            "GPT-NeoX native KV attention requires FP16 queries"
        )

    q_scale_input = network.add_cast(
        q_4d,
        trt.float32,
    ).get_output(0)
    scale_t = add_constant(
        network,
        (1, 1, 1, 1),
        np.array([[[[scale]]]], dtype=np.float32),
        dtype=np.float32,
    )
    q_scaled = network.add_elementwise(
        q_scale_input,
        scale_t,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    q_scaled = network.add_cast(
        q_scaled,
        trt.float16,
    ).get_output(0)

    attention = network.add_attention_v2(
        q_scaled,
        updated_k,
        updated_v,
        trt.AttentionNormalizationOp.SOFTMAX,
        trt.CausalMaskKind.LOWER_RIGHT,
    )
    if attention is None:
        raise RuntimeError(
            "TensorRT failed to create GPT-NeoX native attention"
        )
    attention.decomposable = False
    attention.key_value_lengths = key_value_lengths
    if tag:
        attention.name = tag

    context = reshape_heads_4d_to_rows(
        network,
        attention.get_output(0),
        num_heads * head_dim,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".ctx",
    )
    return {
        "context": context,
        "present_k": updated_k,
        "present_v": updated_v,
    }
