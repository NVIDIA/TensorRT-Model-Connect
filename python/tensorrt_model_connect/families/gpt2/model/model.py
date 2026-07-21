# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

if TYPE_CHECKING:
    from ....quantization.context import QuantContext
    from ..config import ModelConfig
    from ..weights import WeightDict


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


def add_last_token_logits(
    network: trt.INetworkDefinition,
    hidden_state: trt.ITensor,
    hidden: int,
    output_weights: np.ndarray,
    dtype: np.dtype,
) -> trt.ITensor:
    shape = network.add_shape(hidden_state).get_output(0)
    one_hidden = add_constant(network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
    start = network.add_elementwise(shape, one_hidden, trt.ElementWiseOperation.SUB).get_output(0)
    slicer = network.add_slice(hidden_state, (0, 0), (0, 0), (1, 1))
    slicer.set_input(1, start)
    slicer.set_input(2, one_hidden)
    return add_matmul_rhs_constant(
        network,
        slicer.get_output(0),
        hidden,
        output_weights.shape[1],
        output_weights,
        dtype=dtype,
    )


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
    need_cast = dtype != np.float32
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    # mean = reduce_mean(x)
    mean = network.add_reduce(inp, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # x - mean
    centered = network.add_elementwise(inp, mean.get_output(0), trt.ElementWiseOperation.SUB)
    # variance = mean((x - mean)^2)
    sq = network.add_elementwise(
        centered.get_output(0), centered.get_output(0), trt.ElementWiseOperation.PROD
    )
    var = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # sqrt(var + eps)
    denom_in = network.add_elementwise(var.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    # normalized = (x - mean) / sqrt(var + eps)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0), trt.ElementWiseOperation.PROD
    )
    # gamma * normalized + beta
    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
    beta_t = add_constant(network, (1, hidden_size), beta, dtype=np.float32)
    result = network.add_elementwise(scaled.get_output(0), beta_t, trt.ElementWiseOperation.SUM)
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
    param_shape = (hidden_size,) if rank <= 1 else (1,) * (rank - 1) + (hidden_size,)
    gamma_t = add_constant(
        network, param_shape, np.asarray(gamma).reshape(param_shape), dtype=dtype
    )
    beta_t = add_constant(network, param_shape, np.asarray(beta).reshape(param_shape), dtype=dtype)
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
    ones = add_constant(network, (2,), np.array([1, 1], dtype=np.int64), dtype=np.int64)
    target = network.add_concatenation([ones, mask_shape])
    target.axis = 0
    mask_4d = network.add_shuffle(mask_2d)
    mask_4d.set_input(1, target.get_output(0))
    return mask_4d.get_output(0)


def add_attention_core(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    mask: trt.ITensor,
    scale: float,
) -> trt.ITensor:
    """Run GPT-2's scaled, additively masked noncausal attention."""
    output_dtype = q_4d.dtype
    scale_np_dtype = np.float16 if q_4d.dtype == trt.float16 else np.float32
    scale_t = add_constant(
        network,
        (1, 1, 1, 1),
        np.array([[[[scale]]]], dtype=scale_np_dtype),
        dtype=scale_np_dtype,
    )
    if q_4d.dtype == trt.bfloat16:
        scale_t = network.add_cast(scale_t, trt.bfloat16).get_output(0)
    q_scaled = network.add_elementwise(q_4d, scale_t, trt.ElementWiseOperation.PROD).get_output(0)

    attn = network.add_attention(
        q_scaled,
        k_4d,
        v_4d,
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    attn.decomposable = True
    attn.mask = mask
    return _cast_back_to_trt_dtype(network, attn.get_output(0), output_dtype)


def add_attention_from_rows(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
    q_seq: int | None,
    kv_seq: int | None,
    mask: trt.ITensor,
    scale: float,
    tag: str | None = None,
) -> trt.ITensor:
    """Run GPT-2 attention for row-major Q/K/V tensors."""
    attention_size = num_heads * head_dim
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
        num_heads,
        head_dim,
        sequence_length=kv_seq,
        tag=None if tag is None else tag + ".k",
    )
    v_4d = reshape_rows_to_heads_4d(
        network,
        v,
        num_heads,
        head_dim,
        sequence_length=kv_seq,
        tag=None if tag is None else tag + ".v",
    )
    ctx_4d = add_attention_core(network, q_4d, k_4d, v_4d, mask=mask, scale=scale)
    return reshape_heads_4d_to_rows(
        network,
        ctx_4d,
        attention_size,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".ctx",
    )


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


def add_attention_block(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    max_cache_length: int,
    attention_scale: float,
    eps: float,
    dtype: np.dtype,
    quant_ctx: QuantContext | None,
) -> dict[str, trt.ITensor]:
    """Add GPT-2 LayerNorm, self-attention, and output projection."""
    matmul = make_matmul_fn(network, dtype, quant_ctx)
    attention_window = max_cache_length + 1
    normed = add_layer_norm_native(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        weights[f"{prefix}.input_norm_beta"],
        eps,
        dtype=dtype,
    )
    q = matmul(normed, hidden_size, hidden_size, weights[f"{prefix}.w_q"], f"{prefix}.w_q")
    k = matmul(normed, hidden_size, hidden_size, weights[f"{prefix}.w_k"], f"{prefix}.w_k")
    v = matmul(normed, hidden_size, hidden_size, weights[f"{prefix}.w_v"], f"{prefix}.w_v")
    q = add_bias_sum(network, q, hidden_size, weights[f"{prefix}.q_bias"], dtype=dtype)
    k = add_bias_sum(network, k, hidden_size, weights[f"{prefix}.k_bias"], dtype=dtype)
    v = add_bias_sum(network, v, hidden_size, weights[f"{prefix}.v_bias"], dtype=dtype)
    present_k = k
    present_v = v
    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (1, hidden_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (1, hidden_size)
    all_k = network.add_concatenation([cache_k, k_reshape.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation([cache_v, v_reshape.get_output(0)])
    all_v.axis = 0
    context = add_attention_from_rows(
        network,
        q,
        all_k.get_output(0),
        all_v.get_output(0),
        num_heads=num_heads,
        head_dim=head_dim,
        q_seq=1,
        kv_seq=attention_window,
        mask=add_2d_mask_to_4d(network, attention_mask),
        scale=attention_scale,
    )
    attn_out = matmul(
        context,
        hidden_size,
        hidden_size,
        weights[f"{prefix}.w_o"],
        f"{prefix}.w_o",
    )
    attn_out = add_bias_sum(
        network, attn_out, hidden_size, weights[f"{prefix}.o_bias"], dtype=dtype
    )
    return {
        "normed": normed,
        "attn_out": attn_out,
        "present_k": present_k,
        "present_v": present_v,
    }


def add_gelu_fc_projection(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: WeightDict,
    prefix: str,
    hidden: int,
    mlp_size: int,
    dtype: np.dtype,
) -> trt.ITensor:
    fc1 = matmul(
        inp,
        hidden,
        mlp_size,
        weights[f"{prefix}.w_fc1"],
        f"{prefix}.w_fc1",
    )
    fc1 = add_bias_sum(network, fc1, mlp_size, weights[f"{prefix}.fc1_bias"], dtype=dtype)
    activated = add_gelu_new(network, fc1, dtype=dtype)
    fc2 = matmul(
        activated,
        mlp_size,
        hidden,
        weights[f"{prefix}.w_fc2"],
        f"{prefix}.w_fc2",
    )
    return fc2


@dataclass(frozen=True)
class BuilderContext:
    """TensorRT objects shared by engine builders."""

    logger: trt.Logger
    builder: trt.Builder
    network: trt.INetworkDefinition
    config: trt.IBuilderConfig


def work_dtypes(precision: str):
    if precision == "fp16":
        return np.float16, trt.float16
    if precision == "bf16":
        return np.float16, trt.bfloat16
    return np.float32, trt.float32


def create_builder_context(
    *,
    verbose: bool,
) -> BuilderContext:
    """Create GPT-2's strongly typed TensorRT builder objects."""
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    return BuilderContext(
        logger=logger,
        builder=builder,
        network=network,
        config=config,
    )


def const_in_work_dtype(
    network: trt.INetworkDefinition,
    shape: tuple,
    values: np.ndarray,
    work_np_dtype: np.dtype,
    work_trt_dtype: trt.DataType,
) -> trt.ITensor:
    """Create a constant in storage dtype and cast it to runtime dtype."""
    const = add_constant(network, shape, values, dtype=work_np_dtype)
    if const.dtype != work_trt_dtype:
        const = network.add_cast(const, work_trt_dtype).get_output(0)
    return const


def norm_multi(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype,
) -> trt.ITensor:
    """Apply GPT-2 LayerNorm to a dynamic sequence."""
    return add_layer_norm(network, inp, hidden, gamma, beta, eps_tensor, dtype=dtype)


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    """Mark a tensor as a network output for debug inspection."""
    # Use an identity layer to avoid aliasing issues with existing outputs.
    cast = network.add_cast(tensor, trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


def build_standard_decoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx: QuantContext | None = None,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
) -> bytes:
    """Build GPT-2's fixed-token or delegated dual-profile engine."""
    import os as _os

    config.raw["_decoder_engine_layout_supported"] = True
    decoder_engine_role = str(config.raw.get("_decoder_engine_role", "dual_profile"))
    _dual_profile_disabled_for = (
        debug_layer_outputs or _os.environ.get("TRTMC_NO_DUAL_PROFILE") == "1"
    )
    if decoder_engine_role == "prefill" and _dual_profile_disabled_for:
        raise NotImplementedError(
            "split prefill engine is not supported for this standard decoder configuration"
        )
    if not _dual_profile_disabled_for and decoder_engine_role in ("dual_profile", "prefill"):
        return build_dual_profile_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
            profile_mode=("prefill" if decoder_engine_role == "prefill" else "dual_profile"),
        )
    hidden = config.hidden_size
    mlp_size: int = weights.get("_mlp_size", config.intermediate_size)
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = hidden // num_heads
    attention_window = max_cache_length + 1
    builder_context = create_builder_context(verbose=verbose)
    builder = builder_context.builder
    network = builder_context.network
    trt_config = builder_context.config
    work_np_dtype, work_trt_dtype = work_dtypes(precision)
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (1, attention_window))
    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        ck = network.add_input(
            layer_tensor_name("cache_k", i),
            work_trt_dtype,
            (max_cache_length, hidden),
        )
        cv = network.add_input(
            layer_tensor_name("cache_v", i),
            work_trt_dtype,
            (max_cache_length, hidden),
        )
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)
    if work_trt_dtype != trt.float32:
        mask_cast = network.add_cast(attention_mask, work_trt_dtype)
        attention_mask = mask_cast.get_output(0)

    embedding_table = add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
    )
    position_weights = weights["position_embedding"]
    position_embed_table = add_constant(
        network,
        position_weights.shape,
        position_weights,
        dtype=work_np_dtype,
    )
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))
    gather = network.add_gather(embedding_table, token_id, 0)
    pos_gather = network.add_gather(position_embed_table, position_id, 0)
    hidden_state = network.add_elementwise(
        gather.get_output(0),
        pos_gather.get_output(0),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)
    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")
    present_k_outputs = []
    present_v_outputs = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        result = _add_decoder_layer(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_inputs[layer_idx],
            cache_v=cache_v_inputs[layer_idx],
            attention_mask=attention_mask,
            attention_scale=attn_scale,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            mlp_size=mlp_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_cache_length=max_cache_length,
            eps=config.rms_norm_eps,
            dtype=work_np_dtype,
            quant_ctx=quant_ctx,
        )
        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])
        if debug_layer_outputs:
            _mark_debug_output(network, result["post_attn"], f"debug_post_attn_{layer_idx}")
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")
    hidden_state = _apply_norm(
        network,
        hidden_state,
        hidden,
        weights["final_norm"],
        weights["final_norm_beta"],
        config.rms_norm_eps,
        work_np_dtype,
    )
    out_vocab = weights["w_out"].shape[1]
    logits = add_matmul_rhs_constant(
        network, hidden_state, hidden, out_vocab, weights["w_out"], dtype=work_np_dtype
    )
    if work_trt_dtype != trt.float32:
        logits_cast = network.add_cast(logits, trt.float32)
        logits = logits_cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    for i in range(num_layers):
        pk = present_k_outputs[i]
        pv = present_v_outputs[i]
        pk.name = layer_tensor_name("present_k", i)
        pv.name = layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)
    if verbose:
        print(
            f"[trtmc build] Building TRT engine ({num_layers} layers, hidden={hidden}, mlp={mlp_size}, cache={max_cache_length}, precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(plan)


def _apply_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    dtype: np.dtype,
) -> trt.ITensor:
    return add_layer_norm_native(network, inp, hidden_size, gamma, beta, eps, dtype=dtype)


def _add_decoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    attention_scale: float,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    num_heads: int,
    head_dim: int,
    max_cache_length: int,
    eps: float,
    dtype: np.dtype,
    quant_ctx: QuantContext | None,
) -> dict[str, trt.ITensor]:
    attn = add_attention_block(
        network,
        hidden,
        cache_k,
        cache_v,
        attention_mask,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        max_cache_length=max_cache_length,
        attention_scale=attention_scale,
        eps=eps,
        dtype=dtype,
        quant_ctx=quant_ctx,
    )
    residual = network.add_elementwise(
        hidden, attn["attn_out"], trt.ElementWiseOperation.SUM
    ).get_output(0)
    normed = _apply_norm(
        network,
        residual,
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        weights[f"{prefix}.post_attn_norm_beta"],
        eps,
        dtype,
    )
    mlp_out = add_gelu_fc_projection(
        network,
        normed,
        matmul=make_matmul_fn(network, dtype, quant_ctx),
        weights=weights,
        prefix=prefix,
        hidden=hidden_size,
        mlp_size=mlp_size,
        dtype=dtype,
    )
    mlp_out = add_bias_sum(
        network,
        mlp_out,
        hidden_size,
        weights[f"{prefix}.fc2_bias"],
        dtype=dtype,
    )
    hidden_out = network.add_elementwise(
        residual, mlp_out, trt.ElementWiseOperation.SUM
    ).get_output(0)
    return {
        "hidden": hidden_out,
        "post_attn": residual,
        "present_k": attn["present_k"],
        "present_v": attn["present_v"],
    }


def build_dual_profile_decoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp16",
    opt_prefill_length: int = 64,
    max_prefill_length: int | None = None,
    quant_ctx: QuantContext | None = None,
    verbose: bool = False,
    profile_mode: str = "dual_profile",
) -> bytes:
    """Build GPT-2 with dynamic prefill/decode sequence profiles."""
    if profile_mode not in {"dual_profile", "prefill"}:
        raise ValueError(f"profile_mode must be 'dual_profile' or 'prefill', got {profile_mode!r}")
    if max_prefill_length is None:
        max_prefill_length = max_cache_length
    max_prefill_length = max(1, min(max_prefill_length, max_cache_length))
    opt_prefill_length = max(1, min(opt_prefill_length, max_prefill_length))

    hidden = config.hidden_size
    mlp_size = weights.get("_mlp_size", config.intermediate_size)
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = hidden // num_heads
    builder_context = create_builder_context(verbose=verbose)
    builder = builder_context.builder
    network = builder_context.network
    trt_config = builder_context.config
    work_np_dtype, work_trt_dtype = work_dtypes(precision)

    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (-1, -1))
    cache_shape = (max_cache_length, hidden)
    cache_k_inputs = [
        network.add_input(layer_tensor_name("cache_k", i), work_trt_dtype, cache_shape)
        for i in range(num_layers)
    ]
    cache_v_inputs = [
        network.add_input(layer_tensor_name("cache_v", i), work_trt_dtype, cache_shape)
        for i in range(num_layers)
    ]
    attention_mask_work = attention_mask
    if work_trt_dtype != trt.float32:
        attention_mask_work = network.add_cast(attention_mask, work_trt_dtype).get_output(0)

    def _add_profile(opt_sq: int, max_sq: int, *, fixed: bool = False) -> None:
        profile = builder.create_optimization_profile()
        min_sq = opt_sq if fixed else 1
        profile.set_shape("token_id", (min_sq,), (opt_sq,), (max_sq,))
        profile.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
        profile.set_shape(
            "attention_mask",
            (min_sq, max_cache_length + min_sq),
            (opt_sq, max_cache_length + opt_sq),
            (max_sq, max_cache_length + max_sq),
        )
        trt_config.add_optimization_profile(profile)

    import os as _os_dbg

    if profile_mode == "prefill":
        _add_profile(opt_prefill_length, max_prefill_length)
    elif _os_dbg.environ.get("TRTMC_DECODE_ONLY_DEBUG") == "1":
        _add_profile(1, 1, fixed=True)
    elif _os_dbg.environ.get("TRTMC_REVERSE_PROFILE_ORDER") == "1":
        _add_profile(1, 1, fixed=True)
        _add_profile(opt_prefill_length, max_prefill_length)
    else:
        _add_profile(opt_prefill_length, max_prefill_length)
        _add_profile(1, 1, fixed=True)

    embedding_table = const_in_work_dtype(
        network,
        (vocab, hidden),
        weights["embedding"],
        work_np_dtype,
        work_trt_dtype,
    )
    position_weights = weights["position_embedding"]
    position_table = const_in_work_dtype(
        network,
        position_weights.shape,
        position_weights,
        work_np_dtype,
        work_trt_dtype,
    )
    eps_tensor = add_constant(
        network,
        (1, 1),
        np.array([[config.rms_norm_eps]], dtype=np.float32),
        dtype=np.float32,
    )
    matmul = make_matmul_fn(network, work_np_dtype, quant_ctx)
    token_embedding = network.add_gather(embedding_table, token_id, 0).get_output(0)
    position_embedding = network.add_gather(position_table, position_id, 0).get_output(0)
    hidden_state = network.add_elementwise(
        token_embedding, position_embedding, trt.ElementWiseOperation.SUM
    ).get_output(0)
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    mask_4d = add_2d_mask_to_4d(network, attention_mask_work)
    present_k_outs: list[trt.ITensor] = []
    present_v_outs: list[trt.ITensor] = []
    attention_scale = 1.0 / np.sqrt(max(head_dim, 1))
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        normed = norm_multi(
            network,
            hidden_state,
            hidden,
            weights[f"{prefix}.input_norm"],
            weights[f"{prefix}.input_norm_beta"],
            eps_tensor,
            work_np_dtype,
        )
        q = matmul(normed, hidden, hidden, weights[f"{prefix}.w_q"], f"{prefix}.w_q")
        k = matmul(normed, hidden, hidden, weights[f"{prefix}.w_k"], f"{prefix}.w_k")
        v = matmul(normed, hidden, hidden, weights[f"{prefix}.w_v"], f"{prefix}.w_v")
        q = add_bias_sum(network, q, hidden, weights[f"{prefix}.q_bias"], dtype=work_np_dtype)
        k = add_bias_sum(network, k, hidden, weights[f"{prefix}.k_bias"], dtype=work_np_dtype)
        v = add_bias_sum(network, v, hidden, weights[f"{prefix}.v_bias"], dtype=work_np_dtype)
        present_k_outs.append(k)
        present_v_outs.append(v)
        all_k = network.add_concatenation([cache_k_inputs[layer_idx], k])
        all_k.axis = 0
        all_v = network.add_concatenation([cache_v_inputs[layer_idx], v])
        all_v.axis = 0
        context = add_attention_from_rows(
            network,
            q,
            all_k.get_output(0),
            all_v.get_output(0),
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq=None,
            kv_seq=None,
            mask=mask_4d,
            scale=attention_scale,
            tag=f"{prefix}.attn",
        )
        attn_out = matmul(context, hidden, hidden, weights[f"{prefix}.w_o"], f"{prefix}.w_o")
        attn_out = add_bias_sum(
            network,
            attn_out,
            hidden,
            weights[f"{prefix}.o_bias"],
            dtype=work_np_dtype,
        )
        residual = network.add_elementwise(
            hidden_state, attn_out, trt.ElementWiseOperation.SUM
        ).get_output(0)
        normed = norm_multi(
            network,
            residual,
            hidden,
            weights[f"{prefix}.post_attn_norm"],
            weights[f"{prefix}.post_attn_norm_beta"],
            eps_tensor,
            work_np_dtype,
        )
        mlp_out = add_gelu_fc_projection(
            network,
            normed,
            matmul=matmul,
            weights=weights,
            prefix=prefix,
            hidden=hidden,
            mlp_size=mlp_size,
            dtype=work_np_dtype,
        )
        mlp_out = add_bias_sum(
            network,
            mlp_out,
            hidden,
            weights[f"{prefix}.fc2_bias"],
            dtype=work_np_dtype,
        )
        hidden_state = network.add_elementwise(
            residual, mlp_out, trt.ElementWiseOperation.SUM
        ).get_output(0)

    hidden_state = norm_multi(
        network,
        hidden_state,
        hidden,
        weights["final_norm"],
        weights["final_norm_beta"],
        eps_tensor,
        work_np_dtype,
    )
    logits = add_last_token_logits(network, hidden_state, hidden, weights["w_out"], work_np_dtype)
    if work_trt_dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    for layer_idx, (present_k, present_v) in enumerate(zip(present_k_outs, present_v_outs)):
        present_k.name = layer_tensor_name("present_k", layer_idx)
        present_v.name = layer_tensor_name("present_v", layer_idx)
        network.mark_output(present_k)
        network.mark_output(present_v)
    if verbose:
        print(
            f"[trtmc build] Building GPT-2 {profile_mode} engine "
            f"(layers={num_layers}, hidden={hidden}, mlp={mlp_size}, "
            f"cache={max_cache_length}, precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("dual-profile decoder engine build failed")
    return bytes(plan)
