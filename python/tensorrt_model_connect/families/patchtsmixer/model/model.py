"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations


import numpy as np
from tensorrt_model_connect import trt_compat
import sys
from ..weights import (
    _target_np_dtype,
)
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


# Backward-compatible name used by existing tests and call sites.
_add_attention_core = add_attention_core


# Time Series Trt


trt = trt_compat.get_trt()


def maybe_return_replicated_tp_plan(weights: dict, parallel_config) -> bytes | None:
    if parallel_config is None or not getattr(parallel_config, "enabled", False):
        return None
    rank = int(getattr(parallel_config, "rank", -1))
    if rank > 0 and "_replicated_tp_engine_plan" in weights:
        return weights["_replicated_tp_engine_plan"]
    return None


def cache_replicated_tp_plan(weights: dict, parallel_config, plan: bytes) -> None:
    if parallel_config is not None and getattr(parallel_config, "enabled", False):
        weights["_replicated_tp_engine_plan"] = plan


def build_serialized_network(
    builder: trt.Builder,
    network: trt.INetworkDefinition,
    *,
    precision: str,
    verbose: bool = False,
    tag: str = "time_series",
) -> bytes:
    config = builder.create_builder_config()
    config.avg_timing_iterations = 8
    config.max_aux_streams = 0
    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if precision == "fp32" and hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    if precision in {"fp16", "bf16"}:
        config.set_flag(trt.BuilderFlag.FP16)

    if verbose:
        print(
            f"[trtmc build] {tag}: building native TRT network "
            f"({network.num_layers} layers, precision={precision}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT {tag} engine build failed")
    return bytes(plan)


def create_network(*, verbose: bool = False) -> tuple[trt.Builder, trt.INetworkDefinition]:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    return builder, network


def add_linear(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight_out_in: np.ndarray,
    bias: np.ndarray | None,
    *,
    precision: str = "fp32",
) -> trt.ITensor:
    target_dtype = _target_np_dtype(precision)
    w = np.ascontiguousarray(weight_out_in.T, dtype=target_dtype)
    out_features = int(weight_out_in.shape[0])
    out = add_matmul_rhs_constant(
        network,
        inp,
        int(weight_out_in.shape[1]),
        out_features,
        w,
        dtype=target_dtype,
    )
    if bias is not None:
        out = add_bias_sum(
            network,
            out,
            out_features,
            np.ascontiguousarray(bias, dtype=target_dtype),
            dtype=target_dtype,
        )
    return out


def add_scalar(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    value: float,
    *,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    return add_constant(
        network,
        shape,
        np.full(shape, value, dtype=dtype),
        dtype=dtype,
    )


def add_std_scale(
    network: trt.INetworkDefinition,
    data: trt.ITensor,
    observed: trt.ITensor,
    *,
    channels: int,
    minimum_scale: float,
) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    mask = network.add_cast(observed, trt.float32).get_output(0)
    denominator = network.add_reduce(
        mask, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    ).get_output(0)
    one = add_scalar(network, (1, 1, channels), 1.0)
    denominator = network.add_elementwise(
        denominator, one, trt.ElementWiseOperation.MAX
    ).get_output(0)

    masked = network.add_elementwise(data, mask, trt.ElementWiseOperation.PROD).get_output(0)
    summed = network.add_reduce(masked, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True).get_output(
        0
    )
    loc = network.add_elementwise(summed, denominator, trt.ElementWiseOperation.DIV).get_output(0)

    centered = network.add_elementwise(data, loc, trt.ElementWiseOperation.SUB).get_output(0)
    centered_masked = network.add_elementwise(
        centered, mask, trt.ElementWiseOperation.PROD
    ).get_output(0)
    sq = network.add_elementwise(
        centered_masked, centered_masked, trt.ElementWiseOperation.PROD
    ).get_output(0)
    var_sum = network.add_reduce(sq, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True).get_output(0)
    var = network.add_elementwise(var_sum, denominator, trt.ElementWiseOperation.DIV).get_output(0)
    eps = add_scalar(network, (1, 1, channels), minimum_scale)
    var_eps = network.add_elementwise(var, eps, trt.ElementWiseOperation.SUM).get_output(0)
    scale = network.add_unary(var_eps, trt.UnaryOperation.SQRT).get_output(0)
    scaled = network.add_elementwise(centered, scale, trt.ElementWiseOperation.DIV).get_output(0)
    return scaled, loc, scale


def add_patchify(
    network: trt.INetworkDefinition,
    values: trt.ITensor,
    *,
    context_length: int,
    channels: int,
    patch_length: int,
    patch_stride: int,
    num_patches: int,
) -> trt.ITensor:
    new_sequence_length = patch_length + patch_stride * (num_patches - 1)
    sequence_start = context_length - new_sequence_length
    if sequence_start < 0:
        raise ValueError("Patch configuration exceeds context length")

    channel_tensors: list[trt.ITensor] = []
    for channel in range(channels):
        patch_tensors: list[trt.ITensor] = []
        for patch_idx in range(num_patches):
            start = sequence_start + patch_idx * patch_stride
            sliced = network.add_slice(
                values,
                start=(0, start, channel),
                shape=(1, patch_length, 1),
                stride=(1, 1, 1),
            ).get_output(0)
            shuf = network.add_shuffle(sliced)
            shuf.first_transpose = (0, 2, 1)
            shuf.reshape_dims = (1, 1, 1, patch_length)
            patch_tensors.append(shuf.get_output(0))
        cat_patches = network.add_concatenation(patch_tensors)
        cat_patches.axis = 2
        channel_tensors.append(cat_patches.get_output(0))
    cat_channels = network.add_concatenation(channel_tensors)
    cat_channels.axis = 1
    return cat_channels.get_output(0)


def add_named_output(network: trt.INetworkDefinition, tensor: trt.ITensor, name: str) -> None:
    tensor.name = name
    network.mark_output(tensor)


def add_gelu(network: trt.INetworkDefinition, inp: trt.ITensor) -> trt.ITensor:
    if hasattr(trt.UnaryOperation, "ERF"):
        inv_sqrt2 = add_scalar(
            network, (1,) * len(tuple(inp.shape)), 1.0 / np.sqrt(2.0), dtype=np.float32
        )
        half = add_scalar(network, (1,) * len(tuple(inp.shape)), 0.5, dtype=np.float32)
        one = add_scalar(network, (1,) * len(tuple(inp.shape)), 1.0, dtype=np.float32)
        scaled = network.add_elementwise(inp, inv_sqrt2, trt.ElementWiseOperation.PROD).get_output(
            0
        )
        erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
        one_plus = network.add_elementwise(erf, one, trt.ElementWiseOperation.SUM).get_output(0)
        half_x = network.add_elementwise(inp, half, trt.ElementWiseOperation.PROD).get_output(0)
        return network.add_elementwise(half_x, one_plus, trt.ElementWiseOperation.PROD).get_output(
            0
        )
    return add_gelu_new(network, inp, dtype=np.float32)
