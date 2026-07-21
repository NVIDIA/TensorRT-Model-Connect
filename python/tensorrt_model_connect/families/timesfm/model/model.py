# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations

import sys

import numpy as np
from tensorrt_model_connect import trt_compat

from ..weights import _target_np_dtype


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
    rhs_shape = (lhs_width, rhs_width) if rank <= 2 else (1,) * (rank - 2) + (lhs_width, rhs_width)
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
            lhs,
            trt.MatrixOperation.NONE,
            rhs,
            trt.MatrixOperation.NONE,
        )
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


def add_rms_norm_last_dim(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """RMSNorm over the final dimension for any-rank tensors.

    Generalises :func:`add_rms_norm` (rank 2) to inputs of rank >= 2 by
    reducing over the last axis. Diffusion batched builders use this for
    ``[B, S, D]`` tensors.
    """
    rank = len(tuple(inp.shape))
    if rank < 2:
        raise ValueError("add_rms_norm_last_dim expects rank >= 2")

    need_cast = dtype != np.float32
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)

    sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << (rank - 1), keep_dims=True
    )
    denom_in = network.add_elementwise(mean.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(inp, recip.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_shape = (1,) * (rank - 1) + (hidden_size,)
    gamma_t = add_constant(
        network, gamma_shape, np.asarray(gamma).reshape(gamma_shape), dtype=np.float32
    )
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
    result = scaled.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def add_silu(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
) -> trt.ITensor:
    """SiLU (Swish): x * sigmoid(x)."""
    sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID)
    return network.add_elementwise(
        inp, sigmoid.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)


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
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)

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
    target_trt_dtype = trt.float16 if target_dtype == np.float16 else trt.float32
    if inp.dtype != target_trt_dtype:
        inp = network.add_cast(inp, target_trt_dtype).get_output(0)
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


def add_named_output(network: trt.INetworkDefinition, tensor: trt.ITensor, name: str) -> None:
    tensor.name = name
    network.mark_output(tensor)
