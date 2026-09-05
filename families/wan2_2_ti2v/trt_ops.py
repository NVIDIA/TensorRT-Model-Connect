# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small plugin-free TensorRT graph vocabulary owned by Wan2.2 TI2V."""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from .model_config import SUPPORTED_GENERATION_PROFILES


def cast(network, tensor, dtype):
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def bf16_barrier(network, tensor):
    """Preserve the source BF16 value boundary with native TensorRT typing."""

    return cast(network, tensor, trt.bfloat16)


def constant(network, value, *, dtype=np.float32):
    array = np.ascontiguousarray(value, dtype=dtype)
    return network.add_constant(tuple(array.shape), array).get_output(0)


def linear(network, x, weight, bias=None, *, bf16=True):
    """Linear with PyTorch ``[out, in]`` weights."""

    rhs = constant(network, np.asarray(weight, dtype=np.float32).T)
    if bf16:
        x = cast(network, x, trt.bfloat16)
        rhs = cast(network, rhs, trt.bfloat16)
    y = network.add_matrix_multiply(
        x,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    if bias is not None:
        bias_tensor = constant(network, np.asarray(bias, dtype=np.float32).reshape(1, -1))
        bias_tensor = cast(network, bias_tensor, y.dtype)
        y = network.add_elementwise(y, bias_tensor, trt.ElementWiseOperation.SUM).get_output(0)
    return bf16_barrier(network, y) if bf16 else y


def fused_qkv_linear(
    network,
    x,
    q_weight,
    q_bias,
    k_weight,
    k_bias,
    v_weight,
    v_bias,
    *,
    rows: int,
    hidden_size: int,
):
    """Run self-attention Q/K/V as one BF16 linear and return Q, K, V rows."""

    weights = tuple(
        np.asarray(weight, dtype=np.float32) for weight in (q_weight, k_weight, v_weight)
    )
    biases = tuple(np.asarray(bias, dtype=np.float32) for bias in (q_bias, k_bias, v_bias))
    expected_weight_shape = (hidden_size, hidden_size)
    expected_bias_shape = (hidden_size,)
    if any(weight.shape != expected_weight_shape for weight in weights):
        raise ValueError(
            "Q/K/V weights must all have shape "
            f"{expected_weight_shape}; got {[weight.shape for weight in weights]}"
        )
    if any(bias.shape != expected_bias_shape for bias in biases):
        raise ValueError(
            "Q/K/V biases must all have shape "
            f"{expected_bias_shape}; got {[bias.shape for bias in biases]}"
        )

    packed = linear(
        network,
        x,
        np.concatenate(weights, axis=0),
        np.concatenate(biases, axis=0),
    )
    return tuple(
        network.add_slice(
            packed,
            (0, index * hidden_size),
            (rows, hidden_size),
            (1, 1),
        ).get_output(0)
        for index in range(3)
    )


def fp8_e4m3_weight_scale(weight) -> float:
    """Return the smallest finite per-tensor E4M3 scale for ``weight``."""

    maximum = float(np.max(np.abs(np.asarray(weight, dtype=np.float32))))
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("FP8 weight must have a positive finite absolute maximum")
    return maximum / 448.0


def _fp8_e4m3_tn(weight, scale: float):
    """Pre-quantize PyTorch ``[out, in]`` weights in TRT's FP8 TN layout."""

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("FP8 weight scale must be positive and finite")

    minimum_scale = fp8_e4m3_weight_scale(weight)
    if scale < minimum_scale * (1.0 - 1.0e-6):
        raise ValueError(
            f"FP8 weight scale would overflow E4M3: provided={scale}, minimum={minimum_scale}"
        )

    try:
        import ml_dtypes
    except ImportError as exc:
        raise RuntimeError("Wan2.2 FFN FP8 builds require the ml_dtypes build dependency") from exc

    # ``weight`` is already [out, in]. Keeping that storage and transposing
    # operand B at the matmul is the TN form that TensorRT fuses on Blackwell.
    scaled = np.ascontiguousarray(np.asarray(weight, dtype=np.float32) / scale)
    scaled = np.clip(scaled, -448.0, 448.0)
    return np.ascontiguousarray(scaled.astype(ml_dtypes.float8_e4m3fn))


def linear_fp8_e4m3(
    network,
    x,
    weight,
    bias=None,
    *,
    input_scale: float,
    weight_scale: float | None = None,
    weight_refs: list[np.ndarray] | None = None,
):
    """FFN linear using native TensorRT E4M3 Q/DQ and an FP8 TN weight.

    The output and bias boundary stay BF16, matching :func:`linear`. Only the
    matmul operands are quantized. ``weight_refs`` keeps the NumPy-owned FP8
    storage alive until TensorRT has finished serializing the network.
    """

    if not math.isfinite(input_scale) or input_scale <= 0.0:
        raise ValueError("FP8 input scale must be positive and finite")
    if weight_scale is None:
        weight_scale = fp8_e4m3_weight_scale(weight)
    fp8_weight = _fp8_e4m3_tn(weight, weight_scale)
    if weight_refs is not None:
        weight_refs.append(fp8_weight)

    x = cast(network, x, trt.bfloat16)
    input_scale_tensor = constant(network, np.asarray(input_scale, dtype=np.float32))
    quantized_x = network.add_quantize(x, input_scale_tensor, trt.DataType.FP8)
    dequantized_x = network.add_dequantize(
        quantized_x.get_output(0),
        input_scale_tensor,
        trt.bfloat16,
    )

    out_features, in_features = np.asarray(weight).shape
    fp8_weight_tensor = network.add_constant(
        (out_features, in_features),
        trt.Weights(
            trt.DataType.FP8,
            fp8_weight.ctypes.data,
            fp8_weight.size,
        ),
    ).get_output(0)
    weight_scale_tensor = constant(network, np.asarray(weight_scale, dtype=np.float32))
    dequantized_weight = network.add_dequantize(
        fp8_weight_tensor,
        weight_scale_tensor,
        trt.bfloat16,
    )

    y = network.add_matrix_multiply(
        dequantized_x.get_output(0),
        trt.MatrixOperation.NONE,
        dequantized_weight.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    if bias is not None:
        bias_tensor = constant(network, np.asarray(bias, dtype=np.float32).reshape(1, -1))
        bias_tensor = cast(network, bias_tensor, y.dtype)
        y = network.add_elementwise(y, bias_tensor, trt.ElementWiseOperation.SUM).get_output(0)
    return bf16_barrier(network, y)


def layer_norm(network, x, hidden_size: int, eps: float, *, round_bf16: bool = False):
    x = cast(network, x, trt.float32)
    gamma = constant(network, np.ones((1, hidden_size), dtype=np.float32))
    beta = constant(network, np.zeros((1, hidden_size), dtype=np.float32))
    norm = network.add_normalization_v2(x, gamma, beta, 1 << 1)
    norm.epsilon = eps
    output = norm.get_output(0)
    if round_bf16:
        output = cast(network, bf16_barrier(network, output), trt.float32)
    return output


def affine_layer_norm(network, x, weight, bias, hidden_size: int, eps: float):
    x = cast(network, x, trt.float32)
    gamma = constant(network, np.asarray(weight, dtype=np.float32).reshape(1, hidden_size))
    beta = constant(network, np.asarray(bias, dtype=np.float32).reshape(1, hidden_size))
    norm = network.add_normalization_v2(x, gamma, beta, 1 << 1)
    norm.epsilon = eps
    return norm.get_output(0)


def rms_norm(network, x, weight, hidden_size: int, eps: float):
    """Match WanRMSNorm's BF16-normalized, FP32-affine boundary."""

    x_fp32 = cast(network, x, trt.float32)
    squared = network.add_elementwise(x_fp32, x_fp32, trt.ElementWiseOperation.PROD).get_output(0)
    mean = network.add_reduce(squared, trt.ReduceOperation.AVG, 1 << 1, True).get_output(0)
    epsilon = constant(network, np.array([[eps]], dtype=np.float32))
    variance = network.add_elementwise(mean, epsilon, trt.ElementWiseOperation.SUM).get_output(0)
    inverse = network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
    inverse = network.add_unary(inverse, trt.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(x_fp32, inverse, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    normalized = cast(network, bf16_barrier(network, normalized), trt.float32)
    gamma = constant(network, np.asarray(weight, dtype=np.float32).reshape(1, hidden_size))
    return network.add_elementwise(normalized, gamma, trt.ElementWiseOperation.PROD).get_output(0)


def adaptive_norm(network, normalized, shift, scale):
    normalized = cast(network, normalized, trt.float32)
    one = constant(network, np.ones((1, 1), dtype=np.float32))
    scale_plus_one = network.add_elementwise(scale, one, trt.ElementWiseOperation.SUM).get_output(0)
    scaled = network.add_elementwise(
        normalized, scale_plus_one, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return network.add_elementwise(scaled, shift, trt.ElementWiseOperation.SUM).get_output(0)


def gelu_tanh(network, x):
    output = network.add_activation(x, trt.ActivationType.GELU_TANH).get_output(0)
    return bf16_barrier(network, output)


def silu(network, x):
    sigmoid = network.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
    return network.add_elementwise(x, sigmoid, trt.ElementWiseOperation.PROD).get_output(0)


def rows_to_heads(network, x, seq: int, heads: int, head_dim: int):
    reshape = network.add_shuffle(x)
    reshape.reshape_dims = (seq, heads, head_dim)
    reshape.second_transpose = trt.Permutation([1, 0, 2])
    batched = network.add_shuffle(reshape.get_output(0))
    batched.reshape_dims = (1, heads, seq, head_dim)
    return batched.get_output(0)


def heads_to_rows(network, x, seq: int, hidden_size: int):
    shuffle = network.add_shuffle(x)
    shuffle.first_transpose = trt.Permutation([0, 2, 1, 3])
    shuffle.reshape_dims = (seq, hidden_size)
    return shuffle.get_output(0)


def rotary(network, x, cos_half, sin_half, seq: int, heads: int, head_dim: int):
    """Apply native TensorRT RoPE and materialize its BF16 boundary."""

    x = cast(network, rows_to_heads(network, x, seq, heads, head_dim), trt.float32)
    cos_tensor = constant(
        network,
        np.asarray(cos_half, dtype=np.float32).reshape(1, seq, head_dim // 2),
    )
    sin_tensor = constant(
        network,
        np.asarray(sin_half, dtype=np.float32).reshape(1, seq, head_dim // 2),
    )
    layer = network.add_rotary_embedding(x, cos_tensor, sin_tensor, True, head_dim)
    if layer is None:
        raise RuntimeError("Could not add native TensorRT RoPE")
    rotated = bf16_barrier(network, layer.get_output(0))
    return heads_to_rows(network, rotated, seq, heads * head_dim)


def attention(
    network,
    q,
    k,
    v,
    *,
    q_seq: int,
    kv_seq: int,
    heads: int,
    head_dim: int,
):
    """Build non-decomposable native TensorRT attention for a qualified profile."""

    q = cast(network, q, v.dtype)
    k = cast(network, k, v.dtype)
    q4 = rows_to_heads(network, q, q_seq, heads, head_dim)
    k4 = rows_to_heads(network, k, kv_seq, heads, head_dim)
    v4 = rows_to_heads(network, v, kv_seq, heads, head_dim)

    qualified = any(
        q_seq == profile.num_patches
        and kv_seq in (profile.num_patches, profile.text_seq_len)
        and heads == profile.num_heads
        and head_dim == profile.head_dim
        for profile in SUPPORTED_GENERATION_PROFILES
    )
    if (
        not qualified
        or q4.dtype != trt.bfloat16
        or k4.dtype != trt.bfloat16
        or v4.dtype != trt.bfloat16
    ):
        raise ValueError("Native TensorRT attention requires a qualified Wan2.2 contract")

    factor = constant(
        network,
        np.array([[[[1.0 / math.sqrt(head_dim)]]]], dtype=np.float32),
    )
    factor = cast(network, factor, q4.dtype)
    q4 = network.add_elementwise(q4, factor, trt.ElementWiseOperation.PROD).get_output(0)
    layer = network.add_attention(
        q4,
        k4,
        v4,
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    if layer is None:
        raise RuntimeError("Could not add native TensorRT IAttention")
    layer.decomposable = False
    if layer.decomposable:
        raise RuntimeError("TensorRT did not retain non-decomposable IAttention")
    result = bf16_barrier(network, layer.get_output(0))
    return heads_to_rows(network, result, q_seq, heads * head_dim)


def add_fp32_residual(network, x, update, gate=None):
    x = cast(network, x, trt.float32)
    update = cast(network, update, trt.float32)
    if gate is not None:
        update = network.add_elementwise(update, gate, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(x, update, trt.ElementWiseOperation.SUM).get_output(0)
