# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTST-owned TensorRT graph helpers."""

from __future__ import annotations

import sys

import numpy as np

from tensorrt_model_connect import trt_compat

from ..weights import _target_np_dtype


trt = trt_compat.get_trt()


def _cast(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    dtype: trt.DataType,
) -> trt.ITensor:
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    layer = network.add_constant(
        shape,
        trt.Weights(np.ascontiguousarray(values, dtype=dtype)),
    )
    return layer.get_output(0)


def add_matmul_rhs_constant(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    rhs_weights: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    rank = len(tuple(lhs.shape))
    rhs_shape = (lhs_width, rhs_width) if rank <= 2 else (1,) * (rank - 2) + (lhs_width, rhs_width)
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(rhs_weights).reshape(rhs_shape),
        dtype=dtype,
    )
    rhs = _cast(network, rhs, lhs.dtype)

    output_dtype = lhs.dtype
    lhs = _cast(network, lhs, trt.float32)
    rhs = _cast(network, rhs, trt.float32)
    output = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    return _cast(network, output, output_dtype)


def add_bias_sum(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    bias: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    rank = len(tuple(inp.shape))
    shape = (width,) if rank <= 1 else (1,) * (rank - 1) + (width,)
    bias_t = add_constant(
        network,
        shape,
        np.asarray(bias).reshape(shape),
        dtype=dtype,
    )
    return network.add_elementwise(
        inp,
        _cast(network, bias_t, inp.dtype),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)


def add_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
) -> trt.ITensor:
    rank = len(tuple(inp.shape))
    shape = (hidden_size,) if rank <= 1 else (1,) * (rank - 1) + (hidden_size,)
    gamma_t = add_constant(network, shape, gamma.reshape(shape))
    beta_t = add_constant(network, shape, beta.reshape(shape))
    norm = network.add_normalization_v2(
        inp,
        _cast(network, gamma_t, inp.dtype),
        _cast(network, beta_t, inp.dtype),
        1 << (rank - 1),
    )
    norm.epsilon = eps
    if hasattr(norm, "compute_precision"):
        norm.compute_precision = trt.float32
    return norm.get_output(0)


def add_self_attention(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
    sequence_length: int,
) -> trt.ITensor:
    """Apply PatchTST's non-causal, equal-head self-attention."""

    def to_heads(tensor: trt.ITensor) -> trt.ITensor:
        rows = network.add_shuffle(tensor)
        rows.reshape_dims = (sequence_length, num_heads, head_dim)
        rows.second_transpose = trt.Permutation([1, 0, 2])
        batched = network.add_shuffle(rows.get_output(0))
        batched.reshape_dims = (1, num_heads, sequence_length, head_dim)
        return batched.get_output(0)

    q_heads = to_heads(q)
    k_heads = to_heads(k)
    v_heads = to_heads(v)
    scale_dtype = np.float16 if q_heads.dtype == trt.float16 else np.float32
    scale = add_constant(
        network,
        (1, 1, 1, 1),
        np.array([1.0 / np.sqrt(head_dim)], dtype=scale_dtype),
        dtype=scale_dtype,
    )
    q_heads = network.add_elementwise(
        q_heads,
        _cast(network, scale, q_heads.dtype),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    attention = network.add_attention(
        q_heads,
        k_heads,
        v_heads,
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    attention.decomposable = True
    rows = network.add_shuffle(attention.get_output(0))
    rows.first_transpose = trt.Permutation([0, 2, 1, 3])
    rows.reshape_dims = (sequence_length, num_heads * head_dim)
    return rows.get_output(0)


def maybe_return_replicated_tp_plan(weights: dict, parallel_config) -> bytes | None:
    if parallel_config is None or not getattr(parallel_config, "enabled", False):
        return None
    if int(getattr(parallel_config, "rank", -1)) > 0:
        return weights.get("_replicated_tp_engine_plan")
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
            f"[trtmc build] patchtst: building native TRT network "
            f"({network.num_layers} layers, precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT patchtst engine build failed")
    return bytes(plan)


def _target_trt_dtype(precision: str) -> trt.DataType:
    if precision == "fp16":
        return trt.float16
    if precision == "bf16":
        return trt.bfloat16
    return trt.float32


def add_linear(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight_out_in: np.ndarray,
    bias: np.ndarray | None,
    *,
    precision: str,
) -> trt.ITensor:
    target_np_dtype = _target_np_dtype(precision)
    inp = _cast(network, inp, _target_trt_dtype(precision))
    out_features, in_features = weight_out_in.shape
    out = add_matmul_rhs_constant(
        network,
        inp,
        int(in_features),
        int(out_features),
        np.ascontiguousarray(weight_out_in.T, dtype=target_np_dtype),
        dtype=target_np_dtype,
    )
    if bias is not None:
        out = add_bias_sum(
            network,
            out,
            int(out_features),
            np.ascontiguousarray(bias, dtype=target_np_dtype),
            dtype=target_np_dtype,
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


def _scalar_like(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    value: float,
) -> trt.ITensor:
    dtype = np.float16 if inp.dtype == trt.float16 else np.float32
    scalar = add_scalar(network, (1,) * len(tuple(inp.shape)), value, dtype=dtype)
    return _cast(network, scalar, inp.dtype)


def add_std_scale(
    network: trt.INetworkDefinition,
    data: trt.ITensor,
    observed: trt.ITensor,
    *,
    channels: int,
    minimum_scale: float,
) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    mask = _cast(network, observed, trt.float32)
    denominator = network.add_reduce(
        mask, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    ).get_output(0)
    denominator = network.add_elementwise(
        denominator,
        add_scalar(network, (1, 1, channels), 1.0),
        trt.ElementWiseOperation.MAX,
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
    squared = network.add_elementwise(
        centered_masked,
        centered_masked,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    variance_sum = network.add_reduce(
        squared, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    ).get_output(0)
    variance = network.add_elementwise(
        variance_sum, denominator, trt.ElementWiseOperation.DIV
    ).get_output(0)
    variance = network.add_elementwise(
        variance,
        add_scalar(network, (1, 1, channels), minimum_scale),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    scale = network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
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
    sequence_start = context_length - (patch_length + patch_stride * (num_patches - 1))
    if sequence_start < 0:
        raise ValueError("Patch configuration exceeds context length")

    channel_tensors: list[trt.ITensor] = []
    for channel in range(channels):
        patch_tensors: list[trt.ITensor] = []
        for patch_idx in range(num_patches):
            sliced = network.add_slice(
                values,
                start=(0, sequence_start + patch_idx * patch_stride, channel),
                shape=(1, patch_length, 1),
                stride=(1, 1, 1),
            ).get_output(0)
            patch = network.add_shuffle(sliced)
            patch.first_transpose = (0, 2, 1)
            patch.reshape_dims = (1, 1, 1, patch_length)
            patch_tensors.append(patch.get_output(0))
        patches = network.add_concatenation(patch_tensors)
        patches.axis = 2
        channel_tensors.append(patches.get_output(0))
    result = network.add_concatenation(channel_tensors)
    result.axis = 1
    return result.get_output(0)


def add_batch_norm_last_dim(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    width: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float,
) -> trt.ITensor:
    output_dtype = inp.dtype
    inp = _cast(network, inp, trt.float32)
    denominator = np.sqrt(np.asarray(running_var, dtype=np.float32) + eps)
    scale = np.asarray(gamma, dtype=np.float32) / denominator
    bias = np.asarray(beta, dtype=np.float32) - np.asarray(running_mean, dtype=np.float32) * scale
    rank = len(tuple(inp.shape))
    shape = (width,) if rank <= 1 else (1,) * (rank - 1) + (width,)
    scaled = network.add_elementwise(
        inp,
        add_constant(network, shape, scale.reshape(shape)),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    output = network.add_elementwise(
        scaled,
        add_constant(network, shape, bias.reshape(shape)),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    return _cast(network, output, output_dtype)


def add_named_output(network: trt.INetworkDefinition, tensor: trt.ITensor, name: str) -> None:
    tensor.name = name
    network.mark_output(tensor)


def add_gelu(network: trt.INetworkDefinition, inp: trt.ITensor) -> trt.ITensor:
    if hasattr(trt.UnaryOperation, "ERF"):
        scaled = network.add_elementwise(
            inp,
            _scalar_like(network, inp, 1.0 / np.sqrt(2.0)),
            trt.ElementWiseOperation.PROD,
        ).get_output(0)
        erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
        one_plus = network.add_elementwise(
            erf,
            _scalar_like(network, inp, 1.0),
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        half_x = network.add_elementwise(
            inp,
            _scalar_like(network, inp, 0.5),
            trt.ElementWiseOperation.PROD,
        ).get_output(0)
        return network.add_elementwise(half_x, one_plus, trt.ElementWiseOperation.PROD).get_output(
            0
        )

    squared = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD).get_output(0)
    cubed = network.add_elementwise(squared, inp, trt.ElementWiseOperation.PROD).get_output(0)
    cubed = network.add_elementwise(
        cubed,
        _scalar_like(network, inp, 0.044715),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    inner = network.add_elementwise(inp, cubed, trt.ElementWiseOperation.SUM).get_output(0)
    inner = network.add_elementwise(
        inner,
        _scalar_like(network, inp, np.sqrt(2.0 / np.pi)),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    tanh = network.add_activation(inner, trt.ActivationType.TANH).get_output(0)
    tanh = network.add_elementwise(
        tanh,
        _scalar_like(network, inp, 1.0),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    half_x = network.add_elementwise(
        inp,
        _scalar_like(network, inp, 0.5),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    return network.add_elementwise(half_x, tanh, trt.ElementWiseOperation.PROD).get_output(0)


def add_squareplus(network: trt.INetworkDefinition, inp: trt.ITensor) -> trt.ITensor:
    squared = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD).get_output(0)
    rooted = network.add_unary(
        network.add_elementwise(
            squared,
            _scalar_like(network, inp, 4.0),
            trt.ElementWiseOperation.SUM,
        ).get_output(0),
        trt.UnaryOperation.SQRT,
    ).get_output(0)
    summed = network.add_elementwise(inp, rooted, trt.ElementWiseOperation.SUM).get_output(0)
    return network.add_elementwise(
        summed,
        _scalar_like(network, inp, 0.5),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
