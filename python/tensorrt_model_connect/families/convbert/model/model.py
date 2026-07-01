# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ConvBERT TensorRT graph construction and tensor-parallel sharding."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from ....parallel_config import add_all_reduce_sum, normalize_parallel_config
from ..config import ModelConfig

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ....parallel_config import ParallelConfig
    from ..weights import WeightDict


def _constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    return network.add_constant(
        shape, trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    ).get_output(0)


def _cast_like(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target: trt.ITensor,
) -> trt.ITensor:
    if tensor.dtype == target.dtype:
        return tensor
    return network.add_cast(tensor, target.dtype).get_output(0)


def _linear(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    in_features: int,
    out_features: int,
    weight: np.ndarray,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    rank = len(tuple(inp.shape))
    shape = (1,) * max(0, rank - 2) + (in_features, out_features)
    rhs = _constant(network, shape, np.asarray(weight).reshape(shape), dtype)
    rhs = _cast_like(network, rhs, inp)
    return network.add_matrix_multiply(
        inp, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
    ).get_output(0)


def _bias(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    values: np.ndarray,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    rank = len(tuple(inp.shape))
    shape = (1,) * max(0, rank - 1) + (width,)
    bias = _constant(network, shape, np.asarray(values).reshape(shape), dtype)
    return network.add_elementwise(
        inp, _cast_like(network, bias, inp), trt.ElementWiseOperation.SUM
    ).get_output(0)


def _gelu(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    shape = (1,) * max(1, len(tuple(inp.shape)))

    def scalar(value: float) -> trt.ITensor:
        value_t = _constant(network, shape, np.array([value]), dtype)
        return _cast_like(network, value_t, inp)

    squared = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD).get_output(0)
    cubed = network.add_elementwise(squared, inp, trt.ElementWiseOperation.PROD).get_output(0)
    cubic_term = network.add_elementwise(
        cubed, scalar(0.044715), trt.ElementWiseOperation.PROD
    ).get_output(0)
    inner = network.add_elementwise(inp, cubic_term, trt.ElementWiseOperation.SUM).get_output(0)
    scaled = network.add_elementwise(
        inner, scalar(float(np.sqrt(2.0 / np.pi))), trt.ElementWiseOperation.PROD
    ).get_output(0)
    tanh = network.add_activation(scaled, trt.ActivationType.TANH).get_output(0)
    shifted = network.add_elementwise(tanh, scalar(1.0), trt.ElementWiseOperation.SUM).get_output(0)
    half = network.add_elementwise(inp, scalar(0.5), trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(half, shifted, trt.ElementWiseOperation.PROD).get_output(0)


def _activation(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    name: str,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    if name in {"gelu", "gelu_new"}:
        return _gelu(network, inp, dtype=dtype)
    if name == "relu":
        return network.add_activation(inp, trt.ActivationType.RELU).get_output(0)
    if name == "silu":
        sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID).get_output(0)
        return network.add_elementwise(inp, sigmoid, trt.ElementWiseOperation.PROD).get_output(0)
    raise ValueError(f"Unsupported ConvBERT activation: {name}")


def _layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    rank = len(tuple(inp.shape))
    shape = (1,) * max(0, rank - 1) + (width,)
    scale = _cast_like(
        network, _constant(network, shape, np.asarray(gamma).reshape(shape), dtype), inp
    )
    bias = _cast_like(
        network, _constant(network, shape, np.asarray(beta).reshape(shape), dtype), inp
    )
    layer = network.add_normalization_v2(inp, scale, bias, 1 << (rank - 1))
    layer.epsilon = eps
    if hasattr(layer, "compute_precision"):
        layer.compute_precision = trt.float32
    return layer.get_output(0)


def _rows_to_heads(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_size: int,
    sequence_length: int,
) -> trt.ITensor:
    rows = network.add_shuffle(inp)
    rows.reshape_dims = (sequence_length, num_heads, head_size)
    rows.second_transpose = trt.Permutation([1, 0, 2])
    batched = network.add_shuffle(rows.get_output(0))
    batched.reshape_dims = (1, num_heads, sequence_length, head_size)
    return batched.get_output(0)


def _heads_to_rows(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    sequence_length: int,
) -> trt.ITensor:
    rows = network.add_shuffle(inp)
    rows.first_transpose = trt.Permutation([0, 2, 1, 3])
    rows.reshape_dims = (sequence_length, width)
    return rows.get_output(0)


def _attention_mask(
    network: trt.INetworkDefinition,
    mask: trt.ITensor,
) -> trt.ITensor:
    shape = network.add_shape(mask).get_output(0)
    ones = _constant(network, (2,), np.array([1, 1], dtype=np.int64), np.int64)
    target = network.add_concatenation([ones, shape])
    target.axis = 0
    expanded = network.add_shuffle(mask)
    expanded.set_input(1, target.get_output(0))
    return expanded.get_output(0)


def _self_attention(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    *,
    num_heads: int,
    head_size: int,
    sequence_length: int,
    mask: trt.ITensor,
) -> trt.ITensor:
    q_4d = _rows_to_heads(network, q, num_heads, head_size, sequence_length)
    k_4d = _rows_to_heads(network, k, num_heads, head_size, sequence_length)
    v_4d = _rows_to_heads(network, v, num_heads, head_size, sequence_length)
    scale_dtype = np.float16 if q_4d.dtype == trt.float16 else np.float32
    scale = _constant(
        network,
        (1, 1, 1, 1),
        np.array([1.0 / np.sqrt(head_size)]),
        scale_dtype,
    )
    scaled_q = network.add_elementwise(
        q_4d, _cast_like(network, scale, q_4d), trt.ElementWiseOperation.PROD
    ).get_output(0)
    layer = network.add_attention(
        scaled_q,
        k_4d,
        v_4d,
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    layer.decomposable = True
    layer.mask = mask
    return _heads_to_rows(network, layer.get_output(0), num_heads * head_size, sequence_length)


def _slice_last(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _slice_output_rows(
    arr: np.ndarray,
    *,
    rank: int,
    tp_size: int,
    all_head_size: int,
) -> np.ndarray:
    local = all_head_size // tp_size
    start = rank * local
    return np.ascontiguousarray(
        np.concatenate(
            [
                arr[start : start + local],
                arr[all_head_size + start : all_head_size + start + local],
            ],
            axis=0,
        )
    )


def _validate_convbert_tp(
    config: ModelConfig,
    weights: WeightDict,
    parallel: ParallelConfig,
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("ConvBERT tensor-parallel build requires a concrete rank")

    tp_size = parallel.tp_size
    new_num_heads = int(weights["_convbert_new_num_heads"][0])
    all_head_size = int(weights["_convbert_all_head_size"][0])
    for name, value in (
        ("effective attention heads", new_num_heads),
        ("all_head_size", all_head_size),
        ("intermediate_size", config.intermediate_size),
    ):
        if value % tp_size:
            raise ValueError(
                f"ConvBERT tensor parallel requires {name} divisible by "
                f"tp_size ({value} vs {tp_size})"
            )

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layer.{layer_idx}"
        for suffix in ("w_q", "w_k", "w_v", "conv_out_w", "w_fc1"):
            key = f"{prefix}.{suffix}"
            if weights[key].shape[-1] % tp_size:
                raise ValueError(f"{key} output dim must be divisible by tp_size")
        for suffix in ("q_bias", "k_bias", "v_bias", "w_fc2"):
            key = f"{prefix}.{suffix}"
            if weights[key].shape[0] % tp_size:
                raise ValueError(f"{key} dim must be divisible by tp_size")


def shard_convbert_weights(
    config: ModelConfig,
    weights: WeightDict,
    *,
    parallel: ParallelConfig,
) -> WeightDict:
    """Return the rank-local ConvBERT projection and convolution weights."""
    _validate_convbert_tp(config, weights, parallel)
    if not parallel.enabled:
        return weights

    full_heads = int(weights["_convbert_new_num_heads"][0])
    full_all = int(weights["_convbert_all_head_size"][0])
    output_last = (".w_q", ".w_k", ".w_v", ".conv_out_w", ".w_fc1")
    output_first = (
        ".q_bias",
        ".k_bias",
        ".v_bias",
        ".sep_conv_pw",
        ".sep_conv_bias",
        ".conv_out_bias",
        ".fc1_bias",
        ".conv_kernel_w",
        ".w_fc2",
    )

    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
        elif key.endswith(output_last):
            out[key] = _slice_last(value, parallel.rank, parallel.tp_size)
        elif key.endswith(output_first):
            out[key] = _slice_first(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".w_o"):
            out[key] = _slice_output_rows(
                value,
                rank=parallel.rank,
                tp_size=parallel.tp_size,
                all_head_size=full_all,
            )
        else:
            out[key] = value

    out["_convbert_full_new_num_heads"] = np.array([full_heads], dtype=np.int32)
    out["_convbert_full_all_head_size"] = np.array([full_all], dtype=np.int32)
    out["_convbert_new_num_heads"] = np.array([full_heads // parallel.tp_size], dtype=np.int32)
    out["_convbert_all_head_size"] = np.array([full_all // parallel.tp_size], dtype=np.int32)
    out["_intermediate_size"] = config.intermediate_size // parallel.tp_size
    out["_tensor_parallel_size"] = parallel.tp_size
    out["_tensor_parallel_rank"] = parallel.rank
    return out


def _separable_conv1d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    hidden_size: int,
    all_head_size: int,
    kernel_size: int,
    sequence_length: int,
    depthwise: np.ndarray,
    pointwise: np.ndarray,
    bias: np.ndarray,
    dtype: np.dtype,
) -> trt.ITensor:
    rows = network.add_shuffle(inp)
    rows.first_transpose = trt.Permutation([1, 0])
    rows.reshape_dims = (1, hidden_size, sequence_length, 1)
    depthwise_layer = network.add_convolution_nd(
        rows.get_output(0),
        hidden_size,
        (kernel_size, 1),
        trt.Weights(
            np.ascontiguousarray(depthwise.reshape(hidden_size, 1, kernel_size, 1), dtype=dtype)
        ),
    )
    depthwise_layer.padding_nd = (kernel_size // 2, 0)
    depthwise_layer.num_groups = hidden_size
    pointwise_layer = network.add_convolution_nd(
        depthwise_layer.get_output(0),
        all_head_size,
        (1, 1),
        trt.Weights(
            np.ascontiguousarray(pointwise.reshape(all_head_size, hidden_size, 1, 1), dtype=dtype)
        ),
    )
    bias_4d = _constant(
        network,
        (1, all_head_size, 1, 1),
        bias.reshape(1, all_head_size, 1, 1),
        dtype,
    )
    biased = network.add_elementwise(
        pointwise_layer.get_output(0),
        _cast_like(network, bias_4d, pointwise_layer.get_output(0)),
        trt.ElementWiseOperation.SUM,
    )
    result = network.add_shuffle(biased.get_output(0))
    result.reshape_dims = (all_head_size, sequence_length)
    result.second_transpose = trt.Permutation([1, 0])
    return result.get_output(0)


def _unfold(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    channels: int,
    sequence_length: int,
    kernel_size: int,
) -> trt.ITensor:
    expanded = network.add_shuffle(inp)
    expanded.reshape_dims = (1, channels, sequence_length, 1)
    inp_4d = expanded.get_output(0)
    shifted = []
    for kernel_idx in range(kernel_size):
        offset = kernel_idx - kernel_size // 2
        if offset == 0:
            value = inp_4d
        else:
            before, after = (-offset, 0) if offset < 0 else (0, offset)
            padded = network.add_padding_nd(
                inp_4d, pre_padding=(before, 0), post_padding=(after, 0)
            )
            value = network.add_slice(
                padded.get_output(0),
                start=(0, 0, max(offset, 0), 0),
                shape=(1, channels, sequence_length, 1),
                stride=(1, 1, 1, 1),
            ).get_output(0)
        rows = network.add_shuffle(value)
        rows.reshape_dims = (channels, sequence_length)
        shifted.append(rows.get_output(0))
    if len(shifted) == 1:
        return shifted[0]
    result = network.add_concatenation(shifted)
    result.axis = 0
    return result.get_output(0)


def _convbert_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    intermediate_size: int,
    new_num_heads: int,
    full_new_num_heads: int,
    head_size: int,
    all_head_size: int,
    kernel_size: int,
    sequence_length: int,
    attention_mask: trt.ITensor,
    hidden_act: str,
    eps: float,
    tp_size: int,
    tp_rank: int,
    dtype: np.dtype,
) -> trt.ITensor:
    def linear(inp, in_features, out_features, suffix):
        return _linear(
            network,
            inp,
            in_features,
            out_features,
            weights[f"{prefix}.{suffix}"],
            dtype=dtype,
        )

    def bias(inp, width, suffix):
        return _bias(network, inp, width, weights[f"{prefix}.{suffix}"], dtype=dtype)

    q = bias(linear(hidden, hidden_size, all_head_size, "w_q"), all_head_size, "q_bias")
    k = bias(linear(hidden, hidden_size, all_head_size, "w_k"), all_head_size, "k_bias")
    v = bias(linear(hidden, hidden_size, all_head_size, "w_v"), all_head_size, "v_bias")

    mask_row = network.add_shuffle(attention_mask)
    mask_row.reshape_dims = (1, sequence_length)
    zero_col = _constant(
        network,
        (sequence_length, 1),
        np.zeros((sequence_length, 1)),
        dtype,
    )
    mask_2d = network.add_elementwise(
        zero_col, mask_row.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)
    context = _self_attention(
        network,
        q,
        k,
        v,
        num_heads=new_num_heads,
        head_size=head_size,
        sequence_length=sequence_length,
        mask=_attention_mask(network, mask_2d),
    )
    attention_branch = network.add_shuffle(context)
    attention_branch.reshape_dims = (sequence_length, new_num_heads, head_size)

    key_conv = _separable_conv1d(
        network,
        hidden,
        hidden_size=hidden_size,
        all_head_size=all_head_size,
        kernel_size=kernel_size,
        sequence_length=sequence_length,
        depthwise=weights[f"{prefix}.sep_conv_dw"],
        pointwise=weights[f"{prefix}.sep_conv_pw"],
        bias=weights[f"{prefix}.sep_conv_bias"],
        dtype=dtype,
    )
    conv_attention = network.add_elementwise(key_conv, q, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    full_kernel_width = full_new_num_heads * kernel_size
    conv_kernel = linear(conv_attention, all_head_size, full_kernel_width, "conv_kernel_w")
    if tp_size > 1:
        conv_kernel = add_all_reduce_sum(network, conv_kernel, tp_size)
    conv_kernel = bias(conv_kernel, full_kernel_width, "conv_kernel_bias")
    if tp_size > 1:
        conv_kernel = network.add_slice(
            conv_kernel,
            start=(0, tp_rank * new_num_heads * kernel_size),
            shape=(sequence_length, new_num_heads * kernel_size),
            stride=(1, 1),
        ).get_output(0)

    kernel_rows = network.add_shuffle(conv_kernel)
    kernel_rows.reshape_dims = (sequence_length * new_num_heads, kernel_size, 1)
    kernel_probs = network.add_softmax(kernel_rows.get_output(0))
    kernel_probs.axes = 1 << 1

    conv_out = bias(
        linear(hidden, hidden_size, all_head_size, "conv_out_w"),
        all_head_size,
        "conv_out_bias",
    )
    channel_first = network.add_shuffle(conv_out)
    channel_first.first_transpose = trt.Permutation([1, 0])
    unfolded = _unfold(
        network,
        channel_first.get_output(0),
        channels=all_head_size,
        sequence_length=sequence_length,
        kernel_size=kernel_size,
    )
    reordered = network.add_shuffle(unfolded)
    reordered.reshape_dims = (kernel_size, all_head_size, sequence_length)
    reordered.second_transpose = trt.Permutation([1, 0, 2])
    sequence_first = network.add_shuffle(reordered.get_output(0))
    sequence_first.first_transpose = trt.Permutation([2, 0, 1])
    convolution_rows = network.add_shuffle(sequence_first.get_output(0))
    convolution_rows.reshape_dims = (
        sequence_length * new_num_heads,
        head_size,
        kernel_size,
    )
    convolution = network.add_matrix_multiply(
        convolution_rows.get_output(0),
        trt.MatrixOperation.NONE,
        kernel_probs.get_output(0),
        trt.MatrixOperation.NONE,
    )
    convolution_branch = network.add_shuffle(convolution.get_output(0))
    convolution_branch.reshape_dims = (sequence_length, new_num_heads, head_size)

    combined = network.add_concatenation(
        [attention_branch.get_output(0), convolution_branch.get_output(0)]
    )
    combined.axis = 1
    flattened = network.add_shuffle(combined.get_output(0))
    flattened.reshape_dims = (sequence_length, 2 * all_head_size)
    attention_out = linear(flattened.get_output(0), 2 * all_head_size, hidden_size, "w_o")
    if tp_size > 1:
        attention_out = add_all_reduce_sum(network, attention_out, tp_size)
    attention_out = bias(attention_out, hidden_size, "o_bias")
    residual = network.add_elementwise(
        hidden, attention_out, trt.ElementWiseOperation.SUM
    ).get_output(0)
    normalized = _layer_norm(
        network,
        residual,
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        weights[f"{prefix}.post_attn_norm_beta"],
        eps,
        dtype=dtype,
    )

    intermediate = bias(
        linear(normalized, hidden_size, intermediate_size, "w_fc1"),
        intermediate_size,
        "fc1_bias",
    )
    activated = _activation(network, intermediate, hidden_act, dtype=dtype)
    output = linear(activated, intermediate_size, hidden_size, "w_fc2")
    if tp_size > 1:
        output = add_all_reduce_sum(network, output, tp_size)
    output = bias(output, hidden_size, "fc2_bias")
    residual = network.add_elementwise(normalized, output, trt.ElementWiseOperation.SUM).get_output(
        0
    )
    return _layer_norm(
        network,
        residual,
        hidden_size,
        weights[f"{prefix}.output_norm"],
        weights[f"{prefix}.output_norm_beta"],
        eps,
        dtype=dtype,
    )


def build_convbert_encoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_seq_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build a serial or rank-local ConvBERT encoder engine."""
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        if precision != "fp32":
            raise ValueError("ConvBERT tensor-parallel builds support fp32 only")
        weights = shard_convbert_weights(config, weights, parallel=parallel)
        tp_size, tp_rank = parallel.tp_size, parallel.rank
    else:
        tp_size, tp_rank = 1, 0

    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Unsupported ConvBERT precision: {precision}")

    hidden_size = config.hidden_size
    embedding_size = int(config.raw.get("embedding_size", hidden_size))
    type_vocab_size = int(config.raw.get("type_vocab_size", 2))
    num_heads = int(weights["_convbert_new_num_heads"][0])
    full_num_heads = int(weights.get("_convbert_full_new_num_heads", [num_heads])[0])
    head_size = int(weights["_convbert_head_size"][0])
    all_head_size = int(weights["_convbert_all_head_size"][0])
    kernel_size = int(weights["_convbert_conv_kernel_size"][0])
    intermediate_size = config.intermediate_size // tp_size

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    builder_config.clear_flag(trt.BuilderFlag.TF32)

    sequence_length = max_seq_length
    input_ids = network.add_input("input_ids", trt.int32, (sequence_length,))
    attention_mask_input = network.add_input("attention_mask", trt.int32, (sequence_length,))
    token_type_ids = network.add_constant(
        (sequence_length,), trt.Weights(np.zeros(sequence_length, dtype=np.int32))
    ).get_output(0)

    embedding = _constant(
        network,
        (config.vocab_size, embedding_size),
        weights["embedding"],
        work_np_dtype,
    )
    position_embedding = _constant(
        network,
        weights["position_embedding"].shape,
        weights["position_embedding"],
        work_np_dtype,
    )
    token_type_embedding = _constant(
        network,
        (type_vocab_size, embedding_size),
        weights["token_type_embedding"],
        work_np_dtype,
    )

    mask = network.add_cast(attention_mask_input, work_trt_dtype).get_output(0)
    one = _constant(network, (1,), np.array([1.0]), work_np_dtype)
    penalty = _constant(
        network,
        (1,),
        np.array([-1e4 if precision == "fp16" else -1e10]),
        work_np_dtype,
    )
    inverted = network.add_elementwise(one, mask, trt.ElementWiseOperation.SUB).get_output(0)
    additive_mask = network.add_elementwise(
        inverted, penalty, trt.ElementWiseOperation.PROD
    ).get_output(0)
    mask_3d = network.add_shuffle(additive_mask)
    mask_3d.reshape_dims = (1, 1, sequence_length)

    positions = network.add_constant(
        (sequence_length,),
        trt.Weights(np.arange(sequence_length, dtype=np.int32)),
    ).get_output(0)
    word = network.add_gather(embedding, input_ids, 0).get_output(0)
    position = network.add_gather(position_embedding, positions, 0).get_output(0)
    token_type = network.add_gather(token_type_embedding, token_type_ids, 0).get_output(0)
    hidden = network.add_elementwise(word, position, trt.ElementWiseOperation.SUM).get_output(0)
    hidden = network.add_elementwise(hidden, token_type, trt.ElementWiseOperation.SUM).get_output(0)
    hidden = _layer_norm(
        network,
        hidden,
        embedding_size,
        weights["embed_norm"],
        weights["embed_norm_beta"],
        config.rms_norm_eps,
        dtype=work_np_dtype,
    )

    for layer_idx in range(config.num_hidden_layers):
        hidden = _convbert_layer(
            network=network,
            hidden=hidden,
            weights=weights,
            prefix=f"layer.{layer_idx}",
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            new_num_heads=num_heads,
            full_new_num_heads=full_num_heads,
            head_size=head_size,
            all_head_size=all_head_size,
            kernel_size=kernel_size,
            sequence_length=sequence_length,
            attention_mask=mask_3d.get_output(0),
            hidden_act=config.hidden_act or "gelu",
            eps=config.rms_norm_eps,
            tp_size=tp_size,
            tp_rank=tp_rank,
            dtype=work_np_dtype,
        )

    if hidden.dtype != trt.float32:
        hidden = network.add_cast(hidden, trt.float32).get_output(0)
    hidden.name = "hidden_states"
    network.mark_output(hidden)

    if verbose:
        print(
            f"[trtmc build] Building ConvBERT encoder TRT engine "
            f"({config.num_hidden_layers} layers, hidden={hidden_size}, "
            f"tp={tp_size}, seq_len={sequence_length}, "
            f"conv_kernel={kernel_size}, precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(plan)


def build_tp_convbert_encoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_seq_length: int,
    *,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build one FP32 tensor-parallel ConvBERT rank."""
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_tp_convbert_encoder_engine requires tensor_parallel mode and tp_size > 1"
        )
    return build_convbert_encoder_engine(
        config,
        weights,
        max_seq_length,
        verbose=verbose,
        parallel_config=parallel,
    )
