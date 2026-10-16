# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure TensorRT-Python graphs for dense LFM2 causal decoders."""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt
from .native_kv_attention_builder import (
    NativeKvMasks,
    add_active_prefix_causal_masks,
    add_explicit_masked_grouped_query_attention,
)

from .checkpoint_mapper import WeightDict, load_lfm2_weights
from .config import DenseLfm2Config, validate_dense_lfm2_config


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


def layer_tensor_name(stem: str, layer: int) -> str:
    return f"{stem}_{layer}"


def _precision_dtypes(precision: str) -> tuple[np.dtype, trt.DataType]:
    if precision == "fp16":
        return np.dtype(np.float16), trt.float16
    if precision == "bf16":
        # NumPy has no portable BF16 weights type. Build constants in FP32 and
        # cast once inside the strongly typed TensorRT graph.
        return np.dtype(np.float32), trt.bfloat16
    raise ValueError("dense LFM2 supports only fp16 and bf16 builds")


def _cast(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    dtype: trt.DataType,
) -> trt.ITensor:
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def _residual_sum(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    rhs: trt.ITensor,
) -> trt.ITensor:
    if lhs.dtype != rhs.dtype:
        raise ValueError("dense LFM2 residual operands must have matching dtypes")
    # Each branch is already rounded to the model dtype. Preserve that boundary
    # before addition instead of consuming an unrounded projection accumulator.
    dtype = lhs.dtype
    summed = network.add_elementwise(
        _cast(network, lhs, trt.float32),
        _cast(network, rhs, trt.float32),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    return _cast(network, summed, dtype)


def _constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    *,
    np_dtype: np.dtype = np.dtype(np.float32),
) -> trt.ITensor:
    array = np.ascontiguousarray(values, dtype=np_dtype).reshape(shape)
    return network.add_constant(shape, trt.Weights(array)).get_output(0)


def _work_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    *,
    storage_dtype: np.dtype,
    work_dtype: trt.DataType,
) -> trt.ITensor:
    tensor = _constant(network, shape, values, np_dtype=storage_dtype)
    return _cast(network, tensor, work_dtype)


def _matmul(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    rhs: np.ndarray,
    *,
    lhs_width: int,
    rhs_width: int,
    storage_dtype: np.dtype,
    work_dtype: trt.DataType,
    name: str,
) -> trt.ITensor:
    array = np.asarray(rhs)
    expected = (lhs_width, rhs_width)
    if tuple(array.shape) != expected:
        raise ValueError(
            f"mapped LFM2 weight {name} must have shape {expected}, got {tuple(array.shape)}"
        )
    rhs_tensor = _work_constant(
        network,
        expected,
        array,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    layer = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs_tensor,
        trt.MatrixOperation.NONE,
    )
    if layer is None:
        raise RuntimeError(f"TensorRT failed to add LFM2 matmul {name}")
    layer.name = name
    return layer.get_output(0)


def _bias(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    values: np.ndarray | None,
    *,
    width: int,
    storage_dtype: np.dtype,
    work_dtype: trt.DataType,
) -> trt.ITensor:
    if values is None:
        return tensor
    if tuple(np.asarray(values).shape) != (width,):
        raise ValueError(f"mapped LFM2 bias must have shape ({width},)")
    bias = _work_constant(
        network,
        (1, width),
        values,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    return network.add_elementwise(
        tensor,
        bias,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)


def _mean64(
    network: trt.INetworkDefinition,
    squared: trt.ITensor,
    bridge: Path,
) -> trt.ITensor:
    from tensorrt_model_connect.byok import add_kernel

    shape = tuple(squared.shape)
    if (
        squared.dtype != trt.float32
        or len(shape) != 3
        or shape[1] not in {8, 32}
        or shape[2] != 64
        or (shape[0] != -1 and not 1 <= shape[0] <= 131072)
    ):
        raise ValueError("LFM2 Q/K mean64 requires FP32 [R,8|32,64] squares")
    index = _constant(network, (1,), np.zeros((1,), dtype=np.int32), np_dtype=np.int32)
    shape_like = network.add_gather(squared, index, 2).get_output(0)
    return add_kernel(
        network,
        plugin_library=bridge,
        kernel_name=_MEAN64_KERNEL,
        inputs=[squared, shape_like],
        output_specs=[{"dims": "same_as_input_1", "dtype": "float32"}],
        workspace_bytes=0,
    )[0]


def _rms_norm(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    gamma: np.ndarray,
    *,
    width: int,
    epsilon: float,
    output_dtype: trt.DataType,
    mean64_bridge: Path | None = None,
) -> trt.ITensor:
    if mean64_bridge is not None and (width != 64 or output_dtype != trt.bfloat16):
        raise ValueError("LFM2 mean64 is limited to BF16 Q/K normalization")
    if tuple(np.asarray(gamma).shape) != (width,):
        raise ValueError(f"mapped LFM2 RMSNorm weight must have shape ({width},)")
    fp32 = _cast(network, tensor, trt.float32)
    rank = len(tuple(fp32.shape))
    axes = 1 << (rank - 1)
    squared = network.add_elementwise(
        fp32,
        fp32,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    mean = (
        _mean64(network, squared, mean64_bridge)
        if mean64_bridge is not None
        else network.add_reduce(squared, trt.ReduceOperation.AVG, axes, True).get_output(0)
    )
    eps_shape = (1,) * rank
    eps = _constant(
        network,
        eps_shape,
        np.full(eps_shape, epsilon, dtype=np.float32),
    )
    variance = network.add_elementwise(
        mean,
        eps,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    root = network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
    reciprocal = network.add_unary(root, trt.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(
        fp32,
        reciprocal,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    # Match HF Lfm2RMSNorm exactly: variance and normalization are FP32, then
    # normalized activations return to the model dtype before model-dtype gamma
    # is applied. Multiplying gamma in FP32 changes FP16/BF16 rounding.
    normalized = _cast(network, normalized, output_dtype)
    gamma_shape = (1,) * (rank - 1) + (width,)
    gamma_storage_dtype = (
        np.dtype(np.float16) if output_dtype == trt.float16 else np.dtype(np.float32)
    )
    gamma_tensor = _work_constant(
        network,
        gamma_shape,
        np.asarray(gamma, dtype=np.float32).reshape(gamma_shape),
        storage_dtype=gamma_storage_dtype,
        work_dtype=output_dtype,
    )
    scaled = network.add_elementwise(
        normalized,
        gamma_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    return scaled


def _per_head_rms_norm(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    gamma: np.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    epsilon: float,
    output_dtype: trt.DataType,
    mean64_bridge: Path | None = None,
) -> trt.ITensor:
    if tuple(np.asarray(gamma).shape) != (head_dim,):
        raise ValueError(f"mapped LFM2 QK norm must have shape ({head_dim},)")
    shaped = network.add_shuffle(tensor)
    shaped.reshape_dims = (-1, num_heads, head_dim)
    normalized = _rms_norm(
        network,
        shaped.get_output(0),
        gamma,
        width=head_dim,
        epsilon=epsilon,
        output_dtype=output_dtype,
        mean64_bridge=mean64_bridge,
    )
    flattened = network.add_shuffle(normalized)
    flattened.reshape_dims = (-1, num_heads * head_dim)
    return flattened.get_output(0)


def _silu(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
) -> trt.ITensor:
    # F.silu rounds once after sigmoid and multiplication, not between them.
    dtype = tensor.dtype
    fp32 = _cast(network, tensor, trt.float32)
    sigmoid = network.add_activation(fp32, trt.ActivationType.SIGMOID)
    output = network.add_elementwise(
        fp32, sigmoid.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    return _cast(network, output, dtype)


def _active_rope_cache(
    network: trt.INetworkDefinition,
    position_id: trt.ITensor,
    *,
    head_dim: int,
    rope_theta: float,
    work_dtype: trt.DataType,
) -> tuple[trt.ITensor, trt.ITensor]:
    if head_dim % 2:
        raise ValueError("dense LFM2 RoPE requires an even head_dim")
    inverse_frequency = 1.0 / (
        float(rope_theta) ** (np.arange(0, head_dim, 2, dtype=np.float32) / float(head_dim))
    )
    position = _cast(network, position_id, trt.float32)
    position_column = network.add_shuffle(position)
    position_column.reshape_dims = (-1, 1)
    inverse = _constant(
        network,
        (1, head_dim // 2),
        inverse_frequency.reshape(1, -1),
    )
    angles = network.add_elementwise(
        position_column.get_output(0),
        inverse,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    cos = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    sin = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    cos_3d = network.add_shuffle(cos)
    cos_3d.reshape_dims = (1, -1, head_dim // 2)
    sin_3d = network.add_shuffle(sin)
    sin_3d.reshape_dims = (1, -1, head_dim // 2)
    return (
        _cast(network, cos_3d.get_output(0), work_dtype),
        _cast(network, sin_3d.get_output(0), work_dtype),
    )


def _reshape_rows_to_heads(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
) -> trt.ITensor:
    shaped = network.add_shuffle(tensor)
    shaped.reshape_dims = (1, -1, num_heads, head_dim)
    shaped.second_transpose = (0, 2, 1, 3)
    return shaped.get_output(0)


def _reshape_heads_to_rows(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    *,
    width: int,
) -> trt.ITensor:
    shaped = network.add_shuffle(tensor)
    shaped.first_transpose = (0, 2, 1, 3)
    shaped.reshape_dims = (-1, width)
    return shaped.get_output(0)


def _apply_rope(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    cos_cache: trt.ITensor,
    sin_cache: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
) -> trt.ITensor:
    heads = _reshape_rows_to_heads(
        network,
        tensor,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    # The reference rounds the cosine and sine products separately before
    # adding them. Keep native rotary operators, with a model-dtype boundary
    # for each component, instead of one fused multiply-add rounding.
    # RotaryEmbedding requires both caches to have the input sequence length;
    # its cache inputs do not broadcast a single zero row across the prompt.
    zero = network.add_elementwise(cos_cache, cos_cache, trt.ElementWiseOperation.SUB).get_output(0)
    cosine = network.add_rotary_embedding(heads, cos_cache, zero, False, head_dim)
    sine = network.add_rotary_embedding(heads, zero, sin_cache, False, head_dim)
    if cosine is None or sine is None:
        raise RuntimeError("TensorRT failed to create rotary components")
    summed = network.add_elementwise(
        _cast(network, cosine.get_output(0), trt.float32),
        _cast(network, sine.get_output(0), trt.float32),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    return _reshape_heads_to_rows(
        network,
        _cast(network, summed, heads.dtype),
        width=num_heads * head_dim,
    )


def add_native_kv_attention(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k_update: trt.ITensor,
    v_update: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    cache_write_indices: trt.ITensor,
    attention_masks: NativeKvMasks | None,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    work_dtype: trt.DataType,
    tag: str,
    attention_bridge: Path | None = None,
    active_length_extent: trt.ITensor | None = None,
) -> dict[str, trt.ITensor]:
    """Use TensorRT's in-place KV update and explicit masked attention."""

    if work_dtype not in {trt.float16, trt.bfloat16}:
        raise ValueError("LFM2 native KV attention requires FP16 or BF16")

    k_4d = _reshape_rows_to_heads(
        network,
        k_update,
        num_heads=num_kv_heads,
        head_dim=head_dim,
    )
    v_4d = _reshape_rows_to_heads(
        network,
        v_update,
        num_heads=num_kv_heads,
        head_dim=head_dim,
    )
    update_k = network.add_kv_cache_update(
        cache_k,
        k_4d,
        cache_write_indices,
        trt.KVCacheMode.LINEAR,
    )
    update_v = network.add_kv_cache_update(
        cache_v,
        v_4d,
        cache_write_indices,
        trt.KVCacheMode.LINEAR,
    )
    if update_k is None or update_v is None:
        raise RuntimeError("TensorRT failed to create LFM2 KV-cache update layers")
    update_k.name = f"{tag}.cache_k_update"
    update_v.name = f"{tag}.cache_v_update"
    present_k = update_k.get_output(0)
    present_v = update_v.get_output(0)

    q_4d = _reshape_rows_to_heads(
        network,
        q,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    if attention_bridge is not None:
        from tensorrt_model_connect.byok import add_kernel

        if (
            active_length_extent is None
            or work_dtype != trt.bfloat16
            or (num_heads, num_kv_heads, head_dim) != (32, 8, 64)
        ):
            raise ValueError("LFM2 cuDNN attention requires BF16 GQA 32/8/64 and active length")
        (context_4d,) = add_kernel(
            network,
            plugin_library=attention_bridge,
            kernel_name=_ATTENTION_KERNEL,
            inputs=[q_4d, present_k, present_v, active_length_extent],
            output_specs=[{"dims": "same_as_input_0", "dtype": "bfloat16"}],
            workspace_bytes=0,
        )
        context_4d.name = f"{tag}.context"
    else:
        if attention_masks is None or active_length_extent is not None:
            raise ValueError("LFM2 explicit attention requires its active-prefix masks")
        context_4d = add_explicit_masked_grouped_query_attention(
            network,
            q_4d,
            present_k,
            present_v,
            attention_masks,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=float(1.0 / np.sqrt(head_dim)),
            tag=tag,
        )
    context = _reshape_heads_to_rows(
        network,
        context_4d,
        width=num_heads * head_dim,
    )
    return {
        "context": context,
        "present_k": present_k,
        "present_v": present_v,
    }


def _row_count(network: trt.INetworkDefinition, tensor: trt.ITensor) -> trt.ITensor:
    shape = network.add_shape(tensor).get_output(0)
    return network.add_slice(shape, (0,), (1,), (1,)).get_output(0)


def _last_row(network: trt.INetworkDefinition, tensor: trt.ITensor) -> trt.ITensor:
    one = _constant(network, (1,), np.ones(1, np.int64), np_dtype=np.dtype(np.int64))
    index = network.add_elementwise(
        _row_count(network, tensor), one, trt.ElementWiseOperation.SUB
    ).get_output(0)
    return network.add_gather(tensor, index, 0).get_output(0)


def _add_batched_conv_layer(
    network: trt.INetworkDefinition,
    hidden_state: trt.ITensor,
    conv_state: trt.ITensor,
    decode_phase: trt.ITensor,
    weights: WeightDict,
    *,
    position_id: trt.ITensor,
    prefix: str,
    config: DenseLfm2Config,
    storage_dtype: np.dtype,
    work_dtype: trt.DataType,
) -> tuple[trt.ITensor, trt.ITensor]:
    """Full causal prefill and the original cached single-row convolution."""
    hidden, length = config.hidden_size, config.conv_l_cache
    projected = _matmul(
        network,
        hidden_state,
        weights[f"{prefix}.conv_in"],
        lhs_width=hidden,
        rhs_width=3 * hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name=f"{prefix}.conv.in_proj",
    )
    projected = _bias(
        network,
        projected,
        weights.get(f"{prefix}.conv_in_bias"),
        width=3 * hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    parts = []
    projected_shape = network.add_shape(projected).get_output(0)
    rows = network.add_slice(projected_shape, (0,), (1,), (1,)).get_output(0)
    width = _constant(network, (1,), np.array([hidden], np.int64), np_dtype=np.dtype(np.int64))
    part_shape = network.add_concatenation([rows, width])
    part_shape.axis = 0
    for index in range(3):
        part = network.add_slice(projected, (0, index * hidden), (1, hidden), (1, 1))
        part.set_input(2, part_shape.get_output(0))
        parts.append(part.get_output(0))
    b_gate, c_gate, x_value = parts
    bx = network.add_elementwise(b_gate, x_value, trt.ElementWiseOperation.PROD).get_output(0)
    bx_columns = network.add_shuffle(bx)
    bx_columns.first_transpose = (1, 0)
    history = network.add_concatenation([conv_state, bx_columns.get_output(0)])
    history.axis = 1

    # Every row sees the preceding L-1 values and its own B*x. The initial
    # zero state supplies left padding even when the whole prompt has <L rows.
    first_position = network.add_slice(position_id, (0,), (1,), (1,)).get_output(0)
    relative = network.add_elementwise(
        position_id, first_position, trt.ElementWiseOperation.SUB
    ).get_output(0)
    relative_column = network.add_shuffle(relative)
    relative_column.reshape_dims = (-1, 1)
    taps = _constant(
        network,
        (1, length),
        np.arange(1, length + 1, dtype=np.int32),
        np_dtype=np.dtype(np.int32),
    )
    window_indices = network.add_elementwise(
        relative_column.get_output(0), taps, trt.ElementWiseOperation.SUM
    ).get_output(0)
    windows = network.add_gather(history.get_output(0), window_indices, 1).get_output(0)
    windows_rows = network.add_shuffle(windows)
    windows_rows.first_transpose = (1, 0, 2)

    # Compute a well-shaped cached branch for either profile. Only its first
    # input row participates; the runtime permits decode exclusively at R=1.
    first_bx = network.add_slice(bx, (0, 0), (1, hidden), (1, 1)).get_output(0)
    first_bx_column = network.add_shuffle(first_bx)
    first_bx_column.reshape_dims = (hidden, 1)
    if length > 1:
        prior = network.add_slice(conv_state, (0, 1), (hidden, length - 1), (1, 1)).get_output(0)
        first = network.add_slice(conv_state, (0, 0), (hidden, 1), (1, 1)).get_output(0)
        rolled = network.add_concatenation([prior, first])
        rolled.axis = 1
        position = network.add_shuffle(first_position)
        position.reshape_dims = (1, 1)
        last = _constant(
            network, (1, 1), np.array([[length - 1]], np.int32), np_dtype=np.dtype(np.int32)
        )
        cached_index = network.add_elementwise(
            position.get_output(0), last, trt.ElementWiseOperation.MIN
        ).get_output(0)
        columns = _constant(
            network, (1, length), np.arange(length, dtype=np.int32), np_dtype=np.dtype(np.int32)
        )
        write_mask = network.add_elementwise(
            columns, cached_index, trt.ElementWiseOperation.EQUAL
        ).get_output(0)
        decode_present = network.add_select(
            write_mask, first_bx_column.get_output(0), rolled.get_output(0)
        ).get_output(0)
    else:
        decode_present = first_bx_column.get_output(0)

    tail_offsets = _constant(
        network, (length,), np.arange(length, dtype=np.int64), np_dtype=np.dtype(np.int64)
    )
    tail_indices = network.add_elementwise(
        rows, tail_offsets, trt.ElementWiseOperation.SUM
    ).get_output(0)
    prefill_present = network.add_gather(history.get_output(0), tail_indices, 1).get_output(0)
    present = network.add_select(decode_phase, decode_present, prefill_present).get_output(0)
    decode_window = network.add_shuffle(decode_present)
    decode_window.reshape_dims = (1, hidden, length)
    phase = network.add_shuffle(decode_phase)
    phase.reshape_dims = (1, 1, 1)
    active_windows = network.add_select(
        phase.get_output(0), decode_window.get_output(0), windows_rows.get_output(0)
    ).get_output(0)
    conv_weights = _work_constant(
        network,
        (1, hidden, length),
        weights[f"{prefix}.conv_weight"],
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    weighted = network.add_elementwise(
        _cast(network, active_windows, trt.float32),
        _cast(network, conv_weights, trt.float32),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    rounded = _cast(network, _cast(network, weighted, work_dtype), trt.float32)
    weighted = network.add_select(phase.get_output(0), rounded, weighted).get_output(0)
    convolved = network.add_reduce(weighted, trt.ReduceOperation.SUM, 1 << 2, False).get_output(0)
    convolved = _bias(
        network,
        _cast(network, convolved, work_dtype),
        weights.get(f"{prefix}.conv_bias"),
        width=hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    gated = network.add_elementwise(c_gate, convolved, trt.ElementWiseOperation.PROD).get_output(0)
    output = _matmul(
        network,
        gated,
        weights[f"{prefix}.conv_out"],
        lhs_width=hidden,
        rhs_width=hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name=f"{prefix}.conv.out_proj",
    )
    output = _bias(
        network,
        output,
        weights.get(f"{prefix}.conv_out_bias"),
        width=hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    return output, present


def _add_conv_layer(
    network: trt.INetworkDefinition,
    hidden_state: trt.ITensor,
    conv_state: trt.ITensor,
    decode_phase: trt.ITensor,
    weights: WeightDict,
    *,
    position_id: trt.ITensor,
    prefix: str,
    config: DenseLfm2Config,
    storage_dtype: np.dtype,
    work_dtype: trt.DataType,
) -> tuple[trt.ITensor, trt.ITensor]:
    hidden = config.hidden_size
    projected = _matmul(
        network,
        hidden_state,
        weights[f"{prefix}.conv_in"],
        lhs_width=hidden,
        rhs_width=3 * hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name=f"{prefix}.conv.in_proj",
    )
    projected = _bias(
        network,
        projected,
        weights.get(f"{prefix}.conv_in_bias"),
        width=3 * hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )

    parts: list[trt.ITensor] = []
    for index in range(3):
        part = network.add_slice(
            projected,
            (0, index * hidden),
            (1, hidden),
            (1, 1),
        )
        parts.append(part.get_output(0))
    b_gate, c_gate, x_value = parts
    bx = network.add_elementwise(
        b_gate,
        x_value,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    bx_column = network.add_shuffle(bx)
    bx_column.reshape_dims = (hidden, 1)
    if config.conv_l_cache > 1:
        prior = network.add_slice(
            conv_state,
            (0, 1),
            (hidden, config.conv_l_cache - 1),
            (1, 1),
        ).get_output(0)
        first = network.add_slice(conv_state, (0, 0), (hidden, 1), (1, 1)).get_output(0)
        rolled = network.add_concatenation([prior, first])
        rolled.axis = 1
        position = network.add_shuffle(position_id)
        position.reshape_dims = (1, 1)
        last = _constant(
            network,
            (1, 1),
            np.array([[config.conv_l_cache - 1]], dtype=np.int32),
            np_dtype=np.dtype(np.int32),
        )
        cached_index = network.add_elementwise(
            position.get_output(0), last, trt.ElementWiseOperation.MIN
        ).get_output(0)
        # The original slow cached path writes at clamp(cache_position, 0, L-1)
        # after rolling, even when a short prefix has not yet filled the cache.
        # Prefill remains an append at the last column.
        write_index = network.add_select(decode_phase, cached_index, last).get_output(0)
        columns = _constant(
            network,
            (1, config.conv_l_cache),
            np.arange(config.conv_l_cache, dtype=np.int32),
            np_dtype=np.dtype(np.int32),
        )
        write_mask = network.add_elementwise(
            columns, write_index, trt.ElementWiseOperation.EQUAL
        ).get_output(0)
        present_conv = network.add_select(
            write_mask, bx_column.get_output(0), rolled.get_output(0)
        ).get_output(0)
    else:
        present_conv = bx_column.get_output(0)

    conv_weights = _work_constant(
        network,
        (hidden, config.conv_l_cache),
        weights[f"{prefix}.conv_weight"],
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    weighted = network.add_elementwise(
        _cast(network, present_conv, trt.float32),
        _cast(network, conv_weights, trt.float32),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    # Cached reference decode rounds elementwise products to model dtype;
    # Conv1d prefill accumulates unrounded products until its output boundary.
    rounded_products = _cast(network, _cast(network, weighted, work_dtype), trt.float32)
    weighted = network.add_select(decode_phase, rounded_products, weighted).get_output(0)
    convolved = network.add_reduce(
        weighted,
        trt.ReduceOperation.SUM,
        1 << 1,
        True,
    ).get_output(0)
    conv_row = network.add_shuffle(_cast(network, convolved, work_dtype))
    conv_row.reshape_dims = (1, hidden)
    conv_row_tensor = _bias(
        network,
        conv_row.get_output(0),
        weights.get(f"{prefix}.conv_bias"),
        width=hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    gated = network.add_elementwise(
        c_gate,
        conv_row_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    output = _matmul(
        network,
        gated,
        weights[f"{prefix}.conv_out"],
        lhs_width=hidden,
        rhs_width=hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name=f"{prefix}.conv.out_proj",
    )
    output = _bias(
        network,
        output,
        weights.get(f"{prefix}.conv_out_bias"),
        width=hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    return output, present_conv


def _add_swiglu(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    weights: WeightDict,
    *,
    prefix: str,
    config: DenseLfm2Config,
    storage_dtype: np.dtype,
    work_dtype: trt.DataType,
    w2_bridge: Path | None = None,
    batched_prefill: bool = False,
) -> trt.ITensor:
    left = _matmul(
        network,
        tensor,
        weights[f"{prefix}.w1"],
        lhs_width=config.hidden_size,
        rhs_width=config.intermediate_size,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name=f"{prefix}.feed_forward.w1",
    )
    right = _matmul(
        network,
        tensor,
        weights[f"{prefix}.w3"],
        lhs_width=config.hidden_size,
        rhs_width=config.intermediate_size,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name=f"{prefix}.feed_forward.w3",
    )
    gated = network.add_elementwise(
        _silu(network, left),
        right,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    if w2_bridge is not None:
        return _add_w2(
            network,
            gated,
            weights[f"{prefix}.w2"],
            w2_bridge,
            prefix=prefix,
            shape_like=tensor if batched_prefill else None,
        )
    return _matmul(
        network,
        gated,
        weights[f"{prefix}.w2"],
        lhs_width=config.intermediate_size,
        rhs_width=config.hidden_size,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name=f"{prefix}.feed_forward.w2",
    )


def _add_w2(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    weight: np.ndarray,
    bridge: Path,
    *,
    prefix: str,
    shape_like: trt.ITensor | None = None,
) -> trt.ITensor:
    from tensorrt_model_connect.byok import add_kernel

    rows = tuple(tensor.shape)
    if tensor.dtype != trt.bfloat16 or len(rows) != 2 or rows[1] != 10752:
        raise ValueError("LFM2 W2 requires BF16 input shape [R, 10752]")
    if shape_like is None:
        if rows[0] != 1:
            raise ValueError("LFM2 W2 requires one row without a batched shape carrier")
    elif (
        tuple(shape_like.shape) != (rows[0], 2048)
        or shape_like.dtype != trt.bfloat16
        or rows[0] == 0
        or rows[0] < -1
    ):
        raise ValueError("LFM2 W2 requires a matching BF16 [R, 2048] shape carrier")
    if tuple(np.asarray(weight).shape) != (10752, 2048):
        raise ValueError("LFM2 W2 requires mapped weight shape [10752, 2048]")
    rhs = _work_constant(
        network,
        (10752, 2048),
        weight,
        storage_dtype=np.dtype(np.float32),
        work_dtype=trt.bfloat16,
    )
    transpose = network.add_shuffle(rhs)
    transpose.name = f"{prefix}.feed_forward.w2_transpose"
    transpose.first_transpose = (1, 0)
    transpose.reshape_dims = (2048, 10752)
    (output,) = add_kernel(
        network,
        plugin_library=bridge,
        kernel_name=_W2_ROWS_KERNEL if shape_like is not None else _W2_KERNEL,
        inputs=[tensor, transpose.get_output(0)] + ([] if shape_like is None else [shape_like]),
        output_specs=[
            {
                "dims": "same_as_input_2" if shape_like is not None else [1, 2048],
                "dtype": "bfloat16",
            }
        ],
        workspace_bytes=0,
    )
    output.name = f"{prefix}.feed_forward.w2"
    return output


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    output = network.add_identity(_cast(network, tensor, trt.float32)).get_output(0)
    output.name = name
    network.mark_output(output)


def _validate_mapped_metadata(config: DenseLfm2Config, weights: WeightDict) -> None:
    expected = {
        "_layer_types": list(config.layer_types),
        "_num_attention_layers": config.num_attention_layers,
        "_num_conv_layers": config.num_conv_layers,
        "_hidden_size": config.hidden_size,
        "_intermediate_size": config.intermediate_size,
        "_head_dim": config.head_dim,
        "_conv_l_cache": config.conv_l_cache,
    }
    for name, value in expected.items():
        if weights.get(name) != value:
            raise ValueError(
                f"mapped LFM2 metadata {name} must be {value!r}, got {weights.get(name)!r}"
            )


def build_lfm2_engine(
    config: object,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "bf16",
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    w2_bridge: Path | None = None,
    attention_bridge: Path | None = None,
) -> bytes:
    """Build one native-KV engine, with batched prefill for the dense BF16 path."""

    cfg = validate_dense_lfm2_config(config)
    if w2_bridge is not None and not _uses_w2(cfg, precision):
        raise ValueError("LFM2 W2 requires its qualified BF16 model geometry")
    if attention_bridge is not None and not _uses_attention(cfg, precision):
        raise ValueError("LFM2 cuDNN attention requires its qualified BF16 model geometry")
    if attention_bridge is not None and w2_bridge is None:
        raise ValueError("LFM2 batched attention requires its batched W2 binding")
    _validate_mapped_metadata(cfg, weights)
    if max_cache_length <= 0:
        raise ValueError("max_cache_length must be positive")
    if max_cache_length > cfg.max_position_embeddings:
        raise ValueError(
            f"max_cache_length exceeds LFM2 max_position_embeddings ({cfg.max_position_embeddings})"
        )
    if attention_bridge is not None and max_cache_length > 131072:
        raise ValueError("LFM2 cuDNN attention supports at most 131072 cached positions")
    storage_dtype, work_dtype = _precision_dtypes(precision)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 1
    builder_config.clear_flag(trt.BuilderFlag.TF32)
    # Match the qualified native-KV builder ceiling. This is build-time tactic
    # scratch, not an eager runtime allocation.
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)

    decode_phase = network.add_input("decode_phase", trt.int32, (1, 1))
    zero = _constant(network, (1, 1), np.zeros((1, 1), dtype=np.int32), np_dtype=np.dtype(np.int32))
    decode_phase = network.add_elementwise(
        decode_phase, zero, trt.ElementWiseOperation.GREATER
    ).get_output(0)
    row_shape = (-1,) if attention_bridge is not None else (1,)
    token_id = network.add_input("token_id", trt.int32, row_shape)
    position_id = network.add_input("position_id", trt.int32, row_shape)
    if attention_bridge is not None:
        token_id.set_dimension_name(0, "query_rows")
        position_id.set_dimension_name(0, "query_rows")
    cache_write_indices = network.add_input(
        "cache_write_indices",
        trt.int32,
        (1,),
    )
    active_length_extent = None
    attention_masks = None
    if attention_bridge is not None:
        # Only the dimension carries the logical length. The native binding
        # never reads these values; K/V retain their full-capacity geometry.
        active_length_extent = network.add_input("active_length_extent", trt.int32, (-1,))
        for prefill in (True, False):
            profile = builder.create_optimization_profile()
            extent_shapes = ((1,), (max_cache_length,), (max_cache_length,))
            row_shapes = extent_shapes if prefill else ((1,), (1,), (1,))
            for name, shapes in (
                ("token_id", row_shapes),
                ("position_id", row_shapes),
                ("active_length_extent", extent_shapes),
            ):
                profile.set_shape(name, *shapes)
                if tuple(tuple(shape) for shape in profile.get_shape(name)) != shapes:
                    raise RuntimeError(f"TensorRT rejected the LFM2 {name} profile")
            if builder_config.add_optimization_profile(profile) < 0:
                raise RuntimeError("TensorRT failed to add the LFM2 decoder profile")
    else:
        key_value_lengths = network.add_input("key_value_lengths", trt.int32, (1,))
        attention_masks = add_active_prefix_causal_masks(
            network,
            token_id,
            cache_write_indices,
            key_value_lengths,
            max_cache_length,
        )

    cache_shape = (
        1,
        cfg.num_key_value_heads,
        max_cache_length,
        cfg.head_dim,
    )
    cache_k_inputs: list[trt.ITensor] = []
    cache_v_inputs: list[trt.ITensor] = []
    for attention_index in range(cfg.num_attention_layers):
        cache_k_inputs.append(
            network.add_input(
                layer_tensor_name("cache_k", attention_index),
                work_dtype,
                cache_shape,
            )
        )
        cache_v_inputs.append(
            network.add_input(
                layer_tensor_name("cache_v", attention_index),
                work_dtype,
                cache_shape,
            )
        )

    conv_state_inputs: list[trt.ITensor] = []
    for conv_index in range(cfg.num_conv_layers):
        conv_state_inputs.append(
            network.add_input(
                layer_tensor_name("conv_state", conv_index),
                work_dtype,
                (cfg.hidden_size, cfg.conv_l_cache),
            )
        )

    embedding = _work_constant(
        network,
        (cfg.vocab_size, cfg.hidden_size),
        weights["embedding"],
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    hidden_state = network.add_gather(embedding, token_id, 0).get_output(0)
    hidden_state = _cast(network, hidden_state, work_dtype)
    cos_cache, sin_cache = _active_rope_cache(
        network,
        position_id,
        head_dim=cfg.head_dim,
        rope_theta=cfg.rope_theta,
        work_dtype=work_dtype,
    )

    present_k_outputs: list[trt.ITensor] = []
    present_v_outputs: list[trt.ITensor] = []
    present_conv_outputs: list[trt.ITensor] = []
    attention_index = 0
    conv_index = 0

    for layer_index, layer_type in enumerate(cfg.layer_types):
        prefix = f"layer.{layer_index}"
        operator_input = _rms_norm(
            network,
            hidden_state,
            weights[f"{prefix}.operator_norm"],
            width=cfg.hidden_size,
            epsilon=cfg.norm_eps,
            output_dtype=work_dtype,
        )
        if layer_type == "conv":
            conv_builder = (
                _add_batched_conv_layer if attention_bridge is not None else _add_conv_layer
            )
            operator_output, present_conv = conv_builder(
                network,
                operator_input,
                conv_state_inputs[conv_index],
                decode_phase,
                weights,
                position_id=position_id,
                prefix=prefix,
                config=cfg,
                storage_dtype=storage_dtype,
                work_dtype=work_dtype,
            )
            present_conv_outputs.append(present_conv)
            conv_index += 1
        else:
            q = _matmul(
                network,
                operator_input,
                weights[f"{prefix}.w_q"],
                lhs_width=cfg.hidden_size,
                rhs_width=cfg.attention_size,
                storage_dtype=storage_dtype,
                work_dtype=work_dtype,
                name=f"{prefix}.self_attn.q_proj",
            )
            k = _matmul(
                network,
                operator_input,
                weights[f"{prefix}.w_k"],
                lhs_width=cfg.hidden_size,
                rhs_width=cfg.kv_attention_size,
                storage_dtype=storage_dtype,
                work_dtype=work_dtype,
                name=f"{prefix}.self_attn.k_proj",
            )
            v = _matmul(
                network,
                operator_input,
                weights[f"{prefix}.w_v"],
                lhs_width=cfg.hidden_size,
                rhs_width=cfg.kv_attention_size,
                storage_dtype=storage_dtype,
                work_dtype=work_dtype,
                name=f"{prefix}.self_attn.v_proj",
            )
            q = _per_head_rms_norm(
                network,
                q,
                weights[f"{prefix}.q_norm"],
                num_heads=cfg.num_attention_heads,
                head_dim=cfg.head_dim,
                epsilon=cfg.norm_eps,
                output_dtype=work_dtype,
                mean64_bridge=attention_bridge,
            )
            k = _per_head_rms_norm(
                network,
                k,
                weights[f"{prefix}.k_norm"],
                num_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                epsilon=cfg.norm_eps,
                output_dtype=work_dtype,
                mean64_bridge=attention_bridge,
            )
            q = _apply_rope(
                network,
                q,
                cos_cache,
                sin_cache,
                num_heads=cfg.num_attention_heads,
                head_dim=cfg.head_dim,
            )
            k = _apply_rope(
                network,
                k,
                cos_cache,
                sin_cache,
                num_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
            )
            native = add_native_kv_attention(
                network,
                q,
                k,
                v,
                cache_k_inputs[attention_index],
                cache_v_inputs[attention_index],
                cache_write_indices,
                attention_masks,
                num_heads=cfg.num_attention_heads,
                num_kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                work_dtype=work_dtype,
                tag=f"{prefix}.self_attn",
                attention_bridge=attention_bridge,
                active_length_extent=active_length_extent,
            )
            operator_output = _matmul(
                network,
                native["context"],
                weights[f"{prefix}.w_o"],
                lhs_width=cfg.attention_size,
                rhs_width=cfg.hidden_size,
                storage_dtype=storage_dtype,
                work_dtype=work_dtype,
                name=f"{prefix}.self_attn.out_proj",
            )
            present_k_outputs.append(native["present_k"])
            present_v_outputs.append(native["present_v"])
            attention_index += 1

        hidden_state = _residual_sum(network, hidden_state, operator_output)
        ffn_input = _rms_norm(
            network,
            hidden_state,
            weights[f"{prefix}.ffn_norm"],
            width=cfg.hidden_size,
            epsilon=cfg.norm_eps,
            output_dtype=work_dtype,
        )
        ffn_output = _add_swiglu(
            network,
            ffn_input,
            weights,
            prefix=prefix,
            config=cfg,
            storage_dtype=storage_dtype,
            work_dtype=work_dtype,
            w2_bridge=w2_bridge,
            batched_prefill=attention_bridge is not None,
        )
        hidden_state = _residual_sum(network, hidden_state, ffn_output)
        if debug_layer_outputs:
            _mark_debug_output(
                network,
                hidden_state,
                f"debug_hidden_{layer_index}",
            )

    if attention_index != cfg.num_attention_layers or conv_index != cfg.num_conv_layers:
        raise RuntimeError("LFM2 layer/state counters did not consume every input")

    hidden_state = _rms_norm(
        network,
        hidden_state,
        weights["final_norm"],
        width=cfg.hidden_size,
        epsilon=cfg.norm_eps,
        output_dtype=work_dtype,
    )
    if attention_bridge is not None:
        hidden_state = _last_row(network, hidden_state)
    logits = _matmul(
        network,
        hidden_state,
        weights["w_lm_head"],
        lhs_width=cfg.hidden_size,
        rhs_width=cfg.vocab_size,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name="lm_head",
    )
    logits = _cast(network, logits, trt.float32)
    logits.name = "logits"
    network.mark_output(logits)

    for index, tensor in enumerate(present_k_outputs):
        tensor.name = layer_tensor_name("present_k", index)
        network.mark_output(tensor)
    for index, tensor in enumerate(present_v_outputs):
        tensor.name = layer_tensor_name("present_v", index)
        network.mark_output(tensor)
    for index, tensor in enumerate(present_conv_outputs):
        tensor.name = layer_tensor_name("present_conv", index)
        network.mark_output(tensor)

    if verbose:
        print(
            "[trtmc build] Building LFM2 native hybrid engine "
            f"(layers={cfg.num_hidden_layers}, conv={cfg.num_conv_layers}, "
            f"attention={cfg.num_attention_layers}, hidden={cfg.hidden_size}, "
            f"cache={max_cache_length}, precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the LFM2 engine")
    return bytes(plan)


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


def _load_config(model_dir: Path) -> SimpleNamespace:
    config_path = model_dir / "config.json"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"LFM2 checkpoint is missing {config_path}") from error
    if not isinstance(raw, dict):
        raise ValueError("LFM2 config.json must contain one JSON object")
    return SimpleNamespace(raw=raw)


def _runtime_config(
    model_dir: Path,
    resolved: DenseLfm2Config,
    **updates,
) -> dict:
    runtime = dict(resolved.bundle_overrides())
    runtime.update(
        {
            "vocab_size": resolved.vocab_size,
            "bos_token_id": resolved.bos_token_id,
            "eos_token_id": resolved.eos_token_id,
            "pad_token_id": resolved.pad_token_id,
        }
    )
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            runtime["eos_token_id"] = generation["eos_token_id"]
    runtime.update(updates)
    return runtime


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


_SAMPLER_MODE = "philox_exponential_v1"
_SAMPLER_KERNEL = "lfm2.sampler." + _SAMPLER_MODE
_SAMPLER_VOCAB_SIZE = 128000
_W2_MODE = "cublas_bf16_all_w2_v1"
_W2_KERNEL = "lfm2.w2.cublas_bf16_v1"
_W2_ROWS_MODE = "cublas_bf16_batched_w2_v1"
_W2_ROWS_KERNEL = "lfm2.w2.cublas_bf16_rows_v1"
_ATTENTION_MODE = "cudnn_bf16_batched_v1"
_ATTENTION_KERNEL = "lfm2.attention." + _ATTENTION_MODE
_MEAN64_KERNEL = "lfm2.attention.mean64_fp32_v1"


def _uses_w2(config: DenseLfm2Config, precision: str) -> bool:
    return (
        precision == "bf16"
        and config.vocab_size == 128000
        and config.hidden_size == 2048
        and config.intermediate_size == 10752
        and config.num_hidden_layers == 30
        and config.layer_types[:2] == ("conv", "conv")
    )


def _w2_library(bridge: Path) -> Path:
    library = bridge.parent / "libtrtmc_lfm2_w2_byok.so"
    if not library.is_file():
        raise FileNotFoundError("LFM2 W2 installation is incomplete: " + str(library))
    return library


def _uses_attention(config: DenseLfm2Config, precision: str) -> bool:
    return (
        _uses_w2(config, precision)
        and config.num_attention_heads == 32
        and config.num_key_value_heads == 8
        and config.head_dim == 64
        and config.num_attention_layers == 8
    )


def _attention_library(bridge: Path) -> Path:
    library = bridge.parent / "libtrtmc_lfm2_sdpa_byok.so"
    if not library.is_file():
        raise FileNotFoundError("LFM2 cuDNN attention installation is incomplete: " + str(library))
    return library


def _register_attention(extension: ctypes.CDLL, library: Path) -> None:
    error = extension.trtmc_load_byok_kernel(
        str(library).encode("utf-8"), b"run_rows", _ATTENTION_KERNEL.encode("utf-8")
    )
    if error is not None:
        raise RuntimeError(
            "LFM2 cuDNN attention registration failed: " + error.decode("utf-8", errors="replace")
        )

    error = extension.trtmc_load_byok_kernel(
        str(library).encode("utf-8"), b"mean64", _MEAN64_KERNEL.encode("utf-8")
    )
    if error is not None:
        raise RuntimeError(
            "LFM2 Q/K mean64 registration failed: " + error.decode("utf-8", errors="replace")
        )


def _register_w2(extension: ctypes.CDLL, library: Path, *, batched_prefill: bool = False) -> None:
    error = extension.trtmc_load_byok_kernel(
        str(library).encode("utf-8"),
        b"run_rows" if batched_prefill else b"run",
        (_W2_ROWS_KERNEL if batched_prefill else _W2_KERNEL).encode("utf-8"),
    )
    if error is not None:
        raise RuntimeError(
            "LFM2 W2 registration failed: " + error.decode("utf-8", errors="replace")
        )


def _sampler_libraries() -> tuple[Path, Path]:
    from tensorrt_model_connect.build import cmake_prefixes
    import tensorrt_model_connect.byok as byok

    names = ("libtrtmc_byok_tvm_ffi.so", "libtrtmc_lfm2_sampler_byok.so")

    def require_pair(directory: Path) -> tuple[Path, Path]:
        directory = directory.resolve()
        pair = (directory / names[0], directory / names[1])
        missing = [str(path) for path in pair if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "LFM2 sampler installation is incomplete: " + ", ".join(missing)
            )
        return pair

    directories = [
        prefix / libdir for prefix in cmake_prefixes() for libdir in ("", "bin", "lib", "lib64")
    ]
    directories.append(Path(byok.__file__).resolve().parent / "bin")
    for directory in directories:
        if any((directory / name).exists() for name in names):
            return require_pair(directory)
    raise FileNotFoundError(
        "LFM2 sampling requires the installed BYOK bridge and LFM2 sampler library; "
        "set CMAKE_PREFIX_PATH to their shared directory or installation prefix"
    )


def _register_sampler(bridge: Path, library: Path) -> ctypes.CDLL:
    extension = ctypes.CDLL(str(bridge), mode=ctypes.RTLD_GLOBAL)
    register = extension.trtmc_load_byok_kernel
    register.argtypes = [ctypes.c_char_p] * 3
    register.restype = ctypes.c_char_p
    error = register(str(library).encode("utf-8"), b"sample", _SAMPLER_KERNEL.encode("utf-8"))
    if error is not None:
        raise RuntimeError(
            "LFM2 sampler registration failed: " + error.decode("utf-8", errors="replace")
        )
    if trt.get_plugin_registry().get_creator("TvmFfiKernel", "1", "") is None:
        raise RuntimeError("LFM2 sampler installation did not register TvmFfiKernel version 1")
    return extension


def build_lfm2_sampler_engine(bridge: Path, *, verbose: bool = False) -> bytes:
    from tensorrt_model_connect.byok import add_kernel

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 32 << 20)
    probabilities = network.add_input("probabilities", trt.float32, (1, _SAMPLER_VOCAB_SIZE))
    state = network.add_input("rng_state", trt.int32, (4,))
    outputs = add_kernel(
        network,
        plugin_library=bridge,
        kernel_name=_SAMPLER_KERNEL,
        inputs=[probabilities, state],
        output_specs=[{"dims": [1], "dtype": "int32"}, {"dims": [4], "dtype": "int32"}],
        workspace_bytes=_SAMPLER_VOCAB_SIZE * 8,
    )
    for tensor, name in zip(outputs, ("token_id", "next_state"), strict=True):
        tensor.name = name
        network.mark_output(tensor)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the LFM2 sampler engine")
    data = bytes(plan)
    if not data:
        raise RuntimeError("TensorRT returned an empty LFM2 sampler engine")
    return data


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one dense LFM2 bundle without shared model orchestration."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("lfm2 does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("lfm2 does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("lfm2 does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("lfm2 does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("lfm2 does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("lfm2 supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = _load_config(model_dir)
    resolved = validate_dense_lfm2_config(config)
    precision = str(request.precision).lower()
    if precision not in {"fp16", "bf16"}:
        raise ValueError("dense LFM2 supports only fp16 and bf16 builds")
    max_cache_length = _positive_int(
        request.max_sequence_length or resolved.default_cache_length,
        "max_sequence_length",
    )
    if max_cache_length > resolved.max_position_embeddings:
        raise ValueError(
            "max_cache_length exceeds the checkpoint context capacity: "
            f"{max_cache_length} > {resolved.max_position_embeddings}"
        )
    if _uses_attention(resolved, precision) and max_cache_length > 131072:
        raise ValueError("LFM2 cuDNN attention supports at most 131072 cached positions")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("dense LFM2 does not support tensor-parallel builds")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("dense LFM2 does not support quantized builds")
    if request.fp32_layers:
        raise NotImplementedError("dense LFM2 does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("dense LFM2 does not support MTP speculative decoding")

    sampler_bridge = None
    sampler_extension = None
    w2_bridge = None
    attention_bridge = None
    if resolved.vocab_size == _SAMPLER_VOCAB_SIZE:
        if request.backend != "trt":
            raise ValueError("LFM2 Philox sampling requires backend=trt")
        # The decoder emits FP32 logits for both supported model precisions.
        # Resolve and register dependencies before allocating checkpoint weights.
        sampler_bridge, sampler_library = _sampler_libraries()
        w2_library = _w2_library(sampler_bridge) if _uses_w2(resolved, precision) else None
        attention_library = (
            _attention_library(sampler_bridge) if _uses_attention(resolved, precision) else None
        )
        sampler_extension = _register_sampler(sampler_bridge, sampler_library)
        if w2_library is not None:
            _register_w2(
                sampler_extension, w2_library, batched_prefill=attention_library is not None
            )
            w2_bridge = sampler_bridge
        if attention_library is not None:
            _register_attention(sampler_extension, attention_library)
            attention_bridge = sampler_bridge

    weights = load_lfm2_weights(str(model_dir), config, precision=precision)
    plan = build_lfm2_engine(
        config,
        weights,
        max_cache_length,
        precision=precision,
        verbose=bool(request.verbose),
        debug_layer_outputs=False,
        w2_bridge=w2_bridge,
        attention_bridge=attention_bridge,
    )

    runtime_config = _runtime_config(
        model_dir,
        resolved,
        precision=precision,
        max_cache_length=max_cache_length,
        decoder_engine_layout="dual_profile" if attention_bridge is not None else "single",
    )
    sampler_plan = None
    if sampler_bridge is not None:
        sampler_plan = build_lfm2_sampler_engine(sampler_bridge, verbose=bool(request.verbose))
        runtime_config["sampler_mode"] = _SAMPLER_MODE
    if w2_bridge is not None:
        runtime_config["w2_mode"] = _W2_ROWS_MODE if attention_bridge is not None else _W2_MODE
    if attention_bridge is not None:
        runtime_config["attention_mode"] = _ATTENTION_MODE
        runtime_config["native_kv_contract_version"] = 3
    # Keep the registration library alive until both plans have been serialized.
    del sampler_extension

    writer.set_header(family="lfm2", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    if sampler_plan is not None:
        writer.add_bytes("sampler.plan", sampler_plan)
    writer.add_json("runtime.json", runtime_config)
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
