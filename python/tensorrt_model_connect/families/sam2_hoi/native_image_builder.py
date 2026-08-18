# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT Network Definition builder for the SAM2-HOI image encoder.

The graph is fixed to the reviewed 1024 by 1024 Hiera-S checkpoint.  Learned
operators execute in the selected precision while the two source-model
promotion boundaries stay FP32: Hiera's initial positional residual and the
64 by 64 FPN top-down sum.  Model construction is deliberately limited to
checkpoint reads and TensorRT graph creation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import sys
from typing import Any

import numpy as np

from tensorrt_model_connect import trt_compat

from . import native_graph_ops as graph_ops


_IMAGE_SIZE = 1024
_WORKSPACE_BYTES = 8 << 30
_HIERA_PREFIX = "image_encoder.trunk"
_FPN_PREFIX = "image_encoder.neck"
_PAFPN_PREFIX = "image_encoder.learnable_fpn_module.learnable_fpn_module"
_EXACT_BF16_NCHW_1X1_EINSUM_EQUATION = "nchw,oc->nohw"
_EXACT_BF16_NCHW_1X1_CONTRACTS = frozenset(
    {
        ((1, 96, 256, 256), (256, 96, 1, 1), (256,)),
        ((1, 192, 128, 128), (256, 192, 1, 1), (256,)),
        ((1, 384, 64, 64), (256, 384, 1, 1), (256,)),
        ((1, 768, 32, 32), (256, 768, 1, 1), (256,)),
        ((1, 256, 256, 256), (32, 256, 1, 1), (32,)),
        ((1, 256, 128, 128), (64, 256, 1, 1), (64,)),
    }
)


@dataclass(frozen=True)
class HieraBlockSpec:
    """Static shape and attention contract for one reviewed Hiera block."""

    index: int
    height: int
    width: int
    dim: int
    dim_out: int
    heads: int
    window: int
    q_pool: bool


def _hiera_block_specs() -> tuple[HieraBlockSpec, ...]:
    stage_ends = (0, 2, 13, 15)
    q_pool_blocks = {1, 3, 14}
    global_blocks = {7, 10, 13}
    stage_windows = (8, 4, 14, 7)
    specs: list[HieraBlockSpec] = []
    dim = 96
    heads = 1
    height = width = 256
    stage = 0
    for index in range(16):
        dim_out = dim
        window = 0 if index in global_blocks else stage_windows[stage]
        if index > 0 and index - 1 in stage_ends:
            dim_out *= 2
            heads *= 2
            stage += 1
        specs.append(
            HieraBlockSpec(
                index=index,
                height=height,
                width=width,
                dim=dim,
                dim_out=dim_out,
                heads=heads,
                window=window,
                q_pool=index in q_pool_blocks,
            )
        )
        if index in q_pool_blocks:
            height //= 2
            width //= 2
        dim = dim_out
    return tuple(specs)


_HIERA_BLOCKS = _hiera_block_specs()
_STAGE_ENDS = frozenset({0, 2, 13, 15})


def _weight(weights: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    value = np.asarray(weights[name], dtype=np.float32)
    return np.ascontiguousarray(value)


def _optional_weight(
    weights: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if name not in weights:
        return np.zeros(shape, dtype=np.float32)
    value = _weight(weights, name)
    if value.shape != shape:
        raise ValueError(f"SAM2 HOI parameter {name!r} expected {shape}, got {value.shape}")
    return value


def _precision_for_tensor(tensor: Any) -> str:
    return "bf16" if tensor.dtype == graph_ops._trt().bfloat16 else "fp32"


def _cast_to_work(network: Any, tensor: Any, precision: str) -> Any:
    return graph_ops.cast(network, tensor, graph_ops.runtime_dtype(precision))


def _add_hiera_layer_norm(
    network: Any,
    tensor: Any,
    gamma: np.ndarray,
    beta: np.ndarray,
    *,
    instance_name: str,
) -> Any:
    """Match Hiera's autocast boundary: every LayerNorm executes in FP32."""

    trt = graph_ops._trt()
    gamma = np.ascontiguousarray(gamma, dtype=np.float32)
    beta = np.ascontiguousarray(beta, dtype=np.float32)
    if gamma.ndim != 1 or beta.shape != gamma.shape:
        raise ValueError(f"invalid Hiera LayerNorm parameters {gamma.shape}, {beta.shape}")
    width = int(gamma.size)
    if width not in {96, 192, 384, 768}:
        raise ValueError(f"unsupported Hiera LayerNorm width {width}")
    work = graph_ops.cast(network, tensor, trt.float32)
    gamma_tensor = graph_ops.add_constant(network, (width,), gamma, precision="fp32")
    beta_tensor = graph_ops.add_constant(network, (width,), beta, precision="fp32")
    output = graph_ops.add_plugin(
        network,
        "Sam2HoiHieraLayerNorm",
        (work, gamma_tensor, beta_tensor),
        instance_name=instance_name,
    )
    return graph_ops.cast(network, output, trt.float32)


def _shuffle(
    network: Any,
    tensor: Any,
    *,
    reshape: tuple[int, ...] | None = None,
    first_transpose: tuple[int, ...] | None = None,
    second_transpose: tuple[int, ...] | None = None,
) -> Any:
    trt = graph_ops._trt()
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI image shuffle")
    if first_transpose is not None:
        layer.first_transpose = trt.Permutation(first_transpose)
    if reshape is not None:
        layer.reshape_dims = reshape
    if second_transpose is not None:
        layer.second_transpose = trt.Permutation(second_transpose)
    return layer.get_output(0)


def _nhwc_to_nchw(network: Any, tensor: Any) -> Any:
    return _shuffle(network, tensor, first_transpose=(0, 3, 1, 2))


def _nchw_to_nhwc(network: Any, tensor: Any) -> Any:
    return _shuffle(network, tensor, first_transpose=(0, 2, 3, 1))


def _promoted_sum(network: Any, lhs: Any, rhs: Any) -> Any:
    trt = graph_ops._trt()
    if lhs.dtype != rhs.dtype:
        lhs = graph_ops.cast(network, lhs, trt.float32)
        rhs = graph_ops.cast(network, rhs, trt.float32)
    layer = network.add_elementwise(lhs, rhs, trt.ElementWiseOperation.SUM)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI promoted sum")
    return layer.get_output(0)


def _fused_multiply_add_float32(
    multiplier: np.ndarray | np.float32,
    multiplicand: np.ndarray | np.float32,
    addend: np.ndarray | np.float32,
) -> np.ndarray:
    """Emulate one finite binary32 FFMA with a single final rounding."""

    product = np.asarray(
        np.multiply(
            np.asarray(multiplier, dtype=np.float32),
            np.asarray(multiplicand, dtype=np.float32),
            dtype=np.float64,
        ),
        dtype=np.float64,
    )
    np.add(product, np.asarray(addend, dtype=np.float32), out=product)
    return np.asarray(product, dtype=np.float32)


def _hiera_cubic_convolution1(value: np.ndarray) -> np.ndarray:
    work = _fused_multiply_add_float32(np.float32(1.25), value, np.float32(-2.25))
    work = np.multiply(work, value, dtype=np.float32)
    return _fused_multiply_add_float32(work, value, np.float32(1.0))


def _hiera_cubic_convolution2(value: np.ndarray) -> np.ndarray:
    work = _fused_multiply_add_float32(np.float32(-0.75), value, np.float32(3.75))
    work = _fused_multiply_add_float32(work, value, np.float32(-6.0))
    return _fused_multiply_add_float32(work, value, np.float32(3.0))


def _hiera_cubic_coefficients(fraction: np.ndarray) -> np.ndarray:
    opposite = np.asarray(np.float32(1.0) - fraction, dtype=np.float32)
    return np.stack(
        (
            _hiera_cubic_convolution2(np.asarray(fraction + np.float32(1.0), dtype=np.float32)),
            _hiera_cubic_convolution1(fraction),
            _hiera_cubic_convolution1(opposite),
            _hiera_cubic_convolution2(np.asarray(opposite + np.float32(1.0), dtype=np.float32)),
        ),
        axis=-1,
    )


def _hiera_cubic_interpolate1d(values: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Match the CUDA compiler's contraction schedule for cubic_interp1d."""

    result = np.multiply(values[..., 1], coefficients[..., 1], dtype=np.float32)
    result = _fused_multiply_add_float32(values[..., 0], coefficients[..., 0], result)
    result = _fused_multiply_add_float32(values[..., 2], coefficients[..., 2], result)
    return _fused_multiply_add_float32(values[..., 3], coefficients[..., 3], result)


def _hiera_bicubic_7x7_to_256x256(values: np.ndarray) -> np.ndarray:
    """Reproduce the fixed source CUDA bicubic resize on the build host."""

    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim != 4 or values.shape[-2:] != (7, 7):
        raise ValueError(f"SAM2 HOI Hiera global position expected NCHW 7x7, got {values.shape}")

    destination = np.arange(256, dtype=np.float32)
    centers = np.asarray(destination + np.float32(0.5), dtype=np.float32)
    source = _fused_multiply_add_float32(np.float32(7.0 / 256.0), centers, np.float32(-0.5))
    base = np.floor(source).astype(np.int32)
    fraction = np.asarray(source - base.astype(np.float32), dtype=np.float32)
    indices = np.stack(
        (
            np.clip(base - 1, 0, 6),
            np.clip(base, 0, 6),
            np.clip(base + 1, 0, 6),
            np.clip(base + 2, 0, 6),
        ),
        axis=-1,
    )
    coefficients = _hiera_cubic_coefficients(fraction)

    horizontal_values = values[..., indices]
    horizontal = _hiera_cubic_interpolate1d(
        horizontal_values,
        coefficients[None, None, None, :, :],
    )
    vertical_values = np.moveaxis(horizontal[:, :, indices, :], -2, -1)
    return np.ascontiguousarray(
        _hiera_cubic_interpolate1d(
            vertical_values,
            coefficients[None, None, :, None, :],
        ),
        dtype=np.float32,
    )


def _concatenate(network: Any, tensors: list[Any], *, axis: int) -> Any:
    trt = graph_ops._trt()
    if any(tensor.dtype == trt.float32 for tensor in tensors):
        tensors = [graph_ops.cast(network, tensor, trt.float32) for tensor in tensors]
    layer = network.add_concatenation(tensors)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI concatenation")
    layer.axis = axis
    return layer.get_output(0)


def _max_pool2d(network: Any, tensor_nhwc: Any) -> Any:
    trt = graph_ops._trt()
    tensor = _nhwc_to_nchw(network, tensor_nhwc)
    layer = network.add_pooling_nd(tensor, trt.PoolingType.MAX, (2, 2))
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI max pool")
    layer.stride_nd = (2, 2)
    return _nchw_to_nhwc(network, layer.get_output(0))


def _window_partition_indices(
    height: int,
    width: int,
    window: int,
) -> tuple[np.ndarray, int, int, int]:
    """Return padded window order, using ``H*W`` as the zero-token index."""

    padded_h = ((height + window - 1) // window) * window
    padded_w = ((width + window - 1) // window) * window
    sentinel = height * width
    indices: list[int] = []
    for window_y in range(0, padded_h, window):
        for window_x in range(0, padded_w, window):
            for y in range(window_y, window_y + window):
                for x in range(window_x, window_x + window):
                    indices.append(y * width + x if y < height and x < width else sentinel)
    return np.asarray(indices, dtype=np.int32), padded_h, padded_w, sentinel


def _window_unpartition_indices(
    height: int,
    width: int,
    padded_h: int,
    padded_w: int,
    window: int,
) -> np.ndarray:
    indices = np.empty(height * width, dtype=np.int32)
    windows_wide = padded_w // window
    for y in range(height):
        for x in range(width):
            window_y, local_y = divmod(y, window)
            window_x, local_x = divmod(x, window)
            window_index = window_y * windows_wide + window_x
            indices[y * width + x] = window_index * window * window + local_y * window + local_x
    return indices


def _partition_windows(
    network: Any,
    hidden: Any,
    *,
    height: int,
    width: int,
    channels: int,
    window: int,
) -> tuple[Any, int, int, int]:
    indices, padded_h, padded_w, sentinel = _window_partition_indices(height, width, window)
    rows = _shuffle(network, hidden, reshape=(1, height * width, channels))
    if padded_h != height or padded_w != width:
        zero = graph_ops.add_constant(
            network,
            (1, 1, channels),
            np.zeros((1, 1, channels), dtype=np.float32),
            precision=_precision_for_tensor(rows),
        )
        rows = _concatenate(network, [rows, zero], axis=1)
    if indices.max(initial=-1) > sentinel:
        raise AssertionError("SAM2 HOI window indices exceeded their zero-token sentinel")
    index_tensor = graph_ops.add_int32_constant(network, indices.shape, indices)
    gathered = network.add_gather(rows, index_tensor, axis=1)
    if gathered is None:
        raise RuntimeError("TensorRT failed to partition SAM2 HOI Hiera windows")
    window_count = (padded_h // window) * (padded_w // window)
    partitioned = _shuffle(
        network,
        gathered.get_output(0),
        reshape=(window_count, window * window, channels),
    )
    return partitioned, padded_h, padded_w, window_count


def _unpartition_windows(
    network: Any,
    windows: Any,
    *,
    height: int,
    width: int,
    padded_h: int,
    padded_w: int,
    window: int,
    channels: int,
) -> Any:
    flat = _shuffle(
        network,
        windows,
        reshape=(1, (padded_h // window) * (padded_w // window) * window * window, channels),
    )
    indices = _window_unpartition_indices(height, width, padded_h, padded_w, window)
    index_tensor = graph_ops.add_int32_constant(network, indices.shape, indices)
    gathered = network.add_gather(flat, index_tensor, axis=1)
    if gathered is None:
        raise RuntimeError("TensorRT failed to unpartition SAM2 HOI Hiera windows")
    return _shuffle(network, gathered.get_output(0), reshape=(1, height, width, channels))


def _slice_qkv(
    network: Any,
    qkv: Any,
    *,
    batches: int,
    sequence: int,
    heads: int,
    head_dim: int,
    component: int,
) -> Any:
    shaped = _shuffle(
        network,
        qkv,
        reshape=(batches, sequence, 3, heads, head_dim),
    )
    layer = network.add_slice(
        shaped,
        start=(0, 0, component, 0, 0),
        shape=(batches, sequence, 1, heads, head_dim),
        stride=(1, 1, 1, 1, 1),
    )
    if layer is None:
        raise RuntimeError("TensorRT failed to slice SAM2 HOI Hiera QKV")
    return _shuffle(
        network,
        layer.get_output(0),
        reshape=(batches, sequence, heads, head_dim),
    )


def _rows_to_heads(network: Any, tensor: Any) -> Any:
    return _shuffle(network, tensor, first_transpose=(0, 2, 1, 3))


def _heads_to_rows(
    network: Any,
    tensor: Any,
    *,
    batches: int,
    sequence: int,
    channels: int,
) -> Any:
    return _shuffle(
        network,
        tensor,
        first_transpose=(0, 2, 1, 3),
        reshape=(batches, sequence, channels),
    )


def _pool_query_rows(
    network: Any,
    query: Any,
    *,
    batches: int,
    window: int,
    heads: int,
    head_dim: int,
) -> Any:
    channels = heads * head_dim
    query = _shuffle(network, query, reshape=(batches, window, window, channels))
    query = _max_pool2d(network, query)
    pooled_window = window // 2
    return _shuffle(
        network,
        query,
        reshape=(batches, pooled_window * pooled_window, heads, head_dim),
    )


def _scale_query(network: Any, query: Any, *, head_dim: int) -> Any:
    trt = graph_ops._trt()
    output_dtype = query.dtype
    compute = graph_ops.cast(network, query, trt.float32)
    scale = graph_ops.add_constant(
        network,
        (1, 1, 1, 1),
        np.asarray([head_dim**-0.5], dtype=np.float32),
        precision="fp32",
    )
    scaled = network.add_elementwise(compute, scale, trt.ElementWiseOperation.PROD)
    if scaled is None:
        raise RuntimeError("TensorRT failed to scale SAM2 HOI Hiera queries")
    return graph_ops.cast(network, scaled.get_output(0), output_dtype)


def _add_attention(
    network: Any,
    hidden_rows: Any,
    weights: Mapping[str, np.ndarray],
    spec: HieraBlockSpec,
    *,
    batches: int,
    input_window: int,
    precision: str,
) -> tuple[Any, int]:
    trt = graph_ops._trt()
    prefix = f"{_HIERA_PREFIX}.blocks.{spec.index}.attn"
    work = _cast_to_work(network, hidden_rows, precision)
    qkv = graph_ops.add_linear(
        network,
        work,
        _weight(weights, f"{prefix}.qkv.weight"),
        _weight(weights, f"{prefix}.qkv.bias"),
        precision=precision,
    )
    input_sequence = input_window * input_window
    head_dim = spec.dim_out // spec.heads
    query = _slice_qkv(
        network,
        qkv,
        batches=batches,
        sequence=input_sequence,
        heads=spec.heads,
        head_dim=head_dim,
        component=0,
    )
    key = _slice_qkv(
        network,
        qkv,
        batches=batches,
        sequence=input_sequence,
        heads=spec.heads,
        head_dim=head_dim,
        component=1,
    )
    value = _slice_qkv(
        network,
        qkv,
        batches=batches,
        sequence=input_sequence,
        heads=spec.heads,
        head_dim=head_dim,
        component=2,
    )
    output_window = input_window
    if spec.q_pool:
        query = _pool_query_rows(
            network,
            query,
            batches=batches,
            window=input_window,
            heads=spec.heads,
            head_dim=head_dim,
        )
        output_window //= 2
    query = _rows_to_heads(network, query)
    key = _rows_to_heads(network, key)
    value = _rows_to_heads(network, value)
    if precision == "bf16":
        if head_dim != 96:
            raise ValueError(f"unsupported Hiera FlashAttention head dimension {head_dim}")
        attended = graph_ops.add_plugin(
            network,
            "Sam2HoiHieraFlashAttention96",
            (query, key, value),
            instance_name=f"hiera_flash_attention96_block_{spec.index:02d}",
        )
        attended = graph_ops.cast(network, attended, trt.bfloat16)
    else:
        query = _scale_query(network, query, head_dim=head_dim)
        logits = network.add_matrix_multiply(
            query,
            trt.MatrixOperation.NONE,
            key,
            trt.MatrixOperation.TRANSPOSE,
        )
        if logits is None:
            raise RuntimeError("TensorRT failed to create SAM2 HOI Hiera attention logits")
        softmax = network.add_softmax(logits.get_output(0))
        if softmax is None:
            raise RuntimeError("TensorRT failed to create SAM2 HOI Hiera softmax")
        softmax.axes = 1 << 3
        attended_layer = network.add_matrix_multiply(
            softmax.get_output(0),
            trt.MatrixOperation.NONE,
            value,
            trt.MatrixOperation.NONE,
        )
        if attended_layer is None:
            raise RuntimeError("TensorRT failed to create SAM2 HOI Hiera value projection")
        attended = attended_layer.get_output(0)
    output_sequence = output_window * output_window
    rows = _heads_to_rows(
        network,
        attended,
        batches=batches,
        sequence=output_sequence,
        channels=spec.dim_out,
    )
    projection_weight = _weight(weights, f"{prefix}.proj.weight")
    projection_bias = _weight(weights, f"{prefix}.proj.bias")
    if precision == "bf16" and spec.index in {14, 15}:
        if (batches, output_sequence, spec.dim_out) != (25, 49, 768):
            raise ValueError(
                "unsupported Hiera block14/15 projection shape "
                f"{(batches, output_sequence, spec.dim_out)}"
            )
        flat_rows = _shuffle(network, rows, reshape=(1, 1225, 768))
        weight_tensor = graph_ops.add_constant(
            network, (768, 768), projection_weight, precision="bf16"
        )
        bias_tensor = graph_ops.add_constant(network, (768,), projection_bias, precision="bf16")
        projected = graph_ops.add_plugin(
            network,
            "Sam2HoiHieraBlock1415Projection",
            (flat_rows, weight_tensor, bias_tensor),
            instance_name=f"hiera_block1415_projection_block_{spec.index:02d}",
        )
        projected = graph_ops.cast(network, projected, trt.bfloat16)
        projected = _shuffle(network, projected, reshape=(batches, output_sequence, 768))
    else:
        projected = graph_ops.add_linear(
            network,
            rows,
            projection_weight,
            projection_bias,
            precision=precision,
        )
    return projected, output_window


def _add_hiera_gelu(
    network: Any,
    tensor: Any,
    *,
    precision: str,
    block_index: int,
) -> Any:
    if precision != "bf16":
        return graph_ops.add_activation(network, tensor, "gelu")
    shape = tuple(int(value) for value in tensor.shape)
    allowed = {
        (1, 256, 256, 384),
        (1, 128, 128, 768),
        (1, 64, 64, 1536),
        (1, 32, 32, 3072),
    }
    if shape not in allowed:
        raise ValueError(f"unsupported Hiera GELU shape {shape}")
    output = graph_ops.add_plugin(
        network,
        "Sam2HoiHieraGeluErfBF16",
        (tensor,),
        instance_name=f"hiera_gelu_erf_bf16_block_{block_index:02d}",
    )
    return graph_ops.cast(network, output, graph_ops._trt().bfloat16)


def _add_hiera_block(
    network: Any,
    hidden: Any,
    weights: Mapping[str, np.ndarray],
    spec: HieraBlockSpec,
    *,
    precision: str,
) -> Any:
    prefix = f"{_HIERA_PREFIX}.blocks.{spec.index}"
    normed = _add_hiera_layer_norm(
        network,
        hidden,
        _weight(weights, f"{prefix}.norm1.weight"),
        _weight(weights, f"{prefix}.norm1.bias"),
        instance_name=f"hiera_layer_norm_block_{spec.index:02d}_norm1",
    )

    shortcut = hidden
    if spec.dim != spec.dim_out:
        shortcut = graph_ops.add_linear(
            network,
            _cast_to_work(network, normed, precision),
            _weight(weights, f"{prefix}.proj.weight"),
            _weight(weights, f"{prefix}.proj.bias"),
            precision=precision,
        )
        shortcut = _max_pool2d(network, shortcut)

    if spec.window:
        rows, padded_h, padded_w, batches = _partition_windows(
            network,
            normed,
            height=spec.height,
            width=spec.width,
            channels=spec.dim,
            window=spec.window,
        )
        attention, output_window = _add_attention(
            network,
            rows,
            weights,
            spec,
            batches=batches,
            input_window=spec.window,
            precision=precision,
        )
        output_height = spec.height // (2 if spec.q_pool else 1)
        output_width = spec.width // (2 if spec.q_pool else 1)
        attention = _unpartition_windows(
            network,
            attention,
            height=output_height,
            width=output_width,
            padded_h=padded_h // (2 if spec.q_pool else 1),
            padded_w=padded_w // (2 if spec.q_pool else 1),
            window=output_window,
            channels=spec.dim_out,
        )
    else:
        rows = _shuffle(
            network,
            normed,
            reshape=(1, spec.height * spec.width, spec.dim),
        )
        attention, _ = _add_attention(
            network,
            rows,
            weights,
            spec,
            batches=1,
            input_window=spec.height,
            precision=precision,
        )
        attention = _shuffle(
            network,
            attention,
            reshape=(1, spec.height, spec.width, spec.dim_out),
        )

    hidden = _promoted_sum(network, shortcut, attention)
    normed = _add_hiera_layer_norm(
        network,
        hidden,
        _weight(weights, f"{prefix}.norm2.weight"),
        _weight(weights, f"{prefix}.norm2.bias"),
        instance_name=f"hiera_layer_norm_block_{spec.index:02d}_norm2",
    )
    mlp = graph_ops.add_linear(
        network,
        _cast_to_work(network, normed, precision),
        _weight(weights, f"{prefix}.mlp.layers.0.weight"),
        _weight(weights, f"{prefix}.mlp.layers.0.bias"),
        precision=precision,
    )
    mlp = _add_hiera_gelu(
        network,
        mlp,
        precision=precision,
        block_index=spec.index,
    )
    mlp = graph_ops.add_linear(
        network,
        mlp,
        _weight(weights, f"{prefix}.mlp.layers.1.weight"),
        _weight(weights, f"{prefix}.mlp.layers.1.bias"),
        precision=precision,
    )
    return _promoted_sum(network, hidden, mlp)


def _add_hiera_position(network: Any, weights: Mapping[str, np.ndarray]) -> Any:
    global_position = _weight(weights, f"{_HIERA_PREFIX}.pos_embed")
    window_position = _weight(weights, f"{_HIERA_PREFIX}.pos_embed_window")
    if global_position.shape != (1, 96, 7, 7):
        raise ValueError(
            "SAM2 HOI Hiera global position must have shape (1, 96, 7, 7), "
            f"got {global_position.shape}"
        )
    if window_position.shape != (1, 96, 8, 8):
        raise ValueError(
            "SAM2 HOI Hiera window position must have shape (1, 96, 8, 8), "
            f"got {window_position.shape}"
        )

    resized = _hiera_bicubic_7x7_to_256x256(global_position)
    tiled = np.tile(window_position, (1, 1, 32, 32))
    position = np.ascontiguousarray(resized + tiled, dtype=np.float32)
    return graph_ops.add_constant(network, position.shape, position, precision="fp32")


def _add_hiera_patch_conv(
    network: Any,
    pixel_values: Any,
    weights: Mapping[str, np.ndarray],
    *,
    precision: str,
) -> Any:
    weight_values = _weight(weights, f"{_HIERA_PREFIX}.patch_embed.proj.weight")
    bias_values = _weight(weights, f"{_HIERA_PREFIX}.patch_embed.proj.bias")
    if tuple(weight_values.shape) != (96, 3, 7, 7):
        raise ValueError("SAM2 HOI Hiera patch-convolution weight must have shape (96, 3, 7, 7)")
    if tuple(bias_values.shape) != (96,):
        raise ValueError("SAM2 HOI Hiera patch-convolution bias must have shape (96,)")
    work = _cast_to_work(network, pixel_values, precision)
    if precision != "bf16":
        return graph_ops.add_conv2d(
            network,
            work,
            weight_values,
            bias_values,
            stride=(4, 4),
            padding=(3, 3),
            precision=precision,
        )
    weight = graph_ops.add_constant(
        network,
        tuple(weight_values.shape),
        weight_values,
        precision="bf16",
    )
    bias = graph_ops.add_constant(
        network,
        tuple(bias_values.shape),
        bias_values,
        precision="bf16",
    )
    output = graph_ops.add_plugin(
        network,
        "Sam2HoiHieraPatchConv",
        (work, weight, bias),
        instance_name="hiera_patch_conv",
    )
    output.name = "hiera_patch_conv.raw_output"
    # TensorRT 11.x initially labels IPluginV2DynamicExt outputs FP32 in the
    # Network Definition even though configure-time descriptors expose BF16.
    # Pin the semantic boundary before the following FP32 positional residual.
    cast_layer = network.add_cast(output, trt_compat.get_trt().bfloat16)
    if cast_layer is None:
        raise RuntimeError("TensorRT failed to type SAM2 HOI Hiera patch convolution")
    return cast_layer.get_output(0)


def _add_hiera(
    network: Any,
    pixel_values: Any,
    weights: Mapping[str, np.ndarray],
    *,
    precision: str,
) -> tuple[Any, Any, Any, Any]:
    patch = _add_hiera_patch_conv(
        network,
        pixel_values,
        weights,
        precision=precision,
    )
    position = _add_hiera_position(network, weights)
    hidden = _promoted_sum(network, patch, position)
    hidden = _nchw_to_nhwc(network, hidden)
    stage_outputs: list[Any] = []
    for spec in _HIERA_BLOCKS:
        hidden = _add_hiera_block(network, hidden, weights, spec, precision=precision)
        if spec.index in _STAGE_ENDS:
            stage_outputs.append(_nhwc_to_nchw(network, hidden))
    if len(stage_outputs) != 4:
        raise AssertionError("SAM2 HOI Hiera did not emit four stages")
    return tuple(stage_outputs)  # type: ignore[return-value]


def _add_position_encoding_sine(
    network: Any,
    height: int,
    width: int,
    channels: int = 256,
) -> Any:
    """Build the fixed tracker sine encoding with source-exact GPU arithmetic."""

    if (height, width, channels) != (64, 64, 256):
        raise ValueError(
            "SAM2 HOI tracker sine position only supports the reviewed "
            f"64x64x256 contract, got {height}x{width}x{channels}"
        )

    trt = graph_ops._trt()
    features = channels // 2
    coordinate = np.arange(1, height + 1, dtype=np.float32)
    coordinate = np.asarray(
        coordinate / (coordinate[-1] + np.float32(1.0e-6)) * np.float32(2.0 * np.pi),
        dtype=np.float32,
    )
    dimensions = np.arange(features, dtype=np.float32)
    exponents = np.asarray(
        np.float32(2.0) * np.floor(dimensions / np.float32(2.0)) / np.float32(features),
        dtype=np.float32,
    )

    coordinate_tensor = graph_ops.add_constant(
        network,
        (height, 1),
        coordinate[:, None],
        precision="fp32",
    )
    exponent_tensor = graph_ops.add_constant(
        network,
        (1, features),
        exponents[None, :],
        precision="fp32",
    )
    base_tensor = graph_ops.add_constant(
        network,
        (1, features),
        np.full((1, features), 10000.0, dtype=np.float32),
        precision="fp32",
    )

    divisor_layer = network.add_elementwise(
        base_tensor,
        exponent_tensor,
        trt.ElementWiseOperation.POW,
    )
    if divisor_layer is None:
        raise RuntimeError("TensorRT failed to create SAM2 HOI tracker position divisors")
    phase_layer = network.add_elementwise(
        coordinate_tensor,
        divisor_layer.get_output(0),
        trt.ElementWiseOperation.DIV,
    )
    if phase_layer is None:
        raise RuntimeError("TensorRT failed to create SAM2 HOI tracker position phases")
    phase = phase_layer.get_output(0)

    sine_layer = network.add_unary(phase, trt.UnaryOperation.SIN)
    cosine_layer = network.add_unary(phase, trt.UnaryOperation.COS)
    if sine_layer is None or cosine_layer is None:
        raise RuntimeError("TensorRT failed to create SAM2 HOI tracker position trigonometry")
    sine_even_layer = network.add_slice(
        sine_layer.get_output(0),
        (0, 0),
        (height, features // 2),
        (1, 2),
    )
    cosine_odd_layer = network.add_slice(
        cosine_layer.get_output(0),
        (0, 1),
        (height, features // 2),
        (1, 2),
    )
    if sine_even_layer is None or cosine_odd_layer is None:
        raise RuntimeError("TensorRT failed to slice SAM2 HOI tracker position channels")

    sine_even = _shuffle(
        network,
        sine_even_layer.get_output(0),
        reshape=(height, features // 2, 1),
    )
    cosine_odd = _shuffle(
        network,
        cosine_odd_layer.get_output(0),
        reshape=(height, features // 2, 1),
    )
    interleaved = _concatenate(network, [sine_even, cosine_odd], axis=2)
    table = _shuffle(network, interleaved, reshape=(height, features))

    position_y_singleton = _shuffle(
        network,
        table,
        first_transpose=(1, 0),
        reshape=(1, features, height, 1),
    )
    position_x_singleton = _shuffle(
        network,
        table,
        first_transpose=(1, 0),
        reshape=(1, features, 1, width),
    )
    position_y = _concatenate(
        network,
        [position_y_singleton] * width,
        axis=3,
    )
    position_x = _concatenate(
        network,
        [position_x_singleton] * height,
        axis=2,
    )
    return _concatenate(network, [position_y, position_x], axis=1)


def _add_exact_bf16_nchw_1x1_projection(
    network: Any,
    tensor: Any,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    precision: str,
) -> Any:
    """Build one of the six source-exact fixed BF16 1x1 projections."""

    work = _cast_to_work(network, tensor, precision)
    if precision != "bf16":
        return graph_ops.add_conv2d(
            network,
            work,
            weight,
            bias,
            precision=precision,
        )

    input_shape = tuple(int(value) for value in work.shape)
    weight_shape = tuple(int(value) for value in weight.shape)
    bias_shape = tuple(int(value) for value in bias.shape)
    contract = (input_shape, weight_shape, bias_shape)
    if contract not in _EXACT_BF16_NCHW_1X1_CONTRACTS:
        raise ValueError(
            "SAM2 HOI exact BF16 NCHW 1x1 projection must use one of the six "
            f"fixed contracts, got input={input_shape}, weight={weight_shape}, "
            f"bias={bias_shape}"
        )
    output_channels = weight_shape[0]
    input_channels = input_shape[1]

    weight_tensor = graph_ops.add_constant(
        network,
        (output_channels, input_channels),
        weight.reshape(output_channels, input_channels),
        precision="bf16",
    )
    einsum = network.add_einsum(
        [work, weight_tensor],
        _EXACT_BF16_NCHW_1X1_EINSUM_EQUATION,
    )
    if einsum is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI exact BF16 1x1 Einsum")
    bias_tensor = graph_ops.add_constant(
        network,
        (1, output_channels, 1, 1),
        bias.reshape(1, output_channels, 1, 1),
        precision="bf16",
    )
    summed = network.add_elementwise(
        einsum.get_output(0),
        bias_tensor,
        graph_ops._trt().ElementWiseOperation.SUM,
    )
    if summed is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI exact BF16 1x1 bias")
    return summed.get_output(0)


def _add_fpn(
    network: Any,
    stages: tuple[Any, Any, Any, Any],
    weights: Mapping[str, np.ndarray],
    *,
    precision: str,
) -> tuple[Any, Any, Any, Any]:
    outputs: list[Any | None] = [None, None, None, None]
    previous = None
    spatial = (256, 128, 64, 32)
    for index in range(3, -1, -1):
        conv_index = 3 - index
        prefix = f"{_FPN_PREFIX}.convs.{conv_index}.conv"
        lateral = _add_exact_bf16_nchw_1x1_projection(
            network,
            stages[index],
            _weight(weights, f"{prefix}.weight"),
            _weight(weights, f"{prefix}.bias"),
            precision=precision,
        )
        if index in {2, 3} and previous is not None:
            previous_fp32 = graph_ops.cast(network, previous, graph_ops._trt().float32)
            size = spatial[index]
            top_down = graph_ops.add_resize(
                network,
                previous_fp32,
                (1, 256, size, size),
                mode="nearest",
                coordinate_transformation="asymmetric",
            )
            previous = _promoted_sum(network, lateral, top_down)
        else:
            previous = lateral
        outputs[index] = previous
    if any(output is None for output in outputs):
        raise AssertionError("SAM2 HOI FPN did not emit four levels")
    return tuple(outputs)  # type: ignore[return-value]


def _add_tracker_front_outputs(
    network: Any,
    fpn: tuple[Any, Any, Any, Any],
    weights: Mapping[str, np.ndarray],
    *,
    precision: str,
) -> tuple[Any, Any, Any, Any]:
    tracker_0 = _add_exact_bf16_nchw_1x1_projection(
        network,
        fpn[0],
        _weight(weights, "sam_mask_decoder.conv_s0.weight"),
        _weight(weights, "sam_mask_decoder.conv_s0.bias"),
        precision=precision,
    )
    tracker_1 = _add_exact_bf16_nchw_1x1_projection(
        network,
        fpn[1],
        _weight(weights, "sam_mask_decoder.conv_s1.weight"),
        _weight(weights, "sam_mask_decoder.conv_s1.bias"),
        precision=precision,
    )
    tracker_2 = graph_ops.cast(network, fpn[2], graph_ops._trt().float32)
    tracker_position_2 = _add_position_encoding_sine(network, 64, 64)
    return tracker_0, tracker_1, tracker_2, tracker_position_2


def _add_conv_bn_silu(
    network: Any,
    tensor: Any,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    precision: str,
) -> Any:
    convolution = _weight(weights, f"{prefix}.conv.weight")
    output_channels = convolution.shape[0]
    bias = _optional_weight(weights, f"{prefix}.conv.bias", (output_channels,))
    if precision == "bf16":
        tensor = graph_ops.add_conv2d(
            network,
            _cast_to_work(network, tensor, precision),
            convolution,
            bias,
            stride=stride,
            padding=padding,
            precision=precision,
        )
        tensor = graph_ops.add_batch_norm2d_affine(
            network,
            tensor,
            _weight(weights, f"{prefix}.bn.weight"),
            _weight(weights, f"{prefix}.bn.bias"),
            _weight(weights, f"{prefix}.bn.running_mean"),
            _weight(weights, f"{prefix}.bn.running_var"),
            epsilon=1.0e-5,
            output_dtype=graph_ops.runtime_dtype(precision),
        )
        return graph_ops.add_activation(network, tensor, "silu")

    folded_weight, folded_bias = graph_ops.fold_batch_norm(
        convolution,
        bias,
        _weight(weights, f"{prefix}.bn.weight"),
        _weight(weights, f"{prefix}.bn.bias"),
        _weight(weights, f"{prefix}.bn.running_mean"),
        _weight(weights, f"{prefix}.bn.running_var"),
        epsilon=1.0e-5,
    )
    tensor = graph_ops.add_conv2d(
        network,
        _cast_to_work(network, tensor, precision),
        folded_weight,
        folded_bias,
        stride=stride,
        padding=padding,
        precision=precision,
    )
    return graph_ops.add_activation(network, tensor, "silu")


def _add_csp_layer(
    network: Any,
    tensor: Any,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    *,
    precision: str,
) -> Any:
    short = _add_conv_bn_silu(network, tensor, weights, f"{prefix}.short_conv", precision=precision)
    main = _add_conv_bn_silu(network, tensor, weights, f"{prefix}.main_conv", precision=precision)
    for index in range(3):
        block = f"{prefix}.blocks.{index}"
        main = _add_conv_bn_silu(network, main, weights, f"{block}.conv1", precision=precision)
        main = _add_conv_bn_silu(
            network,
            main,
            weights,
            f"{block}.conv2",
            padding=(1, 1),
            precision=precision,
        )
    merged = _concatenate(network, [main, short], axis=1)
    return _add_conv_bn_silu(network, merged, weights, f"{prefix}.final_conv", precision=precision)


def _add_pafpn(
    network: Any,
    inputs: tuple[Any, Any, Any],
    weights: Mapping[str, np.ndarray],
    *,
    precision: str,
) -> tuple[Any, Any, Any]:
    reduced = [inputs[0], inputs[1], inputs[2]]
    reduced[2] = _add_conv_bn_silu(
        network,
        reduced[2],
        weights,
        f"{_PAFPN_PREFIX}.reduce_layers.2",
        precision=precision,
    )

    high = graph_ops.add_resize(
        network,
        reduced[2],
        (1, 256, 64, 64),
        mode="nearest",
        coordinate_transformation="asymmetric",
    )
    high = _concatenate(network, [high, reduced[1]], axis=1)
    high = _add_csp_layer(
        network,
        high,
        weights,
        f"{_PAFPN_PREFIX}.top_down_layers.0.0",
        precision=precision,
    )
    high = _add_conv_bn_silu(
        network,
        high,
        weights,
        f"{_PAFPN_PREFIX}.top_down_layers.0.1",
        precision=precision,
    )
    inner_1 = high

    high = graph_ops.add_resize(
        network,
        inner_1,
        (1, 256, 128, 128),
        mode="nearest",
        coordinate_transformation="asymmetric",
    )
    high = _concatenate(network, [high, reduced[0]], axis=1)
    inner_0 = _add_csp_layer(
        network,
        high,
        weights,
        f"{_PAFPN_PREFIX}.top_down_layers.1",
        precision=precision,
    )

    down = _add_conv_bn_silu(
        network,
        inner_0,
        weights,
        f"{_PAFPN_PREFIX}.downsample_layers.0",
        stride=(2, 2),
        padding=(1, 1),
        precision=precision,
    )
    out_1 = _add_csp_layer(
        network,
        _concatenate(network, [down, inner_1], axis=1),
        weights,
        f"{_PAFPN_PREFIX}.bottom_up_layers.0",
        precision=precision,
    )
    down = _add_conv_bn_silu(
        network,
        out_1,
        weights,
        f"{_PAFPN_PREFIX}.downsample_layers.1",
        stride=(2, 2),
        padding=(1, 1),
        precision=precision,
    )
    out_2 = _add_csp_layer(
        network,
        _concatenate(network, [down, reduced[2]], axis=1),
        weights,
        f"{_PAFPN_PREFIX}.bottom_up_layers.1",
        precision=precision,
    )
    outs = (inner_0, out_1, out_2)
    return tuple(
        _add_conv_bn_silu(
            network,
            tensor,
            weights,
            f"{_PAFPN_PREFIX}.out_layers.{index}",
            padding=(1, 1),
            precision=precision,
        )
        for index, tensor in enumerate(outs)
    )  # type: ignore[return-value]


def build_image_feature_engine(
    weights: Mapping[str, np.ndarray],
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build fixed-shape Hiera, tracker FPN, and detector PAFPN plans."""

    precision = graph_ops.normalize_precision(precision)
    trt = graph_ops._trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _WORKSPACE_BYTES)
    pixel_values = network.add_input("pixel_values", trt.float32, (1, 3, _IMAGE_SIZE, _IMAGE_SIZE))
    if pixel_values is None:
        raise RuntimeError("TensorRT failed to create the SAM2 HOI image input")

    graph_ops.reset_weight_refs()
    try:
        stages = _add_hiera(network, pixel_values, weights, precision=precision)
        fpn = _add_fpn(network, stages, weights, precision=precision)
        tracker_0, tracker_1, tracker_2, tracker_position_2 = _add_tracker_front_outputs(
            network,
            fpn,
            weights,
            precision=precision,
        )
        detector = _add_pafpn(network, (fpn[1], fpn[2], fpn[3]), weights, precision=precision)

        output_dtype = graph_ops.runtime_dtype(precision)
        graph_ops.mark_output(network, tracker_0, "tracker_feature_0", dtype=output_dtype)
        graph_ops.mark_output(network, tracker_1, "tracker_feature_1", dtype=output_dtype)
        graph_ops.mark_output(network, tracker_2, "tracker_feature_2", dtype=trt.float32)
        graph_ops.mark_output(
            network,
            tracker_position_2,
            "tracker_position_2",
            dtype=trt.float32,
        )
        for index, tensor in enumerate(detector):
            graph_ops.mark_output(
                network,
                tensor,
                f"detector_feature_{index}",
                dtype=output_dtype,
            )
        if verbose:
            print(
                "[trtmc build] Building native SAM2 HOI Hiera + FPN + PAFPN plan ...",
                file=sys.stderr,
            )
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("TensorRT SAM2 HOI image feature engine build failed")
        return bytes(plan)
    finally:
        graph_ops.reset_weight_refs()


def build_phase_a_image_front_engine(
    weights: Mapping[str, np.ndarray],
    *,
    precision: str = "bf16",
    verbose: bool = False,
) -> bytes:
    """Build Hiera/FPN tracker outputs and the three Phase-A PAFPN roots."""

    precision = graph_ops.normalize_precision(precision)
    if precision != "bf16":
        raise ValueError("SAM2 HOI Phase-A image front is qualified only for bf16")
    trt = graph_ops._trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _WORKSPACE_BYTES)
    pixel_values = network.add_input("pixel_values", trt.float32, (1, 3, _IMAGE_SIZE, _IMAGE_SIZE))
    if pixel_values is None:
        raise RuntimeError("TensorRT failed to create the SAM2 HOI Phase-A image input")

    graph_ops.reset_weight_refs()
    try:
        stages = _add_hiera(network, pixel_values, weights, precision=precision)
        fpn = _add_fpn(network, stages, weights, precision=precision)
        tracker_0, tracker_1, tracker_2, tracker_position_2 = _add_tracker_front_outputs(
            network,
            fpn,
            weights,
            precision=precision,
        )
        output_dtype = graph_ops.runtime_dtype(precision)
        graph_ops.mark_output(network, tracker_0, "tracker_feature_0", dtype=output_dtype)
        graph_ops.mark_output(network, tracker_1, "tracker_feature_1", dtype=output_dtype)
        graph_ops.mark_output(network, tracker_2, "tracker_feature_2", dtype=trt.float32)
        graph_ops.mark_output(
            network,
            tracker_position_2,
            "tracker_position_2",
            dtype=trt.float32,
        )
        graph_ops.mark_output(network, fpn[1], "fpn_input_0", dtype=output_dtype)
        graph_ops.mark_output(network, fpn[3], "fpn_input_2", dtype=output_dtype)
        if verbose:
            print("[trtmc build] Building native SAM2 HOI Phase-A image front ...", file=sys.stderr)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("TensorRT SAM2 HOI Phase-A image front build failed")
        return bytes(plan)
    finally:
        graph_ops.reset_weight_refs()


__all__ = ["build_image_feature_engine", "build_phase_a_image_front_engine"]
