# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-free TensorRT graph vocabulary for MiniMax-H3.

All math in this module uses TensorRT native layers. The qualified single-device
graph uses fused ``IAttention`` and contains no plugin or distributed layer.
"""

from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
from functools import wraps
import math

import ml_dtypes
import numpy as np

from tensorrt_model_connect import trt_compat

from .config import resolve_workspace_bytes


trt = trt_compat.get_trt()


# TensorRT's explicit BF16 ``Weights`` constructor stores a pointer rather
# than owning its input.  Retain every backing array until the associated
# network has finished building, including temporary packed QKV buffers.
_WEIGHT_BUFFER_KEEPALIVE: dict[int, list[np.ndarray]] = {}
_ACTIVE_BUILD_NETWORKS: ContextVar[set[int] | None] = ContextVar(
    "minimax_h3_active_build_networks", default=None
)


def cleanup_failed_build(function):
    """Release retained arrays and consumed weights after graph-build failures."""

    @wraps(function)
    def wrapped(weights, *args, **kwargs):
        network_ids: set[int] = set()
        token = _ACTIVE_BUILD_NETWORKS.set(network_ids)
        try:
            return function(weights, *args, **kwargs)
        except BaseException:
            for network_id in network_ids:
                _WEIGHT_BUFFER_KEEPALIVE.pop(network_id, None)
            if kwargs.get("consume_weights", False):
                weights.clear()
            raise
        finally:
            _ACTIVE_BUILD_NETWORKS.reset(token)

    return wrapped


def configure_builder(config, *, weight_streaming: bool = False) -> None:
    """Retain graph metadata and enable RTX weight streaming when requested."""

    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if not isinstance(weight_streaming, bool):
        raise ValueError("MiniMax-H3 weight_streaming must be a boolean")
    if not weight_streaming:
        return
    flag = getattr(trt.BuilderFlag, "WEIGHT_STREAMING", None)
    if flag is None:
        raise RuntimeError(
            "Selected TensorRT-RTX bindings do not expose BuilderFlag.WEIGHT_STREAMING"
        )
    config.set_flag(flag)
    if not config.get_flag(flag):
        raise RuntimeError("TensorRT-RTX did not enable MiniMax-H3 weight streaming")


def configure_workspace(config, workspace_bytes: int | None, *, default_bytes: int) -> int:
    """Apply and return the exact TensorRT tactic-workspace limit."""

    resolved = resolve_workspace_bytes(workspace_bytes, default_bytes=default_bytes)
    pool = trt.MemoryPoolType.WORKSPACE
    config.set_memory_pool_limit(pool, resolved)
    applied = int(config.get_memory_pool_limit(pool))
    if applied != resolved:
        raise RuntimeError(
            "TensorRT did not apply the requested MiniMax-H3 workspace limit: "
            f"requested={resolved}, applied={applied}"
        )
    return applied


def validate_native_network(
    network,
    *,
    expected_attentions: int,
    label: str,
) -> dict[str, int]:
    """Fail closed on any layer outside the selected native H3 contract."""

    counts = Counter(network.get_layer(index).type for index in range(network.num_layers))
    expected = {
        trt.LayerType.ATTENTION_INPUT: expected_attentions,
        trt.LayerType.ATTENTION_OUTPUT: expected_attentions,
    }
    forbidden = (
        trt.LayerType.PLUGIN,
        trt.LayerType.PLUGIN_V2,
        trt.LayerType.PLUGIN_V3,
        trt.LayerType.DIST_COLLECTIVE,
    )
    violations = {
        str(kind): counts[kind] for kind, wanted in expected.items() if counts[kind] != wanted
    }
    violations.update({str(kind): counts[kind] for kind in forbidden if counts[kind]})
    if violations:
        raise RuntimeError(f"MiniMax-H3 {label} native layer contract failed: {violations}")
    return {
        "attention_input": counts[trt.LayerType.ATTENTION_INPUT],
        "attention_output": counts[trt.LayerType.ATTENTION_OUTPUT],
        "plugin": 0,
        "plugin_v2": 0,
        "plugin_v3": 0,
        "dist_collective": 0,
    }


def _add_constant(network, array: np.ndarray):
    array = np.ascontiguousarray(array)
    network_id = id(network)
    _WEIGHT_BUFFER_KEEPALIVE.setdefault(network_id, []).append(array)
    active_networks = _ACTIVE_BUILD_NETWORKS.get()
    if active_networks is not None:
        active_networks.add(network_id)
    if array.dtype == np.dtype(ml_dtypes.bfloat16):
        weights = trt.Weights(trt.bfloat16, array.ctypes.data, array.size)
    else:
        weights = array
    layer = network.add_constant(tuple(array.shape), weights)
    if layer is None:
        raise RuntimeError(f"TensorRT rejected a MiniMax-H3 {array.dtype} constant")
    return layer.get_output(0)


def release_weight_buffers(network) -> None:
    """Release explicit TensorRT weight buffers after engine serialization."""

    _WEIGHT_BUFFER_KEEPALIVE.pop(id(network), None)


def constant(network, value, *, dtype=np.float32):
    array = np.ascontiguousarray(value, dtype=dtype)
    return _add_constant(network, array)


def weight_constant(network, value):
    """Create a constant without expanding checkpoint-native BF16 to FP32."""

    return _add_constant(network, np.asarray(value))


def cast(network, tensor, dtype):
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def linear(
    network,
    tensor,
    weight,
    bias=None,
    *,
    bf16: bool = True,
    compute_dtype=None,
):
    """PyTorch ``[out, in]`` linear expressed as native TensorRT GEMM."""

    tensor_rank = len(tuple(tensor.shape))
    rhs_value = np.asarray(weight)
    if tensor_rank > 2:
        # TensorRT MatrixMultiply applies NumPy-style broadcast across the
        # leading dimensions only when both operands expose those dimensions.
        # VAE layers are [batch, rows, width], so make the shared weight
        # explicitly [1, out, in] instead of relying on implicit rank lift.
        rhs_value = rhs_value.reshape((1,) * (tensor_rank - 2) + rhs_value.shape)
    rhs = weight_constant(network, rhs_value)
    if compute_dtype is None and bf16:
        compute_dtype = trt.bfloat16
    if compute_dtype is None:
        compute_dtype = tensor.dtype
    tensor = cast(network, tensor, compute_dtype)
    rhs = cast(network, rhs, compute_dtype)
    output = network.add_matrix_multiply(
        tensor, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    if bias is not None:
        shape = (1,) * (len(tuple(output.shape)) - 1) + (-1,)
        bias_tensor = weight_constant(network, np.asarray(bias).reshape(shape))
        bias_tensor = cast(network, bias_tensor, output.dtype)
        output = network.add_elementwise(
            output, bias_tensor, trt.ElementWiseOperation.SUM
        ).get_output(0)
    return output


def silu(network, tensor):
    source_dtype = tensor.dtype
    if source_dtype != trt.bfloat16:
        sigmoid = network.add_activation(tensor, trt.ActivationType.SIGMOID).get_output(0)
        return network.add_elementwise(tensor, sigmoid, trt.ElementWiseOperation.PROD).get_output(0)
    value = cast(network, tensor, trt.float32)
    sigmoid = network.add_activation(value, trt.ActivationType.SIGMOID).get_output(0)
    activated = network.add_elementwise(value, sigmoid, trt.ElementWiseOperation.PROD).get_output(0)
    return cast(network, activated, source_dtype)


def rms_norm(network, tensor, weight, width: int, eps: float):
    """PyTorch RMSNorm using native reduce, unary and elementwise layers."""

    source_dtype = tensor.dtype
    value = cast(network, tensor, trt.float32)
    square = network.add_elementwise(value, value, trt.ElementWiseOperation.PROD).get_output(0)
    axis = 1 << (len(tuple(value.shape)) - 1)
    mean = network.add_reduce(square, trt.ReduceOperation.AVG, axis, True).get_output(0)
    broadcast_shape = (1,) * len(tuple(value.shape))
    eps_tensor = constant(network, np.full(broadcast_shape, eps, dtype=np.float32))
    variance = network.add_elementwise(mean, eps_tensor, trt.ElementWiseOperation.SUM).get_output(0)
    root = network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
    inverse = network.add_unary(root, trt.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(value, inverse, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    gamma_shape = (1,) * (len(tuple(value.shape)) - 1) + (width,)
    gamma = weight_constant(network, np.asarray(weight).reshape(gamma_shape))
    gamma = cast(network, gamma, value.dtype)
    normalized = network.add_elementwise(
        normalized, gamma, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return cast(network, normalized, source_dtype)


def gather_rows(network, table, indices):
    return network.add_gather(table, indices, 0).get_output(0)


def _shape_dim(network, tensor, axis: int):
    """Return one runtime dimension as a one-element shape tensor."""

    shape = network.add_shape(tensor).get_output(0)
    return network.add_slice(shape, (axis,), (1,), (1,)).get_output(0)


def _shape_vector(network, values):
    parts = [
        constant(network, np.asarray([value], dtype=np.int64), dtype=np.int64)
        if isinstance(value, (int, np.integer))
        else value
        for value in values
    ]
    if len(parts) == 1:
        return parts[0]
    concat = network.add_concatenation(parts)
    concat.axis = 0
    return concat.get_output(0)


def dynamic_slice(network, tensor, starts: tuple[int, ...], sizes: tuple[int | None, ...]):
    """Slice a tensor while preserving dimensions marked ``None`` at runtime."""

    if len(starts) != len(sizes):
        raise ValueError("MiniMax-H3 dynamic slice rank mismatch")
    runtime_sizes = []
    for axis, size in enumerate(sizes):
        if size is not None:
            runtime_sizes.append(size)
            continue
        static_size = int(tensor.shape[axis])
        runtime_sizes.append(
            static_size if static_size >= 0 else _shape_dim(network, tensor, axis)
        )
    if all(isinstance(size, (int, np.integer)) for size in runtime_sizes):
        return network.add_slice(
            tensor,
            starts,
            tuple(int(size) for size in runtime_sizes),
            (1,) * len(sizes),
        ).get_output(0)
    initial_sizes = tuple(1 if size is None else size for size in sizes)
    layer = network.add_slice(tensor, starts, initial_sizes, (1,) * len(sizes))
    layer.set_input(2, _shape_vector(network, runtime_sizes))
    return layer.get_output(0)


def slice_rows_from_end(network, tensor, *, offset: int, rows: int):
    """Take fixed rows from a dynamic 2-D tensor, measured from its end."""

    total_rows = _shape_dim(network, tensor, 0)
    offset_tensor = constant(network, np.asarray([offset], dtype=np.int64), dtype=np.int64)
    start_row = network.add_elementwise(
        total_rows, offset_tensor, trt.ElementWiseOperation.SUB
    ).get_output(0)
    width = int(tensor.shape[1])
    layer = network.add_slice(tensor, (0, 0), (rows, width), (1, 1))
    layer.set_input(1, _shape_vector(network, (start_row, 0)))
    layer.set_input(2, _shape_vector(network, (rows, width)))
    return layer.get_output(0)


def slice_rows_like_from_end(network, tensor, reference, *, trailing_reference=None):
    """Take runtime-sized rows from ``tensor`` using modality input shapes.

    ``reference`` supplies the number of rows to return.  When a trailing
    modality is supplied, its runtime row count is included in the offset
    measured from the packed sequence end.  Only shape tensors participate in
    this operation, so the reference contents are never copied or consumed.
    """

    total_rows = _shape_dim(network, tensor, 0)
    rows = _shape_dim(network, reference, 0)
    offset = rows
    if trailing_reference is not None:
        trailing_rows = _shape_dim(network, trailing_reference, 0)
        offset = network.add_elementwise(
            offset, trailing_rows, trt.ElementWiseOperation.SUM
        ).get_output(0)
    start_row = network.add_elementwise(
        total_rows, offset, trt.ElementWiseOperation.SUB
    ).get_output(0)
    width = int(tensor.shape[1])
    layer = network.add_slice(tensor, (0, 0), (1, width), (1, 1))
    layer.set_input(1, _shape_vector(network, (start_row, 0)))
    layer.set_input(2, _shape_vector(network, (rows, width)))
    return layer.get_output(0)


def modulate(network, normalized, shift, scale):
    one = constant(
        network,
        np.ones((1,) * len(tuple(normalized.shape)), dtype=np.float32),
    )
    one = cast(network, one, normalized.dtype)
    scale = cast(network, scale, normalized.dtype)
    shift = cast(network, shift, normalized.dtype)
    scale = network.add_elementwise(scale, one, trt.ElementWiseOperation.SUM).get_output(0)
    value = network.add_elementwise(normalized, scale, trt.ElementWiseOperation.PROD).get_output(0)
    # Match PyTorch's BF16 rounding between the product and shift addition
    # without breaking TensorRT's native elementwise fusion.
    zero = constant(network, np.zeros((1,) * len(tuple(value.shape)), dtype=np.float32))
    zero = cast(network, zero, value.dtype)
    value = network.add_elementwise(value, zero, trt.ElementWiseOperation.SUM).get_output(0)
    return network.add_elementwise(value, shift, trt.ElementWiseOperation.SUM).get_output(0)


def gated_residual(network, residual, update, gate):
    gate = cast(network, gate, update.dtype)
    update = network.add_elementwise(update, gate, trt.ElementWiseOperation.PROD).get_output(0)
    # Preserve PyTorch's BF16 product rounding before the residual sum. The
    # zero-add remains in TensorRT's native fused kernel but prevents it from
    # contracting this sequence into a single-rounding multiply-add.
    zero = constant(network, np.zeros((1,) * len(tuple(update.shape)), dtype=np.float32))
    zero = cast(network, zero, update.dtype)
    update = network.add_elementwise(update, zero, trt.ElementWiseOperation.SUM).get_output(0)
    update = cast(network, update, residual.dtype)
    return network.add_elementwise(residual, update, trt.ElementWiseOperation.SUM).get_output(0)


def swiglu(network, tensor, weight_in, weight_out, ffn_dim: int):
    projected = linear(network, tensor, weight_in)
    value = dynamic_slice(network, projected, (0, 0), (None, ffn_dim))
    gate = dynamic_slice(network, projected, (0, ffn_dim), (None, ffn_dim))
    activated = silu(network, gate)
    hidden = network.add_elementwise(value, activated, trt.ElementWiseOperation.PROD).get_output(0)
    return linear(network, hidden, weight_out)


def fused_qkv(
    network,
    tensor,
    weights: dict,
    prefix: str,
    *,
    consume_weights: bool = False,
):
    """Pack Q/K/V into one TensorRT GEMM, matching Sol-Engine's lossless path."""

    keys = tuple(f"{prefix}.to_{name}.weight" for name in ("q", "k", "v"))
    packed_weight = np.concatenate([weights[key] for key in keys], axis=0)
    packed = linear(network, tensor, packed_weight)
    if consume_weights:
        # The packed array is retained by the network. Its three sources are
        # no longer referenced and can be released before serialization.
        for key in keys:
            weights.pop(key)
    width = int(packed.shape[1]) // 3
    return tuple(
        dynamic_slice(network, packed, (0, index * width), (None, width)) for index in range(3)
    )


def rows_to_heads(network, tensor, heads: int, head_dim: int):
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (-1, heads, head_dim)
    reshape.second_transpose = trt.Permutation([1, 0, 2])
    batch = network.add_shuffle(reshape.get_output(0))
    batch.reshape_dims = (1, heads, -1, head_dim)
    return batch.get_output(0)


def heads_to_rows(network, tensor, width: int):
    reshape = network.add_shuffle(tensor)
    reshape.first_transpose = trt.Permutation([0, 2, 1, 3])
    reshape.reshape_dims = (-1, width)
    return reshape.get_output(0)


def partial_rope(
    network,
    tensor,
    cos_half,
    sin_half,
    *,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    interleaved: bool = False,
):
    """Apply H3's 96-channel rotate-half MM-RoPE with native layers."""

    value = rows_to_heads(network, tensor, heads, head_dim)
    if interleaved:
        raise ValueError("MiniMax-H3 uses rotate-half, non-interleaved RoPE")
    rotary = dynamic_slice(network, value, (0, 0, 0, 0), (1, heads, None, rotary_dim))
    passthrough = dynamic_slice(
        network, value, (0, 0, 0, rotary_dim), (1, heads, None, head_dim - rotary_dim)
    )
    half = rotary_dim // 2
    first = dynamic_slice(network, rotary, (0, 0, 0, 0), (1, heads, None, half))
    second = dynamic_slice(network, rotary, (0, 0, 0, half), (1, heads, None, half))
    negative_second = network.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
    rotated_layer = network.add_concatenation((negative_second, first))
    rotated_layer.axis = 3

    def duplicate_table(table):
        table = cast(network, table, value.dtype)
        reshape = network.add_shuffle(table)
        reshape.reshape_dims = (1, 1, -1, half)
        duplicate = network.add_concatenation((reshape.get_output(0), reshape.get_output(0)))
        duplicate.axis = 3
        return duplicate.get_output(0)

    cos = duplicate_table(cos_half)
    sin = duplicate_table(sin_half)
    left = network.add_elementwise(rotary, cos, trt.ElementWiseOperation.PROD).get_output(0)
    right = network.add_elementwise(
        rotated_layer.get_output(0), sin, trt.ElementWiseOperation.PROD
    ).get_output(0)
    rotated = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)
    result = network.add_concatenation((rotated, passthrough))
    result.axis = 3
    return heads_to_rows(network, result.get_output(0), heads * head_dim)


def native_attention(network, q, k, v, *, heads: int, head_dim: int, name: str):
    """Full-sequence single-device fused TensorRT attention."""

    q = rows_to_heads(network, q, heads, head_dim)
    k = rows_to_heads(network, k, heads, head_dim)
    v = rows_to_heads(network, v, heads, head_dim)
    # Preserve BF16's exponent range. H3 residuals can exceed FP16's finite
    # limit before a later block normalizes them.
    scale = constant(
        network,
        np.full((1, 1, 1, 1), 1.0 / math.sqrt(head_dim), dtype=np.float32),
    )
    scale = cast(network, scale, q.dtype)
    q = network.add_elementwise(q, scale, trt.ElementWiseOperation.PROD).get_output(0)
    layer = network.add_attention(q, k, v, trt.AttentionNormalizationOp.SOFTMAX, False)
    if layer is None:
        raise RuntimeError("TensorRT failed to add MiniMax-H3 native attention")
    layer.name = name
    layer.metadata = f"trtmc.native_op=IAttention;source={name}"
    layer.get_output(0).name = f"{name}.output"
    layer.decomposable = False
    return heads_to_rows(network, layer.get_output(0), heads * head_dim)
