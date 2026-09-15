# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VoiceChat-owned runtime-absmax W8A8 graph construction.

The compressed VoiceChat build keeps the embedding and language-model head in
FP16 and applies symmetric INT8 quantization to the selected static Thinker
projections. Activations use one runtime abs-max scale per input row; weights
use one scale per output channel and are packed before TensorRT sees them.
"""

from __future__ import annotations

import importlib
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np


_INT8_QUANT_CHUNK_BYTES = 64 * 1024 * 1024
_INT8_WEIGHT_KEEPALIVE: weakref.WeakKeyDictionary[Any, list[np.ndarray]] = (
    weakref.WeakKeyDictionary()
)
_INT8_FINALIZED_WEIGHT_NETWORKS: weakref.WeakSet[Any] = weakref.WeakSet()


def _trt():
    """Load TensorRT only when a graph is actually constructed."""
    return importlib.import_module("tensorrt")


def _weak_network_contains(networks: weakref.WeakSet[Any], network: Any) -> bool:
    try:
        return network in networks
    except TypeError:
        # Non-pointer call paths may use small, non-weak-referenceable doubles.
        # Pointer-backed networks validate this requirement when retaining data.
        return False


def _weak_network_add(networks: weakref.WeakSet[Any], network: Any) -> None:
    try:
        networks.add(network)
    except TypeError:
        pass


def _retain_int8_weight_buffer(network: Any, buffer: np.ndarray) -> None:
    if _weak_network_contains(_INT8_FINALIZED_WEIGHT_NETWORKS, network):
        raise RuntimeError(
            "A TensorRT network with pointer-backed INT8 weights is one-shot; "
            "create a new network before adding or serializing it again"
        )
    try:
        _INT8_WEIGHT_KEEPALIVE.setdefault(network, []).append(buffer)
    except TypeError as error:
        raise TypeError(
            "TensorRT networks with pointer-backed INT8 weights must be "
            "weak-referenceable and hashable"
        ) from error


def prepare_int8_weight_serialization(network: Any) -> bool:
    """Validate one-shot state and report whether a network owns raw pointers."""
    if _weak_network_contains(_INT8_FINALIZED_WEIGHT_NETWORKS, network):
        raise RuntimeError(
            "A TensorRT network with pointer-backed INT8 weights can be "
            "serialized only once; create a new network for another build"
        )
    try:
        return network in _INT8_WEIGHT_KEEPALIVE
    except TypeError:
        return False


def release_int8_weight_buffers(network: Any) -> None:
    """Release pointer-backed weights and make their network terminal."""
    try:
        pointer_backed = network in _INT8_WEIGHT_KEEPALIVE
        _INT8_WEIGHT_KEEPALIVE.pop(network, None)
    except TypeError:
        pointer_backed = False
    if pointer_backed:
        _weak_network_add(_INT8_FINALIZED_WEIGHT_NETWORKS, network)


def build_serialized_network(builder: Any, network: Any, config: Any) -> Any:
    """Serialize once while retaining every explicit INT8 weight buffer."""
    pointer_backed = prepare_int8_weight_serialization(network)
    try:
        return builder.build_serialized_network(network, config)
    finally:
        if pointer_backed:
            release_int8_weight_buffers(network)


@contextmanager
def int8_weight_build_scope(network: Any):
    """Release the whole graph's buffers if construction leaves it invalid."""
    try:
        yield
    except BaseException:
        release_int8_weight_buffers(network)
        raise


def derive_weight_scale(
    weight_array: np.ndarray,
    *,
    chunk_bytes: int = 16 * 1024 * 1024,
) -> np.ndarray:
    """Derive symmetric per-output INT8 scales with bounded temporary memory."""
    weight = np.asarray(weight_array)
    if weight.ndim != 2:
        raise ValueError(f"VoiceChat INT8 GEMM weight must be rank 2, got {weight.shape}")
    if chunk_bytes <= 0:
        raise ValueError("INT8 scale chunk_bytes must be positive")

    lhs_width, rhs_width = (int(dim) for dim in weight.shape)
    fp32_values_per_chunk = max(1, chunk_bytes // np.dtype(np.float32).itemsize)
    rows_per_chunk = max(1, fp32_values_per_chunk // rhs_width)
    max_abs = np.zeros(rhs_width, dtype=np.float32)
    for row_start in range(0, lhs_width, rows_per_chunk):
        row_end = min(lhs_width, row_start + rows_per_chunk)
        chunk = np.asarray(weight[row_start:row_end], dtype=np.float32)
        if not np.all(np.isfinite(chunk)):
            raise ValueError("VoiceChat INT8 weight contains non-finite values")
        np.maximum(max_abs, np.max(np.abs(chunk), axis=0), out=max_abs)
    return np.maximum(max_abs / np.float32(127.0), np.float32(1.0e-8))


def quantize_int8_per_output_channel(
    weight_array: np.ndarray,
    weight_scale: np.ndarray,
    *,
    lhs_width: int,
    rhs_width: int,
    chunk_bytes: int = _INT8_QUANT_CHUNK_BYTES,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack an ``[input, output]`` matrix using one scale per output."""
    weight = np.asarray(weight_array)
    expected_size = lhs_width * rhs_width
    if weight.size != expected_size:
        raise ValueError(
            "INT8 weight has %d values; expected %d for shape (%d, %d)"
            % (weight.size, expected_size, lhs_width, rhs_width)
        )
    if weight.shape != (lhs_width, rhs_width):
        weight = weight.reshape(lhs_width, rhs_width)

    scale = np.asarray(weight_scale, dtype=np.float32).reshape(-1)
    if scale.size != rhs_width:
        raise ValueError(
            "INT8 weight scale has %d values; expected %d" % (scale.size, rhs_width)
        )
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("INT8 weight scales must be finite and positive")
    if chunk_bytes <= 0:
        raise ValueError("INT8 quantization chunk_bytes must be positive")

    fp32_values_per_chunk = max(1, chunk_bytes // np.dtype(np.float32).itemsize)
    rows_per_chunk = max(1, fp32_values_per_chunk // rhs_width)
    quantized = np.empty((lhs_width, rhs_width), dtype=np.int8)
    output_scale = scale.reshape(1, rhs_width)
    int8_info = np.iinfo(np.int8)
    for row_start in range(0, lhs_width, rows_per_chunk):
        row_end = min(lhs_width, row_start + rows_per_chunk)
        work = np.array(
            weight[row_start:row_end], dtype=np.float32, order="C", copy=True
        )
        if not np.all(np.isfinite(work)):
            raise ValueError("VoiceChat INT8 weight contains non-finite values")
        np.divide(work, output_scale, out=work)
        np.rint(work, out=work)
        np.clip(work, int8_info.min, int8_info.max, out=work)
        quantized[row_start:row_end] = work.astype(np.int8)
    return quantized, np.ascontiguousarray(scale)


def _cast_output_dtype(network: Any, tensor: Any, target_dtype: Any) -> Any:
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)


def _runtime_activation_scale(network: Any, activation: Any, graph_ops: Any) -> Any:
    """Return guarded runtime per-row scales for rank-2 activations."""
    trt = _trt()
    rank = len(tuple(activation.shape))
    if rank != 2:
        raise ValueError("VoiceChat dynamic INT8 activation scaling requires rank-2 input")

    absolute = network.add_unary(activation, trt.UnaryOperation.ABS).get_output(0)
    absmax = network.add_reduce(
        absolute, trt.ReduceOperation.MAX, 1 << 1, True
    ).get_output(0)
    scale_floor = (
        float(np.finfo(np.float16).tiny)
        if activation.dtype == trt.float16
        else float(np.finfo(np.float32).tiny)
    )
    guarded_amax_floor = graph_ops.add_constant(
        network,
        (1, 1),
        np.array([[np.float32(127.0 * scale_floor)]], dtype=np.float32),
        dtype=np.float32,
    )
    guarded_amax_floor = _cast_output_dtype(
        network, guarded_amax_floor, activation.dtype
    )
    guarded_absmax = network.add_elementwise(
        absmax, guarded_amax_floor, trt.ElementWiseOperation.MAX
    ).get_output(0)
    int8_max = graph_ops.add_constant(
        network,
        (1, 1),
        np.array([[127.0]], dtype=np.float32),
        dtype=np.float32,
    )
    int8_max = _cast_output_dtype(network, int8_max, activation.dtype)
    return network.add_elementwise(
        guarded_absmax, int8_max, trt.ElementWiseOperation.DIV
    ).get_output(0)


def _runtime_activation_unit_dq(
    network: Any,
    activation: Any,
    dynamic_scale: Any,
    output_dtype: Any,
    graph_ops: Any,
) -> Any:
    """Manually quantize by a dynamic scale, then dequantize with unit scale."""
    trt = _trt()
    normalized = network.add_elementwise(
        activation, dynamic_scale, trt.ElementWiseOperation.DIV
    ).get_output(0)
    rounded = network.add_unary(normalized, trt.UnaryOperation.ROUND).get_output(0)

    lower = graph_ops.add_constant(
        network, (1, 1), np.array([[-128.0]], dtype=np.float32), dtype=np.float32
    )
    upper = graph_ops.add_constant(
        network, (1, 1), np.array([[127.0]], dtype=np.float32), dtype=np.float32
    )
    lower = _cast_output_dtype(network, lower, activation.dtype)
    upper = _cast_output_dtype(network, upper, activation.dtype)
    clamped_low = network.add_elementwise(
        rounded, lower, trt.ElementWiseOperation.MAX
    ).get_output(0)
    clamped = network.add_elementwise(
        clamped_low, upper, trt.ElementWiseOperation.MIN
    ).get_output(0)
    quantized = network.add_cast(clamped, trt.int8).get_output(0)

    unit_scale = graph_ops.add_constant(
        network, (), np.array(1.0, dtype=np.float32), dtype=np.float32
    )
    unit_scale = _cast_output_dtype(network, unit_scale, output_dtype)
    dequantize = network.add_dequantize(quantized, unit_scale, output_dtype)
    if dequantize is None:
        raise RuntimeError("TensorRT rejected VoiceChat dynamic INT8 activation DQ")
    return dequantize.get_output(0)


def _wrap_int8_matmul(
    network: Any,
    activation: Any,
    weight_array: np.ndarray,
    weight_scale: np.ndarray,
    *,
    lhs_width: int,
    rhs_width: int,
    graph_ops: Any,
) -> Any:
    trt = _trt()
    output_dtype = activation.dtype
    with int8_weight_build_scope(network):
        quantized_weight, normalized_scale = quantize_int8_per_output_channel(
            weight_array,
            weight_scale,
            lhs_width=lhs_width,
            rhs_width=rhs_width,
        )
        _retain_int8_weight_buffer(network, quantized_weight)
        weight_layer = network.add_constant(
            (lhs_width, rhs_width),
            trt.Weights(
                trt.int8,
                quantized_weight.ctypes.data,
                quantized_weight.size,
            ),
        )
        if weight_layer is None:
            raise RuntimeError("TensorRT rejected a pre-quantized VoiceChat INT8 weight")
        weight_const = weight_layer.get_output(0)

        weight_scale_tensor = graph_ops.add_constant(
            network,
            normalized_scale.shape,
            normalized_scale,
            dtype=np.float32,
        )
        weight_scale_tensor = _cast_output_dtype(
            network, weight_scale_tensor, output_dtype
        )
        dequantized_weight = network.add_dequantize(
            weight_const, weight_scale_tensor, output_dtype
        )
        if dequantized_weight is None:
            raise RuntimeError("TensorRT rejected VoiceChat INT8 weight DQ")
        dequantized_weight.axis = 1

        dynamic_scale = _runtime_activation_scale(network, activation, graph_ops)
        dequantized_activation = _runtime_activation_unit_dq(
            network, activation, dynamic_scale, output_dtype, graph_ops
        )
        matmul = network.add_matrix_multiply(
            dequantized_activation,
            trt.MatrixOperation.NONE,
            dequantized_weight.get_output(0),
            trt.MatrixOperation.NONE,
        )
        output = network.add_elementwise(
            matmul.get_output(0), dynamic_scale, trt.ElementWiseOperation.PROD
        ).get_output(0)
        return _cast_output_dtype(network, output, output_dtype)


@dataclass(frozen=True)
class VoiceChatQuantContext:
    """Selected Thinker weights and their derived per-output INT8 scales."""

    weight_scales: dict[str, np.ndarray]
    graph_ops: Any

    @classmethod
    def from_weights(
        cls,
        weights: dict[str, Any],
        weight_names: list[str] | tuple[str, ...],
        graph_ops: Any,
    ) -> "VoiceChatQuantContext":
        missing = [name for name in weight_names if name not in weights]
        if missing:
            raise ValueError(
                "VoiceChat INT8 weights are missing: " + ", ".join(missing[:8])
            )
        return cls(
            weight_scales={
                name: derive_weight_scale(np.asarray(weights[name]))
                for name in weight_names
            },
            graph_ops=graph_ops,
        )

    def maybe_quantized_matmul(
        self,
        network: Any,
        lhs: Any,
        lhs_width: int,
        rhs_width: int,
        rhs_weights: np.ndarray,
        weight_name: str,
        dtype: np.dtype = np.float32,
    ) -> Any:
        scales = self.weight_scales.get(weight_name)
        if scales is None:
            return self.graph_ops.add_matmul_rhs_constant(
                network,
                lhs,
                lhs_width,
                rhs_width,
                rhs_weights,
                dtype=dtype,
            )
        return _wrap_int8_matmul(
            network,
            lhs,
            rhs_weights,
            scales,
            lhs_width=lhs_width,
            rhs_width=rhs_width,
            graph_ops=self.graph_ops,
        )


# Keep the builder type annotation concise at integration sites.
QuantContext = VoiceChatQuantContext


__all__ = [
    "QuantContext",
    "VoiceChatQuantContext",
    "build_serialized_network",
    "derive_weight_scale",
    "int8_weight_build_scope",
    "prepare_int8_weight_serialization",
    "quantize_int8_per_output_channel",
    "release_int8_weight_buffers",
]
