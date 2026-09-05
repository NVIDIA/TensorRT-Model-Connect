# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT layer vocabulary shared by the SAM2 Python graph builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .checkpoint_mapper import Checkpoint


class NetworkBuildError(RuntimeError):
    """TensorRT rejected the requested static SAM2 graph."""


@dataclass(frozen=True)
class WindowTensor:
    tensor: Any
    padded_height: int
    padded_width: int
    window_size: int


class TrtLayers:
    """Add named, strongly typed layers while retaining all host weights."""

    def __init__(self, trt: Any, network: Any, checkpoint: Checkpoint) -> None:
        self.trt = trt
        self.network = network
        self.checkpoint = checkpoint
        self._owned: list[np.ndarray] = []

    @staticmethod
    def _shape(tensor: Any) -> tuple[int, ...]:
        return tuple(int(value) for value in tensor.shape)

    def _layer(self, layer: Any, operation: str, name: str) -> Any:
        if layer is None:
            raise NetworkBuildError(f"{operation} rejected layer {name}")
        layer.name = name
        return layer

    def _required(self, name: str, shape: Sequence[int]) -> np.ndarray:
        return self.checkpoint.tensor(name, tuple(shape))

    def _bf16_weights(self, values: np.ndarray) -> Any:
        source = np.ascontiguousarray(values, dtype=np.float32)
        bits = source.view(np.uint32)
        rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
        self._owned.append(rounded)
        return self.trt.Weights(self.trt.bfloat16, rounded.ctypes.data, rounded.size)

    def _fp32_weights(self, values: np.ndarray) -> Any:
        source = np.ascontiguousarray(values, dtype=np.float32)
        self._owned.append(source)
        return self.trt.Weights(source)

    def _typed_weights(self, values: np.ndarray, dtype: Any) -> Any:
        if dtype == self.trt.float32:
            return self._fp32_weights(values)
        if dtype == self.trt.bfloat16:
            return self._bf16_weights(values)
        raise NetworkBuildError("SAM2 supports only FP32 and BF16 weights")

    def cast(self, tensor: Any, dtype: Any, name: str) -> Any:
        if tensor.dtype == dtype:
            return tensor
        return self._layer(self.network.add_cast(tensor, dtype), "cast", name).get_output(0)

    def shuffle(self, tensor: Any, shape: Sequence[int], name: str) -> Any:
        layer = self._layer(self.network.add_shuffle(tensor), "shuffle", name)
        layer.reshape_dims = tuple(shape)
        return layer.get_output(0)

    def transpose(self, tensor: Any, order: Sequence[int], name: str) -> Any:
        layer = self._layer(self.network.add_shuffle(tensor), "transpose", name)
        layer.first_transpose = tuple(order)
        return layer.get_output(0)

    def slice(
        self,
        tensor: Any,
        start: Sequence[int],
        shape: Sequence[int],
        stride: Sequence[int],
        mode: Any,
        name: str,
    ) -> Any:
        layer = self._layer(
            self.network.add_slice(tensor, tuple(start), tuple(shape), tuple(stride)),
            "slice",
            name,
        )
        layer.mode = mode
        return layer.get_output(0)

    def concatenate(self, tensors: Sequence[Any], axis: int, name: str) -> Any:
        layer = self._layer(self.network.add_concatenation(list(tensors)), "concatenation", name)
        layer.axis = axis
        return layer.get_output(0)

    def elementwise(self, lhs: Any, rhs: Any, operation: Any, name: str) -> Any:
        return self._layer(
            self.network.add_elementwise(lhs, rhs, operation), "elementwise", name
        ).get_output(0)

    def matrix_multiply(
        self, lhs: Any, lhs_operation: Any, rhs: Any, rhs_operation: Any, name: str
    ) -> Any:
        return self._layer(
            self.network.add_matrix_multiply(lhs, lhs_operation, rhs, rhs_operation),
            "matrix multiplication",
            name,
        ).get_output(0)

    def softmax(self, tensor: Any, axes: int, name: str) -> Any:
        layer = self._layer(self.network.add_softmax(tensor), "softmax", name)
        layer.axes = axes
        return layer.get_output(0)

    def constant(
        self,
        checkpoint_name: str,
        checkpoint_shape: Sequence[int],
        tensor_shape: Sequence[int],
        name: str,
    ) -> Any:
        values = self._required(checkpoint_name, checkpoint_shape)
        return self._layer(
            self.network.add_constant(tuple(tensor_shape), self._fp32_weights(values)),
            "constant",
            name,
        ).get_output(0)

    def owned_constant(
        self, values: Sequence[float] | np.ndarray, shape: Sequence[int], dtype: Any, name: str
    ) -> Any:
        storage = np.ascontiguousarray(values, dtype=np.float32)
        output = self._layer(
            self.network.add_constant(tuple(shape), self._fp32_weights(storage)),
            "constant",
            name,
        ).get_output(0)
        return output if dtype == self.trt.float32 else self.cast(output, dtype, f"{name}.cast")

    def scalar(self, value: float, rank: int, dtype: Any, name: str) -> Any:
        return self.owned_constant([value], (1,) * rank, dtype, name)

    def convolution(
        self,
        tensor: Any,
        weight_name: str,
        bias_name: str,
        input_channels: int,
        output_channels: int,
        kernel: int,
        stride: int,
        padding: int,
        groups: int,
        name: str,
    ) -> Any:
        kernel_values = self._required(
            weight_name, (output_channels, input_channels // groups, kernel, kernel)
        )
        kernel_weights = self._typed_weights(kernel_values, tensor.dtype)
        bias_weights = self.trt.Weights(tensor.dtype)
        if bias_name:
            bias_weights = self._typed_weights(
                self._required(bias_name, (output_channels,)), tensor.dtype
            )
        layer = self._layer(
            self.network.add_convolution_nd(
                tensor, output_channels, (kernel, kernel), kernel_weights, bias_weights
            ),
            "convolution",
            name,
        )
        layer.stride_nd = (stride, stride)
        layer.padding_nd = (padding, padding)
        layer.num_groups = groups
        return layer.get_output(0)

    def _fold_batch_norm(
        self,
        module_name: str,
        input_channels: int,
        output_channels: int,
        kernel: int,
        groups: int,
        epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = self._required(
            f"{module_name}.conv.weight",
            (output_channels, input_channels // groups, kernel, kernel),
        )
        gamma = self._required(f"{module_name}.bn.weight", (output_channels,))
        beta = self._required(f"{module_name}.bn.bias", (output_channels,))
        mean = self._required(f"{module_name}.bn.running_mean", (output_channels,))
        variance = self._required(f"{module_name}.bn.running_var", (output_channels,))
        scale = gamma / np.sqrt(variance + epsilon)
        folded_kernel = np.ascontiguousarray(raw * scale.reshape((-1, 1, 1, 1)))
        folded_bias = np.ascontiguousarray(beta - mean * scale)
        self._owned.extend((folded_kernel, folded_bias))
        return folded_kernel, folded_bias

    def convolution_batch_norm_silu(
        self,
        tensor: Any,
        module_name: str,
        input_channels: int,
        output_channels: int,
        kernel: int,
        stride: int,
        padding: int,
        groups: int,
        epsilon: float,
        name: str,
    ) -> Any:
        tensor = self.cast(tensor, self.trt.bfloat16, f"{name}.input_bf16")
        kernel_values, bias_values = self._fold_batch_norm(
            module_name, input_channels, output_channels, kernel, groups, epsilon
        )
        layer = self._layer(
            self.network.add_convolution_nd(
                tensor,
                output_channels,
                (kernel, kernel),
                self._bf16_weights(kernel_values),
                self._bf16_weights(bias_values),
            ),
            "convolution",
            f"{name}.conv_bn_folded",
        )
        layer.stride_nd = (stride, stride)
        layer.padding_nd = (padding, padding)
        layer.num_groups = groups
        return self.silu(layer.get_output(0), f"{name}.silu")

    def linear_bf16(
        self,
        tensor: Any,
        module_name: str,
        input_features: int,
        output_features: int,
        name: str,
    ) -> Any:
        input_shape = self._shape(tensor)
        tensor = self.cast(tensor, self.trt.bfloat16, f"{name}.input_bf16")
        weight = self.constant(
            f"{module_name}.weight",
            (output_features, input_features),
            (output_features, input_features),
            f"{name}.weight",
        )
        weight = self.cast(weight, self.trt.bfloat16, f"{name}.weight_bf16")
        weight_shape = [1] * len(input_shape)
        weight_shape[-2:] = (output_features, input_features)
        weight = self.shuffle(weight, weight_shape, f"{name}.weight_broadcast")
        output = self.matrix_multiply(
            tensor,
            self.trt.MatrixOperation.NONE,
            weight,
            self.trt.MatrixOperation.TRANSPOSE,
            f"{name}.matmul",
        )
        bias_shape = [1] * len(input_shape)
        bias_shape[-1] = output_features
        bias_values = self._required(f"{module_name}.bias", (output_features,))
        bias = self._layer(
            self.network.add_constant(tuple(bias_shape), self._fp32_weights(bias_values)),
            "constant",
            f"{name}.bias",
        ).get_output(0)
        bias = self.cast(bias, self.trt.bfloat16, f"{name}.bias_bf16")
        return self.elementwise(output, bias, self.trt.ElementWiseOperation.SUM, f"{name}.bias_add")

    def layer_norm_fp32(
        self, tensor: Any, module_name: str, channels: int, epsilon: float, name: str
    ) -> Any:
        shape = self._shape(tensor)
        tensor = self.cast(tensor, self.trt.float32, f"{name}.input_fp32")
        parameter_shape = (1,) * (len(shape) - 1) + (channels,)
        scale = self._layer(
            self.network.add_constant(
                parameter_shape,
                self._fp32_weights(self._required(f"{module_name}.weight", (channels,))),
            ),
            "constant",
            f"{name}.scale",
        ).get_output(0)
        bias = self._layer(
            self.network.add_constant(
                parameter_shape,
                self._fp32_weights(self._required(f"{module_name}.bias", (channels,))),
            ),
            "constant",
            f"{name}.bias",
        ).get_output(0)
        layer = self._layer(
            self.network.add_normalization_v2(tensor, scale, bias, 1 << (len(shape) - 1)),
            "normalization",
            name,
        )
        layer.epsilon = epsilon
        return layer.get_output(0)

    def gelu(self, tensor: Any, name: str) -> Any:
        return self._layer(
            self.network.add_activation(tensor, self.trt.ActivationType.GELU_ERF), "GELU", name
        ).get_output(0)

    def silu(self, tensor: Any, name: str) -> Any:
        sigmoid = self._layer(
            self.network.add_activation(tensor, self.trt.ActivationType.SIGMOID),
            "sigmoid",
            f"{name}.sigmoid",
        ).get_output(0)
        return self.elementwise(tensor, sigmoid, self.trt.ElementWiseOperation.PROD, name)

    def max_pool_nhwc(self, tensor: Any, kernel: int, stride: int, name: str) -> Any:
        tensor = self.transpose(tensor, (0, 3, 1, 2), f"{name}.to_nchw")
        layer = self._layer(
            self.network.add_pooling_nd(tensor, self.trt.PoolingType.MAX, (kernel, kernel)),
            "max pooling",
            f"{name}.pool",
        )
        layer.stride_nd = (stride, stride)
        layer.padding_nd = (0, 0)
        layer.average_count_excludes_padding = True
        return self.transpose(layer.get_output(0), (0, 2, 3, 1), f"{name}.to_nhwc")

    def resize_nchw(
        self,
        tensor: Any,
        output_height: int,
        output_width: int,
        interpolation: Any,
        coordinates: Any,
        name: str,
        cubic_coefficient: float = -0.75,
    ) -> Any:
        shape = self._shape(tensor)
        layer = self._layer(self.network.add_resize(tensor), "resize", name)
        layer.resize_mode = interpolation
        layer.coordinate_transformation = coordinates
        layer.shape = (shape[0], shape[1], output_height, output_width)
        if interpolation == self.trt.InterpolationMode.CUBIC:
            layer.cubic_coeff = cubic_coefficient
        return layer.get_output(0)

    def window_partition(
        self, tensor: Any, height: int, width: int, channels: int, window_size: int, name: str
    ) -> WindowTensor:
        padded_height = height + (-height % window_size)
        padded_width = width + (-width % window_size)
        if (padded_height, padded_width) != (height, width):
            tensor = self.slice(
                tensor,
                (0, 0, 0, 0),
                (1, padded_height, padded_width, channels),
                (1, 1, 1, 1),
                self.trt.SampleMode.FILL,
                f"{name}.pad",
            )
        blocked = self.shuffle(
            tensor,
            (
                1,
                padded_height // window_size,
                window_size,
                padded_width // window_size,
                window_size,
                channels,
            ),
            f"{name}.block",
        )
        ordered = self.transpose(blocked, (0, 1, 3, 2, 4, 5), f"{name}.order")
        windows = self.shuffle(
            ordered,
            (
                (padded_height // window_size) * (padded_width // window_size),
                window_size,
                window_size,
                channels,
            ),
            f"{name}.windows",
        )
        return WindowTensor(windows, padded_height, padded_width, window_size)

    def window_unpartition(
        self,
        windows: WindowTensor,
        tensor: Any,
        output_height: int,
        output_width: int,
        channels: int,
        output_window_size: int,
        name: str,
    ) -> Any:
        reduction = windows.window_size // output_window_size
        padded_height = windows.padded_height // reduction
        padded_width = windows.padded_width // reduction
        blocked = self.shuffle(
            tensor,
            (
                1,
                padded_height // output_window_size,
                padded_width // output_window_size,
                output_window_size,
                output_window_size,
                channels,
            ),
            f"{name}.block",
        )
        ordered = self.transpose(blocked, (0, 1, 3, 2, 4, 5), f"{name}.order")
        padded = self.shuffle(ordered, (1, padded_height, padded_width, channels), f"{name}.merge")
        if (padded_height, padded_width) == (output_height, output_width):
            return padded
        return self.slice(
            padded,
            (0, 0, 0, 0),
            (1, output_height, output_width, channels),
            (1, 1, 1, 1),
            self.trt.SampleMode.STRICT_BOUNDS,
            f"{name}.crop",
        )
