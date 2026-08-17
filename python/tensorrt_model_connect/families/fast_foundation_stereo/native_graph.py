# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, family-owned TensorRT Network Definition helpers.

PyTorch is used only as the checkpoint container.  Every operation below adds
TensorRT layers directly to an ``INetworkDefinition``; no exported interchange
graph is created or parsed.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import numpy as np


class NativeGraph:
    """Build strongly typed TensorRT graphs from the distilled module tree."""

    def __init__(self, network: Any, trt: Any, *, fp16: bool) -> None:
        self.network = network
        self.trt = trt
        self.fp16 = fp16
        self.work_np_dtype = np.float16 if fp16 else np.float32
        self.work_trt_dtype = trt.float16 if fp16 else trt.float32
        # TensorRT weights refer to their host buffers until engine build.
        # Retain every converted checkpoint array for the graph's lifetime.
        self._weight_buffers: list[np.ndarray] = []

    @staticmethod
    def _tuple(value: Any, rank: int) -> tuple[int, ...]:
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            result = tuple(int(item) for item in value)
            if len(result) != rank:
                raise ValueError(f"expected {rank} values, got {result}")
            return result
        return (int(value),) * rank

    @staticmethod
    def _array(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
        if value is None:
            raise ValueError("cannot convert None to TensorRT weights")
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.ascontiguousarray(value, dtype=dtype)

    def _np_dtype_for(self, tensor: Any) -> np.dtype:
        return np.float16 if tensor.dtype == self.trt.float16 else np.float32

    def constant(
        self,
        values: Any,
        shape: tuple[int, ...] | None = None,
        *,
        dtype: np.dtype | None = None,
        target_dtype: Any | None = None,
    ) -> Any:
        array = self._array(values, dtype or np.float32)
        if shape is None:
            shape = tuple(array.shape)
        array = np.ascontiguousarray(array.reshape(shape))
        self._weight_buffers.append(array)
        output = self.network.add_constant(shape, self.trt.Weights(array)).get_output(0)
        return self.cast(output, target_dtype) if target_dtype is not None else output

    def scalar(self, value: float, rank: int, *, like: Any) -> Any:
        shape = (1,) * rank
        return self.constant(
            np.full(shape, value, dtype=self._np_dtype_for(like)),
            shape,
            dtype=self._np_dtype_for(like),
            target_dtype=like.dtype,
        )

    def cast(self, tensor: Any, dtype: Any) -> Any:
        if dtype is None or tensor.dtype == dtype:
            return tensor
        return self.network.add_cast(tensor, dtype).get_output(0)

    def name(self, tensor: Any, name: str) -> Any:
        tensor.name = name
        return tensor

    def reshape(self, tensor: Any, shape: tuple[int, ...]) -> Any:
        layer = self.network.add_shuffle(tensor)
        layer.reshape_dims = tuple(int(dim) for dim in shape)
        return layer.get_output(0)

    def transpose(self, tensor: Any, permutation: tuple[int, ...]) -> Any:
        layer = self.network.add_shuffle(tensor)
        layer.second_transpose = self.trt.Permutation(permutation)
        return layer.get_output(0)

    def concat(self, tensors: Iterable[Any], axis: int) -> Any:
        layer = self.network.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)

    def slice(
        self,
        tensor: Any,
        start: tuple[int, ...],
        shape: tuple[int, ...],
        stride: tuple[int, ...] | None = None,
    ) -> Any:
        stride = stride or (1,) * len(start)
        return self.network.add_slice(tensor, start, shape, stride).get_output(0)

    def elementwise(self, operation: Any, lhs: Any, rhs: Any) -> Any:
        if isinstance(operation, str):
            operation = getattr(self.trt.ElementWiseOperation, operation.upper())
        return self.network.add_elementwise(lhs, rhs, operation).get_output(0)

    def add(self, lhs: Any, rhs: Any) -> Any:
        return self.elementwise("SUM", lhs, rhs)

    def sub(self, lhs: Any, rhs: Any) -> Any:
        return self.elementwise("SUB", lhs, rhs)

    def mul(self, lhs: Any, rhs: Any) -> Any:
        return self.elementwise("PROD", lhs, rhs)

    def div(self, lhs: Any, rhs: Any) -> Any:
        return self.elementwise("DIV", lhs, rhs)

    def reduce(self, tensor: Any, operation: Any, axes: Iterable[int], keep_dims: bool) -> Any:
        if isinstance(operation, str):
            operation = getattr(self.trt.ReduceOperation, operation.upper())
        mask = 0
        for axis in axes:
            mask |= 1 << (axis % len(tuple(tensor.shape)))
        return self.network.add_reduce(tensor, operation, mask, keep_dims).get_output(0)

    def reduce_sum(self, tensor: Any, axes: Iterable[int], keep_dims: bool = False) -> Any:
        return self.reduce(tensor, "SUM", axes, keep_dims)

    def reduce_avg(self, tensor: Any, axes: Iterable[int], keep_dims: bool = False) -> Any:
        return self.reduce(tensor, "AVG", axes, keep_dims)

    def reduce_max(self, tensor: Any, axes: Iterable[int], keep_dims: bool = False) -> Any:
        return self.reduce(tensor, "MAX", axes, keep_dims)

    def unary(self, operation: Any, tensor: Any) -> Any:
        if isinstance(operation, str):
            operation = getattr(self.trt.UnaryOperation, operation.upper())
        return self.network.add_unary(tensor, operation).get_output(0)

    def activation(self, tensor: Any, kind: str, *, alpha: float | None = None) -> Any:
        kind = kind.lower()
        if kind == "gelu":
            return self.gelu(tensor)
        activation = {
            "relu": self.trt.ActivationType.RELU,
            "leaky_relu": self.trt.ActivationType.LEAKY_RELU,
            "sigmoid": self.trt.ActivationType.SIGMOID,
            "tanh": self.trt.ActivationType.TANH,
        }[kind]
        layer = self.network.add_activation(tensor, activation)
        if alpha is not None:
            layer.alpha = alpha
        return layer.get_output(0)

    def gelu(self, tensor: Any, *, approximate: str = "none") -> Any:
        if approximate not in {"none", "tanh"}:
            raise ValueError(
                "Fast Foundation Stereo GELU approximation must be 'none' or 'tanh', "
                f"got {approximate!r}"
            )
        if approximate == "tanh":
            activation = getattr(self.trt.ActivationType, "GELU_TANH", None)
            if activation is None:
                raise RuntimeError("TensorRT does not expose native GELU_TANH activation")
            return self.network.add_activation(tensor, activation).get_output(0)

        # The checkpoint uses torch.nn.GELU(approximate="none").
        if hasattr(self.trt.ActivationType, "GELU_ERF"):
            return self.network.add_activation(tensor, self.trt.ActivationType.GELU_ERF).get_output(
                0
            )
        rank = len(tuple(tensor.shape))
        inv_sqrt_two = self.scalar(1.0 / np.sqrt(2.0), rank, like=tensor)
        half = self.scalar(0.5, rank, like=tensor)
        one = self.scalar(1.0, rank, like=tensor)
        erf = self.unary("ERF", self.mul(tensor, inv_sqrt_two))
        return self.mul(self.mul(tensor, half), self.add(one, erf))

    def softmax(self, tensor: Any, axis: int) -> Any:
        layer = self.network.add_softmax(tensor)
        layer.axes = 1 << (axis % len(tuple(tensor.shape)))
        return layer.get_output(0)

    def matmul(
        self,
        lhs: Any,
        rhs: Any,
        *,
        op_lhs: Any | None = None,
        op_rhs: Any | None = None,
    ) -> Any:
        op_lhs = op_lhs or self.trt.MatrixOperation.NONE
        op_rhs = op_rhs or self.trt.MatrixOperation.NONE
        return self.network.add_matrix_multiply(lhs, op_lhs, rhs, op_rhs).get_output(0)

    def linear(self, tensor: Any, module: Any) -> Any:
        tensor = self.cast(tensor, self.work_trt_dtype)
        weight = self._array(module.weight, self._np_dtype_for(tensor))
        out_features, in_features = weight.shape
        rank = len(tuple(tensor.shape))
        rhs_shape = (1,) * (rank - 2) + (in_features, out_features)
        rhs = self.constant(
            weight.T.reshape(rhs_shape),
            rhs_shape,
            dtype=self._np_dtype_for(tensor),
            target_dtype=tensor.dtype,
        )
        output = self.matmul(tensor, rhs)
        if getattr(module, "bias", None) is not None:
            bias_shape = (1,) * (rank - 1) + (out_features,)
            bias = self.constant(
                self._array(module.bias, self._np_dtype_for(output)).reshape(bias_shape),
                bias_shape,
                dtype=self._np_dtype_for(output),
                target_dtype=output.dtype,
            )
            output = self.add(output, bias)
        return output

    def linear_as_conv2d(self, tensor: Any, module: Any) -> Any:
        """Apply a channel-wise Linear to NCHW data as a 1x1 convolution.

        Keep the bias as a separate elementwise add. This preserves ``linear``'s
        FP16 matrix-product output boundary instead of allowing a convolution
        tactic to fold the bias into its accumulator.
        """

        weight = self._array(module.weight)
        if weight.ndim != 2:
            raise ValueError(f"Linear weight must have rank 2, got {weight.shape}")
        out_features, in_features = (int(value) for value in weight.shape)
        shape = tuple(int(value) for value in tensor.shape)
        if len(shape) != 4:
            raise ValueError(f"linear_as_conv2d requires NCHW rank 4, got {shape}")
        if shape[1] >= 0 and shape[1] != in_features:
            raise ValueError(
                f"Linear input width {in_features} does not match NCHW channels {shape[1]}"
            )

        convolution = SimpleNamespace(
            weight=np.ascontiguousarray(weight.reshape(out_features, in_features, 1, 1)),
            bias=None,
            out_channels=out_features,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
        )
        output = self.conv2d(tensor, convolution)
        bias_value = getattr(module, "bias", None)
        if bias_value is None:
            return output

        bias_shape = (1, out_features, 1, 1)
        bias_dtype = self._np_dtype_for(output)
        bias = self.constant(
            self._array(bias_value, bias_dtype).reshape(bias_shape),
            bias_shape,
            dtype=bias_dtype,
            target_dtype=output.dtype,
        )
        return self.add(output, bias)

    def _convolution(self, tensor: Any, module: Any, *, dimensions: int, deconv: bool) -> Any:
        tensor = self.cast(tensor, self.work_trt_dtype)
        weight_dtype = self._np_dtype_for(tensor)
        weight = self._array(module.weight, weight_dtype)
        bias = (
            self._array(module.bias, weight_dtype)
            if getattr(module, "bias", None) is not None
            else None
        )
        self._weight_buffers.append(weight)
        weights = self.trt.Weights(weight)
        if bias is not None:
            self._weight_buffers.append(bias)
            biases = self.trt.Weights(bias)
        else:
            biases = self.trt.Weights()
        kernel = self._tuple(module.kernel_size, dimensions)
        if deconv:
            layer = self.network.add_deconvolution_nd(
                tensor, int(module.out_channels), kernel, weights, biases
            )
        else:
            layer = self.network.add_convolution_nd(
                tensor, int(module.out_channels), kernel, weights, biases
            )
        layer.stride_nd = self._tuple(module.stride, dimensions)
        padding = self._tuple(module.padding, dimensions)
        dilation = self._tuple(module.dilation, dimensions)
        if hasattr(layer, "dilation_nd"):
            layer.dilation_nd = dilation
        layer.num_groups = int(module.groups)
        if deconv and hasattr(module, "output_padding"):
            output_padding = self._tuple(module.output_padding, dimensions)
            if any(output_padding) and hasattr(layer, "pre_padding"):
                layer.pre_padding = padding
                layer.post_padding = tuple(p - o for p, o in zip(padding, output_padding))
            else:
                layer.padding_nd = padding
        else:
            layer.padding_nd = padding
        return layer.get_output(0)

    def _fold_batch_norm_into_convolution(
        self,
        convolution: Any,
        batch_norm: Any,
        *,
        deconv: bool,
    ) -> SimpleNamespace:
        """Fold one eval BatchNorm into TensorRT convolution weights."""

        if bool(getattr(convolution, "training", False)) or bool(
            getattr(batch_norm, "training", False)
        ):
            raise RuntimeError("Conv-BN folding requires eval modules")
        if not bool(getattr(batch_norm, "affine", False)) or not bool(
            getattr(batch_norm, "track_running_stats", False)
        ):
            raise RuntimeError("Conv-BN folding requires affine running statistics")

        groups = int(convolution.groups)
        input_channels = int(convolution.in_channels)
        output_channels = int(convolution.out_channels)
        kernel = self._tuple(convolution.kernel_size, len(convolution.kernel_size))

        gamma = self._array(batch_norm.weight, np.float32)
        beta = self._array(batch_norm.bias, np.float32)
        mean = self._array(batch_norm.running_mean, np.float32)
        variance = self._array(batch_norm.running_var, np.float32)
        expected_parameters = (output_channels,)
        for name, value in (
            ("weight", gamma),
            ("bias", beta),
            ("running_mean", mean),
            ("running_var", variance),
        ):
            if value.shape != expected_parameters:
                raise RuntimeError(
                    f"BatchNorm {name} has shape {value.shape}, expected {expected_parameters}"
                )
        scale = gamma / np.sqrt(variance + float(batch_norm.eps))
        shift = beta - mean * scale

        # Match the existing native FP16 path: quantize checkpoint weights before
        # moving the FP32 BatchNorm scale across the convolution.
        weight = self._array(convolution.weight, self.work_np_dtype).astype(np.float32, copy=True)
        original_shape = weight.shape
        spatial_ones = (1,) * len(kernel)
        if deconv:
            # TensorRT deconvolution weights use logical CKDHW order. PyTorch's
            # ConvTranspose layout is [C_in, K_out / groups, ...], so the BN
            # output-channel scale is axis 1 within each group.
            expected_weight = (input_channels, output_channels // groups, *kernel)
            if weight.shape != expected_weight:
                raise RuntimeError(
                    f"deconvolution weight has shape {weight.shape}, expected {expected_weight}"
                )
            weight = weight.reshape(
                groups,
                input_channels // groups,
                output_channels // groups,
                *kernel,
            )
            weight *= scale.reshape(
                groups,
                1,
                output_channels // groups,
                *spatial_ones,
            )
            weight = weight.reshape(original_shape)
        else:
            expected_weight = (output_channels, input_channels // groups, *kernel)
            if weight.shape != expected_weight:
                raise RuntimeError(
                    f"convolution weight has shape {weight.shape}, expected {expected_weight}"
                )
            weight *= scale.reshape(output_channels, 1, *spatial_ones)

        bias_value = getattr(convolution, "bias", None)
        bias = (
            np.zeros(output_channels, dtype=np.float32)
            if bias_value is None
            else self._array(bias_value, self.work_np_dtype).astype(np.float32, copy=False)
        )
        if bias.shape != expected_parameters:
            raise RuntimeError(
                f"convolution bias has shape {bias.shape}, expected {expected_parameters}"
            )
        bias = bias * scale + shift

        attributes = {
            "weight": np.ascontiguousarray(weight, dtype=self.work_np_dtype),
            "bias": np.ascontiguousarray(bias, dtype=self.work_np_dtype),
            "in_channels": input_channels,
            "out_channels": output_channels,
            "kernel_size": convolution.kernel_size,
            "stride": convolution.stride,
            "padding": convolution.padding,
            "dilation": convolution.dilation,
            "groups": groups,
        }
        if hasattr(convolution, "output_padding"):
            attributes["output_padding"] = convolution.output_padding
        return SimpleNamespace(**attributes)

    def _convolution_batch_norm(
        self,
        tensor: Any,
        convolution: Any,
        batch_norm: Any,
        *,
        dimensions: int,
        deconv: bool,
    ) -> Any:
        folded = self._fold_batch_norm_into_convolution(
            convolution,
            batch_norm,
            deconv=deconv,
        )
        return self._convolution(tensor, folded, dimensions=dimensions, deconv=deconv)

    def conv2d(self, tensor: Any, module: Any) -> Any:
        return self._convolution(tensor, module, dimensions=2, deconv=False)

    def stacked_conv2d(self, tensor: Any, modules: Iterable[Any]) -> Any:
        """Evaluate compatible convolutions as one wider convolution.

        Stacking output channels is mathematically identical when every
        convolution consumes the same tensor and shares its spatial
        hyperparameters.  It lets TensorRT select one wider tactic and avoids
        rereading the input for several small sibling convolutions.
        """
        modules = tuple(modules)
        if len(modules) < 2:
            raise ValueError("stacked_conv2d requires at least two convolutions")

        first = modules[0]
        if int(first.groups) != 1:
            raise ValueError("stacked_conv2d currently supports ungrouped convolutions only")
        spatial_attributes = ("kernel_size", "stride", "padding", "dilation")
        for module in modules[1:]:
            for attribute in spatial_attributes:
                if self._tuple(getattr(module, attribute), 2) != self._tuple(
                    getattr(first, attribute), 2
                ):
                    raise ValueError(
                        f"stacked convolutions have different {attribute}: "
                        f"{getattr(first, attribute)!r} and {getattr(module, attribute)!r}"
                    )
            if int(module.groups) != int(first.groups):
                raise ValueError("stacked convolutions have different group counts")

        weights = [self._array(module.weight) for module in modules]
        if any(weight.shape[1:] != weights[0].shape[1:] for weight in weights[1:]):
            raise ValueError("stacked convolution weight shapes are incompatible")
        biases = [getattr(module, "bias", None) for module in modules]
        if any(bias is None for bias in biases) != all(bias is None for bias in biases):
            raise ValueError("stacked convolutions must either all have biases or all omit them")

        combined = SimpleNamespace(
            weight=np.concatenate(weights, axis=0),
            bias=(
                None
                if biases[0] is None
                else np.concatenate([self._array(bias) for bias in biases], axis=0)
            ),
            out_channels=sum(int(module.out_channels) for module in modules),
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            dilation=first.dilation,
            groups=first.groups,
        )
        return self.conv2d(tensor, combined)

    def conv3d(self, tensor: Any, module: Any) -> Any:
        return self._convolution(tensor, module, dimensions=3, deconv=False)

    def deconv2d(self, tensor: Any, module: Any) -> Any:
        return self._convolution(tensor, module, dimensions=2, deconv=True)

    def deconv3d(self, tensor: Any, module: Any) -> Any:
        return self._convolution(tensor, module, dimensions=3, deconv=True)

    def batch_norm(self, tensor: Any, module: Any) -> Any:
        rank = len(tuple(tensor.shape))
        channels = int(tuple(tensor.shape)[1])
        fp32 = self.cast(tensor, self.trt.float32)
        gamma = self._array(module.weight, np.float32)
        beta = self._array(module.bias, np.float32)
        mean = self._array(module.running_mean, np.float32)
        variance = self._array(module.running_var, np.float32)
        scale = gamma / np.sqrt(variance + float(module.eps))
        shift = beta - mean * scale
        shape = (1, channels) + (1,) * (rank - 2)
        scale_tensor = self.constant(scale.reshape(shape), shape)
        shift_tensor = self.constant(shift.reshape(shape), shape)
        output = self.add(self.mul(fp32, scale_tensor), shift_tensor)
        return self.cast(output, tensor.dtype)

    def instance_norm(self, tensor: Any, module: Any) -> Any:
        rank = len(tuple(tensor.shape))
        axes = tuple(range(2, rank))
        output_dtype = tensor.dtype
        fp32 = self.cast(tensor, self.trt.float32)
        mean = self.reduce_avg(fp32, axes, keep_dims=True)
        centered = self.sub(fp32, mean)
        variance = self.reduce_avg(self.mul(centered, centered), axes, keep_dims=True)
        epsilon = self.scalar(float(module.eps), rank, like=variance)
        inverse = self.unary("RECIP", self.unary("SQRT", self.add(variance, epsilon)))
        output = self.mul(centered, inverse)
        if bool(getattr(module, "affine", False)):
            channels = int(tuple(tensor.shape)[1])
            shape = (1, channels) + (1,) * (rank - 2)
            gamma = self.constant(self._array(module.weight, np.float32).reshape(shape), shape)
            beta = self.constant(self._array(module.bias, np.float32).reshape(shape), shape)
            output = self.add(self.mul(output, gamma), beta)
        return self.cast(output, output_dtype)

    def instance_norm_2d_normalization(
        self,
        tensor: Any,
        *,
        channels: int,
        epsilon: float,
        spatial_shape: tuple[int, int],
    ) -> Any:
        """Express one affine-free InstanceNorm2d with INormalizationLayer."""

        shape = tuple(int(value) for value in tensor.shape)
        expected_tail = (int(channels), *tuple(int(value) for value in spatial_shape))
        if len(shape) != 4 or shape[1:] != expected_tail:
            raise RuntimeError(
                "native InstanceNorm2d requires NCHW shape "
                f"[N, {expected_tail[0]}, {expected_tail[1]}, {expected_tail[2]}], got {shape}"
            )
        if tensor.dtype != self.trt.float16:
            raise RuntimeError(
                f"native InstanceNorm2d requires an FP16 input/output boundary, got {tensor.dtype}"
            )

        fp32 = self.cast(tensor, self.trt.float32)
        parameter_shape = (1, int(channels), 1, 1)
        scale = self.constant(
            np.ones(parameter_shape, dtype=np.float32),
            parameter_shape,
            dtype=np.float32,
            target_dtype=self.trt.float32,
        )
        shift = self.constant(
            np.zeros(parameter_shape, dtype=np.float32),
            parameter_shape,
            dtype=np.float32,
            target_dtype=self.trt.float32,
        )
        add_normalization = getattr(self.network, "add_normalization_v2", None)
        if add_normalization is None:
            add_normalization = getattr(self.network, "add_normalization", None)
        if add_normalization is None:
            raise RuntimeError("TensorRT INormalizationLayer is required for native InstanceNorm2d")
        # Axes 2 and 3 are the 2-D spatial dimensions.  One group per channel
        # makes TensorRT group normalization exactly InstanceNorm2d here.
        layer = add_normalization(fp32, scale, shift, (1 << 2) | (1 << 3))
        layer.num_groups = int(channels)
        layer.epsilon = float(epsilon)
        return self.cast(layer.get_output(0), self.trt.float16)

    def layer_norm(
        self,
        tensor: Any,
        module: Any,
        *,
        axes: tuple[int, ...],
        parameter_axis: int,
    ) -> Any:
        rank = len(tuple(tensor.shape))
        output_dtype = tensor.dtype
        compute_input = self.cast(tensor, self.trt.float32)
        width = int(self._array(module.weight).size)
        parameter_shape = [1] * rank
        parameter_shape[parameter_axis] = width
        gamma = self.constant(
            self._array(module.weight, np.float32).reshape(parameter_shape),
            tuple(parameter_shape),
            dtype=np.float32,
            target_dtype=self.trt.float32,
        )
        beta = self.constant(
            self._array(module.bias, np.float32).reshape(parameter_shape),
            tuple(parameter_shape),
            dtype=np.float32,
            target_dtype=self.trt.float32,
        )
        axes_mask = sum(1 << (axis % rank) for axis in axes)
        add_norm = getattr(self.network, "add_normalization_v2", None)
        if add_norm is None:
            add_norm = getattr(self.network, "add_normalization", None)
        if add_norm is not None:
            layer = add_norm(compute_input, gamma, beta, axes_mask)
            layer.epsilon = float(module.eps)
            return self.cast(layer.get_output(0), output_dtype)
        mean = self.reduce_avg(compute_input, axes, keep_dims=True)
        centered = self.sub(compute_input, mean)
        variance = self.reduce_avg(self.mul(centered, centered), axes, keep_dims=True)
        epsilon = self.scalar(float(module.eps), rank, like=variance)
        normalized = self.mul(
            centered,
            self.unary("RECIP", self.unary("SQRT", self.add(variance, epsilon))),
        )
        return self.cast(self.add(self.mul(normalized, gamma), beta), output_dtype)

    def layer_norm_channels(self, tensor: Any, module: Any) -> Any:
        return self.layer_norm(
            tensor,
            module,
            axes=(1,),
            parameter_axis=1,
        )

    def layer_norm_last(self, tensor: Any, module: Any) -> Any:
        rank = len(tuple(tensor.shape))
        return self.layer_norm(
            tensor,
            module,
            axes=(rank - 1,),
            parameter_axis=rank - 1,
        )

    def resize(
        self,
        tensor: Any,
        shape: tuple[int, ...],
        *,
        mode: str,
        align_corners: bool = False,
    ) -> Any:
        layer = self.network.add_resize(tensor)
        layer.resize_mode = {
            "nearest": self.trt.InterpolationMode.NEAREST,
            "linear": self.trt.InterpolationMode.LINEAR,
            "bilinear": self.trt.InterpolationMode.LINEAR,
            "trilinear": self.trt.InterpolationMode.LINEAR,
        }[mode]
        if mode != "nearest" and hasattr(layer, "coordinate_transformation"):
            transformation = (
                self.trt.ResizeCoordinateTransformation.ALIGN_CORNERS
                if align_corners
                else self.trt.ResizeCoordinateTransformation.HALF_PIXEL
            )
            layer.coordinate_transformation = transformation
        layer.shape = tuple(int(dim) for dim in shape)
        return layer.get_output(0)

    def pool2d(
        self,
        tensor: Any,
        *,
        kind: str,
        window: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int] = (0, 0),
    ) -> Any:
        pool_type = {
            "avg": self.trt.PoolingType.AVERAGE,
            "max": self.trt.PoolingType.MAX,
        }[kind]
        layer = self.network.add_pooling_nd(tensor, pool_type, window)
        layer.stride_nd = stride
        layer.padding_nd = padding
        return layer.get_output(0)

    def normalize_l2(self, tensor: Any, axis: int, *, epsilon: float = 1.0e-12) -> Any:
        rank = len(tuple(tensor.shape))
        fp32 = self.cast(tensor, self.trt.float32)
        squared = self.mul(fp32, fp32)
        norm_squared = self.reduce_sum(squared, (axis,), keep_dims=True)
        norm = self.unary("SQRT", norm_squared)
        epsilon_tensor = self.scalar(epsilon, rank, like=norm)
        denominator = self.elementwise("MAX", norm, epsilon_tensor)
        return self.div(fp32, denominator)

    def basic_conv(self, tensor: Any, module: Any, *, fold_batch_norm: bool = False) -> Any:
        conv = module.conv
        dimensions = len(self._tuple(conv.kernel_size, len(conv.kernel_size)))
        deconv = "Transpose" in conv.__class__.__name__
        bn = getattr(module, "bn", None)
        if bn is None:
            bn = getattr(module, "IN", None)
        has_batch_norm = bn is not None and bn.__class__.__name__ != "Identity"
        if fold_batch_norm:
            if not has_batch_norm or "InstanceNorm" in bn.__class__.__name__:
                raise RuntimeError("requested Conv-BN folding requires BatchNorm")
            output = self._convolution_batch_norm(
                tensor,
                conv,
                bn,
                dimensions=dimensions,
                deconv=deconv,
            )
        elif dimensions == 2:
            output = self.deconv2d(tensor, conv) if deconv else self.conv2d(tensor, conv)
        else:
            output = self.deconv3d(tensor, conv) if deconv else self.conv3d(tensor, conv)
        if has_batch_norm and not fold_batch_norm:
            if "InstanceNorm" in bn.__class__.__name__:
                output = self.instance_norm(output, bn)
            else:
                output = self.batch_norm(output, bn)
        relu = getattr(module, "relu", None)
        if isinstance(relu, bool):
            if relu:
                output = self.activation(output, "leaky_relu", alpha=0.01)
        elif relu is not None and relu.__class__.__name__ != "Identity":
            if "Leaky" in relu.__class__.__name__:
                output = self.activation(
                    output, "leaky_relu", alpha=float(getattr(relu, "negative_slope", 0.01))
                )
            else:
                output = self.activation(output, "relu")
        return output

    def sequential(self, tensor: Any, module: Any) -> Any:
        output = tensor
        for child in module:
            output = self.module(output, child)
        return output

    def resnet(self, tensor: Any, module: Any, *, fold_batch_norm: bool = False) -> Any:
        identity = tensor
        dimensions = len(module.conv1.kernel_size)
        if fold_batch_norm:
            if not hasattr(module, "bn1") or "InstanceNorm" in module.bn1.__class__.__name__:
                raise RuntimeError("requested ResNet Conv-BN folding requires BatchNorm bn1")
            output = self._convolution_batch_norm(
                tensor,
                module.conv1,
                module.bn1,
                dimensions=dimensions,
                deconv=False,
            )
        else:
            output = (
                self.conv2d(tensor, module.conv1)
                if dimensions == 2
                else self.conv3d(tensor, module.conv1)
            )
        if hasattr(module, "bn1") and not fold_batch_norm:
            bn1 = module.bn1
            output = (
                self.instance_norm(output, bn1)
                if "InstanceNorm" in bn1.__class__.__name__
                else self.batch_norm(output, bn1)
            )
        output = self.activation(output, "relu")
        if fold_batch_norm:
            if not hasattr(module, "bn2") or "InstanceNorm" in module.bn2.__class__.__name__:
                raise RuntimeError("requested ResNet Conv-BN folding requires BatchNorm bn2")
            output = self._convolution_batch_norm(
                output,
                module.conv2,
                module.bn2,
                dimensions=dimensions,
                deconv=False,
            )
        else:
            output = (
                self.conv2d(output, module.conv2)
                if dimensions == 2
                else self.conv3d(output, module.conv2)
            )
        if hasattr(module, "bn2") and not fold_batch_norm:
            bn2 = module.bn2
            output = (
                self.instance_norm(output, bn2)
                if "InstanceNorm" in bn2.__class__.__name__
                else self.batch_norm(output, bn2)
            )
        if getattr(module, "downsample", None) is not None:
            identity = self.sequential(identity, module.downsample)
        return self.activation(self.add(output, identity), "relu")

    def conv3d_reduced(
        self,
        tensor: Any,
        module: Any,
        *,
        fold_batch_norm: bool = False,
    ) -> Any:
        if not fold_batch_norm:
            return self.sequential(self.sequential(tensor, module.conv1), module.conv2)

        output = tensor
        for path in ("conv1", "conv2"):
            children = tuple(getattr(module, path, ()))
            actual = tuple(child.__class__.__name__ for child in children)
            expected = ("Conv3d", "SyncBatchNorm", "ReLU")
            if actual != expected:
                raise RuntimeError(
                    "Conv3dNormActReduced Conv-BN folding requires the distilled topology; "
                    f"{path} is {actual!r}, expected {expected!r}"
                )
            convolution, batch_norm, _relu = children
            output = self._convolution_batch_norm(
                output,
                convolution,
                batch_norm,
                dimensions=3,
                deconv=False,
            )
            output = self.activation(output, "relu")
        return output

    def feature_attention(self, volume: Any, feature: Any, module: Any) -> Any:
        attention = self.sequential(feature, module.feat_att)
        shape = tuple(int(dim) for dim in attention.shape)
        attention = self.reshape(attention, (shape[0], shape[1], 1, shape[2], shape[3]))
        return self.mul(volume, self.activation(attention, "sigmoid"))

    def forward_helper(
        self,
        tensor: Any,
        feature: Any,
        module: Any,
        *,
        fold_batch_norm: bool = False,
    ) -> Any:
        output = tensor
        for child in module.layers:
            if child.__class__.__name__ == "FeatureAtt":
                output = self.feature_attention(output, feature, child)
            elif fold_batch_norm and child.__class__.__name__ == "BasicConv":
                output = self.basic_conv(output, child, fold_batch_norm=True)
            else:
                output = self.module(output, child)
        return output

    def post_forward_helper(
        self,
        skip: Any,
        lower: Any,
        feature: Any,
        module: Any,
    ) -> Any:
        output = self.sequential(lower, module.upsample)
        output = self.add(output, skip) if module.op == "sum" else self.concat((output, skip), 1)
        for child in module.out:
            if child.__class__.__name__ == "FeatureAtt":
                output = self.feature_attention(output, feature, child)
            else:
                output = self.module(output, child)
        return output

    def edge_next_encoder(
        self,
        tensor: Any,
        module: Any,
        *,
        nchw_pointwise: bool = False,
        fold_gamma: bool = False,
        gelu_approximate: str = "none",
    ) -> Any:
        if gelu_approximate not in {"none", "tanh"}:
            raise ValueError(
                f"EdgeNext GELU approximation must be 'none' or 'tanh', got {gelu_approximate!r}"
            )
        if gelu_approximate == "tanh":
            input_shape = tuple(int(dimension) for dimension in tensor.shape)
            normalization = getattr(module, "norm", None)
            activation_module = getattr(module, "act", None)
            depthwise = getattr(module, "dwconv", None)
            pointwise1 = getattr(module, "pwconv1", None)
            pointwise2 = getattr(module, "pwconv2", None)
            topology = (
                module.__class__.__name__,
                normalization.__class__.__name__,
                activation_module.__class__.__name__,
                getattr(activation_module, "approximate", None),
                getattr(depthwise, "in_channels", None),
                getattr(depthwise, "out_channels", None),
                getattr(depthwise, "groups", None),
                getattr(pointwise1, "in_features", None),
                getattr(pointwise1, "out_features", None),
                getattr(pointwise2, "in_features", None),
                getattr(pointwise2, "out_features", None),
            )
            expected_topology = (
                "EdgeNextConvEncoder",
                "Identity",
                "GELU",
                "none",
                36,
                36,
                36,
                36,
                244,
                244,
                36,
            )
            if (
                not nchw_pointwise
                or self.work_trt_dtype != self.trt.float16
                or input_shape != (1, 36, 176, 176)
                or topology != expected_topology
            ):
                raise RuntimeError(
                    "GELU_TANH is scoped to the second validated FP16 DispHead NCHW "
                    "EdgeNext block; got "
                    f"nchw={nchw_pointwise}, dtype={self.work_trt_dtype!r}, "
                    f"input={input_shape}, topology={topology!r}"
                )

        folded_pwconv2 = None
        if fold_gamma:
            if not nchw_pointwise:
                raise RuntimeError("EdgeNext gamma folding requires NCHW pointwise lowering")
            gamma = getattr(module, "gamma", None)
            if gamma is None:
                raise RuntimeError("EdgeNext gamma folding requires a gamma checkpoint parameter")

            input_shape = tuple(int(dimension) for dimension in tensor.shape)
            hidden_width = int(getattr(module.pwconv2, "in_features", -1))
            output_width = int(getattr(module.pwconv2, "out_features", -1))
            expected_input_shape = (1, 36, 176, 176)
            if (
                input_shape != expected_input_shape
                or hidden_width not in (212, 244)
                or output_width != 36
            ):
                raise RuntimeError(
                    "EdgeNext gamma folding is specialized for the two validated DispHead "
                    f"36C blocks, got input={input_shape}, pwconv2={hidden_width}->{output_width}"
                )

            checkpoint_weight = self._array(module.pwconv2.weight)
            checkpoint_bias = self._array(getattr(module.pwconv2, "bias", None))
            checkpoint_gamma = self._array(gamma)
            expected_weight_shape = (36, hidden_width)
            if (
                checkpoint_weight.shape != expected_weight_shape
                or checkpoint_bias.shape != (36,)
                or checkpoint_gamma.shape != (36,)
            ):
                raise RuntimeError(
                    "EdgeNext gamma folding checkpoint shape drift: expected "
                    f"weight={expected_weight_shape}, bias=(36,), gamma=(36,), got "
                    f"weight={checkpoint_weight.shape}, bias={checkpoint_bias.shape}, "
                    f"gamma={checkpoint_gamma.shape}"
                )
            if any(
                array.dtype != np.float32
                for array in (checkpoint_weight, checkpoint_bias, checkpoint_gamma)
            ):
                raise RuntimeError("EdgeNext gamma folding requires FP32 checkpoint parameters")

            # Match the existing FP16 graph's checkpoint boundary exactly:
            # quantize weight and bias first, multiply each output channel by
            # FP32 gamma, then store the folded parameters as FP16.
            gamma_fp32 = checkpoint_gamma.reshape(36, 1)
            folded_weight = np.ascontiguousarray(
                (checkpoint_weight.astype(np.float16).astype(np.float32) * gamma_fp32).astype(
                    np.float16
                )
            )
            folded_bias = np.ascontiguousarray(
                (checkpoint_bias.astype(np.float16).astype(np.float32) * checkpoint_gamma).astype(
                    np.float16
                )
            )
            folded_pwconv2 = SimpleNamespace(
                weight=folded_weight,
                bias=folded_bias,
                in_features=hidden_width,
                out_features=36,
            )

        if nchw_pointwise and self.work_trt_dtype != self.trt.float16:
            raise RuntimeError("NCHW EdgeNext pointwise lowering requires an FP16 TensorRT graph")

        residual = self.cast(tensor, self.trt.float32) if nchw_pointwise else tensor
        # Conv/Linear are autocast-eligible even when the residual branch was
        # promoted to FP32 by the preceding layer-scale parameter.
        output = self.conv2d(self.cast(tensor, self.work_trt_dtype), module.dwconv)
        if module.norm.__class__.__name__ != "Identity":
            if "BatchNorm" in module.norm.__class__.__name__:
                output = self.batch_norm(output, module.norm)
            else:
                output = self.layer_norm_channels(output, module.norm)

        if nchw_pointwise:
            output = self.linear_as_conv2d(
                self.cast(output, self.work_trt_dtype),
                module.pwconv1,
            )
            output = (
                self.gelu(output, approximate="tanh")
                if gelu_approximate == "tanh"
                else self.gelu(output)
            )
            output = self.linear_as_conv2d(
                self.cast(output, self.work_trt_dtype),
                folded_pwconv2 if folded_pwconv2 is not None else module.pwconv2,
            )
        else:
            output = self.transpose(output, (0, 2, 3, 1))
            output = self.linear(output, module.pwconv1)
            output = self.gelu(output)
            output = self.linear(output, module.pwconv2)
        gamma = getattr(module, "gamma", None)
        if gamma is not None and not fold_gamma:
            channel_axis = 1 if nchw_pointwise else -1
            channels = int(output.shape[channel_axis])
            gamma_shape = (1, channels, 1, 1) if nchw_pointwise else (1, 1, 1, channels)
            gamma_tensor = self.constant(
                self._array(gamma, np.float32).reshape(gamma_shape),
                gamma_shape,
                dtype=np.float32,
                target_dtype=self.trt.float32,
            )
            output = self.mul(self.cast(output, self.trt.float32), gamma_tensor)
        elif fold_gamma:
            output = self.cast(output, self.trt.float32)
        if not nchw_pointwise:
            output = self.transpose(output, (0, 3, 1, 2))
        return self.add(self.cast(residual, output.dtype), output)

    def module(self, tensor: Any, module: Any) -> Any:
        name = module.__class__.__name__
        if name == "Identity" or name == "Dropout":
            return tensor
        if name in {"Sequential", "ModuleList"}:
            return self.sequential(tensor, module)
        if name == "Conv2d":
            return self.conv2d(tensor, module)
        if name == "Conv3d":
            return self.conv3d(tensor, module)
        if name == "ConvTranspose2d":
            return self.deconv2d(tensor, module)
        if name == "ConvTranspose3d":
            return self.deconv3d(tensor, module)
        if name in {"BatchNorm2d", "BatchNorm3d", "SyncBatchNorm"}:
            return self.batch_norm(tensor, module)
        if name in {"InstanceNorm2d", "InstanceNorm3d"}:
            return self.instance_norm(tensor, module)
        if name in {"LayerNorm", "LayerNorm2d"}:
            return (
                self.layer_norm_last(tensor, module)
                if name == "LayerNorm"
                else self.layer_norm_channels(tensor, module)
            )
        if name == "Linear":
            return self.linear(tensor, module)
        if name == "ReLU":
            return self.activation(tensor, "relu")
        if name == "LeakyReLU":
            return self.activation(
                tensor, "leaky_relu", alpha=float(getattr(module, "negative_slope", 0.01))
            )
        if name == "GELU":
            return self.gelu(tensor)
        if name == "Sigmoid":
            return self.activation(tensor, "sigmoid")
        if name == "Tanh":
            return self.activation(tensor, "tanh")
        if name in {"BasicConv", "BasicConv_IN"}:
            return self.basic_conv(tensor, module)
        if name == "Conv3dNormActReduced":
            return self.conv3d_reduced(tensor, module)
        if name in {"ResnetBasicBlock", "ResnetBasicBlock3D"}:
            return self.resnet(tensor, module)
        if name == "EdgeNextConvEncoder":
            return self.edge_next_encoder(tensor, module)
        if name == "Upsample":
            shape = tuple(int(dim) for dim in tensor.shape)
            scale = module.scale_factor
            if isinstance(scale, tuple):
                spatial_scale = tuple(float(item) for item in scale)
            else:
                spatial_scale = (float(scale),) * (len(shape) - 2)
            target = shape[:2] + tuple(
                int(round(dim * factor)) for dim, factor in zip(shape[2:], spatial_scale)
            )
            return self.resize(
                tensor,
                target,
                mode=str(module.mode),
                align_corners=bool(module.align_corners),
            )
        raise TypeError(f"unsupported Fast Foundation Stereo module: {name}")
