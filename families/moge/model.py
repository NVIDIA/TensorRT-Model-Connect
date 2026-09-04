# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT NetworkDefinition builder for MoGe-2 ViT-L.

All model topology is owned here. TensorRT receives the original dynamic RGB
image and builds the complete DINOv2 encoder, convolutional neck and heads,
and raw MoGe forward outputs without an exporter, parser, or custom kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_CHECKPOINT = Path("model.pt")
_INTERMEDIATE_LAYERS = (5, 11, 17, 23)
_HIDDEN = 1024
_HEADS = 16
_HEAD_DIM = 64
_PATCH = 14
_POSITION_GRID = 37
_NUM_TOKENS = 1800
_FOCAL_RECOVERY_SIZE = 64
_MIN_IMAGE_SIZE = 64
_OPT_IMAGE_SIZE = 518
_MAX_IMAGE_SIZE = 2048
_FAST_MIN_IMAGE_HEIGHT = 540
_FAST_MIN_IMAGE_WIDTH = 608
_FAST_OPT_IMAGE_HEIGHT = 1080
_FAST_OPT_IMAGE_WIDTH = 1920
_FAST_MAX_IMAGE_HEIGHT = 2160
_FAST_MAX_IMAGE_WIDTH = 3840
_ZERO_PAD_SELECTION = frozenset(
    {
        "mask_head.res_blocks.1.0.layers.5",
        "mask_head.res_blocks.2.0.layers.2",
        "mask_head.res_blocks.2.0.layers.5",
        "mask_head.res_blocks.3.0.layers.2",
        "mask_head.res_blocks.3.0.layers.5",
        "mask_head.resamplers.1.1",
        "mask_head.resamplers.2.1",
        "neck.res_blocks.2.0.layers.2",
        "neck.res_blocks.2.0.layers.5",
        "neck.res_blocks.2.1.layers.2",
        "neck.res_blocks.2.1.layers.5",
        "neck.res_blocks.3.0.layers.2",
        "neck.res_blocks.3.0.layers.5",
        "neck.res_blocks.3.1.layers.2",
        "neck.res_blocks.3.1.layers.5",
        "points_head.res_blocks.1.0.layers.2",
        "points_head.res_blocks.1.0.layers.5",
        "points_head.res_blocks.2.0.layers.2",
        "points_head.res_blocks.2.0.layers.5",
        "points_head.res_blocks.3.0.layers.2",
        "points_head.res_blocks.3.0.layers.5",
        "points_head.resamplers.0.1",
        "points_head.resamplers.1.1",
        "points_head.resamplers.2.1",
    }
)


def _fuse_half_pixel_x2_conv_weight(weight: np.ndarray) -> np.ndarray:
    """Compose HALF_PIXEL bilinear x2 followed by a 3x3 cross-correlation."""

    if weight.ndim != 4 or tuple(weight.shape[2:]) != (3, 3):
        raise ValueError(f"MoGe fused resample requires OI33 weights, got {weight.shape}")
    coefficients = np.asarray((0.25, 0.75, 0.75, 0.25), dtype=weight.dtype)
    fused = np.zeros((weight.shape[1], weight.shape[0], 6, 6), dtype=weight.dtype)
    transposed = weight.transpose(1, 0, 2, 3)
    for resize_y, coefficient_y in enumerate(coefficients):
        for resize_x, coefficient_x in enumerate(coefficients):
            coefficient = coefficient_y * coefficient_x
            for kernel_y in range(3):
                for kernel_x in range(3):
                    fused[
                        :,
                        :,
                        resize_y - kernel_y + 2,
                        resize_x - kernel_x + 2,
                    ] += coefficient * transposed[:, :, kernel_y, kernel_x]
    return np.ascontiguousarray(fused)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "MoGe engine builds require PyTorch to read the official checkpoint. "
            "Use the repository base build environment, which provides PyTorch."
        ) from exc
    return torch


class _NativeMogeGraph:
    """Small family-local vocabulary for composing the exact MoGe graph."""

    def __init__(
        self, trt: Any, network: Any, state: dict[str, Any], *, fast_path: bool = False
    ) -> None:
        self.trt = trt
        self.network = network
        self.state = state
        self.fast_path = fast_path
        # TensorRT may retain host weight views until serialization finishes.
        self._host_weights: list[np.ndarray] = []

    def _layer(self, layer: Any, kind: str, name: str) -> Any:
        if layer is None:
            raise RuntimeError(f"TensorRT rejected MoGe {kind} layer {name!r}")
        layer.name = name
        return layer

    def _array(self, name: str, expected: tuple[int, ...] | None = None) -> np.ndarray:
        if name not in self.state:
            raise KeyError(f"MoGe checkpoint is missing tensor {name!r}")
        value = self.state[name]
        array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        if expected is not None and tuple(array.shape) != expected:
            raise ValueError(
                f"MoGe tensor {name!r} has shape {tuple(array.shape)}, expected {expected}"
            )
        self._host_weights.append(array)
        return array

    def constant(self, values: Any, *, dtype=np.float32, name: str = "constant") -> Any:
        array = np.ascontiguousarray(values, dtype=dtype)
        self._host_weights.append(array)
        layer = self._layer(
            self.network.add_constant(array.shape, self.trt.Weights(array)),
            "constant",
            name,
        )
        return layer.get_output(0)

    def weight_constant(
        self,
        tensor_name: str,
        *,
        expected: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
        name: str,
    ) -> Any:
        array = self._array(tensor_name, expected)
        if shape is not None:
            array = np.ascontiguousarray(array.reshape(shape))
            self._host_weights.append(array)
        layer = self._layer(
            self.network.add_constant(array.shape, self.trt.Weights(array)),
            "weight constant",
            name,
        )
        return layer.get_output(0)

    def binary(self, left: Any, right: Any, operation: Any, name: str) -> Any:
        return self._layer(
            self.network.add_elementwise(left, right, operation), "elementwise", name
        ).get_output(0)

    def unary(self, tensor: Any, operation: Any, name: str) -> Any:
        return self._layer(self.network.add_unary(tensor, operation), "unary", name).get_output(0)

    def cast(self, tensor: Any, dtype: Any, name: str) -> Any:
        if tensor.dtype == dtype:
            return tensor
        return self._layer(self.network.add_cast(tensor, dtype), "cast", name).get_output(0)

    def shape(self, tensor: Any, name: str) -> Any:
        return self._layer(self.network.add_shape(tensor), "shape", name).get_output(0)

    def shape_index(self, shape: Any, index: int, name: str) -> Any:
        indices = self.constant([index], dtype=np.int64, name=f"{name}.index")
        return self._layer(
            self.network.add_gather(shape, indices, 0), "shape gather", name
        ).get_output(0)

    def shape_value(self, value: int, name: str) -> Any:
        return self.constant([value], dtype=np.int64, name=name)

    def shape_concat(self, parts: Sequence[Any], name: str) -> Any:
        layer = self._layer(self.network.add_concatenation(list(parts)), "shape concat", name)
        layer.axis = 0
        return layer.get_output(0)

    def reshape(
        self,
        tensor: Any,
        target: tuple[int, ...] | Any,
        name: str,
        *,
        first_transpose: tuple[int, ...] | None = None,
        second_transpose: tuple[int, ...] | None = None,
    ) -> Any:
        layer = self._layer(self.network.add_shuffle(tensor), "shuffle", name)
        if first_transpose is not None:
            layer.first_transpose = first_transpose
        if isinstance(target, tuple):
            layer.reshape_dims = target
        else:
            layer.set_input(1, target)
        if second_transpose is not None:
            layer.second_transpose = second_transpose
        return layer.get_output(0)

    def dynamic_slice(
        self,
        tensor: Any,
        start: tuple[int, ...],
        output_shape: Any,
        name: str,
        *,
        mode: Any | None = None,
    ) -> Any:
        rank = len(start)
        layer = self._layer(
            self.network.add_slice(tensor, start, (1,) * rank, (1,) * rank),
            "slice",
            name,
        )
        layer.set_input(2, output_shape)
        if mode is not None:
            layer.mode = mode
        return layer.get_output(0)

    def gather(self, tensor: Any, indices: Any, axis: int, name: str) -> Any:
        return self._layer(
            self.network.add_gather(tensor, indices, axis), "gather", name
        ).get_output(0)

    def nearest_sample_indices(self, size: Any, name: str) -> Any:
        positions = self.constant(
            np.arange(_FOCAL_RECOVERY_SIZE, dtype=np.int64),
            dtype=np.int64,
            name=f"{name}.positions",
        )
        scaled = self.binary(positions, size, self.trt.ElementWiseOperation.PROD, f"{name}.scaled")
        divisor = self.shape_value(_FOCAL_RECOVERY_SIZE, f"{name}.divisor")
        return self.binary(
            scaled,
            divisor,
            self.trt.ElementWiseOperation.FLOOR_DIV,
            f"{name}.indices",
        )

    def is_finite(self, tensor: Any, name: str) -> Any:
        absolute = self.unary(tensor, self.trt.UnaryOperation.ABS, f"{name}.abs")
        if tensor.dtype == self.trt.float16:
            constant_dtype = np.float16
        elif tensor.dtype == self.trt.float32:
            constant_dtype = np.float32
        else:
            raise ValueError(f"MoGe finite check {name!r} requires FP16 or FP32 input")
        infinity = self.constant(
            np.full((1,) * len(tuple(tensor.shape)), np.inf, dtype=constant_dtype),
            dtype=constant_dtype,
            name=f"{name}.infinity",
        )
        return self.binary(absolute, infinity, self.trt.ElementWiseOperation.LESS, name)

    def resize(
        self,
        tensor: Any,
        output_shape: Any,
        interpolation: Any,
        name: str,
    ) -> Any:
        layer = self._layer(self.network.add_resize(tensor), "resize", name)
        layer.resize_mode = interpolation
        layer.coordinate_transformation = self.trt.ResizeCoordinateTransformation.HALF_PIXEL
        layer.selector_for_single_pixel = self.trt.ResizeSelector.FORMULA
        layer.exclude_outside = False
        layer.set_input(1, output_shape)
        if interpolation == self.trt.InterpolationMode.CUBIC:
            layer.cubic_coeff = -0.75
        return layer.get_output(0)

    def resize_nchw_to_hw(
        self, tensor: Any, output_h: Any, output_w: Any, interpolation: Any, name: str
    ) -> Any:
        input_shape = self.shape(tensor, f"{name}.input_shape")
        batch = self.shape_index(input_shape, 0, f"{name}.batch")
        channels = self.shape_index(input_shape, 1, f"{name}.channels")
        target = self.shape_concat([batch, channels, output_h, output_w], f"{name}.target_shape")
        return self.resize(tensor, target, interpolation, name)

    def replicate_pad(self, tensor: Any, padding: int, name: str) -> Any:
        if padding == 0:
            return tensor
        input_shape = self.shape(tensor, f"{name}.input_shape")
        batch = self.shape_index(input_shape, 0, f"{name}.batch")
        channels = self.shape_index(input_shape, 1, f"{name}.channels")
        height = self.shape_index(input_shape, 2, f"{name}.height")
        width = self.shape_index(input_shape, 3, f"{name}.width")
        twice = self.shape_value(2 * padding, f"{name}.twice_padding")
        output_h = self.binary(height, twice, self.trt.ElementWiseOperation.SUM, f"{name}.out_h")
        output_w = self.binary(width, twice, self.trt.ElementWiseOperation.SUM, f"{name}.out_w")
        output_shape = self.shape_concat(
            [batch, channels, output_h, output_w], f"{name}.output_shape"
        )
        return self.dynamic_slice(
            tensor,
            (0, 0, -padding, -padding),
            output_shape,
            name,
            mode=self.trt.SampleMode.CLAMP,
        )

    def convolution(
        self,
        tensor: Any,
        module: str,
        name: str,
        *,
        stride: int = 1,
        replicate_padding: int = 0,
        compute_dtype: Any | None = None,
    ) -> Any:
        weight = self._array(f"{module}.weight")
        if weight.ndim != 4:
            raise ValueError(f"MoGe convolution {module!r} does not have a 4D kernel")
        bias = self._array(f"{module}.bias", (weight.shape[0],))
        compute_dtype = compute_dtype or self.trt.float32
        if compute_dtype == self.trt.float16:
            weight = np.ascontiguousarray(weight, dtype=np.float16)
            bias = np.ascontiguousarray(bias, dtype=np.float16)
            self._host_weights.extend((weight, bias))
        elif compute_dtype != self.trt.float32:
            raise ValueError(f"Unsupported MoGe convolution compute dtype: {compute_dtype}")
        tensor = self.cast(tensor, compute_dtype, f"{name}.input_cast")
        zero_pad = module in _ZERO_PAD_SELECTION
        if zero_pad and (
            replicate_padding != 1
            or stride != 1
            or tuple(int(value) for value in weight.shape[2:]) != (3, 3)
        ):
            raise ValueError(
                f"MoGe selected zero-pad convolution {module!r} must be stride-1 3x3 "
                "with one pixel of source replicate padding"
            )
        padded = (
            tensor if zero_pad else self.replicate_pad(tensor, replicate_padding, f"{name}.pad")
        )
        layer = self._layer(
            self.network.add_convolution_nd(
                padded,
                int(weight.shape[0]),
                tuple(int(value) for value in weight.shape[2:]),
                self.trt.Weights(weight),
                self.trt.Weights(bias),
            ),
            "convolution",
            name,
        )
        layer.stride_nd = (stride, stride)
        layer.padding_nd = (1, 1) if zero_pad else (0, 0)
        return layer.get_output(0)

    def deconvolution(
        self,
        tensor: Any,
        module: str,
        name: str,
        *,
        compute_dtype: Any | None = None,
    ) -> Any:
        weight = self._array(f"{module}.weight")
        if weight.ndim != 4:
            raise ValueError(f"MoGe deconvolution {module!r} does not have a 4D kernel")
        output_channels = int(weight.shape[1])
        bias = self._array(f"{module}.bias", (output_channels,))
        compute_dtype = compute_dtype or self.trt.float32
        if compute_dtype == self.trt.float16:
            weight = np.ascontiguousarray(weight, dtype=np.float16)
            bias = np.ascontiguousarray(bias, dtype=np.float16)
            self._host_weights.extend((weight, bias))
        elif compute_dtype != self.trt.float32:
            raise ValueError(f"Unsupported MoGe deconvolution compute dtype: {compute_dtype}")
        tensor = self.cast(tensor, compute_dtype, f"{name}.input_cast")
        layer = self._layer(
            self.network.add_deconvolution_nd(
                tensor,
                output_channels,
                tuple(int(value) for value in weight.shape[2:]),
                self.trt.Weights(weight),
                self.trt.Weights(bias),
            ),
            "deconvolution",
            name,
        )
        layer.stride_nd = tuple(int(value) for value in weight.shape[2:])
        layer.padding_nd = (0, 0)
        return layer.get_output(0)

    def fused_half_pixel_resample(
        self,
        tensor: Any,
        module: str,
        name: str,
        *,
        compute_dtype: Any,
    ) -> Any:
        """Fuse bilinear x2, replicate padding and a 3x3 convolution exactly."""

        weight = self._array(f"{module}.weight")
        bias = self._array(f"{module}.bias", (int(weight.shape[0]),))
        if compute_dtype == self.trt.float16:
            # Match the source convolution's effective checkpoint precision
            # before composing its weights with the exact bilinear kernel.
            weight = np.ascontiguousarray(weight, dtype=np.float16)
            bias = np.ascontiguousarray(bias, dtype=np.float16)
            fused_weight = np.ascontiguousarray(
                _fuse_half_pixel_x2_conv_weight(weight.astype(np.float32)),
                dtype=np.float16,
            )
        elif compute_dtype == self.trt.float32:
            fused_weight = _fuse_half_pixel_x2_conv_weight(weight)
        else:
            raise ValueError(f"Unsupported MoGe fused resample dtype: {compute_dtype}")
        self._host_weights.extend((weight, bias, fused_weight))

        tensor = self.cast(tensor, compute_dtype, f"{name}.input_cast")
        # Replicating one low-resolution pixel supplies the two high-resolution
        # border samples consumed by the original post-resize replicate pad.
        tensor = self.replicate_pad(tensor, 1, f"{name}.input_pad")
        layer = self._layer(
            self.network.add_deconvolution_nd(
                tensor,
                int(weight.shape[0]),
                (6, 6),
                self.trt.Weights(fused_weight),
                self.trt.Weights(bias),
            ),
            "fused resample deconvolution",
            name,
        )
        layer.stride_nd = (2, 2)
        # For padded input H+2, (H+2-1)*2 + 6 - 2*4 == 2H.
        layer.padding_nd = (4, 4)
        return layer.get_output(0)

    def linear(
        self,
        tensor: Any,
        module: str,
        name: str,
        *,
        compute_dtype: Any | None = None,
        output_dtype: Any | None = None,
    ) -> Any:
        weight = self._array(f"{module}.weight")
        if weight.ndim != 2:
            raise ValueError(f"MoGe linear {module!r} does not have a 2D weight")
        output_width, input_width = (int(value) for value in weight.shape)
        bias = self._array(f"{module}.bias", (output_width,))
        compute_dtype = compute_dtype or self.trt.float32
        if compute_dtype == self.trt.float16:
            constant_dtype = np.float16
            weight = np.ascontiguousarray(weight, dtype=np.float16)
            bias = np.ascontiguousarray(bias, dtype=np.float16)
            self._host_weights.extend((weight, bias))
        elif compute_dtype != self.trt.float32:
            raise ValueError(f"Unsupported MoGe linear compute dtype: {compute_dtype}")
        else:
            constant_dtype = np.float32
        tensor = self.cast(tensor, compute_dtype, f"{name}.input_cast")
        rank = len(tuple(tensor.shape))
        restore_shape = None
        if rank > 2:
            input_shape = self.shape(tensor, f"{name}.input_shape")
            leading = [
                self.shape_index(input_shape, index, f"{name}.output_dim_{index}")
                for index in range(rank - 1)
            ]
            restore_shape = self.shape_concat(
                [*leading, self.shape_value(output_width, f"{name}.output_width")],
                f"{name}.output_shape",
            )
            tensor = self.reshape(tensor, (-1, input_width), f"{name}.input_rows")
        rhs = self.constant(weight, dtype=constant_dtype, name=f"{name}.weight")
        product = self._layer(
            self.network.add_matrix_multiply(
                tensor,
                self.trt.MatrixOperation.NONE,
                rhs,
                self.trt.MatrixOperation.TRANSPOSE,
            ),
            "matrix multiply",
            f"{name}.matmul",
        ).get_output(0)
        bias_tensor = self.constant(
            bias.reshape(1, output_width), dtype=constant_dtype, name=f"{name}.bias"
        )
        result = self.binary(
            product, bias_tensor, self.trt.ElementWiseOperation.SUM, f"{name}.bias_add"
        )
        if restore_shape is not None:
            result = self.reshape(result, restore_shape, f"{name}.restore")
        if output_dtype is not None:
            result = self.cast(result, output_dtype, f"{name}.output_cast")
        return result

    def layer_norm(self, tensor: Any, module: str, name: str) -> Any:
        if self.fast_path:
            tensor = self.cast(tensor, self.trt.float32, f"{name}.input_fp32")
        rank = len(tuple(tensor.shape))
        width = int(self.state[f"{module}.weight"].numel())
        parameter_shape = (1,) * (rank - 1) + (width,)
        scale = self.weight_constant(
            f"{module}.weight",
            expected=(width,),
            shape=parameter_shape,
            name=f"{name}.scale",
        )
        bias = self.weight_constant(
            f"{module}.bias",
            expected=(width,),
            shape=parameter_shape,
            name=f"{name}.bias",
        )
        layer = self._layer(
            self.network.add_normalization_v2(tensor, scale, bias, 1 << (rank - 1)),
            "normalization",
            name,
        )
        layer.epsilon = 1.0e-6
        return layer.get_output(0)

    def relu(self, tensor: Any, name: str) -> Any:
        return self._layer(
            self.network.add_activation(tensor, self.trt.ActivationType.RELU), "ReLU", name
        ).get_output(0)

    def sigmoid(self, tensor: Any, name: str) -> Any:
        return self._layer(
            self.network.add_activation(tensor, self.trt.ActivationType.SIGMOID),
            "sigmoid",
            name,
        ).get_output(0)

    def gelu(self, tensor: Any, name: str) -> Any:
        return self._layer(
            self.network.add_activation(tensor, self.trt.ActivationType.GELU_ERF),
            "GELU_ERF",
            name,
        ).get_output(0)

    def _linspace(self, size: Any, size_f: Any, span: Any, name: str) -> Any:
        one_i = self.shape_value(1, f"{name}.one_i")
        two_f = self.constant([2.0], name=f"{name}.two_f")
        size_minus_one = self.binary(
            size, one_i, self.trt.ElementWiseOperation.SUB, f"{name}.size_minus_one"
        )
        size_minus_one_f = self.cast(size_minus_one, self.trt.float32, f"{name}.size_minus_one_f")
        start = self.binary(
            self.binary(
                span,
                size_minus_one_f,
                self.trt.ElementWiseOperation.PROD,
                f"{name}.start_numerator",
            ),
            size_f,
            self.trt.ElementWiseOperation.DIV,
            f"{name}.start_abs",
        )
        start = self.unary(start, self.trt.UnaryOperation.NEG, f"{name}.start")
        step = self.binary(
            self.binary(two_f, span, self.trt.ElementWiseOperation.PROD, f"{name}.step_num"),
            size_f,
            self.trt.ElementWiseOperation.DIV,
            f"{name}.step",
        )
        start_scalar = self.reshape(start, (), f"{name}.start_scalar")
        fill = self._layer(
            self.network.add_fill((1,), self.trt.FillOperation.LINSPACE, self.trt.float32),
            "linspace fill",
            name,
        )
        fill.set_input(0, size)
        fill.set_input(1, start_scalar)
        fill.set_input(2, step)
        return fill.get_output(0)

    def uv(self, height: Any, width: Any, aspect: Any, name: str) -> Any:
        one_f = self.constant([1.0], name=f"{name}.one")
        aspect_sq = self.binary(
            aspect, aspect, self.trt.ElementWiseOperation.PROD, f"{name}.aspect_sq"
        )
        diagonal = self.unary(
            self.binary(one_f, aspect_sq, self.trt.ElementWiseOperation.SUM, f"{name}.diag_sq"),
            self.trt.UnaryOperation.SQRT,
            f"{name}.diagonal",
        )
        span_x = self.binary(aspect, diagonal, self.trt.ElementWiseOperation.DIV, f"{name}.span_x")
        span_y = self.binary(one_f, diagonal, self.trt.ElementWiseOperation.DIV, f"{name}.span_y")
        height_f = self.cast(height, self.trt.float32, f"{name}.height_f")
        width_f = self.cast(width, self.trt.float32, f"{name}.width_f")
        u = self._linspace(width, width_f, span_x, f"{name}.u")
        v = self._linspace(height, height_f, span_y, f"{name}.v")
        u_shape = self.shape_concat(
            [
                self.shape_value(1, f"{name}.u_leading"),
                self.shape_value(1, f"{name}.u_channel"),
                self.shape_value(1, f"{name}.u_row"),
                width,
            ],
            f"{name}.u_shape",
        )
        v_shape = self.shape_concat(
            [
                self.shape_value(1, f"{name}.v_leading"),
                self.shape_value(1, f"{name}.v_channel"),
                height,
                self.shape_value(1, f"{name}.v_column"),
            ],
            f"{name}.v_shape",
        )
        u4 = self.reshape(u, u_shape, f"{name}.u4")
        v4 = self.reshape(v, v_shape, f"{name}.v4")
        zero = self.constant([[[[0.0]]]], name=f"{name}.zero")
        u_grid = self.binary(
            u4,
            self.binary(v4, zero, self.trt.ElementWiseOperation.PROD, f"{name}.v_zero"),
            self.trt.ElementWiseOperation.SUM,
            f"{name}.u_grid",
        )
        v_grid = self.binary(
            v4,
            self.binary(u4, zero, self.trt.ElementWiseOperation.PROD, f"{name}.u_zero"),
            self.trt.ElementWiseOperation.SUM,
            f"{name}.v_grid",
        )
        layer = self._layer(self.network.add_concatenation([u_grid, v_grid]), "UV concat", name)
        layer.axis = 1
        return layer.get_output(0)

    def attention(self, hidden: Any, layer_index: int, total_tokens: Any) -> Any:
        prefix = f"encoder.backbone.blocks.{layer_index}.attn"
        name = f"vit.block.{layer_index}.attention"
        if self.fast_path:
            qkv = self.linear(
                hidden,
                f"{prefix}.qkv",
                f"{name}.qkv",
                compute_dtype=self.trt.float16,
                output_dtype=self.trt.float32,
            )
        else:
            qkv = self.linear(hidden, f"{prefix}.qkv", f"{name}.qkv")
        component_shape = self.shape_concat(
            [
                self.shape_value(1, f"{name}.batch"),
                total_tokens,
                self.shape_value(_HIDDEN, f"{name}.hidden"),
            ],
            f"{name}.component_shape",
        )
        components = [
            self.dynamic_slice(
                qkv,
                (0, 0, component * _HIDDEN),
                component_shape,
                f"{name}.{label}_slice",
            )
            for component, label in enumerate(("q", "k", "v"))
        ]
        heads_shape = self.shape_concat(
            [
                self.shape_value(1, f"{name}.heads_batch"),
                total_tokens,
                self.shape_value(_HEADS, f"{name}.heads"),
                self.shape_value(_HEAD_DIM, f"{name}.head_dim"),
            ],
            f"{name}.heads_shape",
        )
        q, k, v = [
            self.reshape(
                component,
                heads_shape,
                f"{name}.{label}_heads",
                second_transpose=(0, 2, 1, 3),
            )
            for component, label in zip(components, ("q", "k", "v"), strict=True)
        ]
        scale = self.constant([[[[0.125]]]], name=f"{name}.scale")
        q = self.binary(q, scale, self.trt.ElementWiseOperation.PROD, f"{name}.q_scaled")
        if self.fast_path:
            q = self.cast(q, self.trt.float16, f"{name}.q_fp16")
            k = self.cast(k, self.trt.float16, f"{name}.k_fp16")
            v = self.cast(v, self.trt.float16, f"{name}.v_fp16")
        add_attention_v2 = getattr(self.network, "add_attention_v2", None)
        if callable(add_attention_v2):
            layer = add_attention_v2(
                q,
                k,
                v,
                self.trt.AttentionNormalizationOp.SOFTMAX,
                self.trt.CausalMaskKind.NONE,
            )
        else:
            layer = self.network.add_attention(
                q, k, v, self.trt.AttentionNormalizationOp.SOFTMAX, False
            )
        attention = self._layer(layer, "IAttention", name)
        attention.decomposable = not self.fast_path
        if hasattr(attention, "query_form"):
            attention.query_form = self.trt.AttentionIOForm.PADDED_BHND
            attention.key_value_form = self.trt.AttentionIOForm.PADDED_BHND
        context_shape = self.shape_concat(
            [
                self.shape_value(1, f"{name}.context_batch"),
                total_tokens,
                self.shape_value(_HIDDEN, f"{name}.context_hidden"),
            ],
            f"{name}.context_shape",
        )
        context = self.reshape(
            attention.get_output(0),
            context_shape,
            f"{name}.context",
            first_transpose=(0, 2, 1, 3),
        )
        if self.fast_path:
            return self.linear(
                context,
                f"{prefix}.proj",
                f"{name}.projection",
                compute_dtype=self.trt.float16,
                output_dtype=self.trt.float16,
            )
        return self.linear(context, f"{prefix}.proj", f"{name}.projection")

    def transformer_block(self, hidden: Any, index: int, total_tokens: Any) -> Any:
        prefix = f"encoder.backbone.blocks.{index}"
        name = f"vit.block.{index}"
        normalized = self.layer_norm(hidden, f"{prefix}.norm1", f"{name}.norm1")
        attention = self.attention(normalized, index, total_tokens)
        gamma1 = self.weight_constant(
            f"{prefix}.ls1.gamma",
            expected=(_HIDDEN,),
            shape=(1, 1, _HIDDEN),
            name=f"{name}.ls1",
        )
        if self.fast_path:
            gamma1 = self.cast(gamma1, self.trt.float16, f"{name}.ls1_fp16")
        attention = self.binary(
            attention, gamma1, self.trt.ElementWiseOperation.PROD, f"{name}.scaled_attention"
        )
        hidden = self.binary(
            hidden, attention, self.trt.ElementWiseOperation.SUM, f"{name}.attention_residual"
        )
        normalized = self.layer_norm(hidden, f"{prefix}.norm2", f"{name}.norm2")
        if self.fast_path:
            mlp = self.linear(
                normalized,
                f"{prefix}.mlp.fc1",
                f"{name}.mlp.fc1",
                compute_dtype=self.trt.float16,
            )
        else:
            mlp = self.linear(normalized, f"{prefix}.mlp.fc1", f"{name}.mlp.fc1")
        mlp = self.gelu(mlp, f"{name}.mlp.gelu")
        if self.fast_path:
            mlp = self.linear(
                mlp,
                f"{prefix}.mlp.fc2",
                f"{name}.mlp.fc2",
                compute_dtype=self.trt.float16,
                output_dtype=self.trt.float16,
            )
        else:
            mlp = self.linear(mlp, f"{prefix}.mlp.fc2", f"{name}.mlp.fc2")
        gamma2 = self.weight_constant(
            f"{prefix}.ls2.gamma",
            expected=(_HIDDEN,),
            shape=(1, 1, _HIDDEN),
            name=f"{name}.ls2",
        )
        if self.fast_path:
            gamma2 = self.cast(gamma2, self.trt.float16, f"{name}.ls2_fp16")
        mlp = self.binary(mlp, gamma2, self.trt.ElementWiseOperation.PROD, f"{name}.scaled_mlp")
        return self.binary(hidden, mlp, self.trt.ElementWiseOperation.SUM, f"{name}.mlp_residual")

    def projected_intermediate(
        self, hidden: Any, projection_index: int, patch_tokens: Any, base_h: Any, base_w: Any
    ) -> tuple[Any, Any]:
        name = f"vit.intermediate.{projection_index}"
        normalized = self.layer_norm(hidden, "encoder.backbone.norm", f"{name}.norm")
        class_shape = self.shape_concat(
            [
                self.shape_value(1, f"{name}.class_batch"),
                self.shape_value(1, f"{name}.class_rows"),
                self.shape_value(_HIDDEN, f"{name}.class_width"),
            ],
            f"{name}.class_shape",
        )
        class_token = self.dynamic_slice(normalized, (0, 0, 0), class_shape, f"{name}.class")
        patch_shape = self.shape_concat(
            [
                self.shape_value(1, f"{name}.patch_batch"),
                patch_tokens,
                self.shape_value(_HIDDEN, f"{name}.patch_width"),
            ],
            f"{name}.patch_shape",
        )
        patches = self.dynamic_slice(normalized, (0, 1, 0), patch_shape, f"{name}.patches")
        image_shape = self.shape_concat(
            [
                self.shape_value(1, f"{name}.image_batch"),
                base_h,
                base_w,
                self.shape_value(_HIDDEN, f"{name}.image_width"),
            ],
            f"{name}.image_shape",
        )
        image = self.reshape(
            patches,
            image_shape,
            f"{name}.image",
            second_transpose=(0, 3, 1, 2),
        )
        projected = self.convolution(
            image,
            f"encoder.output_projections.{projection_index}",
            f"{name}.projection",
            compute_dtype=self.trt.float16 if self.fast_path else self.trt.float32,
        )
        return projected, class_token

    def encoder(self, image: Any) -> tuple[Any, Any, Any, Any, Any]:
        input_shape = self.shape(image, "input.shape")
        input_h = self.shape_index(input_shape, 2, "input.height")
        input_w = self.shape_index(input_shape, 3, "input.width")
        input_h_f = self.cast(input_h, self.trt.float32, "input.height_f")
        input_w_f = self.cast(input_w, self.trt.float32, "input.width_f")
        aspect = self.binary(
            input_w_f, input_h_f, self.trt.ElementWiseOperation.DIV, "input.aspect"
        )
        token_value = self.constant([float(_NUM_TOKENS)], name="input.num_tokens")
        base_h = self.cast(
            self.unary(
                self.unary(
                    self.binary(
                        token_value,
                        aspect,
                        self.trt.ElementWiseOperation.DIV,
                        "input.base_h_ratio",
                    ),
                    self.trt.UnaryOperation.SQRT,
                    "input.base_h_sqrt",
                ),
                self.trt.UnaryOperation.ROUND,
                "input.base_h_round",
            ),
            self.trt.int64,
            "input.base_h",
        )
        base_w = self.cast(
            self.unary(
                self.unary(
                    self.binary(
                        token_value,
                        aspect,
                        self.trt.ElementWiseOperation.PROD,
                        "input.base_w_ratio",
                    ),
                    self.trt.UnaryOperation.SQRT,
                    "input.base_w_sqrt",
                ),
                self.trt.UnaryOperation.ROUND,
                "input.base_w_round",
            ),
            self.trt.int64,
            "input.base_w",
        )
        fourteen = self.shape_value(_PATCH, "input.patch_size")
        resized_h = self.binary(
            base_h, fourteen, self.trt.ElementWiseOperation.PROD, "input.resized_h"
        )
        resized_w = self.binary(
            base_w, fourteen, self.trt.ElementWiseOperation.PROD, "input.resized_w"
        )
        resized_shape = self.shape_concat(
            [
                self.shape_value(1, "input.resize_batch"),
                self.shape_value(3, "input.resize_channels"),
                resized_h,
                resized_w,
            ],
            "input.resize_shape",
        )
        pixels = self.resize(
            image, resized_shape, self.trt.InterpolationMode.LINEAR, "input.resize"
        )
        mean = self.weight_constant("encoder.image_mean", expected=(1, 3, 1, 1), name="input.mean")
        std = self.weight_constant("encoder.image_std", expected=(1, 3, 1, 1), name="input.std")
        pixels = self.binary(pixels, mean, self.trt.ElementWiseOperation.SUB, "input.center")
        pixels = self.binary(pixels, std, self.trt.ElementWiseOperation.DIV, "input.normalize")
        patches = self.convolution(
            pixels,
            "encoder.backbone.patch_embed.proj",
            "vit.patch_embed",
            stride=_PATCH,
            compute_dtype=self.trt.float16 if self.fast_path else self.trt.float32,
        )
        patch_tokens = self.binary(
            base_h, base_w, self.trt.ElementWiseOperation.PROD, "vit.patch_tokens"
        )
        total_tokens = self.binary(
            patch_tokens,
            self.shape_value(1, "vit.class_count"),
            self.trt.ElementWiseOperation.SUM,
            "vit.total_tokens",
        )
        patch_sequence_shape = self.shape_concat(
            [
                self.shape_value(1, "vit.patch_batch"),
                patch_tokens,
                self.shape_value(_HIDDEN, "vit.patch_hidden"),
            ],
            "vit.patch_sequence_shape",
        )
        hidden = self.reshape(
            patches,
            patch_sequence_shape,
            "vit.patch_sequence",
            first_transpose=(0, 2, 3, 1),
        )
        class_token = self.weight_constant(
            "encoder.backbone.cls_token",
            expected=(1, 1, _HIDDEN),
            name="vit.class_token",
        )
        if self.fast_path:
            class_token = self.cast(class_token, self.trt.float16, "vit.class_token_fp16")
        token_concat = self._layer(
            self.network.add_concatenation([class_token, hidden]), "token concat", "vit.tokens"
        )
        token_concat.axis = 1
        hidden = token_concat.get_output(0)

        position = self._array("encoder.backbone.pos_embed", (1, 1 + _POSITION_GRID**2, _HIDDEN))
        class_position = self.constant(position[:, :1, :], name="vit.position.class")
        patch_position = np.ascontiguousarray(
            position[:, 1:, :]
            .reshape(1, _POSITION_GRID, _POSITION_GRID, _HIDDEN)
            .transpose(0, 3, 1, 2)
        )
        position_map = self.constant(patch_position, name="vit.position.patch_map")
        position_shape = self.shape_concat(
            [
                self.shape_value(1, "vit.position.batch"),
                self.shape_value(_HIDDEN, "vit.position.channels"),
                base_h,
                base_w,
            ],
            "vit.position.resize_shape",
        )
        position_map = self.resize(
            position_map,
            position_shape,
            self.trt.InterpolationMode.CUBIC,
            "vit.position.resize",
        )
        position_sequence = self.reshape(
            position_map,
            patch_sequence_shape,
            "vit.position.sequence",
            first_transpose=(0, 2, 3, 1),
        )
        position_concat = self._layer(
            self.network.add_concatenation([class_position, position_sequence]),
            "position concat",
            "vit.position.tokens",
        )
        position_concat.axis = 1
        position_tokens = position_concat.get_output(0)
        if self.fast_path:
            position_tokens = self.cast(position_tokens, self.trt.float16, "vit.position_fp16")
        hidden = self.binary(
            hidden,
            position_tokens,
            self.trt.ElementWiseOperation.SUM,
            "vit.tokens_plus_position",
        )
        if self.fast_path:
            hidden = self.cast(hidden, self.trt.float16, "vit.residual_fp16")

        captured: list[Any] = []
        last_class = None
        for index in range(24):
            hidden = self.transformer_block(hidden, index, total_tokens)
            if index in _INTERMEDIATE_LAYERS:
                projected, last_class = self.projected_intermediate(
                    hidden, len(captured), patch_tokens, base_h, base_w
                )
                captured.append(projected)
        if len(captured) != 4 or last_class is None:
            raise RuntimeError("MoGe failed to capture all DINOv2 intermediate layers")
        encoded = captured[0]
        for index, value in enumerate(captured[1:], start=1):
            encoded = self.binary(
                encoded,
                value,
                self.trt.ElementWiseOperation.SUM,
                f"vit.intermediate_sum.{index}",
            )
        class_vector = self.reshape(last_class, (1, _HIDDEN), "vit.class_vector")
        return encoded, class_vector, base_h, base_w, aspect

    def residual_conv_block(
        self, tensor: Any, module: str, name: str, *, compute_dtype: Any
    ) -> Any:
        hidden = self.relu(tensor, f"{name}.relu1")
        hidden = self.convolution(
            hidden,
            f"{module}.layers.2",
            f"{name}.conv1",
            replicate_padding=1,
            compute_dtype=compute_dtype,
        )
        hidden = self.relu(hidden, f"{name}.relu2")
        hidden = self.convolution(
            hidden,
            f"{module}.layers.5",
            f"{name}.conv2",
            replicate_padding=1,
            compute_dtype=compute_dtype,
        )
        return self.binary(hidden, tensor, self.trt.ElementWiseOperation.SUM, f"{name}.residual")

    def resample(
        self, tensor: Any, module: str, level: int, name: str, *, compute_dtype: Any
    ) -> Any:
        if level < 3:
            tensor = self.deconvolution(
                tensor,
                f"{module}.0",
                f"{name}.deconvolution",
                compute_dtype=compute_dtype,
            )
        elif level == 3 and self.fast_path:
            return self.fused_half_pixel_resample(
                tensor,
                f"{module}.1",
                f"{name}.fused_deconvolution",
                compute_dtype=compute_dtype,
            )
        else:
            shape = self.shape(tensor, f"{name}.input_shape")
            height = self.shape_index(shape, 2, f"{name}.height")
            width = self.shape_index(shape, 3, f"{name}.width")
            two = self.shape_value(2, f"{name}.two")
            output_h = self.binary(height, two, self.trt.ElementWiseOperation.PROD, f"{name}.out_h")
            output_w = self.binary(width, two, self.trt.ElementWiseOperation.PROD, f"{name}.out_w")
            tensor = self.resize_nchw_to_hw(
                tensor, output_h, output_w, self.trt.InterpolationMode.LINEAR, f"{name}.resize"
            )
        return self.convolution(
            tensor,
            f"{module}.1",
            f"{name}.convolution",
            replicate_padding=1,
            compute_dtype=compute_dtype,
        )

    def conv_stack(
        self,
        inputs: list[Any],
        prefix: str,
        num_res_blocks: tuple[int, ...],
        *,
        final_projection: bool,
        compute_dtype: Any,
    ) -> list[Any]:
        outputs: list[Any] = []
        current = None
        for level, feature in enumerate(inputs):
            projected = self.convolution(
                feature,
                f"{prefix}.input_blocks.{level}",
                f"{prefix}.level.{level}.input",
                compute_dtype=compute_dtype,
            )
            current = (
                projected
                if current is None
                else self.binary(
                    current,
                    projected,
                    self.trt.ElementWiseOperation.SUM,
                    f"{prefix}.level.{level}.fusion",
                )
            )
            for block in range(num_res_blocks[level]):
                current = self.residual_conv_block(
                    current,
                    f"{prefix}.res_blocks.{level}.{block}",
                    f"{prefix}.level.{level}.block.{block}",
                    compute_dtype=compute_dtype,
                )
            output = current
            if final_projection and level == len(inputs) - 1:
                output = self.convolution(
                    current,
                    f"{prefix}.output_blocks.{level}",
                    f"{prefix}.level.{level}.output",
                    compute_dtype=compute_dtype,
                )
            outputs.append(output)
            if level < len(inputs) - 1:
                current = self.resample(
                    current,
                    f"{prefix}.resamplers.{level}",
                    level,
                    f"{prefix}.level.{level}.resample",
                    compute_dtype=compute_dtype,
                )
        return outputs

    def outputs(self, image: Any) -> tuple[Any, Any, Any]:
        encoded, class_vector, base_h, base_w, aspect = self.encoder(image)
        features: list[Any] = [encoded]
        for level in range(5):
            multiplier = self.shape_value(2**level, f"uv.level.{level}.multiplier")
            height = self.binary(
                base_h, multiplier, self.trt.ElementWiseOperation.PROD, f"uv.level.{level}.height"
            )
            width = self.binary(
                base_w, multiplier, self.trt.ElementWiseOperation.PROD, f"uv.level.{level}.width"
            )
            uv = self.uv(height, width, aspect, f"uv.level.{level}")
            if self.fast_path:
                uv = self.cast(uv, self.trt.float16, f"uv.level.{level}.fp16")
            if level == 0:
                concat = self._layer(
                    self.network.add_concatenation([features[0], uv]),
                    "encoder/UV concat",
                    "neck.encoder_plus_uv",
                )
                concat.axis = 1
                features[0] = concat.get_output(0)
            else:
                features.append(uv)

        decoder_dtype = self.trt.float16 if self.fast_path else self.trt.float32
        neck = self.conv_stack(
            features,
            "neck",
            (0, 2, 2, 2, 0),
            final_projection=False,
            compute_dtype=decoder_dtype,
        )
        points = self.conv_stack(
            neck,
            "points_head",
            (0, 1, 1, 1, 0),
            final_projection=True,
            compute_dtype=decoder_dtype,
        )[-1]
        mask = self.conv_stack(
            neck,
            "mask_head",
            (0, 1, 1, 1, 0),
            final_projection=True,
            compute_dtype=decoder_dtype,
        )[-1]
        scale = self.linear(class_vector, "scale_head.0", "scale_head.0")
        scale = self.relu(scale, "scale_head.1")
        scale = self.linear(scale, "scale_head.2", "scale_head.2")
        scale = self.relu(scale, "scale_head.3")
        scale = self.linear(scale, "scale_head.4", "scale_head.4")

        input_shape = self.shape(image, "output.input_shape")
        input_h = self.shape_index(input_shape, 2, "output.height")
        input_w = self.shape_index(input_shape, 3, "output.width")
        raw_points = self.resize_nchw_to_hw(
            points, input_h, input_w, self.trt.InterpolationMode.LINEAR, "output.points_resize"
        )
        mask = self.resize_nchw_to_hw(
            mask, input_h, input_w, self.trt.InterpolationMode.LINEAR, "output.mask_resize"
        )
        xy_shape = self.shape_concat(
            [
                self.shape_value(1, "output.xy_batch"),
                self.shape_value(2, "output.xy_channels"),
                input_h,
                input_w,
            ],
            "output.xy_shape",
        )
        z_shape = self.shape_concat(
            [
                self.shape_value(1, "output.z_batch"),
                self.shape_value(1, "output.z_channels"),
                input_h,
                input_w,
            ],
            "output.z_shape",
        )
        raw_xy = self.dynamic_slice(raw_points, (0, 0, 0, 0), xy_shape, "output.raw_xy")
        raw_z = self.dynamic_slice(raw_points, (0, 2, 0, 0), z_shape, "output.raw_z")
        z = self.unary(raw_z, self.trt.UnaryOperation.EXP, "output.z_exp")
        xy = self.binary(raw_xy, z, self.trt.ElementWiseOperation.PROD, "output.xy_scaled")

        mask_shape = self.shape_concat(
            [self.shape_value(1, "output.mask_batch"), input_h, input_w],
            "output.mask_shape",
        )
        affine_depth = self.reshape(z, mask_shape, "output.affine_depth_squeeze")
        affine_depth = self.cast(affine_depth, self.trt.float32, "output.affine_depth_fp32")

        row_indices = self.nearest_sample_indices(input_h, "output.focal_rows")
        column_indices = self.nearest_sample_indices(input_w, "output.focal_columns")
        sampled_xy = self.gather(xy, row_indices, 2, "output.focal_xy_rows")
        sampled_xy = self.gather(sampled_xy, column_indices, 3, "output.focal_xy_columns")
        sampled_z = self.gather(z, row_indices, 2, "output.focal_z_rows")
        sampled_z = self.gather(sampled_z, column_indices, 3, "output.focal_z_columns")
        sampled_concat = self._layer(
            self.network.add_concatenation([sampled_xy, sampled_z]),
            "sampled point concat",
            "output.focal_samples_nchw",
        )
        sampled_concat.axis = 1
        focal_samples = self.reshape(
            sampled_concat.get_output(0),
            (1, _FOCAL_RECOVERY_SIZE, _FOCAL_RECOVERY_SIZE, 3),
            "output.focal_samples_nhwc",
            first_transpose=(0, 2, 3, 1),
        )
        focal_samples = self.cast(focal_samples, self.trt.float32, "output.focal_samples_fp32")

        x = self.dynamic_slice(xy, (0, 0, 0, 0), z_shape, "output.valid.x")
        y = self.dynamic_slice(xy, (0, 1, 0, 0), z_shape, "output.valid.y")
        x_finite = self.is_finite(x, "output.valid.x_finite")
        y_finite = self.is_finite(y, "output.valid.y_finite")
        z_finite = self.is_finite(z, "output.valid.z_finite")
        points_finite = self.binary(
            x_finite,
            y_finite,
            self.trt.ElementWiseOperation.AND,
            "output.valid.xy_finite",
        )
        points_finite = self.binary(
            points_finite,
            z_finite,
            self.trt.ElementWiseOperation.AND,
            "output.valid.xyz_finite",
        )
        points_finite = self.reshape(points_finite, mask_shape, "output.valid.points_squeeze")

        # Keep the legacy sigmoid and FP16->FP32 boundary. For a tiny positive
        # FP16 logit, sigmoid can round to exactly 0.5, so logit > 0 is not an
        # exact replacement for the public mask predicate.
        mask = self.reshape(mask, mask_shape, "output.mask_squeeze")
        mask = self.sigmoid(mask, "output.mask_sigmoid")
        mask = self.cast(mask, self.trt.float32, "output.mask_fp32")
        mask_threshold = self.constant(
            [[[0.5]]], dtype=np.float32, name="output.valid.mask_threshold"
        )
        mask_selected = self.binary(
            mask,
            mask_threshold,
            self.trt.ElementWiseOperation.GREATER,
            "output.valid.mask_selected",
        )
        # GREATER is ordered: NaN compares false. Sigmoid maps finite values
        # and +/-infinity to finite probabilities, so the legacy isfinite(mask)
        # term cannot reject anything that this comparison would select.
        valid = self.binary(
            points_finite,
            mask_selected,
            self.trt.ElementWiseOperation.AND,
            "output.valid.selected",
        )
        valid = self.cast(valid, self.trt.float16, "output.valid_fp16")

        scale = self.unary(scale, self.trt.UnaryOperation.EXP, "output.metric_scale_exp")
        scale = self.reshape(scale, (1,), "output.metric_scale_squeeze")
        scale = self.cast(scale, self.trt.float32, "output.metric_scale_fp32")
        return affine_depth, valid, focal_samples, scale


def _build_native_engine(
    state: dict[str, Any],
    *,
    precision: str,
    verbose: bool,
) -> bytes:
    fast_path = precision == "fp16"
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    if network is None:
        raise RuntimeError("TensorRT failed to create the MoGe network")
    input_dims = (1, -1, -1, 3)
    image = network.add_input("image", trt.float32, input_dims)
    if image is None:
        raise RuntimeError("TensorRT rejected the MoGe image input")
    graph = _NativeMogeGraph(trt, network, state, fast_path=fast_path)
    input_shape = graph.shape(image, "input_hwc.shape")
    input_h = graph.shape_index(input_shape, 1, "input_hwc.height")
    input_w = graph.shape_index(input_shape, 2, "input_hwc.width")
    nchw_shape = graph.shape_concat(
        [
            graph.shape_value(1, "input_hwc.batch"),
            graph.shape_value(3, "input_hwc.channels"),
            input_h,
            input_w,
        ],
        "input_hwc.nchw_shape",
    )
    image = graph.reshape(
        image,
        nchw_shape,
        "input_hwc.to_nchw",
        first_transpose=(0, 3, 1, 2),
    )
    affine_depth, valid, focal_samples, metric_scale = graph.outputs(image)
    for name, tensor in (
        ("affine_depth", affine_depth),
        ("valid", valid),
        ("focal_samples", focal_samples),
        ("metric_scale", metric_scale),
    ):
        tensor.name = name
        network.mark_output(tensor)

    config = builder.create_builder_config()
    tf32 = getattr(trt.BuilderFlag, "TF32", None)
    if tf32 is not None:
        if fast_path:
            config.set_flag(tf32)
        else:
            config.clear_flag(tf32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)
    profile = builder.create_optimization_profile()
    if fast_path:
        profile.set_shape(
            "image",
            (1, _FAST_MIN_IMAGE_HEIGHT, _FAST_MIN_IMAGE_WIDTH, 3),
            (1, _FAST_OPT_IMAGE_HEIGHT, _FAST_OPT_IMAGE_WIDTH, 3),
            (1, _FAST_MAX_IMAGE_HEIGHT, _FAST_MAX_IMAGE_WIDTH, 3),
        )
    else:
        profile.set_shape(
            "image",
            (1, _MIN_IMAGE_SIZE, _MIN_IMAGE_SIZE, 3),
            (1, _OPT_IMAGE_SIZE, _OPT_IMAGE_SIZE, 3),
            (1, _MAX_IMAGE_SIZE, _MAX_IMAGE_SIZE, 3),
        )
    if not profile:
        raise RuntimeError("Failed to configure the MoGe dynamic image profile")
    config.add_optimization_profile(profile)
    if hasattr(config, "builder_optimization_level"):
        # Level 3 enables the FP16 fused-attention fast path. Level 0 keeps the
        # broad dynamic FP32 attention graph decomposed for reliable builds.
        config.builder_optimization_level = 3 if fast_path else 0
    if hasattr(config, "avg_timing_iterations"):
        config.avg_timing_iterations = 3
    if hasattr(config, "max_aux_streams"):
        config.max_aux_streams = 0
    if verbose:
        profile_label = (
            f"{_FAST_MIN_IMAGE_WIDTH}x{_FAST_MIN_IMAGE_HEIGHT}.."
            f"{_FAST_MAX_IMAGE_WIDTH}x{_FAST_MAX_IMAGE_HEIGHT}"
            f"@{_FAST_OPT_IMAGE_WIDTH}x{_FAST_OPT_IMAGE_HEIGHT}"
            if fast_path
            else f"{_MIN_IMAGE_SIZE}..{_MAX_IMAGE_SIZE}"
        )
        print(
            "[trtmc build] Building native MoGe TensorRT graph "
            f"({network.num_layers} layers, num_tokens={_NUM_TOKENS}, "
            f"precision={precision}, profile={profile_label}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the native MoGe engine")
    return bytes(plan)


def build_moge_engine(
    model_dir: str,
    *,
    precision: str,
    verbose: bool = False,
) -> bytes:
    """Build the exact MoGe-2 ViT-L graph with TensorRT's native Python API."""

    model_root = Path(model_dir).resolve()
    checkpoint_path = model_root / _CHECKPOINT
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MoGe checkpoint not found: {checkpoint_path}")
    if precision not in {"fp32", "fp16"}:
        raise ValueError("The native MoGe-2 ViT-L builder supports precision='fp32' or 'fp16' only")
    torch = _require_torch()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    return _build_native_engine(
        checkpoint["model"],
        precision=precision,
        verbose=verbose,
    )


def build(request, writer) -> None:
    """Build one native MoGe-2 bundle."""

    if request.task != "monocular_geometry":
        raise ValueError("moge supports only task=monocular_geometry")
    if request.precision not in {"fp32", "fp16"}:
        raise ValueError("moge supports only precision=fp32 or precision=fp16")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("moge uses a dynamic image profile")
    if request.video_num_frames is not None:
        raise NotImplementedError("moge does not support video_num_frames")
    if request.max_sequence_length is not None:
        raise NotImplementedError("moge does not support max_sequence_length")
    if request.max_batch_size != 1:
        raise NotImplementedError("moge requires max_batch_size=1")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise NotImplementedError("moge supports only single-device builds")
    if request.quantization not in {None, "none"} or request.fp32_layers:
        raise NotImplementedError("moge does not support quantized or mixed-precision builds")

    plan = build_moge_engine(
        str(Path(request.model_dir).resolve()),
        precision=request.precision,
        verbose=bool(request.verbose),
    )
    writer.set_header(family="moge", task=request.task, backend="trt")
    writer.add_bytes("engine.plan", plan)
