# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT NetworkDefinition builder for MoGe-2 ViT-L.

All model topology is owned here. TensorRT receives the original dynamic RGB
image and builds the complete DINOv2 encoder, convolutional neck and heads,
and raw MoGe forward outputs without an exporter, parser, or custom kernel.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorrt_model_connect import trt_compat


_CHECKPOINT = Path("model.pt")
_CHECKPOINT_SHA256 = "3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5"
_INTERMEDIATE_LAYERS = (5, 11, 17, 23)
_HIDDEN = 1024
_HEADS = 16
_HEAD_DIM = 64
_PATCH = 14
_POSITION_GRID = 37
_NUM_TOKENS = 1800
_MIN_IMAGE_SIZE = 64
_OPT_IMAGE_SIZE = 518
_MAX_IMAGE_SIZE = 2048


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "MoGe engine builds require PyTorch to read the official checkpoint. "
            "Install the model extra with `pip install 'tensorrt-model-connect[moge]'`."
        ) from exc
    return torch


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        while chunk := checkpoint.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class _NativeMogeGraph:
    """Small family-local vocabulary for composing the exact MoGe graph."""

    def __init__(self, trt: Any, network: Any, state: dict[str, Any]) -> None:
        self.trt = trt
        self.network = network
        self.state = state
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
    ) -> Any:
        weight = self._array(f"{module}.weight")
        if weight.ndim != 4:
            raise ValueError(f"MoGe convolution {module!r} does not have a 4D kernel")
        bias = self._array(f"{module}.bias", (weight.shape[0],))
        padded = self.replicate_pad(tensor, replicate_padding, f"{name}.pad")
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
        layer.padding_nd = (0, 0)
        return layer.get_output(0)

    def deconvolution(self, tensor: Any, module: str, name: str) -> Any:
        weight = self._array(f"{module}.weight")
        if weight.ndim != 4:
            raise ValueError(f"MoGe deconvolution {module!r} does not have a 4D kernel")
        output_channels = int(weight.shape[1])
        bias = self._array(f"{module}.bias", (output_channels,))
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

    def linear(self, tensor: Any, module: str, name: str) -> Any:
        weight = self._array(f"{module}.weight")
        if weight.ndim != 2:
            raise ValueError(f"MoGe linear {module!r} does not have a 2D weight")
        output_width, input_width = (int(value) for value in weight.shape)
        bias = self._array(f"{module}.bias", (output_width,))
        rank = len(tuple(tensor.shape))
        matrix_shape = (1,) * max(0, rank - 2) + (output_width, input_width)
        rhs = self.constant(weight.reshape(matrix_shape), name=f"{name}.weight")
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
        bias_shape = (1,) * (rank - 1) + (output_width,)
        bias_tensor = self.constant(bias.reshape(bias_shape), name=f"{name}.bias")
        return self.binary(
            product, bias_tensor, self.trt.ElementWiseOperation.SUM, f"{name}.bias_add"
        )

    def layer_norm(self, tensor: Any, module: str, name: str) -> Any:
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
        attention.decomposable = True
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
        attention = self.binary(
            attention, gamma1, self.trt.ElementWiseOperation.PROD, f"{name}.scaled_attention"
        )
        hidden = self.binary(
            hidden, attention, self.trt.ElementWiseOperation.SUM, f"{name}.attention_residual"
        )
        normalized = self.layer_norm(hidden, f"{prefix}.norm2", f"{name}.norm2")
        mlp = self.linear(normalized, f"{prefix}.mlp.fc1", f"{name}.mlp.fc1")
        mlp = self.gelu(mlp, f"{name}.mlp.gelu")
        mlp = self.linear(mlp, f"{prefix}.mlp.fc2", f"{name}.mlp.fc2")
        gamma2 = self.weight_constant(
            f"{prefix}.ls2.gamma",
            expected=(_HIDDEN,),
            shape=(1, 1, _HIDDEN),
            name=f"{name}.ls2",
        )
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
            pixels, "encoder.backbone.patch_embed.proj", "vit.patch_embed", stride=_PATCH
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
        hidden = self.binary(
            hidden,
            position_concat.get_output(0),
            self.trt.ElementWiseOperation.SUM,
            "vit.tokens_plus_position",
        )

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

    def residual_conv_block(self, tensor: Any, module: str, name: str) -> Any:
        hidden = self.relu(tensor, f"{name}.relu1")
        hidden = self.convolution(
            hidden, f"{module}.layers.2", f"{name}.conv1", replicate_padding=1
        )
        hidden = self.relu(hidden, f"{name}.relu2")
        hidden = self.convolution(
            hidden, f"{module}.layers.5", f"{name}.conv2", replicate_padding=1
        )
        return self.binary(hidden, tensor, self.trt.ElementWiseOperation.SUM, f"{name}.residual")

    def resample(self, tensor: Any, module: str, level: int, name: str) -> Any:
        if level < 3:
            tensor = self.deconvolution(tensor, f"{module}.0", f"{name}.deconvolution")
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
        return self.convolution(tensor, f"{module}.1", f"{name}.convolution", replicate_padding=1)

    def conv_stack(
        self,
        inputs: list[Any],
        prefix: str,
        num_res_blocks: tuple[int, ...],
        *,
        final_projection: bool,
    ) -> list[Any]:
        outputs: list[Any] = []
        current = None
        for level, feature in enumerate(inputs):
            projected = self.convolution(
                feature, f"{prefix}.input_blocks.{level}", f"{prefix}.level.{level}.input"
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
                )
            output = current
            if final_projection and level == len(inputs) - 1:
                output = self.convolution(
                    current,
                    f"{prefix}.output_blocks.{level}",
                    f"{prefix}.level.{level}.output",
                )
            outputs.append(output)
            if level < len(inputs) - 1:
                current = self.resample(
                    current,
                    f"{prefix}.resamplers.{level}",
                    level,
                    f"{prefix}.level.{level}.resample",
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

        neck = self.conv_stack(
            features, "neck", (0, 2, 2, 2, 0), final_projection=False
        )
        points = self.conv_stack(
            neck, "points_head", (0, 1, 1, 1, 0), final_projection=True
        )[-1]
        mask = self.conv_stack(
            neck, "mask_head", (0, 1, 1, 1, 0), final_projection=True
        )[-1]
        scale = self.linear(class_vector, "scale_head.0", "scale_head.0")
        scale = self.relu(scale, "scale_head.1")
        scale = self.linear(scale, "scale_head.2", "scale_head.2")
        scale = self.relu(scale, "scale_head.3")
        scale = self.linear(scale, "scale_head.4", "scale_head.4")

        input_shape = self.shape(image, "output.input_shape")
        input_h = self.shape_index(input_shape, 2, "output.height")
        input_w = self.shape_index(input_shape, 3, "output.width")
        points = self.resize_nchw_to_hw(
            points, input_h, input_w, self.trt.InterpolationMode.LINEAR, "output.points_resize"
        )
        mask = self.resize_nchw_to_hw(
            mask, input_h, input_w, self.trt.InterpolationMode.LINEAR, "output.mask_resize"
        )
        points_shape = self.shape_concat(
            [
                self.shape_value(1, "output.points_batch"),
                input_h,
                input_w,
                self.shape_value(3, "output.points_channels"),
            ],
            "output.points_shape",
        )
        points = self.reshape(
            points,
            points_shape,
            "output.points_nhwc",
            first_transpose=(0, 2, 3, 1),
        )
        xy_shape = self.shape_concat(
            [
                self.shape_value(1, "output.xy_batch"),
                input_h,
                input_w,
                self.shape_value(2, "output.xy_channels"),
            ],
            "output.xy_shape",
        )
        z_shape = self.shape_concat(
            [
                self.shape_value(1, "output.z_batch"),
                input_h,
                input_w,
                self.shape_value(1, "output.z_channels"),
            ],
            "output.z_shape",
        )
        xy = self.dynamic_slice(points, (0, 0, 0, 0), xy_shape, "output.xy")
        z = self.dynamic_slice(points, (0, 0, 0, 2), z_shape, "output.z")
        z = self.unary(z, self.trt.UnaryOperation.EXP, "output.z_exp")
        xy = self.binary(xy, z, self.trt.ElementWiseOperation.PROD, "output.xy_scaled")
        point_concat = self._layer(
            self.network.add_concatenation([xy, z]), "point concat", "output.points_remap"
        )
        point_concat.axis = 3
        points = point_concat.get_output(0)

        mask_shape = self.shape_concat(
            [self.shape_value(1, "output.mask_batch"), input_h, input_w],
            "output.mask_shape",
        )
        mask = self.reshape(mask, mask_shape, "output.mask_squeeze")
        mask = self.sigmoid(mask, "output.mask_sigmoid")
        scale = self.unary(scale, self.trt.UnaryOperation.EXP, "output.metric_scale_exp")
        scale = self.reshape(scale, (1,), "output.metric_scale_squeeze")
        return points, mask, scale


def _build_native_engine(
    state: dict[str, Any],
    *,
    verbose: bool,
) -> bytes:
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    if network is None:
        raise RuntimeError("TensorRT failed to create the MoGe network")
    image = network.add_input("image", trt.float32, (1, -1, -1, 3))
    if image is None:
        raise RuntimeError("TensorRT rejected the MoGe image input")
    graph = _NativeMogeGraph(trt, network, state)
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
    points, mask, metric_scale = graph.outputs(image)
    for name, tensor in (
        ("points", points),
        ("mask", mask),
        ("metric_scale", metric_scale),
    ):
        if tensor.dtype != trt.float32:
            tensor = graph.cast(tensor, trt.float32, f"output.{name}_fp32")
        tensor.name = name
        network.mark_output(tensor)

    config = builder.create_builder_config()
    tf32 = getattr(trt.BuilderFlag, "TF32", None)
    if tf32 is not None:
        config.clear_flag(tf32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)
    profile = builder.create_optimization_profile()
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
        # TRT 11.2 optimization level 3 tries to absorb the dynamic FP32
        # decomposable-attention chain into one Myelin ForeignNode, whose
        # dynamic BMM fallback has no implementation. Level 1 preserves the
        # native IAttention decomposition and full 64..2048 profile while
        # still timing tactics instead of accepting the first valid choice.
        config.builder_optimization_level = 1
    if hasattr(config, "avg_timing_iterations"):
        config.avg_timing_iterations = 3
    if hasattr(config, "max_aux_streams"):
        config.max_aux_streams = 0
    if verbose:
        print(
            "[trtmc build] Building native MoGe TensorRT graph "
            f"({network.num_layers} layers, num_tokens={_NUM_TOKENS}, "
            f"profile={_MIN_IMAGE_SIZE}..{_MAX_IMAGE_SIZE}) ...",
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
    if precision != "fp32":
        raise ValueError(
            "The native MoGe-2 ViT-L accuracy contract supports precision='fp32' only"
        )
    checkpoint_sha256 = _checkpoint_digest(checkpoint_path)
    if checkpoint_sha256 != _CHECKPOINT_SHA256:
        raise ValueError(
            "MoGe checkpoint SHA-256 mismatch: "
            f"expected {_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )

    torch = _require_torch()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    return _build_native_engine(
        checkpoint["model"],
        verbose=verbose,
    )
