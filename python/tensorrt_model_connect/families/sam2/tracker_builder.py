# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python TensorRT graph construction for SAM2 prompt and recurrent tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .graph_ops import NetworkBuildError, TrtLayers
from .float_math import cosf, powf, sinf


_LAYER_NORM_2D_EPSILON = 1.0e-6
_TRANSFORMER_LAYER_NORM_EPSILON = 1.0e-5
_TWO_PI = 6.28318530717958647692
_IMAGE_SIZE = 64
_IMAGE_TOKENS = _IMAGE_SIZE * _IMAGE_SIZE

_FPN_CONTRACTS = (
    ("tracker_fpn_0", "bfloat16", (1, 256, 256, 256)),
    ("tracker_fpn_1", "bfloat16", (1, 256, 128, 128)),
    ("tracker_fpn_2", "float32", (1, 256, 64, 64)),
)
_BOX_CONTRACT = ("box_xyxy_1024", "float32", (1, 4))
_OUTPUT_CONTRACTS = (
    ("mask_logits_256", "float32", (1, 1, 256, 256)),
    ("object_pointer", "float32", (1, 256)),
    ("memory_features", "bfloat16", (1, 64, 64, 64)),
)


@dataclass(frozen=True)
class _DecoderOutput:
    mask: Any
    object_pointer: Any
    object_present: Any


class _TrackerBuilder:
    def __init__(self, trt: Any, network: Any, checkpoint: Any) -> None:
        self.trt = trt
        self.network = network
        self.checkpoint = checkpoint
        self.layers = TrtLayers(trt, network, checkpoint)

    def _add_input(self, contract: tuple[str, str, tuple[int, ...]]) -> Any:
        name, dtype_name, shape = contract
        dtype = getattr(self.trt, dtype_name)
        tensor = self.network.add_input(name, dtype, shape)
        return tensor

    def _float_constant(
        self,
        values: Sequence[float] | np.ndarray,
        shape: Sequence[int],
        name: str,
        dtype: Any = None,
    ) -> Any:
        return self.layers.owned_constant(
            values, tuple(shape), self.trt.float32 if dtype is None else dtype, name
        )

    def _int_constant(self, value: int, shape: Sequence[int], name: str) -> Any:
        storage = np.full(tuple(shape), value, dtype=np.int32)
        self.layers._owned.append(storage)
        return self.layers._layer(
            self.network.add_constant(tuple(shape), storage), "integer constant", name
        ).get_output(0)

    def _unary(self, tensor: Any, operation: Any, name: str) -> Any:
        return self.layers._layer(
            self.network.add_unary(tensor, operation), "unary", name
        ).get_output(0)

    def _activation(self, tensor: Any, kind: Any, name: str) -> Any:
        return self.layers._layer(
            self.network.add_activation(tensor, kind), "activation", name
        ).get_output(0)

    def _reduce(
        self, tensor: Any, operation: Any, axes: int, keep_dimensions: bool, name: str
    ) -> Any:
        return self.layers._layer(
            self.network.add_reduce(tensor, operation, axes, keep_dimensions), "reduction", name
        ).get_output(0)

    def _binary(
        self, lhs: Any, rhs: Any, operation: Any, name: str, *, force_float: bool = False
    ) -> Any:
        dtype = lhs.dtype
        if force_float or lhs.dtype == self.trt.float32 or rhs.dtype == self.trt.float32:
            dtype = self.trt.float32
        lhs = self.layers.cast(lhs, dtype, f"{name}.lhs")
        rhs = self.layers.cast(rhs, dtype, f"{name}.rhs")
        return self.layers.elementwise(lhs, rhs, operation, name)

    def _scale(self, tensor: Any, value: float, name: str) -> Any:
        scalar = self.layers.scalar(
            value, len(self.layers._shape(tensor)), tensor.dtype, f"{name}.scalar"
        )
        return self.layers.elementwise(tensor, scalar, self.trt.ElementWiseOperation.PROD, name)

    def _select(self, condition: Any, then_value: Any, else_value: Any, name: str) -> Any:
        return self.layers._layer(
            self.network.add_select(condition, then_value, else_value), "select", name
        ).get_output(0)

    def _checkpoint_constant(
        self,
        checkpoint_name: str,
        checkpoint_shape: Sequence[int],
        tensor_shape: Sequence[int],
        name: str,
        dtype: Any = None,
    ) -> Any:
        value = self.layers.constant(checkpoint_name, checkpoint_shape, tensor_shape, name)
        return self.layers.cast(value, self.trt.float32 if dtype is None else dtype, f"{name}.cast")

    def _conv(
        self,
        tensor: Any,
        module: str,
        input_channels: int,
        output_channels: int,
        kernel: int,
        stride: int,
        padding: int,
        groups: int,
        name: str,
    ) -> Any:
        tensor = self.layers.cast(tensor, self.trt.bfloat16, f"{name}.input")
        return self.layers.convolution(
            tensor,
            f"{module}.weight",
            f"{module}.bias",
            input_channels,
            output_channels,
            kernel,
            stride,
            padding,
            groups,
            name,
        )

    def _deconv(
        self,
        tensor: Any,
        module: str,
        input_channels: int,
        output_channels: int,
        name: str,
    ) -> Any:
        tensor = self.layers.cast(tensor, self.trt.bfloat16, f"{name}.input")
        source_kernel = self.checkpoint.tensor(
            f"{module}.weight", (input_channels, output_channels, 2, 2)
        )
        source_bias = self.checkpoint.tensor(f"{module}.bias", (output_channels,))
        projection_kernel = np.ascontiguousarray(
            source_kernel.transpose(1, 2, 3, 0).reshape(output_channels * 4, input_channels, 1, 1)
        )
        projection_bias = np.ascontiguousarray(np.repeat(source_bias, 4))
        self.layers._owned.extend((projection_kernel, projection_bias))
        projection = self.layers._layer(
            self.network.add_convolution_nd(
                tensor,
                output_channels * 4,
                (1, 1),
                self.layers._bf16_weights(projection_kernel),
                self.layers._bf16_weights(projection_bias),
            ),
            "pixel-shuffle projection",
            f"{name}.projection",
        )
        projection.stride_nd = (1, 1)
        projection.num_groups = 1
        input_shape = self.layers._shape(tensor)
        height, width = input_shape[2:]
        blocked = self.layers.shuffle(
            projection.get_output(0),
            (1, output_channels, 2, 2, height, width),
            f"{name}.blocked",
        )
        ordered = self.layers.transpose(blocked, (0, 1, 4, 2, 5, 3), f"{name}.ordered")
        result = self.layers.shuffle(
            ordered, (1, output_channels, height * 2, width * 2), f"{name}.pixel_shuffle"
        )
        return result

    def _layer_norm_2d(self, tensor: Any, module: str, channels: int, name: str) -> Any:
        mean = self._reduce(tensor, self.trt.ReduceOperation.AVG, 1 << 1, True, f"{name}.mean_bf16")
        centered = self._binary(
            tensor, mean, self.trt.ElementWiseOperation.SUB, f"{name}.center_bf16"
        )
        centered = self.layers.cast(centered, self.trt.float32, f"{name}.center_fp32")
        squared = self.layers.elementwise(
            centered, centered, self.trt.ElementWiseOperation.PROD, f"{name}.square_fp32"
        )
        variance = self._reduce(
            squared, self.trt.ReduceOperation.AVG, 1 << 1, True, f"{name}.variance_fp32"
        )
        epsilon = self.layers.scalar(
            _LAYER_NORM_2D_EPSILON, 4, self.trt.float32, f"{name}.epsilon_fp32"
        )
        variance = self.layers.elementwise(
            variance, epsilon, self.trt.ElementWiseOperation.SUM, f"{name}.variance_with_epsilon"
        )
        deviation = self._unary(variance, self.trt.UnaryOperation.SQRT, f"{name}.deviation_fp32")
        normalized = self.layers.elementwise(
            centered, deviation, self.trt.ElementWiseOperation.DIV, f"{name}.normalize_fp32"
        )
        weight = self._checkpoint_constant(
            f"{module}.weight", (channels,), (1, channels, 1, 1), f"{name}.weight_fp32"
        )
        scaled = self.layers.elementwise(
            normalized, weight, self.trt.ElementWiseOperation.PROD, f"{name}.scale_fp32"
        )
        bias = self._checkpoint_constant(
            f"{module}.bias", (channels,), (1, channels, 1, 1), f"{name}.bias_fp32"
        )
        return self.layers.elementwise(scaled, bias, self.trt.ElementWiseOperation.SUM, name)

    def _linear_relu(
        self, tensor: Any, module: str, input_features: int, output_features: int, name: str
    ) -> Any:
        value = self.layers.linear_bf16(
            tensor, module, input_features, output_features, f"{name}.linear"
        )
        return self._activation(value, self.trt.ActivationType.RELU, name)

    def _make_sine_position(self, channels: int, name: str) -> Any:
        half = channels // 2
        values = np.empty((1, channels, _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.float32)
        for channel in range(channels):
            use_y = channel < half
            component = channel if use_y else channel - half
            exponent = np.float32(2.0 * (component // 2) / half)
            divisor = np.float32(powf(10000.0, float(exponent)))
            for y in range(_IMAGE_SIZE):
                y_position = np.float32(
                    np.float32(y + 1) / np.float32(_IMAGE_SIZE + np.float32(1.0e-6)) * _TWO_PI
                )
                for x in range(_IMAGE_SIZE):
                    x_position = np.float32(
                        np.float32(x + 1) / np.float32(_IMAGE_SIZE + np.float32(1.0e-6)) * _TWO_PI
                    )
                    phase = np.float32((y_position if use_y else x_position) / divisor)
                    values[0, channel, y, x] = (
                        sinf(float(phase)) if component % 2 == 0 else cosf(float(phase))
                    )
        return self._float_constant(values, values.shape, name)

    def _make_dense_random_position(self) -> Any:
        coordinates = np.empty((1, _IMAGE_TOKENS, 2), dtype=np.float32)
        for y in range(_IMAGE_SIZE):
            normalized_y = np.float32((np.float32(y) + np.float32(0.5)) / _IMAGE_SIZE)
            for x in range(_IMAGE_SIZE):
                normalized_x = np.float32((np.float32(x) + np.float32(0.5)) / _IMAGE_SIZE)
                spatial = y * _IMAGE_SIZE + x
                coordinates[0, spatial, 0] = np.float32(2.0) * normalized_x - np.float32(1.0)
                coordinates[0, spatial, 1] = np.float32(2.0) * normalized_y - np.float32(1.0)
        grid = self._float_constant(
            coordinates, coordinates.shape, "prompt.dense_position.grid", self.trt.bfloat16
        )
        gaussian = self._checkpoint_constant(
            "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
            (2, 128),
            (1, 2, 128),
            "prompt.dense_position.gaussian",
            self.trt.bfloat16,
        )
        phase = self.layers.matrix_multiply(
            grid,
            self.trt.MatrixOperation.NONE,
            gaussian,
            self.trt.MatrixOperation.NONE,
            "prompt.dense_position.fourier",
        )
        radians = self._scale(phase, _TWO_PI, "prompt.dense_position.radians")
        sine = self._unary(radians, self.trt.UnaryOperation.SIN, "prompt.dense_position.sin")
        cosine = self._unary(radians, self.trt.UnaryOperation.COS, "prompt.dense_position.cos")
        encoded = self.layers.concatenate((sine, cosine), 2, "prompt.dense_position.encoded")
        nhwc = self.layers.shuffle(encoded, (1, 64, 64, 256), "prompt.dense_position.nhwc")
        return self.layers.transpose(nhwc, (0, 3, 1, 2), "prompt.dense_position.nchw")

    def _make_pointer_temporal_position(self, distance: int, name: str) -> Any:
        values = np.empty(256, dtype=np.float32)
        position = np.float32(distance / 4.0)
        for feature in range(128):
            exponent = np.float32(2.0 * (feature // 2) / 128.0)
            phase = np.float32(position / np.float32(powf(10000.0, float(exponent))))
            values[feature] = sinf(float(phase))
            values[128 + feature] = cosf(float(phase))
        return self._float_constant(values, (1, 256), name)

    def _build_sparse_prompt(self, box: Any | None, prompt: bool) -> Any:
        not_a_point = self._checkpoint_constant(
            "sam_prompt_encoder.not_a_point_embed.weight",
            (1, 256),
            (1, 1, 256),
            "prompt.not_a_point",
        )
        if not prompt:
            return self.layers.concatenate(
                (not_a_point, not_a_point), 1, "prompt.recurrent_padding"
            )
        if box is None:
            raise NetworkBuildError("SAM2 prompt graph requires box_xyxy_1024")
        coordinates = self.layers.shuffle(box, (1, 2, 2), "prompt.box.reshape")
        half = self.layers.scalar(0.5, 3, self.trt.float32, "prompt.box.half")
        centered = self._binary(
            coordinates, half, self.trt.ElementWiseOperation.SUM, "prompt.box.centered"
        )
        normalized = self._scale(centered, 1.0 / 1024.0, "prompt.box.normalized")
        doubled = self._scale(normalized, 2.0, "prompt.box.doubled")
        minus_one = self.layers.scalar(-1.0, 3, self.trt.float32, "prompt.box.minus_one")
        unit = self._binary(
            doubled, minus_one, self.trt.ElementWiseOperation.SUM, "prompt.box.unit_square"
        )
        unit = self.layers.cast(unit, self.trt.bfloat16, "prompt.box.unit_square_bf16")
        gaussian = self._checkpoint_constant(
            "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
            (2, 128),
            (1, 2, 128),
            "prompt.box.gaussian",
            self.trt.bfloat16,
        )
        phase = self.layers.matrix_multiply(
            unit,
            self.trt.MatrixOperation.NONE,
            gaussian,
            self.trt.MatrixOperation.NONE,
            "prompt.box.fourier",
        )
        radians = self._scale(phase, _TWO_PI, "prompt.box.radians")
        sine = self._unary(radians, self.trt.UnaryOperation.SIN, "prompt.box.sin")
        cosine = self._unary(radians, self.trt.UnaryOperation.COS, "prompt.box.cos")
        encoded = self.layers.concatenate((sine, cosine), 2, "prompt.box.encoded")
        corner0 = self.layers.slice(
            encoded,
            (0, 0, 0),
            (1, 1, 256),
            (1, 1, 1),
            self.trt.SampleMode.STRICT_BOUNDS,
            "prompt.box.corner0",
        )
        corner1 = self.layers.slice(
            encoded,
            (0, 1, 0),
            (1, 1, 256),
            (1, 1, 1),
            self.trt.SampleMode.STRICT_BOUNDS,
            "prompt.box.corner1",
        )
        top_left = self._checkpoint_constant(
            "sam_prompt_encoder.point_embeddings.2.weight",
            (1, 256),
            (1, 1, 256),
            "prompt.box.top_left_embedding",
        )
        bottom_right = self._checkpoint_constant(
            "sam_prompt_encoder.point_embeddings.3.weight",
            (1, 256),
            (1, 1, 256),
            "prompt.box.bottom_right_embedding",
        )
        embedded0 = self._binary(
            corner0, top_left, self.trt.ElementWiseOperation.SUM, "prompt.box.corner0_embedding"
        )
        embedded1 = self._binary(
            corner1,
            bottom_right,
            self.trt.ElementWiseOperation.SUM,
            "prompt.box.corner1_embedding",
        )
        return self.layers.concatenate(
            (embedded0, embedded1, not_a_point), 1, "prompt.box.sparse_embeddings"
        )

    def _rotate(
        self,
        tensor: Any,
        sequence: int,
        cosine: np.ndarray,
        sine: np.ndarray,
        name: str,
    ) -> Any:
        original_type = tensor.dtype
        fp32 = self.layers.cast(tensor, self.trt.float32, f"{name}.fp32")
        paired = self.layers.shuffle(fp32, (1, 1, sequence, 128, 2), f"{name}.pairs")
        real = self.layers.slice(
            paired,
            (0, 0, 0, 0, 0),
            (1, 1, sequence, 128, 1),
            (1, 1, 1, 1, 1),
            self.trt.SampleMode.STRICT_BOUNDS,
            f"{name}.real_slice",
        )
        imag = self.layers.slice(
            paired,
            (0, 0, 0, 0, 1),
            (1, 1, sequence, 128, 1),
            (1, 1, 1, 1, 1),
            self.trt.SampleMode.STRICT_BOUNDS,
            f"{name}.imag_slice",
        )
        cos_tensor = self._float_constant(cosine, (1, 1, sequence, 128, 1), f"{name}.cos")
        sin_tensor = self._float_constant(sine, (1, 1, sequence, 128, 1), f"{name}.sin")
        real_cos = self._binary(
            real,
            cos_tensor,
            self.trt.ElementWiseOperation.PROD,
            f"{name}.real_cos",
            force_float=True,
        )
        imag_sin = self._binary(
            imag,
            sin_tensor,
            self.trt.ElementWiseOperation.PROD,
            f"{name}.imag_sin",
            force_float=True,
        )
        rotated_real = self._binary(
            real_cos,
            imag_sin,
            self.trt.ElementWiseOperation.SUB,
            f"{name}.rotated_real",
            force_float=True,
        )
        real_sin = self._binary(
            real,
            sin_tensor,
            self.trt.ElementWiseOperation.PROD,
            f"{name}.real_sin",
            force_float=True,
        )
        imag_cos = self._binary(
            imag,
            cos_tensor,
            self.trt.ElementWiseOperation.PROD,
            f"{name}.imag_cos",
            force_float=True,
        )
        rotated_imag = self._binary(
            real_sin,
            imag_cos,
            self.trt.ElementWiseOperation.SUM,
            f"{name}.rotated_imag",
            force_float=True,
        )
        interleaved = self.layers.concatenate(
            (rotated_real, rotated_imag), 4, f"{name}.interleaved"
        )
        flattened = self.layers.shuffle(interleaved, (1, 1, sequence, 256), f"{name}.flatten")
        return self.layers.cast(flattened, original_type, f"{name}.restore_type")

    @staticmethod
    def _axial_frequencies(repeats: int) -> tuple[np.ndarray, np.ndarray]:
        cosine = np.empty((repeats * _IMAGE_TOKENS, 128), dtype=np.float32)
        sine = np.empty_like(cosine)
        for repeat in range(repeats):
            for token in range(_IMAGE_TOKENS):
                x = np.float32(token % _IMAGE_SIZE)
                y = np.float32(token // _IMAGE_SIZE)
                for feature in range(64):
                    frequency = np.float32(1.0 / np.float32(powf(10000.0, feature * 4 / 256.0)))
                    for axis in range(2):
                        phase = np.float32((x if axis == 0 else y) * frequency)
                        offset = repeat * _IMAGE_TOKENS + token
                        complex_feature = axis * 64 + feature
                        cosine[offset, complex_feature] = cosf(float(phase))
                        sine[offset, complex_feature] = sinf(float(phase))
        return cosine, sine

    def _attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        module: str,
        query_features: int,
        key_value_features: int,
        internal_features: int,
        heads: int,
        name: str,
        *,
        rope: bool = False,
        rope_key_tokens: int = 0,
    ) -> Any:
        query_tokens = self.layers._shape(query)[1]
        key_tokens = self.layers._shape(key)[1]
        head_dim = internal_features // heads
        projected_q = self.layers.linear_bf16(
            query, f"{module}.q_proj", query_features, internal_features, f"{name}.q_proj"
        )
        projected_k = self.layers.linear_bf16(
            key, f"{module}.k_proj", key_value_features, internal_features, f"{name}.k_proj"
        )
        projected_v = self.layers.linear_bf16(
            value, f"{module}.v_proj", key_value_features, internal_features, f"{name}.v_proj"
        )
        q_grouped = self.layers.shuffle(
            projected_q, (1, query_tokens, heads, head_dim), f"{name}.q_group"
        )
        k_grouped = self.layers.shuffle(
            projected_k, (1, key_tokens, heads, head_dim), f"{name}.k_group"
        )
        v_grouped = self.layers.shuffle(
            projected_v, (1, key_tokens, heads, head_dim), f"{name}.v_group"
        )
        q_heads = self.layers.transpose(q_grouped, (0, 2, 1, 3), f"{name}.q_heads")
        k_heads = self.layers.transpose(k_grouped, (0, 2, 1, 3), f"{name}.k_heads")
        v_heads = self.layers.transpose(v_grouped, (0, 2, 1, 3), f"{name}.v_heads")
        if rope:
            q_cos, q_sin = self._axial_frequencies(1)
            q_heads = self._rotate(q_heads, query_tokens, q_cos, q_sin, f"{name}.q_rope")
            k_rope_part = self.layers.slice(
                k_heads,
                (0, 0, 0, 0),
                (1, 1, rope_key_tokens, 256),
                (1, 1, 1, 1),
                self.trt.SampleMode.STRICT_BOUNDS,
                f"{name}.k_rope_slice",
            )
            k_cos, k_sin = self._axial_frequencies(rope_key_tokens // query_tokens)
            rotated_k = self._rotate(k_rope_part, rope_key_tokens, k_cos, k_sin, f"{name}.k_rope")
            if rope_key_tokens != key_tokens:
                excluded = self.layers.slice(
                    k_heads,
                    (0, 0, rope_key_tokens, 0),
                    (1, 1, key_tokens - rope_key_tokens, 256),
                    (1, 1, 1, 1),
                    self.trt.SampleMode.STRICT_BOUNDS,
                    f"{name}.k_pointer_slice",
                )
                k_heads = self.layers.concatenate(
                    (rotated_k, excluded), 2, f"{name}.k_with_pointers"
                )
            else:
                k_heads = rotated_k
        scores = self.layers.matrix_multiply(
            q_heads,
            self.trt.MatrixOperation.NONE,
            k_heads,
            self.trt.MatrixOperation.TRANSPOSE,
            f"{name}.scores",
        )
        scaled = self._scale(scores, 1.0 / math.sqrt(head_dim), f"{name}.scaled")
        probabilities = self.layers.softmax(scaled, 1 << 3, f"{name}.softmax")
        attended = self.layers.matrix_multiply(
            probabilities,
            self.trt.MatrixOperation.NONE,
            v_heads,
            self.trt.MatrixOperation.NONE,
            f"{name}.values",
        )
        token_major = self.layers.transpose(attended, (0, 2, 1, 3), f"{name}.token_major")
        recombined = self.layers.shuffle(
            token_major, (1, query_tokens, internal_features), f"{name}.recombine"
        )
        return self.layers.linear_bf16(
            recombined,
            f"{module}.out_proj",
            internal_features,
            query_features,
            f"{name}.out_proj",
        )

    def _two_way_transformer(
        self, image: Any, image_position: Any, prompt_tokens: Any
    ) -> tuple[Any, Any]:
        image_flat = self.layers.shuffle(image, (1, 256, _IMAGE_TOKENS), "decoder.image.flatten")
        keys = self.layers.transpose(image_flat, (0, 2, 1), "decoder.image.token_major")
        position_flat = self.layers.shuffle(
            image_position, (1, 256, _IMAGE_TOKENS), "decoder.position.flatten"
        )
        key_position = self.layers.transpose(
            position_flat, (0, 2, 1), "decoder.position.token_major"
        )
        queries = prompt_tokens
        query_position = prompt_tokens
        for index in range(2):
            prefix = f"sam_mask_decoder.transformer.layers.{index}"
            name = f"decoder.two_way.{index}"
            if index == 0:
                queries = self._attention(
                    queries,
                    queries,
                    queries,
                    f"{prefix}.self_attn",
                    256,
                    256,
                    256,
                    8,
                    f"{name}.self",
                )
            else:
                q_with_position = self._binary(
                    queries,
                    query_position,
                    self.trt.ElementWiseOperation.SUM,
                    f"{name}.self.q_with_position",
                    force_float=True,
                )
                attended = self._attention(
                    q_with_position,
                    q_with_position,
                    queries,
                    f"{prefix}.self_attn",
                    256,
                    256,
                    256,
                    8,
                    f"{name}.self.attention",
                )
                queries = self._binary(
                    queries,
                    attended,
                    self.trt.ElementWiseOperation.SUM,
                    f"{name}.self.residual",
                    force_float=True,
                )
            queries = self.layers.layer_norm_fp32(
                queries,
                f"{prefix}.norm1",
                256,
                _TRANSFORMER_LAYER_NORM_EPSILON,
                f"{name}.norm1",
            )

            token_query = self._binary(
                queries,
                query_position,
                self.trt.ElementWiseOperation.SUM,
                f"{name}.token_to_image.query",
                force_float=True,
            )
            image_key = self._binary(
                keys,
                key_position,
                self.trt.ElementWiseOperation.SUM,
                f"{name}.token_to_image.key",
                force_float=True,
            )
            token_attended = self._attention(
                token_query,
                image_key,
                keys,
                f"{prefix}.cross_attn_token_to_image",
                256,
                256,
                128,
                8,
                f"{name}.token_to_image.attention",
            )
            queries = self._binary(
                queries,
                token_attended,
                self.trt.ElementWiseOperation.SUM,
                f"{name}.token_to_image.residual",
                force_float=True,
            )
            queries = self.layers.layer_norm_fp32(
                queries,
                f"{prefix}.norm2",
                256,
                _TRANSFORMER_LAYER_NORM_EPSILON,
                f"{name}.norm2",
            )

            hidden = self._linear_relu(
                queries, f"{prefix}.mlp.layers.0", 256, 2048, f"{name}.mlp.relu"
            )
            mlp = self.layers.linear_bf16(
                hidden, f"{prefix}.mlp.layers.1", 2048, 256, f"{name}.mlp.output"
            )
            queries = self._binary(
                queries,
                mlp,
                self.trt.ElementWiseOperation.SUM,
                f"{name}.mlp.residual",
                force_float=True,
            )
            queries = self.layers.layer_norm_fp32(
                queries,
                f"{prefix}.norm3",
                256,
                _TRANSFORMER_LAYER_NORM_EPSILON,
                f"{name}.norm3",
            )

            image_query = self._binary(
                keys,
                key_position,
                self.trt.ElementWiseOperation.SUM,
                f"{name}.image_to_token.query",
                force_float=True,
            )
            token_key = self._binary(
                queries,
                query_position,
                self.trt.ElementWiseOperation.SUM,
                f"{name}.image_to_token.key",
                force_float=True,
            )
            image_attended = self._attention(
                image_query,
                token_key,
                queries,
                f"{prefix}.cross_attn_image_to_token",
                256,
                256,
                128,
                8,
                f"{name}.image_to_token.attention",
            )
            keys = self._binary(
                keys,
                image_attended,
                self.trt.ElementWiseOperation.SUM,
                f"{name}.image_to_token.residual",
                force_float=True,
            )
            keys = self.layers.layer_norm_fp32(
                keys,
                f"{prefix}.norm4",
                256,
                _TRANSFORMER_LAYER_NORM_EPSILON,
                f"{name}.norm4",
            )

        final_q = self._binary(
            queries,
            query_position,
            self.trt.ElementWiseOperation.SUM,
            "decoder.final.query",
            force_float=True,
        )
        final_k = self._binary(
            keys,
            key_position,
            self.trt.ElementWiseOperation.SUM,
            "decoder.final.key",
            force_float=True,
        )
        final_attention = self._attention(
            final_q,
            final_k,
            keys,
            "sam_mask_decoder.transformer.final_attn_token_to_image",
            256,
            256,
            128,
            8,
            "decoder.final.attention",
        )
        queries = self._binary(
            queries,
            final_attention,
            self.trt.ElementWiseOperation.SUM,
            "decoder.final.residual",
            force_float=True,
        )
        queries = self.layers.layer_norm_fp32(
            queries,
            "sam_mask_decoder.transformer.norm_final_attn",
            256,
            _TRANSFORMER_LAYER_NORM_EPSILON,
            "decoder.final.norm",
        )
        return queries, keys

    def _top_candidate_index(self, scores: Any, name: str) -> Any:
        layer = self.network.add_topk(scores, self.trt.TopKOperation.MAX, 1, 1 << 1, self.trt.int32)
        return self.layers._layer(layer, "top-k", name).get_output(1)

    def _select_candidate(
        self,
        candidates: Any,
        indices: Any,
        candidate_shape: Sequence[int],
        condition_shape: Sequence[int],
        name: str,
    ) -> Any:
        result = None
        for index in range(3):
            start = [0] * len(candidate_shape)
            shape = list(candidate_shape)
            start[1] = index
            shape[1] = 1
            candidate = self.layers.slice(
                candidates,
                start,
                shape,
                (1,) * len(candidate_shape),
                self.trt.SampleMode.STRICT_BOUNDS,
                f"{name}.candidate.{index}",
            )
            if result is None:
                result = candidate
                continue
            expected = self._int_constant(index, (1, 1), f"{name}.index.{index}")
            condition = self.layers.elementwise(
                indices,
                expected,
                self.trt.ElementWiseOperation.EQUAL,
                f"{name}.equal.{index}",
            )
            expanded = self.layers.shuffle(condition, condition_shape, f"{name}.condition.{index}")
            result = self._select(expanded, candidate, result, f"{name}.select.{index}")
        return result

    def _decode(
        self, image: Any, fpn0: Any, fpn1: Any, sparse: Any, prompt: bool
    ) -> _DecoderOutput:
        no_mask = self._checkpoint_constant(
            "sam_prompt_encoder.no_mask_embed.weight",
            (1, 256),
            (1, 256, 1, 1),
            "decoder.no_mask",
        )
        source = self._binary(
            image,
            no_mask,
            self.trt.ElementWiseOperation.SUM,
            "decoder.image_with_no_mask",
            force_float=True,
        )
        dense_position = self._make_dense_random_position()
        object_token = self._checkpoint_constant(
            "sam_mask_decoder.obj_score_token.weight",
            (1, 256),
            (1, 1, 256),
            "decoder.object_token",
        )
        iou_token = self._checkpoint_constant(
            "sam_mask_decoder.iou_token.weight",
            (1, 256),
            (1, 1, 256),
            "decoder.iou_token",
        )
        mask_tokens = self._checkpoint_constant(
            "sam_mask_decoder.mask_tokens.weight",
            (4, 256),
            (1, 4, 256),
            "decoder.mask_tokens",
        )
        tokens = self.layers.concatenate(
            (object_token, iou_token, mask_tokens, sparse), 1, "decoder.input_tokens"
        )
        states, transformed_image = self._two_way_transformer(source, dense_position, tokens)
        strict = self.trt.SampleMode.STRICT_BOUNDS
        object_state = self.layers.slice(
            states, (0, 0, 0), (1, 1, 256), (1, 1, 1), strict, "decoder.object_state"
        )
        iou_state = self.layers.slice(
            states, (0, 1, 0), (1, 1, 256), (1, 1, 1), strict, "decoder.iou_state"
        )
        mask_states = self.layers.slice(
            states, (0, 2, 0), (1, 4, 256), (1, 1, 1), strict, "decoder.mask_states"
        )

        image_channels = self.layers.transpose(
            transformed_image, (0, 2, 1), "decoder.image.channels"
        )
        image_nchw = self.layers.shuffle(image_channels, (1, 256, 64, 64), "decoder.image.nchw")
        up1 = self._deconv(
            image_nchw,
            "sam_mask_decoder.output_upscaling.0",
            256,
            64,
            "decoder.upscale.0",
        )
        high1 = self._conv(
            fpn1, "sam_mask_decoder.conv_s1", 256, 64, 1, 1, 0, 1, "decoder.high_res.1"
        )
        merged1 = self._binary(
            up1, high1, self.trt.ElementWiseOperation.SUM, "decoder.upscale.0.skip"
        )
        norm1 = self._layer_norm_2d(
            merged1,
            "sam_mask_decoder.output_upscaling.1",
            64,
            "decoder.upscale.0.norm",
        )
        act1 = self.layers.gelu(norm1, "decoder.upscale.0.gelu")
        up2 = self._deconv(
            act1,
            "sam_mask_decoder.output_upscaling.3",
            64,
            32,
            "decoder.upscale.1",
        )
        high0 = self._conv(
            fpn0, "sam_mask_decoder.conv_s0", 256, 32, 1, 1, 0, 1, "decoder.high_res.0"
        )
        merged2 = self._binary(
            up2, high0, self.trt.ElementWiseOperation.SUM, "decoder.upscale.1.skip"
        )
        upscaled = self.layers.gelu(merged2, "decoder.upscale.1.gelu")

        hypernetworks = []
        for token in range(4):
            state = self.layers.slice(
                mask_states,
                (0, token, 0),
                (1, 1, 256),
                (1, 1, 1),
                strict,
                f"decoder.hyper.{token}.state",
            )
            module = f"sam_mask_decoder.output_hypernetworks_mlps.{token}.layers."
            hidden0 = self._linear_relu(
                state, f"{module}0", 256, 256, f"decoder.hyper.{token}.relu0"
            )
            hidden1 = self._linear_relu(
                hidden0, f"{module}1", 256, 256, f"decoder.hyper.{token}.relu1"
            )
            hypernetworks.append(
                self.layers.linear_bf16(
                    hidden1, f"{module}2", 256, 32, f"decoder.hyper.{token}.output"
                )
            )
        hyper = self.layers.concatenate(hypernetworks, 1, "decoder.hyper.stack")
        upscaled_flat = self.layers.shuffle(
            upscaled, (1, 32, 256 * 256), "decoder.upscaled.flatten"
        )
        masks_flat = self.layers.matrix_multiply(
            hyper,
            self.trt.MatrixOperation.NONE,
            upscaled_flat,
            self.trt.MatrixOperation.NONE,
            "decoder.masks.matmul",
        )
        all_masks = self.layers.shuffle(masks_flat, (1, 4, 256, 256), "decoder.masks.reshape")

        iou_hidden0 = self._linear_relu(
            iou_state,
            "sam_mask_decoder.iou_prediction_head.layers.0",
            256,
            256,
            "decoder.iou.relu0",
        )
        iou_hidden1 = self._linear_relu(
            iou_hidden0,
            "sam_mask_decoder.iou_prediction_head.layers.1",
            256,
            256,
            "decoder.iou.relu1",
        )
        iou_rank3 = self.layers.linear_bf16(
            iou_hidden1,
            "sam_mask_decoder.iou_prediction_head.layers.2",
            256,
            4,
            "decoder.iou.output",
        )
        iou = self.layers.shuffle(iou_rank3, (1, 4), "decoder.iou.squeeze")
        iou_probability = self._activation(
            iou, self.trt.ActivationType.SIGMOID, "decoder.iou.sigmoid"
        )

        object_hidden0 = self._linear_relu(
            object_state,
            "sam_mask_decoder.pred_obj_score_head.layers.0",
            256,
            256,
            "decoder.object.relu0",
        )
        object_hidden1 = self._linear_relu(
            object_hidden0,
            "sam_mask_decoder.pred_obj_score_head.layers.1",
            256,
            256,
            "decoder.object.relu1",
        )
        object_score_rank3 = self.layers.linear_bf16(
            object_hidden1,
            "sam_mask_decoder.pred_obj_score_head.layers.2",
            256,
            1,
            "decoder.object.output",
        )
        object_score = self.layers.cast(
            self.layers.shuffle(object_score_rank3, (1, 1), "decoder.object.squeeze"),
            self.trt.float32,
            "decoder.object.fp32",
        )
        zero = self.layers.scalar(0.0, 2, self.trt.float32, "decoder.object.zero")
        object_present = self.layers.elementwise(
            object_score,
            zero,
            self.trt.ElementWiseOperation.GREATER,
            "decoder.object.present",
        )
        all_masks_fp32 = self.layers.cast(all_masks, self.trt.float32, "decoder.masks.fp32")

        if prompt:
            stability_mask = self.layers.slice(
                all_masks,
                (0, 0, 0, 0),
                (1, 1, 256, 256),
                (1, 1, 1, 1),
                strict,
                "decoder.stability.mask_bf16",
            )
            single_mask = self.layers.slice(
                all_masks_fp32,
                (0, 0, 0, 0),
                (1, 1, 256, 256),
                (1, 1, 1, 1),
                strict,
                "decoder.mask.single",
            )
            candidate_masks = self.layers.slice(
                all_masks_fp32,
                (0, 1, 0, 0),
                (1, 3, 256, 256),
                (1, 1, 1, 1),
                strict,
                "decoder.mask.fallback_candidates",
            )
            candidate_iou = self.layers.slice(
                iou_probability,
                (0, 1),
                (1, 3),
                (1, 1),
                strict,
                "decoder.iou.fallback_candidates",
            )
            indices = self._top_candidate_index(candidate_iou, "decoder.iou.fallback_argmax")
            best_multimask = self._select_candidate(
                candidate_masks,
                indices,
                (1, 3, 256, 256),
                (1, 1, 1, 1),
                "decoder.mask.fallback_best",
            )
            upper = self.layers.scalar(0.05, 4, self.trt.bfloat16, "decoder.stability.upper")
            lower = self.layers.scalar(-0.05, 4, self.trt.bfloat16, "decoder.stability.lower")
            intersection_bool = self.layers.elementwise(
                stability_mask,
                upper,
                self.trt.ElementWiseOperation.GREATER,
                "decoder.stability.intersection_bool",
            )
            union_bool = self.layers.elementwise(
                stability_mask,
                lower,
                self.trt.ElementWiseOperation.GREATER,
                "decoder.stability.union_bool",
            )
            intersection = self._reduce(
                self.layers.cast(
                    intersection_bool, self.trt.float32, "decoder.stability.intersection_fp32"
                ),
                self.trt.ReduceOperation.SUM,
                (1 << 2) | (1 << 3),
                False,
                "decoder.stability.intersection",
            )
            union_area = self._reduce(
                self.layers.cast(union_bool, self.trt.float32, "decoder.stability.union_fp32"),
                self.trt.ReduceOperation.SUM,
                (1 << 2) | (1 << 3),
                False,
                "decoder.stability.union",
            )
            ratio = self.layers.elementwise(
                intersection,
                union_area,
                self.trt.ElementWiseOperation.DIV,
                "decoder.stability.ratio",
            )
            area_zero = self.layers.scalar(0.0, 2, self.trt.float32, "decoder.stability.area_zero")
            nonempty = self.layers.elementwise(
                union_area,
                area_zero,
                self.trt.ElementWiseOperation.GREATER,
                "decoder.stability.nonempty",
            )
            one = self.layers.scalar(1.0, 2, self.trt.float32, "decoder.stability.one")
            stability = self._select(nonempty, ratio, one, "decoder.stability.score")
            threshold = self.layers.scalar(0.98, 2, self.trt.float32, "decoder.stability.threshold")
            unstable = self.layers.elementwise(
                stability,
                threshold,
                self.trt.ElementWiseOperation.LESS,
                "decoder.stability.unstable",
            )
            unstable_mask = self.layers.shuffle(
                unstable, (1, 1, 1, 1), "decoder.stability.unstable_mask"
            )
            selected_mask = self._select(
                unstable_mask, best_multimask, single_mask, "decoder.mask.dynamic_fallback"
            )
            selected_state = self.layers.slice(
                mask_states,
                (0, 0, 0),
                (1, 1, 256),
                (1, 1, 1),
                strict,
                "decoder.pointer.single_state",
            )
        else:
            candidate_masks = self.layers.slice(
                all_masks_fp32,
                (0, 1, 0, 0),
                (1, 3, 256, 256),
                (1, 1, 1, 1),
                strict,
                "decoder.mask.candidates",
            )
            candidate_iou = self.layers.slice(
                iou_probability,
                (0, 1),
                (1, 3),
                (1, 1),
                strict,
                "decoder.iou.candidates",
            )
            candidate_states = self.layers.slice(
                mask_states,
                (0, 1, 0),
                (1, 3, 256),
                (1, 1, 1),
                strict,
                "decoder.pointer.candidates",
            )
            indices = self._top_candidate_index(candidate_iou, "decoder.iou.argmax")
            selected_mask = self._select_candidate(
                candidate_masks,
                indices,
                (1, 3, 256, 256),
                (1, 1, 1, 1),
                "decoder.mask.best",
            )
            selected_state = self._select_candidate(
                candidate_states,
                indices,
                (1, 3, 256),
                (1, 1, 1),
                "decoder.pointer.best",
            )

        object_present_rank4 = self.layers.shuffle(
            object_present, (1, 1, 1, 1), "decoder.object.present_mask"
        )
        no_object_mask = self.layers.scalar(-1024.0, 4, self.trt.float32, "decoder.mask.no_object")
        selected_mask = self._select(
            object_present_rank4, selected_mask, no_object_mask, "decoder.mask.object_gate"
        )
        pointer_state = self.layers.shuffle(selected_state, (1, 256), "decoder.pointer.state")
        pointer0 = self._linear_relu(
            pointer_state, "obj_ptr_proj.layers.0", 256, 256, "decoder.pointer.relu0"
        )
        pointer1 = self._linear_relu(
            pointer0, "obj_ptr_proj.layers.1", 256, 256, "decoder.pointer.relu1"
        )
        pointer2 = self.layers.linear_bf16(
            pointer1, "obj_ptr_proj.layers.2", 256, 256, "decoder.pointer.output"
        )
        pointer_fp32 = self.layers.cast(pointer2, self.trt.float32, "decoder.pointer.fp32")
        no_object_pointer = self._checkpoint_constant(
            "no_obj_ptr", (1, 256), (1, 256), "decoder.pointer.no_object"
        )
        gated_pointer = self._select(
            object_present, pointer_fp32, no_object_pointer, "decoder.pointer.object_gate"
        )
        return _DecoderOutput(selected_mask, gated_pointer, object_present)

    def _encode_memory(
        self,
        current_image: Any,
        low_resolution_mask: Any,
        object_present: Any,
        binarize_interacted_mask: bool,
    ) -> Any:
        high_resolution_mask = self.layers.resize_nchw(
            low_resolution_mask,
            1024,
            1024,
            self.trt.InterpolationMode.LINEAR,
            self.trt.ResizeCoordinateTransformation.HALF_PIXEL,
            "memory.mask.resize",
        )
        if binarize_interacted_mask:
            zero = self.layers.scalar(0.0, 4, self.trt.float32, "memory.mask.binary_zero")
            foreground = self.layers.elementwise(
                high_resolution_mask,
                zero,
                self.trt.ElementWiseOperation.GREATER,
                "memory.mask.binary_foreground",
            )
            probability = self.layers.cast(foreground, self.trt.float32, "memory.mask.binary_fp32")
        else:
            probability = self._activation(
                high_resolution_mask, self.trt.ActivationType.SIGMOID, "memory.mask.sigmoid"
            )
        scaled = self._scale(probability, 20.0, "memory.mask.scale")
        bias = self.layers.scalar(-10.0, 4, scaled.dtype, "memory.mask.bias")
        mask = self.layers.elementwise(
            scaled, bias, self.trt.ElementWiseOperation.SUM, "memory.mask.scaled_bias"
        )
        channels = (4, 16, 64, 256)
        conv_indices = (0, 3, 6, 9)
        norm_indices = (1, 4, 7, 10)
        input_channels = 1
        for index, output_channels in enumerate(channels):
            base = "memory_encoder.mask_downsampler.encoder."
            convolved = self._conv(
                mask,
                f"{base}{conv_indices[index]}",
                input_channels,
                output_channels,
                3,
                2,
                1,
                1,
                f"memory.downsample.{index}.conv",
            )
            normalized = self._layer_norm_2d(
                convolved,
                f"{base}{norm_indices[index]}",
                output_channels,
                f"memory.downsample.{index}.norm",
            )
            mask = self.layers.gelu(normalized, f"memory.downsample.{index}.gelu")
            input_channels = output_channels
        mask = self._conv(
            mask,
            "memory_encoder.mask_downsampler.encoder.12",
            256,
            256,
            1,
            1,
            0,
            1,
            "memory.downsample.project",
        )
        projected_image = self._conv(
            current_image,
            "memory_encoder.pix_feat_proj",
            256,
            256,
            1,
            1,
            0,
            1,
            "memory.image.project",
        )
        fused = self._binary(
            projected_image, mask, self.trt.ElementWiseOperation.SUM, "memory.fuser.input"
        )
        for index in range(2):
            module = f"memory_encoder.fuser.layers.{index}"
            name = f"memory.fuser.{index}"
            depthwise = self._conv(
                fused, module + ".dwconv", 256, 256, 7, 1, 3, 256, name + ".depthwise"
            )
            normalized = self._layer_norm_2d(depthwise, module + ".norm", 256, name + ".norm")
            nhwc = self.layers.transpose(normalized, (0, 2, 3, 1), name + ".nhwc")
            expanded = self.layers.linear_bf16(
                nhwc, module + ".pwconv1", 256, 1024, name + ".expand"
            )
            hidden = self.layers.gelu(expanded, name + ".gelu")
            projected = self.layers.linear_bf16(
                hidden, module + ".pwconv2", 1024, 256, name + ".project"
            )
            gamma = self._checkpoint_constant(
                module + ".gamma", (256,), (1, 1, 1, 256), name + ".gamma"
            )
            scaled_block = self._binary(
                projected,
                gamma,
                self.trt.ElementWiseOperation.PROD,
                name + ".scale",
                force_float=True,
            )
            nchw = self.layers.transpose(scaled_block, (0, 3, 1, 2), name + ".nchw")
            fused = self._binary(
                fused,
                nchw,
                self.trt.ElementWiseOperation.SUM,
                name + ".residual",
                force_float=True,
            )
        compressed = self._conv(
            fused, "memory_encoder.out_proj", 256, 64, 1, 1, 0, 1, "memory.output.project"
        )
        compressed_fp32 = self.layers.cast(compressed, self.trt.float32, "memory.output.fp32")
        visible = self.layers.shuffle(object_present, (1, 1, 1, 1), "memory.object.visible")
        no_object_embedding = self._checkpoint_constant(
            "no_obj_embed_spatial", (1, 64), (1, 64, 1, 1), "memory.object.embedding"
        )
        zero = self.layers.scalar(0.0, 4, self.trt.float32, "memory.object.zero")
        occlusion = self._select(
            visible, zero, no_object_embedding, "memory.object.select_embedding"
        )
        result = self._binary(
            compressed_fp32,
            occlusion,
            self.trt.ElementWiseOperation.SUM,
            "memory.output.with_object_state",
            force_float=True,
        )
        return self.layers.cast(result, self.trt.bfloat16, "memory.output.bf16")

    def _memory_attention(
        self,
        current: Any,
        current_position: Any,
        history_memory: Any,
        history_pointers: Any,
        history_frames: int,
    ) -> Any:
        strict = self.trt.SampleMode.STRICT_BOUNDS
        memory_features = []
        memory_positions = []
        spatial_position_fp32 = self._make_sine_position(64, "attention.memory.spatial_fp32")
        spatial_position = self.layers.cast(
            spatial_position_fp32, self.trt.bfloat16, "attention.memory.spatial_bf16"
        )
        temporal_table = self._checkpoint_constant(
            "maskmem_tpos_enc",
            (7, 1, 1, 64),
            (7, 1, 1, 64),
            "attention.memory.temporal_table",
        )
        for index in range(history_frames):
            feature_nchw = self.layers.slice(
                history_memory,
                (index, 0, 0, 0),
                (1, 64, 64, 64),
                (1, 1, 1, 1),
                strict,
                f"attention.memory.feature.{index}",
            )
            feature_flat = self.layers.shuffle(
                feature_nchw,
                (1, 64, _IMAGE_TOKENS),
                f"attention.memory.feature_flat.{index}",
            )
            memory_features.append(
                self.layers.transpose(
                    feature_flat, (0, 2, 1), f"attention.memory.feature_tokens.{index}"
                )
            )
            row = 6 if index == 0 else history_frames - index - 1
            temporal_nhwc = self.layers.slice(
                temporal_table,
                (row, 0, 0, 0),
                (1, 1, 1, 64),
                (1, 1, 1, 1),
                strict,
                f"attention.memory.temporal.{index}",
            )
            temporal = self.layers.transpose(
                temporal_nhwc, (0, 3, 1, 2), f"attention.memory.temporal_nchw.{index}"
            )
            positioned = self._binary(
                spatial_position,
                temporal,
                self.trt.ElementWiseOperation.SUM,
                f"attention.memory.position.{index}",
                force_float=True,
            )
            position_flat = self.layers.shuffle(
                positioned,
                (1, 64, _IMAGE_TOKENS),
                f"attention.memory.position_flat.{index}",
            )
            memory_positions.append(
                self.layers.transpose(
                    position_flat, (0, 2, 1), f"attention.memory.position_tokens.{index}"
                )
            )

        pointer_features = []
        pointer_positions = []
        for index in range(history_frames):
            frame = 0 if index == 0 else history_frames - index
            pointer = self.layers.slice(
                history_pointers,
                (frame, 0),
                (1, 256),
                (1, 1),
                strict,
                f"attention.pointer.feature.{index}",
            )
            pointer_features.append(
                self.layers.shuffle(pointer, (1, 4, 64), f"attention.pointer.tokens.{index}")
            )
            raw_position = self._make_pointer_temporal_position(
                history_frames if index == 0 else index,
                f"attention.pointer.raw_position.{index}",
            )
            projected_position = self.layers.linear_bf16(
                raw_position,
                "obj_ptr_tpos_proj",
                256,
                64,
                f"attention.pointer.project_position.{index}",
            )
            position_token = self.layers.shuffle(
                projected_position,
                (1, 1, 64),
                f"attention.pointer.position_token.{index}",
            )
            pointer_positions.append(
                self.layers.concatenate(
                    (position_token, position_token, position_token, position_token),
                    1,
                    f"attention.pointer.positions.{index}",
                )
            )
        spatial_memory = self.layers.concatenate(
            memory_features, 1, "attention.memory.spatial_features"
        )
        spatial_positions = self.layers.concatenate(
            memory_positions, 1, "attention.memory.spatial_positions"
        )
        pointers = self.layers.concatenate(pointer_features, 1, "attention.memory.pointer_features")
        pointer_position = self.layers.concatenate(
            pointer_positions, 1, "attention.memory.pointer_positions"
        )
        memory = self.layers.concatenate(
            (
                self.layers.cast(
                    spatial_memory, self.trt.float32, "attention.memory.features_fp32"
                ),
                pointers,
            ),
            1,
            "attention.memory.features",
        )
        position = self.layers.concatenate(
            (
                spatial_positions,
                self.layers.cast(
                    pointer_position,
                    self.trt.float32,
                    "attention.memory.pointer_position_fp32",
                ),
            ),
            1,
            "attention.memory.positions",
        )

        current_flat = self.layers.shuffle(
            current, (1, 256, _IMAGE_TOKENS), "attention.current.flatten"
        )
        output = self.layers.transpose(current_flat, (0, 2, 1), "attention.current.tokens")
        current_position_flat = self.layers.shuffle(
            current_position,
            (1, 256, _IMAGE_TOKENS),
            "attention.current_position.flatten",
        )
        current_position_tokens = self.layers.transpose(
            current_position_flat, (0, 2, 1), "attention.current_position.tokens"
        )
        scaled_position = self._scale(
            current_position_tokens, 0.1, "attention.current_position.scale"
        )
        output = self._binary(
            output,
            scaled_position,
            self.trt.ElementWiseOperation.SUM,
            "attention.input_with_position",
            force_float=True,
        )
        spatial_tokens = history_frames * _IMAGE_TOKENS
        for index in range(4):
            module = f"memory_attention.layers.{index}"
            name = f"attention.layer.{index}"
            self_norm = self.layers.layer_norm_fp32(
                output,
                module + ".norm1",
                256,
                _TRANSFORMER_LAYER_NORM_EPSILON,
                name + ".self.norm",
            )
            self_attended = self._attention(
                self_norm,
                self_norm,
                self_norm,
                module + ".self_attn",
                256,
                256,
                256,
                1,
                name + ".self.attention",
                rope=True,
                rope_key_tokens=_IMAGE_TOKENS,
            )
            output = self._binary(
                output,
                self_attended,
                self.trt.ElementWiseOperation.SUM,
                name + ".self.residual",
                force_float=True,
            )
            cross_norm = self.layers.layer_norm_fp32(
                output,
                module + ".norm2",
                256,
                _TRANSFORMER_LAYER_NORM_EPSILON,
                name + ".cross.norm",
            )
            positioned_memory = self._binary(
                memory,
                position,
                self.trt.ElementWiseOperation.SUM,
                name + ".cross.positioned_memory",
                force_float=True,
            )
            cross_attended = self._attention(
                cross_norm,
                positioned_memory,
                memory,
                module + ".cross_attn_image",
                256,
                64,
                256,
                1,
                name + ".cross.attention",
                rope=True,
                rope_key_tokens=spatial_tokens,
            )
            output = self._binary(
                output,
                cross_attended,
                self.trt.ElementWiseOperation.SUM,
                name + ".cross.residual",
                force_float=True,
            )
            mlp_norm = self.layers.layer_norm_fp32(
                output,
                module + ".norm3",
                256,
                _TRANSFORMER_LAYER_NORM_EPSILON,
                name + ".mlp.norm",
            )
            hidden = self._linear_relu(mlp_norm, module + ".linear1", 256, 2048, name + ".mlp.relu")
            projected = self.layers.linear_bf16(
                hidden, module + ".linear2", 2048, 256, name + ".mlp.output"
            )
            output = self._binary(
                output,
                projected,
                self.trt.ElementWiseOperation.SUM,
                name + ".mlp.residual",
                force_float=True,
            )
        output = self.layers.layer_norm_fp32(
            output,
            "memory_attention.norm",
            256,
            _TRANSFORMER_LAYER_NORM_EPSILON,
            "attention.output.norm",
        )
        channel_major = self.layers.transpose(output, (0, 2, 1), "attention.output.channel_major")
        return self.layers.shuffle(channel_major, (1, 256, 64, 64), "attention.output.nchw")

    def _mark_output(self, tensor: Any, contract: tuple[str, str, tuple[int, ...]]) -> None:
        name, dtype_name, shape = contract
        if tensor.dtype != getattr(self.trt, dtype_name) or self.layers._shape(tensor) != shape:
            raise NetworkBuildError(
                f"SAM2 tracker output contract mismatch for {name}: "
                f"actual {self.layers._shape(tensor)}"
            )
        tensor.name = name
        self.network.mark_output(tensor)

    def build(self, history_frames: int) -> None:
        prompt = history_frames == 0
        fpn0, fpn1, fpn2 = (self._add_input(contract) for contract in _FPN_CONTRACTS)
        box = None
        history_memory = None
        history_pointers = None
        if prompt:
            box = self._add_input(_BOX_CONTRACT)
        else:
            history_memory = self._add_input(
                (
                    "history_memory_features",
                    "bfloat16",
                    (history_frames, 64, 64, 64),
                )
            )
            history_pointers = self._add_input(
                ("history_object_pointers", "float32", (history_frames, 256))
            )
        current_position = self._make_sine_position(256, "tracker.current_position")
        if prompt:
            no_memory = self._checkpoint_constant(
                "no_mem_embed", (1, 1, 256), (1, 256, 1, 1), "tracker.no_memory"
            )
            decoder_image = self._binary(
                fpn2,
                no_memory,
                self.trt.ElementWiseOperation.SUM,
                "tracker.prompt.no_memory",
                force_float=True,
            )
        else:
            decoder_image = self._memory_attention(
                fpn2,
                current_position,
                history_memory,
                history_pointers,
                history_frames,
            )
        sparse = self._build_sparse_prompt(box, prompt)
        decoded = self._decode(decoder_image, fpn0, fpn1, sparse, prompt)
        mask = self.layers.cast(decoded.mask, self.trt.float32, "tracker.output.mask_fp32")
        pointer = self.layers.cast(
            decoded.object_pointer, self.trt.float32, "tracker.output.pointer_fp32"
        )
        memory = self._encode_memory(fpn2, mask, decoded.object_present, prompt)
        for tensor, contract in zip((mask, pointer, memory), _OUTPUT_CONTRACTS, strict=True):
            self._mark_output(tensor, contract)


def populate_tracker_network(
    trt: Any, network: Any, checkpoint: Any, history_frames: int
) -> _TrackerBuilder:
    """Populate one static prompt (H=0) or recurrent (H=1..4) SAM2 network.

    The returned layer owner must remain alive until TensorRT finishes serializing the
    network because TensorRT may retain pointers to NumPy-backed weights.
    """

    builder = _TrackerBuilder(trt, network, checkpoint)
    builder.build(history_frames)
    return builder
