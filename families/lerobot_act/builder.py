# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT graph for the declared LeRobot ACT inference policy.

The graph is a direct inference-only implementation of LeRobot's ACTPolicy:
mean/std preprocessing, ResNet-18 image features, four post-norm transformer
encoder layers, one post-norm decoder layer, and action unnormalization.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Sequence

import numpy as np
import tensorrt as trt


_IMAGE_HEIGHT = 480
_IMAGE_WIDTH = 640
_FEATURE_HEIGHT = 15
_FEATURE_WIDTH = 20
_IMAGE_TOKENS = _FEATURE_HEIGHT * _FEATURE_WIDTH
_HIDDEN = 512
_HEADS = 8
_HEAD_DIM = 64
_ENCODER_TOKENS = 2 + _IMAGE_TOKENS
_CHUNK_SIZE = 100
_ACTION_DIM = 14
_STATE_DIM = 14
_LATENT_DIM = 32
_FROZEN_BATCH_NORM_EPS = 1.0e-5
_LAYER_NORM_EPS = 1.0e-5


def _position_embedding_2d(height: int, width: int, dimension: int) -> np.ndarray:
    """Match ACTSinusoidalPositionEmbedding2d exactly for one feature map."""
    y_range = np.arange(1, height + 1, dtype=np.float32).reshape(height, 1)
    x_range = np.arange(1, width + 1, dtype=np.float32).reshape(1, width)
    y_range = y_range / (np.float32(height) + np.float32(1.0e-6)) * np.float32(2 * math.pi)
    x_range = x_range / (np.float32(width) + np.float32(1.0e-6)) * np.float32(2 * math.pi)
    inverse_frequency = np.float32(10000.0) ** (
        2 * (np.arange(dimension, dtype=np.float32) // 2) / np.float32(dimension)
    )
    x_angles = np.broadcast_to(x_range[..., None], (height, width, dimension)) / inverse_frequency
    y_angles = np.broadcast_to(y_range[..., None], (height, width, dimension)) / inverse_frequency

    pos_x = np.empty((height, width, dimension), dtype=np.float32)
    pos_y = np.empty((height, width, dimension), dtype=np.float32)
    pos_x[..., 0::2] = np.sin(x_angles[..., 0::2])
    pos_x[..., 1::2] = np.cos(x_angles[..., 1::2])
    pos_y[..., 0::2] = np.sin(y_angles[..., 0::2])
    pos_y[..., 1::2] = np.cos(y_angles[..., 1::2])
    return np.ascontiguousarray(np.concatenate((pos_y, pos_x), axis=-1).reshape(height * width, -1))


class _ActGraph:
    def __init__(self, trt: Any, network: Any, weights: dict[str, np.ndarray]) -> None:
        self.trt = trt
        self.network = network
        self.weights = weights
        self._host_weights: list[np.ndarray] = []

    def layer(self, value: Any, kind: str, name: str) -> Any:
        if value is None:
            raise RuntimeError(f"TensorRT rejected LeRobot ACT {kind} layer {name!r}")
        value.name = name
        return value

    def array(self, name: str, expected: tuple[int, ...] | None = None) -> np.ndarray:
        try:
            value = np.ascontiguousarray(self.weights[name], dtype=np.float32)
        except KeyError as exc:
            raise KeyError(f"LeRobot ACT checkpoint is missing tensor {name!r}") from exc
        if expected is not None and tuple(value.shape) != expected:
            raise ValueError(
                f"LeRobot ACT tensor {name!r} has shape {tuple(value.shape)}, expected {expected}"
            )
        self._host_weights.append(value)
        return value

    def constant(self, value: Any, name: str) -> Any:
        array = np.ascontiguousarray(value, dtype=np.float32)
        self._host_weights.append(array)
        return self.layer(
            self.network.add_constant(array.shape, self.trt.Weights(array)), "constant", name
        ).get_output(0)

    def add(self, left: Any, right: Any, name: str) -> Any:
        return self.layer(
            self.network.add_elementwise(left, right, self.trt.ElementWiseOperation.SUM),
            "sum",
            name,
        ).get_output(0)

    def sub(self, left: Any, right: Any, name: str) -> Any:
        return self.layer(
            self.network.add_elementwise(left, right, self.trt.ElementWiseOperation.SUB),
            "subtraction",
            name,
        ).get_output(0)

    def mul(self, left: Any, right: Any, name: str) -> Any:
        return self.layer(
            self.network.add_elementwise(left, right, self.trt.ElementWiseOperation.PROD),
            "product",
            name,
        ).get_output(0)

    def div(self, left: Any, right: Any, name: str) -> Any:
        return self.layer(
            self.network.add_elementwise(left, right, self.trt.ElementWiseOperation.DIV),
            "division",
            name,
        ).get_output(0)

    def reshape(
        self,
        tensor: Any,
        shape: tuple[int, ...],
        name: str,
        *,
        first_transpose: tuple[int, ...] | None = None,
        second_transpose: tuple[int, ...] | None = None,
    ) -> Any:
        shuffle = self.layer(self.network.add_shuffle(tensor), "shuffle", name)
        if first_transpose is not None:
            shuffle.first_transpose = first_transpose
        shuffle.reshape_dims = shape
        if second_transpose is not None:
            shuffle.second_transpose = second_transpose
        return shuffle.get_output(0)

    def concatenate(self, tensors: Sequence[Any], axis: int, name: str) -> Any:
        concat = self.layer(self.network.add_concatenation(list(tensors)), "concatenation", name)
        concat.axis = axis
        return concat.get_output(0)

    def linear_arrays(
        self,
        tensor: Any,
        weight: np.ndarray,
        bias: np.ndarray | None,
        name: str,
    ) -> Any:
        weight = np.ascontiguousarray(weight, dtype=np.float32)
        self._host_weights.append(weight)
        rhs = self.constant(weight, f"{name}.weight")
        output = self.layer(
            self.network.add_matrix_multiply(
                tensor,
                self.trt.MatrixOperation.NONE,
                rhs,
                self.trt.MatrixOperation.TRANSPOSE,
            ),
            "matrix multiply",
            f"{name}.matmul",
        ).get_output(0)
        if bias is None:
            return output
        bias = np.ascontiguousarray(bias, dtype=np.float32)
        bias_shape = (1,) * (len(tuple(output.shape)) - 1) + (int(bias.shape[0]),)
        return self.add(output, self.constant(bias.reshape(bias_shape), f"{name}.bias"), name)

    def linear(self, tensor: Any, prefix: str, name: str) -> Any:
        weight = self.array(f"{prefix}.weight")
        bias = self.array(f"{prefix}.bias", (weight.shape[0],))
        if weight.ndim != 2:
            raise ValueError(f"LeRobot ACT linear {prefix!r} must have a rank-2 weight")
        return self.linear_arrays(tensor, weight, bias, name)

    def layer_norm(self, tensor: Any, prefix: str, name: str) -> Any:
        rank = len(tuple(tensor.shape))
        scale = self.array(f"{prefix}.weight", (_HIDDEN,)).reshape((1,) * (rank - 1) + (_HIDDEN,))
        bias = self.array(f"{prefix}.bias", (_HIDDEN,)).reshape((1,) * (rank - 1) + (_HIDDEN,))
        norm = self.layer(
            self.network.add_normalization_v2(
                tensor,
                self.constant(scale, f"{name}.scale"),
                self.constant(bias, f"{name}.bias"),
                1 << (rank - 1),
            ),
            "layer normalization",
            name,
        )
        norm.epsilon = _LAYER_NORM_EPS
        return norm.get_output(0)

    def relu(self, tensor: Any, name: str) -> Any:
        return self.layer(
            self.network.add_activation(tensor, self.trt.ActivationType.RELU), "ReLU", name
        ).get_output(0)

    def convolution(
        self,
        tensor: Any,
        weight: np.ndarray,
        bias: np.ndarray,
        name: str,
        *,
        stride: int = 1,
        padding: int = 0,
    ) -> Any:
        weight = np.ascontiguousarray(weight, dtype=np.float32)
        bias = np.ascontiguousarray(bias, dtype=np.float32)
        self._host_weights.extend((weight, bias))
        conv = self.layer(
            self.network.add_convolution_nd(
                tensor,
                int(weight.shape[0]),
                tuple(int(item) for item in weight.shape[2:]),
                self.trt.Weights(weight),
                self.trt.Weights(bias),
            ),
            "convolution",
            name,
        )
        conv.stride_nd = (stride, stride)
        conv.padding_nd = (padding, padding)
        return conv.get_output(0)

    def frozen_batch_norm_conv(
        self,
        tensor: Any,
        conv_prefix: str,
        norm_prefix: str,
        name: str,
        *,
        stride: int = 1,
        padding: int = 0,
    ) -> Any:
        kernel = self.array(f"{conv_prefix}.weight")
        channels = int(kernel.shape[0])
        gamma = self.array(f"{norm_prefix}.weight", (channels,))
        beta = self.array(f"{norm_prefix}.bias", (channels,))
        mean = self.array(f"{norm_prefix}.running_mean", (channels,))
        variance = self.array(f"{norm_prefix}.running_var", (channels,))
        scale = gamma / np.sqrt(variance + np.float32(_FROZEN_BATCH_NORM_EPS))
        fused_kernel = kernel * scale.reshape(channels, 1, 1, 1)
        fused_bias = beta - mean * scale
        return self.convolution(
            tensor, fused_kernel, fused_bias, name, stride=stride, padding=padding
        )

    def resnet_block(
        self,
        tensor: Any,
        stage: int,
        block: int,
        *,
        stride: int,
    ) -> Any:
        prefix = f"model.backbone.layer{stage}.{block}"
        hidden = self.frozen_batch_norm_conv(
            tensor,
            f"{prefix}.conv1",
            f"{prefix}.bn1",
            f"backbone.layer{stage}.{block}.conv1",
            stride=stride,
            padding=1,
        )
        hidden = self.relu(hidden, f"backbone.layer{stage}.{block}.relu1")
        hidden = self.frozen_batch_norm_conv(
            hidden,
            f"{prefix}.conv2",
            f"{prefix}.bn2",
            f"backbone.layer{stage}.{block}.conv2",
            padding=1,
        )
        identity = tensor
        if stride != 1:
            identity = self.frozen_batch_norm_conv(
                tensor,
                f"{prefix}.downsample.0",
                f"{prefix}.downsample.1",
                f"backbone.layer{stage}.{block}.downsample",
                stride=stride,
            )
        return self.relu(
            self.add(hidden, identity, f"backbone.layer{stage}.{block}.residual"),
            f"backbone.layer{stage}.{block}.relu2",
        )

    def image_features(self, image_hwc: Any) -> Any:
        image = self.reshape(
            image_hwc,
            (1, 3, _IMAGE_HEIGHT, _IMAGE_WIDTH),
            "image.to_nchw",
            first_transpose=(0, 3, 1, 2),
        )
        mean = self.array("normalize_inputs.buffer_observation_images_top.mean", (3, 1, 1))
        std = self.array("normalize_inputs.buffer_observation_images_top.std", (3, 1, 1))
        image = self.div(
            self.sub(image, self.constant(mean.reshape(1, 3, 1, 1), "image.mean"), "image.center"),
            self.constant((std + np.float32(1.0e-8)).reshape(1, 3, 1, 1), "image.std"),
            "image.normalize",
        )
        image = self.frozen_batch_norm_conv(
            image,
            "model.backbone.conv1",
            "model.backbone.bn1",
            "backbone.conv1",
            stride=2,
            padding=3,
        )
        image = self.relu(image, "backbone.relu")
        pool = self.layer(
            self.network.add_pooling_nd(image, self.trt.PoolingType.MAX, (3, 3)),
            "max pooling",
            "backbone.maxpool",
        )
        pool.stride_nd = (2, 2)
        pool.padding_nd = (1, 1)
        image = pool.get_output(0)
        for stage in range(1, 5):
            for block in range(2):
                stride = 2 if stage > 1 and block == 0 else 1
                image = self.resnet_block(image, stage, block, stride=stride)

        projection_weight = self.array("model.encoder_img_feat_input_proj.weight", (512, 512, 1, 1))
        projection_bias = self.array("model.encoder_img_feat_input_proj.bias", (512,))
        image = self.convolution(image, projection_weight, projection_bias, "image.projection")
        if tuple(image.shape) != (1, _HIDDEN, _FEATURE_HEIGHT, _FEATURE_WIDTH):
            raise RuntimeError(f"Unexpected ACT image feature shape: {tuple(image.shape)}")
        return self.reshape(
            image,
            (_IMAGE_TOKENS, _HIDDEN),
            "image.to_tokens",
            first_transpose=(0, 2, 3, 1),
        )

    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix: str,
        name: str,
        *,
        query_tokens: int,
        key_tokens: int,
    ) -> Any:
        in_weight = self.array(f"{prefix}.in_proj_weight", (3 * _HIDDEN, _HIDDEN))
        in_bias = self.array(f"{prefix}.in_proj_bias", (3 * _HIDDEN,))
        q = self.linear_arrays(query, in_weight[:_HIDDEN], in_bias[:_HIDDEN], f"{name}.query")
        k = self.linear_arrays(
            key,
            in_weight[_HIDDEN : 2 * _HIDDEN],
            in_bias[_HIDDEN : 2 * _HIDDEN],
            f"{name}.key",
        )
        v = self.linear_arrays(
            value, in_weight[2 * _HIDDEN :], in_bias[2 * _HIDDEN :], f"{name}.value"
        )
        q = self.reshape(
            q,
            (query_tokens, _HEADS, _HEAD_DIM),
            f"{name}.query_heads",
            second_transpose=(1, 0, 2),
        )
        k = self.reshape(
            k,
            (key_tokens, _HEADS, _HEAD_DIM),
            f"{name}.key_heads",
            second_transpose=(1, 0, 2),
        )
        v = self.reshape(
            v,
            (key_tokens, _HEADS, _HEAD_DIM),
            f"{name}.value_heads",
            second_transpose=(1, 0, 2),
        )
        q = self.mul(
            q,
            self.constant(
                np.array([[[1.0 / math.sqrt(_HEAD_DIM)]]], dtype=np.float32), f"{name}.scale"
            ),
            f"{name}.scaled_query",
        )
        scores = self.layer(
            self.network.add_matrix_multiply(
                q,
                self.trt.MatrixOperation.NONE,
                k,
                self.trt.MatrixOperation.TRANSPOSE,
            ),
            "attention scores",
            f"{name}.scores",
        ).get_output(0)
        softmax = self.layer(self.network.add_softmax(scores), "softmax", f"{name}.softmax")
        softmax.axes = 1 << 2
        context = self.layer(
            self.network.add_matrix_multiply(
                softmax.get_output(0),
                self.trt.MatrixOperation.NONE,
                v,
                self.trt.MatrixOperation.NONE,
            ),
            "attention context",
            f"{name}.context_heads",
        ).get_output(0)
        context = self.reshape(
            context,
            (query_tokens, _HIDDEN),
            f"{name}.context",
            first_transpose=(1, 0, 2),
        )
        return self.linear(context, f"{prefix}.out_proj", f"{name}.output")

    def feed_forward(self, tensor: Any, prefix: str, name: str) -> Any:
        tensor = self.linear(tensor, f"{prefix}.linear1", f"{name}.linear1")
        tensor = self.relu(tensor, f"{name}.relu")
        return self.linear(tensor, f"{prefix}.linear2", f"{name}.linear2")

    def encoder(self, hidden: Any, positions: Any) -> Any:
        for index in range(4):
            prefix = f"model.encoder.layers.{index}"
            name = f"encoder.layer{index}"
            query_key = self.add(hidden, positions, f"{name}.positioned")
            attention = self.attention(
                query_key,
                query_key,
                hidden,
                f"{prefix}.self_attn",
                f"{name}.attention",
                query_tokens=_ENCODER_TOKENS,
                key_tokens=_ENCODER_TOKENS,
            )
            hidden = self.layer_norm(
                self.add(hidden, attention, f"{name}.attention_residual"),
                f"{prefix}.norm1",
                f"{name}.norm1",
            )
            feed_forward = self.feed_forward(hidden, prefix, f"{name}.feed_forward")
            hidden = self.layer_norm(
                self.add(hidden, feed_forward, f"{name}.feed_forward_residual"),
                f"{prefix}.norm2",
                f"{name}.norm2",
            )
        return hidden

    def decoder(self, encoder_hidden: Any, encoder_positions: Any) -> Any:
        prefix = "model.decoder.layers.0"
        decoder_positions = self.constant(
            self.array("model.decoder_pos_embed.weight", (_CHUNK_SIZE, _HIDDEN)),
            "decoder.positions",
        )
        hidden = self.constant(np.zeros((_CHUNK_SIZE, _HIDDEN), dtype=np.float32), "decoder.input")
        self_attention = self.attention(
            decoder_positions,
            decoder_positions,
            hidden,
            f"{prefix}.self_attn",
            "decoder.self_attention",
            query_tokens=_CHUNK_SIZE,
            key_tokens=_CHUNK_SIZE,
        )
        hidden = self.layer_norm(
            self.add(hidden, self_attention, "decoder.self_attention_residual"),
            f"{prefix}.norm1",
            "decoder.norm1",
        )
        cross_attention = self.attention(
            self.add(hidden, decoder_positions, "decoder.positioned_query"),
            self.add(encoder_hidden, encoder_positions, "decoder.positioned_key"),
            encoder_hidden,
            f"{prefix}.multihead_attn",
            "decoder.cross_attention",
            query_tokens=_CHUNK_SIZE,
            key_tokens=_ENCODER_TOKENS,
        )
        hidden = self.layer_norm(
            self.add(hidden, cross_attention, "decoder.cross_attention_residual"),
            f"{prefix}.norm2",
            "decoder.norm2",
        )
        feed_forward = self.feed_forward(hidden, prefix, "decoder.feed_forward")
        hidden = self.layer_norm(
            self.add(hidden, feed_forward, "decoder.feed_forward_residual"),
            f"{prefix}.norm3",
            "decoder.norm3",
        )
        return self.layer_norm(hidden, "model.decoder.norm", "decoder.final_norm")

    def outputs(self, image: Any, state: Any) -> Any:
        state_mean = self.array("normalize_inputs.buffer_observation_state.mean", (_STATE_DIM,))
        state_std = self.array("normalize_inputs.buffer_observation_state.std", (_STATE_DIM,))
        normalized_state = self.div(
            self.sub(
                state,
                self.constant(state_mean.reshape(1, _STATE_DIM), "state.mean"),
                "state.center",
            ),
            self.constant((state_std + np.float32(1.0e-8)).reshape(1, _STATE_DIM), "state.std"),
            "state.normalize",
        )
        latent = self.linear(
            self.constant(np.zeros((1, _LATENT_DIM), dtype=np.float32), "latent.zeros"),
            "model.encoder_latent_input_proj",
            "latent.projection",
        )
        state_token = self.linear(
            normalized_state, "model.encoder_robot_state_input_proj", "state.projection"
        )
        image_tokens = self.image_features(image)
        hidden = self.concatenate((latent, state_token, image_tokens), 0, "encoder.input")

        one_d_positions = self.array("model.encoder_1d_feature_pos_embed.weight", (2, _HIDDEN))
        image_positions = _position_embedding_2d(_FEATURE_HEIGHT, _FEATURE_WIDTH, _HIDDEN // 2)
        positions = self.constant(
            np.concatenate((one_d_positions, image_positions), axis=0), "encoder.positions"
        )
        encoder_hidden = self.encoder(hidden, positions)
        decoder_hidden = self.decoder(encoder_hidden, positions)
        normalized_actions = self.linear(decoder_hidden, "model.action_head", "action.head")
        action_mean = self.array("unnormalize_outputs.buffer_action.mean", (_ACTION_DIM,))
        action_std = self.array("unnormalize_outputs.buffer_action.std", (_ACTION_DIM,))
        actions = self.add(
            self.mul(
                normalized_actions,
                self.constant(action_std.reshape(1, _ACTION_DIM), "action.std"),
                "action.scale",
            ),
            self.constant(action_mean.reshape(1, _ACTION_DIM), "action.mean"),
            "action.unnormalize",
        )
        return self.reshape(actions, (1, _CHUNK_SIZE, _ACTION_DIM), "action.output_shape")


def build_act_engine(
    raw_config: dict[str, Any],
    weights: dict[str, np.ndarray],
    *,
    precision: str,
    verbose: bool = False,
) -> bytes:
    del raw_config
    if precision != "fp32":
        raise ValueError("The LeRobot ACT accuracy contract supports fp32 builds only")
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    if network is None:
        raise RuntimeError("TensorRT failed to create the LeRobot ACT network")
    image = network.add_input("observation_image", trt.float32, (1, _IMAGE_HEIGHT, _IMAGE_WIDTH, 3))
    state = network.add_input("observation_state", trt.float32, (1, _STATE_DIM))
    if image is None or state is None:
        raise RuntimeError("TensorRT rejected the LeRobot ACT observation inputs")
    graph = _ActGraph(trt, network, weights)
    actions = graph.outputs(image, state)
    actions.name = "actions"
    network.mark_output(actions)

    config = builder.create_builder_config()
    tf32 = getattr(trt.BuilderFlag, "TF32", None)
    if tf32 is not None:
        config.clear_flag(tf32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)
    if hasattr(config, "avg_timing_iterations"):
        config.avg_timing_iterations = 2
    if hasattr(config, "max_aux_streams"):
        config.max_aux_streams = 0
    if verbose:
        print(
            "[trtmc build] Building LeRobot ACT TensorRT graph "
            f"({network.num_layers} layers, input=480x640, actions=100x14, precision=fp32) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the LeRobot ACT engine")
    return bytes(plan)
