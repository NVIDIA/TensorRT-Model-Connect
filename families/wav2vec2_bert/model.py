# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W2v-BERT normalized spectral projection and relative-key conformer encoder."""

import json
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import graph as g


def build(request, writer):
    if request.dynamic_kv_cache:
        raise NotImplementedError("wav2vec2_bert does not support dynamic_kv_cache")
    if request.task != "spectrogram_to_frame_features" or request.backend != "trt":
        raise ValueError("W2v-BERT requires spectrogram_to_frame_features with backend=trt")
    if (
        request.precision != "fp32"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.image_height is not None
        or request.image_width is not None
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError("W2v-BERT supports one fp32 normalized spectral feature sequence")
    directory = Path(request.model_dir)
    cfg = json.loads((directory / "config.json").read_text())
    if (
        cfg["model_type"] != "wav2vec2-bert"
        or cfg["add_adapter"]
        or cfg["position_embeddings_type"] != "relative_key"
        or cfg["hidden_act"] not in ("swish", "silu")
    ):
        raise ValueError("Unsupported W2v-BERT conformer encoder configuration")
    frames = request.max_sequence_length or 99
    if not 1 <= frames <= 512:
        raise ValueError("W2v-BERT supports 1 to 512 spectral feature frames")
    from safetensors import safe_open

    with safe_open(str(directory / "model.safetensors"), framework="np") as handle:
        weights = {name: handle.get_tensor(name).astype(np.float32) for name in handle.keys()}
    width, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    if width % heads:
        raise ValueError("Invalid W2v-BERT head dimensions")
    dim = width // heads
    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)

    def const(value):
        value = np.ascontiguousarray(value, dtype=np.float32)
        return net.add_constant(value.shape, value).get_output(0)

    def add(a, b):
        return net.add_elementwise(a, b, trt.ElementWiseOperation.SUM).get_output(0)

    def mul(a, b):
        return net.add_elementwise(a, b, trt.ElementWiseOperation.PROD).get_output(0)

    def reshape(x, shape, transpose=None):
        layer = net.add_shuffle(x)
        if transpose is not None:
            layer.first_transpose = transpose
        layer.reshape_dims = shape
        return layer.get_output(0)

    def linear(x, name, convolution=False):
        weight = weights[name + ".weight"]
        if convolution:
            weight = weight[:, :, 0]
        return g.add_linear(net, x, weight, weights.get(name + ".bias"))

    def norm(x, name):
        return g.add_layer_norm(
            net, x, weights[name + ".weight"], weights[name + ".bias"], cfg["layer_norm_eps"]
        )

    def swish(x):
        return mul(x, net.add_activation(x, trt.ActivationType.SIGMOID).get_output(0))

    def ffn(x, name):
        return linear(swish(linear(x, name + ".intermediate_dense")), name + ".output_dense")

    def split_heads(x):
        layer = net.add_shuffle(x)
        layer.reshape_dims = (1, frames, heads, dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    spectrogram = net.add_input(
        "spectrogram", trt.float32, (frames, cfg["feature_projection_input_dim"])
    )
    hidden = linear(
        norm(spectrogram, "feature_projection.layer_norm"), "feature_projection.projection"
    )
    indices = np.arange(frames)
    distance = (
        np.clip(
            indices[None, :] - indices[:, None],
            -cfg["left_max_position_embeddings"],
            cfg["right_max_position_embeddings"],
        )
        + cfg["left_max_position_embeddings"]
    )
    scale = const(np.full((1, 1, 1, 1), dim**-0.5))
    half = const(np.full((1, 1), 0.5))
    for index in range(cfg["num_hidden_layers"]):
        prefix = f"encoder.layers.{index}"
        hidden = add(
            hidden, mul(ffn(norm(hidden, prefix + ".ffn1_layer_norm"), prefix + ".ffn1"), half)
        )
        normalized = norm(hidden, prefix + ".self_attn_layer_norm")
        query, key, value = [
            split_heads(linear(normalized, prefix + ".self_attn.linear_" + name))
            for name in ("q", "k", "v")
        ]
        content = net.add_matrix_multiply(
            query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        # Relative-key vectors depend on (query frame, key frame). Treat query
        # frames as the batch axis to contract the head dimension for all heads.
        by_frame = reshape(query, (frames, heads, dim), (0, 2, 1, 3))
        table = const(weights[prefix + ".self_attn.distance_embedding.weight"][distance])
        relative = net.add_matrix_multiply(
            by_frame, trt.MatrixOperation.NONE, table, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        relative = reshape(relative, (1, heads, frames, frames), (1, 0, 2))
        softmax = net.add_softmax(add(mul(content, scale), mul(relative, scale)))
        softmax.axes = 8
        context = net.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
        ).get_output(0)
        context = reshape(context, (frames, width), (0, 2, 1, 3))
        hidden = add(hidden, linear(context, prefix + ".self_attn.linear_out"))
        name = prefix + ".conv_module"
        inner = linear(norm(hidden, name + ".layer_norm"), name + ".pointwise_conv1", True)
        first = net.add_slice(inner, (0, 0), (frames, width), (1, 1)).get_output(0)
        gate = net.add_slice(inner, (0, width), (frames, width), (1, 1)).get_output(0)
        inner = mul(first, net.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0))
        inner = reshape(inner, (1, width, 1, frames), (1, 0))
        kernel = cfg["conv_depthwise_kernel_size"]
        weight = np.ascontiguousarray(
            weights[name + ".depthwise_conv.weight"].reshape(width, 1, 1, kernel)
        )
        convolution = net.add_convolution_nd(
            inner, width, (1, kernel), weight, trt.Weights(trt.float32)
        )
        convolution.num_groups = width
        convolution.pre_padding = (0, kernel - 1)
        inner = reshape(convolution.get_output(0), (width, frames))
        inner = reshape(inner, (frames, width), (1, 0))
        inner = swish(norm(inner, name + ".depthwise_layer_norm"))
        hidden = add(hidden, linear(inner, name + ".pointwise_conv2", True))
        hidden = add(
            hidden, mul(ffn(norm(hidden, prefix + ".ffn2_layer_norm"), prefix + ".ffn2"), half)
        )
        hidden = norm(hidden, prefix + ".final_layer_norm")
    hidden.name = "features"
    net.mark_output(hidden)
    plan = builder.build_serialized_network(net, options)
    if plan is None:
        raise RuntimeError("W2v-BERT encoder engine build failed")
    writer.set_header(family="wav2vec2_bert", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", bytes(plan))
    writer.add_json(
        "runtime.json",
        dict(frames=frames, bins=cfg["feature_projection_input_dim"], hidden_size=width),
    )
