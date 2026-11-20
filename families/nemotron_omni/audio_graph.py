# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic, single-clip Parakeet encoder and speech-to-language projector.

The native runtime supplies the actual unpadded mel sequence and matching
relative positions. Separate clips are encoded separately; no padding tokens
enter convolutions, and no waveform is silently truncated or substituted.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .component_graph import _Graph


def encoded_length(length):
    """Three symmetric stride-two kernel-three convolution stages."""
    for _ in range(3):
        length = (length + 1) // 2
    return length


def relative_inputs(length, hidden_size, heads):
    """Independent sinusoidal positions and relative-shift gather indices."""
    positions = np.arange(length - 1, -length, -1, dtype=np.float32)
    frequencies = np.float32(1) / np.power(
        np.float32(10000), np.arange(0, hidden_size, 2, dtype=np.float32) / hidden_size
    )
    angles = positions[:, None] * frequencies
    values = np.stack((np.sin(angles), np.cos(angles)), axis=-1).reshape(-1, hidden_size)
    indices = length - 1 - np.arange(length)[:, None] + np.arange(length)[None, :]
    return np.ascontiguousarray(values), np.repeat(indices[None], heads, axis=0).astype(np.int32)


class _AudioGraph(_Graph):
    def permute(self, value, order, shape):
        layer = self.network.add_shuffle(value)
        layer.first_transpose = order
        layer.reshape_dims = shape
        return layer.get_output(0)

    def silu(self, value):
        sigmoid = self.network.add_activation(value, trt.ActivationType.SIGMOID).get_output(0)
        return self.mul(value, sigmoid)

    def convolution(self, value, weights, prefix, *, stride=(1, 1), padding=(0, 0), groups=1):
        dtype = np.float16 if self.dtype == trt.float16 else np.float32
        kernel = np.ascontiguousarray(weights[prefix + ".weight"], dtype=dtype)
        if kernel.ndim == 3:
            kernel = kernel[:, :, None, :]
        self.constants.append(kernel)
        bias = trt.Weights()
        if prefix + ".bias" in weights:
            bias = np.ascontiguousarray(weights[prefix + ".bias"], dtype=dtype)
            self.constants.append(bias)
        layer = self.network.add_convolution_nd(
            value, kernel.shape[0], tuple(kernel.shape[2:]), kernel, bias
        )
        if layer is None:
            raise RuntimeError(f"TensorRT rejected Omni convolution {prefix}")
        layer.stride_nd, layer.padding_nd, layer.num_groups = stride, padding, groups
        return layer.get_output(0)


def build_audio(config, weights, max_time_steps, *, precision="fp32", verbose=False):
    """Build full dynamic encoder plus RMSNorm/ReLU-squared language projector."""
    if type(max_time_steps) is not int or max_time_steps < 8:
        raise ValueError("Omni acoustic capacity must be at least eight mel frames")
    audio = config.audio
    if (
        audio.get("subsampling_factor", 8) != 8
        or audio.get("subsampling_conv_stride", 2) != 2
        or audio.get("subsampling_conv_kernel_size", 3) != 3
    ):
        raise ValueError("Omni acoustic frontend requires its three stride-two kernel-three stages")
    if precision not in {"fp32", "fp16"}:
        raise ValueError("Omni audio precision must be fp32 or fp16")
    graph = _AudioGraph(precision, verbose)
    net = graph.network
    hidden, heads = int(audio["hidden_size"]), int(audio["num_attention_heads"])
    mel_bins = int(audio["num_mel_bins"])
    if hidden % heads or hidden % 2 or mel_bins % 8:
        raise ValueError("Invalid Omni acoustic hidden/head/mel dimensions")
    dim = hidden // heads
    root = "sound_encoder.encoder"
    mel = net.add_input("mel_features", graph.dtype, (-1, mel_bins))
    optimum = min(max_time_steps, 512)
    graph.profile.set_shape(
        "mel_features", (2, mel_bins), (optimum, mel_bins), (max_time_steps, mel_bins)
    )
    maximum, opt = encoded_length(max_time_steps), encoded_length(optimum)
    positions = net.add_input("relative_positions", graph.dtype, (-1, hidden))
    graph.profile.set_shape(
        "relative_positions", (1, hidden), (2 * opt - 1, hidden), (2 * maximum - 1, hidden)
    )
    indices = net.add_input("relative_indices", trt.int32, (heads, -1, -1))
    graph.profile.set_shape(
        "relative_indices", (heads, 1, 1), (heads, opt, opt), (heads, maximum, maximum)
    )
    indices.set_dimension_name(1, "audio_tokens")
    indices.set_dimension_name(2, "audio_tokens")
    state = graph.reshape(mel, (1, 1, -1, mel_bins))
    channels = int(audio["subsampling_conv_channels"])
    for index in (0, 2, 3, 5, 6):
        spatial = index in (0, 2, 5)
        state = graph.convolution(
            state,
            weights,
            f"{root}.subsampling.layers.{index}",
            stride=(2, 2) if spatial else (1, 1),
            padding=(1, 1) if spatial else (0, 0),
            groups=channels if index in (2, 5) else 1,
        )
        if index in (0, 3, 6):
            state = net.add_activation(state, trt.ActivationType.RELU).get_output(0)
    state = graph.permute(state, (0, 2, 1, 3), (-1, channels * (mel_bins // 8)))
    state = graph.linear(state, weights, root + ".subsampling.linear", bias=True)
    if audio.get("scale_input", False):
        state = graph.mul(state, graph.scalar(hidden**0.5, 2))
    attention_bias = bool(audio.get("attention_bias", False))

    def half_ffn(value, prefix):
        value = graph.linear(value, weights, prefix + ".linear1", attention_bias)
        value = graph.silu(value)
        value = graph.linear(value, weights, prefix + ".linear2", attention_bias)
        return graph.mul(value, graph.scalar(0.5, 2))

    def heads_view(value):
        return graph.reshape(value, (-1, heads, dim), (1, 0, 2))

    def matmul(left, right, transpose=False):
        return net.add_matrix_multiply(
            left,
            trt.MatrixOperation.NONE,
            right,
            trt.MatrixOperation.TRANSPOSE if transpose else trt.MatrixOperation.NONE,
        ).get_output(0)

    for layer in range(int(audio["num_hidden_layers"])):
        prefix = f"{root}.layers.{layer}"
        normalized = graph.norm(state, weights, prefix + ".norm_feed_forward1", 1e-5, centered=True)
        state = graph.add(state, half_ffn(normalized, prefix + ".feed_forward1"))
        normalized = graph.norm(state, weights, prefix + ".norm_self_att", 1e-5, centered=True)
        attention = prefix + ".self_attn"
        q, k, v = [
            heads_view(
                graph.linear(normalized, weights, attention + "." + part + "_proj", attention_bias)
            )
            for part in ("q", "k", "v")
        ]
        relative = heads_view(graph.linear(positions, weights, attention + ".relative_k_proj"))
        u = graph.constant(weights[attention + ".bias_u"].reshape(heads, 1, dim))
        v_bias = graph.constant(weights[attention + ".bias_v"].reshape(heads, 1, dim))
        content = matmul(graph.add(q, u), k, transpose=True)
        displacement = matmul(graph.add(q, v_bias), relative, transpose=True)
        gather = net.add_gather(displacement, indices, 2)
        gather.mode = trt.GatherMode.ELEMENT
        scores = graph.mul(graph.add(content, gather.get_output(0)), graph.scalar(dim**-0.5, 3))
        probabilities = net.add_softmax(graph.cast(scores, trt.float32))
        probabilities.axes = 4
        attended = matmul(graph.cast(probabilities.get_output(0), graph.dtype), v)
        attended = graph.permute(attended, (1, 0, 2), (-1, hidden))
        state = graph.add(
            state, graph.linear(attended, weights, attention + ".o_proj", attention_bias)
        )
        normalized = graph.norm(state, weights, prefix + ".norm_conv", 1e-5, centered=True)
        conv = prefix + ".conv"
        x = graph.permute(normalized, (1, 0), (1, hidden, 1, -1))
        x = graph.convolution(x, weights, conv + ".pointwise_conv1")
        x = graph.permute(x, (0, 3, 2, 1), (-1, 2, hidden))
        a = net.add_gather(x, graph.constant([0], trt.int32), 1).get_output(0)
        b = net.add_gather(x, graph.constant([1], trt.int32), 1).get_output(0)
        x = graph.mul(
            graph.reshape(a, (-1, hidden)),
            net.add_activation(
                graph.reshape(b, (-1, hidden)), trt.ActivationType.SIGMOID
            ).get_output(0),
        )
        x = graph.permute(x, (1, 0), (1, hidden, 1, -1))
        x = graph.convolution(
            x,
            weights,
            conv + ".depthwise_conv",
            groups=hidden,
            padding=(0, int(audio["conv_kernel_size"]) // 2),
        )
        x = graph.permute(x, (0, 3, 2, 1), (-1, hidden))
        # Evaluation BatchNorm uses checkpoint running statistics, not sample stats.
        mean = graph.constant(weights[conv + ".norm.running_mean"][None], trt.float32)
        variance = np.asarray(weights[conv + ".norm.running_var"], np.float32)
        denominator = graph.constant(np.sqrt(variance[None] + np.float32(1e-5)), trt.float32)
        x = graph.binary(graph.cast(x, trt.float32), mean, trt.ElementWiseOperation.SUB)
        x = graph.binary(x, denominator, trt.ElementWiseOperation.DIV)
        x = graph.add(
            graph.mul(x, graph.constant(weights[conv + ".norm.weight"][None], trt.float32)),
            graph.constant(weights[conv + ".norm.bias"][None], trt.float32),
        )
        x = graph.silu(graph.cast(x, graph.dtype))
        x = graph.permute(x, (1, 0), (1, hidden, 1, -1))
        x = graph.convolution(x, weights, conv + ".pointwise_conv2")
        x = graph.permute(x, (0, 3, 2, 1), (-1, hidden))
        state = graph.add(state, x)
        normalized = graph.norm(state, weights, prefix + ".norm_feed_forward2", 1e-5, centered=True)
        state = graph.add(state, half_ffn(normalized, prefix + ".feed_forward2"))
        state = graph.norm(state, weights, prefix + ".norm_out", 1e-5, centered=True)
    state = graph.norm(state, weights, "sound_projection.norm", 1e-5)
    state = graph.linear(
        state, weights, "sound_projection.linear1", bool(audio.get("projection_bias", False))
    )
    state = net.add_activation(state, trt.ActivationType.RELU).get_output(0)
    state = graph.linear(
        graph.mul(state, state),
        weights,
        "sound_projection.linear2",
        bool(audio.get("projection_bias", False)),
    )
    graph.output(state, "sound_features", trt.float32)
    return graph.finish()
