# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mimi split residual codebooks, causal transformer and SEANet waveform decoder."""

import json
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import graph as g


def build(request, writer):
    if request.dynamic_kv_cache:
        raise NotImplementedError("mimi does not support dynamic_kv_cache")
    if request.task != "audio_codebook_tokens_to_audio" or request.backend != "trt":
        raise ValueError("Mimi requires audio_codebook_tokens_to_audio with backend=trt")
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
        raise NotImplementedError("Mimi supports one fp32 codec token matrix")
    directory = Path(request.model_dir)
    cfg = json.loads((directory / "config.json").read_text())
    if (
        cfg["model_type"] != "mimi"
        or cfg["architectures"] != ["MimiModel"]
        or not cfg["use_causal_conv"]
        or cfg["pad_mode"] != "constant"
        or cfg["trim_right_ratio"] != 1
        or cfg["use_conv_shortcut"]
        or cfg["hidden_act"] != "gelu"
        or cfg.get("attention_bias", False)
        or cfg["num_attention_heads"] != cfg["num_key_value_heads"]
    ):
        raise ValueError("Unsupported Mimi causal decoder configuration")
    frames = request.max_sequence_length or 25
    if not 1 <= frames <= 125:
        raise ValueError("Mimi decode profile must contain 1 to 125 codec frames")
    from safetensors import safe_open
    import torch

    with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as handle:
        weights = {
            name: handle.get_tensor(name).float().numpy()
            for name in handle.keys()
            if name.startswith(("quantizer.", "upsample.", "decoder_transformer.", "decoder."))
        }
    books = cfg.get("num_quantizers", 32)
    semantic = cfg["num_semantic_quantizers"]
    if not 0 < semantic < books:
        raise ValueError("Mimi requires semantic and acoustic codebooks")
    width, heads, dim = cfg["hidden_size"], cfg["num_attention_heads"], cfg["head_dim"]
    if width != heads * dim or dim % 2:
        raise ValueError("Invalid Mimi attention dimensions")
    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)

    def const(value, dtype=np.float32):
        value = np.ascontiguousarray(value, dtype=dtype)
        return net.add_constant(value.shape, value).get_output(0)

    def reshape(x, shape, transpose=None):
        layer = net.add_shuffle(x)
        if transpose is not None:
            layer.first_transpose = transpose
        layer.reshape_dims = shape
        return layer.get_output(0)

    def add(a, b):
        return net.add_elementwise(a, b, trt.ElementWiseOperation.SUM).get_output(0)

    def mul(a, b):
        return net.add_elementwise(a, b, trt.ElementWiseOperation.PROD).get_output(0)

    def linear(x, name):
        return g.add_linear(net, x, weights[name + ".weight"], weights.get(name + ".bias"))

    def norm(x, name):
        return g.add_layer_norm(
            net, x, weights[name + ".weight"], weights[name + ".bias"], cfg["norm_eps"]
        )

    def conv(x, name, *, transpose=False, stride=1, groups=1):
        weight = weights[name + ".weight"]
        kernel = weight.shape[-1]
        output_channels = weight.shape[1] * groups if transpose else weight.shape[0]
        weight = np.ascontiguousarray(weight.reshape(*weight.shape[:2], 1, kernel))
        operation = net.add_deconvolution_nd if transpose else net.add_convolution_nd
        layer = operation(
            x,
            output_channels,
            (1, kernel),
            weight,
            weights.get(name + ".bias", trt.Weights(trt.float32)),
        )
        layer.stride_nd = (1, stride)
        layer.num_groups = groups
        if transpose:
            layer.post_padding = (0, kernel - stride)
        else:
            layer.pre_padding = (0, kernel - 1)
        return layer.get_output(0)

    def elu(x):
        return net.add_activation(x, trt.ActivationType.ELU).get_output(0)

    codes = net.add_input("codes", trt.int32, (books, frames))
    decoded = []
    for start, count, branch in [
        (0, semantic, "semantic"),
        (semantic, books - semantic, "acoustic"),
    ]:
        prefix = f"quantizer.{branch}_residual_vector_quantizer"
        hidden = None
        for index in range(count):
            name = f"{prefix}.layers.{index}.codebook"
            table = (
                weights[name + ".embed_sum"]
                / np.maximum(weights[name + ".cluster_usage"], np.float32(1e-5))[:, None]
            )
            indices = net.add_slice(codes, (start + index, 0), (1, frames), (1, 1)).get_output(0)
            quantized = net.add_gather(const(table), reshape(indices, (frames,)), 0).get_output(0)
            hidden = quantized if hidden is None else add(hidden, quantized)
        projection = weights.get(prefix + ".output_proj.weight")
        if projection is not None:
            hidden = g.add_linear(net, hidden, projection[:, :, 0])
        decoded.append(hidden)
    hidden = reshape(add(*decoded), (1, width, 1, frames), (1, 0))
    hidden = conv(hidden, "upsample.conv", transpose=True, stride=2, groups=cfg["upsample_groups"])
    length = frames * 2
    hidden = reshape(hidden, (width, length))
    hidden = reshape(hidden, (length, width), (1, 0))
    theta = cfg.get("rope_parameters", {}).get("rope_theta", cfg.get("rope_theta", 10000.0))
    inverse = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    frequency = torch.arange(length, dtype=torch.float32)[:, None] * inverse
    angle = torch.cat((frequency, frequency), dim=-1)
    cosine, sine = [
        const(value.numpy().reshape(1, 1, length, dim)) for value in (angle.cos(), angle.sin())
    ]

    def split_heads(x):
        layer = net.add_shuffle(x)
        layer.reshape_dims = (1, length, heads, dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    def rope(x):
        first = net.add_slice(
            x, (0, 0, 0, 0), (1, heads, length, dim // 2), (1, 1, 1, 1)
        ).get_output(0)
        second = net.add_slice(
            x, (0, 0, 0, dim // 2), (1, heads, length, dim // 2), (1, 1, 1, 1)
        ).get_output(0)
        negative = net.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
        joined = net.add_concatenation([negative, first])
        joined.axis = 3
        return add(mul(x, cosine), mul(joined.get_output(0), sine))

    positions = np.arange(length)
    distance = positions[:, None] - positions[None, :]
    mask = np.where((distance >= 0) & (distance < cfg["sliding_window"]), 0.0, -np.inf).astype(
        np.float32
    )
    for index in range(cfg["num_hidden_layers"]):
        prefix = f"decoder_transformer.layers.{index}"
        normalized = norm(hidden, prefix + ".input_layernorm")
        query, key, value = [
            split_heads(linear(normalized, prefix + ".self_attn." + name + "_proj"))
            for name in ("q", "k", "v")
        ]
        scores = net.add_matrix_multiply(
            rope(query), trt.MatrixOperation.NONE, rope(key), trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = add(
            mul(scores, const(np.full((1, 1, 1, 1), dim**-0.5))),
            const(mask.reshape(1, 1, length, length)),
        )
        softmax = net.add_softmax(scores)
        softmax.axes = 8
        context = net.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
        ).get_output(0)
        context = reshape(context, (length, width), (0, 2, 1, 3))
        attention = linear(context, prefix + ".self_attn.o_proj")
        hidden = add(
            hidden,
            mul(
                attention, const(weights[prefix + ".self_attn_layer_scale.scale"].reshape(1, width))
            ),
        )
        inner = linear(norm(hidden, prefix + ".post_attention_layernorm"), prefix + ".mlp.fc1")
        inner = net.add_activation(inner, trt.ActivationType.GELU_ERF).get_output(0)
        inner = linear(inner, prefix + ".mlp.fc2")
        hidden = add(
            hidden, mul(inner, const(weights[prefix + ".mlp_layer_scale.scale"].reshape(1, width)))
        )
    hidden = reshape(hidden, (1, width, 1, length), (1, 0))
    hidden = conv(hidden, "decoder.layers.0.conv")
    module_index = 1
    for ratio in cfg["upsampling_ratios"]:
        hidden = conv(
            elu(hidden), f"decoder.layers.{module_index + 1}.conv", transpose=True, stride=ratio
        )
        for block in range(cfg["num_residual_layers"]):
            if cfg["dilation_growth_rate"] ** block != 1:
                raise ValueError("This Mimi decoder profile requires unit residual dilation")
            prefix = f"decoder.layers.{module_index + 2 + block}.block"
            inner = conv(elu(hidden), prefix + ".1.conv")
            inner = conv(elu(inner), prefix + ".3.conv")
            hidden = add(hidden, inner)
        module_index += 2 + cfg["num_residual_layers"]
    waveform = conv(elu(hidden), f"decoder.layers.{module_index + 1}.conv")
    waveform.name = "waveform"
    net.mark_output(waveform)
    plan = builder.build_serialized_network(net, options)
    if plan is None:
        raise RuntimeError("Mimi decoder engine build failed")
    writer.set_header(family="mimi", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", bytes(plan))
    writer.add_json(
        "runtime.json",
        dict(
            frames=frames,
            codebooks=books,
            vocabulary_size=cfg["codebook_size"],
            samples=length * int(np.prod(cfg["upsampling_ratios"])),
            sample_rate=cfg["sampling_rate"],
        ),
    )
