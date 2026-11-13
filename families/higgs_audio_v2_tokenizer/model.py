# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Higgs Audio V2 residual codebooks and DAC waveform decoder."""

import json
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import graph as g


def build(request, writer):
    if request.task != "audio_codebook_tokens_to_audio" or request.backend != "trt":
        raise ValueError("Higgs Audio V2 requires audio_codebook_tokens_to_audio with backend=trt")
    if (
        request.precision != "fp32"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.dynamic_kv_cache
        or request.image_height is not None
        or request.image_width is not None
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError("Higgs Audio V2 supports one fp32 codec token matrix")
    directory = Path(request.model_dir)
    cfg = json.loads((directory / "config.json").read_text())
    if (
        cfg["model_type"] != "higgs_audio_v2_tokenizer"
        or cfg["acoustic_model_config"]["model_type"] != "dac"
    ):
        raise ValueError("Expected a Higgs Audio V2 tokenizer with a DAC decoder")
    frames = request.max_sequence_length or 50
    if not 1 <= frames <= 250:
        raise ValueError("Higgs Audio V2 supports 1 to 250 codec frames")
    from safetensors import safe_open

    with safe_open(str(directory / "model.safetensors"), framework="np") as handle:
        weights = {
            name: handle.get_tensor(name).astype(np.float32)
            for name in handle.keys()
            if name.startswith(("quantizer.", "fc2.", "acoustic_decoder."))
        }
    books = len([name for name in weights if name.endswith(".codebook.embed")])
    if not books:
        raise ValueError("Higgs Audio V2 checkpoint has no residual codebooks")
    decoder = cfg["acoustic_model_config"]
    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)

    def const(value):
        value = np.ascontiguousarray(value, dtype=np.float32)
        return net.add_constant(value.shape, value).get_output(0)

    def op(a, b, operation):
        return net.add_elementwise(a, b, operation).get_output(0)

    def add(a, b):
        return op(a, b, trt.ElementWiseOperation.SUM)

    def mul(a, b):
        return op(a, b, trt.ElementWiseOperation.PROD)

    def reshape(x, shape, transpose=None):
        layer = net.add_shuffle(x)
        if transpose is not None:
            layer.first_transpose = transpose
        layer.reshape_dims = shape
        return layer.get_output(0)

    def linear(x, name):
        return g.add_linear(net, x, weights[name + ".weight"], weights.get(name + ".bias"))

    def snake(x, name):
        alpha = weights[name + ".alpha"].reshape(1, x.shape[1], 1, 1)
        sine = net.add_unary(mul(x, const(alpha)), trt.UnaryOperation.SIN).get_output(0)
        reciprocal = np.float32(1.0) / (alpha + np.float32(1e-9))
        return add(x, mul(const(reciprocal), mul(sine, sine)))

    def conv(x, name, *, stride=1, dilation=1, transpose=False):
        weight = weights[name + ".weight"]
        kernel = weight.shape[-1]
        weight = np.ascontiguousarray(weight.reshape(*weight.shape[:2], 1, kernel))
        operation = net.add_deconvolution_nd if transpose else net.add_convolution_nd
        layer = operation(
            x,
            weight.shape[1] if transpose else weight.shape[0],
            (1, kernel),
            weight,
            weights.get(name + ".bias", trt.Weights(trt.float32)),
        )
        layer.stride_nd = (1, stride)
        if transpose:
            # HF Higgs retains DAC's ceil(stride/2) left padding and adds an
            # odd-stride output sample on the right. Asymmetric cropping gives
            # the same full convolution support without fabricating samples.
            padding = (stride + 1) // 2
            layer.pre_padding = (0, padding)
            layer.post_padding = (0, padding - stride % 2)
        else:
            layer.dilation_nd = (1, dilation)
            layer.padding_nd = (0, (kernel - 1) * dilation // 2)
        return layer.get_output(0)

    codes = net.add_input("codes", trt.int32, (books, frames))
    hidden = None
    for index in range(books):
        prefix = f"quantizer.quantizers.{index}"
        indices = net.add_slice(codes, (index, 0), (1, frames), (1, 1)).get_output(0)
        table = const(weights[prefix + ".codebook.embed"])
        quantized = net.add_gather(table, reshape(indices, (frames,)), 0).get_output(0)
        projected = linear(quantized, prefix + ".project_out")
        hidden = projected if hidden is None else add(hidden, projected)
    hidden = linear(hidden, "fc2")
    hidden = reshape(hidden, (1, decoder["hidden_size"], 1, frames), (1, 0))
    hidden = conv(hidden, "acoustic_decoder.conv1")
    for index, ratio in enumerate(decoder["upsampling_ratios"]):
        prefix = f"acoustic_decoder.block.{index}"
        hidden = conv(
            snake(hidden, prefix + ".snake1"), prefix + ".conv_t1", stride=ratio, transpose=True
        )
        for unit, dilation in enumerate((1, 3, 9), 1):
            name = prefix + f".res_unit{unit}"
            inner = conv(snake(hidden, name + ".snake1"), name + ".conv1", dilation=dilation)
            inner = conv(snake(inner, name + ".snake2"), name + ".conv2")
            hidden = add(hidden, inner)
    waveform = conv(snake(hidden, "acoustic_decoder.snake1"), "acoustic_decoder.conv2")
    # Higgs removes DAC's final tanh; preserve the published linear PCM output.
    waveform.name = "waveform"
    net.mark_output(waveform)
    plan = builder.build_serialized_network(net, options)
    if plan is None:
        raise RuntimeError("Higgs Audio V2 decoder engine build failed")
    writer.set_header(family="higgs_audio_v2_tokenizer", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", bytes(plan))
    writer.add_json(
        "runtime.json",
        dict(
            frames=frames,
            codebooks=books,
            vocabulary_size=cfg["codebook_size"],
            samples=frames * int(np.prod(decoder["upsampling_ratios"])),
            sample_rate=cfg["sample_rate"],
        ),
    )
