# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SpeechT5 log-mel normalization, HiFiGAN upsampling and residual convolutions."""

import json
from pathlib import Path

import numpy as np
import tensorrt as trt


def build(request, writer):
    if request.task != "spectrogram_to_audio" or request.backend != "trt":
        raise ValueError("SpeechT5 HiFiGAN requires spectrogram_to_audio with backend=trt")
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
        raise NotImplementedError("SpeechT5 HiFiGAN supports one fp32 spectrogram")
    directory = Path(request.model_dir)
    cfg = json.loads((directory / "config.json").read_text())
    if cfg["model_type"] != "hifigan" or cfg["architectures"] != ["SpeechT5HifiGan"]:
        raise ValueError("Expected the SpeechT5 HiFiGAN vocoder checkpoint")
    frames = request.max_sequence_length or 64
    if not 1 <= frames <= 1024:
        raise ValueError("HiFiGAN spectrogram profile must contain 1 to 1024 frames")
    import torch

    weights = {
        key: value.float().numpy()
        for key, value in torch.load(
            directory / "pytorch_model.bin", map_location="cpu", weights_only=True
        ).items()
    }
    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)

    def const(value):
        value = np.ascontiguousarray(value, dtype=np.float32)
        return net.add_constant(value.shape, value).get_output(0)

    def op(a, b, kind):
        return net.add_elementwise(a, b, kind).get_output(0)

    def leaky(x, slope):
        activation = net.add_activation(x, trt.ActivationType.LEAKY_RELU)
        activation.alpha = slope
        return activation.get_output(0)

    def conv(x, name, *, stride=1, dilation=1, transpose=False, padding=None):
        weight = weights[name + ".weight"]
        weight = np.ascontiguousarray(weight.reshape(*weight.shape[:2], 1, weight.shape[2]))
        kernel = weight.shape[-1]
        operation = net.add_deconvolution_nd if transpose else net.add_convolution_nd
        channels = weight.shape[1] if transpose else weight.shape[0]
        layer = operation(x, channels, (1, kernel), weight, weights[name + ".bias"])
        layer.stride_nd = (1, stride)
        if not transpose:
            layer.dilation_nd = (1, dilation)
        layer.padding_nd = (0, (kernel * dilation - dilation) // 2 if padding is None else padding)
        return layer.get_output(0)

    bins = cfg["model_in_dim"]
    spectrogram = net.add_input("spectrogram", trt.float32, (frames, bins))
    hidden = spectrogram
    if cfg["normalize_before"]:
        if np.any(weights["scale"] == 0):
            raise ValueError("HiFiGAN scale contains zero")
        hidden = op(
            op(hidden, const(weights["mean"].reshape(1, bins)), trt.ElementWiseOperation.SUB),
            const(weights["scale"].reshape(1, bins)),
            trt.ElementWiseOperation.DIV,
        )
    reshape = net.add_shuffle(hidden)
    reshape.first_transpose = (1, 0)
    reshape.reshape_dims = (1, bins, 1, frames)
    hidden = conv(reshape.get_output(0), "conv_pre")
    kernels = cfg["resblock_kernel_sizes"]
    for stage, (rate, kernel) in enumerate(
        zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"])
    ):
        hidden = leaky(hidden, cfg["leaky_relu_slope"])
        hidden = conv(
            hidden, f"upsampler.{stage}", stride=rate, transpose=True, padding=(kernel - rate) // 2
        )
        outputs = []
        for block, dilations in enumerate(cfg["resblock_dilation_sizes"]):
            name = f"resblocks.{stage * len(kernels) + block}"
            current = hidden
            for index, dilation in enumerate(dilations):
                inner = conv(
                    leaky(current, cfg["leaky_relu_slope"]),
                    f"{name}.convs1.{index}",
                    dilation=dilation,
                )
                inner = conv(leaky(inner, cfg["leaky_relu_slope"]), f"{name}.convs2.{index}")
                current = op(current, inner, trt.ElementWiseOperation.SUM)
            outputs.append(current)
        hidden = outputs[0]
        for output in outputs[1:]:
            hidden = op(hidden, output, trt.ElementWiseOperation.SUM)
        hidden = op(
            hidden, const(np.full((1, 1, 1, 1), len(kernels))), trt.ElementWiseOperation.DIV
        )
    # The published implementation deliberately uses PyTorch's default 0.01
    # slope for the final activation, rather than the residual blocks' 0.1.
    hidden = conv(leaky(hidden, 0.01), "conv_post")
    waveform = net.add_activation(hidden, trt.ActivationType.TANH).get_output(0)
    waveform.name = "waveform"
    net.mark_output(waveform)
    plan = builder.build_serialized_network(net, options)
    if plan is None:
        raise RuntimeError("SpeechT5 HiFiGAN engine build failed")
    writer.set_header(family="speecht5_hifigan", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", bytes(plan))
    writer.add_json(
        "runtime.json",
        dict(
            frames=frames,
            bins=bins,
            samples=frames * int(np.prod(cfg["upsample_rates"])),
            sample_rate=cfg["sampling_rate"],
        ),
    )
