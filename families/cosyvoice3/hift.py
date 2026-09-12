# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CosyVoice3 offline CausalHiFT: Mel -> F0 -> NSF -> waveform in TensorRT.

Equations: CosyVoice 074ca6dc, CausalHiFTGenerator, SineGen2 and Snake.
Only finalize=True, B=1, 24 kHz. F0 is explicitly FP32, unlike the upstream
inference method's FP64 predictor; qualification must measure this difference.
The caller supplies uniform excitation noise, matching upstream eval/causal
SineGen2 (which uses torch.rand, NOT Gaussian noise). No PyTorch/ONNX fallback.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .components import ComponentEngine, Graph
from .config import ShapeProfile

SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 480


def conv_specs():
    """name -> (out_channels, in_channels, kernel, weight_normalized)."""
    result = {"conv_pre": (512, 80, 5, True), "conv_post": (18, 64, 7, True)}
    for i, (out, inp, kernel) in enumerate(((256, 512, 16), (128, 256, 11), (64, 128, 7))):
        result[f"ups.{i}"] = (out, inp, kernel, True)
    for i, (channels, kernel) in enumerate(((256, 30), (128, 6), (64, 1))):
        result[f"source_downs.{i}"] = (channels, 18, kernel, False)
    for i in range(5):
        result[f"f0_predictor.condnet.{2 * i}"] = (512, 80 if i == 0 else 512, 4 if i == 0 else 3, True)
    for prefix, channels, kernel in resblock_specs():
        for group in (1, 2):
            for i in range(3):
                result[f"{prefix}.convs{group}.{i}"] = (channels, channels, kernel, True)
    return result


def resblock_specs():
    result = [(f"source_resblocks.{i}", ch, k) for i, (ch, k) in enumerate(((256, 7), (128, 7), (64, 11)))]
    result += [(f"resblocks.{i * 3 + j}", ch, k) for i, ch in enumerate((256, 128, 64))
               for j, k in enumerate((3, 7, 11))]
    return result


def weight_shapes(*, checkpoint=False):
    result = {"m_source.l_linear.weight": (1, 9), "m_source.l_linear.bias": (1,),
              "f0_predictor.classifier.weight": (1, 512), "f0_predictor.classifier.bias": (1,)}
    for name, (out, inp, k, normalized) in conv_specs().items():
        result[name + ".bias"] = (out,)
        if checkpoint and normalized:
            result[name + ".parametrizations.weight.original0"] = (out, 1, 1)
            result[name + ".parametrizations.weight.original1"] = (out, inp, k)
        else:
            result[name + ".weight"] = (out, inp, k)
    for prefix, channels, _ in resblock_specs():
        for group in (1, 2):
            for i in range(3):
                result[f"{prefix}.activations{group}.{i}.alpha"] = (channels,)
    return result


def validate_weights(weights):
    shapes = weight_shapes()
    if set(weights) != set(shapes):
        raise ValueError("Unexpected HiFT parameter keys")
    for key, shape in shapes.items():
        a = weights[key]
        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f"Invalid HiFT weight {key}, expected FP32 {shape}")


def load_weights(model_dir):
    import torch
    import yaml
    from .config import read_config

    read_config(model_dir)
    raw = yaml.load((Path(model_dir) / "cosyvoice3.yaml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)["hift"]
    expected = {"in_channels": "80", "base_channels": "512", "nb_harmonics": "8", "sampling_rate": "24000",
                "nsf_alpha": "0.1", "nsf_sigma": "0.003", "nsf_voiced_threshold": "10",
                "upsample_rates": ["8", "5", "3"], "upsample_kernel_sizes": ["16", "11", "7"],
                "istft_params": {"n_fft": "16", "hop_len": "4"}, "lrelu_slope": "0.1", "audio_limit": "0.99",
                "conv_pre_look_right": "4", "resblock_kernel_sizes": ["3", "7", "11"],
                "resblock_dilation_sizes": [["1", "3", "5"]] * 3,
                "source_resblock_kernel_sizes": ["7", "7", "11"],
                "source_resblock_dilation_sizes": [["1", "3", "5"]] * 3,
                "f0_predictor": {"num_class": "1", "in_channels": "80", "cond_channels": "512"}}
    if raw.get("sampling_rate") == "<sample_rate>":
        raw["sampling_rate"] = "24000"  # read_config verified the referenced root value.
    if raw != expected:
        raise ValueError("Only the published CosyVoice3 causal HiFT topology is supported")
    state = torch.load(Path(model_dir) / "hift.pt", weights_only=True, mmap=True, map_location="cpu")
    shapes = weight_shapes(checkpoint=True)
    if set(state) != set(shapes):
        raise ValueError("Unexpected HiFT checkpoint keys")
    for key, shape in shapes.items():
        if tuple(state[key].shape) != shape or not state[key].is_floating_point() or not torch.isfinite(state[key]).all():
            raise ValueError(f"Invalid HiFT checkpoint tensor: {key}")
    weights = {k: v.float().numpy() for k, v in state.items() if ".parametrizations." not in k}
    for name, (_, _, _, normalized) in conv_specs().items():
        if not normalized:
            continue
        # Fold weight parametrization at build time, never at inference time.
        v = state[name + ".parametrizations.weight.original1"]
        scale = state[name + ".parametrizations.weight.original0"]
        dtype = torch.float64 if name.startswith("f0_predictor.") else torch.float32
        if (v.to(dtype).square().sum(dim=(1, 2)) == 0).any():
            raise ValueError(f"Zero weight-norm denominator: {name}")
        weights[name + ".weight"] = torch._weight_norm(v.to(dtype), scale.to(dtype), 0).float().numpy()
    validate_weights(weights)
    return weights


def fourier_filters():
    """16-point periodic Hann STFT and real one-sided inverse transform."""
    n = np.arange(16, dtype=np.float64)
    k = np.arange(9, dtype=np.float64)[:, None]
    window = (.5 - .5 * np.cos(2 * np.pi * n / 16)).astype(np.float32)
    cos, sin = np.cos(2 * np.pi * k * n / 16), np.sin(2 * np.pi * k * n / 16)
    sin[[0, 8]] = 0
    stft = np.concatenate((cos, -sin), axis=0).astype(np.float32)[:, None] * window
    multiplicity = np.array([1] + [2] * 7 + [1], np.float64)[:, None]
    inverse = np.concatenate((cos * multiplicity / 16, -sin * multiplicity / 16), axis=0)
    inverse = inverse.astype(np.float32)[:, None] * window
    return stft, inverse, window


def add_source(g, f0, noise, weights):
    """Causal SineGen2; expose boundaries for maintained, same-F0 unit tests."""
    t, net = g.trt, g.net
    harmonics = g.ew(f0, g.const(np.arange(1, 10, dtype=np.float32)[None, :, None]), "PROD")
    rad = g.ew(harmonics, g.scalar(24000, 3), "DIV")
    rad = g.ew(rad, g.unary(rad, "FLOOR"), "SUB")
    # Preserve the chronological FP32 phase recurrence of the declared causal
    # reference. A parallel prefix sum can round differently, which becomes
    # a waveform mismatch after phase amplification and decoding.
    # This is a model-local recurrence; TensorRT still owns GPU execution.
    loop = net.add_loop()
    count = g.reshape(g.dim(rad, 2), ())
    loop.add_trip_limit(count, t.TripLimit.COUNT)
    current = loop.add_iterator(rad, 2, False).get_output(0)
    state = loop.add_recurrence(g.const(np.zeros((1, 9), np.float32)))
    update = g.ew(state.get_output(0), current, "SUM")
    state.set_input(1, update)
    collected = loop.add_loop_output(update, t.LoopOutput.CONCATENATE, 2)
    collected.set_input(1, count)
    phase = collected.get_output(0)
    cumulative = phase
    phase = g.ew(g.ew(phase, g.scalar(2 * np.pi, 3), "PROD"), g.scalar(480, 3), "PROD")
    sine = g.ew(g.unary(g.repeat(phase, 480), "SIN"), g.scalar(.1, 3), "PROD")
    voiced = g.ew(f0, g.scalar(10, 3), "GREATER")
    voiced = net.add_cast(voiced, t.float32).get_output(0)
    voiced = g.repeat(voiced, 480)
    amplitude = g.ew(g.ew(voiced, g.scalar(.003, 3), "PROD"),
                     g.ew(g.ew(g.scalar(1, 3), voiced, "SUB"), g.scalar(.1 / 3, 3), "PROD"), "SUM")
    excitation = g.ew(g.ew(sine, voiced, "PROD"), g.ew(noise, amplitude, "PROD"), "SUM")
    source = g.activation(g.conv(excitation, weights["m_source.l_linear.weight"][..., None],
                                weights["m_source.l_linear.bias"]), "TANH")
    return {"source": source, "rad": rad, "cumulative": cumulative, "phase": phase}


def build_engine(weights, profile=ShapeProfile(), *, workspace_mib=256):
    validate_weights(weights)
    g = Graph()
    t, net = g.trt, g.net
    mel = net.add_input("mel", t.float32, (1, 80, -1))
    noise = net.add_input("noise", t.float32, (1, 9, -1))

    def conv(x, key, *, right=False, stride=1, dilation=1, left=None):
        w = weights[key + ".weight"]
        padding = (w.shape[2] - 1) * dilation
        if key.startswith("f0_predictor.condnet."):
            # The official predictor uses FP64. Short channel reductions and a
            # balanced sum reduce FP32 accumulation error without target-specific
            # kernels, altered weights, or a CPU inference fallback.
            terms = []
            for start in range(0, w.shape[1], 16):
                indices = g.const(np.arange(start, min(start + 16, w.shape[1]), dtype=np.int32))
                block = g.gather(x, indices, 1)
                terms.append(g.conv(block, np.ascontiguousarray(w[:, start:start + 16]),
                                    left=0 if right else padding, right=padding if right else 0))
            while len(terms) > 1:
                terms = [g.ew(terms[i], terms[i + 1], "SUM") if i + 1 < len(terms) else terms[i]
                         for i in range(0, len(terms), 2)]
            return g.ew(terms[0], g.const(weights[key + ".bias"][None, :, None]), "SUM")
        return g.conv(x, w, weights[key + ".bias"], left=left if left is not None else (0 if right else padding),
                      right=padding if right else 0, stride=stride, dilation=dilation)

    def snake(x, key):
        alpha = weights[key + ".alpha"][None, :, None]
        sine = g.unary(g.ew(x, g.const(alpha), "PROD"), "SIN")
        residual = g.ew(g.ew(sine, sine, "PROD"), g.const((1 / (alpha + np.float32(1e-9))).astype(np.float32)), "PROD")
        return g.ew(x, residual, "SUM")

    def resblock(x, key):
        for j, dilation in enumerate((1, 3, 5)):
            a = conv(snake(x, f"{key}.activations1.{j}"), f"{key}.convs1.{j}", dilation=dilation)
            a = conv(snake(a, f"{key}.activations2.{j}"), f"{key}.convs2.{j}")
            x = g.ew(x, a, "SUM")
        return x

    f0 = mel
    for i in range(5):
        f0 = g.activation(conv(f0, f"f0_predictor.condnet.{i * 2}", right=i == 0), "ELU", 1.)
    f0 = g.conv(f0, weights["f0_predictor.classifier.weight"][..., None], weights["f0_predictor.classifier.bias"])
    f0 = g.unary(f0, "ABS")
    g.mark(f0, "f0")
    # SineGen2 first repeats F0 480x then linearly downsamples by 480.
    # Each interpolation pair is the same repeated F0. The random initial phase
    # at sample zero is not selected by half-pixel downsampling (239.5).
    source = add_source(g, f0, noise, weights)["source"]
    g.mark(source, "source")
    stft_w, inverse_w, window = fourier_filters()
    left = g.gather(source, g.const(np.arange(8, 0, -1, dtype=np.int32)), 2)
    right_ids = g.ew(g.dim(source, 2), g.const(np.arange(-2, -10, -1, dtype=np.int32)), "SUM")
    padded = g.cat([left, source, g.gather(source, right_ids, 2)], 2)
    spectrum = g.conv(padded, stft_w, stride=4)
    x = conv(mel, "conv_pre", right=True)
    for i, factor in enumerate((8, 5, 3)):
        x = conv(g.repeat(g.activation(x, "LEAKY_RELU", .1), factor), f"ups.{i}")
        if i == 2:
            x = g.cat([g.gather(x, g.const([1], np.int32), 2), x], 2)
        stride = (15, 3, 1)[i]
        si = conv(spectrum, f"source_downs.{i}", stride=stride, left=stride - 1)
        x = g.ew(x, resblock(si, f"source_resblocks.{i}"), "SUM")
        a, b, c = [resblock(x, f"resblocks.{i * 3 + j}") for j in range(3)]
        x = g.ew(g.ew(g.ew(a, b, "SUM"), c, "SUM"), g.scalar(3, 3), "DIV")
    # Upstream's FINAL leaky_relu has its default slope .01, not .1.
    x = conv(g.activation(x, "LEAKY_RELU", .01), "conv_post")
    magnitude = g.unary(g.gather(x, g.const(np.arange(9, dtype=np.int32)), 1), "EXP")
    magnitude = g.ew(magnitude, g.scalar(100, 3), "MIN")
    angle = g.unary(g.gather(x, g.const(np.arange(9, 18, dtype=np.int32)), 1), "SIN")
    spectrum = g.cat([g.ew(magnitude, g.unary(angle, op), "PROD") for op in ("COS", "SIN")], 1)
    audio = g.conv(spectrum, inverse_w, stride=4, transpose=True)
    ones = g.ew(g.ew(g.gather(magnitude, g.const([0], np.int32), 1), g.scalar(0, 3), "PROD"), g.scalar(1, 3), "SUM")
    denominator = g.conv(ones, (window * window)[None, None], stride=4, transpose=True)
    # Center=True inverse STFT: trim 8 samples from each end BEFORE division;
    # Hann endpoints are zero outside the retained signal.
    length = g.ew(g.dim(audio, 2), g.const([16], np.int32), "SUB")

    def crop(a):
        layer = net.add_slice(a, (0, 0, 8), (1, 1, 1), (1, 1, 1))
        layer.set_input(2, g.shape(1, 1, length))
        return layer.get_output(0)

    audio = g.ew(crop(audio), crop(denominator), "DIV")
    audio = g.ew(g.ew(audio, g.scalar(-.99, 3), "MAX"), g.scalar(.99, 3), "MIN")
    g.mark(g.reshape(audio, (1, -1)), "audio")
    frames = tuple(asdict(profile).values())
    return g.build({"mel": [(1, 80, n) for n in frames],
                    "noise": [(1, 9, n * 480) for n in frames]}, workspace_mib)


class HiFTEngine(ComponentEngine):
    def __init__(self, directory, *, device=0):
        super().__init__(directory, "hift", {"mel": "float32", "noise": "float32"},
                         {"f0": "float32", "source": "float32", "audio": "float32"}, device=device)
        self.profile = ShapeProfile(**self.manifest["profile"])
        if self.manifest.get("sample_rate") != 24000 or self.manifest.get("finalize") is not True:
            raise ValueError("Unsupported HiFT output/finalization contract")

    def __call__(self, mel, *, noise, finalize=True):
        if not finalize:
            raise ValueError("This HiFT engine is offline-only (finalize=True)")
        if mel.ndim != 3 or mel.shape[:2] != (1, 80):
            raise ValueError("mel must have shape [1, 80, frames]")
        self.profile.validate_frames(mel.shape[2])
        if noise.shape != (1, 9, mel.shape[2] * 480):
            raise ValueError("noise must have shape [1, 9, frames * 480]")
        if (noise < 0).any().item() or (noise >= 1).any().item():
            raise ValueError("Causal HiFT requires uniform excitation noise in [0, 1)")
        return self.run(mel=mel, noise=noise)
