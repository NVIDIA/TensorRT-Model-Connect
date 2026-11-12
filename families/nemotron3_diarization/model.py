# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a Nemotron 3 Diarization bundle.

Two engines. ``embedder.plan`` projects stacked mel frames to the encoder width;
``encoder.plan`` runs one step of the offline chunk loop: the speaker cache,
the FIFO queue and one chunk with its look-ahead, all already in the encoder
input space, through the 31-layer RoPE transformer and the speaker head, and
returns per-10 ms speaker logits for every row of the step. The speaker cache
lives between the two in the runtime, because it stores encoder *inputs*
selected by the head's own scores, so a step's input depends on every earlier
step's output.

The checkpoint is float32 and is built in float32. The cache keeps the frames
it ranks highest, so a precision that moved the ranking would move which
frames the later chunks attend to, not only the logits of one chunk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module
from . import frontend, rotary
from . import graph as g

_FAMILY = "nemotron3_diarization"
_TASK = "audio_to_timed_labels"
_ENCODER = "model.audio_tower"


def _load(model_dir: Path):
    from safetensors import safe_open

    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Required safetensors checkpoint is missing: {path}")
    with safe_open(str(path), framework="numpy") as handle:
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
        weights = {key: np.array(handle.get_tensor(key), dtype=np.float32, copy=True)
                   for key in handle.keys()}
    return weights, shapes


def _builder(verbose: bool):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 would round every matmul and convolution input to 10 mantissa bits.
    config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def _serialize(builder, network, config, what: str) -> bytes:
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"{_FAMILY} {what} engine build failed")
    return bytes(plan)


def build_embedder(weights: dict, cfg: dict, *, verbose: bool = False) -> bytes:
    """Stacked mel frames ``[frames, factor * mel]`` to encoder inputs ``[frames, width]``."""
    builder, network, config = _builder(verbose)
    stacked = cfg["subsampling_factor"] * cfg["num_mel_bins"]
    features = network.add_input("stacked_features", trt.float32, (-1, stacked))
    embeds = g.add_linear(network, features, weights[f"{_ENCODER}.embedder.projection.weight"])
    embeds.name = "inputs_embeds"
    network.mark_output(embeds)
    profile = builder.create_optimization_profile()
    rows = cfg["max_step_rows"]
    profile.set_shape("stacked_features", (1, stacked), (rows, stacked), (rows, stacked))
    config.add_optimization_profile(profile)
    return _serialize(builder, network, config, "embedder")


def _encoder_layers(network, hidden, weights, cfg, rope, key_bias):
    eps = cfg["layer_norm_eps"]
    for index in range(cfg["num_layers"]):
        prefix = f"{_ENCODER}.layers.{index}"
        normed = g.add_layer_norm(network, hidden, weights[f"{prefix}.layer_norm1.weight"],
                                  weights[f"{prefix}.layer_norm1.bias"], eps)
        attention = g.add_self_attention(network, normed, weights, f"{prefix}.self_attn", cfg,
                                         rope, key_bias)
        hidden = g.add_sum(network, hidden, attention)
        normed = g.add_layer_norm(network, hidden, weights[f"{prefix}.layer_norm2.weight"],
                                  weights[f"{prefix}.layer_norm2.bias"], eps)
        inner = g.add_linear(network, normed, weights[f"{prefix}.mlp.fc1.weight"],
                             weights[f"{prefix}.mlp.fc1.bias"])
        # hidden_act "gelu" is the exact erf GELU, not the tanh approximation.
        inner = g.add_activation(network, inner, trt.ActivationType.GELU_ERF)
        inner = g.add_linear(network, inner, weights[f"{prefix}.mlp.fc2.weight"],
                             weights[f"{prefix}.mlp.fc2.bias"])
        hidden = g.add_sum(network, hidden, inner)
    return hidden


def _speaker_head(network, hidden, weights, cfg):
    projected = g.add_linear(network, hidden, weights["model.proj.weight"],
                             weights["model.proj.bias"])
    frames = g.add_subpixel_upsampler(network, projected, weights["model.upsampler.conv.weight"],
                                      weights["model.upsampler.conv.bias"],
                                      cfg["subsampling_factor"])
    relu = trt.ActivationType.RELU
    dense = g.add_linear(network, g.add_activation(network, frames, relu),
                         weights["classifier.dense.weight"], weights["classifier.dense.bias"])
    return g.add_linear(network, g.add_activation(network, dense, relu),
                        weights["classifier.out_proj.weight"], weights["classifier.out_proj.bias"])


def build_encoder(weights: dict, cfg: dict, *, verbose: bool = False) -> bytes:
    """One chunk step: ``[rows, width]`` inputs and ``[rows]`` key bias to ``[rows * factor, speakers]``."""
    builder, network, config = _builder(verbose)
    width = cfg["hidden_size"]
    embeds = network.add_input("inputs_embeds", trt.float32, (-1, width))
    key_bias = network.add_input("key_bias", trt.float32, (-1,))
    eps = cfg["layer_norm_eps"]
    hidden = g.add_layer_norm(network, embeds, weights[f"{_ENCODER}.input_layer_norm.weight"],
                              weights[f"{_ENCODER}.input_layer_norm.bias"], eps)
    rope = g.rotary_tables(network, embeds, *rotary.rotary_tables(
        cfg["head_dim"], cfg["rope_theta"], cfg["max_step_rows"]))
    bias = g.reshape(network, key_bias, (1, 1, -1))
    hidden = _encoder_layers(network, hidden, weights, cfg, rope, bias)
    hidden = g.add_layer_norm(network, hidden, weights[f"{_ENCODER}.layer_norm.weight"],
                              weights[f"{_ENCODER}.layer_norm.bias"], eps)
    logits = _speaker_head(network, hidden, weights, cfg)
    logits.name = "logits"
    network.mark_output(logits)
    profile = builder.create_optimization_profile()
    rows = cfg["max_step_rows"]
    profile.set_shape("inputs_embeds", (1, width), (rows, width), (rows, width))
    profile.set_shape("key_bias", (1,), (rows,), (rows,))
    config.add_optimization_profile(profile)
    return _serialize(builder, network, config, "encoder")


def build(request, writer) -> None:
    """Build one Nemotron 3 Diarization bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("nemotron3_diarization does not support dynamic_kv_cache")
    required = (
        ("task", request.task, _TASK),
        ("backend", request.backend, "trt"),
        ("precision", request.precision, "fp32"),
        ("tensor_parallel_size", request.tensor_parallel_size, 1),
        ("context_parallel_size", request.context_parallel_size, 1),
        ("max_batch_size", request.max_batch_size, 1),
        # The offline loop runs any length in chunks; nothing in the bundle bounds it.
        ("max_sequence_length", request.max_sequence_length, None),
        ("image_height", request.image_height, None),
        ("image_width", request.image_width, None),
        ("video_num_frames", request.video_num_frames, None),
        ("fp32_layers", request.fp32_layers, ()),
        ("mtp_seq_len", request.mtp_seq_len, None),
    )
    for name, actual, expected in required:
        if actual != expected:
            raise ValueError(f"{_FAMILY} requires {name}={expected!r}")
    if request.quantization not in {None, "none"}:
        raise ValueError(f"{_FAMILY} does not support quantization")
    model_dir = Path(request.model_dir)
    weights, shapes = _load(model_dir)
    cfg = config_module.resolve(model_dir, shapes)
    verbose = bool(request.verbose)

    writer.set_header(family="nemotron3_diarization", task=request.task, backend=request.backend)
    writer.add_bytes("embedder.plan", build_embedder(weights, cfg, verbose=verbose))
    writer.add_bytes("encoder.plan", build_encoder(weights, cfg, verbose=verbose))
    writer.add_bytes("mel_filters.f32", frontend.mel_filters(
        cfg["num_mel_bins"], cfg["n_fft"], cfg["sample_rate"]).tobytes())
    writer.add_bytes("window.f32", frontend.symmetric_hann(cfg["win_length"]).tobytes())
    writer.add_bytes("silence_embeds.f32",
                     np.ascontiguousarray(weights["silence_embeds"], dtype=np.float32).tobytes())
    writer.add_json("runtime.json", config_module.runtime_config(cfg))
