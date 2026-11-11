# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build WavLM CTC engines that keep each clip's true length.

WavLM is wav2vec2's convolutional front end and encoder with one change in
every attention block: the scores get a relative position bias, looked up once
from bucketed key-minus-query offsets and rescaled per head and frame by a gate
computed from that block's own input.
"""

from __future__ import annotations

import math
from pathlib import Path

import ml_dtypes  # noqa: F401
import numpy as np
import tensorrt as trt

from . import config as config_module, graph as g

_SUPPORTED_TASKS = ("speech_transcription",)
_PREFIX = "wavlm."
_PRECISIONS = ("fp32", "fp16")


def _open_safetensors(model_dir: Path):
    from safetensors import safe_open

    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Required safetensors checkpoint is missing: {path}")
    return safe_open(str(path), framework="numpy")


def _load(model_dir: Path):
    with _open_safetensors(model_dir) as handle:
        weights = {name: handle.get_tensor(name).astype(np.float32) for name in handle.keys()}
    shapes = {name: tuple(value.shape) for name, value in weights.items()}
    return weights, shapes


def _require_weights(weights: dict, cfg: dict) -> None:
    needed = ["lm_head.weight", "lm_head.bias",
              _PREFIX + "feature_projection.projection.weight",
              _PREFIX + "encoder.pos_conv_embed.conv.weight_g",
              _PREFIX + "encoder.pos_conv_embed.conv.weight_v",
              _PREFIX + "encoder.layer_norm.weight",
              _PREFIX + "encoder.layers.0.attention.rel_attn_embed.weight"]
    for index in range(len(cfg["conv_kernel"])):
        base = f"{_PREFIX}feature_extractor.conv_layers.{index}."
        needed.append(base + "conv.weight")
        if cfg["conv_bias"]:
            needed.append(base + "conv.bias")
        if index == 0 or cfg["feat_extract_norm"] == "layer":
            needed.extend((base + "layer_norm.weight", base + "layer_norm.bias"))
    for index in range(cfg["num_hidden_layers"]):
        base = f"{_PREFIX}encoder.layers.{index}."
        needed.append(base + "attention.q_proj.weight")
        needed.append(base + "attention.gru_rel_pos_linear.weight")
        needed.append(base + "attention.gru_rel_pos_const")
        needed.append(base + "final_layer_norm.weight")
    missing = [name for name in needed if name not in weights]
    if missing:
        raise KeyError("wavlm checkpoint is missing: " + ", ".join(sorted(missing)))


def relative_buckets(offsets, cfg) -> np.ndarray:
    """The reference's bucket for each key-minus-query offset.

    Half of the buckets serve positive offsets. Within each half the first
    ``max_exact`` distances are their own bucket and the rest are spaced
    logarithmically up to ``max_distance``, past which every distance shares
    the last bucket. The logarithm is taken in float32 and truncated, as the
    reference does, so that a distance near a bucket edge lands on the same
    side of it.
    """
    half = cfg["num_buckets"] // 2
    max_exact = half // 2
    offsets = np.asarray(offsets, dtype=np.int64)
    distance = np.abs(offsets)
    # Distances below max_exact take the exact branch; clipping them keeps the
    # logarithm finite on the branch that is discarded for them.
    ratio = np.maximum(distance, max_exact).astype(np.float32) / np.float32(max_exact)
    spaced = np.log(ratio) / np.float32(math.log(cfg["max_distance"] / max_exact))
    spaced = (np.float32(max_exact) + spaced * np.float32(half - max_exact)).astype(np.int64)
    spaced = np.minimum(spaced, half - 1)
    return (offsets > 0).astype(np.int64) * half + np.where(distance < max_exact, distance, spaced)


def position_bias_table(weights, cfg) -> np.ndarray:
    """``[2 * max_distance + 1, heads]``: the bias of every offset from ``-max_distance``
    to ``max_distance``. Only the first block carries the bucket embedding; every
    block reuses its bias."""
    embedding = np.asarray(weights[_PREFIX + "encoder.layers.0.attention.rel_attn_embed.weight"],
                           dtype=np.float32)
    if embedding.shape != (cfg["num_buckets"], cfg["num_attention_heads"]):
        raise ValueError(f"wavlm rel_attn_embed {embedding.shape} does not match "
                         f"({cfg['num_buckets']}, {cfg['num_attention_heads']})")
    offsets = np.arange(-cfg["max_distance"], cfg["max_distance"] + 1)
    return embedding[relative_buckets(offsets, cfg)]


def gate_projection(weights, base):
    """The gate's two logits as one ``[2, head_dim]`` projection.

    The reference projects each head's slice to eight values and sums them in
    two groups of four; summing the rows of the weight and the bias first gives
    the same two logits with one quarter of the work.
    """
    weight = np.asarray(weights[base + "attention.gru_rel_pos_linear.weight"], dtype=np.float64)
    bias = np.asarray(weights[base + "attention.gru_rel_pos_linear.bias"], dtype=np.float64)
    if weight.shape[0] != 8 or bias.shape != (8,):
        raise ValueError(f"wavlm gru_rel_pos_linear {weight.shape} is not an eight-way projection")
    return (weight.reshape(2, 4, -1).sum(axis=1).astype(np.float32),
            bias.reshape(2, 4).sum(axis=1).astype(np.float32))


def _feature_encoder(network, audio, weights, cfg):
    hidden = audio
    for index, (kernel, stride) in enumerate(zip(cfg["conv_kernel"], cfg["conv_stride"])):
        base = f"{_PREFIX}feature_extractor.conv_layers.{index}."
        hidden = g.add_conv1d(network, hidden, weights[base + "conv.weight"], stride=stride,
                              bias=weights[base + "conv.bias"] if cfg["conv_bias"] else None)
        if cfg["feat_extract_norm"] == "layer" or index == 0:
            axis = 1 if cfg["feat_extract_norm"] == "layer" else 3
            hidden = g.add_norm(network, hidden, weights[base + "layer_norm.weight"],
                                weights[base + "layer_norm.bias"], 1e-5, axis,
                                param_axis=1)
        hidden = g.add_gelu(network, hidden)
    return hidden


def _positional(network, rows, weights, cfg):
    """Add the weight-normed grouped convolution over the time axis."""
    hidden_size = cfg["hidden_size"]
    channels = network.add_shuffle(rows)
    channels.first_transpose = (1, 0)
    channels.reshape_dims = (1, hidden_size, 1, -1)
    source = channels.get_output(0)

    base = _PREFIX + "encoder.pos_conv_embed.conv."
    kernel = g.weight_normed_kernel(weights[base + "weight_g"], weights[base + "weight_v"])
    conv = g.add_conv1d(network, source, kernel, padding=cfg["pos_kernel"] // 2,
                        groups=cfg["pos_groups"], bias=weights[base + "bias"])
    # The even kernel leaves one column more than it was given, and the
    # reference drops the last one, not the first.
    trim = network.add_slice(conv, (0, 0, 0, 0), (1, hidden_size, 1, 1), (1, 1, 1, 1))
    trim.set_input(2, network.add_shape(source).get_output(0))
    activated = g.add_gelu(network, trim.get_output(0))

    flat = network.add_shuffle(activated)
    flat.first_transpose = (0, 3, 1, 2)
    flat.reshape_dims = (-1, hidden_size)
    return g.add_sum(network, rows, flat.get_output(0))


def _encoder_layers(network, hidden, position_bias, weights, cfg):
    heads = cfg["num_attention_heads"]
    head_dim = cfg["hidden_size"] // heads
    for index in range(cfg["num_hidden_layers"]):
        base = f"{_PREFIX}encoder.layers.{index}."
        attention_input = hidden
        if cfg["do_stable_layer_norm"]:
            attention_input = g.add_norm(network, hidden, weights[base + "layer_norm.weight"],
                                         weights[base + "layer_norm.bias"], cfg["eps"], 1)
        # The gate reads the same rows the projections read, so in the stable
        # layout it follows the block's pre-norm.
        gate_weight, gate_bias = gate_projection(weights, base)
        gate = g.add_head_gate(network, attention_input, gate_weight, gate_bias,
                               np.asarray(weights[base + "attention.gru_rel_pos_const"]).reshape(-1),
                               num_heads=heads, head_dim=head_dim)
        query = g.add_linear(network, attention_input, weights[base + "attention.q_proj.weight"],
                             weights[base + "attention.q_proj.bias"])
        key = g.add_linear(network, attention_input, weights[base + "attention.k_proj.weight"],
                           weights[base + "attention.k_proj.bias"])
        value = g.add_linear(network, attention_input, weights[base + "attention.v_proj.weight"],
                             weights[base + "attention.v_proj.bias"])
        attended = g.add_attention(network, query, key, value,
                                   g.add_prod(network, gate, position_bias),
                                   num_heads=heads, head_dim=head_dim)
        attended = g.add_linear(network, attended, weights[base + "attention.out_proj.weight"],
                                weights[base + "attention.out_proj.bias"])
        hidden = g.add_sum(network, hidden, attended)
        if not cfg["do_stable_layer_norm"]:
            hidden = g.add_norm(network, hidden, weights[base + "layer_norm.weight"],
                                weights[base + "layer_norm.bias"], cfg["eps"], 1)
        residual = hidden
        if cfg["do_stable_layer_norm"]:
            hidden = g.add_norm(network, hidden, weights[base + "final_layer_norm.weight"],
                                weights[base + "final_layer_norm.bias"], cfg["eps"], 1)
        inner = g.add_linear(network, hidden,
                             weights[base + "feed_forward.intermediate_dense.weight"],
                             weights[base + "feed_forward.intermediate_dense.bias"])
        inner = g.add_gelu(network, inner)
        inner = g.add_linear(network, inner,
                             weights[base + "feed_forward.output_dense.weight"],
                             weights[base + "feed_forward.output_dense.bias"])
        hidden = g.add_sum(network, residual, inner)
        if not cfg["do_stable_layer_norm"]:
            hidden = g.add_norm(network, hidden, weights[base + "final_layer_norm.weight"],
                                weights[base + "final_layer_norm.bias"], cfg["eps"], 1)
    return hidden


def _encoder(network, projected, weights, cfg):
    hidden = _positional(network, projected, weights, cfg)
    if not cfg["do_stable_layer_norm"]:
        hidden = g.add_norm(network, hidden, weights[_PREFIX + "encoder.layer_norm.weight"],
                            weights[_PREFIX + "encoder.layer_norm.bias"], cfg["eps"], 1)
    position_bias = g.add_position_bias(network, hidden, position_bias_table(weights, cfg),
                                        cfg["max_distance"])
    hidden = _encoder_layers(network, hidden, position_bias, weights, cfg)
    if cfg["do_stable_layer_norm"]:
        hidden = g.add_norm(network, hidden, weights[_PREFIX + "encoder.layer_norm.weight"],
                            weights[_PREFIX + "encoder.layer_norm.bias"], cfg["eps"], 1)
    return hidden


def _builder(verbose, *, strongly_typed=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # The fp16 graph states every tensor type itself (fp16 compute, fp32 norms);
    # the fp32 graph keeps the default network flags.
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
    network = builder.create_network(flags)
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 would cap the convolutions near 1e-3 and hide a real graph error.
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    return builder, network, config


def build_engine(weights, cfg, *, max_samples, precision="fp32", verbose=False) -> bytes:
    """Build the CTC engine. ``precision`` is the compute type of the convolutions,
    projections and attention; the input waveform and the output logits stay fp32."""
    if precision not in _PRECISIONS:
        raise NotImplementedError(f"wavlm builds {' or '.join(_PRECISIONS)}, not {precision}")
    half = precision == "fp16"
    builder, network, config = _builder(verbose, strongly_typed=half)
    audio = network.add_input("audio", trt.float32, (1, 1, 1, -1))
    if half:
        audio = network.add_cast(audio, trt.float16).get_output(0)

    hidden = _feature_encoder(network, audio, weights, cfg)
    rows = network.add_shuffle(hidden)
    rows.first_transpose = (0, 3, 1, 2)
    rows.reshape_dims = (-1, cfg["conv_dim"][-1])
    projected = g.add_norm(network, rows.get_output(0),
                           weights[_PREFIX + "feature_projection.layer_norm.weight"],
                           weights[_PREFIX + "feature_projection.layer_norm.bias"],
                           cfg["eps"], 1)
    projected = g.add_linear(network, projected,
                             weights[_PREFIX + "feature_projection.projection.weight"],
                             weights[_PREFIX + "feature_projection.projection.bias"])

    hidden = _encoder(network, projected, weights, cfg)

    logits = g.add_linear(network, hidden, weights["lm_head.weight"], weights["lm_head.bias"])
    if half:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    profile = builder.create_optimization_profile()
    profile.set_shape("audio", (1, 1, 1, config_module.MIN_SAMPLES),
                      (1, 1, 1, min(config_module.OPT_SAMPLES, max_samples)),
                      (1, 1, 1, max_samples))
    config.add_optimization_profile(profile)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("wavlm engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one WavLM CTC bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("wavlm supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend != "trt":
        raise ValueError("wavlm supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("wavlm does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("wavlm does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("wavlm does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("wavlm does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("wavlm does not support video_num_frames")
    if request.image_height is not None or request.image_width is not None:
        # The encoder reads a waveform, so there is no image to size.
        raise NotImplementedError("wavlm takes audio, not an image size")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("wavlm does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("wavlm does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("wavlm does not support MTP speculative decoding")

    precision = str(request.precision).lower()
    if precision not in _PRECISIONS:
        # fp32 stays the default. fp16 is accepted when asked for: it keeps the
        # norms in fp32 and its transcripts may differ from the fp32 reference
        # on a small share of clips (see the README).
        raise NotImplementedError(
            f"wavlm builds {' or '.join(_PRECISIONS)}, not {precision}")

    model_dir = Path(request.model_dir)
    # The configuration is resolved before any weight is read, so a checkpoint
    # without a CTC head is refused by name rather than by a missing file.
    cfg = config_module.resolve(model_dir, {})
    weights, shapes = _load(model_dir)
    config_module.require_head(cfg, shapes)
    _require_weights(weights, cfg)

    # max_sequence_length is a sample count here, not a token count: it is the
    # longest clip the engine's profile will accept in one pass.
    max_samples = config_module.MAX_SAMPLES
    if request.max_sequence_length is not None:
        max_samples = int(request.max_sequence_length)
        if max_samples < config_module.MIN_SAMPLES:
            raise ValueError(
                f"wavlm needs at least {config_module.MIN_SAMPLES} samples, not {max_samples}")

    writer.set_header(family="wavlm", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(
        weights, cfg, max_samples=max_samples, precision=precision,
        verbose=bool(request.verbose)))
    writer.add_json(
        "runtime.json",
        {
            "sample_rate": cfg["sample_rate"],
            "do_normalize": cfg["do_normalize"],
            "conv_kernel": cfg["conv_kernel"],
            "conv_stride": cfg["conv_stride"],
            "tokens": cfg["tokens"],
            "blank_token_id": cfg["blank_token_id"],
            "word_delimiter_token": cfg["word_delimiter_token"],
            "min_samples": config_module.MIN_SAMPLES,
            "max_samples": max_samples,
        },
    )
