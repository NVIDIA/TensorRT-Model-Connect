# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a wav2vec2 bundle.

One engine, waveform in and CTC logits out. There is no spectrogram: the
feature encoder is seven strided convolutions over the raw samples, and the
only frontend the runtime owns is a mean/variance normalisation of the clip.

The engine's time axis is dynamic. That is forced rather than chosen: the
feature encoder's first layer normalises per channel across time, so zero
padding a clip to a fixed window shifts every frame's statistics and the
trimmed result is not the unpadded one. The family measures that rather than
asserting it, in ``tests/test_model.py``.
"""

from __future__ import annotations

from pathlib import Path

import ml_dtypes  # noqa: F401
import numpy as np
import tensorrt as trt

from . import config as config_module, graph as g

_SUPPORTED_TASKS = ("speech_transcription",)
_PREFIX = "wav2vec2."


def _open_safetensors(model_dir: Path):
    from safetensors import safe_open

    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Required safetensors checkpoint is missing: {path}")
    return safe_open(str(path), framework="numpy")


def _load(model_dir: Path):
    handle = _open_safetensors(model_dir)
    weights = {name: handle.get_tensor(name).astype(np.float32) for name in handle.keys()}
    shapes = {name: tuple(value.shape) for name, value in weights.items()}
    return weights, shapes


def _require_weights(weights: dict, cfg: dict) -> None:
    needed = ["lm_head.weight", "lm_head.bias",
              _PREFIX + "feature_projection.projection.weight",
              _PREFIX + "encoder.pos_conv_embed.conv.weight_g",
              _PREFIX + "encoder.pos_conv_embed.conv.weight_v",
              _PREFIX + "encoder.layer_norm.weight"]
    for index in range(len(cfg["conv_kernel"])):
        needed.append(f"{_PREFIX}feature_extractor.conv_layers.{index}.conv.weight")
    # Only the first feature-encoder layer carries a normalisation under
    # feat_extract_norm='group'; a checkpoint that normalised every layer would
    # be the 'layer' variant config.py already refuses.
    needed.append(_PREFIX + "feature_extractor.conv_layers.0.layer_norm.weight")
    for index in range(cfg["num_hidden_layers"]):
        needed.append(f"{_PREFIX}encoder.layers.{index}.attention.q_proj.weight")
        needed.append(f"{_PREFIX}encoder.layers.{index}.final_layer_norm.weight")
    missing = [name for name in needed if name not in weights]
    if missing:
        raise KeyError("wav2vec2 checkpoint is missing: " + ", ".join(sorted(missing)))


def _feature_encoder(network, audio, weights, cfg):
    """Seven strided convolutions; only layer 0 is normalised."""
    hidden = audio
    for index, (kernel, stride) in enumerate(zip(cfg["conv_kernel"], cfg["conv_stride"])):
        base = f"{_PREFIX}feature_extractor.conv_layers.{index}."
        hidden = g.add_conv1d(network, hidden, weights[base + "conv.weight"], stride=stride)
        if index == 0:
            # GroupNorm(num_groups=C, num_channels=C): the statistics are taken
            # per channel over time (axis 3), while the affine term rides the
            # channel axis (axis 1).
            hidden = g.add_norm(network, hidden, weights[base + "layer_norm.weight"],
                                weights[base + "layer_norm.bias"], cfg["eps"], 3,
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
    # The even kernel leaves one column more than it was given. The reference
    # drops the last one; dropping the first instead shifts every position by a
    # frame and still produces a near-correct transcript, so the side matters.
    trim = network.add_slice(conv, (0, 0, 0, 0), (1, hidden_size, 1, 1), (1, 1, 1, 1))
    trim.set_input(2, network.add_shape(source).get_output(0))
    activated = g.add_gelu(network, trim.get_output(0))

    flat = network.add_shuffle(activated)
    flat.first_transpose = (0, 3, 1, 2)
    flat.reshape_dims = (-1, hidden_size)
    return g.add_sum(network, rows, flat.get_output(0))


def _encoder_layers(network, hidden, weights, cfg):
    """Twelve post-norm layers: the norms follow each residual, not precede it."""
    heads = cfg["num_attention_heads"]
    head_dim = cfg["hidden_size"] // heads
    for index in range(cfg["num_hidden_layers"]):
        base = f"{_PREFIX}encoder.layers.{index}."
        query = g.add_linear(network, hidden, weights[base + "attention.q_proj.weight"],
                             weights[base + "attention.q_proj.bias"])
        key = g.add_linear(network, hidden, weights[base + "attention.k_proj.weight"],
                           weights[base + "attention.k_proj.bias"])
        value = g.add_linear(network, hidden, weights[base + "attention.v_proj.weight"],
                             weights[base + "attention.v_proj.bias"])
        attended = g.add_attention(network, query, key, value,
                                   num_heads=heads, head_dim=head_dim)
        attended = g.add_linear(network, attended, weights[base + "attention.out_proj.weight"],
                                weights[base + "attention.out_proj.bias"])
        hidden = g.add_sum(network, hidden, attended)
        hidden = g.add_norm(network, hidden, weights[base + "layer_norm.weight"],
                            weights[base + "layer_norm.bias"], cfg["eps"], 1)
        inner = g.add_linear(network, hidden,
                             weights[base + "feed_forward.intermediate_dense.weight"],
                             weights[base + "feed_forward.intermediate_dense.bias"])
        inner = g.add_gelu(network, inner)
        inner = g.add_linear(network, inner,
                             weights[base + "feed_forward.output_dense.weight"],
                             weights[base + "feed_forward.output_dense.bias"])
        hidden = g.add_sum(network, hidden, inner)
        hidden = g.add_norm(network, hidden, weights[base + "final_layer_norm.weight"],
                            weights[base + "final_layer_norm.bias"], cfg["eps"], 1)
    return hidden


def _builder(verbose):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 would cap the convolutions near 1e-3 and hide a real graph error.
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    return builder, network, config


def build_engine(weights, cfg, *, max_samples, verbose=False) -> bytes:
    builder, network, config = _builder(verbose)
    audio = network.add_input("audio", trt.float32, (1, 1, 1, -1))

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

    hidden = _positional(network, projected, weights, cfg)
    hidden = g.add_norm(network, hidden, weights[_PREFIX + "encoder.layer_norm.weight"],
                        weights[_PREFIX + "encoder.layer_norm.bias"], cfg["eps"], 1)
    hidden = _encoder_layers(network, hidden, weights, cfg)

    logits = g.add_linear(network, hidden, weights["lm_head.weight"], weights["lm_head.bias"])
    logits.name = "logits"
    network.mark_output(logits)

    profile = builder.create_optimization_profile()
    profile.set_shape("audio", (1, 1, 1, config_module.MIN_SAMPLES),
                      (1, 1, 1, min(config_module.OPT_SAMPLES, max_samples)),
                      (1, 1, 1, max_samples))
    config.add_optimization_profile(profile)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("wav2vec2 engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one wav2vec2 bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("wav2vec2 supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("wav2vec2 supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("wav2vec2 does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("wav2vec2 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("wav2vec2 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("wav2vec2 does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("wav2vec2 does not support video_num_frames")
    if request.image_height is not None or request.image_width is not None:
        # The encoder reads a waveform, so there is no image to size.
        raise NotImplementedError("wav2vec2 takes audio, not an image size")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("wav2vec2 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("wav2vec2 does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("wav2vec2 does not support MTP speculative decoding")

    precision = str(request.precision).lower()
    if precision != "fp32":
        # The E2E gate is an exact transcript match, so any drift that flips a
        # single frame's argmax changes the output. fp32 costs little on a 94M
        # encoder and is the only precision this family has been measured at.
        raise NotImplementedError(
            f"wav2vec2 is qualified at fp32 only, not {precision}")

    model_dir = Path(request.model_dir)
    weights, shapes = _load(model_dir)
    cfg = config_module.resolve(model_dir, shapes)
    _require_weights(weights, cfg)

    # max_sequence_length is a sample count here, not a token count: it is the
    # longest clip the engine's profile will accept in one pass.
    max_samples = config_module.MAX_SAMPLES
    if request.max_sequence_length is not None:
        max_samples = int(request.max_sequence_length)
        if max_samples < config_module.MIN_SAMPLES:
            raise ValueError(
                f"wav2vec2 needs at least {config_module.MIN_SAMPLES} samples, not {max_samples}")

    writer.set_header(family="wav2vec2", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(
        weights, cfg, max_samples=max_samples, verbose=bool(request.verbose)))
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
