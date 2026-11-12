# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one HubertForCTC bundle: waveform in, CTC logits out, true clip length."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import ml_dtypes  # noqa: F401
import numpy as np
import tensorrt as trt

from . import config as config_module, graph as g

_SUPPORTED_TASKS = ("speech_transcription",)
_PRECISIONS = ("fp32", "fp16")
_PREFIX = config_module.PREFIX

# A checkpoint saved by current torch names the two tensors of the weight-normed
# positional kernel as parametrizations; the official checkpoints name them
# weight_g and weight_v. Transformers loads either spelling into the same kernel.
_PARAMETRIZED = {
    "parametrizations.weight.original0": "weight_g",
    "parametrizations.weight.original1": "weight_v",
}


def _canonical(name: str) -> str:
    for spelled, canonical in _PARAMETRIZED.items():
        if name.endswith(spelled):
            return name[: -len(spelled)] + canonical
    return name


def _open_torch_checkpoint(model_dir: Path) -> dict:
    """Open the checkpoint's single-file pytorch_model.bin, the format the
    official HuBERT CTC checkpoints ship, with restricted (weights-only) loading."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("This family requires torch in the build environment") from error

    path = Path(model_dir) / "pytorch_model.bin"
    if not path.is_file():
        raise FileNotFoundError(f"Required PyTorch checkpoint is missing: {path}")
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state.items()
    ):
        raise ValueError("hubert pytorch_model.bin must hold a nonempty tensor state dictionary")
    return {name: value.detach().to(torch.float32).numpy() for name, value in state.items()}


def _load(model_dir: Path):
    weights = {}
    for name, value in _open_torch_checkpoint(model_dir).items():
        canonical = _canonical(name)
        if canonical in weights:
            raise ValueError(f"hubert checkpoint names {canonical} twice")
        weights[canonical] = np.asarray(value, dtype=np.float32)
    shapes = {name: tuple(value.shape) for name, value in weights.items()}
    return weights, shapes


def _feature_encoder(network, audio, weights, cfg):
    hidden = audio
    for index, (kernel, stride) in enumerate(zip(cfg["conv_kernel"], cfg["conv_stride"])):
        base = f"{_PREFIX}feature_extractor.conv_layers.{index}."
        hidden = g.add_conv1d(network, hidden, weights[base + "conv.weight"], stride=stride,
                              bias=weights[base + "conv.bias"] if cfg["conv_bias"] else None)
        if cfg["feat_extract_norm"] == "layer" or index == 0:
            # LayerNorm over the channels of each frame, or the first layer's
            # GroupNorm(C, C) over the time axis of each channel. Both use
            # torch's default epsilon, not the config's layer_norm_eps.
            axis = 1 if cfg["feat_extract_norm"] == "layer" else 3
            hidden = g.add_norm(network, hidden, weights[base + "layer_norm.weight"],
                                weights[base + "layer_norm.bias"], 1e-5, axis, param_axis=1)
        hidden = g.add_gelu(network, hidden)
    return hidden


def _feature_projection(network, rows, weights, cfg):
    if cfg["feat_proj_layer_norm"]:
        rows = g.add_norm(network, rows, weights[_PREFIX + "feature_projection.layer_norm.weight"],
                          weights[_PREFIX + "feature_projection.layer_norm.bias"], cfg["eps"], 1)
    return g.add_linear(network, rows, weights[_PREFIX + "feature_projection.projection.weight"],
                        weights[_PREFIX + "feature_projection.projection.bias"])


def _positional(network, rows, weights, cfg):
    """Add the weight-normed grouped convolution over the time axis."""
    source = g.rows_to_frames(network, rows, cfg["hidden_size"])
    base = _PREFIX + "encoder.pos_conv_embed.conv."
    kernel = g.weight_normed_kernel(weights[base + "weight_g"], weights[base + "weight_v"])
    conv = g.add_conv1d(network, source, kernel, padding=cfg["pos_kernel"] // 2,
                        groups=cfg["pos_groups"], bias=weights[base + "bias"])
    # The even kernel leaves one frame more than it was given and the reference
    # drops the last one. Dropping the first instead shifts every position by a
    # frame and still produces a near-correct transcript, so the side matters.
    activated = g.add_gelu(network, g.trim_like(network, conv, source))
    return g.add_sum(network, rows, g.frames_to_rows(network, activated, cfg["hidden_size"]))


def _encoder_layers(network, hidden, weights, cfg):
    heads = cfg["num_attention_heads"]
    head_dim = cfg["hidden_size"] // heads
    stable = cfg["do_stable_layer_norm"]
    for index in range(cfg["num_hidden_layers"]):
        base = f"{_PREFIX}encoder.layers.{index}."

        def norm(tensor, name, base=base):
            return g.add_norm(network, tensor, weights[f"{base}{name}.weight"],
                              weights[f"{base}{name}.bias"], cfg["eps"], 1)

        def linear(tensor, name, base=base):
            return g.add_linear(network, tensor, weights[f"{base}{name}.weight"],
                                weights[f"{base}{name}.bias"])

        attention_input = norm(hidden, "layer_norm") if stable else hidden
        attended = g.add_attention(
            network, linear(attention_input, "attention.q_proj"),
            linear(attention_input, "attention.k_proj"),
            linear(attention_input, "attention.v_proj"), num_heads=heads, head_dim=head_dim)
        hidden = g.add_sum(network, hidden, linear(attended, "attention.out_proj"))
        if not stable:
            hidden = norm(hidden, "layer_norm")
        inner = norm(hidden, "final_layer_norm") if stable else hidden
        inner = g.add_gelu(network, linear(inner, "feed_forward.intermediate_dense"))
        hidden = g.add_sum(network, hidden, linear(inner, "feed_forward.output_dense"))
        if not stable:
            hidden = norm(hidden, "final_layer_norm")
    return hidden


def _encoder(network, projected, weights, cfg):
    """Positional convolution, then the blocks; the encoder norm sits before the
    blocks in the post-norm layout and after them in the stable layout."""
    hidden = _positional(network, projected, weights, cfg)
    norm_weight = weights[_PREFIX + "encoder.layer_norm.weight"]
    norm_bias = weights[_PREFIX + "encoder.layer_norm.bias"]
    if not cfg["do_stable_layer_norm"]:
        hidden = g.add_norm(network, hidden, norm_weight, norm_bias, cfg["eps"], 1)
    hidden = _encoder_layers(network, hidden, weights, cfg)
    if cfg["do_stable_layer_norm"]:
        hidden = g.add_norm(network, hidden, norm_weight, norm_bias, cfg["eps"], 1)
    return hidden


def _forward(network, audio, weights, cfg):
    """Waveform ``[1, 1, 1, samples]`` to CTC logits ``[frames, vocab]``."""
    features = _feature_encoder(network, audio, weights, cfg)
    rows = g.frames_to_rows(network, features, cfg["conv_dim"][-1])
    hidden = _encoder(network, _feature_projection(network, rows, weights, cfg), weights, cfg)
    return g.add_linear(network, hidden, weights["lm_head.weight"], weights["lm_head.bias"])


def _builder(verbose, *, strongly_typed):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # The fp16 graph states every tensor type itself (fp16 compute, fp32 norms);
    # the fp32 graph keeps the default network flags.
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
    network = builder.create_network(flags)
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 would cap the convolutions and GEMMs near 1e-3 and hide a graph error.
    config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def build_engine(weights, cfg, *, max_samples, precision="fp32", verbose=False) -> bytes:
    """Build the CTC engine. ``precision`` is the compute type of the convolutions,
    projections and attention; the waveform input and the logits stay fp32."""
    if precision not in _PRECISIONS:
        raise NotImplementedError(f"hubert builds {' or '.join(_PRECISIONS)}, not {precision}")
    half = precision == "fp16"
    builder, network, config = _builder(verbose, strongly_typed=half)
    audio = network.add_input("audio", trt.float32, (1, 1, 1, -1))
    if half:
        audio = network.add_cast(audio, trt.float16).get_output(0)

    logits = _forward(network, audio, weights, cfg)
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
        raise RuntimeError("hubert engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one HuBERT CTC bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("hubert supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend != "trt":
        raise ValueError("hubert supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("hubert does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("hubert builds one clip per call, not max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("hubert does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("hubert does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("hubert takes audio, not video_num_frames")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("hubert takes audio, not an image size")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("hubert does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("hubert does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("hubert does not support MTP speculative decoding")

    precision = str(request.precision).lower()
    if precision not in _PRECISIONS:
        raise NotImplementedError(f"hubert builds {' or '.join(_PRECISIONS)}, not {precision}")

    # max_sequence_length is a sample count here, not a token count: the longest
    # clip the engine's profile accepts in one pass.
    max_samples = config_module.MAX_SAMPLES
    if request.max_sequence_length is not None:
        max_samples = int(request.max_sequence_length)
        if max_samples < config_module.MIN_SAMPLES:
            raise ValueError(
                f"hubert needs at least {config_module.MIN_SAMPLES} samples, not {max_samples}")

    model_dir = Path(request.model_dir)
    weights, shapes = _load(model_dir)
    cfg = config_module.resolve(model_dir, shapes)

    writer.set_header(family="hubert", task=request.task, backend=request.backend)
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
