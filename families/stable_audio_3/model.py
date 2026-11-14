# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one Stable Audio 3 text-to-audio bundle.

Three engines go into the bundle: the conditioning engine (T5Gemma prompt
encoder plus the ``seconds_total`` embedder), the DiT, and the SAME decoder.
The bundle also carries the T5Gemma tokenizer and the constants the runtime
needs to size a render and to run the reference sampler on the host.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt

from . import checkpoint, conditioner_builder, decoder_builder, dit_builder
from . import config as config_module
from .graph import LOW_PRECISION

# Latent length the engines are tuned for: a 30 s render.
_OPT_RENDER_SECONDS = 30.0
_PRECISIONS = {"fp16": trt.float16, "fp32": trt.float32}


def _compile(populate, *, verbose: bool, profile=None) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    graph = populate(network)
    # TF32 rounds FP32 operands; it is enabled only when that rounding is exact.
    if products_exact_in_tf32(network):
        builder_config.set_flag(trt.BuilderFlag.TF32)
    else:
        builder_config.clear_flag(trt.BuilderFlag.TF32)
    if profile is not None:
        optimization = builder.create_optimization_profile()
        for name, shapes in profile.items():
            optimization.set_shape(name, *shapes)
        builder_config.add_optimization_profile(optimization)
    plan = builder.build_serialized_network(network, builder_config)
    del graph  # the constants' host buffers must outlive the build
    if plan is None:
        raise RuntimeError("stable_audio_3 TensorRT engine build failed")
    return bytes(plan)


# Other layers TF32 would affect in FP32. Their operands cannot be checked here: convolution and einsum weights,
# attention's internal softmax probabilities.
_UNCHECKED_PRODUCTS = (trt.LayerType.CONVOLUTION, trt.LayerType.DECONVOLUTION, trt.LayerType.EINSUM,
                       trt.LayerType.ATTENTION_INPUT, trt.LayerType.ATTENTION_OUTPUT)


def products_exact_in_tf32(network: trt.INetworkDefinition) -> bool:
    """True when the network has FP32 products and every FP32 operand of a product is a cast FP16 / BF16 tensor.

    ``Graph.linear`` / ``Graph.matmul`` multiply low-precision values in FP32. TF32 represents those values
    exactly, so it runs these products on tensor cores with FP32 accumulation, as the reference's cuBLAS GEMMs do;
    any other FP32 product (FP32 bundles, FP32 attention, convolutions) keeps full FP32.
    """
    producers = {}
    layers = [network.get_layer(index) for index in range(network.num_layers)]
    for layer in layers:
        for output in range(layer.num_outputs):
            producers[layer.get_output(output).name] = layer
    products = 0
    for layer in layers:
        operands = [layer.get_input(index) for index in range(layer.num_inputs)]
        wide = [tensor for tensor in operands if tensor is not None and tensor.dtype == trt.float32]
        wide_output = layer.num_outputs > 0 and layer.get_output(0).dtype == trt.float32
        if layer.type in _UNCHECKED_PRODUCTS and (wide or wide_output):
            return False
        if layer.type != trt.LayerType.MATRIX_MULTIPLY or not wide:
            continue
        products += 1
        for tensor in wide:
            source = producers.get(tensor.name)
            if source is None or source.type != trt.LayerType.CAST or source.get_input(0).dtype not in LOW_PRECISION:
                return False
    return products > 0


def as_model_dtype(weights: dict, work) -> dict:
    """Round FP32 parameters and buffers to the trunk dtype, as the reference's ``model.to(torch.float16)`` does.

    The FP32 islands (rotary frequencies, norm gains) then see the values the reference's FP16 model holds.
    """
    if work == trt.float32:
        return weights
    return {name: value.astype(np.float16).astype(np.float32) if value.dtype == np.float32 else value
            for name, value in weights.items()}


def render_latents(cfg: dict, seconds: float) -> int:
    """Latent length ``StableAudioModel._adapt_sample_size`` picks for one duration."""
    ratio = cfg["downsampling_ratio"]
    samples = int((seconds + config_module.DURATION_PADDING_SECONDS) * cfg["sample_rate"])
    samples = -(-samples // ratio) * ratio
    align = ratio * cfg["latent_align"]
    samples = -(-samples // align) * align
    return min(samples, config_module.DEFAULT_MAX_SAMPLES) // ratio


def build_conditioner_engine(model_dir: Path, cfg: dict, model_dtype, *, verbose: bool = False) -> bytes:
    text = checkpoint.text_encoder_weights(model_dir).get_many(conditioner_builder.text_encoder_names(cfg["text"]))
    cond = checkpoint.model_weights(model_dir, checkpoint.CONDITIONER_PREFIX).get_many(
        conditioner_builder.CONDITIONER_NAMES)
    # T5Gemma stays BF16 as stored; the reference does not convert it.
    cond = as_model_dtype(cond, model_dtype)
    return _compile(lambda network: conditioner_builder.populate(network, text, cond, cfg, model_dtype),
                    verbose=verbose)


def build_dit_engine(model_dir: Path, cfg: dict, work, max_latents: int, *, verbose: bool = False) -> bytes:
    weights = as_model_dtype(checkpoint.model_weights(model_dir, checkpoint.DIT_PREFIX).get_many(
        dit_builder.weight_names(cfg["dit"])), work)
    shape = (1, cfg["latent_dim"])
    smallest = cfg["latent_align"]
    opt = min(render_latents(cfg, _OPT_RENDER_SECONDS), max_latents)
    return _compile(lambda network: dit_builder.populate(network, weights, cfg, work, max_latents),
                    verbose=verbose,
                    profile={dit_builder.LATENTS: ((*shape, smallest), (*shape, opt), (*shape, max_latents))})


def build_decoder_engine(model_dir: Path, cfg: dict, work, max_latents: int, *, verbose: bool = False) -> bytes:
    weights = as_model_dtype(checkpoint.model_weights(model_dir, checkpoint.DECODER_PREFIX).get_many(
        decoder_builder.weight_names(cfg["decoder"])), work)
    running_std = as_model_dtype(
        {"running_std": checkpoint.model_weights(model_dir, checkpoint.BOTTLENECK_PREFIX).get("running_std")},
        work)["running_std"]
    shape = (1, cfg["latent_dim"])
    smallest = cfg["latent_align"]
    opt = min(render_latents(cfg, _OPT_RENDER_SECONDS), max_latents)
    return _compile(lambda network: decoder_builder.populate(network, weights, running_std, cfg, work, max_latents),
                    verbose=verbose,
                    profile={decoder_builder.LATENTS: ((*shape, smallest), (*shape, opt), (*shape, max_latents))})


def runtime_config(cfg: dict, precision: str, max_latents: int) -> dict:
    shift = cfg["sampling_shift"]
    return {
        "sample_rate": cfg["sample_rate"],
        "audio_channels": cfg["audio_channels"],
        "latent_dim": cfg["latent_dim"],
        "downsampling_ratio": cfg["downsampling_ratio"],
        "latent_align": cfg["latent_align"],
        "max_latents": max_latents,
        "headroom_seconds": config_module.DURATION_PADDING_SECONDS,
        "headroom_latents": cfg["headroom_latents"],
        "prompt_tokens": cfg["text"]["max_length"],
        "pad_token_id": cfg["text"]["pad_token_id"],
        "context_tokens": cfg["text"]["max_length"] + 1,
        "cond_dim": cfg["dit"]["cond_dim"],
        "shift_anchor_length": float(shift["anchor_length"]),
        "shift_anchor_logsnr": float(shift["anchor_logsnr"]),
        "shift_rate": float(shift["rate"]),
        "shift_logsnr_end": float(shift["logsnr_end"]),
        # The reference rounds the schedule to the model dtype before using it.
        "time_precision": precision,
    }


def build(request, writer) -> None:
    """Build one Stable Audio 3 text-to-audio bundle."""
    if request.task != "text_to_audio":
        raise ValueError("stable_audio_3 supports only task=text_to_audio")
    if request.backend != "trt":
        raise ValueError("stable_audio_3 supports only backend=trt")
    if request.max_batch_size != 1:
        raise NotImplementedError("stable_audio_3 renders one prompt at a time (max_batch_size=1)")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("stable_audio_3 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("stable_audio_3 does not support context parallelism")
    if request.max_sequence_length is not None:
        # The prompt length is the conditioner's and the render length follows the duration.
        raise NotImplementedError("stable_audio_3 does not support max_sequence_length")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("stable_audio_3 does not support image sizes")
    if request.video_num_frames is not None:
        raise NotImplementedError("stable_audio_3 does not support video_num_frames")
    if request.quantization not in (None, "none"):
        raise NotImplementedError(f"stable_audio_3 does not support quantization {request.quantization!r}")
    if request.fp32_layers:
        raise NotImplementedError("stable_audio_3 does not support mixed-precision layers")
    if request.dynamic_kv_cache:
        raise NotImplementedError("stable_audio_3 does not support dynamic_kv_cache")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("stable_audio_3 does not support MTP speculative decoding")
    precision = str(request.precision).lower()
    if precision not in _PRECISIONS:
        raise ValueError(f"stable_audio_3 supports precision fp16 or fp32, got {precision!r}")
    work = _PRECISIONS[precision]
    verbose = bool(request.verbose)

    model_dir = Path(request.model_dir)
    cfg = config_module.resolve(model_dir)
    max_latents = config_module.max_latents(cfg)
    tokenizer = model_dir / config_module.TEXT_ENCODER_DIR / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"stable_audio_3 checkpoint is missing {tokenizer}")

    writer.set_header(family="stable_audio_3", task=request.task, backend=request.backend)
    writer.add_bytes("conditioner.plan", build_conditioner_engine(model_dir, cfg, work, verbose=verbose))
    writer.add_bytes("dit.plan", build_dit_engine(model_dir, cfg, work, max_latents, verbose=verbose))
    writer.add_bytes("decoder.plan", build_decoder_engine(model_dir, cfg, work, max_latents, verbose=verbose))
    writer.add_bytes("tokenizer.json", tokenizer.read_bytes())
    writer.add_json("runtime.json", runtime_config(cfg, precision, max_latents))
