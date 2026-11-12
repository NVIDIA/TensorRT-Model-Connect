# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one Chroma text-to-image bundle.

The bundle holds three TensorRT plans built from the checkpoint - the T5
text encoder, the MMDiT denoiser and the VAE decoder - plus the tokenizer
and ``runtime.json``. Weights are embedded in the plans. The image size and
the text length are fixed per bundle; sampling settings are runtime Config.
"""

from __future__ import annotations

import time
from pathlib import Path

from .checkpoint import ChromaCheckpoint, SafetensorsDirectory, load_checkpoint

PRECISIONS = ("bf16", "fp32")
DEFAULT_IMAGE_SIZE = 1024
# Up to 4096 px the 2x2-patch positions (at most 256) are exact in BF16, as the pipeline's position ids are.
MAX_IMAGE_SIZE = 4096
MAX_TEXT_LENGTH = 512
# ChromaPipeline.__call__ defaults.
DEFAULT_STEPS = 35
DEFAULT_GUIDANCE = 5.0


def _plan(name: str, precision: str, define, verbose: bool) -> "object":
    """Build one strongly typed network; returns the serialized plan (IHostMemory)."""
    import tensorrt as trt

    from .graph_ops import Graph

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = Graph(network, precision)
    define(graph)
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    # FP32 tensors stay true FP32: the reference runs them without TF32.
    config.clear_flag(trt.BuilderFlag.TF32)
    begin = time.monotonic()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT failed to build the chroma {name}")
    print(f"[chroma] built {name} ({precision}) in {time.monotonic() - begin:.1f} s, {plan.nbytes / 2**30:.2f} GiB")
    return plan


def _runtime_config(checkpoint: ChromaCheckpoint, precision: str, height: int, width: int, length: int) -> dict:
    return {
        "precision": precision,
        "height": height,
        "width": width,
        "text_length": length,
        "latent_channels": checkpoint.vae.latent_channels,
        "vae_scale_factor": checkpoint.vae_scale_factor,
        "vae_scaling_factor": checkpoint.vae.scaling_factor,
        "vae_shift_factor": checkpoint.vae.shift_factor,
        "scheduler_shift": checkpoint.scheduler.shift,
        "num_train_timesteps": checkpoint.scheduler.num_train_timesteps,
        "default_num_inference_steps": DEFAULT_STEPS,
        "default_guidance_scale": DEFAULT_GUIDANCE,
    }


def _write_components(writer, checkpoint: ChromaCheckpoint, precision: str, verbose: bool,
                      height: int, width: int, length: int) -> None:
    from .text_encoder_builder import build_text_encoder_network
    from .transformer_builder import build_transformer_network
    from .vae_builder import build_vae_decoder_network

    scale = checkpoint.vae_scale_factor
    components = (
        ("text_encoder.plan", "text encoder", "text_encoder",
         lambda graph, weights: build_text_encoder_network(graph, weights, checkpoint.text_encoder, length)),
        ("transformer.plan", "transformer", "transformer",
         lambda graph, weights: build_transformer_network(
             graph, weights, checkpoint.transformer, length, height // (2 * scale), width // (2 * scale))),
        ("vae_decoder.plan", "VAE decoder", "vae",
         lambda graph, weights: build_vae_decoder_network(
             graph, weights, checkpoint.vae, height // scale, width // scale)),
    )
    for section, name, directory, define in components:
        _write_component(writer, section, name, precision, verbose, checkpoint.root / directory, define)


def _write_component(writer, section: str, name: str, precision: str, verbose: bool, directory: Path,
                     define) -> None:
    """Build and write one plan; its weights map and plan are released on return."""
    weights = SafetensorsDirectory(directory)
    plan = _plan(name, precision, lambda graph: define(graph, weights), verbose)
    with writer.open_section(section) as handle:
        handle.write(memoryview(plan))


def build(request, writer) -> None:
    """Build one Chroma text-to-image bundle."""
    if request.task != "text_to_image":
        raise ValueError("chroma supports only task=text_to_image")
    if request.backend != "trt":
        raise ValueError("chroma supports only backend=trt")
    if request.precision not in PRECISIONS:
        raise ValueError(f"chroma builds precision bf16 or fp32, not {request.precision}")
    if request.dynamic_kv_cache:
        raise NotImplementedError("chroma does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("chroma does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("chroma does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("chroma does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("chroma does not support video_num_frames")
    if request.quantization is not None:
        raise NotImplementedError("chroma does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("chroma does not support fp32_layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("chroma does not support mtp_seq_len")
    height = request.image_height or DEFAULT_IMAGE_SIZE
    width = request.image_width or DEFAULT_IMAGE_SIZE
    if height % 16 or width % 16 or height > MAX_IMAGE_SIZE or width > MAX_IMAGE_SIZE:
        raise ValueError(f"chroma image height and width must be multiples of 16 up to {MAX_IMAGE_SIZE}")
    length = request.max_sequence_length or MAX_TEXT_LENGTH
    if not 1 <= length <= MAX_TEXT_LENGTH:
        raise ValueError(f"chroma max_sequence_length must be in [1, {MAX_TEXT_LENGTH}]")

    checkpoint = load_checkpoint(Path(request.model_dir))
    from .tokenizer_asset import tokenizer_json

    tokenizer = tokenizer_json(checkpoint.root)
    writer.set_header(family="chroma", task=request.task, backend=request.backend)
    writer.add_json("runtime.json", _runtime_config(checkpoint, request.precision, height, width, length))
    writer.add_bytes("tokenizer.json", tokenizer)
    _write_components(writer, checkpoint, request.precision, request.verbose, height, width, length)
