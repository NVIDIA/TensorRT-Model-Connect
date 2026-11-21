# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Cosmos3-Edge reasoning bundle construction."""

from pathlib import Path

from .checkpoint_mapper import CheckpointWeights
from .config import ModelConfig
from .native_graph import build_decoder, build_visual
from .support import SUPPORTED_TASKS


def build(request, writer):
    """Own every build argument, then choose complete-network offload or native."""
    from .edge_llm import dispatch
    if request.backend != "trt":
        raise ValueError("Cosmos3-Edge requires the TensorRT backend")
    if request.task not in SUPPORTED_TASKS:
        raise ValueError("Cosmos3-Edge requires a reasoning text/image/video task")
    if request.dynamic_kv_cache:
        raise NotImplementedError("cosmos3_edge does not support dynamic_kv_cache")
    for name, value, default in (
        ("max_batch_size", request.max_batch_size, 1),
        ("tensor_parallel_size", request.tensor_parallel_size, 1),
        ("context_parallel_size", request.context_parallel_size, 1),
        ("graph_transform", request.graph_transform, None), ("mtp_seq_len", request.mtp_seq_len, None),
        ("fp32_layers", request.fp32_layers, ()), ("image_height", request.image_height, None),
        ("image_width", request.image_width, None), ("video_num_frames", request.video_num_frames, None),
    ):
        if value != default:
            raise ValueError(f"Cosmos3-Edge does not expose {name}; media uses dynamic shapes")
    if request.quantization not in (None, "none"):
        raise ValueError("Cosmos3-Edge reasoning expects the unquantized checkpoint")
    precision = request.precision.lower()
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("Cosmos3-Edge precision must be fp32, fp16, or bf16")
    root = Path(request.model_dir)
    config = ModelConfig.from_dir(root)
    if (config.raw.get("quantization_config") is not None
            or any((root / name).exists() for name in
                   ("hf_quant_config.json", "quantize_config.json", "quant_config.json"))):
        raise ValueError("Cosmos3-Edge reasoning expects original, unquantized checkpoint weights")
    capacity = 2048 if request.max_sequence_length is None else request.max_sequence_length
    if type(capacity) is not int or not 1 <= capacity <= config.max_position_embeddings:
        raise ValueError("Cosmos3-Edge max_sequence_length must be between 1 and context capacity")
    # A visual invocation is one frame; the full prompt shares the decoder budget.
    max_patches = max(256, min(capacity * config.vision["spatial_merge_size"] ** 2, 16384))

    def native(request, writer):
        weights = CheckpointWeights(root)
        tokenizer = (root / "tokenizer.json").read_bytes()
        writer.set_header(family="cosmos3_edge", task=request.task, backend=request.backend)
        writer.add_bytes("decoder.plan", build_decoder(
            config, weights, capacity, precision=precision, max_input_length=capacity,
            verbose=request.verbose))
        writer.add_bytes("visual.plan", build_visual(
            config, weights, max_patches, precision=precision, verbose=request.verbose))
        with writer.open_section("embeddings.f32") as output:
            output.write(weights["embed_tokens.weight"].astype("<f4", copy=False).tobytes())
        writer.add_bytes("tokenizer.json", tokenizer)
        writer.add_json("runtime.json", {
            "format": 2, "config": config.raw, "precision": precision,
            "max_cache_length": capacity, "max_input_length": capacity,
            "max_patches": max_patches,
        })
        for name in ("chat_template.jinja", "preprocessor_config.json",
                     "video_preprocessor_config.json", "generation_config.json"):
            writer.add_bytes(name, (root / name).read_bytes())

    dispatch.build(request, writer, native, capacity)
