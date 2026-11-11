# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the complete Qwen3-Next BF16 checkpoint as family-owned engines."""

from dataclasses import asdict
from pathlib import Path
import shutil
import tensorrt as trt

from .cli import coerce_request
from .config import Config
from .weights import Weights
from .graph import head_network, layer_network


def engine(weights, config, layer, limit, verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = (
        head_network(network, weights, config)
        if layer is None
        else layer_network(network, weights, config, layer)
    )
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024**3)
    options.clear_flag(trt.BuilderFlag.TF32)
    if layer is not None and config.layer_types[layer] == "full_attention":
        profile = builder.create_optimization_profile()
        for name in ("past_k", "past_v"):
            profile.set_shape(
                name,
                (config.num_key_value_heads, 1, config.head_dim),
                (config.num_key_value_heads, min(32, limit), config.head_dim),
                (config.num_key_value_heads, limit, config.head_dim),
            )
        profile.set_shape("mask", (1, 1, 2), (1, 1, min(32, limit) + 1), (1, 1, limit + 1))
        options.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, options)
    assert graph is not None  # keeps all constant storage alive through serialization
    if plan is None:
        raise RuntimeError(f"TensorRT failed to build Qwen3-Next layer {layer}")
    return bytes(plan)


def build(request, writer):
    request = coerce_request(request)
    directory = Path(request.model_dir)
    config = Config.load(directory)
    limit = request.max_sequence_length
    if limit > config.max_position_embeddings:
        raise ValueError("Qwen3-Next context exceeds checkpoint capacity")
    weights = Weights(directory)
    writer.set_header(family="qwen3_next", task=request.task, backend=request.backend)
    writer.add_json(
        "runtime.json",
        {
            **asdict(config),
            "layer_types": config.layer_types,
            "conv_dim": config.conv_dim,
            "max_sequence_length": limit,
            "precision": request.precision,
        },
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        with writer.open_section(name) as stream, (directory / name).open("rb") as source:
            shutil.copyfileobj(source, stream)
    template = directory / "chat_template.jinja"
    if template.is_file():
        writer.add_bytes("chat_template.jinja", template.read_bytes())
    with writer.open_section("embedding.bf16") as stream:
        _, _, _, entry = weights.describe("model.embed_tokens.weight")
        if entry["dtype"] != "BF16" or entry["shape"] != [config.vocab_size, config.hidden_size]:
            raise ValueError("Qwen3-Next embedding must be BF16 [vocabulary, hidden]")
        weights.copy("model.embed_tokens.weight", stream)
    for layer in range(config.num_hidden_layers):
        print(f"Building Qwen3-Next layer {layer + 1}/{config.num_hidden_layers}", flush=True)
        writer.add_bytes(
            f"layers/{layer}.plan", engine(weights, config, layer, limit, request.verbose)
        )
    writer.add_bytes("head.plan", engine(weights, config, None, limit, request.verbose))
    weights.finish()
