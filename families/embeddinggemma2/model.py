# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from .config import Config
from .graph import encode
from .weights import load_weights


def build_engine(config, weights, capacity, precision, verbose=False):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 1
    settings.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    for name in ("token_id", "position_id"):
        profile.set_shape(name, (2,), (min(128, capacity),), (capacity,))
    settings.add_optimization_profile(profile)
    tokens = network.add_input("token_id", trt.int32, (-1,))
    positions = network.add_input("position_id", trt.int32, (-1,))
    dtype = trt.bfloat16 if precision == "bf16" else trt.float32
    output = encode(network, tokens, positions, config, weights, dtype, capacity)
    output.name = "hidden_states"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, settings)
    if plan is None:
        raise RuntimeError("EmbeddingGemma2 TensorRT engine build failed")
    return bytes(plan)


def build(request, writer):
    if request.mtp_seq_len is not None:
        raise NotImplementedError("embeddinggemma2 does not support MTP speculative decoding")
    if request.dynamic_kv_cache:
        raise NotImplementedError("embeddinggemma2 does not support dynamic_kv_cache")
    if request.task != "embedding":
        raise ValueError("EmbeddingGemma2 supports the embedding task only")
    if request.precision not in {"fp32", "bf16"}:
        raise ValueError("EmbeddingGemma2 supports FP32 or BF16, not FP16")
    if (
        request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
    ):
        raise ValueError("EmbeddingGemma2 requires TP1, CP1 and batch size 1")
    if (
        request.fp32_layers
        or request.quantization not in {None, "none"}
        or any(
            value is not None
            for value in (request.image_height, request.image_width, request.video_num_frames)
        )
    ):
        raise ValueError(
            "EmbeddingGemma2 text embeddings do not support multimodal, cache or quantized build options"
        )
    directory = Path(request.model_dir)
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config = Config.from_dict(raw)
    capacity = (
        request.max_sequence_length
        if request.max_sequence_length is not None
        else min(8192, config.max_position_embeddings)
    )
    if type(capacity) is not int or not 2 <= capacity <= min(8192, config.max_position_embeddings):
        raise ValueError(
            "EmbeddingGemma2 sequence capacity must cover BOS/EOS and be within checkpoint capacity and 8192"
        )
    tokenizer = (directory / "tokenizer.json").read_bytes()
    weights = load_weights(directory, config)
    plan = build_engine(config, weights, capacity, request.precision, request.verbose)
    writer.set_header(family="embeddinggemma2", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_bytes("tokenizer.json", tokenizer)
    writer.add_json(
        "runtime.json",
        {
            "format_version": 1,
            "embedding_dimension": config.embedding_dim,
            "vocab_size": config.vocab_size,
            "max_sequence_length": capacity,
            "pooling": "mean",
            "normalization": "l2",
            "model_id": "google/embeddinggemma-2",
            "unsupported_token_ids": [
                raw[key]
                for key in (
                    "image_token_id",
                    "audio_token_id",
                    "video_token_id",
                    "boi_token_id",
                    "boa_token_id",
                    "eoi_token_id",
                    "eoa_token_index",
                )
                if type(raw.get(key)) is int
            ],
        },
    )
