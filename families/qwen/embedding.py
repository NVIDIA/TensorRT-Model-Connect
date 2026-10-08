# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the pinned Qwen3 embedding task within its Qwen family owner."""

from __future__ import annotations
import json
from pathlib import Path
from .embedding_config import ModelConfig
from .embedding_contract import detect_qwen3_embedding_contract


def validate_request(request):
    if request.task != "embedding":
        raise ValueError("Qwen embedding requires task=embedding")
    if request.precision not in {"fp16", "bf16"}:
        raise ValueError("Qwen embedding precision must be fp16 or bf16")
    if request.quantization not in {None, "none"} or request.fp32_layers:
        raise ValueError("Qwen embedding does not support quantization or mixed precision")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise ValueError("Qwen embedding does not support parallel builds")
    if request.dynamic_kv_cache:
        raise ValueError("Qwen embedding does not use a KV cache")
    if request.max_batch_size != 1:
        raise ValueError("Qwen embedding supports one text per request")
    if any(
        value is not None
        for value in (request.image_height, request.image_width, request.video_num_frames)
    ):
        raise ValueError("Qwen embedding accepts text only")
    config = ModelConfig.from_dir(request.model_dir)
    contract = detect_qwen3_embedding_contract(config)
    if contract is None:
        raise ValueError(
            "Qwen embedding requires the Qwen3-Embedding-0.6B sentence-transformers last-token pooling contract"
        )
    length = request.max_sequence_length or config.max_position_embeddings
    if length < 1 or length > config.max_position_embeddings:
        raise ValueError("max_sequence_length exceeds checkpoint context capacity")
    if not (Path(request.model_dir) / "tokenizer.json").is_file():
        raise ValueError("Qwen embedding requires tokenizer.json")
    return config, contract, length


def build_embedding(request, writer):
    config, contract, length = validate_request(request)
    from .embedding_weights import load_standard_weights
    from .embedding_builder import build_qwen3_embedding_engine

    weights = load_standard_weights(
        str(request.model_dir), config, precision=request.precision, include_lm_head=False
    )
    plan = build_qwen3_embedding_engine(
        config, weights, length, precision=request.precision, verbose=request.verbose
    )
    writer.set_header(family="qwen", task="embedding", backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_bytes(
        "runtime.json",
        json.dumps(
            {
                "embedding_pooling": contract.pooling,
                "embedding_normalize": contract.normalize,
                "embedding_dimension": contract.embedding_dimension,
                "embedding_eos_token_id": contract.eos_token_id,
                "max_sequence_length": length,
                "precision": request.precision,
            }
        ).encode(),
    )
    writer.add_bytes("tokenizer.json", (Path(request.model_dir) / "tokenizer.json").read_bytes())
