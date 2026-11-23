# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the bounded, greedy Cohere2 text-continuation bundle."""

import json
from pathlib import Path

from .config import parse_config


def tokenizer_contract(raw: dict, bos: int) -> None:
    post = raw.get("post_processor", {})
    if (
        post.get("type") != "TemplateProcessing"
        or post.get("single")
        != [
            {"SpecialToken": {"id": "<BOS_TOKEN>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
        ]
        or post.get("special_tokens", {}).get("<BOS_TOKEN>", {}).get("ids") != [bos]
    ):
        raise ValueError("cohere2 requires the checkpoint's single BOS tokenizer template")


def build(request, writer) -> None:
    if (
        request.task != "text_continuation"
        or request.backend != "trt"
        or request.precision != "fp32"
    ):
        raise ValueError("cohere2 supports text_continuation with backend=trt and precision=fp32")
    if (
        request.max_batch_size != 1
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
    ):
        raise NotImplementedError("cohere2 supports batch one on a single GPU")
    if (
        request.quantization not in (None, "none")
        or request.fp32_layers
        or request.dynamic_kv_cache
    ):
        raise NotImplementedError(
            "cohere2 does not implement quantization, layer overrides, or a dynamic KV cache"
        )
    for name in ("image_height", "image_width", "video_num_frames", "mtp_seq_len"):
        if getattr(request, name) is not None:
            raise NotImplementedError(f"cohere2 does not implement {name}")
    directory = Path(request.model_dir)
    cfg = parse_config(json.loads((directory / "config.json").read_text()))
    length = request.max_sequence_length if request.max_sequence_length is not None else 128
    if type(length) is not int or not 1 <= length <= cfg.max_position_embeddings:
        raise ValueError(
            f"cohere2 max_sequence_length must be in [1, {cfg.max_position_embeddings}]"
        )
    tokenizer = (directory / "tokenizer.json").read_bytes()
    tokenizer_contract(json.loads(tokenizer), cfg.bos_token_id)
    from .graph import build_engine
    from .weights import load_weights

    weights = load_weights(directory, cfg)
    plan = build_engine(cfg, weights, length, bool(request.verbose))
    writer.set_header(family="cohere2", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_bytes("tokenizer.json", tokenizer)
    writer.add_json(
        "runtime.json",
        {
            "max_sequence_length": length,
            "vocab_size": cfg.vocab_size,
            "bos_token_id": cfg.bos_token_id,
            "eos_token_id": cfg.eos_token_id,
            "pad_token_id": cfg.pad_token_id,
            "precision": "fp32",
        },
    )
