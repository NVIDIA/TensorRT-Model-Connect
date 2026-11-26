# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI GPT family builder: one ``text_continuation`` bundle for ``OpenAIGPTLMHeadModel``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .config import OpenAIGPTConfig, load_config
from .tokenizer_contract import check_tokenizer, load_tokenizer

if TYPE_CHECKING:
    from tensorrt_model_connect.bundle_writer import BundleWriter

FAMILY = "openai_gpt"
TASK = "text_continuation"
# The checkpoint is float32, and float32 is the only precision this family builds.
PRECISION = "fp32"


def runtime_config(config: OpenAIGPTConfig, max_cache_length: int) -> dict:
    return {
        "vocab_size": config.vocab_size,
        "num_hidden_layers": config.n_layer,
        "num_attention_heads": config.n_head,
        "head_dim": config.head_dim,
        "max_cache_length": max_cache_length,
        "precision": PRECISION,
        "eos_token_ids": list(config.eos_token_ids),
    }


def build(request, writer: "BundleWriter") -> None:
    """Build one OpenAI GPT bundle; every build option is honored or rejected."""
    if request.task != TASK:
        raise ValueError(f"{FAMILY} supports only task={TASK}")
    if request.backend != "trt":
        raise ValueError(f"{FAMILY} supports only the trt backend")
    precision = str(request.precision).lower()
    if precision != PRECISION:
        raise ValueError(f"{FAMILY} builds the float32 checkpoint at fp32 only, got {precision}")
    if request.quantization not in (None, "none"):
        raise NotImplementedError(f"{FAMILY} does not support quantization={request.quantization!r}")
    for name, value in (
        ("image_height", request.image_height),
        ("image_width", request.image_width),
        ("video_num_frames", request.video_num_frames),
        ("mtp_seq_len", request.mtp_seq_len),
    ):
        if value is not None:
            raise NotImplementedError(f"{FAMILY} does not support {name}")
    if request.max_batch_size != 1:
        raise NotImplementedError(f"{FAMILY} supports only max_batch_size=1")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise NotImplementedError(f"{FAMILY} runs on one GPU without tensor or context parallelism")
    if request.fp32_layers:
        raise NotImplementedError(f"{FAMILY} does not support per-layer precision overrides")
    if request.dynamic_kv_cache:
        raise NotImplementedError("openai_gpt does not support dynamic_kv_cache")

    model_dir = Path(request.model_dir)
    config = load_config(model_dir)
    # KV-cache rows: max_sequence_length, by default every learned position.
    max_cache_length = request.max_sequence_length
    if max_cache_length is None:
        max_cache_length = config.n_positions
    if isinstance(max_cache_length, bool) or not isinstance(max_cache_length, int):
        raise ValueError(f"{FAMILY} max_sequence_length must be an integer")
    if not 1 <= max_cache_length <= config.n_positions:
        raise ValueError(f"{FAMILY} max_sequence_length must be in [1, {config.n_positions}]")
    tokenizer_bytes = (model_dir / "tokenizer.json").read_bytes()
    check_tokenizer(load_tokenizer(tokenizer_bytes), config.vocab_size)

    from .graph import build_engine
    from .weights import load_weights

    weights = load_weights(model_dir, config)
    plan = build_engine(
        config, weights, max_cache_length=max_cache_length, verbose=bool(request.verbose)
    )
    del weights
    writer.set_header(family="openai_gpt", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json("runtime.json", runtime_config(config, max_cache_length))
    writer.add_bytes("tokenizer.json", tokenizer_bytes)
