# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apertus family builder: one ``text_continuation`` bundle for ``ApertusForCausalLM``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .config import ApertusConfig, load_config
from .graph import PREFILL_CHUNK_TOKENS, build_engine
from .weights import load_weights

if TYPE_CHECKING:
    from tensorrt_model_connect.bundle_writer import BundleWriter

FAMILY = "apertus"
TASK = "text_continuation"
PRECISIONS = ("bf16", "fp32")


def _special_ids(item: dict, special_tokens: dict) -> list[int]:
    token = item["SpecialToken"]["id"]
    entry = special_tokens.get(token)
    ids = entry.get("ids") if isinstance(entry, dict) else None
    if not isinstance(ids, list) or not all(isinstance(value, int) for value in ids):
        raise ValueError(f"Apertus tokenizer post-processor has no IDs for {token!r}")
    return ids


def tokenizer_contract(tokenizer: dict) -> dict:
    """Special tokens that ``tokenizer(text)`` adds around one sequence."""
    processor = tokenizer.get("post_processor")
    if processor is None:
        return {
            "tokenizer_add_special_tokens": True,
            "tokenizer_prefix_ids": [],
            "tokenizer_suffix_ids": [],
        }
    if processor.get("type") != "TemplateProcessing":
        raise ValueError("Apertus supports only a TemplateProcessing tokenizer post-processor")
    single = processor.get("single")
    sequences = [index for index, item in enumerate(single or []) if "Sequence" in item]
    if (
        not isinstance(single, list)
        or len(sequences) != 1
        or single[sequences[0]]["Sequence"]["id"] != "A"
    ):
        raise ValueError("Apertus tokenizer post-processor must wrap exactly one sequence")
    special_tokens = processor.get("special_tokens") or {}
    parts = ([], [])
    for index, item in enumerate(single):
        if index == sequences[0]:
            continue
        if set(item) != {"SpecialToken"}:
            raise ValueError("Apertus tokenizer post-processor may add only special tokens")
        parts[index > sequences[0]].extend(_special_ids(item, special_tokens))
    return {
        "tokenizer_add_special_tokens": True,
        "tokenizer_prefix_ids": parts[0],
        "tokenizer_suffix_ids": parts[1],
    }


def _runtime_config(
    config: ApertusConfig, precision: str, max_cache_length: int, tokenizer: dict
) -> dict:
    contract = tokenizer_contract(tokenizer)
    # The runtime plugin rejects token IDs outside the vocabulary when it loads the bundle
    # (config.py already checks the EOS IDs); reject them before the engine is built.
    for name in ("tokenizer_prefix_ids", "tokenizer_suffix_ids"):
        if any(
            isinstance(value, bool) or not 0 <= value < config.vocab_size
            for value in contract[name]
        ):
            raise ValueError(
                f"Apertus {name} must be inside the vocabulary [0, {config.vocab_size})"
            )
    return {
        "vocab_size": config.vocab_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "max_cache_length": max_cache_length,
        "prefill_chunk_tokens": min(PREFILL_CHUNK_TOKENS, max_cache_length),
        "precision": precision,
        "eos_token_ids": list(config.eos_token_ids),
        **contract,
    }


def build(request, writer: "BundleWriter") -> None:
    """Build one Apertus bundle through family-owned code."""
    if request.task != TASK:
        raise ValueError(f"{FAMILY} supports only task={TASK}")
    if request.backend != "trt":
        raise ValueError(f"{FAMILY} supports only the trt backend")
    precision = str(request.precision).lower()
    if precision not in PRECISIONS:
        raise ValueError(f"{FAMILY} precision must be one of {PRECISIONS} (the checkpoint is BF16)")
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
    if request.quantization not in (None, "none"):
        raise NotImplementedError(
            f"{FAMILY} does not support quantization={request.quantization!r}"
        )
    if request.fp32_layers:
        raise NotImplementedError(f"{FAMILY} does not support per-layer precision overrides")
    if request.dynamic_kv_cache:
        raise NotImplementedError("apertus does not support dynamic_kv_cache")

    model_dir = Path(request.model_dir)
    config = load_config(model_dir)
    max_cache_length = request.max_sequence_length or config.max_position_embeddings
    if not 1 <= max_cache_length <= config.max_position_embeddings:
        raise ValueError(
            f"{FAMILY} max_sequence_length must be in [1, {config.max_position_embeddings}]"
        )
    tokenizer_bytes = (model_dir / "tokenizer.json").read_bytes()
    runtime = _runtime_config(config, precision, max_cache_length, json.loads(tokenizer_bytes))
    weights = load_weights(model_dir, config, precision)
    plan = build_engine(
        config,
        weights,
        precision=precision,
        max_cache_length=max_cache_length,
        verbose=bool(request.verbose),
    )
    del weights
    writer.set_header(family="apertus", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json("runtime.json", runtime)
    writer.add_bytes("tokenizer.json", tokenizer_bytes)
