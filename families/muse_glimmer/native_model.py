# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-owned native bundle construction; no exporter or plugin graph dependency."""
from __future__ import annotations

import json
from pathlib import Path

from .checkpoint import Checkpoint
from .config import TextConfig
from .native_graph import build_target


def build(request, writer):
    model_dir = Path(request.model_dir)
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    generation_path = model_dir / "generation_config.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8")) if generation_path.is_file() else None
    config = TextConfig.from_dict(raw, generation)
    # Source admission is identical to the existing owning Edge route. This
    # checker reads the source recipe only; it does not query hardware or run Edge.
    from .edge_llm.builder import request_weight_format

    request_weight_format(request, raw)
    capacity = request.max_sequence_length or 1024
    if type(capacity) is not int or not 1 <= capacity <= config.max_position_embeddings:
        raise ValueError("Muse native capacity must fit the checkpoint context")
    if request.execution_variant == "dflash" and capacity < 16:
        raise ValueError("Muse native DFlash capacity must fit the complete block of 16")
    draft_config = None
    if request.execution_variant == "dflash":
        from .draft import DraftConfig, load_draft_weights, build_draft_engine, build_selector_engine
        from .edge_llm.paired import companion_version

        companion = Path(request.companion)
        companion_version(companion)
        draft_config = DraftConfig.from_dict(json.loads((companion / "config.json").read_text()), config)
        draft_weights = load_draft_weights(companion, draft_config)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if not (model_dir / name).is_file():
            raise ValueError(f"Muse native bundle requires {name}")
    with Checkpoint(model_dir) as checkpoint:
        features = draft_config.feature_layers if draft_config else ()
        engine = build_target(config, checkpoint, capacity, target_layer_ids=features, verbose=request.verbose)
        writer.add_bytes("engine.plan", engine)
        del engine
        if draft_config:
            head = checkpoint.projection("lm_head", config.vocab_size, config.hidden_size)
            embedding = checkpoint.projection("embed_tokens", config.vocab_size, config.hidden_size)
            writer.add_bytes("draft.plan", build_draft_engine(
                draft_config, draft_weights, head, embedding, capacity, verbose=request.verbose))
            if draft_config.version == 2:
                writer.add_bytes("selector.plan", build_selector_engine(
                    draft_config, draft_weights, verbose=request.verbose))
    writer.add_json("runtime.json", {
        "version": 1,
        "dflash_version": draft_config.version if draft_config else 0,
        "dflash_block_size": draft_config.query_rows if draft_config else 0,
        "feature_width": len(draft_config.feature_layers) * config.hidden_size if draft_config else 0,
        "mask_token_id": draft_config.mask_token if draft_config else -1,
        "draft_window": draft_config.window if draft_config else 0,
        "selector_rank": draft_config.selector_rank if draft_config else 0,
        "selector_top_k": draft_config.selector_top_k if draft_config else 0,
        "execution_variant": request.execution_variant,
        "precision": "fp16",
        "weight_format": "nvfp4_mxfp8",
        "max_cache_length": capacity,
        "max_input_length": min(capacity, 512),
        # One target engine: a multi-row prefill profile and a one-row decode profile.
        "decoder_engine_layout": "dual_profile",
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "bos_token_id": config.bos_token_id,
        "eos_token_ids": list(config.eos_token_ids),
    })
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        writer.add_bytes(name, (model_dir / name).read_bytes())
