# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native complete Nemotron-H target/draft bundle, independent of Edge installation."""

import json
from pathlib import Path

from .draft import DraftConfig, build_draft_engine, build_markov_engine, load_draft_weights
from .model import ModelConfig, _BUNDLE_FILES, _NemotronHModel, _runtime_config
from .prepared_quantization import PreparedQuantContext, checkpoint_policy
from .tokenizer_assets import native_tokenizer_assets


def build(request, writer, draft: Path, variant: str) -> None:
    """Preserve prepared source formats and the requested speculative protocol."""
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    draft_config = DraftConfig.from_dict(
        json.loads((draft / "config.json").read_text(encoding="utf-8")), config.raw, variant
    )
    policy = checkpoint_policy(
        model_dir, allow_legacy_producer=request.quantization == "nvfp4"
    )
    if (
        request.backend != "trt"
        or request.task != "text_generation"
        or request.precision != "fp16"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.dynamic_kv_cache
        or request.fp32_layers
        or request.graph_transform is not None
        or request.mtp_seq_len is not None
        or any(
            x is not None
            for x in (request.image_height, request.image_width, request.video_num_frames)
        )
        or policy is None
        or policy.request_format != "nvfp4"
        or request.quantization not in {None, "none", "nvfp4"}
    ):
        raise ValueError(
            "Native Nemotron paired execution requires an NVFP4 target and FP16 TP1 text"
        )
    capacity = request.max_sequence_length or min(config.max_position_embeddings, 256)
    if not draft_config.query_rows <= capacity <= config.max_position_embeddings:
        raise ValueError("Native Nemotron paired capacity is outside target/draft limits")
    assets = native_tokenizer_assets(model_dir, _BUNDLE_FILES)
    model = _NemotronHModel()
    config.raw["_model_dir"] = str(model_dir)
    weights = model.load_weights(str(model_dir), config, prepared_policy=policy)
    writer.set_header(family="nemotron_h", task=request.task, backend=request.backend)
    writer.add_bytes(
        "engine.plan",
        model.build_engine(
            config,
            weights,
            capacity,
            precision="fp16",
            quant_ctx=PreparedQuantContext(),
            verbose=bool(request.verbose),
            target_feature_layers=draft_config.feature_layers,
        ),
    )
    # Keep only the actual (potentially untied) target head before loading draft
    # weights. Do not reread the target or substitute draft token embeddings.
    target_head = weights["w_lm_head"]
    fp8_cache = bool(weights.get("_fp8_kv_cache", False))
    del weights
    draft_weights = load_draft_weights(draft, draft_config)
    writer.add_bytes(
        "native_draft.plan",
        build_draft_engine(
            draft_config, draft_weights, target_head, capacity, verbose=bool(request.verbose)
        ),
    )
    if variant != "dflash":
        writer.add_bytes(
            "native_markov.plan",
            build_markov_engine(draft_config, draft_weights, verbose=bool(request.verbose)),
        )
    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            model,
            precision="fp16",
            max_cache_length=capacity,
            quantization="nvfp4",
            kv_cache_quantization="fp8" if fp8_cache else "none",
            decoder_engine_layout="single",
            tensor_parallel_size=1,
            tensor_parallel_mode="single",
        ),
    )
    writer.add_json(
        "native_paired.json",
        {
            "schema_version": 1,
            "execution_variant": variant,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "max_cache_length": capacity,
            "max_input_length": min(capacity, 1024),
            "feature_layers": list(draft_config.feature_layers),
            "draft_rows": draft_config.query_rows,
            "mask_token_id": draft_config.mask_token,
            "sliding_window": draft_config.sliding_window,
            "draft_top_k": 4 if variant == "dspark_tree" else 1,
            "verify_size": 9 if variant == "dspark" else 16,
            "markov_rank": draft_config.markov_rank,
        },
    )
    for filename, content in assets.items():
        writer.add_bytes(filename, content)
