# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Qwen3.8 checkpoint-to-bundle build."""

from __future__ import annotations

import json
from pathlib import Path

from .config import ModelConfig
from .engine_builder import Qwen38Model


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _runtime_config(model_dir: Path, config: ModelConfig, model: Qwen38Model, **updates) -> dict:
    runtime = model.get_bundle_config_overrides(config)
    eos_token_ids = [config.eos_token_id]
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            eos = generation["eos_token_id"]
            eos_token_ids = eos if isinstance(eos, list) else [eos]
    if not eos_token_ids or any(
        isinstance(value, bool) or not isinstance(value, int) for value in eos_token_ids
    ):
        raise ValueError("Qwen3.8 eos_token_id must contain one or more integer IDs")
    runtime["eos_token_ids"] = eos_token_ids
    runtime.update(updates)
    fields = {
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "bos_token_id",
        "eos_token_ids",
        "num_attention_layers",
        "num_mamba_layers",
        "d_inner",
        "mamba_d_state",
        "mamba_d_conv",
        "mamba_nheads",
        "mamba_head_dim",
        "conv_dim",
        "max_cache_length",
        "precision",
        "layer_types",
        "decoder_engine_layout",
    }
    missing = fields - runtime.keys()
    if missing:
        raise ValueError(f"Qwen3.8 runtime config is missing {sorted(missing)}")
    return {name: runtime[name] for name in fields}


def build(request, writer) -> None:
    """Build one Qwen3.8 hybrid text-generation bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("qwen3_8 does not support dynamic_kv_cache")

    if request.task != "text_generation":
        raise ValueError("qwen3_8 supports only task=text_generation")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("qwen3_8 does not support image dimensions")
    if request.video_num_frames is not None:
        raise NotImplementedError("qwen3_8 does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("qwen3_8 requires max_batch_size=1")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise NotImplementedError("qwen3_8 supports only single-device builds")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("qwen3_8 does not expose quantized engine builds")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    text_config = config.raw.get("text_config", config.raw)
    if (
        not isinstance(text_config, dict)
        or "output_gate_type" not in text_config
        or ("mlp_only_layers" in text_config)
    ):
        raise ValueError("checkpoint is not a Qwen3.8 model")
    precision = str(request.precision).lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError("qwen3_8 precision must be fp16 or fp32")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("qwen3_8 max_sequence_length exceeds checkpoint context capacity")

    model = Qwen38Model()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_fp32_layers"] = list(request.fp32_layers)
    config.raw["_resolved_build_precision"] = precision
    weights = model.load_weights(str(model_dir), config)
    plan = model.build_engine(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        verbose=bool(request.verbose),
        debug_layer_outputs=False,
    )

    writer.set_header(family="qwen3_8", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            model,
            precision=precision,
            max_cache_length=max_sequence_length,
            decoder_engine_layout="single",
        ),
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
