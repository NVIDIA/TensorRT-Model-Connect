# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete Mistral checkpoint-to-bundle build path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .checkpoint_mapper import WeightDict, load_standard_weights
from .config import ModelConfig
from .default_decoder import build_standard_decoder_engine


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


def _build_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_sequence_length: int,
    *,
    precision: str,
    verbose: bool,
) -> bytes:
    return build_standard_decoder_engine(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        position_type="rope",
        activation="silu",
        verbose=verbose,
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


def _runtime_config(model_dir: Path, config: ModelConfig, **updates) -> dict:
    runtime = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
    }
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            runtime["eos_token_id"] = generation["eos_token_id"]
    runtime.update(updates)
    return runtime


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Mistral bundle through family-owned code only."""
    if request.image_height is not None:
        raise NotImplementedError("mistral does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("mistral does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("mistral does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("mistral does not support max_batch_size")


    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("mistral supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if not str(config.model_type).lower().startswith("mistral"):
        raise ValueError(f"Mistral does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Mistral precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("Mistral max_sequence_length exceeds checkpoint context capacity")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("Mistral does not expose a tensor-parallel builder")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("Mistral has no qualified family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("Mistral does not expose mixed-precision layer selection")

    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    weights = load_standard_weights(str(model_dir), config, precision=precision)
    config.raw["_decoder_engine_role"] = "prefill"
    prefill = _build_engine(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        verbose=bool(request.verbose),
    )
    config.raw["_decoder_engine_role"] = "decode"
    decode = _build_engine(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        verbose=bool(request.verbose),
    )
    config.raw.pop("_decoder_engine_role", None)

    writer.set_header(family="mistral", task=request.task, backend="trt")
    writer.add_bytes("engine.plan", decode)
    writer.add_bytes("prefill.plan", prefill)
    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            precision=precision,
            max_cache_length=max_sequence_length,
            decoder_engine_layout="split",
        ),
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
