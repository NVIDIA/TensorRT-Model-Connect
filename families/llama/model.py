# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete dense Llama checkpoint-to-bundle build path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .build_routing import native_kv_architecture_capability, native_kv_build_capability
from .checkpoint_mapper import WeightDict, load_standard_weights
from .config import ModelConfig
from .dual_profile_decoder_builder import build_dual_profile_decoder_engine
from .native_kv_contract import validate_native_kv_weights
from .standard_decoder_builder import build_standard_decoder_engine


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


def _build_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_sequence_length: int,
    *,
    precision: str,
    verbose: bool,
) -> bytes:
    capability = native_kv_build_capability(
        config,
        precision=precision,
        max_cache_length=max_sequence_length,
        quantized=False,
        debug_layer_outputs=False,
    )
    if capability.eligible:
        validate_native_kv_weights(config, weights)
        config.raw["_native_kv_cache_metadata"] = {
            "native_kv_contract_version": 1,
            "native_kv_cache": True,
        }
        role = str(config.raw.get("_decoder_engine_role", ""))
        if role not in {"prefill", "decode"}:
            raise ValueError("native Llama build requires an explicit prefill or decode role")
        return build_dual_profile_decoder_engine(
            config,
            weights,
            max_sequence_length,
            precision="bf16",
            verbose=verbose,
            profile_mode=role,
            native_kv_cache=True,
        )

    config.raw.pop("_native_kv_cache_metadata", None)
    return build_standard_decoder_engine(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        verbose=verbose,
    )


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
    runtime.update(config.raw.get("_native_kv_cache_metadata", {}))
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
    """Build one dense Llama bundle through family-owned code only."""
    if request.image_height is not None:
        raise NotImplementedError("llama does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("llama does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("llama does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("llama does not support max_batch_size")


    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("llama supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if not str(config.model_type).lower().startswith("llama"):
        raise ValueError(f"Llama does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Llama precision must be fp32, fp16, or bf16")
    architecture = native_kv_architecture_capability(config)
    default_length = (
        config.max_position_embeddings
        if architecture.eligible
        else min(config.max_position_embeddings, 256)
    )
    max_sequence_length = _positive_int(
        request.max_sequence_length or default_length,
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("Llama max_sequence_length exceeds checkpoint context capacity")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("Llama does not expose a tensor-parallel builder")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("Llama has no qualified family-owned quantized build")

    config.raw["_model_dir"] = str(model_dir)
    config.raw["_fp32_layers"] = list(request.fp32_layers)
    config.raw["_resolved_build_precision"] = precision
    weights = load_standard_weights(
        str(model_dir),
        config,
        precision=precision,
        fp32_layers=request.fp32_layers,
    )

    writer.set_header(family="llama", task=request.task, backend="trt")
    if request.fp32_layers:
        config.raw["_decoder_engine_role"] = "dual_profile"
        plan = _build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            verbose=bool(request.verbose),
        )
        writer.add_bytes("engine.plan", plan)
        layout = "dual_profile"
    else:
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
        writer.add_bytes("engine.plan", decode)
        writer.add_bytes("prefill.plan", prefill)
        layout = "split"
    config.raw.pop("_decoder_engine_role", None)

    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            precision=precision,
            max_cache_length=max_sequence_length,
            decoder_engine_layout=layout,
        ),
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
